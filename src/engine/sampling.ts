/**
 * Сэмплинг следующего токена: маска structured output → temperature → softmax → min-p → top-k → top-p → выбор.
 *
 * Порядок как в сэмплере vLLM V1 (vllm/v1/sample/sampler.py): маска грамматики пишет −∞ в логиты до
 * сэмплера, затем temperature, затем min-p (MinPLogitsProcessor), затем top-k и top-p
 * (apply_top_k_top_p: top-p считается по распределению, уже обрезанному top-k).
 * Формула маскирования: p_i = exp(z_i/T) / Σ_{j∈A} exp(z_j/T) для i ∈ A, иначе 0
 * (research/notes-tokens-templates-structured.md §1.14, §2.6). T = 0 означает greedy (argmax).
 *
 * ГПСЧ передаёт вызывающий (rng: () => number в [0, 1)), сам модуль случайности не создаёт.
 */
import type { Candidate, SamplingParams, SamplingResult } from './types'

/** Параметры сэмплинга с необязательным min-p (0 или undefined — выключен). */
export interface SamplingParamsEx extends SamplingParams {
  minP?: number
}

/** SamplingResult с промежуточными шагами для визуализации. */
export interface SamplingResultEx extends SamplingResult {
  /** Распределение среди разрешённых маской (после temperature, до min-p/top-k/top-p). */
  maskedProbs: number[]
  /** Разрешён ли кандидат маской (structured output). */
  allowed: boolean[]
  /** Почему кандидат отброшен: 'mask' | 'min-p' | 'top-k' | 'top-p' | 'greedy'; null — остался. */
  dropReason: Array<'mask' | 'min-p' | 'top-k' | 'top-p' | 'greedy' | null>
  /** Энтропия итогового распределения в битах. */
  entropyBits: number
  /** Случайное число, которым выбирали (undefined при greedy). */
  u?: number
}

/** Softmax логитов с температурой (T > 0); −Infinity даёт вероятность 0. */
export function softmax(logits: number[], temperature = 1): number[] {
  if (logits.length === 0) return []
  const t = temperature > 0 ? temperature : 1
  let max = -Infinity
  for (const z of logits) if (z > max) max = z
  if (max === -Infinity) return logits.map(() => 0)
  const ex = logits.map((z) => (z === -Infinity ? 0 : Math.exp((z - max) / t)))
  const sum = ex.reduce((a, b) => a + b, 0)
  return ex.map((e) => e / sum)
}

/** Энтропия распределения: −Σ p·log p (по умолчанию в битах; base='e' — в натах). */
export function entropy(probs: number[], base: 2 | 'e' = 2): number {
  let h = 0
  for (const p of probs) if (p > 0) h -= p * Math.log(p)
  return base === 2 ? h / Math.LN2 : h
}

/** Индекс максимального логита среди разрешённых (при равенстве меньший индекс); -1, если разрешённых нет. */
export function argmax(logits: number[], allowed?: boolean[]): number {
  let best = -1
  for (let i = 0; i < logits.length; i++) {
    if (allowed && !allowed[i]) continue
    if (best < 0 || logits[i] > logits[best]) best = i
  }
  return best
}

/** Маска top-k: true у k лучших по вероятности (ничьи на границе остаются, как в vLLM). k ≤ 0 — выключено. */
export function topKMask(probs: number[], k: number, keep?: boolean[]): boolean[] {
  const base = keep ?? probs.map(() => true)
  // k — число токенов: дробное значение (например, со слайдера) округляем вниз, как int(top_k) в vLLM;
  // без этого vals[k − 1] = undefined и выкидывалось всё.
  const kk = Math.floor(k)
  if (!(kk > 0)) return base.slice()
  const vals = probs.filter((_, i) => base[i]).sort((a, b) => b - a)
  if (vals.length <= kk) return base.slice()
  const kth = vals[kk - 1]
  return probs.map((p, i) => base[i] && p >= kth)
}

/**
 * Маска top-p (nucleus) как в vLLM: сортировка по возрастанию, отбрасываются токены, у которых
 * накопленная сумма ≤ 1 − p; самый вероятный остаётся всегда. p ≥ 1 — выключено.
 */
export function topPMask(probs: number[], p: number, keep?: boolean[]): boolean[] {
  const base = keep ?? probs.map(() => true)
  if (!(p < 1)) return base.slice()
  const idx = probs.map((_, i) => i).filter((i) => base[i])
  const total = idx.reduce((s, i) => s + probs[i], 0)
  if (total <= 0) return base.slice()
  idx.sort((a, b) => probs[a] - probs[b] || b - a)
  const out = base.slice()
  let cum = 0
  for (let k = 0; k < idx.length - 1; k++) {
    cum += probs[idx[k]] / total
    // допуск 1e-12 гасит ошибку округления: 0.05 + 0.15 = 0.2, а 1 − 0.8 = 0.19999999999999996
    if (cum <= 1 - p + 1e-12) out[idx[k]] = false
    else break
  }
  return out
}

/** Маска min-p: остаются токены с p ≥ minP · p_max. minP ≤ 0 — выключено. */
export function minPMask(probs: number[], minP: number, keep?: boolean[]): boolean[] {
  const base = keep ?? probs.map(() => true)
  if (!(minP > 0)) return base.slice()
  // vLLM принимает min_p ∈ [0, 1]; больше 1 выкинуло бы и самый вероятный токен
  minP = Math.min(minP, 1)
  let pmax = 0
  probs.forEach((p, i) => {
    if (base[i] && p > pmax) pmax = p
  })
  return probs.map((p, i) => base[i] && p >= minP * pmax)
}

/**
 * Всё распределение без выбора: probs (temperature + softmax по всем кандидатам, до маски и отсечений),
 * маска, отсечения и перенормированные finalProbs. chosen = argmax finalProbs (для показа).
 */
export function distribution(cands: Candidate[], params: SamplingParamsEx, allowed?: boolean[]): SamplingResultEx {
  if (cands.length === 0) throw new Error('Сэмплинг: пустой список кандидатов')
  const logits = cands.map((c) => c.logit)
  const allow = cands.map((_, i) => (allowed ? !!allowed[i] : true))
  const greedy = !(params.temperature > 0)
  const dropReason: SamplingResultEx['dropReason'] = allow.map((a) => (a ? null : 'mask'))

  if (greedy) {
    const best = argmax(logits, allow)
    const top = argmax(logits)
    const probs = logits.map((_, i) => (i === top ? 1 : 0))
    const maskedProbs = logits.map((_, i) => (i === best ? 1 : 0))
    const kept = logits.map((_, i) => i === best)
    kept.forEach((k, i) => {
      if (!k && dropReason[i] === null) dropReason[i] = 'greedy'
    })
    return { probs, kept, finalProbs: maskedProbs.slice(), chosen: best, maskedProbs, allowed: allow, dropReason, entropyBits: 0 }
  }

  const T = params.temperature
  const probs = softmax(logits, T)
  const maskedProbs = softmax(
    logits.map((z, i) => (allow[i] ? z : -Infinity)),
    T,
  )
  let keep = allow.slice()
  const afterMinP = minPMask(maskedProbs, params.minP ?? 0, keep)
  afterMinP.forEach((k, i) => {
    if (keep[i] && !k) dropReason[i] = 'min-p'
  })
  keep = afterMinP
  const afterK = topKMask(maskedProbs, params.topK, keep)
  afterK.forEach((k, i) => {
    if (keep[i] && !k) dropReason[i] = 'top-k'
  })
  keep = afterK
  const afterP = topPMask(maskedProbs, params.topP, keep)
  afterP.forEach((k, i) => {
    if (keep[i] && !k) dropReason[i] = 'top-p'
  })
  keep = afterP

  const mass = maskedProbs.reduce((s, p, i) => (keep[i] ? s + p : s), 0)
  const finalProbs = maskedProbs.map((p, i) => (keep[i] && mass > 0 ? p / mass : 0))
  let chosen = -1
  finalProbs.forEach((p, i) => {
    if (p > 0 && (chosen < 0 || p > finalProbs[chosen])) chosen = i
  })
  return { probs, kept: keep, finalProbs, chosen, maskedProbs, allowed: allow, dropReason, entropyBits: entropy(finalProbs) }
}

/**
 * Выбор токена: при temperature > 0 — обратная функция распределения по finalProbs с u = rng(),
 * при temperature = 0 — argmax среди разрешённых. allowed — маска structured output (необязательно).
 * Если разрешённых кандидатов нет, chosen = -1. params.seed игнорируется: rng создаёт вызывающий.
 */
export function sample(cands: Candidate[], params: SamplingParamsEx, rng: () => number, allowed?: boolean[]): SamplingResultEx {
  const d = distribution(cands, params, allowed)
  if (!(params.temperature > 0) || d.chosen < 0) return d
  const u = rng()
  let acc = 0
  let chosen = -1
  for (let i = 0; i < d.finalProbs.length; i++) {
    if (d.finalProbs[i] <= 0) continue
    acc += d.finalProbs[i]
    chosen = i
    if (u < acc) break
  }
  return { ...d, chosen, u }
}
