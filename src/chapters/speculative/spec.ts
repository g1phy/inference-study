/**
 * Математика speculative decoding для главы 24.
 *
 * Формулы Leviathan, Kalman, Matias (ICML 2023, arXiv 2211.17192), как в
 * research/notes-techniques.md §2.3 и §3.8 (и notes-numbers.md §3.19):
 *   ожидаемое число токенов за проход target: E = (1 − α^(k+1)) / (1 − α)
 *   ускорение по времени:                     S = E / (k·c + 1)
 * α: вероятность принять очередной черновой токен (i.i.d.), k: длина черновика,
 * c: время шага черновика относительно шага target.
 */
import { decodeStepBreakdown, stepTime, type ModelLike, type GpuLike } from '@/engine/perf'
import type { DType } from '@/engine/types'
import { mulberry32 } from '@/engine/rng'

/** Ожидаемое число токенов за один проход target (принятые черновики + 1 свой токен). */
export function expectedTokens(alpha: number, k: number): number {
  const a = Math.min(Math.max(alpha, 0), 1)
  if (a === 1) return k + 1
  return (1 - Math.pow(a, k + 1)) / (1 - a)
}

/** Ожидаемое ускорение по времени с учётом цены черновика c (доля шага target на один черновой токен). */
export function expectedSpeedup(alpha: number, k: number, c: number): number {
  return expectedTokens(alpha, k) / (k * Math.max(0, c) + 1)
}

/** P(N = j) для числа принятых черновиков N: α^j(1 − α) при j < k, α^k при j = k. */
export function acceptedDistribution(alpha: number, k: number): number[] {
  const out: number[] = []
  for (let j = 0; j < k; j++) out.push(Math.pow(alpha, j) * (1 - alpha))
  out.push(Math.pow(alpha, k))
  return out
}

/** Доля принятых от предложенных черновиков: E[N] / k (не путать с α и с acceptance length E). */
export function acceptedFraction(alpha: number, k: number): number {
  return k > 0 ? (expectedTokens(alpha, k) - 1) / k : 0
}

/** Длина черновика с максимальным ожидаемым ускорением (перебор 1..kMax). */
export function optimalK(alpha: number, c: number, kMax = 16): { k: number; speedup: number } {
  let best = { k: 1, speedup: expectedSpeedup(alpha, 1, c) }
  for (let k = 2; k <= kMax; k++) {
    const s = expectedSpeedup(alpha, k, c)
    if (s > best.speedup + 1e-12) best = { k, speedup: s }
  }
  return best
}

/* ============================================================================
 * Пошаговая симуляция для анимации
 * ========================================================================== */

export type DraftVerdict = 'accepted' | 'rejected' | 'discarded'

export interface SpecRound {
  /** Индекс первого токена ответа, с которого начинается раунд. */
  start: number
  /** Черновые позиции раунда (k штук, может быть меньше у конца ответа). */
  drafts: Array<{ pos: number; verdict: DraftVerdict }>
  /** Сколько черновиков принято. */
  accepted: number
  /** Токен target в конце раунда: исправление на месте первого отказа или бонус, если приняты все. */
  targetPos: number
  targetKind: 'correction' | 'bonus'
  /** Сколько токенов ответа добавил раунд (принятые + 1). */
  produced: number
}

/**
 * Разбить ответ длиной total токенов на раунды спекуляции: в каждом раунде черновик
 * предлагает k токенов, каждый принимается с вероятностью α (пока не случится первый отказ),
 * затем target добавляет свой токен. Детерминированно по seed.
 */
export function simulateRounds(total: number, alpha: number, k: number, seed = 1): SpecRound[] {
  const rng = mulberry32(seed)
  const rounds: SpecRound[] = []
  let pos = 0
  while (pos < total) {
    const room = total - pos - 1 // последний токен раунда всегда за target
    const kk = Math.max(0, Math.min(k, room))
    const drafts: SpecRound['drafts'] = []
    let accepted = 0
    let rejectedAt = -1
    for (let i = 0; i < kk; i++) {
      if (rejectedAt >= 0) {
        drafts.push({ pos: pos + i, verdict: 'discarded' })
        continue
      }
      if (rng() < alpha) {
        drafts.push({ pos: pos + i, verdict: 'accepted' })
        accepted++
      } else {
        drafts.push({ pos: pos + i, verdict: 'rejected' })
        rejectedAt = i
      }
    }
    const targetPos = pos + accepted
    rounds.push({
      start: pos,
      drafts,
      accepted,
      targetPos,
      targetKind: rejectedAt >= 0 ? 'correction' : 'bonus',
      produced: accepted + 1,
    })
    pos = targetPos + 1
  }
  return rounds
}

/* ============================================================================
 * Где спекуляция помогает: зависимость от batch (упрощённо, roofline из engine/perf)
 * ========================================================================== */

export interface BatchSpecPoint {
  batch: number
  /** Время обычного шага decode, мс. */
  baseMs: number
  /** Время проверочного прохода target по k+1 позициям у каждой последовательности, мс. */
  verifyMs: number
  /** Время k шагов черновика, мс (k · c · baseMs). */
  draftMs: number
  /** Ожидаемое ускорение токенов на пользователя. */
  speedup: number
  /** Проверочный проход упирается в вычисления. */
  verifyBound: 'memory' | 'compute'
}

/**
 * Ускорение спекуляции при разном batch. Проверка k черновиков стоит как prefill-кусок
 * из k+1 токенов на каждую последовательность: веса читаются один раз, KV каждой
 * последовательности тоже, а FLOPs растут в k+1 раз. Пока шаг упирается в память,
 * это почти бесплатно; когда в вычисления, проверка дорожает и выигрыш пропадает.
 * Упрощения: LM head считается для одной позиции на последовательность, черновик стоит
 * k · c обычных шагов при том же batch, MFU/MBU по умолчанию engine/perf.
 */
export function batchSpeedupSweep(
  model: ModelLike,
  gpu: GpuLike,
  opts: { alpha: number; k: number; c: number; avgContext: number; batches: number[]; dtype?: DType; kvDtype?: DType },
): BatchSpecPoint[] {
  const E = expectedTokens(opts.alpha, opts.k)
  const perf = { dtype: opts.dtype, kvDtype: opts.kvDtype }
  return opts.batches.map((batch) => {
    const base = decodeStepBreakdown(model, gpu, { ...perf, batch, avgContext: opts.avgContext })
    const chunk = { tokens: opts.k + 1, context: opts.avgContext, last: true }
    const verify = stepTime(model, gpu, { prefill: Array.from({ length: batch }, () => chunk) }, perf)
    const draftMs = opts.k * opts.c * base.ms
    return {
      batch,
      baseMs: base.ms,
      verifyMs: verify.ms,
      draftMs,
      speedup: (E * base.ms) / (verify.ms + draftMs),
      verifyBound: verify.bound,
    }
  })
}

/** Время одного прохода target по n токенам сразу (как проверка n − 1 черновиков), мс. */
export function passMs(model: ModelLike, gpu: GpuLike, tokens: number, avgContext: number, dtype?: DType): { ms: number; bound: 'memory' | 'compute'; weightBytes: number } {
  const s = stepTime(model, gpu, { prefill: [{ tokens, context: avgContext, last: true }] }, { dtype })
  return { ms: s.ms, bound: s.bound, weightBytes: s.weightBytes }
}
