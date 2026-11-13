/**
 * Распределение следующего токена: явные кандидаты («голова») + «хвост» из M токенов с одинаковым
 * низким логитом, так что вместе получается вся ширина логитов Qwen3 (151 936).
 *
 * Математика та же, что в src/engine/sampling.ts (порядок vLLM V1: temperature → softmax → min-p →
 * top-k → top-p → перенормировка), но хвост считается аналитически, без массива на 150 тысяч чисел.
 * Эквивалентность с движком на развёрнутом словаре проверяет dist.test.ts.
 *
 * Упрощение: настоящие логиты хвоста разные; здесь у всех M одинаковое значение. Чтобы top-k > числа
 * кандидатов вёл себя как на настоящих (различных) логитах, из хвоста оставляется ровно k − n токенов.
 */
import { argmax, entropy, softmax } from '@/engine/sampling'

export interface HeadCandidate {
  text: string
  logit: number
}

export interface TailGroup {
  count: number
  logit: number
}

export interface DistParams {
  temperature: number
  topK: number
  topP: number
  minP: number
}

export type DropReason = 'greedy' | 'min-p' | 'top-k' | 'top-p' | null

export interface DistRow {
  key: string
  text: string
  tail: boolean
  /** Сколько токенов в строке (1 у кандидата, M у хвоста). */
  count: number
  /** Логит z (у хвоста логит каждого из M токенов). */
  logit: number
  /** Вероятность строки после temperature, до отсечений (у хвоста сумма по всем M). */
  before: number
  /** Итоговая вероятность после отсечений и перенормировки (у хвоста сумма по оставшимся). */
  after: number
  /** Сколько токенов строки осталось. */
  kept: number
  reason: DropReason
}

export interface DistResult {
  rows: DistRow[]
  greedy: boolean
  /** Энтропия в битах: до отсечений (вся ширина словаря) и после. */
  entropyBefore: number
  entropyAfter: number
  /** Сколько токенов словаря осталось после отсечений. */
  keptTokens: number
  /** Вероятность одного токена хвоста до и после (для подписи). */
  tailEachBefore: number
  tailEachAfter: number
}

const H2 = (p: number) => (p > 0 ? -p * Math.log2(p) : 0)

export function computeDist(head: HeadCandidate[], tail: TailGroup, params: DistParams): DistResult {
  const n = head.length
  const M = Math.max(0, Math.floor(tail.count))
  const greedy = !(params.temperature > 0)

  if (greedy) {
    const best = argmax(head.map((c) => c.logit))
    const rows: DistRow[] = head.map((c, i) => ({
      key: `h${i}`,
      text: c.text,
      tail: false,
      count: 1,
      logit: c.logit,
      before: i === best ? 1 : 0,
      after: i === best ? 1 : 0,
      kept: i === best ? 1 : 0,
      reason: i === best ? null : 'greedy',
    }))
    rows.push({ key: 'tail', text: '', tail: true, count: M, logit: tail.logit, before: 0, after: 0, kept: 0, reason: 'greedy' })
    return { rows, greedy, entropyBefore: 0, entropyAfter: 0, keptTokens: 1, tailEachBefore: 0, tailEachAfter: 0 }
  }

  const T = params.temperature
  // softmax по голове и хвосту вместе: хвост как M одинаковых слагаемых
  const zmax = Math.max(tail.logit, ...head.map((c) => c.logit))
  const wHead = head.map((c) => Math.exp((c.logit - zmax) / T))
  const wTail = M > 0 ? Math.exp((tail.logit - zmax) / T) : 0
  const Z = wHead.reduce((s, w) => s + w, 0) + M * wTail
  const p = wHead.map((w) => w / Z)
  const q = wTail / Z

  // min-p: порог относительно самого вероятного токена
  const keepHead = head.map(() => true)
  let keepTail = M
  const reason: DropReason[] = head.map(() => null)
  let tailReason: DropReason = null
  const minP = Math.min(1, params.minP)
  if (minP > 0) {
    const pmax = Math.max(q, ...p)
    p.forEach((v, i) => {
      if (v < minP * pmax) {
        keepHead[i] = false
        reason[i] = 'min-p'
      }
    })
    if (M > 0 && q < minP * pmax) {
      keepTail = 0
      tailReason = 'min-p'
    }
  }

  // top-k: k лучших; ничьи на границе остаются (как в движке), хвост ниже любого кандидата
  const k = Math.floor(params.topK)
  if (k > 0) {
    const keptVals = p.filter((_, i) => keepHead[i]).sort((a, b) => b - a)
    if (keptVals.length >= k) {
      const kth = keptVals[k - 1]
      p.forEach((v, i) => {
        if (keepHead[i] && v < kth) {
          keepHead[i] = false
          reason[i] = 'top-k'
        }
      })
      if (keepTail > 0 && q < kth) {
        keepTail = 0
        tailReason = 'top-k'
      }
    } else if (keepTail > 0) {
      const room = k - keptVals.length
      if (room < keepTail) {
        keepTail = room
        if (room === 0) tailReason = 'top-k'
      }
    }
  }

  // top-p как в движке: по возрастанию отбрасываем, пока накопленная доля ≤ 1 − p; лучший остаётся всегда
  if (params.topP < 1) {
    const S = p.reduce((s, v, i) => (keepHead[i] ? s + v : s), 0) + keepTail * q
    if (S > 0) {
      let budget = (1 - params.topP + 1e-12) * S
      const keptHeadIdx = p.map((_, i) => i).filter((i) => keepHead[i])
      const totalKept = keptHeadIdx.length + keepTail
      let dropped = 0
      // сначала хвост (самые маленькие)
      if (keepTail > 0) {
        // хвост с нулевой вероятностью (переполнение вниз при малой T) отбрасывается целиком
        const canDrop = Math.min(keepTail, q > 0 ? Math.floor(budget / q) : keepTail, totalKept - 1)
        budget -= canDrop * q
        dropped += canDrop
        keepTail -= canDrop
        if (keepTail === 0) tailReason = 'top-p'
      }
      if (keepTail === 0) {
        keptHeadIdx.sort((a, b) => p[a] - p[b] || b - a)
        for (const i of keptHeadIdx) {
          if (dropped >= totalKept - 1) break
          if (p[i] <= budget) {
            budget -= p[i]
            dropped++
            keepHead[i] = false
            reason[i] = 'top-p'
          } else break
        }
      }
    }
  }

  const mass = p.reduce((s, v, i) => (keepHead[i] ? s + v : s), 0) + keepTail * q
  const after = p.map((v, i) => (keepHead[i] && mass > 0 ? v / mass : 0))
  const qAfter = keepTail > 0 && mass > 0 ? q / mass : 0

  const rows: DistRow[] = head.map((c, i) => ({
    key: `h${i}`,
    text: c.text,
    tail: false,
    count: 1,
    logit: c.logit,
    before: p[i],
    after: after[i],
    kept: keepHead[i] ? 1 : 0,
    reason: reason[i],
  }))
  rows.push({ key: 'tail', text: '', tail: true, count: M, logit: tail.logit, before: M * q, after: keepTail * qAfter, kept: keepTail, reason: keepTail > 0 ? null : tailReason })

  // энтропия через движок по голове плюс вклад M (или оставшихся) одинаковых токенов хвоста
  const entropyBefore = entropy(p) + M * H2(q)
  const entropyAfter = entropy(after) + keepTail * H2(qAfter)
  const keptTokens = keepHead.filter(Boolean).length + keepTail
  return { rows, greedy, entropyBefore, entropyAfter, keptTokens, tailEachBefore: q, tailEachAfter: qAfter }
}

/** Выбор по обратной функции распределения в порядке строк (кандидаты, затем хвост). */
export function pickRow(d: DistResult, u: number): { row: number; tailIndex?: number } {
  let acc = 0
  let last = -1
  for (let i = 0; i < d.rows.length; i++) {
    const r = d.rows[i]
    if (r.after <= 0) continue
    last = i
    if (u < acc + r.after) {
      if (r.tail && r.kept > 0) return { row: i, tailIndex: Math.min(r.kept - 1, Math.floor(((u - acc) / r.after) * r.kept)) }
      return { row: i }
    }
    acc += r.after
  }
  return { row: last }
}

/** Вероятности после temperature без хвоста (для мелких демо). */
export function headProbs(head: HeadCandidate[], T: number): number[] {
  return softmax(
    head.map((c) => c.logit),
    T,
  )
}
