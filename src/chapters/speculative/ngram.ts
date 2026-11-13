/**
 * N-gram speculation (prompt lookup), как в vLLM: ищем последние n токенов (n от nMax до nMin)
 * раньше в контексте (промпт + уже сгенерированное) и предлагаем черновиком k токенов,
 * которые шли после совпадения (notes-engines.md §1.11, notes-techniques.md §3.11).
 * Проверка жадная (temperature 0): черновой токен принят, если совпал с тем, что выдала бы модель.
 * Здесь «что выдала бы модель» = заранее известный ответ (иллюстрация, текст ответа выдуман).
 */

export interface LookupRound {
  /** Позиция в ответе, с которой начинается раунд. */
  start: number
  /** Предложенные черновые токены (id). Пусто, если совпадения нет. */
  drafts: number[]
  /** Сколько черновиков совпало с ответом. */
  accepted: number
  /** Длина найденного совпадения n (0, если нет). */
  matchLen: number
  /** Где в контексте (промпт + ответ) начинается найденное совпадение, -1 если нет. */
  matchAt: number
  /** Сколько токенов ответа дал раунд: принятые + 1 токен target. */
  produced: number
}

/** Последнее (самое правое) вхождение suffix в ctx, которое заканчивается до начала самого suffix. */
function findEarlier(ctx: readonly number[], n: number): number {
  const L = ctx.length
  if (L <= n) return -1
  for (let i = L - n - 1; i >= 0; i--) {
    let ok = true
    for (let j = 0; j < n; j++) {
      if (ctx[i + j] !== ctx[L - n + j]) {
        ok = false
        break
      }
    }
    if (ok) return i
  }
  return -1
}

export function promptLookupRounds(
  prompt: readonly number[],
  output: readonly number[],
  opts: { k: number; nMin: number; nMax: number },
): LookupRound[] {
  const rounds: LookupRound[] = []
  let pos = 0
  while (pos < output.length) {
    const ctx = [...prompt, ...output.slice(0, pos)]
    let matchLen = 0
    let matchAt = -1
    for (let n = opts.nMax; n >= opts.nMin; n--) {
      const at = findEarlier(ctx, n)
      if (at >= 0) {
        matchLen = n
        matchAt = at
        break
      }
    }
    let drafts: number[] = []
    if (matchAt >= 0) drafts = ctx.slice(matchAt + matchLen, matchAt + matchLen + opts.k)
    // последний токен раунда всегда за target, поэтому черновик не длиннее остатка − 1
    drafts = drafts.slice(0, Math.max(0, output.length - pos - 1))
    let accepted = 0
    while (accepted < drafts.length && drafts[accepted] === output[pos + accepted]) accepted++
    rounds.push({ start: pos, drafts, accepted, matchLen, matchAt, produced: accepted + 1 })
    pos += accepted + 1
  }
  return rounds
}
