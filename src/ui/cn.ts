/** Склеить классы, пропуская пустые значения: cn('a', cond && 'b', undefined). */
export function cn(...parts: Array<string | false | null | undefined | 0>): string {
  let out = ''
  for (const p of parts) {
    if (p) out = out ? `${out} ${p}` : p
  }
  return out
}
