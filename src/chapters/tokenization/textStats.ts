/**
 * Помощники главы «Токенизация»: подсчёт символов, классы пре-токенов, поиск слова под токеном.
 * Чистые функции без React (тест: textStats.test.ts).
 */
import type { Token } from '@/engine/types'

/** Число символов, как их видит человек (графемы; 🧑‍💻 = 1). Без Intl.Segmenter: code points. */
export function charCount(text: string): number {
  const Seg = (Intl as unknown as { Segmenter?: new (l: string, o: { granularity: 'grapheme' }) => { segment: (s: string) => Iterable<unknown> } }).Segmenter
  if (Seg) {
    let n = 0
    for (const _ of new Seg('ru', { granularity: 'grapheme' }).segment(text)) n++
    return n
  }
  return Array.from(text).length
}

export function utf8Length(text: string): number {
  return new TextEncoder().encode(text).length
}

export type PieceClass = 'word' | 'digit' | 'punct' | 'space' | 'contraction' | 'special'

/** Класс пре-токена по содержимому (для подсветки; само разбиение делает регулярка движка). */
export function classifyPiece(piece: string, isSpecial = false): PieceClass {
  if (isSpecial) return 'special'
  if (/^\s+$/u.test(piece)) return 'space'
  if (/^'(?:s|t|re|ve|m|ll|d)$/iu.test(piece)) return 'contraction'
  if (/\p{L}/u.test(piece)) return 'word'
  if (/^\p{N}+$/u.test(piece)) return 'digit'
  return 'punct'
}

/** Диапазоны пре-токенов в строке (UTF-16), если склейка кусков даёт исходный текст. */
export function pieceSpans(text: string, pieces: string[]): Array<{ start: number; end: number }> | null {
  const spans: Array<{ start: number; end: number }> = []
  let pos = 0
  for (const p of pieces) {
    if (!text.startsWith(p, pos)) return null
    spans.push({ start: pos, end: pos + p.length })
    pos += p.length
  }
  return pos === text.length ? spans : null
}

/**
 * Слово (пре-токен) под токеном и индексы всех токенов этого слова.
 * Текст ожидается в NFC (иначе смещения пре-токенов и токенов расходятся): тогда берём сам токен.
 */
export function wordOfToken(
  text: string,
  pieces: string[],
  tokens: Token[],
  index: number,
): { word: string; indices: number[] } | null {
  const t = tokens[index]
  if (!t || t.special) return null
  const spans = text === text.normalize('NFC') ? pieceSpans(text, pieces) : null
  if (!spans || t.start === undefined) return { word: t.text, indices: [index] }
  const k = spans.findIndex((s) => t.start! >= s.start && t.start! < s.end)
  if (k < 0) return { word: t.text, indices: [index] }
  const s = spans[k]
  const indices: number[] = []
  tokens.forEach((x, i) => {
    if (x.start !== undefined && x.start >= s.start && x.start < s.end) indices.push(i)
  })
  return { word: pieces[k], indices }
}

/** Индексы токенов другой ленты, которые пересекают символы [start, end). */
export function overlapping(tokens: Token[], start: number, end: number): number[] {
  const out: number[] = []
  tokens.forEach((t, i) => {
    if (t.start === undefined || t.end === undefined) return
    if (t.start < end && t.end > start) out.push(i)
  })
  return out
}
