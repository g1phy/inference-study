import type { Token } from '@/engine/types'

/** Короткая видимая форма токена для подписей в SVG: пробел «·», перевод строки «↵». */
export function tokenLabel(t: Token): string {
  if (t.special) return t.text || t.raw
  const s = t.text.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '→')
  return s.length ? s : '∅'
}

/** Короткая форма в «ёлочках» для прозы и подписей. */
export function quoted(t: Token): string {
  return `«${tokenLabel(t)}»`
}
