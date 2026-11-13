/**
 * Маленькие помощники для фигур главы «Детокенизация»: разбор UTF-8 и видимые пробелы.
 * Чистые функции без React (покрыты detok.test.ts).
 */

/** Сколько байтов занимает символ, который начинается с этого байта (0: байт продолжения или мусор). */
export function utf8SeqLen(lead: number): number {
  if (lead < 0x80) return 1
  if (lead >= 0xc2 && lead <= 0xdf) return 2
  if (lead >= 0xe0 && lead <= 0xef) return 3
  if (lead >= 0xf0 && lead <= 0xf4) return 4
  return 0
}

/** Роль байта в UTF-8: ASCII, начало многобайтного символа или продолжение. */
export type ByteRole = 'ascii' | 'lead' | 'cont' | 'bad'

export function byteRole(b: number): ByteRole {
  if (b < 0x80) return 'ascii'
  if (b >= 0x80 && b <= 0xbf) return 'cont'
  return utf8SeqLen(b) > 0 ? 'lead' : 'bad'
}

/** Недописанный символ в конце потока байтов: что уже пришло и сколько байтов нужно всего. */
export interface PendingChar {
  /** Байты недописанного символа (начиная с ведущего). */
  bytes: number[]
  /** Длина символа целиком. */
  need: number
  /** Целые байты перед ним (например, пробел из того же токена). */
  before: number[]
}

/**
 * Находит неполный символ в конце байтов (то, из-за чего декодер выдаёт «�» на конце).
 * null, если байты заканчиваются целым символом.
 */
export function pendingChar(bytes: readonly number[]): PendingChar | null {
  let i = 0
  let lastStart = -1
  let lastNeed = 0
  while (i < bytes.length) {
    const n = utf8SeqLen(bytes[i])
    if (n === 0) {
      i++
      continue
    }
    lastStart = i
    lastNeed = n
    i += n
  }
  if (lastStart < 0) return null
  const have = bytes.length - lastStart
  if (have >= lastNeed) return null
  return { bytes: bytes.slice(lastStart), need: lastNeed, before: bytes.slice(0, lastStart) }
}

/** Пробел «·», перевод строки «↵», таб «→»: так их видно в подписях и коде фигур. */
export function visibleText(s: string): string {
  return s.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '→')
}

/** Сколько символов замены U+FFFD («�») в строке. */
export function countReplacement(s: string): number {
  let n = 0
  for (const ch of s) if (ch === '�') n++
  return n
}

/** Человекочитаемое имя длины символа UTF-8. */
export function seqName(n: number): string {
  return n === 2 ? '2-байтный' : n === 3 ? '3-байтный' : n === 4 ? '4-байтный' : 'однобайтный'
}

/** Байт в hex: 0xD1 → «D1». */
export function hexByte(b: number): string {
  return b.toString(16).toUpperCase().padStart(2, '0')
}

/** Код символа в записи Unicode: 0x1F3FD → «U+1F3FD». */
export function codePointName(cp: number): string {
  return 'U+' + cp.toString(16).toUpperCase().padStart(4, '0')
}

/** Вид символа, который сам по себе не рисуется, а меняет соседний (модификатор, соединитель, селектор). */
export type ModifierKind = 'skin' | 'zwj' | 'variation' | 'other'

export function modifierKind(cp: number): ModifierKind | null {
  if (cp >= 0x1f3fb && cp <= 0x1f3ff) return 'skin'
  if (cp === 0x200d) return 'zwj'
  if (cp === 0xfe0f || cp === 0xfe0e) return 'variation'
  if (cp === 0x20e3 || (cp >= 0xe0020 && cp <= 0xe007f)) return 'other'
  return null
}

/** Короткое имя модификатора для подписей. */
export function modifierLabel(kind: ModifierKind): string {
  return kind === 'skin' ? 'оттенок кожи' : kind === 'zwj' ? 'соединитель ZWJ' : kind === 'variation' ? 'селектор варианта' : 'служебный символ'
}

/**
 * Если строка состоит только из модификаторов (их не видно без соседа): список кодов, иначе null.
 * Так приходит, например, оттенок кожи 🏽 отдельным токеном после 👍.
 */
export function onlyModifiers(s: string): Array<{ cp: number; kind: ModifierKind }> | null {
  if (!s) return null
  const out: Array<{ cp: number; kind: ModifierKind }> = []
  for (const ch of s) {
    const cp = ch.codePointAt(0) ?? 0
    const kind = modifierKind(cp)
    if (!kind) return null
    out.push({ cp, kind })
  }
  return out
}

/** Модификаторы записаны кодами («U+1F3FD»), остальное как в visibleText. */
export function shownText(s: string): string {
  let out = ''
  for (const ch of s) {
    const cp = ch.codePointAt(0) ?? 0
    out += modifierKind(cp) ? `[${codePointName(cp)}]` : ch
  }
  return visibleText(out)
}
