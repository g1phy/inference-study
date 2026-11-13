/**
 * Байтовые утилиты для byte-level BPE: отображение байт ↔ символ (GPT-2 bytes_to_unicode),
 * UTF-8 кодирование/декодирование и перевод байтовых смещений в UTF-16.
 *
 * byte-level маппинг: 188 «печатных» байтов (! .. ~, ¡ .. ¬, ® .. ÿ) отображаются сами в себя,
 * остальные 68 (0x00–0x20, 0x7F–0xA0, 0xAD) уходят в U+0100…U+0143. Пробел 0x20 → 'Ġ' (U+0120),
 * \n 0x0A → 'Ċ' (U+010A). Источник: research/notes-tokens-templates-structured.md §1.5 (GPT-2 bytes_to_unicode).
 */

/** Символ byte-level алфавита для каждого байта 0..255. */
export const BYTE_TO_CHAR: readonly string[] = (() => {
  const printable: number[] = []
  for (let b = 0x21; b <= 0x7e; b++) printable.push(b)
  for (let b = 0xa1; b <= 0xac; b++) printable.push(b)
  for (let b = 0xae; b <= 0xff; b++) printable.push(b)
  const out: string[] = new Array(256)
  for (const b of printable) out[b] = String.fromCodePoint(b)
  let n = 0
  for (let b = 0; b < 256; b++) {
    if (out[b] === undefined) {
      out[b] = String.fromCodePoint(256 + n)
      n++
    }
  }
  return out
})()

/** Обратная таблица: code point символа byte-level алфавита → байт (или -1). */
const CHAR_TO_BYTE: Int16Array = (() => {
  const t = new Int16Array(0x144).fill(-1)
  BYTE_TO_CHAR.forEach((ch, b) => {
    t[ch.codePointAt(0)!] = b
  })
  return t
})()

/** Байт для символа byte-level алфавита или -1, если символ не из алфавита. */
export function byteOfChar(ch: string): number {
  const cp = ch.codePointAt(0)!
  return cp < CHAR_TO_BYTE.length ? CHAR_TO_BYTE[cp] : -1
}

/** Байты → строка в byte-level форме словаря ('Ġworld'). */
export function bytesToRaw(bytes: ArrayLike<number>): string {
  let s = ''
  for (let i = 0; i < bytes.length; i++) s += BYTE_TO_CHAR[bytes[i]]
  return s
}

/** byte-level строка → байты; null, если встретился символ не из алфавита. */
export function rawToBytes(raw: string): number[] | null {
  const out: number[] = []
  for (let i = 0; i < raw.length; i++) {
    const cp = raw.charCodeAt(i)
    const b = cp < CHAR_TO_BYTE.length ? CHAR_TO_BYTE[cp] : -1
    if (b < 0) return null
    out.push(b)
  }
  return out
}

const encoder = new TextEncoder()
// ignoreBOM: иначе TextDecoder молча съедает ведущий U+FEFF (токен 3225 Qwen3 = байты EF BB BF стал бы пустым).
const decoder = new TextDecoder('utf-8', { fatal: false, ignoreBOM: true })

/** UTF-8 байты строки (одиночные суррогаты заменяются на U+FFFD, как в TextEncoder). */
export function utf8Bytes(text: string): Uint8Array {
  return encoder.encode(text)
}

/** Декодирование UTF-8 с заменой неполных/битых последовательностей на U+FFFD (как from_utf8_lossy в Rust). */
export function utf8DecodeLossy(bytes: ArrayLike<number>): string {
  return decoder.decode(bytes instanceof Uint8Array ? bytes : Uint8Array.from(bytes as ArrayLike<number>))
}

/** Длина последовательности UTF-8 по первому байту (0 для байта продолжения и невалидных). */
function seqLen(b: number): number {
  if (b < 0x80) return 1
  if (b >= 0xc2 && b <= 0xdf) return 2
  if (b >= 0xe0 && b <= 0xef) return 3
  if (b >= 0xf0 && b <= 0xf4) return 4
  return 0
}

/** Валиден ли байт продолжения на позиции k последовательности с первым байтом lead (RFC 3629). */
function contOk(lead: number, k: number, b: number): boolean {
  if (b < 0x80 || b > 0xbf) return false
  if (k !== 1) return true
  if (lead === 0xe0) return b >= 0xa0
  if (lead === 0xed) return b <= 0x9f
  if (lead === 0xf0) return b >= 0x90
  if (lead === 0xf4) return b <= 0x8f
  return true
}

/**
 * «Мягкое» декодирование для показа: валидные символы как есть, а неполные или битые байты
 * в виде «\xNN» (например, первый байт «п» → '\xD0').
 */
export function utf8DecodeSoft(bytes: ArrayLike<number>): string {
  let out = ''
  let i = 0
  while (i < bytes.length) {
    const b = bytes[i]
    const n = seqLen(b)
    let ok = n > 0 && i + n <= bytes.length
    for (let k = 1; ok && k < n; k++) ok = contOk(b, k, bytes[i + k])
    if (ok) {
      out += utf8DecodeLossy(Array.prototype.slice.call(bytes, i, i + n) as number[])
      i += n
    } else {
      out += '\\x' + b.toString(16).toUpperCase().padStart(2, '0')
      i += 1
    }
  }
  return out
}

/** true, если байты — целиком валидный UTF-8 (без неполных хвостов). */
export function isCompleteUtf8(bytes: ArrayLike<number>): boolean {
  let i = 0
  while (i < bytes.length) {
    const b = bytes[i]
    const n = seqLen(b)
    if (n === 0 || i + n > bytes.length) return false
    for (let k = 1; k < n; k++) if (!contOk(b, k, bytes[i + k])) return false
    i += n
  }
  return true
}

/**
 * Для строки s строит таблицу: для каждого байта её UTF-8 представления — UTF-16 индекс начала
 * и конца символа, которому байт принадлежит. Нужна, чтобы перевести байтовые границы
 * токенов в смещения исходной строки (токен, режущий символ, покрывает этот символ целиком).
 */
export function byteToUtf16Spans(s: string): { start: Int32Array; end: Int32Array } {
  const starts: number[] = []
  const ends: number[] = []
  let u = 0
  for (const ch of s) {
    const cp = ch.codePointAt(0)!
    // Одиночный суррогат TextEncoder кодирует как U+FFFD (3 байта), это попадает в ветку < 0x10000.
    const blen = cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4
    const ulen = ch.length
    for (let k = 0; k < blen; k++) {
      starts.push(u)
      ends.push(u + ulen)
    }
    u += ulen
  }
  return { start: Int32Array.from(starts), end: Int32Array.from(ends) }
}
