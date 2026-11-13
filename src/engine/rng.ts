/**
 * Детерминированный ГПСЧ и хэши для движка (агент A2, «engine-sim»).
 *
 * Никакого Math.random: всё, что «случайно» на сайте (нагрузка, выбор воркера при
 * temperature > 0, синтетические токены), воспроизводится по seed.
 */

/** Генератор: каждый вызов возвращает число в [0, 1). */
export type Rng = () => number

/**
 * mulberry32: маленький 32-битный ГПСЧ (Tommy Ettinger, public domain,
 * https://gist.github.com/tommyettinger/46a874533244883189143505d203312c).
 * Качества хватает для симуляций и анимаций, криптостойкости нет.
 */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Удобная обёртка: seed из числа или строки (строка хэшируется). */
export function makeRng(seed: number | string): Rng {
  return mulberry32(typeof seed === 'string' ? hashString(seed) : seed)
}

/**
 * 32-битный хэш строки (FNV-1a по UTF-16 code units с финальным перемешиванием
 * murmur3 fmix32, чтобы похожие строки давали непохожие хэши).
 * FNV: http://www.isthe.com/chongo/tech/comp/fnv/ ; fmix32: MurmurHash3 (public domain).
 */
export function hashString(s: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return fmix32(h)
}

/** Финальное перемешивание MurmurHash3 (fmix32): лавинный эффект для 32-битного слова. */
export function fmix32(h: number): number {
  h ^= h >>> 16
  h = Math.imul(h, 0x85ebca6b)
  h ^= h >>> 13
  h = Math.imul(h, 0xc2b2ae35)
  h ^= h >>> 16
  return h >>> 0
}

/**
 * 64-битный некриптографический хэш массива 32-битных слов (две независимые
 * 32-битные дорожки в стиле cyrb53, https://github.com/bryc/code/blob/master/jshash/experimental/cyrb53.js,
 * расширено до полных 64 бит). Возвращает [hi, lo] как беззнаковые 32-битные числа.
 * Используется для цепочки хэшей блоков KV (kv.ts): быстро и детерминированно.
 */
export function hash64Words(words: ArrayLike<number>, seed = 0): [number, number] {
  let h1 = (0xdeadbeef ^ seed) >>> 0
  let h2 = (0x41c6ce57 ^ seed) >>> 0
  for (let i = 0; i < words.length; i++) {
    const w = words[i] >>> 0
    h1 = Math.imul(h1 ^ w, 2654435761)
    h2 = Math.imul(h2 ^ w, 1597334677)
    // Перемешиваем дорожки между собой, чтобы порядок слов влиял на обе.
    h1 = (h1 ^ (h2 >>> 15)) >>> 0
    h2 = (h2 ^ (h1 >>> 13)) >>> 0
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909)
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909)
  // Длина входа участвует в хэше: [] и [0] различаются.
  h1 = fmix32((h1 ^ words.length) >>> 0)
  h2 = fmix32((h2 + Math.imul(words.length, 0x9e3779b9)) >>> 0)
  return [h1 >>> 0, h2 >>> 0]
}

/** 32-битное число → 8 hex-символов с ведущими нулями. */
export function hex32(n: number): string {
  return (n >>> 0).toString(16).padStart(8, '0')
}

/**
 * Короткий хэш для показа: 8 hex-символов.
 * Строка хэшируется hashString, число и массив чисел (token ids) хэшируются как 32-битные слова.
 */
export function shortHash(input: string | number | ArrayLike<number>): string {
  if (typeof input === 'string') return hex32(hashString(input))
  const words = typeof input === 'number' ? [input] : input
  return hex32(hash64Words(words)[0])
}

/** Целое в [lo, hi] включительно. */
export function randInt(rng: Rng, lo: number, hi: number): number {
  return lo + Math.floor(rng() * (hi - lo + 1))
}

/** Случайный элемент массива (массив не должен быть пустым). */
export function pick<T>(rng: Rng, arr: readonly T[]): T {
  if (arr.length === 0) throw new Error('pick: пустой массив')
  return arr[Math.floor(rng() * arr.length)]
}

/** Выбор индекса по весам (веса неотрицательные, хотя бы один > 0). */
export function pickWeighted(rng: Rng, weights: readonly number[]): number {
  let total = 0
  for (const w of weights) total += Math.max(0, w)
  if (!(total > 0)) throw new Error('pickWeighted: все веса нулевые')
  let r = rng() * total
  for (let i = 0; i < weights.length; i++) {
    r -= Math.max(0, weights[i])
    if (r < 0) return i
  }
  // Защита от ошибок округления: последний элемент с положительным весом.
  for (let i = weights.length - 1; i >= 0; i--) if (weights[i] > 0) return i
  return weights.length - 1
}

/** Перемешивание Фишера–Йетса; возвращает НОВЫЙ массив, исходный не меняется. */
export function shuffle<T>(rng: Rng, arr: readonly T[]): T[] {
  const out = arr.slice()
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    const tmp = out[i]
    out[i] = out[j]
    out[j] = tmp
  }
  return out
}

/** Нормальное распределение (преобразование Бокса–Мюллера). */
export function normal(rng: Rng, mean = 0, sd = 1): number {
  let u = 0
  while (u === 0) u = rng() // log(0) недопустим
  const v = rng()
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

/** Логнормальное распределение, заданное медианой и sigma (стандартное отклонение ln X). */
export function logNormal(rng: Rng, median: number, sigma: number): number {
  return median * Math.exp(sigma * normal(rng))
}

/** Экспоненциальное распределение с интенсивностью rate (среднее 1 / rate): интервалы пуассоновского потока. */
export function exponential(rng: Rng, rate: number): number {
  let u = 0
  while (u === 0) u = rng()
  return -Math.log(u) / rate
}
