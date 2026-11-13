/**
 * Условные (псевдослучайные) векторы для иллюстраций: настоящие веса моделей мы в браузер не грузим.
 * Значения детерминированы: один и тот же ID всегда даёт один и тот же вектор, поэтому повторы токена
 * в промпте видны как одинаковые «штрих-коды». Масштаб условный.
 */
import { fmix32, mulberry32, normal } from '@/engine/rng'

/** Сколько первых координат показываем в штрих-коде (из 4096 и т. п.). */
export const BARCODE_DIMS = 64

/** Множитель для подписей чисел: веса эмбеддингов порядка сотых (условно). */
export const VALUE_SCALE = 0.02

const cache = new Map<string, number[]>()

/**
 * Условный вектор длины `dims` для пары (seed, salt): нормальное распределение N(0, 1).
 * salt разделяет независимые семейства векторов (эмбеддинги, вклады слоёв и т. п.).
 */
export function pseudoVector(seed: number, dims = BARCODE_DIMS, salt = 0): number[] {
  const key = `${seed}|${dims}|${salt}`
  const hit = cache.get(key)
  if (hit) return hit
  const rng = mulberry32(fmix32(((seed >>> 0) ^ Math.imul(salt + 1, 0x9e3779b9)) >>> 0))
  const out = Array.from({ length: dims }, () => normal(rng, 0, 1))
  if (cache.size > 4000) cache.clear()
  cache.set(key, out)
  return out
}

/** Строка таблицы эмбеддингов для ID токена (условно). */
export function embeddingRow(tokenId: number, dims = BARCODE_DIMS): number[] {
  return pseudoVector(tokenId, dims, 0)
}

/** Евклидова норма. */
export function norm(v: readonly number[]): number {
  let s = 0
  for (const x of v) s += x * x
  return Math.sqrt(s)
}

/** Поэлементная сумма (длина по первому вектору). */
export function addVec(a: readonly number[], b: readonly number[], k = 1): number[] {
  return a.map((x, i) => x + k * (b[i] ?? 0))
}
