/**
 * Блочная квантизация вектора весов в FP4 (E2M1) для фигуры «один выброс портит всё».
 *
 * Три схемы масштаба (research/notes-techniques.md §1.1.5, §3.3, §3.4):
 * - 'tensor': один масштаб FP32 на весь вектор: s = amax / 6;
 * - 'mxfp4':  блоки по 32, масштаб E8M0 (только степень двойки): s = 2^(floor(log2 amax_b) − 2),
 *             где 2 = максимальная степень E2M1 (правило OCP MX, Rouhani и др. 2023, алгоритм 1):
 *             после деления на s максимум блока попадает в [4, 8), всё выше 6 насыщается
 *             (clamping error); верх сетки пустует, только если amax_b / s < 5;
 * - 'nvfp4':  блоки по 16, масштаб E4M3 (дробный) плюс глобальный FP32 на тензор:
 *             s_global = amax / (6 · 448), код блока = E4M3((amax_b / 6) / s_global), s_b = код · s_global.
 * Бит на значение: 4 + биты масштаба / размер блока (NVFP4 4,5; MXFP4 4,25).
 */
import { getFormat, quantize, roundHalfEven } from './formats'
import { mulberry32 } from '@/engine/rng'

export type BlockScheme = 'tensor' | 'mxfp4' | 'nvfp4'

export interface BlockSchemeInfo {
  id: BlockScheme
  label: string
  blockSize: number | null
  scaleBits: number
  scaleName: string
}

export const BLOCK_SCHEMES: readonly BlockSchemeInfo[] = [
  { id: 'tensor', label: 'Один масштаб', blockSize: null, scaleBits: 32, scaleName: 'FP32' },
  { id: 'mxfp4', label: 'MXFP4', blockSize: 32, scaleBits: 8, scaleName: 'E8M0, степень двойки' },
  { id: 'nvfp4', label: 'NVFP4', blockSize: 16, scaleBits: 8, scaleName: 'E4M3 + FP32 на тензор' },
]

const E2M1 = getFormat('e2m1')
const E4M3 = getFormat('e4m3')
const FP4_MAX = 6
const FP4_EMAX = 2
const E4M3_MAX = 448

export interface QuantBlock {
  start: number
  end: number
  amax: number
  /** Применённый масштаб блока (после квантования самого масштаба). */
  scale: number
}

export interface BlockQuantResult {
  scheme: BlockScheme
  q: number[]
  blocks: QuantBlock[]
  mse: number
  /** Среднеквадратичная ошибка относительно среднего квадрата значений. */
  relMse: number
  /** Ненулевые значения, ставшие нулём. */
  zeroed: number
  /** Значения, упёршиеся в верх сетки (насыщение). */
  clipped: number
  maxAbsError: number
  bitsPerValue: number
}

function amaxOf(xs: readonly number[]): number {
  return xs.reduce((m, v) => Math.max(m, Math.abs(v)), 0)
}

/** Квантовать вектор в FP4 E2M1 по схеме масштабов. */
export function quantizeBlocks(values: readonly number[], scheme: BlockScheme): BlockQuantResult {
  const info = BLOCK_SCHEMES.find((s) => s.id === scheme)!
  const n = values.length
  const size = info.blockSize ?? n
  const tensorAmax = amaxOf(values)
  const globalScale = tensorAmax > 0 ? tensorAmax / (FP4_MAX * E4M3_MAX) : 1
  const q: number[] = new Array(n).fill(0)
  const blocks: QuantBlock[] = []
  let clipped = 0
  for (let start = 0; start < n; start += size) {
    const end = Math.min(n, start + size)
    const slice = values.slice(start, end)
    const amax = amaxOf(slice)
    let scale: number
    if (amax === 0) scale = 1
    else if (scheme === 'tensor') scale = amax / FP4_MAX
    else if (scheme === 'mxfp4') scale = Math.pow(2, Math.floor(Math.log2(amax)) - FP4_EMAX)
    else {
      const code = quantize(amax / FP4_MAX / globalScale, E4M3).value
      scale = code * globalScale
    }
    for (let i = start; i < end; i++) {
      const r = quantize(values[i] / scale, E2M1)
      if (r.overflow) clipped++
      // −0 → 0: для графика и подписей знак нуля не важен
      q[i] = r.value * scale || 0
    }
    blocks.push({ start, end, amax, scale })
  }
  let se = 0
  let sq = 0
  let zeroed = 0
  let maxAbsError = 0
  for (let i = 0; i < n; i++) {
    const e = q[i] - values[i]
    se += e * e
    sq += values[i] * values[i]
    maxAbsError = Math.max(maxAbsError, Math.abs(e))
    if (values[i] !== 0 && q[i] === 0) zeroed++
  }
  const mse = n ? se / n : 0
  return {
    scheme,
    q,
    blocks,
    mse,
    relMse: sq > 0 ? se / sq : 0,
    zeroed,
    clipped,
    maxAbsError,
    bitsPerValue: 4 + info.scaleBits / size,
  }
}

/**
 * Условные веса одного ряда матрицы: 64 значения ~ N(0, 0,04), округлённые до 3 знаков
 * (детерминированно, seed). С выбросом одно значение заменяется большим (как «outlier» каналы LLM).
 */
export function sampleWeights(opts: { outlier: boolean; seed?: number; n?: number; outlierIndex?: number; outlierValue?: number }): number[] {
  const n = opts.n ?? 64
  const rng = mulberry32(opts.seed ?? 7)
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    // Бокс-Мюллер
    const u = Math.max(1e-9, rng())
    const v = rng()
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
    out.push(roundHalfEven(Math.abs(z * 0.04) * 1000) / 1000 * Math.sign(z))
  }
  if (opts.outlier) out[opts.outlierIndex ?? 5] = opts.outlierValue ?? 0.9
  return out
}
