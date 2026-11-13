/**
 * Данные игрушечного примера главы (SPEC §6.6): «Кошка не перепрыгнула забор, потому что она устала».
 * Все числа берутся из движка (TOY_ATTENTION, attention(), dotTerms), здесь только удобные выборки
 * для фигур: строки, столбцы, слагаемые скалярных произведений, шаг decode по кэшу.
 *
 * Опыт «а если бы забор был женского рода»: вариант OGRADA заменяет «забор» на «ограду»
 * (существительное женского рода) и пересчитывает весь след attention() с теми же матрицами весов.
 */
import { TOY_ATTENTION, attention, dotTerms, mat } from '@/engine/attention'
import type { AttentionTrace, Matrix } from '@/engine/types'

export const TOY = TOY_ATTENTION
export const TRACE = TOY.trace
export const WORDS: readonly string[] = TOY.tokens
export const N = WORDS.length
export const FEATURES: readonly string[] = TOY.featureNames
export const SQRT_DK = Math.sqrt(TOY.dK)
export const FOCUS = TOY.focus

/** Какая проекция: запрос, ключ или значение. */
export type Proj = 'q' | 'k' | 'v'

export const PROJ_W: Record<Proj, Matrix> = { q: TOY.Wq, k: TOY.Wk, v: TOY.Wv }
export const PROJ_DIMS: Record<Proj, readonly string[]> = {
  q: TOY.queryDimNames,
  k: TOY.keyDimNames,
  v: TOY.valueDimNames,
}

/* ============================================================================
 * Варианты предложения: «забор» (как в движке) и «ограду» (тот же расчёт, другой род)
 * ========================================================================== */

export type Variant = 'zabor' | 'ograda'

/** Формы слова в столбце FOCUS.distractor: «забор» или «ограда». */
export interface WordForms {
  nom: string
  gen: string
  dat: string
  ins: string
}

export interface ToyCase {
  variant: Variant
  words: readonly string[]
  X: Matrix
  trace: AttentionTrace
  other: WordForms
}

export const ZABOR: ToyCase = {
  variant: 'zabor',
  words: WORDS,
  X: TOY.X,
  trace: TRACE,
  other: { nom: 'забор', gen: 'забора', dat: 'забору', ins: 'забором' },
}

/** «Ограда»: существительное женского рода, признаки [сущ, жен, мест, глаг] = [1, 1, 0, 0]. */
const X_OGRADA: Matrix = TOY.X.map((r, i) => (i === FOCUS.distractor ? [1, 1, 0, 0] : r.slice()))

export const OGRADA: ToyCase = {
  variant: 'ograda',
  words: WORDS.map((w, i) => (i === FOCUS.distractor ? 'ограду' : w)),
  X: X_OGRADA,
  trace: attention(X_OGRADA, TOY.Wq, TOY.Wk, TOY.Wv, { causal: true }),
  other: { nom: 'ограда', gen: 'ограды', dat: 'ограде', ins: 'оградой' },
}

export const CASES: Record<Variant, ToyCase> = { zabor: ZABOR, ograda: OGRADA }

/** Q, K или V варианта. */
export function projOut(proj: Proj, c: ToyCase = ZABOR): Matrix {
  return proj === 'q' ? c.trace.Q : proj === 'k' ? c.trace.K : c.trace.V
}

/* ============================================================================
 * Выборки для фигуры по шагам
 * ========================================================================== */

/** Веса, которые отличаются меньше чем на TIE, считаем равными (softmax равных оценок). */
export const TIE = 1e-9

/** Столбец матрицы. */
export function column(M: Matrix, j: number): number[] {
  return M.map((row) => row[j])
}

/** Номера признаков, которые у слова равны 1 (X из нулей и единиц). */
export function activeFeatures(row: number, c: ToyCase = ZABOR): number[] {
  const out: number[] = []
  c.X[row].forEach((x, f) => {
    if (x !== 0) out.push(f)
  })
  return out
}

/** Ячейка проекции (слово row, измерение dim): x_row · W[:, dim]. */
export function projectionTerms(proj: Proj, row: number, dim: number, c: ToyCase = ZABOR) {
  return dotTerms(c.X[row], column(PROJ_W[proj], dim))
}

/** Оценка «row смотрит на col»: q_row · k_col. */
export function scoreTerms(row: number, col: number, c: ToyCase = ZABOR) {
  return dotTerms(c.trace.Q[row], c.trace.K[col])
}

/** Выход (слово row, измерение dim): Σ_j w_row,j · v_j[dim]. */
export function outTerms(row: number, dim: number, c: ToyCase = ZABOR) {
  return dotTerms(c.trace.weights[row], column(c.trace.V, dim))
}

/** Экспоненты строки после маски (без вычитания максимума: числа здесь маленькие) и их сумма. */
export function softmaxParts(row: number, c: ToyCase = ZABOR): { exps: number[]; sum: number } {
  const exps = c.trace.masked[row].map((x) => (x === -Infinity ? 0 : Math.exp(x)))
  return { exps, sum: exps.reduce((a, b) => a + b, 0) }
}

/** Индекс самого большого веса в строке (среди видимых слов; при равенстве левый). */
export function argmaxWeight(row: number, c: ToyCase = ZABOR): number {
  let best = 0
  for (let j = 0; j <= row; j++) if (c.trace.weights[row][j] > c.trace.weights[row][best] + TIE) best = j
  return best
}

/** Слова строки, отсортированные по весу (только видимые: j ≤ row). */
export function rankedRow(row: number, c: ToyCase = ZABOR): Array<{ index: number; word: string; weight: number }> {
  return c.words
    .slice(0, row + 1)
    .map((word, index) => ({ index, word, weight: c.trace.weights[row][index] }))
    .sort((a, b) => (Math.abs(b.weight - a.weight) > TIE ? b.weight - a.weight : a.index - b.index))
}

/** Все видимые веса строки одинаковые (строка из одних нулевых оценок). */
export function isUniformRow(row: number, c: ToyCase = ZABOR): boolean {
  const w = c.trace.weights[row].slice(0, row + 1)
  return Math.max(...w) - Math.min(...w) < TIE
}

/** Слова строки с самым большим весом (несколько, если веса равны). */
export function topIndices(row: number, c: ToyCase = ZABOR): number[] {
  const w = c.trace.weights[row]
  const max = Math.max(...w.slice(0, row + 1))
  return w.slice(0, row + 1).flatMap((x, j) => (max - x < TIE ? [j] : []))
}

/**
 * Начало рейтинга строки для подписи: первые три слова и все, кто делит вес с третьим,
 * плюс признак, что после них остались слова с весом строго меньше.
 */
export function rankedHead(row: number, c: ToyCase = ZABOR, k = 3): { head: ReturnType<typeof rankedRow>; more: boolean } {
  const r = rankedRow(row, c)
  if (r.length <= k) return { head: r, more: false }
  const edge = r[k - 1].weight
  let end = k
  while (end < r.length && Math.abs(r[end].weight - edge) < TIE) end++
  return { head: r.slice(0, end), more: end < r.length }
}

/* ============================================================================
 * Шаг decode по KV cache (фигура «мост к KV cache»)
 * ========================================================================== */

/** Сколько слов считаем промптом: «Кошка не перепрыгнула забор, потому что». Дальше генерация. */
export const PROMPT_LEN = 7

export interface DecodeRow {
  /** Номер нового токена. */
  index: number
  /** Его запрос, ключ и значение: считаются только для него. */
  q: number[]
  k: number[]
  v: number[]
  /** Оценки против всех ключей из кэша (и своего), после деления на √d_k. */
  scaled: number[]
  weights: number[]
  out: number[]
}

/**
 * Одна строка attention для нового токена: q нового токена против K из кэша (плюс его собственный k),
 * softmax, смесь V. Считается заново, без готового следа из TRACE, чтобы показать, что результат
 * совпадает со строкой полного треугольника (проверено тестом).
 */
export function decodeRow(index: number): DecodeRow {
  const x = [TOY.X[index]]
  const q = mat.matmul(x, TOY.Wq)[0]
  const k = mat.matmul(x, TOY.Wk)[0]
  const v = mat.matmul(x, TOY.Wv)[0]
  const cacheK = [...mat.matmul(TOY.X.slice(0, index), TOY.Wk), k]
  const cacheV = [...mat.matmul(TOY.X.slice(0, index), TOY.Wv), v]
  const scaled = cacheK.map((kj) => mat.dot(q, kj) / SQRT_DK)
  const weights = mat.softmaxRows([scaled])[0]
  const out = mat.matmul([weights], cacheV)[0]
  return { index, q, k, v, scaled, weights, out }
}

/** Пары (запрос, ключ) в causal-треугольнике из n токенов: n(n+1)/2. */
export function trianglePairs(n: number): number {
  return (n * (n + 1)) / 2
}
