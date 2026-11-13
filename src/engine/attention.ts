/**
 * Матричная математика attention, игрушечный пример для главы 6 и паттерны голов (агент A2).
 *
 * Формула: softmax(Q·Kᵀ / √d_k + M)·V с causal mask M (−∞ над диагональю): токен не видит будущее
 * (research/notes-models-attention-bottlenecks.md §1.11, §2.3 п. 5, §3.5–3.6).
 */
import type { AttentionTrace, HeadPattern, Matrix, Token } from './types'

/* ============================================================================
 * Матричные помощники
 * ========================================================================== */

/** Матрица r × c из нулей (или из value). */
export function zeros(rows: number, cols: number, value = 0): Matrix {
  return Array.from({ length: rows }, () => new Array<number>(cols).fill(value))
}

/** Размер матрицы [строки, столбцы]. */
export function shape(A: Matrix): [number, number] {
  return [A.length, A.length ? A[0].length : 0]
}

/** Произведение матриц A (n × k) · B (k × m). */
export function matmul(A: Matrix, B: Matrix): Matrix {
  const [n, k] = shape(A)
  const [k2, m] = shape(B)
  if (k !== k2) throw new Error(`matmul: несовместимые размеры ${n}×${k} и ${k2}×${m}`)
  const out = zeros(n, m)
  for (let i = 0; i < n; i++) {
    for (let t = 0; t < k; t++) {
      const a = A[i][t]
      if (a === 0) continue
      for (let j = 0; j < m; j++) out[i][j] += a * B[t][j]
    }
  }
  return out
}

/** Транспонирование. */
export function transpose(A: Matrix): Matrix {
  const [n, m] = shape(A)
  const out = zeros(m, n)
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[j][i] = A[i][j]
  return out
}

/** Поэлементная сумма (−∞ + x = −∞). */
export function add(A: Matrix, B: Matrix): Matrix {
  return A.map((row, i) => row.map((x, j) => x + B[i][j]))
}

/** Умножение на число. */
export function scale(A: Matrix, s: number): Matrix {
  return A.map((row) => row.map((x) => x * s))
}

/** Поэлементное преобразование. */
export function mapMatrix(A: Matrix, f: (x: number, i: number, j: number) => number): Matrix {
  return A.map((row, i) => row.map((x, j) => f(x, i, j)))
}

/** Causal mask n × n: true там, где j ≤ i (токен видит себя и прошлое). */
export function causalMask(n: number): boolean[][] {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => j <= i))
}

/** Применяет маску: где mask = false, ставит −∞. */
export function applyMask(A: Matrix, mask: boolean[][]): Matrix {
  return A.map((row, i) => row.map((x, j) => (mask[i][j] ? x : -Infinity)))
}

/**
 * Softmax по строкам (численно устойчиво: вычитаем максимум строки). −∞ и замаскированные
 * позиции (mask = false) получают вес 0. Строка целиком из −∞ даёт нули.
 */
export function softmaxRows(A: Matrix, mask?: boolean[][]): Matrix {
  return A.map((row, i) => {
    const vals = row.map((x, j) => (mask && !mask[i][j] ? -Infinity : x))
    const m = Math.max(...vals)
    if (m === -Infinity) return vals.map(() => 0)
    const e = vals.map((x) => (x === -Infinity ? 0 : Math.exp(x - m)))
    const s = e.reduce((a, b) => a + b, 0)
    return e.map((x) => x / s)
  })
}

/** Скалярное произведение. */
export function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

/** Слагаемые скалярного произведения (для подсказки «как посчитана ячейка»): a_i·b_i. */
export function dotTerms(a: readonly number[], b: readonly number[]): { terms: number[]; sum: number } {
  const terms = a.map((x, i) => x * b[i])
  return { terms, sum: terms.reduce((s, x) => s + x, 0) }
}

/** Округление всех элементов (для показа). */
export function roundMatrix(A: Matrix, digits = 2): Matrix {
  const f = 10 ** digits
  return A.map((row) => row.map((x) => (Number.isFinite(x) ? Math.round(x * f) / f : x)))
}

/** Суммы строк. */
export function rowSums(A: Matrix): number[] {
  return A.map((row) => row.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0))
}

/** Все помощники одним объектом: mat.matmul, mat.softmaxRows, … */
export const mat = {
  zeros,
  shape,
  matmul,
  transpose,
  add,
  scale,
  map: mapMatrix,
  causalMask,
  applyMask,
  softmaxRows,
  dot,
  dotTerms,
  round: roundMatrix,
  rowSums,
}

/* ============================================================================
 * Attention
 * ========================================================================== */

/**
 * Одна голова attention по шагам: Q = X·W_Q, K = X·W_K, V = X·W_V, scores = Q·Kᵀ, деление на √d_k,
 * causal mask (−∞ над диагональю), softmax по строкам, выход = веса·V.
 */
export function attention(X: Matrix, Wq: Matrix, Wk: Matrix, Wv: Matrix, opts: { causal?: boolean; scale?: boolean } = {}): AttentionTrace {
  const causal = opts.causal !== false
  const Q = matmul(X, Wq)
  const K = matmul(X, Wk)
  const V = matmul(X, Wv)
  const rawScores = matmul(Q, transpose(K))
  const dK = shape(K)[1]
  const scaled = opts.scale === false ? rawScores.map((r) => r.slice()) : scale(rawScores, 1 / Math.sqrt(dK))
  const masked = causal ? applyMask(scaled, causalMask(X.length)) : scaled.map((r) => r.slice())
  const weights = softmaxRows(masked)
  const out = matmul(weights, V)
  return { Q, K, V, rawScores, scaled, masked, weights, out }
}

/* ============================================================================
 * Игрушечный пример для главы 6 (SPEC §6.6)
 * ========================================================================== */

/** Игрушечный пример attention с подобранными вручную матрицами. */
export interface ToyAttention {
  /** Слова предложения (упрощённо: по словам, не по токенам). */
  tokens: string[]
  /** Имена признаков d_model (столбцы X, строки W). */
  featureNames: string[]
  /** Имена измерений Q, K и V (столбцы W_Q, W_K, W_V). */
  queryDimNames: string[]
  keyDimNames: string[]
  valueDimNames: string[]
  X: Matrix
  Wq: Matrix
  Wk: Matrix
  Wv: Matrix
  dK: number
  /** Индексы для подписи главного вывода: строка «она», столбцы «Кошка» и «забор». */
  focus: { query: number; target: number; distractor: number }
  /** Объяснения по-русски к каждому шагу. */
  explanations: Record<'X' | 'Wq' | 'Wk' | 'Wv' | 'Q' | 'K' | 'V' | 'scores' | 'scale' | 'mask' | 'softmax' | 'out', string>
  /** Пометка об упрощениях. */
  note: string
  /** Полный расчёт (causal, с делением на √d_k). */
  trace: AttentionTrace
}

const TOY_TOKENS = ['Кошка', 'не', 'перепрыгнула', 'забор', ',', 'потому', 'что', 'она', 'устала']
const TOY_FEATURES = ['существительное', 'женский род', 'местоимение', 'глагол']

// Признаки слов: 1 = «да». Служебные слова и запятая без признаков.
//                         сущ  жен  мест глаг
const TOY_X: Matrix = [
  [1, 1, 0, 0], // Кошка
  [0, 0, 0, 0], // не
  [0, 1, 0, 1], // перепрыгнула (глагол прошедшего времени женского рода)
  [1, 0, 0, 0], // забор (мужской род)
  [0, 0, 0, 0], // ,
  [0, 0, 0, 0], // потому
  [0, 0, 0, 0], // что
  [0, 1, 1, 0], // она
  [0, 1, 0, 1], // устала (глагол женского рода)
]

// W_Q: что слово ищет. Столбцы: «ищу предмет», «ищу женский род».
const TOY_WQ: Matrix = [
  [-0.5, 0], // существительное слегка ищет глагол (у глагола минус по «предмету»)
  [0, 1], // женский род ищет женский род (согласование)
  [2, 2], // местоимение ищет существительное и род: «на кого я указываю?»
  [1, 1], // глагол ищет подлежащее
]

// W_K: по чему слово находят. Столбцы: «я предмет», «я женского рода».
const TOY_WK: Matrix = [
  [1, 0], // существительное: «я предмет»
  [0, 1], // женский род
  [0, 0], // местоимение само по себе не предмет
  [-1, 0], // глагол точно не предмет
]

// W_V: что слово передаст. Столбцы: «существительное», «женский род».
const TOY_WV: Matrix = [
  [1, 0],
  [0, 1],
  [0, 0],
  [0, 0],
]

/**
 * Игрушечный пример главы 6: «Кошка не перепрыгнула забор, потому что она устала».
 * Матрицы подобраны так, что с causal mask и делением на √2 строка «она» отдаёт «Кошке» 65 %
 * веса, а «забору» 8 %: по роду «она» указывает на кошку. Запятая оставлена отдельным токеном:
 * так режут текст настоящие токенизаторы, и видно, что пунктуация здесь почти не получает внимания.
 */
export const TOY_ATTENTION: ToyAttention = {
  tokens: TOY_TOKENS,
  featureNames: TOY_FEATURES,
  queryDimNames: ['ищу предмет', 'ищу женский род'],
  keyDimNames: ['я предмет', 'я женского рода'],
  valueDimNames: ['существительное', 'женский род'],
  X: TOY_X,
  Wq: TOY_WQ,
  Wk: TOY_WK,
  Wv: TOY_WV,
  dK: 2,
  focus: { query: 7, target: 0, distractor: 3 },
  explanations: {
    X:
      'Каждое слово это вектор из четырёх признаков: 1 значит «да», 0 значит «нет». «Кошка»: существительное женского рода, «забор»: мужского, «она»: местоимение женского рода. У служебных слов и запятой признаков нет.',
    Wq: 'W_Q превращает признаки в запрос «что я ищу». Местоимение ищет предмет женского рода, глагол ищет, кто действует, слово женского рода ищет пару по роду.',
    Wk: 'W_K превращает признаки в ключ «по чему меня найти». Существительное отвечает «я предмет», женский род отвечает «я женского рода». У глагола минус по «предмету»: «она» не может указывать на действие.',
    Wv: 'W_V решает, что слово передаст тем, кто его выбрал: здесь признаки «существительное» и «женский род».',
    Q: 'Q = X·W_Q: строка это запрос слова. У «она» запрос [2, 3]: сильно ищет предмет и женский род.',
    K: 'K = X·W_K: строка это ключ слова. У «Кошки» [1, 1]: предмет и женский род, у «забора» [1, 0]: только предмет.',
    V: 'V = X·W_V: строка это то, что слово отдаст. У «Кошки» [1, 1], у «забора» [1, 0].',
    scores: 'Q·Kᵀ: насколько запрос строки подходит к ключу столбца (скалярное произведение). «Она»·«Кошка» = 2·1 + 3·1 = 5, «она»·«забор» = 2·1 + 3·0 = 2.',
    scale: 'Делим на √d_k = √2 ≈ 1,41, чтобы при большой размерности softmax не превращался в «всё или ничего».',
    mask: 'Causal mask: над диагональю −∞. Слово видит только себя и слова слева: при генерации будущего ещё нет.',
    softmax: 'Softmax по строке превращает оценки в веса, которые в сумме дают 1. Строка «она»: 65 % на «Кошку», 8 % на «забор».',
    out: 'Выход = веса·V: смесь значений тех, на кого слово смотрит; она прибавляется к вектору слова (через W_O и residual stream). «Она» получила признак «существительное» почти целиком от «Кошки».',
  },
  note:
    'Упрощённо: по словам, а не по токенам; одна голова, d_model = 4 признака с понятными именами, d_k = 2. В настоящей модели признаки не названы, голов 32 на слой и 36 слоёв (Qwen3-8B).',
  trace: attention(TOY_X, TOY_WQ, TOY_WK, TOY_WV, { causal: true }),
}

/* ============================================================================
 * Паттерны голов для любого промпта (SPEC §6.6, фигура 2)
 * ========================================================================== */

/** Минимум полей токена, нужный для паттернов. */
export type PatternToken = Pick<Token, 'id' | 'text'> & Partial<Pick<Token, 'special'>>

/** Разделитель: спецтокен, перевод строки или чистая пунктуация. */
function isDelimiter(t: PatternToken): boolean {
  if (t.special) return true
  const s = t.text
  if (s.includes('\n')) return true
  const trimmed = s.trim()
  return trimmed.length > 0 && /^[\p{P}]+$/u.test(trimmed)
}

/** Нормирует строку на 1 (если сумма 0, весь вес на первый токен: «стоянка» внимания). */
function normalizeRow(row: number[]): number[] {
  const s = row.reduce((a, b) => a + b, 0)
  if (!(s > 0)) return row.map((_, j) => (j === 0 ? 1 : 0))
  return row.map((x) => x / s)
}

/** Строит causal-матрицу по функции весов строки i (ненормированных). */
function buildPattern(n: number, rowFn: (i: number) => number[]): Matrix {
  return Array.from({ length: n }, (_, i) => {
    const raw = rowFn(i)
    const row = new Array<number>(n).fill(0)
    for (let j = 0; j <= i; j++) row[j] = Math.max(0, raw[j] ?? 0)
    return normalizeRow(row)
  })
}

/**
 * Иллюстративные паттерны голов для любого промпта: предыдущий токен, attention sink на первый
 * токен, локальное окно, «тот же токен раньше», induction, разделители, почти равномерное внимание.
 * Все матрицы causal (верхний треугольник 0), строки в сумме 1. Это схемы типичного поведения
 * голов, а не веса настоящей модели: доли (15 % на первый токен, ≈ 75 % у sink-головы), окно
 * в 4 токена и затухания иллюстративные, подобраны для наглядности.
 */
export function headPatterns(tokens: readonly PatternToken[]): HeadPattern[] {
  const n = tokens.length
  const SINK = 0.15 // доля внимания, которая у реальных голов «паркуется» на первом токене
  const withSink = (i: number, main: number[]): number[] => {
    const row = main.slice()
    const s = row.reduce((a, b) => a + b, 0)
    if (!(s > 0)) return row
    for (let j = 0; j < row.length; j++) row[j] = (row[j] / s) * (1 - SINK)
    row[0] = (row[0] ?? 0) + SINK
    return row
  }
  const earlierSame = (i: number): number[] => {
    const out: number[] = []
    for (let j = 0; j < i; j++) if (tokens[j].id === tokens[i].id) out.push(j)
    return out
  }
  const recencyWeights = (positions: number[], i: number): number[] => {
    const row = new Array<number>(i + 1).fill(0)
    for (const p of positions) row[p] += Math.exp(-(i - p) / 8)
    return row
  }

  const patterns: HeadPattern[] = [
    {
      id: 'previous-token',
      name: 'Предыдущий токен',
      description: 'Каждый токен смотрит на соседа слева. Такие головы помогают собирать слова из кусочков и замечать пары токенов.',
      weights: buildPattern(n, (i) => {
        const row = new Array<number>(i + 1).fill(0)
        row[Math.max(0, i - 1)] = 1
        return i >= 2 ? withSink(i, row) : row
      }),
    },
    {
      id: 'attention-sink',
      name: 'Attention sink: первый токен',
      description:
        'Большая часть внимания уходит на самый первый токен, даже если он ничего не значит. Голове «нечего искать», а softmax обязан куда-то отдать вес, вот она и паркуется на начале.',
      weights: buildPattern(n, (i) => {
        const row = new Array<number>(i + 1).fill(0)
        for (let j = 1; j <= i; j++) row[j] = 0.3 * Math.exp(-(i - j) / 2)
        const rest = row.reduce((a, b) => a + b, 0)
        row[0] = rest > 0 ? (rest * 0.75) / 0.25 : 1 // ≈ 75 % на первый токен
        return row
      }),
    },
    {
      id: 'local-window',
      name: 'Локальное окно',
      description: 'Внимание на несколько ближайших токенов слева, чем ближе, тем сильнее. Так модель собирает ближний контекст: словосочетания и окончания.',
      weights: buildPattern(n, (i) => {
        const row = new Array<number>(i + 1).fill(0)
        for (let j = Math.max(0, i - 3); j <= i; j++) row[j] = Math.exp(-(i - j) / 1.5)
        return row
      }),
    },
    {
      id: 'duplicate-token',
      name: 'Тот же токен раньше',
      description: 'Токен ищет, где он уже встречался в тексте. Если нигде, внимание уходит на первый токен.',
      weights: buildPattern(n, (i) => {
        const prev = earlierSame(i)
        if (prev.length === 0) return [1]
        return withSink(i, recencyWeights(prev, i))
      }),
    },
    {
      id: 'induction',
      name: 'Induction: что было после такого же токена',
      description:
        'Токен находит своё прошлое появление и смотрит на токен, который шёл сразу после него: «в прошлый раз после A было B, значит, сейчас, скорее всего, тоже B». На этом держится копирование из контекста.',
      weights: buildPattern(n, (i) => {
        const prev = earlierSame(i)
        if (prev.length === 0) return [1]
        return withSink(
          i,
          recencyWeights(
            prev.map((p) => p + 1),
            i,
          ),
        )
      }),
    },
    {
      id: 'delimiter',
      name: 'Разделители и пунктуация',
      description: 'Внимание на ближайшие знаки препинания, переводы строки и спецтокены шаблона: так голова следит за границами фраз и сообщений.',
      weights: buildPattern(n, (i) => {
        const delims: number[] = []
        for (let j = 0; j <= i; j++) if (isDelimiter(tokens[j])) delims.push(j)
        if (delims.length === 0) return [1]
        const row = new Array<number>(i + 1).fill(0)
        delims.forEach((p, k) => (row[p] = Math.pow(0.5, delims.length - 1 - k)))
        return withSink(i, row)
      }),
    },
    {
      id: 'broad',
      name: 'Почти равномерное внимание',
      description: 'Вес размазан по всем предыдущим токенам примерно поровну: голова собирает «общий смысл» всего контекста.',
      weights: buildPattern(n, (i) => new Array<number>(i + 1).fill(1)),
    },
  ]
  return patterns
}

/* ============================================================================
 * Multi-head и GQA
 * ========================================================================== */

/** Группа GQA: одна KV-голова и Q-головы, которые её делят. */
export interface GqaGroup {
  kvHead: number
  queryHeads: number[]
}

/**
 * Раскладка Q-голов по KV-головам: при GQA каждые heads / kvHeads соседних Q-голов делят одну
 * пару K/V (Qwen3-8B: 32 Q-головы на 8 KV-голов, по 4 в группе). kvHeads = heads — MHA, 1 — MQA.
 */
export function gqaGroups(heads: number, kvHeads: number): GqaGroup[] {
  if (!Number.isInteger(heads) || !Number.isInteger(kvHeads) || heads < 1 || kvHeads < 1) {
    throw new Error('gqaGroups: число голов должно быть целым ≥ 1')
  }
  if (heads % kvHeads !== 0) throw new Error(`gqaGroups: ${heads} Q-голов не делятся на ${kvHeads} KV-голов поровну`)
  const g = heads / kvHeads
  return Array.from({ length: kvHeads }, (_, k) => ({ kvHead: k, queryHeads: Array.from({ length: g }, (_, j) => k * g + j) }))
}

/** Вид attention по числу голов: MHA (kv = heads), MQA (kv = 1) или GQA. */
export function attentionKind(heads: number, kvHeads: number): 'MHA' | 'GQA' | 'MQA' {
  if (kvHeads === heads) return 'MHA'
  if (kvHeads === 1) return 'MQA'
  return 'GQA'
}
