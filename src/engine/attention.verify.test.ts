/**
 * Независимая проверка attention.ts (верификатор A2): требования SPEC §6.6 к TOY_ATTENTION
 * (строка «она» отдаёт «Кошке» > 50 %, «забору» заметно меньше; causal; строки в сумме 1; числа
 * матриц с ≤ 1 знаком после запятой; d_k = 2 или 3; числа в объяснениях совпадают с расчётом) и
 * инварианты headPatterns на случайных последовательностях токенов (causal, неотрицательно, строки = 1).
 */
import { describe, expect, it } from 'vitest'
import { TOY_ATTENTION, attention, gqaGroups, headPatterns, matmul, softmaxRows, transpose, type PatternToken } from './attention'
import { mulberry32, randInt } from './rng'

const T = TOY_ATTENTION
const tr = T.trace

describe('TOY_ATTENTION (SPEC §6.6)', () => {
  it('предложение, признаки и размеры: d_model = 4, d_k = 2', () => {
    expect(T.tokens.join(' ').replace(' ,', ',')).toBe('Кошка не перепрыгнула забор, потому что она устала')
    expect(T.X).toHaveLength(T.tokens.length)
    expect(T.X.every((r) => r.length === 4)).toBe(true)
    expect(T.featureNames).toHaveLength(4)
    for (const W of [T.Wq, T.Wk, T.Wv]) {
      expect(W).toHaveLength(4)
      expect(W.every((r) => r.length === T.dK)).toBe(true)
    }
    expect([2, 3]).toContain(T.dK)
    expect(T.queryDimNames).toHaveLength(T.dK)
    expect(T.keyDimNames).toHaveLength(T.dK)
  })
  it('все числа X и W — с одним знаком после запятой максимум', () => {
    for (const M of [T.X, T.Wq, T.Wk, T.Wv]) for (const row of M) for (const x of row) expect(Math.abs(x * 10 - Math.round(x * 10))).toBeLessThan(1e-12)
    // Q, K, V тоже получаются «круглыми» (их показывают в шагах плеера)
    for (const M of [tr.Q, tr.K, tr.V]) for (const row of M) for (const x of row) expect(Math.abs(x * 10 - Math.round(x * 10))).toBeLessThan(1e-9)
  })
  it('строка «она»: «Кошка» > 50 %, «забор» заметно меньше; это максимум строки', () => {
    const { query, target, distractor } = T.focus
    expect(T.tokens[query]).toBe('она')
    expect(T.tokens[target]).toBe('Кошка')
    expect(T.tokens[distractor]).toBe('забор')
    const row = tr.weights[query]
    expect(row[target]).toBeGreaterThan(0.5)
    expect(row[distractor]).toBeLessThan(row[target] / 4)
    expect(Math.max(...row)).toBe(row[target])
    expect(row[target]).toBeCloseTo(0.65, 2)
    expect(row[distractor]).toBeCloseTo(0.078, 3)
  })
  it('causal: над диагональю −∞ и вес 0; строки softmax в сумме 1', () => {
    const n = T.tokens.length
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (j > i) {
          expect(tr.masked[i][j]).toBe(-Infinity)
          expect(tr.weights[i][j]).toBe(0)
        } else expect(Number.isFinite(tr.masked[i][j])).toBe(true)
      }
      expect(tr.weights[i].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    }
  })
  it('trace пересчитывается с нуля: Q = XW_Q, scores = QKᵀ, /√d_k, softmax, out = weights·V', () => {
    expect(tr.Q).toEqual(matmul(T.X, T.Wq))
    expect(tr.K).toEqual(matmul(T.X, T.Wk))
    expect(tr.V).toEqual(matmul(T.X, T.Wv))
    expect(tr.rawScores).toEqual(matmul(tr.Q, transpose(tr.K)))
    tr.scaled.forEach((row, i) => row.forEach((x, j) => expect(x).toBeCloseTo(tr.rawScores[i][j] / Math.sqrt(T.dK), 12)))
    expect(tr.weights).toEqual(softmaxRows(tr.masked))
    expect(tr.out).toEqual(matmul(tr.weights, tr.V))
    expect(attention(T.X, T.Wq, T.Wk, T.Wv, { causal: true })).toEqual(tr)
  })
  it('числа в объяснениях совпадают с расчётом', () => {
    const q = T.focus.query
    expect(tr.Q[q]).toEqual([2, 3])
    expect(tr.K[0]).toEqual([1, 1])
    expect(tr.K[3]).toEqual([1, 0])
    expect(tr.rawScores[q][0]).toBe(5)
    expect(tr.rawScores[q][3]).toBe(2)
    expect(T.explanations.Q).toContain('[2, 3]')
    expect(T.explanations.scores).toContain('= 5')
    expect(T.explanations.scores).toContain('= 2')
    expect(T.explanations.softmax).toContain(`${Math.round(tr.weights[q][0] * 100)} %`)
    expect(T.explanations.softmax).toContain(`${Math.round(tr.weights[q][3] * 100)} %`)
    // «Она» получила признак «существительное» в основном от «Кошки»
    expect((tr.weights[q][0] * tr.V[0][0]) / tr.out[q][0]).toBeGreaterThan(0.85)
  })
  it('без маски «она» всё равно выбирает «Кошку», но может подсматривать в будущее («устала»)', () => {
    const full = attention(T.X, T.Wq, T.Wk, T.Wv, { causal: false })
    expect(full.weights[T.focus.query][T.tokens.length - 1]).toBeGreaterThan(0)
    expect(Math.max(...full.weights[T.focus.query])).toBe(full.weights[T.focus.query][0])
  })
})

describe('headPatterns: инварианты на случайных промптах', () => {
  const PIECES = ['Кошка', ' не', ',', '.', '\n', ' она', '!', ' что', '<|im_start|>', ' the', ' cat', '?', ' ', '–', ' 42']
  function randomTokens(seed: number): PatternToken[] {
    const rng = mulberry32(seed)
    const n = randInt(rng, 0, 40)
    return Array.from({ length: n }, () => {
      const k = randInt(rng, 0, PIECES.length - 1)
      return { id: 1000 + k, text: PIECES[k], special: PIECES[k].startsWith('<|') }
    })
  }
  it('для 200 случайных промптов: n×n, causal, неотрицательно, конечно, строки = 1; детерминированно', () => {
    for (let seed = 0; seed < 200; seed++) {
      const toks = randomTokens(seed)
      const n = toks.length
      const pats = headPatterns(toks)
      expect(pats.length).toBeGreaterThanOrEqual(5)
      expect(new Set(pats.map((p) => p.id)).size).toBe(pats.length)
      for (const p of pats) {
        expect(p.name).toMatch(/[а-яё]/i)
        expect(p.description).toMatch(/[а-яё]/i)
        expect(p.weights).toHaveLength(n)
        p.weights.forEach((row, i) => {
          expect(row).toHaveLength(n)
          row.forEach((x, j) => {
            expect(Number.isFinite(x)).toBe(true)
            expect(x).toBeGreaterThanOrEqual(0)
            if (j > i) expect(x).toBe(0)
          })
          expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
        })
      }
      expect(headPatterns(toks)).toEqual(pats)
    }
  })
  it('смысл паттернов: предыдущий токен, sink ≈ 75 %, «тот же токен раньше» и induction', () => {
    const toks: PatternToken[] = [10, 20, 30, 10, 20, 40].map((id) => ({ id, text: `t${id}` }))
    const byId = Object.fromEntries(headPatterns(toks).map((p) => [p.id, p.weights]))
    const prev = byId['previous-token']
    for (let i = 1; i < toks.length; i++) expect(prev[i].indexOf(Math.max(...prev[i]))).toBe(i - 1)
    const sink = byId['attention-sink']
    for (let i = 1; i < toks.length; i++) expect(sink[i][0]).toBeCloseTo(0.75, 9)
    // токен 3 (id 10) уже был на позиции 0; токен 4 (id 20) был на позиции 1
    expect(byId['duplicate-token'][3][0]).toBeCloseTo(1, 9) // и sink, и совпадение на позиции 0
    expect(byId['duplicate-token'][4].indexOf(Math.max(...byId['duplicate-token'][4]))).toBe(1)
    // induction: после прошлого «10» шёл «20» (позиция 1)
    expect(byId['induction'][3].indexOf(Math.max(...byId['induction'][3]))).toBe(1)
    expect(byId['induction'][4].indexOf(Math.max(...byId['induction'][4]))).toBe(2)
    expect(byId['broad'][5].every((x) => Math.abs(x - 1 / 6) < 1e-12)).toBe(true)
  })
})

describe('gqaGroups', () => {
  it('разбиение Q-голов: каждая ровно в одной группе, группы равные', () => {
    for (const [h, kv] of [
      [32, 8],
      [64, 4],
      [64, 64],
      [32, 1],
    ]) {
      const groups = gqaGroups(h, kv)
      expect(groups).toHaveLength(kv)
      const all = groups.flatMap((g) => g.queryHeads).sort((a, b) => a - b)
      expect(all).toEqual(Array.from({ length: h }, (_, i) => i))
      expect(groups.every((g) => g.queryHeads.length === h / kv)).toBe(true)
    }
    expect(() => gqaGroups(32, 5)).toThrow()
  })
})
