import { describe, expect, it } from 'vitest'
import {
  TOY_ATTENTION,
  attention,
  attentionKind,
  causalMask,
  dotTerms,
  gqaGroups,
  headPatterns,
  mat,
  matmul,
  rowSums,
  softmaxRows,
  transpose,
  type PatternToken,
} from './attention'

describe('матричные помощники', () => {
  it('matmul, transpose, add, scale', () => {
    const A = [
      [1, 2],
      [3, 4],
    ]
    const B = [
      [5, 6],
      [7, 8],
    ]
    expect(matmul(A, B)).toEqual([
      [19, 22],
      [43, 50],
    ])
    expect(transpose([[1, 2, 3]])).toEqual([[1], [2], [3]])
    expect(mat.add(A, B)).toEqual([
      [6, 8],
      [10, 12],
    ])
    expect(mat.scale(A, 0.5)).toEqual([
      [0.5, 1],
      [1.5, 2],
    ])
    expect(() => matmul(A, [[1, 2, 3]])).toThrow()
  })
  it('softmax по строкам с маской и −∞', () => {
    const s = softmaxRows([[0, 0, -Infinity]])
    expect(s[0][0]).toBeCloseTo(0.5, 12)
    expect(s[0][2]).toBe(0)
    const m = softmaxRows(
      [
        [1, 2],
        [1, 2],
      ],
      causalMask(2),
    )
    expect(m[0]).toEqual([1, 0])
    expect(rowSums(m).every((x) => Math.abs(x - 1) < 1e-12)).toBe(true)
    expect(softmaxRows([[-Infinity, -Infinity]])[0]).toEqual([0, 0])
    expect(softmaxRows([[1000, 1000]])[0][0]).toBeCloseTo(0.5, 12) // численно устойчиво
  })
  it('слагаемые скалярного произведения', () => {
    expect(dotTerms([2, 3], [1, 1])).toEqual({ terms: [2, 3], sum: 5 })
  })
})

describe('attention на числах из заметок (notes-models-attention-bottlenecks.md §3.5)', () => {
  // «Я написал книгу она»: строка «она» = [0.0434, 0.0434, 0.7346, 0.1786]
  const X = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1],
  ]
  const Wq = [
    [2, 0],
    [0, 0],
    [0, 0],
    [0, 2],
  ]
  const Wk = [
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1],
  ]
  const Wv = [
    [1, 0],
    [0, 1],
    [2, 0],
    [0, 2],
  ]
  it('веса и выход совпадают с разбором', () => {
    const tr = attention(X, Wq, Wk, Wv, { causal: true })
    expect(tr.rawScores[3]).toEqual([0, 0, 4, 2])
    expect(tr.masked[0].slice(1).every((x) => x === -Infinity)).toBe(true)
    const w = tr.weights[3]
    expect(w[0]).toBeCloseTo(0.0434, 4)
    expect(w[2]).toBeCloseTo(0.7346, 4)
    expect(w[3]).toBeCloseTo(0.1786, 4)
    expect(tr.weights[2][2]).toBeCloseTo(0.6728, 4)
    expect(tr.out[3][0]).toBeCloseTo(1.6912, 3)
    expect(tr.out[3][1]).toBeCloseTo(1.8697, 3)
  })
  it('без деления на √d_k распределение острее', () => {
    const tr = attention(X, Wq, Wk, Wv, { causal: true, scale: false })
    expect(tr.weights[3][2]).toBeCloseTo(0.8533, 4)
  })
  it('без causal mask токен видит будущее', () => {
    const tr = attention(X, Wq, Wk, Wv, { causal: false })
    expect(tr.weights[0][2]).toBeGreaterThan(0)
  })
})

describe('TOY_ATTENTION: «Кошка не перепрыгнула забор, потому что она устала»', () => {
  const T = TOY_ATTENTION
  const idx = (w: string) => T.tokens.indexOf(w)
  const W = T.trace.weights
  it('размеры и подписи', () => {
    expect(T.tokens).toEqual(['Кошка', 'не', 'перепрыгнула', 'забор', ',', 'потому', 'что', 'она', 'устала'])
    expect(T.featureNames).toHaveLength(4)
    expect(T.X.every((r) => r.length === 4)).toBe(true)
    expect(T.Wq).toHaveLength(4)
    expect(T.Wq[0]).toHaveLength(T.dK)
    expect(T.dK).toBe(2)
    expect(T.focus).toEqual({ query: idx('она'), target: idx('Кошка'), distractor: idx('забор') })
    for (const key of Object.keys(T.explanations)) expect((T.explanations as Record<string, string>)[key].length).toBeGreaterThan(20)
  })
  it('все числа матриц с не более чем одним знаком после запятой', () => {
    for (const M of [T.X, T.Wq, T.Wk, T.Wv]) for (const row of M) for (const x of row) expect(Math.round(x * 10) / 10).toBe(x)
  })
  it('строка «она»: «Кошка» > 50 %, «забор» < 20 %', () => {
    const row = W[idx('она')]
    expect(row[idx('Кошка')]).toBeGreaterThan(0.5)
    expect(row[idx('Кошка')]).toBeCloseTo(0.65, 2)
    expect(row[idx('забор')]).toBeLessThan(0.2)
    expect(row[idx('забор')]).toBeCloseTo(0.078, 2)
    expect(row[idx('устала')]).toBe(0) // будущее замаскировано
    // «Кошка» — самый сильный ключ в строке
    expect(Math.max(...row)).toBe(row[idx('Кошка')])
  })
  it('правдоподобно в остальных строках', () => {
    const tired = W[idx('устала')]
    const ranked = tired.map((w, j) => [w, j] as const).sort((a, b) => b[0] - a[0])
    expect([ranked[0][1], ranked[1][1]].sort()).toEqual([idx('Кошка'), idx('она')].sort())
    expect(W[idx('перепрыгнула')][idx('Кошка')]).toBeGreaterThan(0.6) // глагол → подлежащее
    expect(W[idx('забор')][idx('перепрыгнула')]).toBeGreaterThan(W[idx('забор')][idx('Кошка')]) // дополнение → глагол
    expect(W[0][0]).toBe(1)
  })
  it('causal, строки суммируются в 1, выход «она» получил признак «существительное»', () => {
    for (let i = 0; i < W.length; i++) {
      expect(W[i].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
      for (let j = i + 1; j < W.length; j++) {
        expect(W[i][j]).toBe(0)
        expect(T.trace.masked[i][j]).toBe(-Infinity)
      }
    }
    expect(T.trace.out[idx('она')][0]).toBeGreaterThan(0.6)
    expect(T.trace.Q[idx('она')]).toEqual([2, 3])
  })
})

/** Токены для паттернов. */
function toks(texts: string[], ids?: number[]): PatternToken[] {
  return texts.map((text, i) => ({ id: ids ? ids[i] : 1000 + i, text }))
}

describe('headPatterns', () => {
  const t = toks(['<|im_start|>', 'user', '\n', 'Мама', ' мыла', ' раму', '.', ' Мама', ' мыла'], [151644, 872, 198, 10, 11, 12, 13, 10, 11])
  t[0].special = true
  const patterns = headPatterns(t)
  it('все паттерны causal, строки в сумме 1, n × n, id уникальны, тексты по-русски', () => {
    expect(patterns.length).toBeGreaterThanOrEqual(5)
    expect(new Set(patterns.map((p) => p.id)).size).toBe(patterns.length)
    for (const p of patterns) {
      expect(p.weights).toHaveLength(t.length)
      p.weights.forEach((row, i) => {
        expect(row).toHaveLength(t.length)
        expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
        for (let j = i + 1; j < row.length; j++) expect(row[j]).toBe(0)
        expect(row.every((x) => x >= 0)).toBe(true)
      })
      expect(p.name).toMatch(/[а-яё]/i)
      expect(p.description.length).toBeGreaterThan(30)
    }
  })
  const byId = (id: string) => patterns.find((p) => p.id === id)!.weights
  it('предыдущий токен', () => {
    const w = byId('previous-token')
    expect(w[5][4]).toBeGreaterThan(0.8)
    expect(w[1][0]).toBe(1)
  })
  it('attention sink: большая часть на первом токене', () => {
    const w = byId('attention-sink')
    for (let i = 1; i < t.length; i++) expect(w[i][0]).toBeGreaterThan(0.7)
  })
  it('induction: второе «Мама» смотрит на « мыла» после первого «Мама»', () => {
    const w = byId('induction')
    const i = 7 // второе « Мама» (id 10), первое на позиции 3 → цель 4
    expect(w[i][4]).toBeGreaterThan(0.8)
    const dup = byId('duplicate-token')
    expect(dup[i][3]).toBeGreaterThan(0.8)
    expect(dup[5][0]).toBe(1) // « раму» раньше не встречалось → на первый токен
  })
  it('разделители: точка, перевод строки и спецтокен', () => {
    const w = byId('delimiter')
    expect(w[8][6]).toBeGreaterThan(w[8][4]) // после «.» внимание на точку
    expect(w[4][2]).toBeGreaterThan(0.5) // перевод строки
  })
  it('пустой и одиночный промпт', () => {
    expect(headPatterns([]).every((p) => p.weights.length === 0)).toBe(true)
    expect(headPatterns(toks(['a'])).every((p) => p.weights[0][0] === 1)).toBe(true)
  })
})

describe('GQA', () => {
  it('Qwen3-8B: 32 Q-головы на 8 KV-голов, по 4 в группе', () => {
    const g = gqaGroups(32, 8)
    expect(g).toHaveLength(8)
    expect(g.every((x) => x.queryHeads.length === 4)).toBe(true)
    expect(g[1].queryHeads).toEqual([4, 5, 6, 7])
    expect(attentionKind(32, 8)).toBe('GQA')
  })
  it('MHA и MQA; некратное число голов — ошибка', () => {
    expect(gqaGroups(4, 4).map((x) => x.queryHeads)).toEqual([[0], [1], [2], [3]])
    expect(gqaGroups(4, 1)[0].queryHeads).toEqual([0, 1, 2, 3])
    expect(attentionKind(4, 4)).toBe('MHA')
    expect(attentionKind(4, 1)).toBe('MQA')
    expect(() => gqaGroups(32, 5)).toThrow()
    expect(() => gqaGroups(0, 1)).toThrow()
  })
})
