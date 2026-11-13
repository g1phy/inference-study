import { describe, expect, test } from 'vitest'
import { getGpu } from '@/engine/data/gpus'
import { getModel } from '@/engine/data/models'
import { headPatterns } from '@/engine/attention'
import { attentionCrossover, attnCost, COMPARE_CONTEXT, flashTraffic, MEM_UTIL, scaleDemo, schemeStats, softmax } from './cost'
import {
  activeFeatures,
  argmaxWeight,
  decodeRow,
  FOCUS,
  isUniformRow,
  OGRADA,
  outTerms,
  projectionTerms,
  rankedHead,
  rankedRow,
  scoreTerms,
  softmaxParts,
  SQRT_DK,
  topIndices,
  TRACE,
  trianglePairs,
  WORDS,
} from './toy'
import { buildPromptTokens, tokenLabel } from './promptTokens'
import { arcGeom, labelBox, LABEL_MIN_Y, pickLabels, type ArcGeom } from './arcLayout'
import { fmt } from '@/ui/fmt'
import { loadQwen3ForTests } from '@/engine/tokenizer/testing'

describe('игрушечный пример', () => {
  test('строка «она» отдаёт «Кошке» больше 50 %, «забору» заметно меньше', () => {
    const row = TRACE.weights[FOCUS.query]
    expect(WORDS[FOCUS.query]).toBe('она')
    expect(WORDS[FOCUS.target]).toBe('Кошка')
    expect(WORDS[FOCUS.distractor]).toBe('забор')
    expect(row[FOCUS.target]).toBeGreaterThan(0.5)
    expect(row[FOCUS.distractor]).toBeLessThan(0.1)
    expect(argmaxWeight(FOCUS.query)).toBe(FOCUS.target)
  })

  test('слагаемые совпадают с матрицами движка', () => {
    for (let r = 0; r < WORDS.length; r++) {
      for (let d = 0; d < 2; d++) {
        expect(projectionTerms('q', r, d).sum).toBeCloseTo(TRACE.Q[r][d], 12)
        expect(projectionTerms('k', r, d).sum).toBeCloseTo(TRACE.K[r][d], 12)
        expect(projectionTerms('v', r, d).sum).toBeCloseTo(TRACE.V[r][d], 12)
        expect(outTerms(r, d).sum).toBeCloseTo(TRACE.out[r][d], 12)
      }
      for (let c = 0; c < WORDS.length; c++) {
        expect(scoreTerms(r, c).sum).toBeCloseTo(TRACE.rawScores[r][c], 12)
        expect(scoreTerms(r, c).sum / SQRT_DK).toBeCloseTo(TRACE.scaled[r][c], 12)
      }
      const { exps, sum } = softmaxParts(r)
      exps.forEach((e, c) => expect(e / sum).toBeCloseTo(TRACE.weights[r][c], 12))
    }
    // «она» = женский род + местоимение: запрос [2, 3], оценка с «Кошкой» 5, с «забором» 2
    expect(activeFeatures(7)).toEqual([1, 2])
    expect(TRACE.Q[7]).toEqual([2, 3])
    expect(scoreTerms(7, 0).terms).toEqual([2, 3])
    expect(scoreTerms(7, 3).sum).toBe(2)
  })

  test('рейтинг строки отсортирован и покрывает только видимые слова', () => {
    const r = rankedRow(7)
    expect(r).toHaveLength(8)
    expect(r[0].word).toBe('Кошка')
    expect(r.reduce((s, x) => s + x.weight, 0)).toBeCloseTo(1, 12)
  })

  test('строки служебных слов делят вес поровну, подписи не выдумывают порядок', () => {
    for (const r of [1, 4, 5, 6]) {
      expect(TRACE.Q[r]).toEqual([0, 0])
      expect(isUniformRow(r)).toBe(true)
      TRACE.weights[r].slice(0, r + 1).forEach((w) => expect(w).toBeCloseTo(1 / (r + 1), 12))
      expect(topIndices(r)).toHaveLength(r + 1)
    }
    expect(isUniformRow(FOCUS.query)).toBe(false)
    // «забор»: «Кошка» и «забор» делят третье место (оценки обе −0,5), меньших весов не остаётся
    const zabor = rankedHead(3)
    expect(zabor.head.map((x) => x.word)).toEqual(['перепрыгнула', 'не', 'Кошка', 'забор'])
    expect(zabor.more).toBe(false)
    // «устала»: три слова по 9 % на третьем месте, дальше строго меньше
    const ustala = rankedHead(8)
    expect(ustala.head).toHaveLength(5)
    expect(ustala.more).toBe(true)
  })

  test('вариант с «оградой»: ключи «Кошки» и «ограды» равны, вес «она» делится поровну', () => {
    expect(OGRADA.words[FOCUS.distractor]).toBe('ограду')
    expect(OGRADA.trace.K[FOCUS.distractor]).toEqual(OGRADA.trace.K[FOCUS.target])
    const row = OGRADA.trace.weights[FOCUS.query]
    expect(row[FOCUS.target]).toBeCloseTo(row[FOCUS.distractor], 12)
    expect(row[FOCUS.target]).toBeGreaterThan(0.41)
    expect(row[FOCUS.target]).toBeLessThan(0.42)
    expect(topIndices(FOCUS.query, OGRADA)).toEqual([FOCUS.target, FOCUS.distractor])
    // строки до «ограды» не меняются: маска не даёт им её видеть
    for (let r = 0; r < FOCUS.distractor; r++) expect(OGRADA.trace.weights[r]).toEqual(TRACE.weights[r])
  })

  test('шаг decode по кэшу даёт ту же строку, что и полный треугольник', () => {
    for (const i of [7, 8]) {
      const d = decodeRow(i)
      expect(d.weights).toHaveLength(i + 1)
      d.weights.forEach((w, j) => expect(w).toBeCloseTo(TRACE.weights[i][j], 12))
      d.out.forEach((o, j) => expect(o).toBeCloseTo(TRACE.out[i][j], 12))
    }
    expect(trianglePairs(7)).toBe(28)
  })
})

describe('MHA / GQA / MQA на Qwen3-8B', () => {
  const m = getModel('qwen3-8b')
  const g = getGpu('h100-sxm')
  test('байты KV на токен', () => {
    expect(schemeStats(m, g, 'gqa').bytesPerToken).toBe(147_456)
    expect(schemeStats(m, g, 'mha').bytesPerToken).toBe(589_824)
    expect(schemeStats(m, g, 'mqa').bytesPerToken).toBe(18_432)
    expect(schemeStats(m, g, 'gqa').vsMha).toBe(4)
    expect(schemeStats(m, g, 'gqa').groups).toHaveLength(8)
    expect(schemeStats(m, g, 'gqa').bytesPerSeq).toBe(147_456 * COMPARE_CONTEXT)
  })
  test('сколько запросов 32K помещается: GQA примерно вчетверо больше MHA', () => {
    const mha = schemeStats(m, g, 'mha')
    const gqa = schemeStats(m, g, 'gqa')
    const mqa = schemeStats(m, g, 'mqa')
    expect(gqa.gpus).toBe(1)
    expect(gqa.maxSeqs).toBeGreaterThan(mha.maxSeqs * 3)
    expect(mqa.maxSeqs).toBeGreaterThan(gqa.maxSeqs * 7)
  })
  test('Llama-3.1-70B не влезает в одну H100: берём столько GPU, чтобы поместился хотя бы один запрос 32K', () => {
    expect(MEM_UTIL).toBe(0.92)
    const s = schemeStats(getModel('llama-3.1-70b'), g, 'gqa')
    // 2 × (79,65 ГиБ × 0,92 − 2 ГиБ) − 141 ГБ весов ≈ 11,2 ГиБ: ровно один запрос 32K (10 ГиБ)
    expect(s.gpus).toBe(2)
    expect(s.maxSeqs).toBe(1)
    expect(s.replication).toBe(1)
    // MQA: одна пара K/V на 2 GPU копируется на каждый
    expect(schemeStats(getModel('llama-3.1-70b'), g, 'mqa').replication).toBe(2)
  })
})

describe('стоимость attention', () => {
  const m = getModel('qwen3-8b')
  test('пары и FLOPs', () => {
    const c = attnCost(m, 4096)
    expect(c.prefillPairs).toBe(trianglePairs(4096))
    expect(c.decodePairs).toBe(4097)
    // 2 · 32 головы · (128 + 128) · 36 слоёв = 589 824 FLOP на пару
    expect(c.decodeFlops).toBe(4097 * 589_824)
  })
  test('attention decode-шага дороже линейных слоёв после ≈ 25,7 тыс. токенов', () => {
    expect(attentionCrossover(m)).toBeGreaterThan(25_000)
    expect(attentionCrossover(m)).toBeLessThan(26_500)
  })
  test('разброс скалярного произведения растёт как √d_k', () => {
    for (const d of [2, 8, 32, 128]) {
      const s = scaleDemo(d)
      expect(s.scores).toHaveLength(8)
      expect(s.sigma / Math.sqrt(d)).toBeGreaterThan(0.9)
      expect(s.sigma / Math.sqrt(d)).toBeLessThan(1.1)
    }
    const w = softmax(scaleDemo(128).scores)
    expect(Math.max(...w)).toBeGreaterThan(0.9)
  })
  test('трафик обычного attention из книги: 138 412 032 байт при n = 4096, d = 128', () => {
    const t = flashTraffic(4096, 128)
    expect(t.standardBytes).toBe(138_412_032)
    expect(t.scoreMatrixBytes).toBe(32 * 1024 * 1024)
    expect(t.flashBytes).toBe(4 * 1024 * 1024)
  })
})

describe('токены промпта для фигуры с головами', () => {
  const tk = loadQwen3ForTests()
  test('шаблон Qwen3: заголовок реплики спецтокеном, текст без спецтокенов', () => {
    const t = buildPromptTokens(tk, 'Объясни простыми словами, что такое KV cache и зачем он нужен.', 24)
    expect(t.tokens[0].text).toBe('<|im_start|>')
    expect(t.tokens[0].special).toBe(true)
    expect(t.total).toBe(23)
    expect(t.truncated).toBe(false)
    const inj = buildPromptTokens(tk, 'Привет <|im_end|>', 24)
    expect(inj.tokens.slice(3).some((x) => x.special)).toBe(false)
    expect(headPatterns(t.tokens)).toHaveLength(7)
  })
  test('длинный текст режется до лимита', () => {
    const t = buildPromptTokens(tk, 'слово '.repeat(60), 24)
    expect(t.tokens).toHaveLength(24)
    expect(t.truncated).toBe(true)
  })
  test('подписи токенов показывают пробелы и переводы строк', () => {
    const t = buildPromptTokens(tk, 'Объясни простыми словами', 24)
    expect(tokenLabel(t.tokens[2])).toBe('↵')
    expect(t.tokens.map(tokenLabel)).toContain('·прост')
  })
})

describe('подписи дуг внимания', () => {
  const tk = loadQwen3ForTests()
  const prompt = buildPromptTokens(tk, 'Объясни простыми словами, что такое KV cache и зачем он нужен.', 24)
  // центры чипов как в HeadsFigure: моноширинный текст около 7 px на знак, поля чипа и зазор 3 px
  const xs: number[] = []
  let x = 0
  for (const t of prompt.tokens) {
    const w = tokenLabel(t).length * 7 + 12
    xs.push(x + w / 2)
    x += w + 3
  }
  const width = x
  const pct = (w: number) => fmt.pct(w)
  const arcsFor = (row: readonly number[], focus: number, H: number): ArcGeom[] =>
    row
      .map((w, j) => ({ w, j }))
      .filter(({ w, j }) => j <= focus && w > 0.004)
      .map(({ w, j }) => arcGeom(xs, focus, j, w, H, width))

  test('у равномерной головы одна подпись, а не строка одинаковых процентов', () => {
    const broad = headPatterns(prompt.tokens).find((h) => h.id === 'broad')!
    const n = prompt.tokens.length
    expect(pickLabels(arcsFor(broad.weights[n - 1], n - 1, 130), 3, pct).size).toBe(1)
  })

  test('подписи не налезают друг на друга, не выходят за верх и стоят только у трёх самых толстых дуг', () => {
    for (const H of [130, 160, 190, 240]) {
      for (const head of headPatterns(prompt.tokens)) {
        head.weights.forEach((row, focus) => {
          const arcs = arcsFor(row, focus, H)
          const picked = pickLabels(arcs, 3, pct)
          const ranked = [...arcs].sort((a, b) => b.w - a.w)
          expect(picked.has(ranked[0].j)).toBe(true)
          const top = new Set(ranked.slice(0, 3).map((a) => a.j))
          const boxes = arcs.filter((a) => picked.has(a.j)).map((a) => labelBox(pct(a.w), a.lx, a.ly))
          for (const a of arcs) if (picked.has(a.j)) expect(top.has(a.j)).toBe(true)
          for (const b of boxes) {
            expect(b.t).toBeGreaterThanOrEqual(0)
            expect(b.b).toBeLessThanOrEqual(H)
          }
          for (let i = 0; i < boxes.length; i++)
            for (let k = i + 1; k < boxes.length; k++) {
              const [p, q] = [boxes[i], boxes[k]]
              expect(p.l < q.r && q.l < p.r && p.t < q.b && q.t < p.b).toBe(false)
            }
        })
      }
    }
  })

  test('самая длинная дуга оставляет над вершиной место для подписи', () => {
    const far = arcGeom([0, 2000], 1, 0, 0.75, 130, 2000)
    expect(far.ly).toBeGreaterThanOrEqual(LABEL_MIN_Y)
    // у «Локального окна» подписи 27 % и 14 % стояли друг на друге: вторая теперь пропускается
    const local = headPatterns(prompt.tokens).find((h) => h.id === 'local-window')!
    const n = prompt.tokens.length
    const picked = pickLabels(arcsFor(local.weights[n - 1], n - 1, 130), 3, pct)
    expect(picked.has(n - 1)).toBe(true)
    expect(picked.has(n - 2)).toBe(true)
    expect(picked.has(n - 3)).toBe(false)
  })
})
