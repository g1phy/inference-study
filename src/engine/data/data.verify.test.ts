/**
 * Независимая проверка данных (верификатор A2): точечные сверки с datasheet-арифметикой
 * из research/notes-numbers.md (§2.1–2.5, §3.1) и с полями config.json, не через numbers.json.
 */
import { describe, expect, it } from 'vitest'
import { GPUS, getGpu } from './gpus'
import { MODELS, getModel } from './models'
import { INTERCONNECTS, getInterconnect } from './interconnects'
import { kvBytesPerToken, weightBytes } from '../perf'

describe('GPU: dense против sparse (notes-numbers.md §3.1)', () => {
  it('H100 SXM: datasheet 1,979* BF16 / 3,958* FP8 «with sparsity» → dense = половина', () => {
    const g = getGpu('h100-sxm')
    expect(g.tflops.bf16).toBe(Math.floor(1979 / 2)) // 989.5 → 989 (книга с. 62)
    expect(g.tflops.fp8).toBe(3958 / 2)
    expect(g.tflops.fp4).toBeUndefined() // у Hopper нет FP4 в тензорных ядрах
    expect(g.bwTBs).toBe(3.35)
    expect(g.usableMiB).toBe(81559)
  })
  it('HGX B200 (8 GPU): FP4 72 PF dense, FP8 72 PF sparse, BF16 36 PF sparse → на GPU', () => {
    const g = getGpu('b200')
    expect(g.tflops.fp4).toBe((72 / 8) * 1000)
    expect(g.tflops.fp8).toBe((72 / 2 / 8) * 1000)
    expect(g.tflops.bf16).toBe((36 / 2 / 8) * 1000)
    expect(g.bwTBs).toBe(8)
    expect(g.memGB).toBe(180)
  })
  it('B300: удвоился только FP4 dense (108 PF на 8 GPU), FP8/BF16 как у B200', () => {
    const b300 = getGpu('b300')
    const b200 = getGpu('b200')
    expect(b300.tflops.fp4).toBe((108 / 8) * 1000)
    expect(b300.tflops.fp8).toBe(b200.tflops.fp8)
    expect(b300.tflops.bf16).toBe(b200.tflops.bf16)
    expect(b300.memGB).toBe(288)
  })
  it('GB200 superchip (2 GPU): FP8 20 PF sparse → 5 PF dense на GPU; BF16 10 PF sparse → 2.5', () => {
    const g = getGpu('gb200')
    expect(g.tflops.fp8).toBe((20 / 2 / 2) * 1000)
    expect(g.tflops.bf16).toBe((10 / 2 / 2) * 1000)
    expect(g.tflops.fp4).toBe((20 / 2) * 1000)
  })
  it('A100: 312 | 624* BF16, FP8 нет; L40S 362 | 733*; L4 242* / 485*', () => {
    expect(getGpu('a100-80g').tflops).toEqual({ bf16: 624 / 2 })
    expect(getGpu('l40s').tflops.bf16).toBe(362)
    expect(getGpu('l40s').tflops.fp8).toBe(1466 / 2)
    expect(getGpu('l4').tflops.bf16).toBe(242 / 2)
    expect(getGpu('l4').tflops.fp8).toBe(485 / 2)
  })
  it('GeForce: FP32-accumulate (половина маркетинга); RTX 5090 «3352 AI TOPS» = sparse FP4', () => {
    expect(getGpu('rtx-4090').tflops.bf16).toBeCloseTo(330.4 / 2, 1)
    expect(getGpu('rtx-5090').tflops.fp4).toBe(3352 / 2)
    expect(getGpu('rtx-5090').bwTBs).toBe(1.792) // книга с. 90
  })
  it('H200: тот же compute, что H100, память больше и быстрее', () => {
    const h = getGpu('h200')
    expect(h.tflops).toEqual(getGpu('h100-sxm').tflops)
    expect([h.memGB, h.bwTBs, h.usableMiB]).toEqual([141, 4.8, 143771])
  })
  it('у всех GPU FP-точности удваиваются вниз (кроме исключений B300/GB300 по FP4)', () => {
    for (const g of GPUS) {
      if (g.tflops.fp8 !== undefined) expect(g.tflops.fp8 / g.tflops.bf16).toBeCloseTo(2, 1)
      if (g.tflops.fp4 !== undefined && g.tflops.fp8 !== undefined) expect(g.tflops.fp4 / g.tflops.fp8).toBeGreaterThanOrEqual(1.99)
      if (g.usableMiB) expect(g.usableMiB / 1024).toBeLessThanOrEqual(g.memGB * 1.075) // nvidia-smi ≈ «GB» по паспорту
    }
  })
})

describe('модели: поля config.json (notes-numbers.md §2.4) и посчитанные величины (§2.5)', () => {
  it('таблица §2.4', () => {
    const rows: Array<[string, number, number, number, number, number, number]> = [
      // id, layers, hidden, heads, kvHeads, headDim, vocab
      ['llama-3.1-8b', 32, 4096, 32, 8, 128, 128256],
      ['llama-3.1-70b', 80, 8192, 64, 8, 128, 128256],
      ['qwen3-8b', 36, 4096, 32, 8, 128, 151936],
      ['qwen3-32b', 64, 5120, 64, 8, 128, 151936],
      ['qwen3-30b-a3b', 48, 2048, 32, 4, 128, 151936],
      ['qwen3-235b-a22b', 94, 4096, 64, 4, 128, 151936],
      ['gpt-oss-120b', 36, 2880, 64, 8, 64, 201088],
      ['gpt-oss-20b', 24, 2880, 64, 8, 64, 201088],
      ['gemma-3-27b', 62, 5376, 32, 16, 128, 262208],
      ['mistral-small-24b', 40, 5120, 32, 8, 128, 131072],
    ]
    for (const [id, L, h, nh, nkv, hd, V] of rows) {
      const m = getModel(id)
      expect([m.layers, m.hidden, m.heads, m.kvHeads, m.headDim, m.vocab], id).toEqual([L, h, nh, nkv, hd, V])
    }
    const ds = getModel('deepseek-v3')
    expect([ds.layers, ds.hidden, ds.heads, ds.vocab, ds.kvLoraRank, ds.qkRopeHeadDim, ds.vHeadDim]).toEqual([61, 7168, 128, 129280, 512, 64, 128])
    expect(ds.experts).toEqual({ total: 256, active: 8 })
    expect(getModel('qwen3-235b-a22b').experts).toEqual({ total: 128, active: 8 }) // книга с. 53–54
    expect(getModel('gpt-oss-120b').experts).toEqual({ total: 128, active: 4 })
  })
  it('Llama-3.1-8B: ровно 8 030 261 248 параметров (§3.3), 7.505B «умножаемых»', () => {
    const m = getModel('llama-3.1-8b')
    const embed = 128256 * 4096
    const layer = 4096 * 4096 * 2 + 2 * 4096 * 1024 + 3 * 4096 * 14336 + 2 * 4096
    expect(embed * 2 + 32 * layer + 4096).toBe(8030261248)
    expect(m.paramsB * 1e9).toBe(8030261248)
    expect(m.activeMatmulParamsB).toBeCloseTo((8030261248 - embed) / 1e9, 3)
    expect(m.lmHeadParamsB).toBeCloseTo(embed / 1e9, 4)
  })
  it('KV на токен из формулы, а не из таблицы: 2·L·kv·hd·2 и MLA L·576·2', () => {
    for (const m of MODELS) {
      const expected = m.attention === 'mla' ? m.layers * 576 * 2 : 2 * m.layers * m.kvHeads * m.headDim * 2
      expect(kvBytesPerToken(m), m.id).toBe(expected)
    }
    expect(kvBytesPerToken(getModel('qwen3-8b')) / 1024).toBe(144)
    expect(kvBytesPerToken(getModel('deepseek-v3')) / 1024).toBeCloseTo(68.625, 6)
  })
  it('MoE: активные = всё − (1 − k/E)·эксперты у всех MoE-моделей (с точностью округления)', () => {
    for (const m of MODELS) {
      if (!m.experts || !m.expertParamsB) continue
      const idle = 1 - m.experts.active / m.experts.total
      const derived = m.paramsB - m.expertParamsB * idle
      // gpt-oss: activeParamsB по OpenAI без таблицы эмбеддингов (vocab × hidden); у экспертов gpt-oss
      // есть bias (2·I + h на эксперта), которые не входят в expertParamsB (только матрицы)
      const embed = m.family === 'gpt-oss' ? (m.vocab * m.hidden) / 1e9 : 0
      const biases = m.family === 'gpt-oss' ? (m.layers * m.experts.total * (2 * m.intermediate + m.hidden) * idle) / 1e9 : 0
      expect(derived - embed - biases, m.id).toBeCloseTo(m.activeParamsB, 2)
    }
  })
  it('веса: NVFP4 = 4.5 бит, gpt-oss нативно (MXFP4 4.25 бит + BF16) = 65.25 ГБ / 60.8 ГиБ (§3.6)', () => {
    expect(weightBytes(getModel('llama-3.1-70b'), 'fp4') / 1e9).toBeCloseTo((70.554 * 4.5) / 8, 1)
    expect(weightBytes(getModel('gpt-oss-120b'), 'fp4') / 2 ** 30).toBeCloseTo(60.77, 2)
    expect(weightBytes(getModel('gpt-oss-20b'), 'fp4') / 2 ** 30).toBeCloseTo(12.82, 2)
  })
})

describe('интерконнекты: в одну сторону, биты и байты', () => {
  it('NVLink и PCIe: в одну сторону = половина суммы двух направлений', () => {
    for (const l of INTERCONNECTS) {
      if (l.GBsBidirectional !== undefined) expect(l.GBsBidirectional, l.id).toBeCloseTo(2 * l.GBs, 6)
    }
  })
  it('сеть: 400 Гбит/с = 50 ГБ/с, 800 Гбит/с = 100 ГБ/с, 25 Гбит/с = 3.125 ГБ/с', () => {
    expect(getInterconnect('ib-ndr').GBs).toBe(400 / 8)
    expect(getInterconnect('ib-xdr').GBs).toBe(800 / 8)
    expect(getInterconnect('roce-400').GBs).toBe(400 / 8)
    expect(getInterconnect('tcp-25g').GBs).toBe(25 / 8)
    expect(getInterconnect('ib-ndr-x8').GBs).toBe((8 * 400) / 8)
  })
  it('PCIe по поколениям: Gen4 31.5, Gen5 63 (128b/130b), Gen6 121', () => {
    expect(getInterconnect('pcie4').GBs).toBeCloseTo((16 * 16 * 128) / 130 / 8, 0)
    expect(getInterconnect('pcie5').GBs).toBeCloseTo((32 * 16 * 128) / 130 / 8, 0)
    expect(getInterconnect('pcie6').GBs).toBe(121)
  })
})
