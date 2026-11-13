import { describe, expect, it } from 'vitest'
import numbersJson from '../../../research/numbers.json'
import { FEATURED_GPU_IDS, GPUS, findGpu, getGpu } from './gpus'
import { FEATURED_MODEL_IDS, MODELS, findModel, getModel, slidingLayers } from './models'
import { INTERCONNECTS, KV_TIERS, KV_TRANSFER_LINK_IDS, getInterconnect } from './interconnects'
import { kvBytesPerToken, weightBytes } from '../perf'

/** research/numbers.json — первоисточник данных (импорт JSON-модулем, без node:fs). */
const numbers = numbersJson as unknown as {
  gpus: Array<{ id: string; memGB: number; bwTBs: number; tflops: { bf16: number; fp8: number | null; fp4: number | null } }>
  models: Array<{ id: string; layers: number; hidden: number; heads: number; kvHeads: number; headDim: number; vocab: number; paramsB: number; kvBytesPerTokenBF16: number; kvBytesPerTokenFP8: number; activeMatmulParamsB: number }>
  interconnects: Array<{ id: string; GBs: number }>
}

describe('GPU', () => {
  it('обязательные GPU есть, у каждого источник', () => {
    for (const id of ['a100-80g', 'h100-sxm', 'h200', 'b200', 'l40s', 'gb200', 'b300']) expect(getGpu(id).id).toBe(id)
    for (const g of GPUS) {
      expect(g.source.length).toBeGreaterThan(10)
      expect(g.bwTBs).toBeGreaterThan(0)
      expect(g.tflops.bf16).toBeGreaterThan(0)
    }
    expect(new Set(GPUS.map((g) => g.id)).size).toBe(GPUS.length)
    FEATURED_GPU_IDS.forEach((id) => expect(getGpu(id).featured).toBe(true))
  })
  it('алиасы id из numbers.json и ошибка на неизвестный', () => {
    expect(getGpu('h200-sxm').id).toBe('h200')
    expect(getGpu('b200-hgx').id).toBe('b200')
    expect(getGpu('a100-sxm-80gb').id).toBe('a100-80g')
    expect(() => getGpu('nope')).toThrow()
    expect(findGpu('nope')).toBeUndefined()
  })
  it('числа совпадают с research/numbers.json', () => {
    for (const src of numbers.gpus) {
      const g = findGpu(src.id)
      if (!g) continue
      expect(g.memGB).toBe(src.memGB)
      expect(g.bwTBs).toBe(src.bwTBs)
      expect(g.tflops.bf16).toBe(src.tflops.bf16)
      expect(g.tflops.fp8 ?? null).toBe(src.tflops.fp8)
      expect(g.tflops.fp4 ?? null).toBe(src.tflops.fp4)
    }
  })
})

describe('модели', () => {
  it('обязательные модели есть, у каждой источник', () => {
    for (const id of ['qwen3-8b', 'qwen3-32b', 'qwen3-30b-a3b', 'qwen3-235b-a22b', 'llama-3.1-8b', 'llama-3.1-70b', 'deepseek-v3', 'gpt-oss-120b']) {
      expect(getModel(id).id).toBe(id)
    }
    for (const m of MODELS) expect(m.source).toMatch(/huggingface\.co/)
    expect(new Set(MODELS.map((m) => m.id)).size).toBe(MODELS.length)
    FEATURED_MODEL_IDS.forEach((id) => expect(getModel(id).featured).toBe(true))
    expect(() => getModel('gpt-5')).toThrow()
    expect(findModel('gpt-5')).toBeUndefined()
  })
  it('поля config и KV совпадают с research/numbers.json', () => {
    for (const src of numbers.models) {
      const m = findModel(src.id)
      if (!m) continue
      expect([m.layers, m.hidden, m.heads, m.vocab]).toEqual([src.layers, src.hidden, src.heads, src.vocab])
      if (m.attention !== 'mla') expect([m.kvHeads, m.headDim]).toEqual([src.kvHeads, src.headDim])
      expect(m.paramsB).toBeCloseTo(src.paramsB, 2)
      expect(m.activeMatmulParamsB).toBeCloseTo(src.activeMatmulParamsB, 3)
      expect(kvBytesPerToken(m, 'bf16')).toBe(src.kvBytesPerTokenBF16)
      expect(kvBytesPerToken(m, 'fp8')).toBe(src.kvBytesPerTokenFP8)
    }
  })
  it('MoE: активные = всегда активная часть + k/E экспертов', () => {
    for (const m of MODELS) {
      if (!m.experts) continue
      expect(m.expertParamsB).toBeGreaterThan(0)
      expect(m.expertParamsB!).toBeLessThan(m.paramsB)
    }
    const q = getModel('qwen3-235b-a22b')
    expect(q.paramsB - q.expertParamsB! * (1 - 8 / 128)).toBeCloseTo(q.activeParamsB, 2)
    const ds = getModel('deepseek-v3')
    expect(ds.paramsB - ds.expertParamsB! * (1 - 8 / 256)).toBeCloseTo(ds.activeParamsB, 2)
  })
  it('DeepSeek-V3: MLA 512 + 64; gpt-oss-120b: 18 + 18 слоёв, окно 128', () => {
    const ds = getModel('deepseek-v3')
    expect(ds.attention).toBe('mla')
    expect(ds.mlaDim).toBe(576)
    expect(ds.kvLoraRank! + ds.qkRopeHeadDim!).toBe(576)
    const g = getModel('gpt-oss-120b')
    expect(g.attention).toBe('gqa-swa')
    expect(g.slidingWindow).toEqual({ window: 128, fullLayers: 18 })
    expect(slidingLayers(g)).toBe(18)
    expect(slidingLayers(getModel('qwen3-8b'))).toBe(0)
    expect(weightBytes(g, 'fp4') / 1e9).toBeCloseTo(65.25, 1)
  })
})

describe('интерконнекты и уровни KV', () => {
  it('обязательные каналы есть, полосы в одну сторону', () => {
    const ids = ['pcie4', 'pcie5', 'nvlink4', 'nvlink5', 'ib-ndr', 'ib-xdr', 'roce-400', 'nvme4', 'nvme5', 'host-dram']
    for (const id of ids) expect(getInterconnect(id).GBs).toBeGreaterThan(0)
    expect(getInterconnect('nvlink4').GBs).toBe(450)
    expect(getInterconnect('nvlink4').GBsBidirectional).toBe(900)
    expect(getInterconnect('ib-ndr').GBs).toBe(50)
    expect(getInterconnect('pcie5-x16').id).toBe('pcie5')
    for (const l of INTERCONNECTS) expect(l.source.length).toBeGreaterThan(5)
    expect(new Set(INTERCONNECTS.map((l) => l.id)).size).toBe(INTERCONNECTS.length)
    expect(() => getInterconnect('carrier-pigeon')).toThrow()
  })
  it('совпадают с research/numbers.json', () => {
    for (const src of numbers.interconnects) {
      const l = INTERCONNECTS.find((x) => x.id === src.id || x.aliases?.includes(src.id))
      if (l) expect(l.GBs).toBe(src.GBs)
    }
  })
  it('уровни KV ссылаются на существующие каналы, лестница убывает по полосе', () => {
    expect(KV_TIERS.map((t) => t.id)).toEqual(['g1', 'g2', 'g3', 'g4'])
    for (const t of KV_TIERS) getInterconnect(t.linkId)
    const bw = KV_TIERS.slice(0, 3).map((t) => getInterconnect(t.linkId).GBs)
    expect(bw[0]).toBeGreaterThan(bw[1])
    expect(bw[1]).toBeGreaterThan(bw[2])
    KV_TRANSFER_LINK_IDS.forEach((id) => getInterconnect(id))
  })
})
