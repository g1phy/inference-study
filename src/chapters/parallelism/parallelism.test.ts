import { describe, expect, test } from 'vitest'
import { getModel } from '@/engine/data/models'
import { getGpu } from '@/engine/data/gpus'
import { LAUNCH_MS, allReduceMs, bubbleFraction, decodeBubbleFraction, decodeSchedule, pipelineSchedule, plan, usablePerGpu } from './plan'

const llama70 = getModel('llama-3.1-70b')
const qwen235 = getModel('qwen3-235b-a22b')
const dsv3 = getModel('deepseek-v3')
const h100 = getGpu('h100-sxm')
const b200 = getGpu('b200')

describe('plan: память', () => {
  test('TP делит веса, DP нет', () => {
    const tp = plan({ model: llama70, gpu: h100, dtype: 'bf16', strategy: 'tp', gpus: 4, concurrency: 32, context: 2048 })
    expect(tp.weightsPerGpu).toBeCloseTo((70.554e9 * 2) / 4, -6)
    expect(tp.feasible).toBe(true)
    const dp = plan({ model: llama70, gpu: h100, dtype: 'bf16', strategy: 'dp', gpus: 4, concurrency: 32, context: 2048 })
    expect(dp.feasible).toBe(false)
    expect(dp.weightsFit).toBe(false)
  })
  test('EP: только для MoE; эксперты делятся, остальное копируется', () => {
    const dense = plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'ep', gpus: 8, concurrency: 32, context: 2048 })
    expect(dense.feasible).toBe(false)
    expect(dense.reason).toMatch(/MoE/)
    const ep = plan({ model: qwen235, gpu: h100, dtype: 'fp8', strategy: 'ep', gpus: 8, concurrency: 64, context: 2048 })
    expect(ep.expertsPerGpu).toBe(16)
    expect(ep.weightsPerGpu / 1e9).toBeCloseTo(7.998 + 227.096 / 8, 1)
  })
  test('KV дублируется при TP > kv_heads (Qwen3-235B, TP8 → ×2)', () => {
    const tp = plan({ model: qwen235, gpu: h100, dtype: 'fp8', strategy: 'tp', gpus: 8, concurrency: 64, context: 8192 })
    expect(tp.kvReplication).toBe(2)
    const ep = plan({ model: qwen235, gpu: h100, dtype: 'fp8', strategy: 'ep', gpus: 8, concurrency: 64, context: 8192 })
    expect(ep.maxSeqs).toBeGreaterThan(tp.maxSeqs)
  })
  test('usable как в fitsCheck с memUtil vLLM: 79,65 ГиБ × 0,92 − 2 ГиБ', () => {
    expect(usablePerGpu(h100) / 2 ** 30).toBeCloseTo(79.647 * 0.92 - 2, 1)
  })
})

describe('plan: время и связь', () => {
  test('all-reduce по NVLink (notes-techniques §3.7): Llama-70B, batch 32, TP8 ≈ 0,16 мс полосы и 1,6 мс запусков за шаг', () => {
    const msg = 32 * 8192 * 2
    const perStep = 160 * allReduceMs(msg, 8, 900)
    expect(perStep).toBeCloseTo(0.163, 2)
    const p = plan({ model: llama70, gpu: b200, dtype: 'fp8', strategy: 'tp', gpus: 8, concurrency: 32, context: 512 })
    expect(p.commOps).toBe(160)
    expect(p.commLaunchMs).toBeCloseTo(160 * LAUNCH_MS.nvlink, 6)
    expect(p.commMs).toBeCloseTo(0.163 + 1.6, 2)
    expect(p.commLink).toBe('nvlink')
  })
  test('TP16 через две ноды: связь по InfiniBand дороже чтения весов', () => {
    const p = plan({ model: llama70, gpu: b200, dtype: 'fp8', strategy: 'tp', gpus: 16, concurrency: 32, context: 512 })
    expect(p.commLink).toBe('ib')
    expect(p.commMs).toBeGreaterThan(p.workMs)
  })
  test('PP: при одном запросе общий throughput равен скорости одного пользователя', () => {
    const pp = plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'pp', gpus: 8, concurrency: 1, context: 2048 })
    expect(pp.totalTokS).toBeCloseTo(pp.perUserTokS, 6)
    expect(pp.totalTokS).toBeLessThan(40)
    const tp = plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'tp', gpus: 8, concurrency: 1, context: 2048 })
    expect(tp.totalTokS).toBeGreaterThan(pp.totalTokS * 4)
    // запросов меньше, чем стадий: throughput растёт с числом запросов, а не держится на «полном конвейере»
    const pp4 = plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'pp', gpus: 8, concurrency: 4, context: 2048 })
    expect(pp4.totalTokS).toBeCloseTo(4 * pp4.perUserTokS, 6)
  })
  test('16 GPU: TP16 через InfiniBand проигрывает TP8, а TP8 × PP2 держит all-reduce внутри ноды', () => {
    const tp8 = plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'tp', gpus: 8, concurrency: 1, context: 2048 })
    const tp16 = plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'tp', gpus: 16, concurrency: 1, context: 2048 })
    const hyb = plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'tppp', gpus: 16, concurrency: 1, context: 2048 })
    expect(tp16.tokenMs).toBeGreaterThan(tp8.tokenMs)
    expect(hyb.tokenMs).toBeLessThan(tp16.tokenMs)
    expect(hyb.commIbMs).toBeLessThan(0.1) // одна передача активаций между нодами
    expect(tp16.commIbMs).toBeGreaterThan(4)
    const tp16b = plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'tp', gpus: 16, concurrency: 256, context: 2048 })
    const hybb = plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'tppp', gpus: 16, concurrency: 256, context: 2048 })
    const pp16 = plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'pp', gpus: 16, concurrency: 256, context: 2048 })
    expect(hybb.totalTokS).toBeGreaterThan(2 * tp16b.totalTokS)
    expect(hybb.totalTokS).toBeGreaterThan(2 * pp16.totalTokS)
    expect(plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'tppp', gpus: 8, concurrency: 1, context: 2048 }).feasible).toBe(false)
  })
  test('EP: токен медленнее, чем при TP, зато throughput при большой нагрузке выше', () => {
    const tp1 = plan({ model: qwen235, gpu: h100, dtype: 'fp8', strategy: 'tp', gpus: 8, concurrency: 1, context: 2048 })
    const ep1 = plan({ model: qwen235, gpu: h100, dtype: 'fp8', strategy: 'ep', gpus: 8, concurrency: 1, context: 2048 })
    expect(ep1.tokenMs).toBeGreaterThan(tp1.tokenMs)
    const tp256 = plan({ model: qwen235, gpu: h100, dtype: 'fp8', strategy: 'tp', gpus: 8, concurrency: 256, context: 2048 })
    const ep256 = plan({ model: qwen235, gpu: h100, dtype: 'fp8', strategy: 'ep', gpus: 8, concurrency: 256, context: 2048 })
    expect(ep256.totalTokS).toBeGreaterThan(tp256.totalTokS)
  })
  test('EP16 и TP8 × PP2 на 16 × H100 (текст «Глубже» о двух нодах)', () => {
    const two = (model: typeof dsv3, strategy: 'ep' | 'tppp') => plan({ model, gpu: h100, dtype: 'fp8', strategy, gpus: 16, concurrency: 256, context: 2048 })
    // DeepSeek-V3: TP8 копирует MLA-latent на все 8 GPU, EP16 нет: в память влезает в разы больше запросов, и throughput выше уже при 256
    const dsEp = two(dsv3, 'ep')
    const dsHy = two(dsv3, 'tppp')
    expect(dsHy.kvReplication).toBe(8)
    expect(dsEp.kvReplication).toBe(1)
    expect(dsEp.maxSeqs).toBeGreaterThan(4 * dsHy.maxSeqs)
    expect(dsEp.totalTokS).toBeGreaterThan(dsHy.totalTokS)
    // Qwen3-235B-A22B: KV копируется вдвое, и при 256 запросах TP8 × PP2 «даже чуть впереди»
    const qEp = two(qwen235, 'ep')
    const qHy = two(qwen235, 'tppp')
    expect(qHy.kvReplication).toBe(2)
    expect(qHy.totalTokS).toBeGreaterThan(qEp.totalTokS)
    expect(qHy.totalTokS).toBeLessThan(qEp.totalTokS * 1.1)
    // задержка на пользователя у TP8 × PP2 ниже (строка «MoE, задержка»)
    const one = (strategy: 'ep' | 'tppp') => plan({ model: qwen235, gpu: h100, dtype: 'fp8', strategy, gpus: 16, concurrency: 1, context: 2048 })
    expect(one('tppp').tokenMs).toBeLessThan(one('ep').tokenMs)
  })
  test('PP не дублирует KV даже у MLA (каждая стадия хранит KV только своих слоёв)', () => {
    const pp = plan({ model: dsv3, gpu: h100, dtype: 'fp8', strategy: 'pp', gpus: 16, concurrency: 1, context: 2048 })
    const tp = plan({ model: dsv3, gpu: h100, dtype: 'fp8', strategy: 'tp', gpus: 16, concurrency: 1, context: 2048 })
    expect(pp.kvReplication).toBe(1)
    expect(tp.kvReplication).toBe(16)
    expect(pp.maxSeqs).toBeGreaterThan(tp.maxSeqs * 8)
  })
  test('TP ускоряет одного пользователя, PP нет; PP выигрывает памятью', () => {
    const one = plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'tp', gpus: 1, concurrency: 64, context: 1024 })
    const tp = plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'tp', gpus: 4, concurrency: 64, context: 1024 })
    const pp = plan({ model: llama70, gpu: h100, dtype: 'fp8', strategy: 'pp', gpus: 4, concurrency: 64, context: 1024 })
    expect(one.concurrencyUsed).toBeLessThan(64) // на одном GPU почти нет места под KV
    expect(tp.tokenMs).toBeLessThan(one.tokenMs / 2)
    expect(pp.tokenMs).toBeGreaterThan(tp.tokenMs * 2)
    expect(pp.totalTokS).toBeGreaterThan(one.totalTokS)
  })
})

describe('конвейер', () => {
  test('непрерывный decode: при M ≥ S простоя нет, при M < S простой 1 − M/S', () => {
    expect(decodeBubbleFraction(4, 1)).toBeCloseTo(0.75)
    expect(decodeBubbleFraction(4, 4)).toBe(0)
    expect(decodeBubbleFraction(4, 16)).toBe(0)
    const sch = decodeSchedule(4, 1)
    expect(sch[0]).toEqual([0, -1, -1, -1, 0, -1, -1, -1])
    expect(sch[3]).toEqual([-1, -1, -1, 0, -1, -1, -1, 0])
    const full = decodeSchedule(3, 5)
    for (const row of full) expect(row.every((m) => m >= 0)).toBe(true)
    expect(full[0].slice(0, 5)).toEqual([0, 1, 2, 3, 4])
  })
  test('пузыри: (S − 1) / (M + S − 1)', () => {
    expect(bubbleFraction(4, 1)).toBeCloseTo(0.75)
    expect(bubbleFraction(4, 8)).toBeCloseTo(3 / 11)
    const sch = pipelineSchedule(3, 2)
    expect(sch).toEqual([
      [0, 1, -1, -1],
      [-1, 0, 1, -1],
      [-1, -1, 0, 1],
    ])
  })
})
