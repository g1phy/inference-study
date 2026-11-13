import { describe, expect, it } from 'vitest'
import { getGpu } from '@/engine/data/gpus'
import { getModel } from '@/engine/data/models'
import { coldStartStages, coldStartTotal } from './coldstartModel'
import { planPools, plannerDecodeReplicas, plannerPrefillReplicas, replicaGpus } from './planner'

describe('cold start', () => {
  const m = getModel('qwen3-8b')
  it('веса считаются из размера модели: 16,4 ГБ BF16 с Hugging Face по 0,2 ГБ/с ≈ 82 с', () => {
    const s = coldStartStages({ model: m, dtype: 'bf16', source: 'hf', warmPool: true, imageCached: true, engine: 'vllm', compileCache: true })
    const dl = s.find((x) => x.id === 'download')!
    expect(dl.seconds).toBeCloseTo(16.382 / 0.2, 1)
    expect(dl.computed).toBe(true)
  })
  it('FP8 вдвое сокращает скачивание, кэш на узле убирает его совсем', () => {
    const bf = coldStartStages({ model: m, dtype: 'bf16', source: 's3', warmPool: true, imageCached: true, engine: 'vllm', compileCache: true })
    const f8 = coldStartStages({ model: m, dtype: 'fp8', source: 's3', warmPool: true, imageCached: true, engine: 'vllm', compileCache: true })
    const nd = coldStartStages({ model: m, dtype: 'fp8', source: 'node', warmPool: true, imageCached: true, engine: 'vllm', compileCache: true })
    const d = (s: typeof bf) => s.find((x) => x.id === 'download')!.seconds
    expect(d(f8)).toBeCloseTo(d(bf) / 2, 6)
    expect(d(nd)).toBe(0)
  })
  it('TensorRT-LLM 1.x без сборки engine: компиляция того же порядка, что у vLLM; кэш её сокращает; худший случай дольше лучшего в разы', () => {
    const worst = coldStartStages({ model: m, dtype: 'bf16', source: 'hf', warmPool: false, imageCached: false, engine: 'trtllm', compileCache: false })
    const cached = coldStartStages({ model: m, dtype: 'bf16', source: 'hf', warmPool: false, imageCached: false, engine: 'trtllm', compileCache: true })
    const vllm = coldStartStages({ model: m, dtype: 'bf16', source: 'hf', warmPool: false, imageCached: false, engine: 'vllm', compileCache: false })
    const best = coldStartStages({ model: m, dtype: 'fp8', source: 'node', warmPool: true, imageCached: true, engine: 'vllm', compileCache: true })
    const compile = (s: typeof worst) => s.find((x) => x.id === 'compile')!.seconds
    expect(compile(worst)).toBeLessThan(180)
    expect(compile(worst) / compile(vllm)).toBeLessThan(2)
    expect(compile(cached)).toBeLessThan(compile(worst))
    expect(coldStartTotal(worst) / coldStartTotal(best)).toBeGreaterThan(10)
  })
})

describe('Planner', () => {
  it('классические формулы на примере из заметок (notes-dynamo §3.9): 2P3D', () => {
    expect(plannerPrefillReplicas({ requests: 1800, isl: 3000, intervalSec: 180, correction: 0.9, throughputPerGpu: 15000, gpusPerEngine: 1 })).toBe(2)
    expect(plannerDecodeReplicas({ requests: 1800, osl: 300, intervalSec: 180, throughputPerGpu: 600, gpusPerEngine: 2 })).toBe(3)
  })
  it('длинные промпты растят prefill-пул, длинные ответы растят decode-пул', () => {
    const m = getModel('qwen3-8b')
    const g = getGpu('h100-sxm')
    const base = planPools(m, g, { rps: 40, isl: 2000, osl: 300, ttftMs: 500, itlMs: 50 })
    const longIn = planPools(m, g, { rps: 40, isl: 16000, osl: 300, ttftMs: 500, itlMs: 50 })
    const longOut = planPools(m, g, { rps: 40, isl: 2000, osl: 2000, ttftMs: 500, itlMs: 50 })
    expect(longIn.prefill.replicas).toBeGreaterThan(base.prefill.replicas)
    expect(longOut.decode.replicas).toBeGreaterThan(base.decode.replicas)
    expect(longOut.prefill.replicas).toBe(base.prefill.replicas)
    expect(base.decode.stepMs).toBeLessThanOrEqual(50)
  })
  it('длинный промпт растит и decode-пул: batch decode упирается в память под KV', () => {
    const m = getModel('qwen3-8b')
    const g = getGpu('h100-sxm')
    const base = planPools(m, g, { rps: 20, isl: 2048, osl: 300, ttftMs: 500, itlMs: 50 })
    const long = planPools(m, g, { rps: 20, isl: 16384, osl: 300, ttftMs: 500, itlMs: 50 })
    expect(long.decode.replicas).toBeGreaterThan(base.decode.replicas)
    expect(long.decode.batch).toBeLessThan(base.decode.batch / 3)
    expect(long.decode.limit).toBe('kv')
  })
  it('реплике DeepSeek-V3 на H100 не хватает одного узла: FP8 на двух, BF16 на четырёх', () => {
    const ds = getModel('deepseek-v3')
    const g = getGpu('h100-sxm')
    expect(replicaGpus(ds, g, 4096, 'fp8')).toEqual({ gpus: 16, fits: true })
    expect(replicaGpus(ds, g, 4096, 'bf16')).toEqual({ gpus: 32, fits: true })
    expect(replicaGpus(getModel('qwen3-8b'), g, 4096, 'bf16')).toEqual({ gpus: 1, fits: true })
  })
  it('70B на H100 требует TP', () => {
    const p = planPools(getModel('llama-3.1-70b'), getGpu('h100-sxm'), { rps: 5, isl: 2000, osl: 300, ttftMs: 500, itlMs: 50 })
    expect(p.decode.gpusPerEngine).toBeGreaterThanOrEqual(2)
  })
})
