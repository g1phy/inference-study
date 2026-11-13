/**
 * Независимая проверка perf.ts (верификатор A2): ВСЕ тест-векторы из research/numbers.json (`checks`,
 * их генерирует research/scripts/numbers_calc.py) и разобранные примеры notes-numbers.md §3.8–3.16,
 * плюс свойства (монотонность, согласованность разбивок), которых нет в исходных тестах.
 */
import { describe, expect, it } from 'vitest'
import numbersJson from '../../research/numbers.json'
import { getGpu, GPUS } from './data/gpus'
import { getModel } from './data/models'
import type { DType } from './types'
import {
  GiB,
  attentionPairs,
  decodeLowerBoundMs,
  decodeStepBreakdown,
  decodeStepMs,
  kvBudgetBytes,
  kvBytesForSeq,
  maxConcurrentSeqs,
  prefillBreakdown,
  prefillFlops,
  prefillFlopsBreakdown,
  prefillMs,
  ridgePoint,
  stepTime,
  tpKvReplication,
  transferMs,
  weightBytes,
} from './perf'

type Check = { what: string } & Record<string, number | string | null>
const checks = (numbersJson as unknown as { checks: Check[] }).checks
const byWhat = (what: string): Check => {
  const c = checks.find((x) => x.what === what)
  if (!c) throw new Error(`нет check «${what}» в numbers.json`)
  return c
}
const IDEAL = { mbu: 1, mfu: 1, overheadMs: 0 } as const

describe('numbers.json checks: нижняя граница decode при batch 1 (все 23)', () => {
  // [what-хвост, модель, dtype, gpu, gpus]
  const cases: Array<[string, string, DType, string, number]> = [
    ['Llama-3.1-8B BF16, 1x H100', 'llama-3.1-8b', 'bf16', 'h100-sxm', 1],
    ['Llama-3.1-8B BF16, 1x H200', 'llama-3.1-8b', 'bf16', 'h200', 1],
    ['Llama-3.1-8B BF16, 1x B200', 'llama-3.1-8b', 'bf16', 'b200', 1],
    ['Llama-3.1-8B BF16, 1x L4', 'llama-3.1-8b', 'bf16', 'l4', 1],
    ['Llama-3.1-8B BF16, 1x RTX 4090', 'llama-3.1-8b', 'bf16', 'rtx-4090', 1],
    ['Llama-3.1-8B FP8, 1x H100', 'llama-3.1-8b', 'fp8', 'h100-sxm', 1],
    ['Llama-3.1-8B FP8, 1x L4', 'llama-3.1-8b', 'fp8', 'l4', 1],
    ['Llama-3.1-70B FP8, 1x H100', 'llama-3.1-70b', 'fp8', 'h100-sxm', 1],
    ['Llama-3.1-70B FP8, 2x H100 TP2', 'llama-3.1-70b', 'fp8', 'h100-sxm', 2],
    ['Llama-3.1-70B FP8, 4x H100 TP4', 'llama-3.1-70b', 'fp8', 'h100-sxm', 4],
    ['Llama-3.1-70B FP8, 8x H100 TP8', 'llama-3.1-70b', 'fp8', 'h100-sxm', 8],
    ['Llama-3.1-70B BF16, 4x H100 TP4', 'llama-3.1-70b', 'bf16', 'h100-sxm', 4],
    ['Llama-3.1-70B FP8, 1x H200', 'llama-3.1-70b', 'fp8', 'h200', 1],
    ['Llama-3.1-70B FP8, 1x B200', 'llama-3.1-70b', 'fp8', 'b200', 1],
    ['Llama-3.1-70B NVFP4, 1x B200', 'llama-3.1-70b', 'fp4', 'b200', 1],
    ['Llama-3.1-405B FP8, 8x H100 TP8', 'llama-3.1-405b', 'fp8', 'h100-sxm', 8],
    ['Llama-3.1-405B FP8, 8x H200 TP8', 'llama-3.1-405b', 'fp8', 'h200', 8],
    ['Llama-3.1-405B FP8, 8x B200 TP8', 'llama-3.1-405b', 'fp8', 'b200', 8],
    ['Qwen3-30B-A3B BF16 (active only), 1x H100', 'qwen3-30b-a3b', 'bf16', 'h100-sxm', 1],
    ['Qwen3-235B-A22B FP8 (active only), 8x H100 ideal spread', 'qwen3-235b-a22b', 'fp8', 'h100-sxm', 8],
    ['DeepSeek-V3 FP8 (active only), 8x H200 ideal spread', 'deepseek-v3', 'fp8', 'h200', 8],
    ['gpt-oss-120b native (active only), 1x H100', 'gpt-oss-120b', 'fp4', 'h100-sxm', 1],
    ['gpt-oss-20b native (active only), 1x RTX 4090', 'gpt-oss-20b', 'fp4', 'rtx-4090', 1],
  ]
  it.each(cases)('%s', (name, mid, dtype, gid, gpus) => {
    const c = byWhat(`decode lower bound batch=1 (weights only): ${name}`)
    const ms = decodeLowerBoundMs(getModel(mid), getGpu(gid), dtype, gpus)
    expect(ms / (c.msPerToken as number)).toBeCloseTo(1, 3) // ≤ 0.05 % (параметры в models.ts округлены до 1e6)
    // тот же результат через общий stepTime с идеальными параметрами
    const viaStep = decodeStepMs(getModel(mid), getGpu(gid), { dtype, gpus, batch: 1, avgContext: 0, ...IDEAL })
    expect(viaStep).toBeGreaterThanOrEqual(ms)
    expect(viaStep - ms).toBeLessThan(0.05) // разница только на KV одного токена
  })
  it('все decode-checks покрыты', () => {
    expect(checks.filter((c) => c.what.startsWith('decode lower bound')).length).toBe(cases.length)
  })
})

describe('numbers.json checks: сколько 8k-последовательностей влезает (все 28)', () => {
  // [what-хвост, модель, dtype весов, gpu, gpus, kvDtype, dpAttention]
  const cases: Array<[string, string, DType, string, number, DType, boolean]> = [
    ['Llama-3.1-8B BF16 on 1x H100', 'llama-3.1-8b', 'bf16', 'h100-sxm', 1, 'bf16', false],
    ['Llama-3.1-8B BF16 on 1x H100, FP8 KV', 'llama-3.1-8b', 'bf16', 'h100-sxm', 1, 'fp8', false],
    ['Llama-3.1-8B BF16 on 1x L4', 'llama-3.1-8b', 'bf16', 'l4', 1, 'bf16', false],
    ['Llama-3.1-8B FP8 on 1x L4', 'llama-3.1-8b', 'fp8', 'l4', 1, 'bf16', false],
    ['Llama-3.1-8B FP8 on 1x L4, FP8 KV', 'llama-3.1-8b', 'fp8', 'l4', 1, 'fp8', false],
    ['Llama-3.1-70B FP8 on 1x H100', 'llama-3.1-70b', 'fp8', 'h100-sxm', 1, 'bf16', false],
    ['Llama-3.1-70B FP8 on 1x H100, FP8 KV', 'llama-3.1-70b', 'fp8', 'h100-sxm', 1, 'fp8', false],
    ['Llama-3.1-70B FP8 on 2x H100 TP2', 'llama-3.1-70b', 'fp8', 'h100-sxm', 2, 'bf16', false],
    ['Llama-3.1-70B BF16 on 4x H100 TP4', 'llama-3.1-70b', 'bf16', 'h100-sxm', 4, 'bf16', false],
    ['Llama-3.1-70B BF16 on 8x H100 TP8', 'llama-3.1-70b', 'bf16', 'h100-sxm', 8, 'bf16', false],
    ['Llama-3.1-70B FP8 on 1x H200', 'llama-3.1-70b', 'fp8', 'h200', 1, 'bf16', false],
    ['Llama-3.1-70B FP8 on 1x H200, FP8 KV', 'llama-3.1-70b', 'fp8', 'h200', 1, 'fp8', false],
    ['Llama-3.1-70B FP8 on 1x B200', 'llama-3.1-70b', 'fp8', 'b200', 1, 'bf16', false],
    ['Llama-3.1-405B FP8 on 8x H100 TP8', 'llama-3.1-405b', 'fp8', 'h100-sxm', 8, 'bf16', false],
    ['Llama-3.1-405B FP8 on 8x H100 TP8, FP8 KV', 'llama-3.1-405b', 'fp8', 'h100-sxm', 8, 'fp8', false],
    ['Llama-3.1-405B FP8 on 8x H200 TP8', 'llama-3.1-405b', 'fp8', 'h200', 8, 'bf16', false],
    ['Llama-3.1-405B FP8 on 8x B200 TP8', 'llama-3.1-405b', 'fp8', 'b200', 8, 'bf16', false],
    // «если KV не дублируется» = DP-attention (каждый ранг держит KV только своих запросов)
    ['Qwen3-235B-A22B FP8 on 8x H100 (если KV не дублируется)', 'qwen3-235b-a22b', 'fp8', 'h100-sxm', 8, 'bf16', true],
    ['Qwen3-235B-A22B FP8 on 8x H100 TP8 (4 KV heads -> x2 дубли)', 'qwen3-235b-a22b', 'fp8', 'h100-sxm', 8, 'bf16', false],
    ['DeepSeek-V3 FP8 on 8x H200, TP8 attention (latent KV дублируется x8)', 'deepseek-v3', 'fp8', 'h200', 8, 'bf16', false],
    ['DeepSeek-V3 FP8 on 8x H200, DP-attention (без дублей)', 'deepseek-v3', 'fp8', 'h200', 8, 'bf16', true],
    ['DeepSeek-V3 FP8 on 8x B200, DP-attention', 'deepseek-v3', 'fp8', 'b200', 8, 'bf16', true],
    ['gpt-oss-120b native MXFP4 on 1x H100 (SWA-aware)', 'gpt-oss-120b', 'fp4', 'h100-sxm', 1, 'bf16', false],
    ['gpt-oss-20b native MXFP4 on 1x L4 (SWA-aware)', 'gpt-oss-20b', 'fp4', 'l4', 1, 'bf16', false],
    ['Qwen3-30B-A3B BF16 on 1x H100', 'qwen3-30b-a3b', 'bf16', 'h100-sxm', 1, 'bf16', false],
    ['Qwen3-30B-A3B FP8 on 1x H100', 'qwen3-30b-a3b', 'fp8', 'h100-sxm', 1, 'bf16', false],
    ['Gemma-3-27B BF16 on 1x H100 (SWA-aware; +0.4B vision)', 'gemma-3-27b', 'bf16', 'h100-sxm', 1, 'bf16', false],
    ['Mistral-Small-24B FP8 on 1x L40S', 'mistral-small-24b', 'fp8', 'l40s', 1, 'bf16', false],
  ]
  it.each(cases)('%s', (name, mid, dtype, gid, gpus, kvDtype, dpAttention) => {
    const c = byWhat(`concurrent 8192-token sequences: ${name}`)
    const m = getModel(mid)
    const g = getGpu(gid)
    const opts = { contextLen: 8192, dtype, gpus, kvDtype, dpAttention }
    expect(maxConcurrentSeqs(m, g, opts)).toBe(c.maxSeqs)
    const vision = mid === 'gemma-3-27b' ? 0.4e9 * 2 : 0 // numbers_calc добавляет vision-башню Gemma
    expect((weightBytes(m, dtype) + vision) / GiB).toBeCloseTo(c.weightsGiB as number, 1)
    expect((kvBudgetBytes(m, g, opts) - vision) / GiB).toBeCloseTo(c.kvBudgetGiB as number, 1)
    const perSeq = kvBytesForSeq(m, 8192, kvDtype) * tpKvReplication(m, gpus, dpAttention)
    expect(perSeq / GiB).toBeCloseTo(c.perSeqGiB as number, 3)
  })
  it('все checks про 8k-последовательности покрыты', () => {
    expect(checks.filter((c) => c.what.startsWith('concurrent 8192')).length).toBe(cases.length)
  })
})

describe('numbers.json checks: prefill FLOPs и идеальное время (все 24)', () => {
  const runs: Array<[string, DType, string, string]> = [
    ['llama-3.1-8b', 'bf16', 'h100-sxm', 'h100-sxm'],
    ['llama-3.1-8b', 'fp8', 'h100-sxm', 'h100-sxm'],
    ['llama-3.1-70b', 'fp8', 'h100-sxm', 'h100-sxm'],
    ['llama-3.1-70b', 'fp8', 'b200', 'b200-hgx'],
    ['llama-3.1-405b', 'fp8', 'h100-sxm', 'h100-sxm'],
    ['qwen3-235b-a22b', 'fp8', 'h100-sxm', 'h100-sxm'],
    ['deepseek-v3', 'fp8', 'h200', 'h200-sxm'],
    ['gpt-oss-120b', 'bf16', 'h100-sxm', 'h100-sxm'],
  ]
  const cases = runs.flatMap(([mid, dt, gid, jsonGid]) => [1024, 8192, 32768].map((S) => [mid, dt, gid, jsonGid, S] as const))
  it.each(cases)('%s %s %s (%s) S=%i', (mid, dtype, gid, jsonGid, S) => {
    const c = byWhat(`prefill ideal time ${mid} ${dtype} on 1x ${jsonGid} S=${S}`)
    const f = prefillFlopsBreakdown(getModel(mid), S)
    expect(f.linear / (c.linearFLOPs as number)).toBeCloseTo(1, 3)
    expect(f.attention / (c.attentionFLOPs as number)).toBeCloseTo(1, 3)
    const ms = prefillBreakdown(getModel(mid), getGpu(gid), S, { dtype, ...IDEAL }).computeMs
    expect(ms / (c.idealMs as number)).toBeCloseTo(1, 3)
  })
  it('все prefill-checks покрыты', () => {
    expect(checks.filter((c) => c.what.startsWith('prefill ideal')).length).toBe(cases.length)
  })
})

describe('ridge point для всех GPU = numbers.json ridgeFlopsPerByte', () => {
  const gpusJson = (numbersJson as unknown as { gpus: Array<{ id: string; ridgeFlopsPerByte: Record<string, number | null> }> }).gpus
  it('BF16 / FP8 / FP4, где формат поддержан', () => {
    for (const src of gpusJson) {
      const g = GPUS.find((x) => x.id === src.id || x.aliases?.includes(src.id))!
      for (const dt of ['bf16', 'fp8', 'fp4'] as const) {
        const want = src.ridgeFlopsPerByte[dt]
        if (want === null) continue
        expect(ridgePoint(g, dt), `${g.id} ${dt}`).toBeCloseTo(want, 0)
      }
    }
  })
})

describe('разобранные примеры notes-numbers.md', () => {
  it('§3.14: Llama-3.1-70B FP8 на 1 × H200, S = 4096: 14.98 / 16.94 / 19.17 / 23.65 / 27.56 мс', () => {
    const m = getModel('llama-3.1-70b')
    const g = getGpu('h200')
    const at = (batch: number) => decodeStepMs(m, g, { batch, avgContext: 4096, dtype: 'fp8', ...IDEAL })
    expect(at(1)).toBeCloseTo(14.98, 1)
    expect(at(8)).toBeCloseTo(16.94, 1)
    expect(at(16)).toBeCloseTo(19.17, 1)
    expect(at(32)).toBeCloseTo(23.65, 1)
    expect(at(46)).toBeCloseTo(27.56, 1)
    // 46 — предел по памяти с FP8 KV (§3.15)
    expect(maxConcurrentSeqs(m, g, { contextLen: 8192, dtype: 'fp8', kvDtype: 'fp8' })).toBe(46)
  })
  it('§3.16: пересылка KV 8k-промпта (1.07 / 2.68 / 0.58 ГБ) по разным каналам', () => {
    const row = (mid: string) => kvBytesForSeq(getModel(mid), 8192)
    expect(transferMs(row('llama-3.1-8b'), 450)).toBeCloseTo(2.4, 1)
    expect(transferMs(row('llama-3.1-8b'), 63)).toBeCloseTo(17.0, 1)
    expect(transferMs(row('llama-3.1-70b'), 100)).toBeCloseTo(26.8, 1)
    expect(transferMs(row('llama-3.1-70b'), 14)).toBeCloseTo(191.7, 1)
    expect(transferMs(row('deepseek-v3'), 50)).toBeCloseTo(11.5, 1)
    expect(transferMs(row('deepseek-v3'), 7)).toBeCloseTo(82.2, 1)
  })
  it('§3.17: L3 8k-префикс 70B на TP8 FP8: пересчёт ≈ 76 мс, 32k ≈ 372 мс', () => {
    const m = getModel('llama-3.1-70b')
    const g = getGpu('h100-sxm')
    expect(prefillBreakdown(m, g, 8192, { dtype: 'fp8', gpus: 8, ...IDEAL }).computeMs).toBeCloseTo(76.4, 0)
    expect(prefillBreakdown(m, g, 32768, { dtype: 'fp8', gpus: 8, ...IDEAL }).computeMs).toBeCloseTo(372.2, 0)
  })
  it('§3.20: чанк 2048 токенов в decode-шаге 64 × 2k ≈ 29 мс compute', () => {
    const m = getModel('llama-3.1-8b')
    const g = getGpu('h100-sxm')
    const s = stepTime(m, g, { prefill: [{ tokens: 2048, last: false }] }, IDEAL)
    expect(s.linearFlops / 1e13).toBeCloseTo(2.86, 2) // 2 × 6.98e9 × 2048
    expect(s.computeMs).toBeCloseTo(29.9, 0)
  })
})

describe('sliding window: формулы против брутфорса по токенам', () => {
  it('attentionPairs(S новых поверх P в кэше) = Σ min(P + i, W) и S·P + S(S+1)/2', () => {
    const goss = getModel('gpt-oss-120b') // W = 128
    const W = 128
    for (const [S, P] of [
      [0, 0], [1, 0], [5, 0], [128, 0], [129, 0], [300, 0], [1, 127], [1, 128], [10, 120], [10, 200], [500, 50], [64, 64],
    ]) {
      let sliding = 0
      let full = 0
      for (let i = 1; i <= S; i++) {
        sliding += Math.min(P + i, W)
        full += P + i
      }
      const got = attentionPairs(goss, S, P)
      expect(got.sliding, `S=${S} P=${P}`).toBe(sliding)
      expect(got.full, `S=${S} P=${P}`).toBe(full)
    }
  })
  it('kvBytesForSeq у gpt-oss и Gemma 3 = Σ по слоям (полный слой: S токенов, sliding: min(S, W))', () => {
    for (const mid of ['gpt-oss-120b', 'gpt-oss-20b', 'gemma-3-27b']) {
      const model = getModel(mid)
      const W = model.slidingWindow!.window
      const perLayer = 2 * model.kvHeads * model.headDim * 2
      for (const S of [0, 1, W - 1, W, W + 1, 3000]) {
        const want = model.slidingWindow!.fullLayers * S * perLayer + (model.layers - model.slidingWindow!.fullLayers) * Math.min(S, W) * perLayer
        expect(kvBytesForSeq(model, S), `${mid} S=${S}`).toBe(want)
      }
    }
  })
  it('decode-шаг SWA-модели: KV читается по окну в sliding-слоях', () => {
    const goss = getModel('gpt-oss-120b')
    const g = getGpu('h100-sxm')
    const s = stepTime(goss, g, { decode: [1000] }, { ...IDEAL, dtype: 'fp4' })
    expect(s.kvBytes).toBe(kvBytesForSeq(goss, 1000))
    const naive = stepTime(goss, g, { decode: [1000] }, { ...IDEAL, dtype: 'fp4', swaAware: false })
    expect(naive.kvBytes).toBe(1000 * 73728)
  })
})

describe('свойства perf.ts', () => {
  const m = getModel('qwen3-8b')
  const g = getGpu('h100-sxm')
  it('prefill монотонен по длине и не убывает по MoE/SWA-моделям', () => {
    for (const mid of ['qwen3-8b', 'qwen3-30b-a3b', 'gpt-oss-120b', 'deepseek-v3', 'gemma-3-27b']) {
      let prev = 0
      for (const S of [1, 2, 16, 100, 127, 128, 129, 1000, 4096, 9000, 32768]) {
        const ms = prefillMs(getModel(mid), getGpu('h200'), S, { dtype: 'fp8' })
        expect(ms, `${mid} ${S}`).toBeGreaterThanOrEqual(prev)
        prev = ms
      }
    }
  })
  it('decode монотонен по batch и по контексту', () => {
    let prev = 0
    for (const b of [1, 2, 4, 8, 16, 64, 256]) {
      const ms = decodeStepMs(m, g, { batch: b, avgContext: 2048 })
      expect(ms).toBeGreaterThanOrEqual(prev)
      prev = ms
    }
    prev = 0
    for (const c of [0, 1, 100, 1000, 10000, 40000]) {
      const ms = decodeStepMs(m, g, { batch: 8, avgContext: c })
      expect(ms).toBeGreaterThanOrEqual(prev)
      prev = ms
    }
  })
  it('разбивка prefill согласована: линейная часть + attention = итог; chunked = цельный по FLOPs', () => {
    const whole = prefillFlopsBreakdown(m, 6000)
    expect(whole.linear + whole.attention).toBe(whole.total)
    // три куска по 2000 с контекстом = тот же attention, линейная часть больше на 2 лишних lm_head
    const chunks = stepTime(m, g, {
      prefill: [
        { tokens: 2000, context: 0, last: false },
        { tokens: 2000, context: 2000, last: false },
        { tokens: 2000, context: 4000, last: true },
      ],
    }, IDEAL)
    expect(chunks.attentionFlops / whole.attention).toBeCloseTo(1, 9)
    expect(chunks.linearFlops / whole.linear).toBeCloseTo(1, 9)
  })
  it('prefix cache: закэшированная часть не считается, но attention новых токенов видит кэш', () => {
    const full = prefillFlopsBreakdown(m, 4096)
    const half = prefillFlopsBreakdown(m, 4096, { cachedTokens: 2048 })
    const fresh = prefillFlopsBreakdown(m, 2048)
    expect(half.attention).toBeGreaterThan(fresh.attention * 2.9) // 2048 новых смотрят на 2048 + себя
    expect(half.total).toBeLessThan(full.total)
    // разница = линейная часть первых 2048 токенов + их attention между собой (lm_head один раз в обоих)
    const lm = m.lmHeadParamsB * 1e9
    expect((full.total - half.total) / (fresh.total - 2 * lm)).toBeCloseTo(1, 9)
  })
  it('весь промпт в кэше: движок всё равно считает последний токен (TTFT не 0)', () => {
    // vLLM: «When all tokens hit the cache, we must recompute the last token to obtain logits»
    const ms = prefillMs(m, g, 4096, { cachedTokens: 4096 })
    expect(ms).toBeGreaterThan(0)
    expect(prefillFlops(m, 4096, { cachedTokens: 4096 })).toBeGreaterThan(0)
    expect(prefillMs(m, g, 4096, { cachedTokens: 4096 })).toBeCloseTo(prefillMs(m, g, 4096, { cachedTokens: 4095 }), 9)
    // и это примерно один decode-шаг по чтению весов, а не пересчёт промпта
    expect(ms).toBeLessThan(prefillMs(m, g, 4096) / 10)
  })
  it('DP-attention снимает дублирование KV и у GQA-моделей (Qwen3-235B: 4 KV-головы на 8 GPU)', () => {
    const q = getModel('qwen3-235b-a22b')
    expect(tpKvReplication(q, 8)).toBe(2)
    expect(tpKvReplication(q, 8, true)).toBe(1)
    expect(tpKvReplication(getModel('deepseek-v3'), 8, true)).toBe(1)
  })
  it('нижняя граница decode ≤ любого реального шага; шаг ≥ накладных', () => {
    for (const mid of ['qwen3-8b', 'llama-3.1-70b', 'qwen3-30b-a3b', 'gpt-oss-120b']) {
      const model = getModel(mid)
      const lb = decodeLowerBoundMs(model, g, 'fp8', 4)
      const real = decodeStepBreakdown(model, g, { dtype: 'fp8', gpus: 4, batch: 1, avgContext: 100 })
      expect(real.ms).toBeGreaterThan(lb)
      expect(real.ms).toBeGreaterThanOrEqual(real.overheadMs)
    }
  })
})
