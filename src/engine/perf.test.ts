import { describe, expect, it } from 'vitest'
import { getGpu } from './data/gpus'
import { getModel } from './data/models'
import {
  GiB,
  arithmeticIntensity,
  attainableTflops,
  attentionPairs,
  decodeAttentionIntensity,
  decodeBatchSweep,
  decodeLowerBoundMs,
  decodeStepBreakdown,
  decodeStepMs,
  fitsCheck,
  kvBytesForSeq,
  kvBytesPerToken,
  kvBudgetBytes,
  kvLayout,
  linearLayerIntensity,
  loadVsRecompute,
  maxConcurrentSeqs,
  moeExpertsTouched,
  prefillBreakdown,
  prefillFlops,
  prefillFlopsBreakdown,
  prefillMs,
  ridgePoint,
  stepTime,
  stepWeightBytes,
  tpKvReplication,
  transferMs,
  weightBytes,
} from './perf'

const qwen8 = getModel('qwen3-8b')
const llama8 = getModel('llama-3.1-8b')
const llama70 = getModel('llama-3.1-70b')
const ds = getModel('deepseek-v3')
const goss = getModel('gpt-oss-120b')
const h100 = getGpu('h100-sxm')
const h200 = getGpu('h200')
const b200 = getGpu('b200')

/** Идеальная граница: 100 % полосы и пика, без накладных (так считают заметки). */
const IDEAL = { mbu: 1, mfu: 1, overheadMs: 0 } as const

describe('KV на токен (notes-numbers.md §2.5)', () => {
  it('стандартный GQA', () => {
    expect(kvBytesPerToken(qwen8)).toBe(147456)
    expect(kvBytesPerToken(llama8)).toBe(131072)
    expect(kvBytesPerToken(llama70)).toBe(327680)
    expect(kvBytesPerToken(getModel('qwen3-32b'))).toBe(262144)
    expect(kvBytesPerToken(getModel('qwen3-235b-a22b'))).toBe(192512)
  })
  it('MLA DeepSeek-V3: 61 × 576 × 2 = 70 272, без множителя 2', () => {
    expect(kvBytesPerToken(ds)).toBe(70272)
    expect(kvBytesPerToken(ds, 'fp8')).toBe(35136)
  })
  it('FP8 KV вдвое меньше', () => {
    expect(kvBytesPerToken(qwen8, 'fp8')).toBe(73728)
  })
  it('sliding window gpt-oss-120b (§3.6)', () => {
    expect(kvBytesPerToken(goss)).toBe(73728) // наивно
    const L = kvLayout(goss)
    expect(L.bytesPerTokenFullLayers).toBe(36864)
    expect(L.slidingConstBytes).toBe(4718592) // 4.5 МиБ
    expect(kvBytesForSeq(goss, 8192)).toBe(306708480) // 0.286 ГиБ
    expect(kvBytesForSeq(goss, 8192, 'bf16', { swaAware: false })).toBe(8192 * 73728)
    expect(kvBytesForSeq(goss, 100)).toBe(100 * 73728) // до заполнения окна разницы нет
  })
  it('Gemma 3 27B: 1.031 ГиБ на 8k против 3.875 наивно (§3.7)', () => {
    const g = getModel('gemma-3-27b')
    expect(kvBytesForSeq(g, 8192) / GiB).toBeCloseTo(1.03125, 4)
    expect(kvBytesForSeq(g, 8192, 'bf16', { swaAware: false }) / GiB).toBeCloseTo(3.875, 4)
  })
  it('Llama-3.1-8B: ровно 1 ГиБ на 8192 токена', () => {
    expect(kvBytesForSeq(llama8, 8192)).toBe(GiB)
  })
})

describe('веса', () => {
  it('Llama-3.1-8B BF16 ≈ 16.06 ГБ, FP8 ≈ 8.03 ГБ', () => {
    expect(weightBytes(llama8, 'bf16') / 1e9).toBeCloseTo(16.06, 2)
    expect(weightBytes(llama8, 'fp8') / 1e9).toBeCloseTo(8.03, 2)
  })
  it('gpt-oss-120b нативно (MXFP4 + BF16) ≈ 65.25 ГБ = 60.8 ГиБ', () => {
    expect(weightBytes(goss, 'fp4') / 1e9).toBeCloseTo(65.25, 1)
    expect(weightBytes(goss, 'fp4') / GiB).toBeCloseTo(60.77, 1)
  })
  it('NVFP4 = 4.5 бита на параметр', () => {
    expect(weightBytes(llama70, 'fp4') / 1e9).toBeCloseTo(39.69, 1)
  })
  it('MoE: при batch 1 читаются только активные; при большом batch почти все эксперты', () => {
    const a3b = getModel('qwen3-30b-a3b')
    expect(stepWeightBytes(a3b, 'bf16', 1) / 1e9).toBeCloseTo(6.706, 2)
    expect(moeExpertsTouched(a3b, 1)).toBeCloseTo(8, 6)
    expect(moeExpertsTouched(a3b, 64)).toBeCloseTo(125.9, 1) // §3.18
    expect(moeExpertsTouched(ds, 32)).toBeCloseTo(163.3, 1)
    expect(stepWeightBytes(a3b, 'bf16', 10000) / 1e9).toBeCloseTo(weightBytes(a3b, 'bf16') / 1e9, 1)
  })
})

describe('decode: нижняя граница при batch 1 (notes-numbers.md §3.8)', () => {
  it('Llama-3.1-8B BF16 на H100 ≈ 4.79 мс (≤ 209 ток/с)', () => {
    const ms = decodeLowerBoundMs(llama8, h100, 'bf16')
    expect(ms).toBeCloseTo(4.794, 2)
    expect(1000 / ms).toBeCloseTo(208.6, 0)
    // то же через общий decodeStepMs с идеальными параметрами
    expect(decodeStepMs(llama8, h100, { batch: 1, avgContext: 0, ...IDEAL })).toBeCloseTo(4.794, 2)
  })
  it('H200, B200, FP8, TP', () => {
    expect(decodeLowerBoundMs(llama8, h200)).toBeCloseTo(3.346, 2)
    expect(decodeLowerBoundMs(llama8, b200)).toBeCloseTo(2.008, 2)
    expect(decodeLowerBoundMs(llama8, h100, 'fp8')).toBeCloseTo(2.397, 2)
    expect(decodeLowerBoundMs(llama70, h100, 'fp8', 2)).toBeCloseTo(10.53, 1)
    expect(decodeLowerBoundMs(llama70, h100, 'fp8', 8)).toBeCloseTo(2.633, 2)
  })
  it('MoE: Qwen3-30B-A3B BF16 ≈ 2.0 мс, gpt-oss-120b нативно ≈ 1.49 мс', () => {
    expect(decodeLowerBoundMs(getModel('qwen3-30b-a3b'), h100)).toBeCloseTo(2.002, 2)
    expect(decodeLowerBoundMs(goss, h100, 'fp4')).toBeCloseTo(1.493, 1)
  })
})

describe('decode: throughput против latency (notes-numbers.md §3.14)', () => {
  it('Llama-3.1-8B BF16, H100, S = 2048', () => {
    const at = (batch: number) => decodeStepMs(llama8, h100, { batch, avgContext: 2048, ...IDEAL })
    expect(at(1)).toBeCloseTo(4.87, 1)
    expect(at(8)).toBeCloseTo(5.44, 1)
    expect(at(32)).toBeCloseTo(7.36, 1)
    expect(at(64)).toBeCloseTo(9.92, 1)
    expect(at(128)).toBeCloseTo(15.05, 1)
    expect(at(200)).toBeCloseTo(20.82, 1)
  })
  it('короткий контекст, огромный batch: шаг упирается в compute (S = 256, B = 1024)', () => {
    const s = decodeStepBreakdown(llama8, h100, { batch: 1024, avgContext: 256, ...IDEAL })
    expect(s.memoryMs).toBeCloseTo(15.05, 1)
    expect(s.computeMs).toBeCloseTo(15.68, 1)
    expect(s.bound).toBe('compute')
    expect((1024 * 1000) / s.ms).toBeCloseTo(65308, -2)
  })
  it('Llama-3.1-70B FP8 на H200, S = 4096 (§3.14)', () => {
    const at = (batch: number) => decodeStepMs(llama70, h200, { batch, avgContext: 4096, dtype: 'fp8', ...IDEAL })
    expect(at(1)).toBeCloseTo(14.98, 1)
    expect(at(32)).toBeCloseTo(23.65, 1)
  })
  it('с дефолтными MBU 0.7 и накладными шаг длиннее идеала', () => {
    const real = decodeStepMs(qwen8, h100, { batch: 1, avgContext: 512 })
    const ideal = decodeStepMs(qwen8, h100, { batch: 1, avgContext: 512, ...IDEAL })
    expect(real).toBeGreaterThan(ideal / 0.7)
    expect(real).toBeLessThan(ideal / 0.7 + 1)
  })
  it('sweep: per-user скорость падает, общий throughput растёт', () => {
    const pts = decodeBatchSweep(llama8, h100, [1, 8, 64], { avgContext: 2048 })
    expect(pts[0].perUserTokS).toBeGreaterThan(pts[2].perUserTokS)
    expect(pts[2].totalTokS).toBeGreaterThan(pts[0].totalTokS * 10)
    expect(pts.every((p) => p.fits)).toBe(true)
  })
})

describe('prefill (notes-numbers.md §3.9, research/numbers.json checks)', () => {
  it('Llama-3.1-8B, 8192 токена: линейная часть 1.144e14, attention 1.759e13', () => {
    const f = prefillFlopsBreakdown(llama8, 8192)
    expect(f.linear / 1e14).toBeCloseTo(1.1435, 3)
    expect(f.attention / 1e13).toBeCloseTo(1.7594, 3)
    expect(prefillFlops(llama8, 8192) / 1e14).toBeCloseTo(1.3195, 3)
  })
  it('идеальное время на H100 (100 % пика): 14.73 / 133.41 / 747.1 мс', () => {
    const at = (S: number) => prefillBreakdown(llama8, h100, S, IDEAL).computeMs
    expect(at(1024)).toBeCloseTo(14.73, 1)
    expect(at(8192)).toBeCloseTo(133.41, 1)
    expect(at(32768)).toBeCloseTo(747.1, 0)
  })
  it('другие модели из checks', () => {
    expect(prefillBreakdown(llama70, h100, 8192, { dtype: 'fp8', ...IDEAL }).computeMs).toBeCloseTo(611.15, 0)
    expect(prefillBreakdown(ds, h200, 1024, { dtype: 'fp8', ...IDEAL }).computeMs).toBeCloseTo(38.27, 0)
    expect(prefillBreakdown(goss, h100, 8192, IDEAL).computeMs).toBeCloseTo(85.73, 0)
    const q235 = getModel('qwen3-235b-a22b')
    expect(prefillBreakdown(q235, h100, 32768, { dtype: 'fp8', ...IDEAL }).computeMs).toBeCloseTo(1529.25, -1)
  })
  it('attention gpt-oss учитывает sliding-слои', () => {
    const f = prefillFlopsBreakdown(goss, 1024)
    expect(f.attention / 1e11).toBeCloseTo(1.9103, 2)
    expect(f.linear / 1e12).toBeCloseTo(9.3242, 2)
  })
  it('пары attention: полный causal и окно', () => {
    expect(attentionPairs(llama8, 4, 0).full).toBe(10)
    expect(attentionPairs(llama8, 2, 3).full).toBe(2 * 3 + 3)
    const p = attentionPairs(goss, 200, 0) // окно 128
    expect(p.sliding).toBe((128 * 129) / 2 + 128 * 72)
  })
  it('prefix cache: считаются только новые токены, но attention смотрит на кэш', () => {
    const full = prefillFlops(llama8, 4096)
    const cached = prefillFlops(llama8, 4096, { cachedTokens: 3072 })
    expect(cached).toBeLessThan(full / 3)
    // весь промпт в кэше: последний токен всё равно считается (как в vLLM), FLOPs ≈ одного токена
    const allCached = prefillFlops(llama8, 4096, { cachedTokens: 4096 })
    expect(allCached).toBe(prefillFlops(llama8, 4096, { cachedTokens: 4095 }))
    expect(allCached).toBeGreaterThan(0)
    expect(allCached).toBeLessThan(full / 1000)
  })
  it('короткий промпт memory-bound, длинный compute-bound (граница ≈ 340 токенов для 8B на H100)', () => {
    expect(prefillBreakdown(llama8, h100, 100, IDEAL).bound).toBe('memory')
    expect(prefillBreakdown(llama8, h100, 2000, IDEAL).bound).toBe('compute')
    // с дефолтами (MFU 0.5) prefill 4096 токенов ≈ 125 мс (notes-models §3.12)
    expect(prefillMs(llama8, h100, 4096)).toBeGreaterThan(115)
    expect(prefillMs(llama8, h100, 4096)).toBeLessThan(135)
  })
})

describe('roofline', () => {
  it('ridge point (notes-numbers.md §2.2)', () => {
    expect(ridgePoint(h100, 'bf16')).toBeCloseTo(295.2, 1)
    expect(ridgePoint(h100, 'fp8')).toBeCloseTo(590.7, 1)
    expect(ridgePoint(h200, 'bf16')).toBeCloseTo(206.0, 1)
    expect(ridgePoint(b200, 'fp4')).toBeCloseTo(1125, 0)
    expect(ridgePoint(getGpu('a100-80g'), 'bf16')).toBeCloseTo(153.0, 0)
  })
  it('attainable: наклонная часть и полка', () => {
    expect(attainableTflops(h100, 'bf16', 1)).toBeCloseTo(3.35, 6)
    expect(attainableTflops(h100, 'bf16', 62.4)).toBeCloseTo(209, 0)
    expect(attainableTflops(h100, 'bf16', 5000)).toBe(989)
  })
  it('интенсивность линейного слоя (§3.10)', () => {
    expect(linearLayerIntensity(64, 4096, 4096)).toBeCloseTo(62.1, 1)
    expect(linearLayerIntensity(345, 4096, 4096)).toBeCloseTo(295.3, 0)
    expect(linearLayerIntensity(1, 8192, 8192)).toBeCloseTo(1.0, 2)
  })
  it('attention в decode: ≈ g у GQA, ≈ 242 у MLA', () => {
    expect(decodeAttentionIntensity(llama8)).toBeCloseTo(4, 6)
    expect(decodeAttentionIntensity(llama70)).toBeCloseTo(8, 6)
    expect(decodeAttentionIntensity(getModel('qwen3-235b-a22b'))).toBeCloseTo(16, 6)
    expect(decodeAttentionIntensity(ds)).toBeCloseTo(241.8, 1)
  })
  it('шаг decode при batch 1 ≈ 1 FLOP/байт, prefill тысячи', () => {
    expect(arithmeticIntensity(llama8, { phase: 'decode', batch: 1, avgContext: 16 }).intensity).toBeCloseTo(1, 0)
    expect(arithmeticIntensity(llama8, { phase: 'prefill', tokens: 4096 }).intensity).toBeGreaterThan(3000)
    // при длинном контексте batch не спасает: предел ≈ 32 при S = 4096 (§3.12)
    const big = arithmeticIntensity(llama8, { phase: 'decode', batch: 100000, avgContext: 4096 }).intensity
    expect(big).toBeCloseTo(31.9, 0)
  })
})

describe('передача KV (notes-numbers.md §3.16)', () => {
  it('8k токенов Llama-3.1-70B BF16', () => {
    const bytes = kvBytesForSeq(llama70, 8192)
    expect(bytes / 1e9).toBeCloseTo(2.684, 2)
    expect(transferMs(bytes, 50)).toBeCloseTo(53.7, 0)
    expect(transferMs(bytes, 900)).toBeCloseTo(2.98, 1)
    expect(transferMs(bytes, 450, 0.5)).toBeCloseTo(6.46, 1)
  })
  it('загрузить или пересчитать: RAM по PCIe быстрее пересчёта длинного префикса', () => {
    const r = loadVsRecompute(llama70, h100, 32768, 63, { dtype: 'fp8', gpus: 8 })
    expect(r.loadIsFaster).toBe(true)
    expect(r.breakEvenGBs).toBeGreaterThan(5)
  })
})

describe('память и сколько последовательностей влезает (notes-numbers.md §3.15)', () => {
  const at8k = { contextLen: 8192 }
  it('Llama-3.1-8B BF16 на H100: 54 (FP8 KV: 109)', () => {
    expect(kvBudgetBytes(llama8, h100) / GiB).toBeCloseTo(54.73, 1)
    expect(maxConcurrentSeqs(llama8, h100, at8k)).toBe(54)
    expect(maxConcurrentSeqs(llama8, h100, { ...at8k, kvDtype: 'fp8' })).toBe(109)
  })
  it('Llama-3.1-70B FP8: 1 × H100 → 1, 2 × H100 → 29, 1 × H200 → 23, 1 × B200 → 37', () => {
    expect(maxConcurrentSeqs(llama70, h100, { ...at8k, dtype: 'fp8' })).toBe(1)
    expect(maxConcurrentSeqs(llama70, h100, { ...at8k, dtype: 'fp8', gpus: 2 })).toBe(29)
    expect(maxConcurrentSeqs(llama70, h200, { ...at8k, dtype: 'fp8' })).toBe(23)
    expect(maxConcurrentSeqs(llama70, h200, { ...at8k, dtype: 'fp8', kvDtype: 'fp8' })).toBe(46)
    expect(maxConcurrentSeqs(llama70, b200, { ...at8k, dtype: 'fp8' })).toBe(37)
  })
  it('дублирование KV при TP: Qwen3-235B TP8 → 115, DeepSeek-V3 TP8 → 86, DP-attention → 690', () => {
    const q235 = getModel('qwen3-235b-a22b')
    expect(tpKvReplication(q235, 8)).toBe(2)
    expect(maxConcurrentSeqs(q235, h100, { ...at8k, dtype: 'fp8', gpus: 8 })).toBe(115)
    expect(maxConcurrentSeqs(ds, h200, { ...at8k, dtype: 'fp8', gpus: 8 })).toBe(86)
    expect(maxConcurrentSeqs(ds, h200, { ...at8k, dtype: 'fp8', gpus: 8, dpAttention: true })).toBe(690)
  })
  it('MoE и sliding window', () => {
    expect(maxConcurrentSeqs(getModel('qwen3-30b-a3b'), h100, at8k)).toBe(17)
    expect(maxConcurrentSeqs(getModel('qwen3-30b-a3b'), h100, { ...at8k, dtype: 'fp8' })).toBe(54)
    expect(maxConcurrentSeqs(goss, h100, { ...at8k, dtype: 'fp4' })).toBe(31)
    expect(maxConcurrentSeqs(goss, h100, { ...at8k, dtype: 'fp4', swaAware: false })).toBe(15)
    expect(maxConcurrentSeqs(getModel('mistral-small-24b'), getGpu('l40s'), { ...at8k, dtype: 'fp8' })).toBe(13)
  })
  it('fitsCheck: понятный ответ по-русски', () => {
    const no = fitsCheck(llama70, h100, { contextLen: 8192, seqs: 4 })
    expect(no.weightsFit).toBe(false)
    expect(no.fits).toBe(false)
    expect(no.message).toMatch(/Веса не помещаются/)
    const kvNo = fitsCheck(llama8, h100, { contextLen: 8192, seqs: 100 })
    expect(kvNo.weightsFit).toBe(true)
    expect(kvNo.fits).toBe(false)
    expect(kvNo.maxSeqs).toBe(54)
    expect(kvNo.message).toMatch(/максимум 54/)
    const ok = fitsCheck(qwen8, h100, { contextLen: 4096, seqs: 10 })
    expect(ok.fits).toBe(true)
    expect(ok.freeBytes).toBeGreaterThan(0)
    expect(ok.message).toMatch(/^Влезает/)
  })
  it('fitsCheck: числа как в легенде (разряды, ТБ) и слова согласованы с числом', () => {
    const big = fitsCheck(qwen8, h100, { contextLen: 131072, seqs: 512 })
    expect(big.message).toContain('KV для 512\u00a0последовательностей по 131\u00a0072\u00a0токена не влезает')
    expect(big.message).toMatch(/нужно \d+(,\d+)?\u00a0ТБ, под KV есть \d+(,\d+)?\u00a0ГБ/)
    expect(fitsCheck(qwen8, h100, { contextLen: 32768, seqs: 21 }).message).toContain('для 21\u00a0последовательности по 32\u00a0768\u00a0токенов')
    expect(fitsCheck(qwen8, h100, { contextLen: 4096, seqs: 1 }).message).toMatch(/KV 604\u00a0МБ на 1\u00a0посл\./)
  })
})

describe('смешанный шаг (для планировщика)', () => {
  it('чанк prefill в decode-шаге удлиняет шаг (notes-numbers.md §3.20)', () => {
    const decodeOnly = stepTime(llama8, h100, { decodeUniform: { batch: 64, context: 2048 } }, IDEAL)
    expect(decodeOnly.ms).toBeCloseTo(9.92, 1)
    const mixed = stepTime(llama8, h100, { decodeUniform: { batch: 64, context: 2048 }, prefill: [{ tokens: 2048 }] }, IDEAL)
    expect(mixed.ms).toBeGreaterThan(25)
    expect(mixed.ms).toBeLessThan(40)
    const small = stepTime(llama8, h100, { decodeUniform: { batch: 64, context: 2048 }, prefill: [{ tokens: 512 }] }, IDEAL)
    expect(small.ms).toBeLessThan(decodeOnly.ms * 1.2)
  })
  it('пустой шаг занимает 0 мс', () => {
    expect(stepTime(llama8, h100, {}).ms).toBe(0)
  })
})
