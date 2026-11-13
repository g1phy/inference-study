/**
 * Формулы производительности инференса (агент A2, «engine-sim»).
 *
 * Модель шага движка: «крыша» (roofline). Один forward-проход читает веса и KV из HBM
 * и делает FLOPs; его время ≈ max(байты / (BW·MBU), FLOPs / (пик·MFU)) + фиксированные накладные.
 * Источники формул: research/notes-numbers.md §2.6 (шпаргалка), §3.8–3.16;
 * notes-models-attention-bottlenecks.md §2.3, §3.11–3.12; numbers_calc.py (prefill_flops, kv_budget).
 *
 * Все величины: байты (не ГБ), FLOPs, миллисекунды. ГБ = 10^9 байт, ГиБ = 2^30 байт.
 * Идеализация: TP делит байты и FLOPs поровну без учёта all-reduce; MoE маршрутизирует
 * токены равномерно; ядра attention считаются «фьюзенными» (матрица n×n в HBM не пишется).
 */
import type { DType, GpuSpec, ModelSpec } from './types'
import type { GpuSpecExt } from './data/gpus'
import type { ModelSpecExt } from './data/models'

/* ============================================================================
 * Константы (все в одном месте, с источниками)
 * ========================================================================== */

/** 1 ГБ = 10^9 байт (так пишут полосы и ёмкости в datasheet). */
export const GB = 1e9
/** 1 ГиБ = 2^30 байт (так считают nvidia-smi и движки). */
export const GiB = 2 ** 30
/** 1 МиБ = 2^20 байт. */
export const MiB = 2 ** 20

/**
 * MFU по умолчанию для prefill: 50 % пика dense FLOPS. Допущение из заметок:
 * notes-models-attention-bottlenecks.md §3.12 («реалистичный TTFT при MFU около 50 %»),
 * notes-kvcache-tiers.md §2.1 F6 (MFU 40–60 % — ползунок).
 */
export const DEFAULT_MFU = 0.5

/**
 * MBU по умолчанию для decode: 70 % пиковой полосы HBM. Между примером Databricks (50 %)
 * и оценкой для 8B на H100 (80 %): notes-hardware-software-engines.md §3.16.
 */
export const DEFAULT_MBU = 0.7

/**
 * Доля памяти GPU под процесс движка: 0.9 (vLLM `gpu_memory_utilization`, default до v0.18;
 * с v0.20 стало 0.92: notes-engines.md §2.3). Так же считает numbers_calc.py (kv_budget, util=0.9):
 * на этом значении держатся сверочные тесты perf.verify.test.ts, поэтому умолчание движка не меняем.
 */
export const DEFAULT_MEM_UTIL = 0.9

/**
 * gpu_memory_utilization vLLM по умолчанию в нынешних релизах (v0.20 и новее, SPEC §8.8). Главы, которые
 * показывают ёмкость vLLM «как из коробки», передают его явно: `{ memUtil: VLLM_MEM_UTIL }`.
 */
export const VLLM_MEM_UTIL = 0.92

/**
 * Резерв на активации, CUDA graphs и буферы: 2 ГиБ на GPU (упрощение калькуляторов
 * notes-numbers.md §1.8 и §3.15: «KV budget = 0.9 × nvidia-smi − 2 GiB/GPU − веса»).
 */
export const DEFAULT_RESERVE_GIB = 2

/**
 * Фиксированные накладные на один шаг движка: 0.3 мс. Допущение: планирование на CPU
 * и запуск CUDA graph; с графами и async scheduling (vLLM V1) почти всё прячется
 * за работой GPU (notes-hardware-software-engines.md §3.9; notes-engines.md §1.4).
 */
export const STEP_OVERHEAD_MS = 0.3

/**
 * Байт на параметр. BF16 2, FP8 1 (notes-numbers.md §2.7). FP4 = NVFP4: E2M1 + FP8-scale
 * на 16 значений = 4.5 бит = 0.5625 байта (NVIDIA NVFP4 blog; книга с. 124).
 */
export const BYTES_PER_PARAM: Readonly<Record<DType, number>> = { bf16: 2, fp8: 1, fp4: 0.5625 }

/**
 * Байт на элемент KV cache. BF16 2, FP8 1 (`--kv-cache-dtype fp8` в vLLM/SGLang),
 * FP4 как NVFP4 0.5625 (SGLang `--kv-cache-dtype nvfp4`, notes-numbers.md §2.8).
 */
export const KV_BYTES_PER_ELEM: Readonly<Record<DType, number>> = { bf16: 2, fp8: 1, fp4: 0.5625 }

/** Модель с необязательными расширенными полями из data/models.ts. */
export type ModelLike = ModelSpec & Partial<ModelSpecExt>
/** GPU с необязательными расширенными полями из data/gpus.ts. */
export type GpuLike = GpuSpec & Partial<GpuSpecExt>

/* ============================================================================
 * Базовые величины модели и GPU
 * ========================================================================== */

/** Байт на параметр весов в формате dtype. */
export function bytesPerParam(dtype: DType): number {
  return BYTES_PER_PARAM[dtype]
}

/**
 * Пиковые dense TFLOPS GPU для формата. Если тензорные ядра формат не поддерживают
 * (FP8 на A100, FP4 на Hopper), считаем математику в BF16: это weight-only квантизация
 * (веса хранятся сжатыми, умножение в BF16), как gpt-oss MXFP4 на H100 (notes-numbers.md §3.9).
 */
export function peakTflops(gpu: GpuSpec, dtype: DType): number {
  return gpu.tflops[dtype] ?? gpu.tflops.bf16
}

/** Реальная ёмкость памяти одного GPU в байтах (nvidia-smi memory.total, иначе memGB как ГиБ, как numbers_calc.py). */
export function usableMemBytes(gpu: GpuLike): number {
  return gpu.usableMiB ? gpu.usableMiB * MiB : gpu.memGB * GiB
}

/** Параметры, которые умножаются на каждый токен (без таблицы эмбеддингов), шт. */
function matmulParams(model: ModelLike): number {
  return (model.activeMatmulParamsB ?? model.activeParamsB) * 1e9
}

/** Параметры LM head (vocab × hidden), шт. */
function lmHeadParams(model: ModelLike): number {
  return model.lmHeadParamsB !== undefined ? model.lmHeadParamsB * 1e9 : model.vocab * model.hidden
}

/* ============================================================================
 * KV cache
 * ========================================================================== */

/** Раскладка KV cache модели: сколько байт на слой и на токен, полные и sliding-слои. */
export interface KvLayout {
  bytesPerElem: number
  /** Чисел на токен на слой: 2·kv_heads·head_dim (K и V) или mlaDim у MLA (один latent, без множителя 2). */
  valuesPerLayerPerToken: number
  bytesPerLayerPerToken: number
  fullLayers: number
  slidingLayers: number
  /** Окно sliding window или null. */
  window: number | null
  /** «Наивно»: все слои хранят весь контекст (= kvBytesPerToken). */
  bytesPerTokenNaive: number
  /** Только слои с полным контекстом (растёт с каждым токеном). */
  bytesPerTokenFullLayers: number
  /** Константа sliding-слоёв при контексте ≥ окна: slidingLayers × window × bytesPerLayerPerToken. */
  slidingConstBytes: number
}

/**
 * Раскладка KV: стандарт 2·L·kv_heads·head_dim·b; MLA L·(kv_lora_rank + rope)·b;
 * sliding window: KV(S) = S·full + min(S, W)·sliding (notes-numbers.md §1.7, §2.6).
 */
export function kvLayout(model: ModelLike, kvDtype: DType = 'bf16'): KvLayout {
  const b = KV_BYTES_PER_ELEM[kvDtype]
  const values = model.attention === 'mla' && model.mlaDim ? model.mlaDim : 2 * model.kvHeads * model.headDim
  const perLayer = values * b
  const window = model.slidingWindow ? model.slidingWindow.window : null
  const fullLayers = model.slidingWindow ? model.slidingWindow.fullLayers : model.layers
  const slidingLayers = model.layers - fullLayers
  return {
    bytesPerElem: b,
    valuesPerLayerPerToken: values,
    bytesPerLayerPerToken: perLayer,
    fullLayers,
    slidingLayers,
    window,
    bytesPerTokenNaive: model.layers * perLayer,
    bytesPerTokenFullLayers: fullLayers * perLayer,
    slidingConstBytes: window ? slidingLayers * window * perLayer : 0,
  }
}

/**
 * Байт KV на один токен контекста по всем слоям («наивно»: каждый слой хранит весь контекст).
 * Qwen3-8B BF16: 2·36·8·128·2 = 147 456; DeepSeek-V3 (MLA): 61·576·2 = 70 272 (notes-numbers.md §2.5).
 * Для sliding window точный размер последовательности даёт kvBytesForSeq.
 */
export function kvBytesPerToken(model: ModelLike, kvDtype: DType = 'bf16'): number {
  return kvLayout(model, kvDtype).bytesPerTokenNaive
}

/**
 * Байт KV на последовательность из `tokens` токенов. С swaAware (по умолчанию) sliding-слои
 * хранят только последние window токенов (vLLM Hybrid KV Cache Manager), иначе все слои полные.
 * gpt-oss-120b, 8192 токена: 306 708 480 байт против 603 979 776 наивно (notes-numbers.md §3.6).
 */
export function kvBytesForSeq(
  model: ModelLike,
  tokens: number,
  kvDtype: DType = 'bf16',
  opts: { swaAware?: boolean } = {},
): number {
  return seqKvBytes(kvLayout(model, kvDtype), tokens, opts.swaAware !== false)
}

/** Байт KV последовательности по готовой раскладке (без повторного kvLayout в горячих циклах). */
function seqKvBytes(L: KvLayout, tokens: number, swaAware: boolean): number {
  const t = Math.max(0, tokens)
  if (!L.window || !swaAware) return t * L.bytesPerTokenNaive
  return t * L.bytesPerTokenFullLayers + Math.min(t, L.window) * L.slidingLayers * L.bytesPerLayerPerToken
}

/**
 * Во сколько раз KV дублируется при tensor parallel: если TP > kv_heads, головы копируются
 * (Qwen3-235B, 4 KV-головы, TP8 → ×2); MLA хранит один latent, общий для всех голов, поэтому
 * при TP он целиком на каждом ранге (×TP). DP-attention (SGLang `--enable-dp-attention`) снимает
 * дублирование у любой модели: каждый ранг держит KV только своих запросов (notes-numbers.md §1.9;
 * numbers.json: «Qwen3-235B-A22B … если KV не дублируется» → 230 вместо 115).
 */
export function tpKvReplication(model: ModelLike, tp: number, dpAttention = false): number {
  if (tp <= 1 || dpAttention) return 1
  if (model.attention === 'mla') return tp
  return tp > model.kvHeads ? tp / model.kvHeads : 1
}

/**
 * Сколько токенов промпта реально берётся из prefix cache: не больше tokens − 1. Если в кэше весь
 * промпт, движок всё равно считает последний токен, чтобы получить логиты первого выходного
 * (vLLM KVCacheManager.get_computed_blocks: «When all tokens hit the cache, we must recompute
 * the last token to obtain logits»; так же sched.ts → newState).
 */
export function effectiveCachedTokens(tokens: number, cachedTokens = 0): number {
  const t = Math.max(0, Math.floor(tokens))
  return t > 0 ? Math.min(Math.max(0, Math.floor(cachedTokens)), t - 1) : 0
}

/* ============================================================================
 * Веса
 * ========================================================================== */

/**
 * Байт весов модели целиком (так она лежит в памяти): параметры × байт на параметр.
 * Если у модели смешанный нативный формат и dtype совпадает с ним (gpt-oss: эксперты MXFP4,
 * остальное BF16), считаем по нему: gpt-oss-120b ≈ 65.25 ГБ (notes-numbers.md §3.6).
 */
export function weightBytes(model: ModelLike, dtype: DType = 'bf16'): number {
  const n = model.native
  if (n && n.dtype === dtype && model.expertParamsB) {
    const expert = model.expertParamsB * 1e9
    return expert * n.expertBytesPerParam + (model.paramsB * 1e9 - expert) * n.otherBytesPerParam
  }
  return model.paramsB * 1e9 * BYTES_PER_PARAM[dtype]
}

/**
 * Сколько разных экспертов задето на слой, если в шаге `tokens` токенов, при равномерной
 * маршрутизации: E·(1 − (1 − k/E)^B) (notes-numbers.md §1.10, §3.18). Для плотной модели 0.
 */
export function moeExpertsTouched(model: ModelLike, tokens: number): number {
  if (!model.experts) return 0
  const { total: E, active: k } = model.experts
  return E * (1 - Math.pow(1 - k / E, Math.max(0, tokens)))
}

/**
 * Байт весов, которые читает один forward-шаг с `tokens` токенами. Плотная модель: все веса
 * (как в заметках: Llama-3.1-8B BF16 = 16.06 ГБ → 4.79 мс на H100). MoE: «всегда активная» часть
 * (activeParamsB как в карточке минус routed-эксперты) плюс задетые эксперты; при batch 1
 * это ровно active (Qwen3-30B-A3B BF16: 6.71 ГБ, notes-numbers.md §3.8).
 */
export function stepWeightBytes(model: ModelLike, dtype: DType = 'bf16', tokens = 1): number {
  const bpp = BYTES_PER_PARAM[dtype]
  if (!model.experts || !model.expertParamsB) return model.activeParamsB * 1e9 * bpp
  const n = model.native && model.native.dtype === dtype ? model.native : null
  const bppExpert = n ? n.expertBytesPerParam : bpp
  const bppOther = n ? n.otherBytesPerParam : bpp
  const { total: E, active: k } = model.experts
  const expert = model.expertParamsB * 1e9
  const always = Math.max(0, model.activeParamsB * 1e9 - (expert * k) / E)
  const touchedFrac = moeExpertsTouched(model, Math.max(1, tokens)) / E
  return always * bppOther + expert * touchedFrac * bppExpert
}

/* ============================================================================
 * FLOPs
 * ========================================================================== */

/** Пары (запрос, ключ) attention на один слой: для полных слоёв и для sliding-слоёв. */
export interface AttnPairs {
  full: number
  sliding: number
}

/**
 * Пары attention, когда `newTokens` новых токенов смотрят на `context` уже лежащих в KV
 * и друг на друга (causal): полный слой S·P + S(S+1)/2; sliding-слой Σ min(P + i, W).
 */
export function attentionPairs(model: ModelLike, newTokens: number, context = 0): AttnPairs {
  const S = Math.max(0, newTokens)
  const P = Math.max(0, context)
  const full = S * P + (S * (S + 1)) / 2
  const W = model.slidingWindow?.window
  if (!W) return { full, sliding: 0 }
  // Σ_{x=a}^{b} min(x, W), где x = число ключей у i-го нового токена.
  const a = P + 1
  const b = P + S
  let sliding: number
  if (S === 0) sliding = 0
  else if (b <= W) sliding = ((a + b) * (b - a + 1)) / 2
  else if (a > W) sliding = W * S
  else sliding = ((a + W) * (W - a + 1)) / 2 + W * (b - W)
  return { full, sliding }
}

/**
 * FLOPs attention на одну пару (запрос, ключ) на один слой: QKᵀ и взвешивание V,
 * 2·heads·(d_qk + d_v). Llama-3.1-8B: 2·32·(128 + 128) = 16 384 (notes-numbers.md §3.9).
 * MLA в decode считается в «absorbed»-форме: 2·heads·(576 + 512) = 278 528 у DeepSeek-V3
 * (notes-numbers.md §3.13); в prefill в обычной форме (qk 192, v 128), как numbers_calc.py.
 */
export function attentionFlopsPerPair(model: ModelLike, phase: 'prefill' | 'decode' = 'prefill'): number {
  if (phase === 'decode' && model.attention === 'mla' && model.mlaDim) {
    const latent = model.kvLoraRank ?? model.mlaDim - (model.qkRopeHeadDim ?? 64)
    return 2 * model.heads * (model.mlaDim + latent)
  }
  return 2 * model.heads * (model.headDim + (model.vHeadDim ?? model.headDim))
}

/** Разбивка FLOPs prefill. */
export interface FlopsBreakdown {
  linear: number
  attention: number
  total: number
}

/**
 * FLOPs prefill по частям: линейные слои 2·(N_matmul − N_lmhead)·S + 2·N_lmhead (логиты только
 * для последнего токена) плюс causal attention 2·(d_qk + d_v)·пары по слоям (notes-numbers.md §1.13).
 * `tokens` — полная длина промпта, `cachedTokens` — сколько из них уже в prefix cache:
 * считаются только новые токены, но attention новых смотрит и на закэшированные. Весь промпт
 * в кэше: последний токен всё равно считается (effectiveCachedTokens).
 */
export function prefillFlopsBreakdown(model: ModelLike, tokens: number, opts: { cachedTokens?: number } = {}): FlopsBreakdown {
  const cached = effectiveCachedTokens(tokens, opts.cachedTokens)
  const S = Math.max(0, Math.floor(tokens) - cached)
  if (S === 0) return { linear: 0, attention: 0, total: 0 }
  const lm = lmHeadParams(model)
  const linear = 2 * (matmulParams(model) - lm) * S + 2 * lm
  const pairs = attentionPairs(model, S, cached)
  const L = kvLayout(model)
  const attention = attentionFlopsPerPair(model, 'prefill') * (L.fullLayers * pairs.full + L.slidingLayers * pairs.sliding)
  return { linear, attention, total: linear + attention }
}

/**
 * FLOPs prefill (2 × активные параметры × токены плюс attention-член).
 * Llama-3.1-8B, 8192 токена: ≈ 1.32·10^14 (notes-numbers.md §3.9).
 */
export function prefillFlops(model: ModelLike, tokens: number, opts: { cachedTokens?: number } = {}): number {
  return prefillFlopsBreakdown(model, tokens, opts).total
}

/* ============================================================================
 * Время шага (roofline)
 * ========================================================================== */

/** Общие параметры производительности. */
export interface PerfOptions {
  /** Формат весов и математики, по умолчанию 'bf16'. */
  dtype?: DType
  /** Формат KV cache, по умолчанию 'bf16'. */
  kvDtype?: DType
  /** Доля пика FLOPS (по умолчанию DEFAULT_MFU = 0.5). */
  mfu?: number
  /** Доля пиковой полосы HBM (по умолчанию DEFAULT_MBU = 0.7). */
  mbu?: number
  /** Число GPU (tensor parallel), по умолчанию 1. */
  gpus?: number
  /** Накладные на шаг, мс (по умолчанию STEP_OVERHEAD_MS). Для «нижней границы» ставьте 0. */
  overheadMs?: number
  /** DeepSeek / MLA: DP-attention вместо TP-attention (KV не дублируется). */
  dpAttention?: boolean
  /** Учитывать sliding window при подсчёте KV (по умолчанию true). */
  swaAware?: boolean
}

/** Работа одного шага движка (одной итерации forward). */
export interface StepWork {
  /**
   * Куски prefill: `tokens` новых токенов поверх `context` токенов, уже лежащих в KV
   * (prefix cache и прошлые куски). `last` (по умолчанию true): кусок дописывает промпт до конца,
   * значит считаются логиты и сэмплируется первый токен.
   */
  prefill?: Array<{ tokens: number; context?: number; last?: boolean }>
  /** Длины последовательностей (KV вместе с текущим токеном) у тех, кто делает decode-шаг. */
  decode?: number[]
  /** То же для одинаковых последовательностей: batch штук длиной context (быстрее, чем массив). */
  decodeUniform?: { batch: number; context: number }
}

/** Разбивка времени шага. */
export interface StepBreakdown {
  /** Итог: max(memoryMs, computeMs) + overheadMs. */
  ms: number
  memoryMs: number
  computeMs: number
  overheadMs: number
  /** Байты из HBM на весь шаг (сумма по GPU): веса + KV. */
  bytes: number
  weightBytes: number
  kvBytes: number
  flops: number
  linearFlops: number
  attentionFlops: number
  /** Арифметическая интенсивность шага, FLOP/байт. */
  intensity: number
  bound: 'memory' | 'compute'
  /** Токенов обработано за шаг (prefill + decode). */
  tokens: number
}

/**
 * Время одного шага движка с произвольной смесью prefill-кусков и decode-последовательностей
 * (основа для sched.ts). Память: веса читаются один раз на шаг (для MoE: задетые эксперты),
 * KV каждой последовательности читается целиком; compute: 2·N·T + attention.
 */
export function stepTime(model: ModelLike, gpu: GpuLike, work: StepWork, opts: PerfOptions = {}): StepBreakdown {
  const dtype = opts.dtype ?? 'bf16'
  const kvDtype = opts.kvDtype ?? 'bf16'
  const mfu = opts.mfu ?? DEFAULT_MFU
  const mbu = opts.mbu ?? DEFAULT_MBU
  const gpus = Math.max(1, opts.gpus ?? 1)
  const overheadMs = opts.overheadMs ?? STEP_OVERHEAD_MS
  const swa = opts.swaAware !== false
  const L = kvLayout(model, kvDtype)
  const perPairPrefill = attentionFlopsPerPair(model, 'prefill')
  const perPairDecode = attentionFlopsPerPair(model, 'decode')

  let prefillTokens = 0
  let logitRows = 0
  let attentionFlops = 0
  let kvBytes = 0
  for (const ch of work.prefill ?? []) {
    const t = Math.max(0, ch.tokens)
    const ctx = Math.max(0, ch.context ?? 0)
    if (t === 0) continue
    prefillTokens += t
    if (ch.last !== false) logitRows += 1
    const p = attentionPairs(model, t, ctx)
    attentionFlops += perPairPrefill * (L.fullLayers * p.full + L.slidingLayers * p.sliding)
    // Читаем KV контекста и пишем KV новых токенов.
    kvBytes += seqKvBytes(L, ctx + t, swa)
  }

  let decodeSeqs = 0
  const addDecode = (len: number, count: number) => {
    if (count <= 0) return
    const c = Math.max(1, len)
    decodeSeqs += count
    const keysFull = c
    const keysSliding = L.window ? Math.min(c, L.window) : 0
    attentionFlops += count * perPairDecode * (L.fullLayers * keysFull + L.slidingLayers * keysSliding)
    kvBytes += count * seqKvBytes(L, c, swa)
  }
  for (const len of work.decode ?? []) addDecode(len, 1)
  if (work.decodeUniform) addDecode(work.decodeUniform.context, work.decodeUniform.batch)
  logitRows += decodeSeqs

  const tokens = prefillTokens + decodeSeqs
  if (tokens === 0) {
    return {
      ms: 0, memoryMs: 0, computeMs: 0, overheadMs: 0, bytes: 0, weightBytes: 0, kvBytes: 0,
      flops: 0, linearFlops: 0, attentionFlops: 0, intensity: 0, bound: 'memory', tokens: 0,
    }
  }

  const lm = lmHeadParams(model)
  const linearFlops = 2 * (matmulParams(model) - lm) * tokens + 2 * lm * logitRows
  const flops = linearFlops + attentionFlops
  const wBytes = stepWeightBytes(model, dtype, tokens)
  const rep = tpKvReplication(model, gpus, opts.dpAttention)
  const bytesTotal = wBytes + kvBytes * rep
  const memoryMs = (bytesTotal / gpus / (gpu.bwTBs * 1e12 * mbu)) * 1000
  const computeMs = (flops / gpus / (peakTflops(gpu, dtype) * 1e12 * mfu)) * 1000
  return {
    ms: Math.max(memoryMs, computeMs) + overheadMs,
    memoryMs,
    computeMs,
    overheadMs,
    bytes: bytesTotal,
    weightBytes: wBytes,
    kvBytes: kvBytes * rep,
    flops,
    linearFlops,
    attentionFlops,
    intensity: flops / bytesTotal,
    bound: computeMs > memoryMs ? 'compute' : 'memory',
    tokens,
  }
}

/** Параметры prefill. */
export interface PrefillOptions extends PerfOptions {
  /** Сколько токенов промпта уже в prefix cache (их не считаем). */
  cachedTokens?: number
}

/**
 * Разбивка времени prefill одного промпта (одним шагом, без чанкования). Весь промпт в кэше:
 * считается один последний токен (effectiveCachedTokens), поэтому TTFT не бывает нулевым.
 */
export function prefillBreakdown(model: ModelLike, gpu: GpuLike, tokens: number, opts: PrefillOptions = {}): StepBreakdown {
  const t = Math.max(0, Math.floor(tokens))
  const cached = effectiveCachedTokens(t, opts.cachedTokens)
  return stepTime(model, gpu, { prefill: [{ tokens: t - cached, context: cached }] }, opts)
}

/**
 * Время prefill, мс: max(FLOPs / (gpus·пик·MFU), байты / (gpus·BW·MBU)) + накладные.
 * `tokens` — полная длина промпта; opts.cachedTokens — сколько из них уже в prefix cache.
 * Короткий промпт упирается в чтение весов (memory-bound), длинный в FLOPs
 * (notes-models-attention-bottlenecks.md §3.12: граница около 340 токенов для 8B на H100).
 */
export function prefillMs(model: ModelLike, gpu: GpuLike, tokens: number, opts: PrefillOptions = {}): number {
  return prefillBreakdown(model, gpu, tokens, opts).ms
}

/** Параметры decode-шага. */
export interface DecodeOptions extends PerfOptions {
  /** Сколько последовательностей в batch (по умолчанию 1). */
  batch?: number
  /** Средняя длина последовательности (промпт + сгенерированное), токенов; по умолчанию 0. */
  avgContext?: number
}

/** Разбивка decode-шага: batch последовательностей средней длины avgContext. */
export function decodeStepBreakdown(model: ModelLike, gpu: GpuLike, opts: DecodeOptions = {}): StepBreakdown {
  const batch = Math.max(1, Math.round(opts.batch ?? 1))
  const ctx = Math.max(0, opts.avgContext ?? 0)
  return stepTime(model, gpu, { decodeUniform: { batch, context: ctx } }, opts)
}

/**
 * Время одного decode-шага, мс: max(время памяти, время вычислений) + небольшие накладные.
 * С mbu = 1, mfu = 1, overheadMs = 0 это идеальная граница из заметок: Llama-3.1-8B BF16,
 * H100, batch 1 ≈ 4.79 мс (notes-numbers.md §3.8), S = 2048, batch 8 ≈ 5.44 мс (§3.14).
 */
export function decodeStepMs(model: ModelLike, gpu: GpuLike, opts: DecodeOptions = {}): number {
  return decodeStepBreakdown(model, gpu, opts).ms
}

/**
 * Нижняя граница шага decode при batch 1 и коротком контексте: байты весов / полоса HBM
 * (100 % MBU, без накладных). Llama-3.1-8B BF16 на H100: 4.79 мс → ≤ 209 ток/с.
 */
export function decodeLowerBoundMs(model: ModelLike, gpu: GpuLike, dtype: DType = 'bf16', gpus = 1): number {
  return (stepWeightBytes(model, dtype, 1) / Math.max(1, gpus) / (gpu.bwTBs * 1e12)) * 1000
}

/** Точка кривой «batch → скорость» для roofline-калькулятора decode. */
export interface DecodeSweepPoint {
  batch: number
  stepMs: number
  memoryMs: number
  computeMs: number
  /** Токенов в секунду на одного пользователя (1 / шаг). */
  perUserTokS: number
  /** Токенов в секунду на весь batch. */
  totalTokS: number
  bound: 'memory' | 'compute'
  /** KV всех последовательностей batch, байт. */
  kvBytes: number
  /** Влезает ли KV batch в память (по maxConcurrentSeqs). */
  fits: boolean
}

/**
 * Кривая throughput / latency: время шага и ток/с при разных batch (notes-numbers.md §3.14):
 * пока шаг memory-bound, рост batch почти не удлиняет шаг, и общий throughput растёт почти линейно.
 */
export function decodeBatchSweep(
  model: ModelLike,
  gpu: GpuLike,
  batches: number[],
  opts: DecodeOptions & { memUtil?: number; reserveGiB?: number } = {},
): DecodeSweepPoint[] {
  const ctx = opts.avgContext ?? 0
  const maxSeqs = maxConcurrentSeqs(model, gpu, {
    contextLen: Math.max(1, ctx),
    dtype: opts.dtype,
    kvDtype: opts.kvDtype,
    gpus: opts.gpus,
    memUtil: opts.memUtil,
    reserveGiB: opts.reserveGiB,
    dpAttention: opts.dpAttention,
  })
  return batches.map((batch) => {
    const s = decodeStepBreakdown(model, gpu, { ...opts, batch })
    return {
      batch,
      stepMs: s.ms,
      memoryMs: s.memoryMs,
      computeMs: s.computeMs,
      perUserTokS: 1000 / s.ms,
      totalTokS: (batch * 1000) / s.ms,
      bound: s.bound,
      kvBytes: s.kvBytes,
      fits: batch <= maxSeqs,
    }
  })
}

/* ============================================================================
 * Арифметическая интенсивность и roofline
 * ========================================================================== */

/** Параметры для arithmeticIntensity. */
export interface IntensityOptions {
  phase: 'prefill' | 'decode'
  /** decode: сколько последовательностей в batch (по умолчанию 1). */
  batch?: number
  /** decode: длина последовательности; prefill: сколько токенов уже в KV (по умолчанию 0). */
  avgContext?: number
  /** prefill: сколько токенов считаем (по умолчанию 1024). */
  tokens?: number
  dtype?: DType
  kvDtype?: DType
}

/**
 * Арифметическая интенсивность (FLOP на байт из HBM) целого шага decode или prefill.
 * Decode при batch 1 ≈ 1 FLOP/байт (в ~300 раз ниже ridge H100), prefill тысячи (notes-numbers.md §1.3).
 */
export function arithmeticIntensity(model: ModelLike, opts: IntensityOptions): { flops: number; bytes: number; intensity: number } {
  const work: StepWork =
    opts.phase === 'decode'
      ? { decodeUniform: { batch: Math.max(1, opts.batch ?? 1), context: Math.max(1, opts.avgContext ?? 1) } }
      : { prefill: [{ tokens: Math.max(1, opts.tokens ?? 1024), context: Math.max(0, opts.avgContext ?? 0) }] }
  // GPU не влияет на FLOPs и байты; берём фиктивный с единичными пиками.
  const dummy: GpuSpec = { id: 'dummy', name: 'dummy', memGB: 1, bwTBs: 1, tflops: { bf16: 1, fp8: 1, fp4: 1 }, source: '' }
  const s = stepTime(model, dummy, work, { dtype: opts.dtype, kvDtype: opts.kvDtype, overheadMs: 0 })
  return { flops: s.flops, bytes: s.bytes, intensity: s.intensity }
}

/**
 * Интенсивность линейного слоя d_in × d_out при T токенах за проход:
 * 2·T·d_in·d_out / (b_w·d_in·d_out + b_a·T·(d_in + d_out)) ≈ T в BF16 (notes-numbers.md §3.10:
 * d = 4096, T = 64 → 62.1).
 */
export function linearLayerIntensity(tokens: number, dIn: number, dOut: number, bytesW = 2, bytesA = 2): number {
  const flops = 2 * tokens * dIn * dOut
  const bytes = bytesW * dIn * dOut + bytesA * tokens * (dIn + dOut)
  return flops / bytes
}

/**
 * Интенсивность attention в decode (FLOP на байт прочитанного KV), не зависит от batch:
 * GQA ≈ heads / kv_heads в BF16 (Llama-3.1-8B: 4), MLA DeepSeek-V3 ≈ 242 (notes-numbers.md §1.4, §3.13).
 */
export function decodeAttentionIntensity(model: ModelLike, kvDtype: DType = 'bf16'): number {
  return attentionFlopsPerPair(model, 'decode') / kvLayout(model, kvDtype).bytesPerLayerPerToken
}

/** Ridge point: пиковые dense TFLOPS / ТБ/с = FLOP на байт. H100 BF16: 989 / 3.35 ≈ 295 (книга, с. 62). */
export function ridgePoint(gpu: GpuSpec, dtype: DType = 'bf16'): number {
  return peakTflops(gpu, dtype) / gpu.bwTBs
}

/** Достижимые TFLOPS по roofline: min(пик, интенсивность × полоса). */
export function attainableTflops(gpu: GpuSpec, dtype: DType, intensity: number): number {
  return Math.min(peakTflops(gpu, dtype), Math.max(0, intensity) * gpu.bwTBs)
}

/* ============================================================================
 * Передача данных
 * ========================================================================== */

/**
 * Время передачи, мс: байты / (ГБ/с в одну сторону) + латентность. 8k токенов KV Llama-3.1-70B
 * (2.68 ГБ) по одному NDR 400G (50 ГБ/с) ≈ 53.7 мс (notes-numbers.md §3.16).
 */
export function transferMs(bytes: number, GBs: number, latencyMs = 0): number {
  if (!(GBs > 0)) return Infinity
  return latencyMs + (Math.max(0, bytes) / (GBs * 1e9)) * 1000
}

/** Сравнение «загрузить KV с уровня кэша или пересчитать prefill». */
export interface LoadVsRecompute {
  kvBytes: number
  loadMs: number
  recomputeMs: number
  /** Полоса, при которой загрузка и пересчёт равны, ГБ/с (BW* из notes-kvcache-tiers.md §2.1 F8). */
  breakEvenGBs: number
  loadIsFaster: boolean
}

/**
 * Загрузить ли KV `tokens` токенов по каналу linkGBs или пересчитать prefill
 * (notes-kvcache-tiers.md §1.3: T_load = KV / BW против T_recompute = FLOPs / (N·пик·MFU)).
 */
export function loadVsRecompute(
  model: ModelLike,
  gpu: GpuLike,
  tokens: number,
  linkGBs: number,
  opts: PerfOptions & { latencyMs?: number } = {},
): LoadVsRecompute {
  const kvBytes = kvBytesForSeq(model, tokens, opts.kvDtype ?? 'bf16', { swaAware: opts.swaAware !== false })
  const loadMs = transferMs(kvBytes, linkGBs, opts.latencyMs ?? 0)
  const recomputeMs = prefillMs(model, gpu, tokens, { ...opts, overheadMs: 0 })
  const breakEvenGBs = recomputeMs > 0 ? kvBytes / 1e9 / (recomputeMs / 1000) : Infinity
  return { kvBytes, loadMs, recomputeMs, breakEvenGBs, loadIsFaster: loadMs < recomputeMs }
}

/* ============================================================================
 * Память: бюджет KV, сколько последовательностей влезает
 * ========================================================================== */

/** Параметры бюджета памяти. */
export interface MemoryOptions {
  dtype?: DType
  kvDtype?: DType
  /** Доля памяти под процесс (по умолчанию DEFAULT_MEM_UTIL = 0.9). */
  memUtil?: number
  gpus?: number
  /** Резерв на активации и CUDA graphs, ГиБ на GPU (по умолчанию 2). */
  reserveGiB?: number
  dpAttention?: boolean
  swaAware?: boolean
}

/**
 * Бюджет под KV, байт: gpus × (память × memUtil − резерв) − веса (может быть отрицательным).
 * Llama-3.1-8B BF16 на H100: 79.65 × 0.9 − 2 − 14.96 ≈ 54.7 ГиБ (notes-numbers.md §3.15).
 */
export function kvBudgetBytes(model: ModelLike, gpu: GpuLike, opts: MemoryOptions = {}): number {
  const gpus = Math.max(1, opts.gpus ?? 1)
  const util = opts.memUtil ?? DEFAULT_MEM_UTIL
  const reserve = (opts.reserveGiB ?? DEFAULT_RESERVE_GIB) * GiB
  return gpus * (usableMemBytes(gpu) * util - reserve) - weightBytes(model, opts.dtype ?? 'bf16')
}

/** Байт KV на одну последовательность с учётом дублирования при TP. */
function kvPerSeqWithReplication(model: ModelLike, contextLen: number, opts: MemoryOptions): number {
  const per = kvBytesForSeq(model, contextLen, opts.kvDtype ?? 'bf16', { swaAware: opts.swaAware !== false })
  return per * tpKvReplication(model, Math.max(1, opts.gpus ?? 1), opts.dpAttention)
}

/**
 * Сколько последовательностей длиной contextLen одновременно влезает в KV.
 * Llama-3.1-8B BF16, 1 × H100, 8192 токена: 54 (с FP8 KV 109), notes-numbers.md §3.15.
 */
export function maxConcurrentSeqs(model: ModelLike, gpu: GpuLike, opts: MemoryOptions & { contextLen: number }): number {
  const budget = kvBudgetBytes(model, gpu, opts)
  if (budget <= 0) return 0
  const per = kvPerSeqWithReplication(model, Math.max(1, opts.contextLen), opts)
  return per > 0 ? Math.floor(budget / per) : 0
}

/** Сколько токенов KV влезает в бюджет (для ёмкости кэша в симуляциях). */
export function kvCapacityTokens(model: ModelLike, gpu: GpuLike, opts: MemoryOptions = {}): number {
  const budget = kvBudgetBytes(model, gpu, opts)
  if (budget <= 0) return 0
  const perToken = kvBytesPerToken(model, opts.kvDtype ?? 'bf16') * tpKvReplication(model, Math.max(1, opts.gpus ?? 1), opts.dpAttention)
  return Math.floor(budget / perToken)
}

/** Результат проверки «влезет ли». Все объёмы в байтах, суммарно по всем GPU. */
export interface FitsResult {
  /** Влезают веса и KV для заданного числа последовательностей. */
  fits: boolean
  weightsFit: boolean
  /** Вся память GPU (nvidia-smi). */
  totalBytes: number
  /** Доступно процессу движка: total × memUtil. */
  processBytes: number
  weightBytes: number
  /** Резерв на активации и CUDA graphs. */
  reserveBytes: number
  /** Под KV: process − reserve − weights (может быть отрицательным). */
  kvBudgetBytes: number
  kvPerSeqBytes: number
  kvNeededBytes: number
  /** kvBudget − kvNeeded (отрицательное значит переполнение). */
  freeBytes: number
  maxSeqs: number
  /** По-русски: вывод одной-двумя фразами. */
  message: string
}

const NBSP = '\u00a0'
const RU_NF = new Map<number, Intl.NumberFormat>()
/** Число по-русски (DESIGN §7): разряды неразрывным пробелом, десятичная запятая. */
function ruNum(x: number, maxDigits = 0): string {
  let f = RU_NF.get(maxDigits)
  if (!f) RU_NF.set(maxDigits, (f = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: maxDigits })))
  return f.format(x).replace('-', '−')
}

/** Форма слова после числа: forms = [для 1, для 2–4, для 5–20]. */
function ruPlural(n: number, forms: readonly [string, string, string]): string {
  const a = Math.abs(Math.round(n)) % 100
  const b = a % 10
  if (a > 10 && a < 20) return forms[2]
  if (b === 1) return forms[0]
  if (b >= 2 && b <= 4) return forms[1]
  return forms[2]
}

/** «для 1 последовательности», «до 54 последовательностей». */
const SEQS_GEN = ['последовательности', 'последовательностей', 'последовательностей'] as const
/** «по 1 токену», «по 8 192 токена», «по 4 096 токенов». */
const TOKENS_PO = ['токену', 'токена', 'токенов'] as const

const SI_BYTES = ['Б', 'КБ', 'МБ', 'ГБ', 'ТБ', 'ПБ'] as const

/** Байты в единицах SI (ГБ = 10⁹) до трёх значащих цифр, как fmt.bytes в ui: «604 МБ», «16,4 ГБ», «4,72 ТБ». */
function fmtGB(bytes: number): string {
  const sign = bytes < 0 ? '−' : ''
  let v = Math.abs(bytes)
  let u = 0
  while (u < SI_BYTES.length - 1 && Number(v.toPrecision(3)) >= 1000) {
    v /= 1000
    u++
  }
  const digits = u === 0 || v >= 100 ? 0 : v >= 10 ? 1 : 2
  return `${sign}${ruNum(v, digits)}${NBSP}${SI_BYTES[u]}`
}

/**
 * Влезет ли модель и KV для `seqs` последовательностей по `contextLen` токенов на gpus × GPU:
 * полоса памяти «веса / резерв / KV / свободно» и вывод по-русски (SPEC §6.11, §6.17).
 */
export function fitsCheck(
  model: ModelLike,
  gpu: GpuLike,
  opts: MemoryOptions & { contextLen: number; seqs?: number },
): FitsResult {
  const gpus = Math.max(1, opts.gpus ?? 1)
  const seqs = Math.max(0, opts.seqs ?? 1)
  const total = gpus * usableMemBytes(gpu)
  const processBytes = total * (opts.memUtil ?? DEFAULT_MEM_UTIL)
  const reserveBytes = gpus * (opts.reserveGiB ?? DEFAULT_RESERVE_GIB) * GiB
  const wBytes = weightBytes(model, opts.dtype ?? 'bf16')
  const budget = processBytes - reserveBytes - wBytes
  const perSeq = kvPerSeqWithReplication(model, Math.max(1, opts.contextLen), opts)
  const needed = perSeq * seqs
  const weightsFit = budget > 0
  const maxSeqs = weightsFit && perSeq > 0 ? Math.floor(budget / perSeq) : 0
  const fits = weightsFit && needed <= budget
  const gpuWord = gpus === 1 ? `1 × ${gpu.name}` : `${gpus} × ${gpu.name}`
  let message: string
  if (!weightsFit) {
    message =
      `Веса не помещаются: ${fmtGB(wBytes)} весов при ${fmtGB(processBytes - reserveBytes)} доступной памяти на ${gpuWord}. ` +
      'Нужно больше GPU или квантизация.'
  } else if (!fits) {
    message =
      `Веса помещаются, но KV для ${ruNum(seqs)}${NBSP}${ruPlural(seqs, SEQS_GEN)} по ${ruNum(opts.contextLen)}${NBSP}${ruPlural(opts.contextLen, TOKENS_PO)} ` +
      `не влезает: нужно ${fmtGB(needed)}, под KV есть ${fmtGB(budget)}. Влезет максимум ${ruNum(maxSeqs)}.`
  } else {
    message =
      `Влезает: веса ${fmtGB(wBytes)}, KV ${fmtGB(needed)} на ${ruNum(seqs)}${NBSP}посл., свободно ${fmtGB(budget - needed)}. ` +
      `Всего влезет до ${ruNum(maxSeqs)}${NBSP}${ruPlural(maxSeqs, SEQS_GEN)} такой длины.`
  }
  return {
    fits,
    weightsFit,
    totalBytes: total,
    processBytes,
    weightBytes: wBytes,
    reserveBytes,
    kvBudgetBytes: budget,
    kvPerSeqBytes: perSeq,
    kvNeededBytes: needed,
    freeBytes: budget - needed,
    maxSeqs,
    message,
  }
}
