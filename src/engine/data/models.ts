/**
 * Модели для калькуляторов и симуляций. Перенесено из research/numbers.json (ключ `models`,
 * поля config.json с Hugging Face, параметры посчитаны research/scripts/numbers_calc.py)
 * и research/notes-numbers.md §2.4–2.5, notes-models-attention-bottlenecks.md §2.4.
 *
 * Как читать поля:
 * - paramsB: все параметры (10^9), так модель лежит в памяти;
 * - activeParamsB: как в карточке модели (Qwen/DeepSeek считают с таблицей эмбеддингов,
 *   OpenAI для gpt-oss без неё); по нему считаем байты, которые decode читает за шаг;
 * - activeMatmulParamsB: что реально умножается на каждый токен (без таблицы эмбеддингов),
 *   для FLOPs = 2·N·tokens (notes-numbers.md §1.13);
 * - expertParamsB: сумма матриц всех routed-экспертов (для MoE), посчитано из config:
 *   3 × hidden × moe_intermediate × experts × MoE-слоёв.
 */
import type { DType, ModelSpec } from '../types'

/** Семейство модели (для подписей и шаблонов). */
export type ModelFamilyId = 'qwen3' | 'llama3' | 'deepseek-v3' | 'gpt-oss' | 'mistral' | 'gemma3' | 'kimi-k2'

/** ModelSpec + необязательные поля, которых нет в общем контракте. */
export interface ModelSpecExt extends ModelSpec {
  family: ModelFamilyId
  /** Репозиторий на Hugging Face. */
  hfId: string
  /** intermediate_size плотного MLP (у MoE: размер dense-слоёв, если есть). */
  intermediate: number
  /** Параметры, которые умножаются на каждый токен (без таблицы эмбеддингов), B. Для FLOPs. */
  activeMatmulParamsB: number
  /** LM head (vocab × hidden), B: логиты в prefill нужны только для последнего токена. */
  lmHeadParamsB: number
  /** MoE: сумма матриц всех routed-экспертов, B. */
  expertParamsB?: number
  /** MoE: intermediate одного эксперта. */
  moeIntermediate?: number
  /** MoE: shared-эксперты (активны всегда). */
  sharedExperts?: number
  /** Размерность V на голову, если отличается от headDim (MLA DeepSeek: qk 192, v 128). */
  vHeadDim?: number
  /** MLA: kv_lora_rank (512) и qk_rope_head_dim (64); mlaDim = их сумма. */
  kvLoraRank?: number
  qkRopeHeadDim?: number
  /**
   * Нативный формат весов, если он смешанный (gpt-oss: эксперты MXFP4 4.25 бит, остальное BF16).
   * weightBytes(model, native.dtype) использует эти байты вместо «все параметры в одном формате».
   */
  native?: { dtype: DType; label: string; expertBytesPerParam: number; otherBytesPerParam: number }
  /** Нативный контекст без RoPE-scaling, если отличается от maxContext. */
  nativeContext?: number
  /** Основная модель для селекторов сайта. */
  featured?: boolean
  /** Короткая заметка по-русски. */
  notes?: string
}

/** Все модели: сначала обязательные для сайта, затем дополнительные для сравнений. */
export const MODELS: ModelSpecExt[] = [
  {
    id: 'qwen3-8b',
    name: 'Qwen3-8B',
    family: 'qwen3',
    hfId: 'Qwen/Qwen3-8B',
    layers: 36,
    hidden: 4096,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    intermediate: 12288,
    vocab: 151936, // ширина логитов (строк эмбеддингов); токенизатор 151 669 (SPEC §8.7)
    paramsB: 8.191,
    activeParamsB: 8.191,
    activeMatmulParamsB: 7.568,
    lmHeadParamsB: 0.6223, // 151936 × 4096
    attention: 'gqa', // GQA 32 Q / 8 KV + QK-RMSNorm
    maxContext: 40960, // max_position_embeddings в config
    nativeContext: 32768, // карточка: 32,768 нативно, 131,072 с YaRN
    featured: true,
    notes: 'Плотная модель: 36 слоёв, GQA 32/8, KV 144 КиБ на токен в BF16.',
    source: 'https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json ; research/numbers.json models[qwen3-8b]',
  },
  {
    id: 'qwen3-32b',
    name: 'Qwen3-32B',
    family: 'qwen3',
    hfId: 'Qwen/Qwen3-32B',
    layers: 64,
    hidden: 5120,
    heads: 64,
    kvHeads: 8,
    headDim: 128, // num_heads × head_dim = 8192 > hidden 5120
    intermediate: 25600,
    vocab: 151936,
    paramsB: 32.762,
    activeParamsB: 32.762,
    activeMatmulParamsB: 31.984,
    lmHeadParamsB: 0.7779, // 151936 × 5120
    attention: 'gqa',
    maxContext: 40960,
    nativeContext: 32768,
    featured: true,
    notes: 'Плотная модель: 64 слоя, KV 256 КиБ на токен в BF16.',
    source: 'https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json ; research/numbers.json models[qwen3-32b]',
  },
  {
    id: 'qwen3-30b-a3b',
    name: 'Qwen3-30B-A3B',
    family: 'qwen3',
    hfId: 'Qwen/Qwen3-30B-A3B',
    layers: 48,
    hidden: 2048,
    heads: 32,
    kvHeads: 4,
    headDim: 128,
    intermediate: 6144,
    vocab: 151936,
    paramsB: 30.532,
    activeParamsB: 3.353,
    activeMatmulParamsB: 3.042,
    lmHeadParamsB: 0.3112, // 151936 × 2048
    expertParamsB: 28.991, // 3 × 2048 × 768 × 128 экспертов × 48 слоёв
    moeIntermediate: 768,
    sharedExperts: 0,
    experts: { total: 128, active: 8 },
    attention: 'gqa',
    maxContext: 40960,
    nativeContext: 32768,
    notes: 'MoE: 128 экспертов, 8 активных на токен. В памяти 30,5B параметров, на токен работает 3,4B.',
    source: 'https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json ; research/numbers.json models[qwen3-30b-a3b]',
  },
  {
    id: 'qwen3-235b-a22b',
    name: 'Qwen3-235B-A22B',
    family: 'qwen3',
    hfId: 'Qwen/Qwen3-235B-A22B',
    layers: 94,
    hidden: 4096,
    heads: 64,
    kvHeads: 4,
    headDim: 128,
    intermediate: 12288,
    vocab: 151936,
    paramsB: 235.094,
    activeParamsB: 22.191,
    activeMatmulParamsB: 21.568,
    lmHeadParamsB: 0.6223,
    expertParamsB: 227.096, // 3 × 4096 × 1536 × 128 × 94 (notes-models §3.15: 227,1B)
    moeIntermediate: 1536,
    sharedExperts: 0,
    experts: { total: 128, active: 8 }, // книга с. 53–54
    attention: 'gqa',
    maxContext: 40960,
    nativeContext: 32768,
    notes: 'MoE: 128 экспертов, top-8, 94 слоя; 22B активных из 235B (книга, с. 53–54).',
    source:
      'https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json ; книга с. 53–54 ; research/numbers.json models[qwen3-235b-a22b]',
  },
  {
    id: 'llama-3.1-8b',
    name: 'Llama-3.1-8B',
    family: 'llama3',
    hfId: 'meta-llama/Llama-3.1-8B-Instruct',
    layers: 32,
    hidden: 4096,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    intermediate: 14336,
    vocab: 128256,
    paramsB: 8.030261248, // ровно 8,030,261,248 (notes-numbers.md §3.3)
    activeParamsB: 8.030261248,
    activeMatmulParamsB: 7.505,
    lmHeadParamsB: 0.5253, // 128256 × 4096
    attention: 'gqa',
    maxContext: 131072,
    notes: 'Плотная модель: 32 слоя, GQA 32/8, KV ровно 128 КиБ на токен (1 ГиБ на 8k).',
    source:
      'https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/blob/main/config.json (зеркало gated meta-llama) ; research/numbers.json models[llama-3.1-8b]',
  },
  {
    id: 'llama-3.1-70b',
    name: 'Llama-3.1-70B',
    family: 'llama3',
    hfId: 'meta-llama/Llama-3.1-70B-Instruct',
    layers: 80,
    hidden: 8192,
    heads: 64,
    kvHeads: 8,
    headDim: 128,
    intermediate: 28672,
    vocab: 128256,
    paramsB: 70.554,
    activeParamsB: 70.554,
    activeMatmulParamsB: 69.502,
    lmHeadParamsB: 1.0507, // 128256 × 8192
    attention: 'gqa',
    maxContext: 131072,
    featured: true,
    notes: 'Плотная модель: 80 слоёв, KV 320 КиБ на токен; в BF16 не влезает в один GPU на 80 ГБ.',
    source:
      'https://huggingface.co/unsloth/Meta-Llama-3.1-70B-Instruct/blob/main/config.json (зеркало meta-llama) ; research/numbers.json models[llama-3.1-70b]',
  },
  {
    id: 'deepseek-v3',
    name: 'DeepSeek-V3',
    family: 'deepseek-v3',
    hfId: 'deepseek-ai/DeepSeek-V3',
    layers: 61,
    hidden: 7168,
    heads: 128,
    kvHeads: 1, // условно: в кэше один общий latent-вектор на слой (config пишет 128, для KV не использовать)
    headDim: 192, // qk: 128 nope + 64 rope
    vHeadDim: 128,
    intermediate: 18432,
    vocab: 129280,
    paramsB: 671.026, // + ~14B MTP-модуль = 685B на HF
    activeParamsB: 37.552,
    activeMatmulParamsB: 36.625,
    lmHeadParamsB: 0.9267, // 129280 × 7168
    expertParamsB: 653.909, // 3 × 7168 × 2048 × 256 × 58 MoE-слоёв (первые 3 слоя dense)
    moeIntermediate: 2048,
    sharedExperts: 1,
    experts: { total: 256, active: 8 },
    attention: 'mla',
    mlaDim: 576, // kv_lora_rank 512 + qk_rope_head_dim 64
    kvLoraRank: 512,
    qkRopeHeadDim: 64,
    maxContext: 163840,
    notes:
      'MLA: в KV на слой лежит один сжатый вектор из 576 чисел (512 + 64), без отдельных K и V: 68,6 КиБ на токен при модели в 671B.',
    source:
      'https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json (R1 и V3.1: те же поля) ; research/numbers.json models[deepseek-v3]',
  },
  {
    id: 'gpt-oss-120b',
    name: 'gpt-oss-120b',
    family: 'gpt-oss',
    hfId: 'openai/gpt-oss-120b',
    layers: 36,
    hidden: 2880,
    heads: 64,
    kvHeads: 8,
    headDim: 64,
    intermediate: 2880,
    vocab: 201088,
    paramsB: 116.829,
    activeParamsB: 5.133, // по определению OpenAI: без таблицы эмбеддингов (с ней 5.712)
    activeMatmulParamsB: 5.131,
    lmHeadParamsB: 0.5791, // 201088 × 2880
    expertParamsB: 114.662, // 3 × 2880² × 128 × 36 (только матрицы)
    moeIntermediate: 2880,
    sharedExperts: 0,
    experts: { total: 128, active: 4 },
    attention: 'gqa-swa',
    slidingWindow: { window: 128, fullLayers: 18 }, // слои чередуются: 18 sliding (окно 128) + 18 full
    native: {
      dtype: 'fp4',
      label: 'MXFP4 (эксперты, 4,25 бит) + BF16 (остальное)',
      expertBytesPerParam: 0.53125, // E2M1 + E8M0 scale на 32 значения = 4.25 бит (arXiv 2508.10925)
      otherBytesPerParam: 2,
    },
    maxContext: 131072,
    notes:
      'MoE 128 экспертов, top-4; половина слоёв со скользящим окном 128 токенов, поэтому KV растёт вдвое медленнее; веса экспертов в MXFP4: чекпоинт ≈ 60,8 ГиБ.',
    source:
      'https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json ; arXiv 2508.10925 Table 1 ; research/numbers.json models[gpt-oss-120b]',
  },
  // ---- дополнительные модели (не обязательны для сайта, полезны для сравнений) ----
  {
    id: 'llama-3.1-405b',
    name: 'Llama-3.1-405B',
    family: 'llama3',
    hfId: 'meta-llama/Llama-3.1-405B-Instruct',
    layers: 126,
    hidden: 16384,
    heads: 128,
    kvHeads: 8,
    headDim: 128,
    intermediate: 53248,
    vocab: 128256,
    paramsB: 405.853,
    activeParamsB: 405.853,
    activeMatmulParamsB: 403.748,
    lmHeadParamsB: 2.1013,
    attention: 'gqa',
    maxContext: 131072,
    notes: 'Плотная модель на 405B: только на нескольких GPU (8 × H100 в FP8).',
    source:
      'https://huggingface.co/unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit/blob/main/config.json (архитектурные поля = meta-llama) ; research/numbers.json models[llama-3.1-405b]',
  },
  {
    id: 'gpt-oss-20b',
    name: 'gpt-oss-20b',
    family: 'gpt-oss',
    hfId: 'openai/gpt-oss-20b',
    layers: 24,
    hidden: 2880,
    heads: 64,
    kvHeads: 8,
    headDim: 64,
    intermediate: 2880,
    vocab: 201088,
    paramsB: 20.915,
    activeParamsB: 3.608, // OpenAI: без таблицы эмбеддингов (с ней 4.187)
    activeMatmulParamsB: 3.607,
    lmHeadParamsB: 0.5791,
    expertParamsB: 19.11, // 3 × 2880² × 32 × 24
    moeIntermediate: 2880,
    sharedExperts: 0,
    experts: { total: 32, active: 4 },
    attention: 'gqa-swa',
    slidingWindow: { window: 128, fullLayers: 12 },
    native: {
      dtype: 'fp4',
      label: 'MXFP4 (эксперты, 4,25 бит) + BF16 (остальное)',
      expertBytesPerParam: 0.53125,
      otherBytesPerParam: 2,
    },
    maxContext: 131072,
    notes: 'Маленькая MoE: 32 эксперта, top-4; чекпоинт ≈ 12,8 ГиБ, влезает в потребительскую карту.',
    source:
      'https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json ; arXiv 2508.10925 Table 1 ; research/numbers.json models[gpt-oss-20b]',
  },
  {
    id: 'kimi-k2',
    name: 'Kimi-K2-Instruct',
    family: 'kimi-k2',
    hfId: 'moonshotai/Kimi-K2-Instruct',
    layers: 61,
    hidden: 7168,
    heads: 64,
    kvHeads: 1,
    headDim: 192,
    vHeadDim: 128,
    intermediate: 18432,
    vocab: 163840,
    paramsB: 1026.408,
    activeParamsB: 32.862,
    activeMatmulParamsB: 31.686,
    lmHeadParamsB: 1.1744,
    expertParamsB: 1014.686, // 3 × 7168 × 2048 × 384 × 60 MoE-слоёв
    moeIntermediate: 2048,
    sharedExperts: 1,
    experts: { total: 384, active: 8 },
    attention: 'mla',
    mlaDim: 576,
    kvLoraRank: 512,
    qkRopeHeadDim: 64,
    maxContext: 131072,
    notes: 'MLA как у DeepSeek-V3, 384 эксперта: триллион параметров, 33B активных.',
    source: 'https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json ; research/numbers.json models[kimi-k2]',
  },
  {
    id: 'mistral-small-24b',
    name: 'Mistral-Small-24B',
    family: 'mistral',
    hfId: 'mistralai/Mistral-Small-24B-Instruct-2501',
    layers: 40,
    hidden: 5120,
    heads: 32,
    kvHeads: 8,
    headDim: 128,
    intermediate: 32768,
    vocab: 131072,
    paramsB: 23.572,
    activeParamsB: 23.572,
    activeMatmulParamsB: 22.901,
    lmHeadParamsB: 0.6711,
    attention: 'gqa',
    maxContext: 32768,
    notes: 'Плотная модель среднего размера, полный causal attention.',
    source:
      'https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501/blob/main/config.json ; research/numbers.json models[mistral-small-24b]',
  },
  {
    id: 'gemma-3-27b',
    name: 'Gemma-3-27B',
    family: 'gemma3',
    hfId: 'google/gemma-3-27b-it',
    layers: 62,
    hidden: 5376,
    heads: 32,
    kvHeads: 16,
    headDim: 128,
    intermediate: 21504,
    vocab: 262208,
    paramsB: 27.009, // текстовая часть (+0.4B vision)
    activeParamsB: 27.009,
    activeMatmulParamsB: 27.008,
    lmHeadParamsB: 1.4096, // tied с эмбеддингами
    attention: 'gqa-swa',
    slidingWindow: { window: 1024, fullLayers: 10 }, // 52 локальных слоя (окно 1024) + 10 глобальных
    maxContext: 131072,
    notes: '5 локальных слоёв (окно 1024) на 1 глобальный: на 8k контекста KV в 3,8 раза меньше наивного.',
    source:
      'https://huggingface.co/unsloth/gemma-3-27b-it/blob/main/config.json (зеркало gated google/gemma-3-27b-it) ; research/numbers.json models[gemma-3-27b]',
  },
]

/** Модели для основных селекторов симулятора (SPEC §6.0: Qwen3-8B / Qwen3-32B / Llama-3.1-70B). */
export const FEATURED_MODEL_IDS: readonly string[] = ['qwen3-8b', 'qwen3-32b', 'llama-3.1-70b']

const MODEL_INDEX: Map<string, ModelSpecExt> = new Map(MODELS.map((m) => [m.id, m]))

/** Модель по id; бросает ошибку, если такой нет. */
export function getModel(id: string): ModelSpecExt {
  const m = MODEL_INDEX.get(id)
  if (!m) throw new Error(`getModel: неизвестная модель «${id}»`)
  return m
}

/** То же, что getModel, но возвращает undefined вместо ошибки. */
export function findModel(id: string): ModelSpecExt | undefined {
  return MODEL_INDEX.get(id)
}

/** Сколько слоёв со скользящим окном (0 у моделей без sliding window). */
export function slidingLayers(model: ModelSpec): number {
  return model.slidingWindow ? model.layers - model.slidingWindow.fullLayers : 0
}
