/**
 * Параметры и FLOPs по частям блока трансформера, выведенные из полей модели в src/engine/data/models.ts
 * (hidden, heads, kvHeads, headDim, intermediate, moeIntermediate, experts, layers, vocab).
 *
 * Чего нет в данных и что выведено здесь (сверено тестом arch.test.ts: сумма частей = paramsB модели):
 * - QK-norm у Qwen3 (RMSNorm по head_dim для Q и K в каждом слое): включаем для семейства qwen3;
 * - LM head отдельной матрицей (tie_word_embeddings = false у всех моделей списка): lmHeadParamsB > 0;
 * - у MoE-моделей Qwen3 все слои MoE, shared-экспертов нет (sharedExperts = 0 в данных).
 * FLOPs: 2 FLOP на параметр матрицы на токен (умножение и сложение); attention по контексту через
 * attentionFlopsPerPair из perf.ts (2·heads·(d_qk + d_v) на пару «запрос, ключ» на слой).
 */
import { getModel, type ModelSpecExt } from '@/engine/data/models'
import { attentionFlopsPerPair } from '@/engine/perf'

/** Модели со «стандартным» блоком (GQA + SwiGLU или MoE), для которых разбивка точная. */
export const ARCH_MODEL_IDS = ['qwen3-8b', 'qwen3-32b', 'llama-3.1-8b', 'llama-3.1-70b', 'qwen3-30b-a3b', 'qwen3-235b-a22b'] as const
export type ArchModelId = (typeof ARCH_MODEL_IDS)[number]

export function isArchModel(id: string): id is ArchModelId {
  return (ARCH_MODEL_IDS as readonly string[]).includes(id)
}

export interface WeightMatrix {
  name: string
  rows: number
  cols: number
  /** Сколько таких матриц (эксперты). */
  count?: number
  note?: string
}

export interface Part {
  /** Параметры части (для частей блока: на один слой). */
  params: number
  /** Параметры, которые работают на один токен (для MoE меньше params). */
  activeParams: number
  /** FLOPs на токен без зависимости от контекста. */
  flopsFixed: number
  /** FLOPs на токен на каждый токен контекста (только attention). */
  flopsPerCtx: number
  matrices: WeightMatrix[]
}

export interface Arch {
  model: ModelSpecExt
  qkNorm: boolean
  moe: boolean
  norm: Part
  attn: Part
  mlp: Part
  add: Part
  layerParams: number
  layerActiveParams: number
  embed: Part
  finalNorm: Part
  head: Part
  totalParams: number
  totalActiveParams: number
}

const part = (p: Partial<Part> & { params: number }): Part => ({
  activeParams: p.params,
  flopsFixed: 0,
  flopsPerCtx: 0,
  matrices: [],
  ...p,
})

export function computeArch(model: ModelSpecExt): Arch {
  const d = model.hidden
  const H = model.heads
  const Hkv = model.kvHeads
  const dh = model.headDim
  const qkNorm = model.family === 'qwen3'
  const moe = Boolean(model.experts && model.moeIntermediate)

  // RMSNorm: вектор весов g длины d; операций порядка 4·d (квадраты, сумма, деление, умножение на g)
  const norm = part({ params: d, flopsFixed: 4 * d, matrices: [{ name: 'g', rows: 1, cols: d, note: 'вектор весов' }] })

  const qP = d * H * dh
  const kP = d * Hkv * dh
  const vP = d * Hkv * dh
  const oP = H * dh * d
  const attnMatmul = qP + kP + vP + oP
  const attnMatrices: WeightMatrix[] = [
    { name: 'W_Q', rows: d, cols: H * dh },
    { name: 'W_K', rows: d, cols: Hkv * dh },
    { name: 'W_V', rows: d, cols: Hkv * dh },
    { name: 'W_O', rows: H * dh, cols: d },
  ]
  if (qkNorm) attnMatrices.push({ name: 'q_norm, k_norm', rows: 2, cols: dh, note: 'RMSNorm для Q и K' })
  const attn = part({
    params: attnMatmul + (qkNorm ? 2 * dh : 0),
    flopsFixed: 2 * attnMatmul,
    flopsPerCtx: attentionFlopsPerPair(model, 'decode'),
    matrices: attnMatrices,
  })

  let mlp: Part
  if (moe && model.experts && model.moeIntermediate) {
    const E = model.experts.total
    const k = model.experts.active
    const ff = model.moeIntermediate
    const router = d * E
    const expert = 3 * d * ff
    mlp = part({
      params: router + E * expert,
      activeParams: router + k * expert,
      flopsFixed: 2 * (router + k * expert),
      matrices: [
        { name: 'W_router', rows: d, cols: E },
        { name: 'W_gate, W_up', rows: d, cols: ff, count: 2 * E, note: `по 2 на каждого из ${E} экспертов` },
        { name: 'W_down', rows: ff, cols: d, count: E, note: `по 1 на эксперта` },
      ],
    })
  } else {
    const ff = model.intermediate
    const p = 3 * d * ff
    mlp = part({
      params: p,
      flopsFixed: 2 * p,
      matrices: [
        { name: 'W_gate', rows: d, cols: ff },
        { name: 'W_up', rows: d, cols: ff },
        { name: 'W_down', rows: ff, cols: d },
      ],
    })
  }

  const add = part({ params: 0, flopsFixed: d })
  const layerParams = 2 * norm.params + attn.params + mlp.params
  const layerActiveParams = 2 * norm.params + attn.activeParams + mlp.activeParams

  const embed = part({ params: model.vocab * d, flopsFixed: 0, matrices: [{ name: 'E', rows: model.vocab, cols: d, note: 'таблица эмбеддингов' }] })
  const finalNorm = part({ params: d, flopsFixed: 4 * d })
  const head = part({
    params: d * model.vocab,
    flopsFixed: 2 * d * model.vocab,
    matrices: [{ name: 'W_lm_head', rows: d, cols: model.vocab }],
  })

  const totalParams = model.layers * layerParams + embed.params + finalNorm.params + head.params
  const totalActiveParams = model.layers * layerActiveParams + embed.params + finalNorm.params + head.params
  return { model, qkNorm, moe, norm, attn, mlp, add, layerParams, layerActiveParams, embed, finalNorm, head, totalParams, totalActiveParams }
}

export function archById(id: string): Arch {
  return computeArch(getModel(isArchModel(id) ? id : 'qwen3-8b'))
}

/** FLOPs части на один токен при контексте ctx (токенов, на которые смотрит attention). */
export function partFlops(p: Part, ctx: number): number {
  return p.flopsFixed + p.flopsPerCtx * Math.max(0, ctx)
}

/** FLOPs одного блока на токен при контексте ctx. */
export function layerFlops(a: Arch, ctx: number): number {
  return 2 * partFlops(a.norm, ctx) + partFlops(a.attn, ctx) + partFlops(a.mlp, ctx) + 2 * partFlops(a.add, ctx)
}

/** FLOPs всей модели на один токен decode при контексте ctx. */
export function modelFlops(a: Arch, ctx: number): number {
  return a.model.layers * layerFlops(a, ctx) + partFlops(a.finalNorm, ctx) + partFlops(a.head, ctx)
}

/**
 * Длина контекста, после которой FLOPs attention по контексту (Q·Kᵀ и взвешивание V) на токен
 * превышают все умножения на матрицы весов (2 × активные параметры без таблицы эмбеддингов).
 */
export function attentionCrossover(a: Arch): number {
  const matmul = a.model.layers * (a.attn.flopsFixed + a.mlp.flopsFixed) + a.head.flopsFixed
  return matmul / (a.model.layers * a.attn.flopsPerCtx)
}
