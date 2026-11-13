/**
 * Упрощённая модель параллелизма для главы 17: сколько весов и KV лежит на каждом GPU,
 * сколько стоит шаг decode и связь между GPU при TP, PP, EP, DP и гибриде TP8 × PP2.
 *
 * Основа: roofline как perf.stepTime (max(байты / (BW·MBU), FLOPs / (пик·MFU)) + накладные). Связь:
 * у каждого обмена условная задержка запуска (LAUNCH_MS: 10 мкс по NVLink, 30 мкс по InfiniBand)
 * плюс время по полосе канала. Малые all-reduce стоят микросекунды независимо от размера (NCCL ring
 * около 11 мкс на маленьких сообщениях, собственные all-reduce движков 2–5 мкс), поэтому при 160
 * обменах на шаг задержка запуска важнее полосы. Перекрытие связи с вычислениями не учитываем.
 * Источники формул:
 *   TP: 2 all-reduce на слой в forward (Megatron), ring all-reduce гоняет 2(N−1)/N объёма;
 *       KV-головы дублируются при TP > kv_heads, MLA-latent дублируется на каждом ранге (notes-numbers §1.9);
 *   PP: слои делятся на стадии, токен проходит стадии по очереди (книга с. 143), KV не дублируется;
 *   EP: эксперты целиком на GPU, attention реплицирован (DP-attention), токены едут к экспертам all-to-all
 *       (книга с. 145–146; notes-techniques §1.4.5, §3.15);
 *   DP: полные реплики, связи на каждом шаге нет;
 *   TP8 × PP2: внутри ноды TP по NVLink, между нодами PP по InfiniBand (книга, глава 5: TP8PP2).
 */
import type { GpuSpecExt } from '@/engine/data/gpus'
import { getInterconnect } from '@/engine/data/interconnects'
import type { ModelSpecExt } from '@/engine/data/models'
import { BYTES_PER_PARAM, DEFAULT_MBU, DEFAULT_MFU, DEFAULT_RESERVE_GIB, GiB, STEP_OVERHEAD_MS, kvBytesPerToken, moeExpertsTouched, peakTflops, stepTime, tpKvReplication, usableMemBytes, VLLM_MEM_UTIL, weightBytes } from '@/engine/perf'
import type { DType } from '@/engine/types'
import { nvlinkFor } from '../disaggregation/logic'

/** tppp: TP внутри ноды (8 GPU) и PP между нодами, на 16 GPU это TP8 × PP2. */
export type Strategy = 'tp' | 'pp' | 'ep' | 'dp' | 'tppp'

/** Канал связи: NVLink (или PCIe у карт без него) внутри ноды, InfiniBand между нодами. */
export type LinkKind = 'nvlink' | 'pcie' | 'ib'

export const GPUS_PER_NODE = 8
/** Связь между нодами: одна карта InfiniBand NDR 400G на GPU (DGX H100 / B200: 8 NIC на 8 GPU). */
export const IB_PER_GPU_GBS = getInterconnect('ib-ndr').GBs
/**
 * Условная задержка запуска одного обмена (коллектив или пересылка между стадиями), мс:
 * 10 мкс по NVLink, 30 мкс по InfiniBand. Порядок величин: NCCL ring около 11 мкс на малых
 * сообщениях внутри ноды, между нодами десятки микросекунд и больше.
 */
export const LAUNCH_MS: Readonly<Record<LinkKind, number>> = { nvlink: 0.01, pcie: 0.01, ib: 0.03 }
/** Байт на элемент активаций, которые гоняет связь (BF16). */
const ACT_BYTES = 2


export interface PlanInput {
  model: ModelSpecExt
  gpu: GpuSpecExt
  dtype: DType
  strategy: Strategy
  gpus: number
  /** Сколько последовательностей одновременно генерирует вся система. */
  concurrency: number
  /** Средняя длина последовательности (промпт + ответ), токенов. */
  context: number
}

export interface PlanResult {
  feasible: boolean
  /** Почему схема невозможна (по-русски). */
  reason?: string
  /** Помещаются ли веса в память каждого GPU. */
  weightsFit: boolean
  weightsPerGpu: number
  /** Память GPU под процесс: usable × 0,92 − резерв 2 ГиБ. */
  usablePerGpu: number
  /** Под KV на каждом GPU (может быть отрицательным). */
  kvBudgetPerGpu: number
  /** Сколько последовательностей длиной context помещается во всю систему. */
  maxSeqs: number
  /** Дублирование KV (TP > kv_heads, MLA при TP). */
  kvReplication: number
  /** Шаг decode для одного пользователя (от токена до токена), мс. */
  tokenMs: number
  /** Из него: работа GPU (память или вычисления, плюс накладные), мс. */
  workMs: number
  /** Из него: связь между GPU, мс (задержки запуска и передача по полосе). */
  commMs: number
  /** Из связи: задержки запуска обменов, мс. */
  commLaunchMs: number
  /** Из связи: то, что идёт по InfiniBand между нодами, мс. */
  commIbMs: number
  /** Сколько раз за шаг GPU обмениваются данными. */
  commOps: number
  /** Байт на один GPU за шаг. */
  commBytes: number
  /** По какому каналу идёт связь (самый медленный из использованных). */
  commLink: LinkKind | 'none'
  commGBs: number
  perUserTokS: number
  totalTokS: number
  /** Используемая конкурентность (урезана до maxSeqs, если не помещается). */
  concurrencyUsed: number
  /** Что лежит на каждом GPU, по-русски. */
  holds: string
  /** Для EP: экспертов на GPU. */
  expertsPerGpu?: number
  /** Для PP и TP × PP: слоёв на стадию. */
  layersPerStage?: number
  /** Для PP и TP × PP: стадий конвейера. */
  stages?: number
  /** Для TP и TP × PP: GPU в одной TP-группе. */
  tpSize?: number
}

/** Связь между GPU: NVLink внутри ноды, InfiniBand, если GPU больше, чем в одной ноде. */
export function linkOf(gpu: GpuSpecExt, gpus: number): { kind: LinkKind; GBs: number } {
  if (gpus > GPUS_PER_NODE) return { kind: 'ib', GBs: IB_PER_GPU_GBS }
  return intraNodeLink(gpu)
}

/** Канал внутри ноды: NVLink своего поколения, у карт без NVLink их PCIe. */
function intraNodeLink(gpu: GpuSpecExt): { kind: LinkKind; GBs: number } {
  const nv = nvlinkFor(gpu)
  return { kind: nv.id.startsWith('nvlink') ? 'nvlink' : 'pcie', GBs: nv.GBs }
}

/** Байт весов экспертов MoE в формате dtype. */
function expertBytes(model: ModelSpecExt, dtype: DType): number {
  return (model.expertParamsB ?? 0) * 1e9 * BYTES_PER_PARAM[dtype]
}

/** Память под процесс на одном GPU (как fitsCheck: usable × memUtil − резерв), memUtil как в vLLM по умолчанию (0,92). */
export function usablePerGpu(gpu: GpuSpecExt): number {
  return usableMemBytes(gpu) * VLLM_MEM_UTIL - DEFAULT_RESERVE_GIB * GiB
}

/** Время ring all-reduce для сообщения msg байт на N GPU (только полоса, без задержки запуска). */
export function allReduceMs(msgBytes: number, n: number, GBs: number): number {
  if (n <= 1) return 0
  return ((2 * (n - 1)) / n) * msgBytes / (GBs * 1e9) * 1000
}

/**
 * Шаг decode на одном GPU, если работа batch последовательностей поделена на shards GPU поровну
 * (TP-срезы, стадии PP или их произведение), а KV дублируется в kvRep раз. Roofline как perf.stepTime.
 */
function shardedStepMs(model: ModelSpecExt, gpu: GpuSpecExt, dtype: DType, batch: number, ctx: number, shards: number, kvRep: number): number {
  const raw = stepTime(model, gpu, { decodeUniform: { batch: Math.max(1, batch), context: ctx } }, { dtype, gpus: 1, overheadMs: 0 })
  const memMs = ((raw.weightBytes + raw.kvBytes * kvRep) / shards / (gpu.bwTBs * 1e12 * DEFAULT_MBU)) * 1000
  const compMs = (raw.flops / shards / (peakTflops(gpu, dtype) * 1e12 * DEFAULT_MFU)) * 1000
  return Math.max(memMs, compMs) + STEP_OVERHEAD_MS
}

export function plan(inp: PlanInput): PlanResult {
  const { model, gpu, dtype, strategy } = inp
  const n = Math.max(1, Math.floor(inp.gpus))
  const ctx = Math.max(1, inp.context)
  const totalW = weightBytes(model, dtype)
  const usable = usablePerGpu(gpu)
  const kvTok = kvBytesPerToken(model)
  const isMoe = Boolean(model.experts && model.expertParamsB)
  const link = linkOf(gpu, n)
  const intra = intraNodeLink(gpu)

  const empty = (reason: string): PlanResult => ({
    feasible: false,
    reason,
    weightsFit: false,
    weightsPerGpu: 0,
    usablePerGpu: usable,
    kvBudgetPerGpu: 0,
    maxSeqs: 0,
    kvReplication: 1,
    tokenMs: 0,
    workMs: 0,
    commMs: 0,
    commLaunchMs: 0,
    commIbMs: 0,
    commOps: 0,
    commBytes: 0,
    commLink: 'none',
    commGBs: 0,
    perUserTokS: 0,
    totalTokS: 0,
    concurrencyUsed: 0,
    holds: '',
  })

  if (strategy === 'ep' && !isMoe) return empty('EP раскладывает экспертов MoE, а у этой модели их нет: она плотная.')
  if (strategy === 'tppp' && n <= GPUS_PER_NODE) return empty('TP × PP нужен, когда GPU больше, чем в одной ноде: внутри нод TP, между нодами PP.')
  const tpSize = strategy === 'tppp' ? GPUS_PER_NODE : strategy === 'tp' ? n : undefined
  const stages = strategy === 'tppp' ? Math.round(n / GPUS_PER_NODE) : strategy === 'pp' ? n : undefined

  // --- память: веса и KV на GPU ---
  let weightsPerGpu: number
  let kvPerTokenPerGpu: number // байт KV на токен последовательности, который лежит на этом GPU
  let seqsShareGpu: boolean // true: каждый GPU хранит долю каждой последовательности (TP, PP)
  let rep = 1
  let holds: string
  let expertsPerGpu: number | undefined
  let layersPerStage: number | undefined
  if (strategy === 'tp') {
    rep = tpKvReplication(model, n)
    weightsPerGpu = totalW / n
    kvPerTokenPerGpu = (kvTok * rep) / n
    seqsShareGpu = true
    holds = n === 1 ? 'вся модель' : `1/${n} каждой матрицы во всех ${model.layers} слоях`
  } else if (strategy === 'pp') {
    layersPerStage = model.layers / n
    weightsPerGpu = totalW / n
    kvPerTokenPerGpu = kvTok / n
    seqsShareGpu = true
    holds = n === 1 ? 'вся модель' : `${fmtLayers(model.layers, n)} слоёв подряд`
  } else if (strategy === 'tppp') {
    rep = tpKvReplication(model, tpSize!)
    layersPerStage = model.layers / stages!
    weightsPerGpu = totalW / n
    kvPerTokenPerGpu = (kvTok * rep) / n
    seqsShareGpu = true
    holds = `1/${tpSize} каждой матрицы в ${fmtLayers(model.layers, stages!)} слоях своей ноды`
  } else if (strategy === 'ep') {
    const E = model.experts!.total
    expertsPerGpu = E / n
    const exp = expertBytes(model, dtype)
    weightsPerGpu = totalW - exp + exp / n
    kvPerTokenPerGpu = kvTok
    seqsShareGpu = false
    holds = n === 1 ? 'вся модель' : `${trimNum(expertsPerGpu)} из ${E} экспертов и копия остальных весов`
  } else {
    weightsPerGpu = totalW
    kvPerTokenPerGpu = kvTok
    seqsShareGpu = false
    holds = 'вся модель'
  }
  const kvBudgetPerGpu = usable - weightsPerGpu
  const weightsFit = kvBudgetPerGpu > 0
  const tokensPerGpu = weightsFit ? Math.floor(kvBudgetPerGpu / kvPerTokenPerGpu) : 0
  const systemTokens = seqsShareGpu ? tokensPerGpu : tokensPerGpu * n
  const maxSeqs = Math.floor(systemTokens / ctx)

  const base: PlanResult = {
    ...empty(''),
    feasible: weightsFit,
    reason: weightsFit ? undefined : 'Веса не помещаются в память GPU при такой раскладке.',
    weightsFit,
    weightsPerGpu,
    kvBudgetPerGpu,
    maxSeqs,
    kvReplication: rep,
    holds,
    expertsPerGpu,
    layersPerStage,
    stages,
    tpSize,
  }
  if (!weightsFit) return base

  // --- время шага decode ---
  const c = Math.max(1, Math.min(inp.concurrency, Math.max(1, maxSeqs)))
  base.concurrencyUsed = c
  let launch = 0 // мс задержек запуска
  let ib = 0 // мс связи по InfiniBand
  if (strategy === 'tp') {
    const work = shardedStepMs(model, gpu, dtype, c, ctx, n, rep)
    const msg = c * model.hidden * ACT_BYTES
    const ops = n > 1 ? 2 * model.layers : 0
    launch = ops * LAUNCH_MS[link.kind]
    const commMs = launch + ops * allReduceMs(msg, n, link.GBs)
    if (link.kind === 'ib') ib = commMs
    base.workMs = work
    base.commMs = commMs
    base.commOps = ops
    base.commBytes = n > 1 ? ops * ((2 * (n - 1)) / n) * msg : 0
  } else if (strategy === 'pp' || strategy === 'tppp') {
    // Микробатчи по c / S: каждая стадия считает свои слои, токен проходит все стадии по очереди.
    const S = stages!
    const tp = strategy === 'tppp' ? tpSize! : 1
    const mb = Math.max(1, Math.round(c / S))
    const stage = shardedStepMs(model, gpu, dtype, mb, ctx, n, rep)
    const msg = mb * model.hidden * ACT_BYTES
    // пересылки активаций между соседними стадиями: внутри ноды по NVLink, между нодами по InfiniBand
    const nodes = Math.ceil(n / GPUS_PER_NODE)
    const ibHops = S > 1 ? nodes - 1 : 0
    const nvHops = Math.max(0, S - 1 - ibHops)
    const hopNv = nvHops * (LAUNCH_MS[intra.kind] + (msg / (intra.GBs * 1e9)) * 1000)
    const hopIb = ibHops * (LAUNCH_MS.ib + (msg / (IB_PER_GPU_GBS * 1e9)) * 1000)
    // TP внутри стадии: 2 all-reduce на каждый слой, всего 2·L на токен, по NVLink
    const arOps = tp > 1 ? 2 * model.layers : 0
    const arMs = arOps * (LAUNCH_MS[intra.kind] + allReduceMs(msg, tp, intra.GBs))
    launch = (nvHops + arOps) * LAUNCH_MS[intra.kind] + ibHops * LAUNCH_MS.ib
    ib = hopIb
    base.workMs = S * stage
    base.commMs = hopNv + hopIb + arMs
    base.commOps = S - 1 + arOps
    base.commBytes = (S - 1) * msg + (tp > 1 ? arOps * ((2 * (tp - 1)) / tp) * msg : 0)
  } else if (strategy === 'ep') {
    const E = model.experts!.total
    const k = model.experts!.active
    const b = Math.max(1, c / n)
    const bpp = BYTES_PER_PARAM[dtype]
    const exp = expertBytes(model, dtype)
    const always = Math.max(0, model.activeParamsB * 1e9 - ((model.expertParamsB ?? 0) * 1e9 * k) / E) * bpp
    const touchedLocal = moeExpertsTouched(model, c) / n // задетые эксперты из тех, что лежат на этом GPU
    const local = stepTime(model, gpu, { decodeUniform: { batch: Math.round(b), context: ctx } }, { dtype, gpus: 1, overheadMs: 0 })
    const bytes = always + (exp / E) * touchedLocal + local.kvBytes
    const memMs = (bytes / (gpu.bwTBs * 1e12 * DEFAULT_MBU)) * 1000
    const compMs = (local.flops / (peakTflops(gpu, dtype) * 1e12 * DEFAULT_MFU)) * 1000
    const msg = b * k * model.hidden * ACT_BYTES
    const ops = n > 1 ? 2 * model.layers : 0
    launch = ops * LAUNCH_MS[link.kind]
    const commMs = n > 1 ? launch + (ops * ((msg * (n - 1)) / n)) / (link.GBs * 1e9) * 1000 : 0
    if (link.kind === 'ib') ib = commMs
    base.workMs = Math.max(memMs, compMs) + STEP_OVERHEAD_MS
    base.commMs = commMs
    base.commOps = ops
    base.commBytes = n > 1 ? (ops * msg * (n - 1)) / n : 0
  } else {
    const b = Math.max(1, Math.round(c / n))
    base.workMs = shardedStepMs(model, gpu, dtype, b, ctx, 1, 1)
  }
  base.commLaunchMs = launch
  base.commIbMs = ib
  base.tokenMs = base.workMs + base.commMs
  base.perUserTokS = 1000 / base.tokenMs
  // Каждый из c запросов получает токен раз в tokenMs. Для PP при полном конвейере (c ≥ S) это то же,
  // что mb·1000 / (stage + пересылка); при c < S часть стадий простаивает, и throughput не выше c / tokenMs.
  base.totalTokS = c * base.perUserTokS
  const usesIb = n > GPUS_PER_NODE && base.commOps > 0
  base.commLink = base.commOps > 0 ? (usesIb ? 'ib' : link.kind) : 'none'
  base.commGBs = base.commOps > 0 ? (usesIb ? IB_PER_GPU_GBS : intra.GBs) : 0
  return base
}

function trimNum(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(1).replace('.', ',')
}

function fmtLayers(layers: number, n: number): string {
  const per = layers / n
  return Number.isInteger(per) ? String(per) : `${Math.floor(per)} или ${Math.ceil(per)}`
}

/* ============================================================================
 * Пузыри конвейера (PP)
 * ========================================================================== */

/** Доля времени, когда стадии простаивают, при M микробатчах и S стадиях (один проход). */
export function bubbleFraction(stages: number, microbatches: number): number {
  const s = Math.max(1, stages)
  const m = Math.max(1, microbatches)
  return (s - 1) / (m + s - 1)
}

/** Простой в непрерывном decode: микробатч после последней стадии сразу входит в первую. */
export function decodeBubbleFraction(stages: number, microbatches: number): number {
  const s = Math.max(1, stages)
  const m = Math.max(1, microbatches)
  return Math.max(0, 1 - m / s)
}

/** Расписание GPipe-подобного прохода: [стадия][слот] = номер микробатча или -1 (пузырь). */
export function pipelineSchedule(stages: number, microbatches: number): number[][] {
  const s = Math.max(1, stages)
  const m = Math.max(1, microbatches)
  const slots = m + s - 1
  return Array.from({ length: s }, (_, st) => Array.from({ length: slots }, (_, t) => (t - st >= 0 && t - st < m ? t - st : -1)))
}

/**
 * Установившийся режим decode: микробатчи ходят по кругу, период max(M, S) слотов; стадия s в слоте t
 * считает микробатч (t − s) mod период, если такой есть, иначе простаивает. Слотов: два периода,
 * если период короткий (до 8), иначе один.
 */
export function decodeSchedule(stages: number, microbatches: number): number[][] {
  const s = Math.max(1, stages)
  const m = Math.max(1, microbatches)
  const period = Math.max(m, s)
  const slots = period <= 8 ? 2 * period : period
  return Array.from({ length: s }, (_, st) =>
    Array.from({ length: slots }, (_, t) => {
      const k = (((t - st) % period) + period) % period
      return k < m ? k : -1
    }),
  )
}
