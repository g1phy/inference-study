/**
 * Логика главы 16 (дезагрегация): сценарии, прогоны sched.ts, решение conditional disaggregation.
 * Чистые функции без React, покрыты тестами (disaggregation.test.ts).
 */
import { getInterconnect, type InterconnectSpecExt } from '@/engine/data/interconnects'
import type { GpuSpecExt } from '@/engine/data/gpus'
import { makeWorkload, quantile, simulateContinuous, simulateDisagg, type Dist, type SimResultExt } from '@/engine/sched'
import { decodeStepMs } from '@/engine/perf'
import type { SimRequest } from '@/engine/types'
import { perfOf, type SimConfig } from '../batching/config'

/* ============================================================================
 * Каналы для передачи KV
 * ========================================================================== */

/** NVLink того поколения, что стоит у GPU; у PCIe-карт без NVLink: их хост-линк. */
export function nvlinkFor(gpu: GpuSpecExt): InterconnectSpecExt {
  const gen = gpu.nvlinkGen ?? ''
  if (/NVLink 5/.test(gen)) return getInterconnect('nvlink5')
  if (/NVLink 4/.test(gen)) return getInterconnect('nvlink4')
  if (/NVLink 3/.test(gen)) return getInterconnect('nvlink3')
  return /Gen5/.test(gpu.hostLink ?? '') ? getInterconnect('pcie5') : getInterconnect('pcie4')
}

export type LinkChoice = 'nvlink' | 'rdma' | 'tcp'

/**
 * Полоса передачи KV для воркера с tp GPU. NIXL передаёт KV каждого TP-ранга отдельно,
 * а у каждого GPU свой NVLink и своя сетевая карта (DGX: 8 NIC на 8 GPU), поэтому NVLink
 * и RDMA масштабируются с TP; TCP считаем одним каналом на воркер.
 */
export function linkFor(choice: LinkChoice, gpu: GpuSpecExt, tp: number): { spec: InterconnectSpecExt; GBs: number } {
  const spec = choice === 'nvlink' ? nvlinkFor(gpu) : choice === 'rdma' ? getInterconnect('ib-ndr') : getInterconnect('tcp-25g')
  return { spec, GBs: spec.GBs * (choice === 'tcp' ? 1 : Math.max(1, tp)) }
}

/* ============================================================================
 * Фигура 1: один поток, два режима
 * ========================================================================== */

/** 12 запросов с короткими и длинными промптами (время эталона: Qwen3-8B на H100). */
const LANE_SPEC: ReadonlyArray<readonly [number, number, number]> = [
  [0, 900, 160],
  [40, 7000, 120],
  [120, 500, 150],
  [260, 1200, 140],
  [380, 12000, 100],
  [520, 700, 160],
  [640, 3000, 130],
  [800, 600, 150],
  [950, 9000, 110],
  [1100, 800, 150],
  [1250, 1500, 120],
  [1400, 5000, 100],
]

export function laneRequests(timeScale: number): SimRequest[] {
  return LANE_SPEC.map(([a, p, o], i) => ({ id: `r${i + 1}`, arrivalMs: a * timeScale, promptTokens: p, outputTokens: o }))
}

export interface LaneRuns {
  agg: SimResultExt
  disagg: SimResultExt
  linkGBs: number
  linkName: string
}

export function runLanes(cfg: SimConfig, link: LinkChoice): LaneRuns {
  const reqs = laneRequests(cfg.timeScale)
  const base = perfOf(cfg)
  const l = linkFor(link, cfg.gpu, cfg.tp)
  const agg = simulateContinuous(reqs, { ...base, workers: 2 })
  const disagg = simulateDisagg(reqs, { ...base, prefillWorkers: 1, decodeWorkers: 1, link: l.GBs })
  return { agg, disagg, linkGBs: l.GBs, linkName: l.spec.name }
}

/* ============================================================================
 * Фигура 2: сравнение под нагрузкой
 * ========================================================================== */

export type PresetId = 'chat' | 'long' | 'reason'

export interface Slo {
  ttftMs: number
  /** Ни одна пауза между токенами запроса не длиннее itlMs. */
  itlMs: number
  /**
   * Медианная пауза запроса не длиннее itlP50Factor пауз одиночного запроса на этой конфигурации
   * (условная цель «скорость чтения не ниже половины одиночной» для длинных ответов).
   */
  itlP50Factor?: number
}

export interface Preset {
  id: PresetId
  label: string
  prompt: number
  output: number
  /** Запросов в секунду на всю систему из COMPARE_WORKERS воркеров (во времени эталона). */
  rate: number
  slo: Slo
}

/** Пауза между токенами, которую уже видно глазу как подвисание текста (условный порог SLO). */
export const STALL_MS = 100

/** Профили нагрузки и SLO условные. */
export const PRESETS: readonly Preset[] = [
  { id: 'long', label: 'Длинный контекст', prompt: 6000, output: 150, rate: 14, slo: { ttftMs: 2000, itlMs: STALL_MS } },
  { id: 'chat', label: 'Чат', prompt: 800, output: 250, rate: 40, slo: { ttftMs: 500, itlMs: STALL_MS } },
  { id: 'reason', label: 'Рассуждения', prompt: 600, output: 1500, rate: 18, slo: { ttftMs: 500, itlMs: STALL_MS, itlP50Factor: 2 } },
]

/** Воркеров в сравнении: agg на 4 воркерах против 1P3D, 2P2D, 3P1D. */
export const COMPARE_WORKERS = 4
export const COMPARE_COUNT = 120

export interface Load {
  prompt: number
  output: number
  rate: number
}

export function compareWorkload(load: Load, timeScale: number): SimRequest[] {
  const promptDist: Dist = { kind: 'lognormal', median: load.prompt, sigma: 0.5, min: 16, max: Math.min(40000, load.prompt * 5) }
  const outputDist: Dist = { kind: 'lognormal', median: load.output, sigma: 0.5, min: 8, max: Math.min(8000, load.output * 4) }
  // интенсивность задаётся «в эталоне»: на медленной конфигурации поток реже, чтобы картинка сохраняла форму
  return makeWorkload({ rate: load.rate / timeScale, promptDist, outputDist, seed: 11, count: COMPARE_COUNT })
}

export interface RunSummary {
  key: string
  label: string
  prefill: number
  decode: number
  gpus: number
  ttftP50: number
  ttftP99: number
  itlP50: number
  itlP99: number
  perGpu: number
  slo: number
  /** Медиана времени передачи KV (только disagg). */
  transferP50: number
}

/**
 * Доля запросов, у которых TTFT и все паузы между токенами в пределах SLO (строгий вариант для стриминга);
 * с itlP50Ms ещё и медианная пауза запроса не длиннее itlP50Ms.
 */
export function sloShare(res: SimResultExt, slo: Slo, itlP50Ms?: number): number {
  if (!res.requests.length) return 0
  let ok = 0
  for (const r of res.requests) {
    let worst = 0
    for (const v of r.itlMs) if (v > worst) worst = v
    let medOk = true
    if (itlP50Ms !== undefined && r.itlMs.length > 0) {
      const sorted = r.itlMs.slice().sort((a, b) => a - b)
      medOk = sorted[Math.floor(sorted.length / 2)] <= itlP50Ms
    }
    if (r.ttftMs <= slo.ttftMs && worst <= slo.itlMs && medOk) ok++
  }
  return ok / res.requests.length
}

/** Пауза между токенами одиночного запроса (batch 1) на этой конфигурации, мс. */
export function soloItlMs(cfg: SimConfig, load: Load): number {
  return decodeStepMs(cfg.model, cfg.gpu, { batch: 1, avgContext: load.prompt + load.output / 2, gpus: cfg.tp, dtype: cfg.dtype })
}

function summarizeRun(key: string, label: string, res: SimResultExt, prefill: number, decode: number, gpus: number, slo: Slo, itlP50Ms?: number): RunSummary {
  const s = res.summary
  const tr = res.transfers.map((t) => t.endMs - t.startMs)
  return {
    key,
    label,
    prefill,
    decode,
    gpus,
    ttftP50: s.ttftP50,
    ttftP99: s.ttftP99,
    itlP50: s.itlP50,
    itlP99: s.itlP99,
    perGpu: s.throughputPerGpuTokS,
    slo: sloShare(res, slo, itlP50Ms),
    transferP50: tr.length ? quantile(tr, 0.5) : 0,
  }
}

/** Агрегированный режим на workers воркерах и все разбиения xPyD с x + y = workers. */
export function runComparison(
  cfg: SimConfig,
  load: Load,
  slo: Slo,
  opts: { workers?: number; aggBudget?: number } = {},
): { agg: RunSummary; splits: RunSummary[]; itlP50Ms?: number } {
  const workers = opts.workers ?? COMPARE_WORKERS
  const reqs = compareWorkload(load, cfg.timeScale)
  const base = perfOf(cfg)
  const link = linkFor('rdma', cfg.gpu, cfg.tp).GBs
  const gpus = workers * cfg.tp
  const itlP50Ms = slo.itlP50Factor !== undefined ? slo.itlP50Factor * soloItlMs(cfg, load) : undefined
  const agg = summarizeRun('agg', `Агрегированный, ${workers} воркера`, simulateContinuous(reqs, { ...base, workers, tokenBudget: opts.aggBudget }), 0, 0, gpus, slo, itlP50Ms)
  const splits: RunSummary[] = []
  for (let x = 1; x < workers; x++) {
    const res = simulateDisagg(reqs, { ...base, prefillWorkers: x, decodeWorkers: workers - x, link })
    splits.push(summarizeRun(`${x}P${workers - x}D`, `${x}P${workers - x}D`, res, x, workers - x, gpus, slo, itlP50Ms))
  }
  return { agg, splits, itlP50Ms }
}

/**
 * Лучшая схема: больше доля в SLO; при равенстве больше выдано токенов на GPU, потом короче хвост ITL.
 * null, если все схемы целиком в SLO: тогда «лучшей» нет.
 */
export function bestRun(runs: readonly RunSummary[]): RunSummary | null {
  if (runs.every((r) => r.slo >= 1 - 1e-9)) return null
  return runs.slice().sort((a, b) => b.slo - a.slo || b.perGpu - a.perGpu || a.itlP99 - b.itlP99)[0]
}

/* ============================================================================
 * Conditional disaggregation (Dynamo, экспериментально; SPEC §8.1, notes-dynamo §1.15, §3.6)
 * ========================================================================== */

export type CondPolicy = 'isl_bounding' | 'prefill_load' | 'isl_or_load'

export const EFF_ISL_THRESHOLD = 2048
export const EFF_ISL_RATIO_THRESHOLD = 0.7

export interface CondInput {
  isl: number
  /** Сколько токенов промпта уже лежит в KV cache выбранного decode-воркера. */
  cached: number
  policy: CondPolicy
  /** Выбранный prefill-воркер выше prefill_busy_threshold. */
  prefillBusy: boolean
  effThreshold?: number
  ratioThreshold?: number
}

export interface CondDecision {
  local: boolean
  effIsl: number
  ratio: number
  islOk: boolean
  ratioOk: boolean
  islBounding: boolean
  loadOk: boolean
}

/**
 * Решение роутера Dynamo: делать prefill на decode-воркере (local) или отправить на prefill-воркер.
 * isl_bounding: local ⇔ eff_isl < 2048 и eff_isl / ISL < 0.7 (условия строгие); prefill_load: local, если
 * выбранный prefill-воркер перегружен; isl_or_load: любое из двух.
 */
export function decideConditional(inp: CondInput): CondDecision {
  const isl = Math.max(1, Math.floor(inp.isl))
  const cached = Math.min(Math.max(0, Math.floor(inp.cached)), isl)
  const effIsl = isl - cached
  const ratio = effIsl / isl
  const islOk = effIsl < (inp.effThreshold ?? EFF_ISL_THRESHOLD)
  const ratioOk = ratio < (inp.ratioThreshold ?? EFF_ISL_RATIO_THRESHOLD)
  const islBounding = islOk && ratioOk
  const loadOk = inp.prefillBusy
  const local = inp.policy === 'isl_bounding' ? islBounding : inp.policy === 'prefill_load' ? loadOk : islBounding || loadOk
  return { local, effIsl, ratio, islOk, ratioOk, islBounding, loadOk }
}

/** Четыре запроса из разбора заметок (notes-dynamo §3.6). */
export const COND_EXAMPLES: ReadonlyArray<{ label: string; isl: number; cached: number }> = [
  { label: 'Длинный чат', isl: 10000, cached: 8500 },
  { label: 'Средний без кэша', isl: 3000, cached: 0 },
  { label: 'Короткий без кэша', isl: 1800, cached: 0 },
  { label: 'Частично в кэше', isl: 2600, cached: 1000 },
]
