/**
 * KV-aware роутер в стиле NVIDIA Dynamo, round-robin и random, симуляция потока запросов (агент A2).
 *
 * Формула стоимости Dynamo ≥ 1.2 (research/notes-dynamo.md §1.8, сверено исследователем с кодом
 * v1.6.2; контракт в src/engine/types.ts → RouterOptions):
 *   raw_prefill   = (active_prefill_tokens + ISL) / B                    (дробное)
 *   credit        = c·device_overlap + 0.75·host_overlap + 0.25·disk_overlap
 *                 + 0.5·shared_beyond (общий пул кластера, если задан sharedPoolBlocks)
 *   cost          = prefill_load_scale · max(0, raw_prefill − credit)
 *                 + (active_decode_blocks + additional_active_blocks)
 *                 + active_requests_weight · active_requests
 * additional_active_blocks — по коду Dynamo (lib/router/slot-tracker/src/prompt_registry.rs):
 * `token_sequence.len() − overlap_depth`, где token_sequence — хэши ПОЛНЫХ блоков запроса
 * (floor(ISL / B), хвост не хэшируется: lib/kv-hashing/src/routing.rs), а overlap_depth — совпадение
 * с trie АКТИВНЫХ запросов этого воркера (не с prefix cache: notes-dynamo.md §1.8 «а не промахи кэша»).
 * Активные блоки ведут applyRoute / releaseRoute (RouterWorker.activeBlockRefs).
 * Выбор: temperature = 0 → argmin (в Dynamo ничьи разыгрываются случайно, у нас для детерминизма
 * побеждает первый по порядку); иначе p_i ∝ exp(−(c_i − c_min) / (T·(c_max − c_min))) [C-PICK].
 * 'legacy' (≤ v1.1, книга и старые доки): cost = overlap_score_weight · prefill_blocks + decode_blocks,
 * где prefill_blocks = непокрытые кэшем блоки, decode_blocks = активные блоки (notes-numbers.md §3.21).
 * Не моделируем: decay кредита (по умолчанию 0), preferred taint.
 * Пример документации (notes-dynamo.md §3.3) с «potential decode» 10 / 5 / 9 при 10 блоках запроса
 * воспроизводится, только если закэшированный префикс держат активные запросы (activeBlockHashes).
 */
import type { BlockInfo, DType, GpuSpec, ModelSpec, RouteDecision, RouterOptions, WorkerScore, WorkerState } from './types'
import { exponential, hashString, mulberry32, randInt, type Rng } from './rng'
import { DEFAULT_BLOCK_SIZE, blockHashes } from './kv'
import { decodeStepMs, prefillMs } from './perf'

/* ============================================================================
 * Типы и константы
 * ========================================================================== */

/** Кредит за блок в host RAM (Dynamo host_cache_hit_weight = 0.75, notes-dynamo.md §1.8). */
export const HOST_CACHE_HIT_WEIGHT = 0.75
/** Кредит за блок на локальном диске (Dynamo disk_cache_hit_weight = 0.25). */
export const DISK_CACHE_HIT_WEIGHT = 0.25
/** Кредит за блок из общего пула кластера (Dynamo shared_cache_multiplier = 0.5 при hicache, notes-dynamo.md §1.8). */
export const SHARED_CACHE_MULTIPLIER = 0.5

/** Дефолты Dynamo: overlap_score_credit 1.0, router_temperature 0, prefill_load_scale 1.0, вес запросов 0. */
export const DEFAULT_ROUTER_OPTIONS: RouterOptions = {
  overlapCredit: 1.0,
  temperature: 0,
  prefillLoadScale: 1.0,
  activeRequestsWeight: 0,
  formula: 'v1.2',
  legacyOverlapWeight: 1.0,
}

/** Режим роутинга. */
export type RouteMode = RouteDecision['mode']

/** WorkerState + необязательные поля. */
export interface RouterWorker extends WorkerState {
  /**
   * Хэши блоков, которые сейчас держат активные запросы воркера (trie активных промптов в Dynamo).
   * По ним считается additional_active_blocks при assume_kv_reuse: блоки, уже занятые активными
   * последовательностями, новый запрос не добавляет. Не задано = активных блоков с известными
   * хэшами нет (prefix cache сам по себе decode-слагаемое не уменьшает). Ведут applyRoute / releaseRoute.
   */
  activeBlockHashes?: Set<string>
  /**
   * Сколько активных запросов держат каждый блок из activeBlockHashes (ref count, как unique_blocks
   * в Dynamo): блок освобождается, когда завершился последний запрос, который его держал.
   */
  activeBlockRefs?: Map<string, number>
}

/** RouterOptions + необязательные поля. */
export interface KvRouterOptions extends RouterOptions {
  /** Размер блока B, токенов. По умолчанию длина первого полного блока запроса, иначе 16. */
  blockSize?: number
  /** Dynamo --router-assume-kv-reuse (по умолчанию true): дедуплицировать новые блоки с активными. */
  assumeKvReuse?: boolean
  /** Кредит host-блока (по умолчанию 0.75) и disk-блока (0.25). */
  hostCacheHitWeight?: number
  diskCacheHitWeight?: number
  /**
   * Общий пул кластера (SGLang HiCache L3 на Mooncake, Dynamo `--shared-cache-type hicache`): хэши
   * блоков, которые любой воркер может подтянуть из пула. Кредит = shared_cache_multiplier ·
   * hits_beyond(device_overlap): блоки префикса в пуле за пределами device-префикса воркера
   * (notes-dynamo.md §1.8, §3.5б; код lib/router/scheduler/src/selector/reference.rs).
   */
  sharedPoolBlocks?: ReadonlySet<string>
  /** shared_cache_multiplier: по умолчанию 0.5, если задан sharedPoolBlocks (Dynamo: 0.5 при hicache, иначе 0). */
  sharedCacheMultiplier?: number
  /** Генератор для выбора при temperature > 0 (иначе mulberry32(seed ?? 0)). */
  rng?: Rng
}

/** WorkerScore + подробности расчёта. */
export interface KvWorkerScore extends WorkerScore {
  name: string
  hostOverlapBlocks: number
  diskOverlapBlocks: number
  /** Блоки префикса в общем пуле за пределами device-префикса (кредит ×0.5); 0 без пула. */
  sharedBeyondBlocks: number
  /** (active_prefill_tokens + ISL) / B. */
  rawPrefillBlocks: number
  /** max(0, raw − credit). */
  adjustedPrefillBlocks: number
  /** Новые полные блоки в активной памяти воркера (additional_active_blocks = полные блоки − общие с активными). */
  newBlocks: number
  activeBlocks: number
  queuedPrefillTokens: number
  activeRequests: number
  activeRequestsTerm: number
  /** Токенов запроса уже в GPU-кэше воркера (device overlap × B, не больше ISL). */
  cachedTokens: number
  /** Сколько токенов придётся посчитать в prefill на этом воркере. */
  prefillTokens: number
}

/** RouteDecision с подробностями. */
export interface KvRouteDecision extends RouteDecision {
  scores: KvWorkerScore[]
  /** Длина промпта в токенах. */
  isl: number
  blockSize: number
  formula: 'v1.2' | 'legacy'
  temperature: number
  /** Хэши полных блоков запроса (по ним releaseRoute снимает ссылки активных блоков). */
  fullBlockHashes: string[]
  /** Был ли включён assume_kv_reuse (дедупликация с активными блоками). */
  assumeKvReuse: boolean
}

/* ============================================================================
 * Подсчёт стоимости
 * ========================================================================== */

const RU_NF = new Map<number, Intl.NumberFormat>()
/** Число в русской записи (DESIGN §7): разряды неразрывным пробелом, до `digits` знаков после запятой. */
function ru(x: number, digits = 1): string {
  let f = RU_NF.get(digits)
  if (!f) RU_NF.set(digits, (f = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: digits })))
  const r = Math.round(x * 10 ** digits) / 10 ** digits
  return f.format(Object.is(r, -0) ? 0 : r).replace('-', '−')
}

/** Форма слова после числа: forms = [для 1, для 2–4, для 5–20]; у дробного числа `frac` (родительный ед.: «12,5 блока»). */
function plural(n: number, forms: readonly [string, string, string], frac: string = forms[1]): string {
  if (!Number.isInteger(n)) return frac
  const a = Math.abs(n) % 100
  const b = a % 10
  if (a > 10 && a < 20) return forms[2]
  if (b === 1) return forms[0]
  if (b >= 2 && b <= 4) return forms[1]
  return forms[2]
}

/** «1 блок», «3 блока», «36 блоков». */
const BLOCKS_NOM = ['блок', 'блока', 'блоков'] as const
/** «из 1 блока», «из 7 блоков»; дробное «против 12,5 блока». */
const BLOCKS_GEN = ['блока', 'блоков', 'блоков'] as const
/** «1 024 токена», «4 096 токенов». */
const TOKENS_NOM = ['токен', 'токена', 'токенов'] as const
/** «из 1 воркера», «из 3 воркеров». */
const WORKERS_GEN = ['воркера', 'воркеров', 'воркеров'] as const
const NBSP = '\u00a0'

/** Число и слово в нужной форме: ruCount(1024, TOKENS_NOM) → «1 024 токена». Округляет один раз, чтобы число и форма совпали. */
function ruCount(x: number, forms: readonly [string, string, string], frac?: string, digits = 1): string {
  const r = Math.round(x * 10 ** digits) / 10 ** digits
  return `${ru(r, digits)}${NBSP}${plural(r, forms, frac)}`
}

/** Вероятность в процентах, как подписи столбиков (fmt.pct): «<0,1 %», «5,4 %» (один знак меньше 10 %), «44 %». */
function pct(p: number): string {
  if (p > 0 && p < 0.001) return `<0,1${NBSP}%`
  return `${ru(p * 100, p > 0 && p < 0.1 ? 1 : 0)}${NBSP}%`
}

/** Перечисление имён: «W2», «W2 и W3», «W1, W2 и W3». */
function joinNames(names: readonly string[]): string {
  return names.length < 2 ? (names[0] ?? '') : `${names.slice(0, -1).join(', ')} и ${names[names.length - 1]}`
}

function workerName(w: WorkerState): string {
  return w.name || w.id
}

/** Размер блока: из опций, иначе длина первого полного блока, иначе 16. */
function inferBlockSize(blocks: readonly BlockInfo[], opts: Partial<KvRouterOptions>): number {
  if (opts.blockSize && opts.blockSize > 0) return opts.blockSize
  const full = blocks.find((b) => b.full)
  return full ? full.tokenIds.length : DEFAULT_BLOCK_SIZE
}

/**
 * Перекрытие запроса с кэшем воркера: самый длинный префикс полных блоков, присутствующих у воркера
 * на каком-либо уровне; каждый блок засчитывается на лучшем уровне (GPU, затем host, затем disk).
 */
export function overlapByTier(blocks: readonly BlockInfo[], w: WorkerState): { device: number; host: number; disk: number } {
  let device = 0
  let host = 0
  let disk = 0
  for (const b of blocks) {
    if (!b.full) break
    if (w.cachedBlocks.has(b.hash)) device++
    else if (w.hostBlocks?.has(b.hash)) host++
    else if (w.diskBlocks?.has(b.hash)) disk++
    else break
  }
  return { device, host, disk }
}

/** Стоимость одного воркера по формуле Dynamo (вероятность заполняется при выборе). */
export function scoreWorker(blocks: readonly BlockInfo[], w: RouterWorker, opts: Partial<KvRouterOptions> = {}): KvWorkerScore {
  const o = { ...DEFAULT_ROUTER_OPTIONS, ...opts }
  const B = inferBlockSize(blocks, o)
  const isl = blocks.reduce((a, b) => a + b.tokenIds.length, 0)
  const requestBlocks = Math.ceil(isl / B)
  const ov = overlapByTier(blocks, w)
  const queued = w.queuedPrefillTokens ?? 0
  const activeRequests = w.activeRequests ?? 0
  const cachedTokens = Math.min(isl, ov.device * B)
  const anyTier = ov.device + ov.host + ov.disk

  // additional_active_blocks (Dynamo prompt_registry.rs): хэши полных блоков запроса минус
  // совпадение по префиксу с блоками АКТИВНЫХ запросов воркера. Хвост < B не хэшируется и не
  // считается; закэшированные, но никем не занятые блоки считаются новыми (кэш — скидка на prefill).
  const fullBlocks = blocks.filter((b) => b.full).length
  let newBlocks = fullBlocks
  if (o.assumeKvReuse !== false && w.activeBlockHashes && w.activeBlockHashes.size > 0) {
    let shared = 0
    while (shared < blocks.length && blocks[shared].full && w.activeBlockHashes.has(blocks[shared].hash)) shared++
    newBlocks = fullBlocks - shared
  }

  // Общий пул: глубина префикса запроса в пуле, кредит только за то, что дальше device-префикса.
  let poolDepth = 0
  const pool = o.sharedPoolBlocks
  if (pool) while (poolDepth < blocks.length && blocks[poolDepth].full && pool.has(blocks[poolDepth].hash)) poolDepth++
  const sharedBeyond = Math.max(0, poolDepth - ov.device)
  const sharedM = pool ? (o.sharedCacheMultiplier ?? SHARED_CACHE_MULTIPLIER) : 0

  const hostW = o.hostCacheHitWeight ?? HOST_CACHE_HIT_WEIGHT
  const diskW = o.diskCacheHitWeight ?? DISK_CACHE_HIT_WEIGHT
  const credit = o.overlapCredit * ov.device + hostW * ov.host + diskW * ov.disk + sharedM * sharedBeyond
  const rawPrefill = (queued + isl) / B
  const adjusted = Math.max(0, rawPrefill - credit)

  let prefillTerm: number
  let decodeTerm: number
  let decodeBlocks: number
  let activeRequestsTerm = 0
  if (o.formula === 'legacy') {
    // ≤ v1.1: overlap_score_weight · (непокрытые блоки) + активные decode-блоки.
    prefillTerm = (o.legacyOverlapWeight ?? 1) * Math.max(0, isl / B - ov.device)
    decodeBlocks = w.activeBlocks
    decodeTerm = decodeBlocks
  } else {
    prefillTerm = (o.prefillLoadScale ?? 1) * adjusted
    decodeBlocks = w.activeBlocks + newBlocks
    decodeTerm = decodeBlocks
    activeRequestsTerm = (o.activeRequestsWeight ?? 0) * activeRequests
  }
  return {
    workerId: w.id,
    name: workerName(w),
    overlapBlocks: ov.device,
    hostOverlapBlocks: ov.host,
    diskOverlapBlocks: ov.disk,
    sharedBeyondBlocks: sharedBeyond,
    prefillBlocks: Math.max(0, requestBlocks - Math.max(anyTier, poolDepth)),
    decodeBlocks,
    credit,
    prefillTerm,
    decodeTerm,
    cost: prefillTerm + decodeTerm + activeRequestsTerm,
    prob: 0,
    rawPrefillBlocks: rawPrefill,
    adjustedPrefillBlocks: adjusted,
    newBlocks,
    activeBlocks: w.activeBlocks,
    queuedPrefillTokens: queued,
    activeRequests,
    activeRequestsTerm,
    cachedTokens,
    prefillTokens: isl - cachedTokens,
  }
}

/**
 * Вероятности выбора по стоимостям: T = 0 → one-hot у минимума (ничья: первый по порядку);
 * T > 0 → softmax по min-max нормированным стоимостям (все равны → поровну).
 * Пример доков: стоимости [18, 10, 11] при T = 1 → [0.1635, 0.4444, 0.3922] (notes-dynamo.md §3.3).
 */
export function selectionProbs(costs: readonly number[], temperature: number): number[] {
  const n = costs.length
  if (n === 0) return []
  if (!(temperature > 0)) {
    let best = 0
    for (let i = 1; i < n; i++) if (costs[i] < costs[best]) best = i
    return costs.map((_, i) => (i === best ? 1 : 0))
  }
  const cmin = Math.min(...costs)
  const cmax = Math.max(...costs)
  const range = cmax - cmin
  if (!(range > 1e-12)) return costs.map(() => 1 / n)
  const w = costs.map((c) => Math.exp(-(c - cmin) / (temperature * range)))
  const s = w.reduce((a, b) => a + b, 0)
  return w.map((x) => x / s)
}

/** Выбор индекса по вероятностям (детерминированно при заданном rng). */
function sampleIndex(probs: readonly number[], rng: Rng): number {
  let r = rng()
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i]
    if (r < 0) return i
  }
  return probs.length - 1
}

/** Объяснение выбора по-русски. */
function explainKv(scores: KvWorkerScore[], chosenIdx: number, requestBlocks: number, temperature: number, formula: 'v1.2' | 'legacy'): string {
  const c = scores[chosenIdx]
  const parts: string[] = []
  let head = `Выбран ${c.name}: стоимость ${ru(c.cost)} = prefill ${ru(c.prefillTerm ?? 0)} + decode ${ru(c.decodeTerm ?? 0)}`
  if (c.activeRequestsTerm > 0) head += ` + запросы ${ru(c.activeRequestsTerm)}`
  parts.push(head + '.')
  if (c.overlapBlocks > 0) {
    parts.push(`В его GPU-кэше уже ${ru(c.overlapBlocks)} из ${ruCount(requestBlocks, BLOCKS_GEN)} запроса, prefill посчитает только остальное.`)
  } else if (c.hostOverlapBlocks + c.diskOverlapBlocks > 0) {
    parts.push(`Префикс лежит у него ниже GPU (host: ${ru(c.hostOverlapBlocks)}, диск: ${ru(c.diskOverlapBlocks)} бл.): подгрузить дешевле, чем пересчитать.`)
  } else {
    parts.push('Префикса запроса в его кэше нет: prefill посчитает все блоки.')
  }
  if (c.sharedBeyondBlocks > 0) {
    parts.push(`Ещё ${ru(c.sharedBeyondBlocks)} бл. префикса можно подтянуть из общего пула кластера: за них кредит меньше, чем за блок в его GPU.`)
  }
  // Воркер с самым большим кэшем, если это не выбранный. Сравнение «кэш есть, но дороже» печатается, только если он
  // действительно дороже выбранного: при T > 0 выбор случайный и выбранный может стоить больше, а при ничьей при T = 0
  // выбор объясняет фраза про одинаковую стоимость.
  let best = -1
  for (let i = 0; i < scores.length; i++) {
    if (i === chosenIdx) continue
    const tot = scores[i].overlapBlocks + scores[i].hostOverlapBlocks
    if (tot > c.overlapBlocks + c.hostOverlapBlocks && (best < 0 || tot > scores[best].overlapBlocks + scores[best].hostOverlapBlocks)) best = i
  }
  if (best >= 0 && scores[best].cost > c.cost + 1e-9) {
    const b = scores[best]
    const why: string[] = []
    if ((b.decodeTerm ?? 0) > (c.decodeTerm ?? 0)) why.push(`нагрузка decode выше (${ru(b.decodeTerm ?? 0)} против ${ruCount(c.decodeTerm ?? 0, BLOCKS_GEN, 'блока')})`)
    if (formula === 'v1.2' && b.queuedPrefillTokens > c.queuedPrefillTokens) why.push(`в очереди prefill уже ${ruCount(b.queuedPrefillTokens, TOKENS_NOM)}`)
    const reason = why.length ? why.join(', а ') : 'итоговая стоимость не меньше'
    parts.push(`${b.name} держит больше всего кэша (${ru(b.overlapBlocks + b.hostOverlapBlocks)} бл.), но его стоимость ${ru(b.cost)}: ${reason}. Кэш даёт скидку, а не гарантию.`)
  }
  if (temperature > 0) {
    // Выпал не самый дешёвый: назвать самого дешёвого (или нескольких с одинаковой минимальной стоимостью).
    const minCost = Math.min(...scores.map((s) => s.cost))
    if (c.cost > minCost + 1e-9) {
      const cheapest = scores.filter((s) => Math.abs(s.cost - minCost) < 1e-9)
      const m = cheapest[0]
      parts.push(
        cheapest.length === 1
          ? `${c.name} не самый дешёвый: ${m.name} стоит меньше всех (${ru(m.cost)}, шанс ${pct(m.prob)}).`
          : `${c.name} не самый дешёвый: ${joinNames(cheapest.map((s) => s.name))} стоят меньше всех (${ru(m.cost)}, шанс у каждого ${pct(m.prob)}).`,
      )
    }
    parts.push(`Температура ${ru(temperature, 2)}: воркер выбирается случайно по softmax от стоимостей, шанс выбрать ${c.name} был ${pct(c.prob)}.`)
  } else if (scores.filter((s) => Math.abs(s.cost - c.cost) < 1e-9).length > 1) {
    parts.push('Несколько воркеров с одинаковой стоимостью: берём первого по порядку (Dynamo выбрал бы случайно).')
  }
  return parts.join(' ')
}

/* ============================================================================
 * Роутинг
 * ========================================================================== */

/**
 * KV-aware выбор воркера строго по формуле Dynamo ≥ 1.2 (или 'legacy'). Перекрытие считается как
 * самый длинный префикс цепочки хэшей блоков запроса, лежащий у воркера (GPU, затем host/disk).
 * Пример доков (notes-dynamo.md §3.3): стоимости 18 / 10 / 11 → выбран W2.
 */
export function routeKV(requestBlocks: readonly BlockInfo[], workers: readonly RouterWorker[], opts: Partial<KvRouterOptions> = DEFAULT_ROUTER_OPTIONS): KvRouteDecision {
  if (workers.length === 0) throw new Error('routeKV: нет воркеров')
  const o: KvRouterOptions = { ...DEFAULT_ROUTER_OPTIONS, ...opts }
  const B = inferBlockSize(requestBlocks, o)
  const isl = requestBlocks.reduce((a, b) => a + b.tokenIds.length, 0)
  const scores = workers.map((w) => scoreWorker(requestBlocks, w, { ...o, blockSize: B }))
  const probs = selectionProbs(
    scores.map((s) => s.cost),
    o.temperature,
  )
  scores.forEach((s, i) => (s.prob = probs[i]))
  const chosenIdx = o.temperature > 0 ? sampleIndex(probs, o.rng ?? mulberry32(o.seed ?? 0)) : probs.indexOf(1)
  const requestBlockCount = Math.ceil(isl / B)
  const formula = o.formula ?? 'v1.2'
  return {
    mode: 'kv',
    requestBlocks: requestBlockCount,
    scores,
    chosen: workers[chosenIdx].id,
    explanation: explainKv(scores, chosenIdx, requestBlockCount, o.temperature, formula),
    isl,
    blockSize: B,
    formula,
    temperature: o.temperature,
    fullBlockHashes: fullHashes(requestBlocks),
    assumeKvReuse: o.assumeKvReuse !== false,
  }
}

/** Хэши полных блоков запроса (хвост < B не хэшируется). */
function fullHashes(blocks: readonly BlockInfo[]): string[] {
  const out: string[] = []
  for (const b of blocks) if (b.full) out.push(b.hash)
  return out
}

/** Решение с заданным выбранным воркером (для round-robin и random): стоимости считаются для сравнения. */
function fixedDecision(
  mode: 'round-robin' | 'random',
  requestBlocks: readonly BlockInfo[],
  workers: readonly RouterWorker[],
  chosenIdx: number,
  probOf: (i: number) => number,
  opts: Partial<KvRouterOptions>,
  explanation: string,
): KvRouteDecision {
  const o: KvRouterOptions = { ...DEFAULT_ROUTER_OPTIONS, ...opts }
  const B = inferBlockSize(requestBlocks, o)
  const isl = requestBlocks.reduce((a, b) => a + b.tokenIds.length, 0)
  const scores = workers.map((w, i) => ({ ...scoreWorker(requestBlocks, w, { ...o, blockSize: B }), prob: probOf(i) }))
  return {
    mode,
    requestBlocks: Math.ceil(isl / B),
    scores,
    chosen: workers[chosenIdx].id,
    explanation,
    isl,
    blockSize: B,
    formula: o.formula ?? 'v1.2',
    temperature: 0,
    fullBlockHashes: fullHashes(requestBlocks),
    assumeKvReuse: o.assumeKvReuse !== false,
  }
}

/**
 * Round-robin (режим Dynamo по умолчанию, notes-dynamo.md §1.9): запрос номер `counter` идёт
 * воркеру counter mod N; кэш и нагрузка не учитываются (стоимости в scores только для сравнения).
 */
export function routeRoundRobin(
  requestBlocks: readonly BlockInfo[],
  workers: readonly RouterWorker[],
  counter: number,
  opts: Partial<KvRouterOptions> = {},
): KvRouteDecision {
  if (workers.length === 0) throw new Error('routeRoundRobin: нет воркеров')
  const n = workers.length
  const idx = ((Math.floor(counter) % n) + n) % n
  const tmp = scoreWorker(requestBlocks, workers[idx], opts)
  const explanation =
    `Round-robin: по кругу выбран ${workerName(workers[idx])} (запрос №${Math.floor(counter) + 1}), кэш и нагрузка не учитываются. ` +
    (tmp.overlapBlocks > 0 ? `Повезло: в его кэше ${ruCount(tmp.overlapBlocks, BLOCKS_NOM)} запроса.` : 'Префикса запроса в его кэше нет.')
  return fixedDecision('round-robin', requestBlocks, workers, idx, (i) => (i === idx ? 1 : 0), opts, explanation)
}

/** Случайный воркер (равновероятно), детерминированно по seed или rng. */
export function routeRandom(
  requestBlocks: readonly BlockInfo[],
  workers: readonly RouterWorker[],
  seedOrRng: number | Rng,
  opts: Partial<KvRouterOptions> = {},
): KvRouteDecision {
  if (workers.length === 0) throw new Error('routeRandom: нет воркеров')
  const rng = typeof seedOrRng === 'function' ? seedOrRng : mulberry32(seedOrRng)
  const idx = Math.min(workers.length - 1, Math.floor(rng() * workers.length))
  const tmp = scoreWorker(requestBlocks, workers[idx], opts)
  const explanation =
    `Случайный выбор: ${workerName(workers[idx])} (у каждого из ${ruCount(workers.length, WORKERS_GEN)} шанс ${Math.round(100 / workers.length)}${NBSP}%). ` +
    (tmp.overlapBlocks > 0 ? `В его кэше ${ruCount(tmp.overlapBlocks, BLOCKS_NOM)} запроса.` : 'Префикса запроса в его кэше нет.')
  return fixedDecision('random', requestBlocks, workers, idx, () => 1 / workers.length, opts, explanation)
}

/* ============================================================================
 * Состояние воркеров после назначения
 * ========================================================================== */

/** Копия воркера с новыми Set/Map (исходный не меняется). */
function cloneWorker(w: RouterWorker): RouterWorker {
  return {
    ...w,
    cachedBlocks: new Set(w.cachedBlocks),
    hostBlocks: w.hostBlocks ? new Set(w.hostBlocks) : undefined,
    diskBlocks: w.diskBlocks ? new Set(w.diskBlocks) : undefined,
    activeBlockHashes: w.activeBlockHashes ? new Set(w.activeBlockHashes) : undefined,
    activeBlockRefs: w.activeBlockRefs ? new Map(w.activeBlockRefs) : undefined,
  }
}

/** Счётчики ссылок активных блоков воркера; блоки из activeBlockHashes без счётчика держит «кто-то ещё» (1). */
function activeRefsOf(w: RouterWorker): Map<string, number> {
  const refs = new Map(w.activeBlockRefs ?? [])
  for (const h of w.activeBlockHashes ?? []) if (!refs.has(h)) refs.set(h, 1)
  return refs
}

/**
 * Применяет решение: выбранный воркер кладёт полные блоки запроса в GPU-кэш (порядок Set = LRU:
 * тронутые блоки уходят в конец), растёт нагрузка: блоки запроса становятся активными (ref count +1,
 * active_decode_blocks += сколько из них раньше не держал ни один активный запрос),
 * active_prefill_tokens += токены к расчёту, active_requests += 1. Если у воркера задан
 * capacityBlocks, самые старые блоки вытесняются (и переезжают в hostBlocks, если у воркера
 * есть host-уровень). Возвращает НОВЫЙ массив; остальные воркеры те же объекты.
 */
export function applyRoute(
  workers: readonly RouterWorker[],
  decision: RouteDecision,
  blocks: readonly BlockInfo[],
  opts: { capacityBlocks?: number } = {},
): RouterWorker[] {
  return workers.map((w) => {
    if (w.id !== decision.chosen) return w
    const score = decision.scores.find((s) => s.workerId === w.id) as KvWorkerScore | undefined
    const nw = cloneWorker(w)
    const mine = new Set<string>()
    for (const b of blocks) {
      if (!b.full) continue
      mine.add(b.hash)
      nw.cachedBlocks.delete(b.hash)
      nw.cachedBlocks.add(b.hash)
    }
    const isl = blocks.reduce((a, b) => a + b.tokenIds.length, 0)
    const B = (decision as Partial<KvRouteDecision>).blockSize ?? inferBlockSize(blocks, {})
    // Активные блоки: при assume_kv_reuse запрос добавляет только блоки, которых ещё не держал
    // ни один активный запрос (ref count 0 → 1), как unique_blocks в Dynamo; без него — все полные блоки.
    let newBlocks = 0
    if ((decision as Partial<KvRouteDecision>).assumeKvReuse !== false) {
      const refs = activeRefsOf(w)
      const active = new Set(w.activeBlockHashes ?? [])
      for (const b of blocks) {
        if (!b.full) continue
        const r = refs.get(b.hash) ?? 0
        if (r === 0) newBlocks++
        refs.set(b.hash, r + 1)
        active.add(b.hash)
      }
      nw.activeBlockRefs = refs
      nw.activeBlockHashes = active
    } else newBlocks = score?.newBlocks ?? blocks.filter((b) => b.full).length
    const prefillTokens = score?.prefillTokens ?? Math.max(0, isl - (score?.overlapBlocks ?? 0) * B)
    nw.activeBlocks = w.activeBlocks + newBlocks
    nw.queuedPrefillTokens = (w.queuedPrefillTokens ?? 0) + prefillTokens
    nw.activeRequests = (w.activeRequests ?? 0) + 1
    const cap = opts.capacityBlocks ?? w.capacityBlocks
    if (cap !== undefined && nw.cachedBlocks.size > cap) {
      for (const h of nw.cachedBlocks) {
        if (nw.cachedBlocks.size <= cap) break
        if (mine.has(h)) continue // блоки текущего запроса не вытесняем
        nw.cachedBlocks.delete(h)
        nw.hostBlocks?.add(h)
      }
    }
    return nw
  })
}

/**
 * Снимает нагрузку запроса: 'prefill' — prefill закончился (Dynamo снимает prefill-токены
 * по первому токену), 'finish' — запрос завершён: ref count его блоков −1, освобождаются только
 * блоки, которые больше не держит ни один активный запрос, и слот запроса.
 * Кэш (cachedBlocks) не трогается: блоки остаются переиспользуемыми.
 */
export function releaseRoute(workers: readonly RouterWorker[], decision: RouteDecision, phase: 'prefill' | 'finish'): RouterWorker[] {
  return workers.map((w) => {
    if (w.id !== decision.chosen) return w
    const score = decision.scores.find((s) => s.workerId === w.id) as KvWorkerScore | undefined
    if (!score) return w
    const nw: RouterWorker = { ...w }
    if (phase === 'prefill') {
      nw.queuedPrefillTokens = Math.max(0, (w.queuedPrefillTokens ?? 0) - score.prefillTokens)
      return nw
    }
    const d = decision as Partial<KvRouteDecision>
    let freed = score.newBlocks
    if (d.assumeKvReuse !== false && d.fullBlockHashes && w.activeBlockRefs) {
      const refs = new Map(w.activeBlockRefs)
      const active = new Set(w.activeBlockHashes ?? [])
      freed = 0
      for (const h of d.fullBlockHashes) {
        const r = refs.get(h) ?? 0
        if (r <= 0) continue
        if (r === 1) {
          refs.delete(h)
          active.delete(h)
          freed++
        } else refs.set(h, r - 1)
      }
      nw.activeBlockRefs = refs
      nw.activeBlockHashes = active
    }
    nw.activeBlocks = Math.max(0, w.activeBlocks - freed)
    nw.activeRequests = Math.max(0, (w.activeRequests ?? 0) - 1)
    return nw
  })
}

/**
 * Кладёт полные блоки в GPU-кэш воркера без изменения нагрузки (тёплый кэш для демо: системный
 * промпт уже посчитан на этом воркере). Возвращает новый объект воркера.
 */
export function cacheBlocks(worker: RouterWorker, blocks: readonly BlockInfo[]): RouterWorker {
  const nw = cloneWorker(worker)
  for (const b of blocks) if (b.full) nw.cachedBlocks.add(b.hash)
  return nw
}

/** Пустые воркеры для демо: worker-0 … worker-(n−1) (с учётом активных блоков для assume_kv_reuse). */
export function makeWorkers(n: number, opts: { capacityBlocks?: number; withHostTier?: boolean } = {}): RouterWorker[] {
  return Array.from({ length: n }, (_, i) => ({
    id: `worker-${i}`,
    name: `Воркер ${i + 1}`,
    cachedBlocks: new Set<string>(),
    hostBlocks: opts.withHostTier ? new Set<string>() : undefined,
    activeBlocks: 0,
    queuedPrefillTokens: 0,
    activeRequests: 0,
    capacityBlocks: opts.capacityBlocks,
    activeBlockHashes: new Set<string>(),
    activeBlockRefs: new Map<string, number>(),
  }))
}

/* ============================================================================
 * Симуляция потока запросов
 * ========================================================================== */

/** Запрос для симуляции роутинга. */
export interface RoutingRequest {
  id: string
  arrivalMs: number
  blocks: BlockInfo[]
  outputTokens: number
  /** Тенант / системный промпт (для подписей). */
  tenant?: string
}

/** Параметры simulateRouting. */
export interface RoutingSimOptions {
  /** Параметры роутера (дефолты Dynamo). */
  router?: Partial<KvRouterOptions>
  model: ModelSpec
  gpu: GpuSpec
  dtype?: DType
  kvDtype?: DType
  mfu?: number
  mbu?: number
  /** GPU на одного воркера (TP), по умолчанию 1. */
  gpus?: number
  /** Ёмкость GPU-кэша каждого воркера в блоках (если у воркера не задан capacityBlocks). */
  capacityBlocks?: number
  /** Какие стратегии сравнить (по умолчанию все три). */
  strategies?: RouteMode[]
  seed?: number
}

/** Нагрузка одного воркера за симуляцию. */
export interface RoutingWorkerLoad {
  workerId: string
  name: string
  requests: number
  promptTokens: number
  cachedTokens: number
  /** Посчитано в prefill. */
  prefillTokens: number
  /** Время GPU на prefill, мс. */
  busyMs: number
  hitRate: number
  /** Блоков в GPU-кэше в конце. */
  cachedBlocksEnd: number
}

/** Итог одной стратегии. */
export interface RoutingStrategyResult {
  mode: RouteMode
  /** cached prompt tokens / prompt tokens (определение SGLang, notes-engines.md §2.1). */
  hitRate: number
  meanTtftMs: number
  p50TtftMs: number
  p90TtftMs: number
  p99TtftMs: number
  promptTokens: number
  cachedTokens: number
  /** Суммарное GPU-время prefill, мс. */
  prefillGpuMs: number
  perWorker: RoutingWorkerLoad[]
  /** Максимум запросов на воркер / среднее (1 = идеально ровно). */
  loadImbalance: number
  /** Куда ушёл каждый запрос (id воркера, в порядке прихода). */
  assignments: string[]
  ttftMs: number[]
}

/** Итог simulateRouting. */
export interface RoutingSimResult {
  strategies: RoutingStrategyResult[]
  byMode: Partial<Record<RouteMode, RoutingStrategyResult>>
}

/** Квантиль с линейной интерполяцией (numpy default). */
function quantileLinear(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return 0
  const h = (sorted.length - 1) * q
  const lo = Math.floor(h)
  const hi = Math.ceil(h)
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo])
}

/**
 * Прогоняет один и тот же поток запросов через стратегии роутинга и сравнивает hit rate, TTFT и
 * баланс нагрузки. Модель воркера: prefill-очередь FIFO (время из perf.prefillMs с учётом
 * закэшированного префикса), decode занимает воркер outputTokens × decodeStepMs при текущем числе
 * активных запросов; нагрузка снимается событиями (releaseRoute). Кэш воркера ограничен capacityBlocks
 * (LRU). Учитываются только попадания в GPU-кэш (host/disk-уровни в симуляции не моделируются).
 */
export function simulateRouting(requests: readonly RoutingRequest[], workers: readonly RouterWorker[], opts: RoutingSimOptions): RoutingSimResult {
  const modes = opts.strategies ?? (['kv', 'round-robin', 'random'] as RouteMode[])
  const sorted = requests.slice().sort((a, b) => a.arrivalMs - b.arrivalMs || (a.id < b.id ? -1 : 1))
  const results: RoutingStrategyResult[] = []
  for (const mode of modes) {
    let ws: RouterWorker[] = workers.map(cloneWorker)
    const rng = mulberry32(((opts.seed ?? 0) ^ hashString(mode)) >>> 0)
    const freeAt = new Map<string, number>(ws.map((w) => [w.id, 0]))
    const load = new Map<string, RoutingWorkerLoad>(
      ws.map((w) => [w.id, { workerId: w.id, name: workerName(w), requests: 0, promptTokens: 0, cachedTokens: 0, prefillTokens: 0, busyMs: 0, hitRate: 0, cachedBlocksEnd: 0 }]),
    )
    const events: Array<{ t: number; phase: 'prefill' | 'finish'; decision: RouteDecision }> = []
    const ttfts: number[] = []
    const assignments: string[] = []
    let promptTokens = 0
    let cachedTokens = 0
    let prefillGpuMs = 0
    sorted.forEach((req, k) => {
      // Снимаем нагрузку завершившихся этапов.
      events.sort((a, b) => a.t - b.t)
      while (events.length && events[0].t <= req.arrivalMs) {
        const e = events.shift()!
        ws = releaseRoute(ws, e.decision, e.phase)
      }
      const routerOpts: Partial<KvRouterOptions> = { ...opts.router, rng }
      const decision: KvRouteDecision =
        mode === 'kv' ? routeKV(req.blocks, ws, routerOpts) : mode === 'round-robin' ? routeRoundRobin(req.blocks, ws, k, routerOpts) : routeRandom(req.blocks, ws, rng, routerOpts)
      const score = decision.scores.find((s) => s.workerId === decision.chosen)!
      ws = applyRoute(ws, decision, req.blocks, { capacityBlocks: opts.capacityBlocks })
      const w = ws.find((x) => x.id === decision.chosen)!
      const isl = decision.isl
      const perf = { dtype: opts.dtype, kvDtype: opts.kvDtype, mfu: opts.mfu, mbu: opts.mbu, gpus: opts.gpus }
      const start = Math.max(req.arrivalMs, freeAt.get(w.id) ?? 0)
      const dur = prefillMs(opts.model, opts.gpu, isl, { ...perf, cachedTokens: score.cachedTokens })
      const end = start + dur
      freeAt.set(w.id, end)
      const step = decodeStepMs(opts.model, opts.gpu, { ...perf, batch: Math.max(1, w.activeRequests ?? 1), avgContext: isl + req.outputTokens / 2 })
      events.push({ t: end, phase: 'prefill', decision }, { t: end + req.outputTokens * step, phase: 'finish', decision })
      ttfts.push(end - req.arrivalMs)
      assignments.push(w.id)
      promptTokens += isl
      cachedTokens += score.cachedTokens
      prefillGpuMs += dur
      const l = load.get(w.id)!
      l.requests += 1
      l.promptTokens += isl
      l.cachedTokens += score.cachedTokens
      l.prefillTokens += isl - score.cachedTokens
      l.busyMs += dur
    })
    for (const w of ws) {
      const l = load.get(w.id)!
      l.hitRate = l.promptTokens ? l.cachedTokens / l.promptTokens : 0
      l.cachedBlocksEnd = w.cachedBlocks.size
    }
    const sortedT = ttfts.slice().sort((a, b) => a - b)
    const perWorker = Array.from(load.values())
    const meanReq = sorted.length / Math.max(1, ws.length)
    results.push({
      mode,
      hitRate: promptTokens ? cachedTokens / promptTokens : 0,
      meanTtftMs: ttfts.length ? ttfts.reduce((a, b) => a + b, 0) / ttfts.length : 0,
      p50TtftMs: quantileLinear(sortedT, 0.5),
      p90TtftMs: quantileLinear(sortedT, 0.9),
      p99TtftMs: quantileLinear(sortedT, 0.99),
      promptTokens,
      cachedTokens,
      prefillGpuMs,
      perWorker,
      loadImbalance: meanReq > 0 ? Math.max(...perWorker.map((p) => p.requests)) / meanReq : 1,
      assignments,
      ttftMs: ttfts,
    })
  }
  const byMode: Partial<Record<RouteMode, RoutingStrategyResult>> = {}
  for (const r of results) byMode[r.mode] = r
  return { strategies: results, byMode }
}

/** Параметры генератора «тенантов с общими системными промптами». */
export interface TenantWorkloadOptions {
  /** Сколько разных системных промптов (тенантов). */
  tenants: number
  /** Длина системного промпта, токенов. */
  systemTokens: number
  /** Длина уникальной части запроса: число или [min, max]. */
  userTokens: number | [number, number]
  /** Сколько запросов. */
  requests: number
  /** Интенсивность пуассоновского потока, запросов в секунду. */
  ratePerSec: number
  /** Длина ответа, токенов (по умолчанию 200). */
  outputTokens?: number
  blockSize?: number
  seed?: number
}

/**
 * Поток запросов «у каждого тенанта свой системный промпт» для главы про роутинг (SPEC §6.20,
 * фигура 2): токены синтетические (детерминированно по seed), блоки и хэши настоящие (kv.blockHashes).
 */
export function makeTenantWorkload(opts: TenantWorkloadOptions): RoutingRequest[] {
  const rng = mulberry32(opts.seed ?? 7)
  const B = opts.blockSize ?? DEFAULT_BLOCK_SIZE
  // Синтетический словарь: id токенов в диапазоне обычных токенов Qwen3 (0..151642, types.ts).
  const tok = () => randInt(rng, 0, 151642)
  const systems = Array.from({ length: Math.max(1, opts.tenants) }, () => Array.from({ length: opts.systemTokens }, tok))
  const out: RoutingRequest[] = []
  let t = 0
  for (let i = 0; i < opts.requests; i++) {
    t += exponential(rng, opts.ratePerSec / 1000)
    const tenant = Math.floor(rng() * systems.length)
    const u = Array.isArray(opts.userTokens) ? randInt(rng, opts.userTokens[0], opts.userTokens[1]) : opts.userTokens
    const ids = systems[tenant].concat(Array.from({ length: u }, tok))
    out.push({ id: `req-${i}`, arrivalMs: t, blocks: blockHashes(ids, B), outputTokens: opts.outputTokens ?? 200, tenant: `tenant-${tenant}` })
  }
  return out
}
