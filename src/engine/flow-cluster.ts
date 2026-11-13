/**
 * Кластер симулятора (агент S1a): воркеры, их кэш и нагрузка, решения роутера для agg и disagg.
 *
 * Состояние условное, но устроено честно:
 *  - в кэше воркеров лежат настоящие блоки других клиентов (тенантов): их промпты прогнаны через тот же
 *    шаблон Qwen3, токенизатор и blockHashes, поэтому с промптом читателя они не совпадают с первого блока;
 *  - тёплый режим: Воркер 2 недавно обслужил запрос с тем же началом (system prompt и начало сообщения),
 *    у него лежат все полные блоки промпта, кроме последнего (при одном полном блоке: этот блок);
 *  - нагрузка одинакова в холодном и тёплом режиме (зависит только от длины промпта): в холодном побеждает
 *    наименее загруженный Воркер 3, в тёплом Воркер 2 нагружен чуть сильнее, но скидка за кэш перевешивает
 *    (нагрузка Воркера 2 = нагрузка Воркера 3 + c/2 + 0,25 блока, где c = блоков в тёплом кэше);
 *  - agg: routeKV с дефолтами Dynamo (c = 1, T = 0);
 *  - disagg: PrefillRouter первым выбирает prefill-воркер (KV-aware, decode-нагрузку не считает:
 *    track_active_blocks = false), decode-воркер выбирается только по нагрузке (overlap_score_credit = 0,
 *    assume_kv_reuse = false), research/notes-dynamo.md §1.4, SPEC §8.1.
 */
import type { BlockInfo, FlowClusterInfo, FlowWorkerInfo, GpuSpec, ModelSpec, Tokenizer } from './types'
import { blockHashes, cachedPrefixBlocks } from './kv'
import { kvCapacityTokens } from './perf'

/** gpu_memory_utilization vLLM по умолчанию с v0.20 (SPEC §8.8): от него считается ёмкость KV воркера. */
export const SIM_GPU_MEMORY_UTILIZATION = 0.92
import { cacheBlocks, routeKV, selectionProbs, DEFAULT_ROUTER_OPTIONS, type KvRouteDecision, type KvWorkerScore, type RouterWorker } from './router'
import { renderTemplate } from './templates'

/** Тенанты: чужие префиксы в кэше воркеров. */
interface TenantSpec {
  id: string
  label: string
  system: string
  user: string
}

const TENANTS: readonly TenantSpec[] = [
  {
    id: 'shop',
    label: 'поддержка магазина',
    system:
      'Ты ассистент службы поддержки интернет-магазина электроники. Отвечай вежливо, коротко и только по делу. Если вопрос о возврате, сначала уточни номер заказа.',
    user: 'Здравствуйте! Хочу вернуть наушники, они мне не подошли.',
  },
  {
    id: 'review',
    label: 'код-ревью',
    system: 'You are a senior TypeScript reviewer. Point out bugs first, then style issues. Keep answers under 200 words.',
    user: 'Review this function: export function sum(a: number, b: number) { return a - b }',
  },
  {
    id: 'legal',
    label: 'помощник юриста',
    system:
      'Ты помощник юриста. Отвечай со ссылками на статьи Гражданского кодекса РФ и всегда предупреждай, что это не юридическая консультация.',
    user: 'Можно ли расторгнуть договор аренды квартиры досрочно?',
  },
  {
    id: 'docs',
    label: 'поиск по документации',
    system:
      'Отвечай только по приведённым документам. Документ 1: руководство по эксплуатации GPU-кластера. Узлы перезапускаются по очереди, перед перезапуском воркер дренирует активные запросы.',
    user: 'Как безопасно перезапустить узел?',
  },
  {
    id: 'translate',
    label: 'переводчик',
    system: 'Ты переводчик. Переводи с английского на русский, термины оставляй в оригинале.',
    user: 'Translate: The KV cache grows linearly with the context length.',
  },
]

const tenantCache = new WeakMap<Tokenizer, Map<string, BlockInfo[]>>()

/** Полные блоки промпта тенанта (настоящий шаблон и токенизатор, кэш на токенизатор). */
function tenantBlocks(tokenizer: Tokenizer, id: string, blockSize: number): BlockInfo[] {
  let m = tenantCache.get(tokenizer)
  if (!m) {
    m = new Map()
    tenantCache.set(tokenizer, m)
  }
  const key = `${id}#${blockSize}`
  let blocks = m.get(key)
  if (!blocks) {
    const t = TENANTS.find((x) => x.id === id)!
    const tpl = renderTemplate({
      family: 'qwen3',
      messages: [
        { role: 'system', content: t.system },
        { role: 'user', content: t.user },
      ],
      enableThinking: false,
      addGenerationPrompt: true,
    })
    const ids = tokenizer.encode(tpl.text, { allowSpecial: true }).map((x) => x.id)
    blocks = blockHashes(ids, blockSize).filter((b) => b.full)
    m.set(key, blocks)
  }
  return blocks
}

/** Число в русской записи с одним знаком после запятой. */
function ru(x: number, digits = 1): string {
  const r = Math.round(x * 10 ** digits) / 10 ** digits
  return new Intl.NumberFormat('ru-RU', { maximumFractionDigits: digits }).format(r)
}

/** Склонение: 1 блок, 2 блока, 5 блоков. */
export function plural(n: number, forms: readonly [string, string, string]): string {
  const a = Math.abs(n) % 100
  const b = a % 10
  if (a > 10 && a < 20) return forms[2]
  if (b > 1 && b < 5) return forms[1]
  if (b === 1) return forms[0]
  return forms[2]
}
const BLOCKS: readonly [string, string, string] = ['блок', 'блока', 'блоков']

/** «блока» для дробных («38,8 блока»), иначе по обычному склонению. */
function blocksWord(x: number): string {
  const r = Math.round(x * 10) / 10
  return Number.isInteger(r) ? plural(r, BLOCKS) : BLOCKS[1]
}

interface WorkerSeed {
  tenants: string[]
  activeBlocks: number
  activeRequests: number
  queued: number
}

/** Агрегированные воркеры: индекс 1 тёплый, индекс 2 наименее загруженный. */
const AGG_SEEDS: readonly WorkerSeed[] = [
  { tenants: ['shop', 'review'], activeBlocks: 46, activeRequests: 3, queued: 0 },
  { tenants: ['legal'], activeBlocks: 31, activeRequests: 2, queued: 0 }, // queued считается от длины промпта
  { tenants: ['docs'], activeBlocks: 38, activeRequests: 2, queued: 0 },
  { tenants: ['shop', 'translate'], activeBlocks: 52, activeRequests: 4, queued: 64 },
]
/** Prefill-пул disagg (decode-нагрузку prefill-роутер не считает, решают очередь prefill и кэш). */
const PREFILL_SEEDS: readonly WorkerSeed[] = [
  { tenants: ['shop', 'review'], activeBlocks: 0, activeRequests: 1, queued: 96 },
  { tenants: ['legal'], activeBlocks: 0, activeRequests: 1, queued: 0 }, // queued считается от длины промпта
  { tenants: ['docs'], activeBlocks: 0, activeRequests: 1, queued: 32 },
  { tenants: ['shop', 'translate'], activeBlocks: 0, activeRequests: 2, queued: 160 },
]
/** Decode-пул disagg: большие batch, выбор только по нагрузке. */
const DECODE_SEEDS: readonly WorkerSeed[] = [
  { tenants: ['review', 'translate'], activeBlocks: 186, activeRequests: 9, queued: 0 },
  { tenants: ['shop'], activeBlocks: 141, activeRequests: 7, queued: 0 },
]
export const WARM_WORKER_INDEX = 1
/** Batch шага decode в агрегированном режиме: у обоих воркеров, которых выбирает роутер (2 и 3), по 2 активных запроса плюс наш. */
export const AGG_DECODE_BATCH = AGG_SEEDS[2].activeRequests + 1

/** Сколько полных блоков промпта лежит на тёплом воркере: все, кроме последнего (при одном блоке: он сам). */
export function warmBlockCount(fullBlocks: number): number {
  return fullBlocks >= 2 ? fullBlocks - 1 : fullBlocks
}

export interface ClusterInput {
  tokenizer: Tokenizer
  model: ModelSpec
  gpu: GpuSpec
  /** GPU на воркер (TP). */
  tp: number
  blocks: readonly BlockInfo[]
  blockSize: number
  deployment: 'agg' | 'disagg'
  warm: boolean
}

export interface ClusterBuild {
  /** agg-воркеры или prefill-пул. */
  workers: RouterWorker[]
  routing: KvRouteDecision
  decodeWorkers?: RouterWorker[]
  decodeRouting?: KvRouteDecision
  info: FlowClusterInfo
  notes: string[]
}

function makeWorker(
  id: string,
  name: string,
  seed: WorkerSeed,
  queued: number,
  capacityBlocks: number,
  tokenizer: Tokenizer,
  blockSize: number,
): RouterWorker {
  let w: RouterWorker = {
    id,
    name,
    cachedBlocks: new Set<string>(),
    activeBlocks: seed.activeBlocks,
    queuedPrefillTokens: queued,
    activeRequests: seed.activeRequests,
    capacityBlocks,
    activeBlockHashes: new Set<string>(),
    activeBlockRefs: new Map<string, number>(),
  }
  for (const t of seed.tenants) w = cacheBlocks(w, tenantBlocks(tokenizer, t, blockSize))
  return w
}

function workerInfo(
  w: RouterWorker,
  role: FlowWorkerInfo['role'],
  seed: WorkerSeed,
  tp: number,
  tokenizer: Tokenizer,
  blockSize: number,
  promptBlocks: readonly BlockInfo[],
): FlowWorkerInfo {
  const cached: FlowWorkerInfo['cached'] = seed.tenants.map((t) => ({
    label: TENANTS.find((x) => x.id === t)!.label,
    blocks: tenantBlocks(tokenizer, t, blockSize).length,
    matchesPrompt: false,
  }))
  const own = cachedPrefixBlocks(promptBlocks, w.cachedBlocks)
  if (own > 0) cached.unshift({ label: 'начало вашего промпта', blocks: own, matchesPrompt: true })
  return {
    id: w.id,
    name: w.name,
    role,
    gpus: tp,
    capacityBlocks: w.capacityBlocks ?? 0,
    activeRequests: w.activeRequests ?? 0,
    activeBlocks: w.activeBlocks,
    queuedPrefillTokens: w.queuedPrefillTokens ?? 0,
    cached,
  }
}

/** Нагрузка воркера в блоках, как её видит формула: очередь prefill / B + активные decode-блоки. */
function loadOf(s: KvWorkerScore, blockSize: number, withDecode: boolean): number {
  return s.queuedPrefillTokens / blockSize + (withDecode ? s.activeBlocks : 0)
}

/** Пояснение к выбору (добавляется к объяснению роутера). */
function cacheStory(decision: KvRouteDecision, blockSize: number, withDecode: boolean, warm: boolean, warmBlocks: number): string {
  const scores = decision.scores
  const chosen = scores.find((s) => s.workerId === decision.chosen)!
  const anyOverlap = scores.some((s) => s.overlapBlocks > 0)
  if (warm && warmBlocks === 0) {
    return 'Промпт короче одного блока (16 токенов): полных блоков нет, кэшировать нечего, хвост не хэшируется.'
  }
  if (!anyOverlap) {
    return 'Совпадений нет ни у кого: в кэшах воркеров лежат префиксы других клиентов. Поэтому решает нагрузка, и у выбранного воркера она самая низкая.'
  }
  let least = scores[0]
  for (const s of scores) if (loadOf(s, blockSize, withDecode) < loadOf(least, blockSize, withDecode)) least = s
  if (chosen.overlapBlocks > 0 && least.workerId !== chosen.workerId) {
    return (
      `${least.name} нагружен меньше (${ru(loadOf(least, blockSize, withDecode))} против ${ru(loadOf(chosen, blockSize, withDecode))} ` +
      `${blocksWord(loadOf(chosen, blockSize, withDecode))} нагрузки), но скидка за ${chosen.overlapBlocks} ${plural(chosen.overlapBlocks, BLOCKS)} в кэше перевешивает.`
    )
  }
  if (chosen.overlapBlocks > 0) return 'У него и кэш с началом промпта, и самая низкая нагрузка.'
  return ''
}

/** Prefill-пул: стоимость без decode-слагаемого (Dynamo PrefillRouter: track_active_blocks = false). */
function routePrefillPool(blocks: readonly BlockInfo[], workers: readonly RouterWorker[], blockSize: number): KvRouteDecision {
  const base = routeKV(blocks, workers, { ...DEFAULT_ROUTER_OPTIONS, blockSize })
  const scores: KvWorkerScore[] = base.scores.map((s) => ({
    ...s,
    decodeBlocks: 0,
    decodeTerm: 0,
    newBlocks: 0,
    activeBlocks: 0,
    cost: s.prefillTerm ?? s.adjustedPrefillBlocks,
  }))
  const probs = selectionProbs(
    scores.map((s) => s.cost),
    0,
  )
  scores.forEach((s, i) => (s.prob = probs[i]))
  const idx = Math.max(0, probs.indexOf(1))
  const c = scores[idx]
  const explanation =
    'PrefillRouter выбирает prefill-воркер первым. Стоимость = блоки к prefill с учётом очереди минус кредит за кэш; ' +
    'decode-нагрузку prefill-роутер не считает (track_active_blocks = false). ' +
    `Выбран ${c.name}: стоимость ${ru(c.cost)} = (${ru(c.queuedPrefillTokens, 0)} + ${ru(base.isl, 0)}) / ${blockSize} − ${ru(c.credit ?? 0)}.`
  return { ...base, scores, chosen: workers[idx].id, explanation }
}

/**
 * Decode-пул: только нагрузка. Dynamo строит для decode-этапа роутер с overlap_score_credit = 0, assume_kv_reuse = false и
 * track_prefill_tokens = false (prefill_router/mod.rs: build_decode_router_override): prefill-слагаемое не считается вовсе,
 * стоимость = decode-блоки + active_request_cost (по умолчанию 0). research/notes-dynamo.md §1.4.
 */
function routeDecodePool(blocks: readonly BlockInfo[], workers: readonly RouterWorker[], blockSize: number): KvRouteDecision {
  const ws = workers.map((w) => ({ ...w, queuedPrefillTokens: 0 }))
  const base = routeKV(blocks, ws, { ...DEFAULT_ROUTER_OPTIONS, overlapCredit: 0, assumeKvReuse: false, blockSize })
  const scores: KvWorkerScore[] = base.scores.map((s) => ({
    ...s,
    prefillTerm: 0,
    cost: (s.decodeTerm ?? s.decodeBlocks) + (s.activeRequestsTerm ?? 0),
  }))
  const probs = selectionProbs(
    scores.map((s) => s.cost),
    0,
  )
  scores.forEach((s, i) => (s.prob = probs[i]))
  const idx = Math.max(0, probs.indexOf(1))
  const c = scores[idx]
  const others = scores.filter((_, i) => i !== idx)
  const load = (s: KvWorkerScore) => s.decodeTerm ?? s.decodeBlocks
  const vs = others.map((o) => `${ru(load(o), 0)} у ${o.name}`).join(', ')
  const explanation =
    `Decode-воркер выбирается только по нагрузке (overlap_score_credit = 0, prefill-токены не учитываются): у ${c.name} будет занято ` +
    `${ru(load(c), 0)} ${plural(load(c), BLOCKS)} KV (${ru(c.activeBlocks, 0)} ${plural(c.activeBlocks, ['активный', 'активных', 'активных'])} ` +
    `и ${ru(c.newBlocks, 0)} ${plural(c.newBlocks, BLOCKS)} вашего промпта) против ${vs}. KV промпта он заберёт у prefill-воркера через NIXL.`
  return { ...base, scores, chosen: workers[idx].id, explanation }
}

/** Воркеры, кэш, нагрузка и решения роутера. */
export function buildCluster(input: ClusterInput): ClusterBuild {
  const { tokenizer, model, gpu, tp, blocks, blockSize, deployment, warm } = input
  const capacityBlocks = Math.max(0, Math.floor(kvCapacityTokens(model, gpu, { gpus: tp, memUtil: SIM_GPU_MEMORY_UTILIZATION }) / blockSize))
  const fullBlocks = blocks.filter((b) => b.full)
  const cWarm = warmBlockCount(fullBlocks.length)
  const warmBlocks = warm ? cWarm : 0
  const notes: string[] = []

  const warmUp = (w: RouterWorker): RouterWorker => (warmBlocks > 0 ? cacheBlocks(w, fullBlocks.slice(0, warmBlocks)) : w)

  if (deployment === 'agg') {
    // Нагрузка Воркера 2 = нагрузка Воркера 3 + c/2 + 0,25 блока (в токенах очереди prefill).
    const w3 = AGG_SEEDS[2]
    const q2 = Math.round(blockSize * (w3.activeBlocks + w3.queued / blockSize - AGG_SEEDS[1].activeBlocks) + (blockSize / 2) * cWarm + blockSize / 4)
    let workers = AGG_SEEDS.map((s, i) =>
      makeWorker(`worker-${i}`, `Воркер ${i + 1}`, s, i === WARM_WORKER_INDEX ? q2 : s.queued, capacityBlocks, tokenizer, blockSize),
    )
    workers = workers.map((w, i) => (i === WARM_WORKER_INDEX ? warmUp(w) : w))
    const routing = routeKV(blocks, workers, { ...DEFAULT_ROUTER_OPTIONS, blockSize })
    const story = cacheStory(routing, blockSize, true, warm, warmBlocks)
    if (story) routing.explanation += ' ' + story
    const info: FlowClusterInfo = {
      deployment,
      tp,
      gpuCount: tp * workers.length,
      workers: workers.map((w, i) =>
        workerInfo(w, 'aggregated', { ...AGG_SEEDS[i], queued: w.queuedPrefillTokens ?? 0 }, tp, tokenizer, blockSize, blocks),
      ),
      warmBlocks,
      ...(warmBlocks > 0 ? { warmWorkerId: workers[WARM_WORKER_INDEX].id } : {}),
    }
    if (warm && warmBlocks === 0) notes.push('Промпт короче одного блока (16 токенов): тёплому кэшу нечего хранить.')
    return { workers, routing, info, notes }
  }

  const p3 = PREFILL_SEEDS[2]
  const q2 = Math.round(p3.queued + (blockSize / 2) * cWarm + blockSize / 4)
  let prefill = PREFILL_SEEDS.map((s, i) =>
    makeWorker(`prefill-${i}`, `Prefill ${i + 1}`, s, i === WARM_WORKER_INDEX ? q2 : s.queued, capacityBlocks, tokenizer, blockSize),
  )
  prefill = prefill.map((w, i) => (i === WARM_WORKER_INDEX ? warmUp(w) : w))
  const decode = DECODE_SEEDS.map((s, i) => makeWorker(`decode-${i}`, `Decode ${i + 1}`, s, 0, capacityBlocks, tokenizer, blockSize))
  const routing = routePrefillPool(blocks, prefill, blockSize)
  const story = cacheStory(routing, blockSize, false, warm, warmBlocks)
  if (story) routing.explanation += ' ' + story
  const decodeRouting = routeDecodePool(blocks, decode, blockSize)
  const info: FlowClusterInfo = {
    deployment,
    tp,
    gpuCount: tp * (prefill.length + decode.length),
    workers: [
      ...prefill.map((w, i) => workerInfo(w, 'prefill', { ...PREFILL_SEEDS[i], queued: w.queuedPrefillTokens ?? 0 }, tp, tokenizer, blockSize, blocks)),
      ...decode.map((w, i) => workerInfo(w, 'decode', DECODE_SEEDS[i], tp, tokenizer, blockSize, blocks)),
    ],
    warmBlocks,
    ...(warmBlocks > 0 ? { warmWorkerId: prefill[WARM_WORKER_INDEX].id } : {}),
  }
  if (warm && warmBlocks === 0) notes.push('Промпт короче одного блока (16 токенов): тёплому кэшу нечего хранить.')
  return { workers: prefill, routing, decodeWorkers: decode, decodeRouting, info, notes }
}
