/**
 * Модель главной фигуры главы 20: сценарии, начальные состояния воркеров, поток следующих запросов,
 * решение роутера и формула стоимости с подставленными числами. Чистые функции, всё считает
 * src/engine/router.ts (формула Dynamo ≥ 1.2), токены и блоки настоящие (prompt.ts).
 */
import type { Tokenizer } from '@/engine/types'
import { blockHashes, DEFAULT_BLOCK_SIZE, type BlockInfoExt } from '@/engine/kv'
import {
  applyRoute,
  releaseRoute,
  routeKV,
  routeRandom,
  routeRoundRobin,
  type KvRouteDecision,
  type KvRouterOptions,
  type KvWorkerScore,
  type RouterWorker,
} from '@/engine/router'
import { AGENT_TOOLS, OTHER_QUESTIONS, buildPrompt, type PromptParts } from './prompt'

export type Scenario = 'agent' | 'plain' | 'docs'
export type Strategy = 'kv' | 'round-robin' | 'random'

export interface RouterRequest {
  /** Номер запроса в потоке (0 — запрос читателя). */
  n: number
  /** Подпись: «Ваш запрос» или вопрос другого пользователя. */
  label: string
  /** Чей запрос: читатель, другой пользователь, синтетический из документации. */
  who: 'reader' | 'other' | 'docs'
  blocks: BlockInfoExt[]
  isl: number
  fullBlocks: number
  tailTokens: number
}

/** Воркер фигуры: состояние роутера + подпись роли для читателя. */
export interface DemoWorker extends RouterWorker {
  role: string
}

export const BLOCK = DEFAULT_BLOCK_SIZE

/** Системный промпт «чужого» приложения: его кэш у воркера никак не совпадёт с нашим запросом. */
const OTHER_APP: PromptParts = {
  systemText: 'Ты переводчик с английского на русский. Переводи точно и сохраняй форматирование.',
  userText: 'Translate: The scheduler admits requests while the KV cache has free blocks.',
}

function toRequest(n: number, label: string, who: RouterRequest['who'], blocks: BlockInfoExt[]): RouterRequest {
  const fullBlocks = blocks.filter((b) => b.full).length
  const isl = blocks.reduce((a, b) => a + b.tokenIds.length, 0)
  return { n, label, who, blocks, isl, fullBlocks, tailTokens: isl - fullBlocks * BLOCK }
}

/**
 * Системный промпт агента поддержки. Он свой и не берётся из симулятора: иначе длинный system prompt читателя
 * (вкладка «Длинный документ») раздувал бы общий префикс агента, и тёплый Воркер 1 выигрывал бы каждый запрос.
 */
export const AGENT_SYSTEM = 'Ты агент поддержки платформы инференса. Отвечай коротко и по делу.'

/**
 * Части промпта для сценария. Агент: свой системный промпт и описание инструментов (длинный общий префикс), из
 * симулятора только сообщение и thinking. «Ваш запрос»: промпт симулятора целиком.
 */
export function scenarioParts(scenario: Scenario, base: PromptParts, userText: string): PromptParts {
  if (scenario === 'agent') return { ...base, userText, systemText: AGENT_SYSTEM, tools: AGENT_TOOLS }
  return { ...base, userText }
}

/** Запрос читателя в выбранном сценарии (для 'docs' — синтетические 160 токенов из примера доков). */
export function readerRequest(scenario: Scenario, base: PromptParts, tokenizer: Tokenizer): RouterRequest {
  if (scenario === 'docs') return docsRequest(0)
  const p = buildPrompt(scenarioParts(scenario, base, base.userText), tokenizer)
  return toRequest(0, 'Ваш запрос', 'reader', p.blocks)
}

/** Пример документации Dynamo (notes-dynamo.md §3.3): 160 токенов = 10 полных блоков. */
function docsRequest(n: number): RouterRequest {
  const ids = Array.from({ length: 160 }, (_, i) => 100 + i)
  return toRequest(n, n === 0 ? 'Запрос из примера документации' : `Тот же запрос ещё раз (№${n + 1})`, 'docs', blockHashes(ids, BLOCK))
}

/** Следующий запрос потока: другой пользователь того же приложения (или тот же запрос доков). */
export function nextRequest(scenario: Scenario, n: number, base: PromptParts, tokenizer: Tokenizer): RouterRequest {
  if (scenario === 'docs') return docsRequest(n)
  const q = OTHER_QUESTIONS[(n - 1 + OTHER_QUESTIONS.length) % OTHER_QUESTIONS.length]
  const p = buildPrompt(scenarioParts(scenario, base, q), tokenizer)
  return toRequest(n, q, 'other', p.blocks)
}

const fullHashes = (blocks: readonly BlockInfoExt[]) => blocks.filter((b) => b.full).map((b) => b.hash)

function worker(i: number, role: string, cached: string[], activeBlocks: number, extra: Partial<RouterWorker> = {}): DemoWorker {
  return {
    id: `worker-${i}`,
    name: `Воркер ${i + 1}`,
    role,
    cachedBlocks: new Set(cached),
    activeBlocks,
    queuedPrefillTokens: 0,
    activeRequests: 0,
    activeBlockHashes: new Set<string>(),
    activeBlockRefs: new Map<string, number>(),
    ...extra,
  }
}

/** Активные блоки, которые держат уже идущие запросы воркера (ref count 1). */
function held(hashes: string[]): Partial<RouterWorker> {
  return { activeBlockHashes: new Set(hashes), activeBlockRefs: new Map(hashes.map((h) => [h, 1])) }
}

/**
 * Начальное состояние кластера (условные нагрузки, настоящие хэши):
 *  agent — W1 уже обслуживает этого агента и держит его префикс в активных запросах; W2 тоже видел агента,
 *          но занят prefill длинного документа; W3 обслуживал чужое приложение; W4 только что запущен.
 *          Нагрузка W1 подобрана так, чтобы при настройках по умолчанию он выигрывал первые три запроса,
 *          а четвёртый ушёл на пустой W4. Длинное сообщение читателя сдвигает смену раньше, поэтому подпись
 *          фигуры берёт номер запроса из agentSwitchAt.
 *  plain — W1 уже видел ровно этот запрос, W2 видел другой вопрос с тем же системным промптом. Нагрузка W1
 *          подобрана так, чтобы короткий промпт (до трёх полных блоков: короткие вопросы пресетов симулятора) ушёл
 *          на пустой W4, а длинный (вызов инструмента, длинный документ) остался у W1 ради кэша: так написано в главе.
 *  docs  — три воркера из примера документации (стоимости 18 / 10 / 11).
 */
export function initialWorkers(scenario: Scenario, base: PromptParts, tokenizer: Tokenizer): DemoWorker[] {
  if (scenario === 'docs') {
    const h = fullHashes(docsRequest(0).blocks)
    const w = (i: number, n: number, active: number): DemoWorker => ({
      ...worker(i, `в кэше ${n} из 10 блоков запроса`, h.slice(0, n), active, held(h.slice(0, n))),
      name: `W${i + 1}`,
    })
    return [w(0, 2, 2), w(1, 5, 0), w(2, 8, 7)]
  }
  const other = (k: number) => buildPrompt(scenarioParts(scenario, base, OTHER_QUESTIONS[k]), tokenizer).blocks
  const foreign = buildPrompt(OTHER_APP, tokenizer).blocks
  if (scenario === 'agent') {
    const a = fullHashes(other(0))
    const b = fullHashes(other(1))
    return [
      worker(0, 'обслуживает этого агента прямо сейчас', a, 67, { ...held(a), activeRequests: 3 }),
      worker(1, 'видел агента, но считает prefill длинного документа', b, 20, { queuedPrefillTokens: 1024, activeRequests: 1 }),
      worker(2, 'в кэше чужое приложение', fullHashes(foreign), 24, { activeRequests: 2 }),
      worker(3, 'только что запущен', [], 0),
    ]
  }
  const mine = fullHashes(readerRequest('plain', base, tokenizer).blocks)
  return [
    worker(0, 'уже отвечал ровно на этот запрос', mine, 4, { activeRequests: 1 }),
    worker(1, 'видел тот же системный промпт', fullHashes(other(2)), 6, { activeRequests: 1 }),
    worker(2, 'в кэше чужое приложение', fullHashes(foreign), 20, { activeRequests: 2 }),
    worker(3, 'только что запущен', [], 0),
  ]
}

export interface RouterKnobs {
  overlapCredit: number
  temperature: number
}

/** Решение выбранной стратегии; стоимости по формуле Dynamo считаются всегда (для сравнения). */
export function decide(strategy: Strategy, req: RouterRequest, workers: readonly DemoWorker[], knobs: RouterKnobs): KvRouteDecision {
  const opts: Partial<KvRouterOptions> = { overlapCredit: knobs.overlapCredit, temperature: knobs.temperature, blockSize: BLOCK, seed: 7 + req.n * 31 }
  if (strategy === 'round-robin') return routeRoundRobin(req.blocks, workers, req.n, opts)
  if (strategy === 'random') return routeRandom(req.blocks, workers, 1000 + req.n * 17, opts)
  return routeKV(req.blocks, workers, opts)
}

/**
 * Отправить запрос: выбранный воркер кладёт его полные блоки в кэш и берёт нагрузку (applyRoute).
 * Пока читатель жмёт кнопку, prefill этого запроса успевает закончиться, поэтому prefill-токены сразу
 * снимаются (releaseRoute 'prefill'), а активные decode-блоки остаются: ответы ещё генерируются.
 */
export function commit(
  workers: readonly DemoWorker[],
  decision: KvRouteDecision,
  req: RouterRequest,
  freshRole = 'был пустым, теперь держит этот префикс',
): DemoWorker[] {
  const applied = applyRoute(workers, decision, req.blocks) as DemoWorker[]
  const released = releaseRoute(applied, decision, 'prefill') as DemoWorker[]
  // пустой воркер, получивший первый запрос, больше не «только что запущен»: подпись роли меняется вместе с кэшем
  return released.map((w, i) => (w.id === decision.chosen && workers[i].cachedBlocks.size === 0 && w.cachedBlocks.size > 0 ? { ...w, role: freshRole } : w))
}

/**
 * Номер запроса (с 1), на котором KV-aware роутер при настройках по умолчанию (c = 1, T = 0) впервые отправляет
 * запрос агента на пустой Воркер 4, если читатель жмёт «Отправить запрос» подряд; 0, если за max запросов смены нет.
 */
export function agentSwitchAt(base: PromptParts, tokenizer: Tokenizer, max = 10): number {
  const knobs: RouterKnobs = { overlapCredit: 1, temperature: 0 }
  let workers = initialWorkers('agent', base, tokenizer)
  const fresh = workers[3].id
  let req = readerRequest('agent', base, tokenizer)
  for (let n = 0; n < max; n++) {
    const d = decide('kv', req, workers, knobs)
    if (d.chosen === fresh) return n + 1
    workers = commit(workers, d, req)
    req = nextRequest('agent', n + 1, base, tokenizer)
  }
  return 0
}

/* ---------------------------------------------------------------------------------------------
 * Формула в TeX с подставленными числами
 * ------------------------------------------------------------------------------------------- */

/** Число для TeX: десятичная запятая {,}, не больше двух знаков, без хвостовых нулей. */
export function texNum(x: number, digits = 2): string {
  const r = Math.round(x * 10 ** digits) / 10 ** digits
  const s = Number.isInteger(r) ? String(r) : r.toFixed(digits).replace(/0+$/, '')
  return s.replace('.', '{,}')
}

export const COST_TEX =
  '\\begin{aligned}\\text{cost} &= s\\cdot\\max\\!\\Big(0,\\ \\cprefill{\\tfrac{P + \\text{ISL}}{B} - c\\cdot \\text{overlap}}\\Big)\\\\ &\\quad + \\cdecode{\\big(D_{\\text{акт}} + D_{\\text{нов}}\\big)}\\end{aligned}'

/** Подстановка чисел выбранного воркера в формулу стоимости. */
export function costTexFor(s: KvWorkerScore, knobs: RouterKnobs, isl: number, B = BLOCK): string {
  const prefill = s.prefillTerm ?? 0
  const decodeT = s.decodeTerm ?? 0
  return (
    '\\begin{aligned}' +
    `&= 1\\cdot\\max\\!\\Big(0,\\ \\cprefill{\\tfrac{${texNum(s.queuedPrefillTokens)} + ${texNum(isl)}}{${B}} - ${texNum(knobs.overlapCredit)}\\cdot ${s.overlapBlocks}}\\Big)` +
    `\\\\ &\\quad + \\cdecode{(${texNum(s.activeBlocks)} + ${texNum(s.newBlocks)})}\\\\` +
    `&= ${texNum(prefill)} + ${texNum(decodeT)} = ${texNum(s.cost)}` +
    '\\end{aligned}'
  )
}

export const PROB_TEX =
  'p_i \\propto \\exp\\!\\Big(-\\frac{\\text{cost}_i - \\text{cost}_{\\min}}{T\\,(\\text{cost}_{\\max} - \\text{cost}_{\\min})}\\Big)'

/** Итог по уже отправленным запросам. */
export interface SentRecord {
  n: number
  label: string
  worker: string
  workerName: string
  isl: number
  cachedTokens: number
  strategy: Strategy
}

export function summarize(sent: readonly SentRecord[]) {
  const isl = sent.reduce((a, r) => a + r.isl, 0)
  const cached = sent.reduce((a, r) => a + r.cachedTokens, 0)
  return { count: sent.length, isl, cached, computed: isl - cached, hitRate: isl ? cached / isl : 0 }
}
