/**
 * Полный прогон запроса для симулятора «Путь запроса» (агент S1a): buildRun(settings, tokenizer) → FlowRun.
 *
 * Чистая детерминированная функция: никакого Math.random, Date.now и DOM. Склеивает настоящие модули движка:
 *   запрос OpenAI → renderTemplate (Qwen3) → tokenizer.encode → blockHashes (B = 16) → роутер Dynamo
 *   (agg: routeKV; disagg: PrefillRouter + decode по нагрузке) → prefillMs → NIXL (disagg) → шаги decode
 *   (decodeStepMs, sampling.distribution, structured.allowedMaskAfterTokens) → incrementalDetokenize →
 *   runStreamingParser → toSseChunks.
 * Симулирован только текст ответа модели (scenarios.ts) и синтетические логиты кандидатов (flow-candidates.ts),
 * плюс несколько маленьких условных констант времени (FLOW_TIMING: сеть, разбор HTTP, CPU-стадии).
 * Описание API и полей: src/engine/README.flow.md.
 */
import type {
  BlockInfo,
  Candidate,
  Channel,
  ChatMessage,
  DecodeCandidate,
  DecodeStep,
  FlowFrontendInfo,
  SegmentKind,
  FlowPlaybackStep,
  FlowPreprocessInfo,
  FlowRun,
  FlowSettings,
  FlowStageSummary,
  JsonSchema,
  StageId,
  TimelineItem,
  Token,
  Tokenizer,
  ToolCall,
} from './types'
import { findGpu, getGpu, type GpuSpecExt } from './data/gpus'
import { getModel, type ModelSpecExt } from './data/models'
import { getInterconnect } from './data/interconnects'
import { renderTemplate } from './templates'
import { blockHashes, DEFAULT_BLOCK_SIZE } from './kv'
import {
  decodeStepBreakdown,
  effectiveCachedTokens,
  fitsCheck,
  kvBytesForSeq,
  moeExpertsTouched,
  prefillBreakdown,
  stepTime,
  tpKvReplication,
  transferMs,
  weightBytes,
} from './perf'
import { distribution } from './sampling'
import { allowedMaskAfterTokens, grammarState } from './structured'
import { incrementalDetokenize, type DetokStepEx } from './detok'
import { runStreamingParser, toSseChunks, type ParserEventEx } from './parsers'
import { hashString, mulberry32, shortHash } from './rng'
import { QWEN3_STOP_TOKEN_IDS } from './tokenizer'
import { piecesText, scriptFor, WEATHER_TOOL, type AltSpec, type Script, type ScriptPiece } from './scenarios'
import { AGG_DECODE_BATCH, buildCluster, plural } from './flow-cluster'
import type { KvRouteDecision, RouterWorker } from './router'
import { buildCandidates, specialAlts, type RawCandidate } from './flow-candidates'

/* ============================================================================
 * Константы
 * ========================================================================== */

/** Модели симулятора: только семейство Qwen3, у них настоящие токенизатор и шаблон. */
export const SIM_MODEL_IDS: readonly string[] = ['qwen3-8b', 'qwen3-32b', 'qwen3-30b-a3b', 'qwen3-235b-a22b']
/** GPU симулятора (селектор: первые три, A100 для сравнения). */
export const SIM_GPU_IDS: readonly string[] = ['h100-sxm', 'h200', 'b200', 'a100-80g']
export const SIM_MODELS: readonly ModelSpecExt[] = SIM_MODEL_IDS.map((id) => getModel(id))
export const SIM_GPUS: readonly GpuSpecExt[] = SIM_GPU_IDS.map((id) => getGpu(id))
export const DEFAULT_SIM_MODEL_ID = 'qwen3-8b'
export const DEFAULT_SIM_GPU_ID = 'h100-sxm'

/** max_tokens запроса по умолчанию (ответы сценариев короче, 'length' видно только при меньшем лимите). */
export const DEFAULT_MAX_TOKENS = 512
/**
 * Рекомендованный сэмплинг из generation_config.json моделей Qwen3 (одинаков у 8B, 32B, 30B-A3B и 235B-A22B):
 * temperature 0,6, top_p 0,95, top_k 20. Запрос симулятора задаёт только temperature; frontend Dynamo передаёт незаданные поля
 * пустыми, а воркер vLLM подставляет их из generation_config.json модели (--generation-config auto по умолчанию с vLLM PR #12622).
 */
export const QWEN3_GENERATION_CONFIG = { temperature: 0.6, topP: 0.95, topK: 20 } as const
/** Размер блока KV: vLLM по умолчанию 16 (SPEC §8.8), хэшируются только полные блоки (§8.2). */
export const FLOW_BLOCK_SIZE = DEFAULT_BLOCK_SIZE
/** Варианты tensor parallel для автоподбора числа GPU на воркер. */
export const SIM_TP_OPTIONS: readonly number[] = [1, 2, 4, 8]

/**
 * (X-sim-7) Нагрузка в агрегированном режиме, условно: на воркер нашего запроса приходит длинный промпт соседа.
 * Chunked prefill (в vLLM включён по умолчанию) ставит его в те же шаги, что и decode, и эти шаги растягиваются:
 * так в agg скачет ITL. В disagg тот же промпт считает prefill-воркер, и шаги decode ровные.
 */
export const AGG_NEIGHBOR_PREFILL_TOKENS = 6000

/**
 * (X-sim-r2-4) Сосед приходит только в ответы от стольких токенов. В коротком ответе один долгий шаг соседа
 * задавал бы почти всё среднее ITL и E2E, и обычный шаг decode выглядел бы в разы медленнее, чем он есть.
 */
export const NEIGHBOR_MIN_COMPLETION_TOKENS = 40

/** Шаг decode, на котором приходит сосед: треть ответа, но не позже 12-го токена (виден в начале проигрывания). */
export function neighborArrivalStep(completionTokens: number): number {
  return Math.max(1, Math.min(12, Math.floor(completionTokens / 3)))
}

/**
 * max_num_batched_tokens у `vllm serve` по умолчанию (v0.32, notes-engines §2.3): B200 16384, H100 и H200 8192, остальные 2048.
 * Столько токенов помещается в один шаг; длинный prefill режется на куски этого размера.
 */
export function servingTokenBudget(gpuId: string): number {
  if (gpuId.startsWith('b200') || gpuId.startsWith('gb200')) return 16384
  if (gpuId.startsWith('h100') || gpuId.startsWith('h200')) return 8192
  return 2048
}
/** «Разумный бюджет KV» для автоподбора TP (условно): 8 одновременных последовательностей по 8k токенов. */
export const SIM_KV_REQUIREMENT = { contextLen: 8192, seqs: 8 } as const

/**
 * Условные константы времени (мс). Всё, что не из perf.ts, помечено simulated: true на таймлайне.
 * Порядок величин: Rust-frontend Dynamo, HF tokenizers, request plane TCP + msgpack, RDMA.
 */
export const FLOW_TIMING = {
  /** Сеть клиент ↔ frontend в одну сторону (клиент в том же регионе). */
  networkMs: 5,
  /** Разбор HTTP и JSON, проверка полей, параметры сэмплинга и стоп-условия. */
  frontendMs: 0.2,
  /** Рендер шаблона: база + на символ промпта. */
  templateBaseMs: 0.05,
  templatePerCharMs: 0.0002,
  /** Токенизация: база + на токен. */
  tokenizePerTokenMs: 0.002,
  tokenizeBaseMs: 0.03,
  /** Роутер: хэши блоков и поиск в индексе (база + на блок) и отправка на воркер по request plane. */
  routeBaseMs: 0.05,
  routePerBlockMs: 0.002,
  dispatchMs: 0.15,
  /** disagg: ответ prefill-воркера с disaggregated_params во frontend и выбор decode-воркера. */
  prefillReturnMs: 0.15,
  decodeRouteMs: 0.05,
  /** disagg: регистрация передачи NIXL (READ) и уведомление о завершении. */
  nixlSetupMs: 0.1,
  /** На каждый токен: доставка с воркера во frontend, детокенизация, парсеры, сборка SSE-чанка. */
  postprocessMs: 0.08,
} as const

/** Unix-время для поля created (2026-11-12 00:00:00 UTC) плюс детерминированный сдвиг. */
const CREATED_BASE = 1794441600
const EOS_ID = 151645
const THINK_OPEN = '<think>'
const THINK_CLOSE = '</think>'

/** Стадии пайплайна: подпись узла, что происходит (одна фраза) и главы для ссылки «Подробнее». */
export const STAGE_INFO: Readonly<Record<StageId, { title: string; what: string; chapters: readonly string[] }>> = {
  client: { title: 'Клиент', what: 'Отправляет POST /v1/chat/completions и читает SSE-поток.', chapters: ['request'] },
  frontend: {
    title: 'Frontend',
    what: 'OpenAI-совместимый HTTP-сервер: проверяет запрос, собирает параметры сэмплинга и стоп-условия.',
    chapters: ['request', 'dynamo', 'engines'],
  },
  preprocess: { title: 'Preprocessor', what: 'Собирает промпт по шаблону чата и превращает его в токены.', chapters: ['template', 'tokenization'] },
  route: { title: 'Router', what: 'Выбирает воркер по совпадению префикса в KV cache и по нагрузке.', chapters: ['routing', 'prefix-caching', 'kv-tiers'] },
  prefill: {
    title: 'Prefill',
    what: 'Прогоняет весь промпт через модель, заполняет KV cache и даёт логиты первого токена.',
    chapters: ['prefill-decode', 'attention', 'transformer', 'kv-cache', 'roofline'],
  },
  transfer: { title: 'NIXL', what: 'Переносит KV cache с prefill-воркера на decode-воркер напрямую GPU→GPU.', chapters: ['disaggregation'] },
  decode: {
    title: 'Decode',
    what: 'Один токен за шаг: логиты, маска, сэмплинг; KV cache растёт на токен.',
    chapters: ['prefill-decode', 'attention', 'sampling', 'structured-output', 'roofline', 'batching', 'speculative'],
  },
  postprocess: { title: 'Postprocessor', what: 'Детокенизация, стоп-условия, парсеры reasoning и tool calls.', chapters: ['detokenization', 'parsers'] },
  stream: { title: 'Стрим SSE', what: 'Чанки chat.completion.chunk по мере генерации, в конце usage и data: [DONE].', chapters: ['parsers', 'metrics'] },
}

/** Узлы схемы пайплайна по порядку: в agg prefill и decode на одном воркере, в disagg между ними NIXL. */
export function pipelineStages(deployment: FlowSettings['deployment']): StageId[] {
  return deployment === 'disagg'
    ? ['client', 'frontend', 'preprocess', 'route', 'prefill', 'transfer', 'decode', 'postprocess', 'stream']
    : ['client', 'frontend', 'preprocess', 'route', 'prefill', 'decode', 'postprocess', 'stream']
}

/**
 * FlowRun с точными типами модулей движка (совместим с FlowRun): решения роутера с подробностями (KvRouteDecision),
 * воркеры с учётом активных блоков (RouterWorker), шаги детокенизатора (DetokStepEx) и события парсера (ParserEventEx).
 */
export interface FlowRunEx extends FlowRun {
  routing: KvRouteDecision
  decodeRouting?: KvRouteDecision
  workers: RouterWorker[]
  decodeWorkers?: RouterWorker[]
  detok: DetokStepEx[]
  parser: ParserEventEx[]
  /** (G0) Шаги decode с разбивкой времени (чтение памяти и вычисления). */
  decode: DecodeStepEx[]
  /** (G0) Frontend с пометкой, какие поля сэмплинга взяты из generation_config.json модели. */
  frontend: FlowFrontendInfoEx
  /** (G0) Метрики плюс момент, когда до клиента дошёл первый чанк с ответом (content или tool_calls), а не с рассуждением. */
  metrics: FlowRun['metrics'] & { firstAnswerMs?: number }
  /** (G0) Литералы спецтокенов в тексте сообщений, которые токенизатор превратил в настоящие спецтокены. */
  injection: FlowInjection[]
  /**
   * (X-sim-7) Длинный промпт соседа на воркере decode. agg: applied, шаги steps растянуты (stepMs против plainMs).
   * disagg: не applied, его считает prefill-воркер; stepMs показывает, сколько длился бы такой шаг в agg.
   * Только в ответах от NEIGHBOR_MIN_COMPLETION_TOKENS токенов.
   */
  neighbor?: { promptTokens: number; atStep: number; steps: number[]; stepMs: number; plainMs: number; applied: boolean }
  /**
   * (X-sim-r2-1) Порог упора prefill в вычисления, токенов: с какой длины промпта холодный prefill на этой модели и GPU
   * упирается в вычисления, а не в чтение весов. Оценка по этому промпту без кэша (у холодного и тёплого прогона
   * одно число), до десятков. У плотных моделей сотни токенов, у MoE больше тысячи: токены промпта вместе задевают
   * почти всех экспертов, и prefill читает почти все веса, а считает только по активным экспертам.
   */
  prefillThreshold?: number
}

/** (G0) Шаг decode с разбивкой времени из perf.decodeStepBreakdown. */
export interface DecodeStepEx extends DecodeStep {
  /**
   * Из чего сложилось время шага: чтение весов и KV из HBM против вычислений (ms = max(memoryMs, computeMs) + overheadMs).
   * У первого токена в agg нет: он выбран в конце prefill (разбивка в run.prefill).
   */
  breakdown?: {
    memoryMs: number
    computeMs: number
    overheadMs: number
    /** Байт весов, прочитанных за шаг (у MoE только задетые эксперты), сумма по GPU воркера. */
    weightBytes: number
    /** Байт KV, прочитанных за шаг всеми последовательностями batch. */
    kvReadBytes: number
    bound: 'memory' | 'compute'
  }
  /** (X-sim-7) В этот шаг chunked prefill поставил кусок промпта соседа (только agg). plainMs: тот же шаг без него. */
  neighborPrefill?: { tokens: number; doneBefore: number; promptTokens: number; plainMs: number; budget: number }
}

/** (G0) FlowFrontendInfo с источником параметров сэмплинга. */
export interface FlowFrontendInfoEx extends FlowFrontendInfo {
  /** Поля сэмплинга, которых нет в запросе: frontend передаёт их пустыми, воркер vLLM берёт значения из generation_config.json. */
  samplingFromModel: Array<'top_p' | 'top_k'>
  /** Рекомендованные значения из generation_config.json модели. */
  generationConfig: { temperature: number; topP: number; topK: number }
}

/** (G0) Спецтокен, который появился в промпте из текста сообщения (а не из шаблона). */
export interface FlowInjection {
  /** Индекс в promptTokens. */
  index: number
  id: number
  text: string
  role: ChatMessage['role']
}

/** Необязательные параметры прогона. */
export interface BuildRunOptions {
  /** max_tokens запроса (по умолчанию DEFAULT_MAX_TOKENS). Меньше длины ответа → finish_reason 'length'. */
  maxTokens?: number
}

/* ============================================================================
 * Модель, GPU, TP
 * ========================================================================== */

/** Модель симулятора: id из SIM_MODEL_IDS, иначе Qwen3-8B (fallback = true). */
export function resolveSimModel(modelId: string): { model: ModelSpecExt; fallback: boolean } {
  return SIM_MODEL_IDS.includes(modelId) ? { model: getModel(modelId), fallback: false } : { model: getModel(DEFAULT_SIM_MODEL_ID), fallback: true }
}

/** GPU: любой из data/gpus (по id или алиасу), иначе H100 SXM (fallback = true). */
export function resolveSimGpu(gpuId: string): { gpu: GpuSpecExt; fallback: boolean } {
  const g = findGpu(gpuId)
  return g ? { gpu: g, fallback: false } : { gpu: getGpu(DEFAULT_SIM_GPU_ID), fallback: true }
}

/**
 * Сколько GPU на воркер: наименьший TP из {1, 2, 4, 8}, при котором влезают веса BF16 и бюджет KV
 * SIM_KV_REQUIREMENT (perf.fitsCheck). Если не влезает даже TP 8: tp = 8, fits = false.
 */
export function simTensorParallel(model: ModelSpecExt, gpu: GpuSpecExt): { tp: number; fits: boolean; message: string } {
  for (const tp of SIM_TP_OPTIONS) {
    const f = fitsCheck(model, gpu, { ...SIM_KV_REQUIREMENT, gpus: tp })
    if (f.fits) return { tp, fits: true, message: f.message }
  }
  const tp = SIM_TP_OPTIONS[SIM_TP_OPTIONS.length - 1]
  return { tp, fits: false, message: fitsCheck(model, gpu, { ...SIM_KV_REQUIREMENT, gpus: tp }).message }
}

/* ============================================================================
 * Форматирование для подписей
 * ========================================================================== */

function ruNum(x: number, digits = 2): string {
  return new Intl.NumberFormat('ru-RU', { maximumFractionDigits: digits }).format(x)
}
function ruMs(ms: number): string {
  if (ms >= 1000) return `${ruNum(ms / 1000, 2)} с`
  return `${ruNum(ms, ms < 1 ? 2 : ms < 10 ? 2 : 1)} мс`
}
const SIG3 = new Intl.NumberFormat('ru-RU', { maximumSignificantDigits: 3 })
/** Байты десятичными единицами, три значащие цифры: так же, как fmt.bytes в интерфейсе («6,34 МБ»). */
function ruBytes(b: number): string {
  const units = ['Б', 'КБ', 'МБ', 'ГБ', 'ТБ']
  let v = Math.max(0, b)
  let u = 0
  while (v >= 1000 && u < units.length - 1) {
    v /= 1000
    u++
  }
  if (u > 0 && Number(v.toPrecision(3)) >= 1000 && u < units.length - 1) {
    v /= 1000
    u++
  }
  return `${u === 0 ? Math.round(v) : SIG3.format(v)} ${units[u]}`
}
const TOKENS: readonly [string, string, string] = ['токен', 'токена', 'токенов']
const tokensWord = (n: number) => `${ruNum(n, 0)} ${plural(n, TOKENS)}`
const fullBlocksWord = (n: number) => `${ruNum(n, 0)} ${plural(n, ['полный блок', 'полных блока', 'полных блоков'])}`
const round2 = (x: number) => Math.round(x * 100) / 100

/* ============================================================================
 * Вспомогательное
 * ========================================================================== */

/** Нормализация текста для seed (тот же ответ при тех же словах). */
function seedOf(script: Script, settings: FlowSettings): number {
  const text = settings.userText.replace(/\s+/g, ' ').trim()
  return hashString(['flow', script.scenarioId, text, settings.thinking, settings.tools, settings.json].join('|'))
}

/** Индекс сегмента шаблона для каждого токена (по смещению начала токена). */
function segmentIndexOfTokens(segments: ReadonlyArray<{ text: string }>, tokens: readonly Token[]): number[] {
  const starts: number[] = []
  let off = 0
  for (const s of segments) {
    starts.push(off)
    off += s.text.length
  }
  let fallback = 0
  return tokens.map((t) => {
    const at = t.start ?? fallback
    fallback = (t.end ?? at + t.text.length)
    let lo = 0
    let hi = starts.length - 1
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1
      if (starts[mid] <= at) lo = mid
      else hi = mid - 1
    }
    return lo
  })
}

/**
 * Сегменты-спецтокены, которые пришли из текста сообщения, а не из шаблона. Шаблон Qwen3 открывает сообщение
 * своим <|im_start|> и закрывает своим <|im_end|>; описание tools (с <tool_call> в инструкции) идёт после текста
 * system. Всё special между ними и до блока tools взялось из литералов в тексте сообщения.
 */
function injectedSegments(segments: ReadonlyArray<{ kind: SegmentKind; messageIndex?: number }>): Set<number> {
  const byMsg = new Map<number, number[]>()
  segments.forEach((s, i) => {
    if (s.messageIndex === undefined) return
    const list = byMsg.get(s.messageIndex)
    if (list) list.push(i)
    else byMsg.set(s.messageIndex, [i])
  })
  const out = new Set<number>()
  for (const idxs of byMsg.values()) {
    const specials = idxs.filter((i) => segments[i].kind === 'special')
    const first = idxs[0]
    const lastSpecial = specials[specials.length - 1]
    const firstTools = idxs.find((i) => segments[i].kind === 'tools') ?? Number.POSITIVE_INFINITY
    for (const i of specials) if (i !== first && i !== lastSpecial && i < firstTools) out.add(i)
  }
  return out
}

/** Отрезок выходного текста и его свойства. */
interface OutRange {
  start: number
  end: number
  channel: Channel
  /** JSON-разметка внутри (тело structured output или аргументы вызова). */
  inJson: boolean
  /** Грамматика structured output активна. */
  grammar: boolean
}

interface StepPlan {
  alts?: AltSpec[]
  confidence?: ScriptPiece['confidence']
  note?: string
}

function cloneJson<T>(x: T): T {
  return JSON.parse(JSON.stringify(x)) as T
}

/* ============================================================================
 * buildRun
 * ========================================================================== */

/**
 * Полный прогон запроса: всё, что показывает симулятор, на настоящих данных (кроме текста ответа модели).
 * Детерминирован (seed из сценария и текста), обычный прогон занимает единицы миллисекунд.
 */
export function buildRun(settings: FlowSettings, tokenizer: Tokenizer, opts: BuildRunOptions = {}): FlowRunEx {
  const maxTokens = Math.max(1, Math.floor(opts.maxTokens ?? DEFAULT_MAX_TOKENS))
  const { model, fallback: modelFallback } = resolveSimModel(settings.modelId)
  const { gpu, fallback: gpuFallback } = resolveSimGpu(settings.gpuId)
  const tpPick = simTensorParallel(model, gpu)
  const tp = tpPick.tp
  const B = FLOW_BLOCK_SIZE
  const T = FLOW_TIMING
  const notes: string[] = []
  if (modelFallback) {
    notes.push(`Модель «${settings.modelId}» не из семейства Qwen3: симулятор взял ${model.name}, потому что настоящие токенизатор и шаблон здесь только у Qwen3.`)
  }
  if (gpuFallback) notes.push(`GPU «${settings.gpuId}» нет в данных: взят ${gpu.name}.`)
  if (!tpPick.fits) notes.push(`Даже на 8 GPU не хватает памяти: ${tpPick.message}`)

  const script = scriptFor(settings)
  const seed = seedOf(script, settings)
  const schema: JsonSchema | undefined = settings.json ? script.json?.schema : undefined

  /* ---------- 1. запрос ---------- */
  const messages: ChatMessage[] = []
  if (settings.systemText.trim()) messages.push({ role: 'system', content: settings.systemText })
  messages.push({ role: 'user', content: settings.userText })
  const tools = settings.tools ? [cloneJson(WEATHER_TOOL)] : undefined
  const request: Record<string, unknown> = {
    model: model.hfId,
    messages: messages.map((m) => ({ role: m.role, content: m.content })),
  }
  if (tools) {
    request.tools = tools
    request.tool_choice = 'auto'
  }
  if (schema && script.json) {
    request.response_format = { type: 'json_schema', json_schema: { name: script.json.name, schema: cloneJson(schema), strict: true } }
  }
  request.temperature = settings.temperature
  request.max_tokens = maxTokens
  request.stream = true
  request.stream_options = { include_usage: true }
  request.chat_template_kwargs = { enable_thinking: settings.thinking }

  /* ---------- 2. шаблон и токены ---------- */
  const tplInput = { family: 'qwen3' as const, messages, enableThinking: settings.thinking, addGenerationPrompt: true }
  const template = renderTemplate(tools ? { ...tplInput, tools } : tplInput)
  const promptTokens = tokenizer.encode(template.text, { allowSpecial: true })
  const promptIds = promptTokens.map((t) => t.id)
  const total = promptTokens.length
  let toolsTokens = 0
  if (tools) {
    const bare = renderTemplate(tplInput)
    toolsTokens = total - tokenizer.encode(bare.text, { allowSpecial: true }).length
  }
  const segmentOfToken = segmentIndexOfTokens(template.segments, promptTokens)

  // Инъекция спецтокенов: литералы вроде «<|im_end|>» в тексте сообщения токенизатор (как HF по умолчанию,
  // split_special_tokens=False) превращает в настоящие управляющие токены.
  const injectedSegs = injectedSegments(template.segments)
  const injection: FlowInjection[] = []
  promptTokens.forEach((tok, i) => {
    const si = segmentOfToken[i]
    if (!tok.special || !injectedSegs.has(si)) return
    const mi = template.segments[si].messageIndex
    injection.push({ index: i, id: tok.id, text: tok.text, role: mi !== undefined ? (messages[mi]?.role ?? 'user') : 'user' })
  })
  if (injection.length > 0) {
    const list = [...new Set(injection.map((x) => x.text))].join(', ')
    notes.push(
      `В тексте сообщения есть литералы служебных токенов (${list}). Токенизатор с настройками HF по умолчанию превратил их в настоящие токены: ` +
        'модель увидит конец реплики, чужую роль или маркер рассуждения и вызова там, где их не было. Это инъекция спецтокенов. ' +
        'split_special_tokens=True обезвреживает только special-токены вроде <|im_start|> и <|im_end|>, а <think> и <tool_call> у Qwen3 не special ' +
        'и распознаются всё равно, поэтому текст пользователя нужно чистить.',
    )
  }
  notes.push(...script.notes)

  /* ---------- 3. блоки и хэши ---------- */
  const blocks = blockHashes(promptIds, B)

  /* ---------- 4. кластер и роутинг ---------- */
  const cluster = buildCluster({ tokenizer, model, gpu, tp, blocks, blockSize: B, deployment: settings.deployment, warm: settings.cacheWarm })
  notes.push(...cluster.notes)
  const disagg = settings.deployment === 'disagg'
  const pScore = cluster.routing.scores.find((s) => s.workerId === cluster.routing.chosen)!
  const prefillWorker = cluster.workers.find((w) => w.id === cluster.routing.chosen)!
  const decodeWorker = disagg ? cluster.decodeWorkers!.find((w) => w.id === cluster.decodeRouting!.chosen)! : prefillWorker

  /* ---------- 5. prefill ---------- */
  const cachedTokens = effectiveCachedTokens(total, pScore.overlapBlocks * B)
  const computedTokens = total - cachedTokens
  const pb = prefillBreakdown(model, gpu, total, { cachedTokens, gpus: tp })
  // порог упора в вычисления: время вычислений растёт с числом токенов, чтение весов почти нет (у MoE растёт до всех весов)
  const pbCold = cachedTokens > 0 ? prefillBreakdown(model, gpu, total, { cachedTokens: 0, gpus: tp }) : pb
  const prefillThreshold = pbCold.computeMs > 0 && total > 0 ? Math.round((total * pbCold.memoryMs) / pbCold.computeMs / 10) * 10 : 0
  const prefill: FlowRun['prefill'] = {
    totalTokens: total,
    cachedTokens,
    computedTokens,
    ms: pb.ms,
    worker: prefillWorker.id,
    kvBytes: kvBytesForSeq(model, computedTokens),
    workerName: prefillWorker.name,
    cachedBlocks: pScore.overlapBlocks,
    kvTotalBytes: kvBytesForSeq(model, total),
    layers: model.layers,
    layerMs: pb.ms / model.layers,
    bound: pb.bound,
    computeMs: pb.computeMs,
    memoryMs: pb.memoryMs,
    flops: pb.flops,
  }

  /* ---------- 6. текст ответа и его токены ---------- */
  const ranges: OutRange[] = []
  const marks: Array<{ offset: number; piece: ScriptPiece }> = []
  let outText = ''
  const pushPieces = (pieces: readonly ScriptPiece[]) => {
    for (const piece of pieces) {
      if (piece.alts || piece.confidence) marks.push({ offset: outText.length, piece })
      outText += piece.text
    }
  }
  const bodyJson = script.bodyKind === 'json'
  if (script.reasoning) {
    outText += THINK_OPEN + '\n'
    pushPieces(script.reasoning)
    outText += '\n' + THINK_CLOSE
    ranges.push({ start: 0, end: outText.length, channel: 'reasoning', inJson: false, grammar: false })
    // Обычно после </think> идёт «\n\n»; под маской JSON модель сразу пишет «{».
    if (!bodyJson) {
      const s = outText.length
      outText += '\n\n'
      ranges.push({ start: s, end: outText.length, channel: 'content', inJson: false, grammar: false })
    }
  }
  const bodyStart = outText.length
  pushPieces(script.body)
  ranges.push({
    start: bodyStart,
    end: outText.length,
    channel: script.bodyKind === 'tool_call' ? 'tool_call' : 'content',
    inJson: script.bodyKind !== 'text',
    grammar: bodyJson && !!schema,
  })
  const eosStart = outText.length
  outText += '<|im_end|>'
  ranges.push({ start: eosStart, end: outText.length, channel: 'content', inJson: false, grammar: bodyJson && !!schema })

  const encodedOut = tokenizer.encode(outText, { allowSpecial: true })
  const truncated = encodedOut.length > maxTokens
  const outTokens = truncated ? encodedOut.slice(0, maxTokens) : encodedOut
  const N = outTokens.length
  if (truncated) notes.push(`Ответ длиннее max_tokens = ${maxTokens}: генерация оборвана на лимите, finish_reason = length.`)

  // Позиции токенов в тексте ответа (у Qwen3 encode всегда даёт start/end; запасной путь по длине текста).
  const starts: number[] = []
  {
    let off = 0
    for (const t of outTokens) {
      const s = t.start ?? off
      starts.push(s)
      off = t.end ?? s + t.text.length
    }
  }
  const rangeOf = (i: number): OutRange => {
    const at = starts[i]
    return ranges.find((r) => at >= r.start && at < r.end) ?? ranges[ranges.length - 1]
  }
  const tokenAt = (offset: number): number => {
    for (let i = 0; i < N; i++) {
      const end = i + 1 < N ? starts[i + 1] : outText.length
      if (offset >= starts[i] && offset < end) return i
    }
    return -1
  }

  // План шагов: ручные альтернативы из скрипта, служебные токены, пояснения.
  const plan: StepPlan[] = outTokens.map(() => ({}))
  for (const m of marks) {
    const i = tokenAt(m.offset)
    if (i < 0 || plan[i].alts || plan[i].confidence) continue
    if (m.piece.alts) plan[i].alts = m.piece.alts
    if (m.piece.confidence) plan[i].confidence = m.piece.confidence
  }
  const grammarStart = schema ? outTokens.findIndex((_, i) => rangeOf(i).grammar) : -1
  for (let i = 0; i < N; i++) {
    const tok = outTokens[i]
    const prev = outTokens[i - 1]
    if (tok.special && !plan[i].alts) {
      const alts = specialAlts(tok, { nextText: outText.slice(bodyStart, bodyStart + 40), jsonBody: bodyJson })
      if (alts) plan[i].alts = alts
      plan[i].confidence = 'high'
    }
    // «\n» сразу после <think> и «\n\n» сразу после </think>: модель почти уверена.
    if (prev?.text === THINK_OPEN || prev?.text === THINK_CLOSE) plan[i].confidence ??= 'high'
    if (tok.text === THINK_CLOSE) plan[i].note = 'Конец рассуждения: дальше парсер шлёт текст в content.'
    if (tok.text === '<tool_call>') plan[i].note = 'Начало вызова инструмента: парсер копит JSON и не отправляет его как content.'
    if (tok.id === EOS_ID) {
      plan[i].note =
        script.bodyKind === 'tool_call'
          ? 'Стоп-токен <|im_end|>: генерация закончена; был вызов инструмента, поэтому finish_reason = tool_calls.'
          : 'Стоп-токен <|im_end|>: генерация закончена, finish_reason = stop.'
    }
  }
  if (grammarStart >= 0) {
    const afterThink = outTokens[grammarStart - 1]?.text === THINK_CLOSE
    const extra: AltSpec[] = afterThink ? [{ text: '\n\n', preview: 'как обычно после </think>', delta: 1.2 }] : []
    plan[grammarStart].alts = [...extra, ...(plan[grammarStart].alts ?? [])]
    plan[grammarStart].note = afterThink
      ? 'Рассуждение закончилось, включилась грамматика JSON: «\\n\\n» после </think> и болтовню маска отрезает, остаётся начало документа.'
      : 'Первый шаг грамматики JSON: маска оставляет только токены, с которых может начаться документ по схеме.'
  }

  // Слова из ответа и промпта: правдоподобные замены в начале слова.
  const context: Token[] = []
  {
    const seen = new Set<number>()
    for (const t of [...outTokens, ...promptTokens]) {
      if (t.special || seen.has(t.id) || !/^ \p{L}{2,}/u.test(t.text)) continue
      seen.add(t.id)
      context.push(tokenizer.token(t.id))
    }
  }

  // Язык ответа: по числу кириллических и латинских букв (латинские термины в русском ответе получают русские замены).
  const cyr = (outText.match(/[А-Яа-яЁё]/g) ?? []).length
  const lat = (outText.match(/[A-Za-z]/g) ?? []).length
  const lang: 'ru' | 'en' = cyr >= lat ? 'ru' : 'en'

  /* ---------- 7. таймлайн до первого токена ---------- */
  const timeline: TimelineItem[] = []
  const add = (stage: StageId, startMs: number, endMs: number, label: string, extra: Partial<TimelineItem> = {}) => {
    timeline.push({ stage, startMs, endMs, label, ...extra })
  }
  let t = 0
  add('client', t, t + T.networkMs, 'Запрос летит к серверу', {
    simulated: true,
    detail: `HTTP POST /v1/chat/completions; сеть условно ${ruMs(T.networkMs)} в одну сторону.`,
  })
  t += T.networkMs
  add('frontend', t, t + T.frontendMs, 'Frontend принимает запрос', {
    simulated: true,
    detail: 'Разбор HTTP и JSON, проверка полей, параметры сэмплинга и стоп-условия.',
  })
  t += T.frontendMs
  const templateMs = T.templateBaseMs + T.templatePerCharMs * template.text.length
  const tokenizeMs = T.tokenizeBaseMs + T.tokenizePerTokenMs * total
  add('preprocess', t, t + templateMs + tokenizeMs, 'Шаблон и токенизация', {
    simulated: true,
    detail: `Шаблон Qwen3 и токенизатор: ${tokensWord(total)}, ${fullBlocksWord(blocks.filter((b) => b.full).length)} по ${B}.`,
  })
  t += templateMs + tokenizeMs
  const routeMs = T.routeBaseMs + T.routePerBlockMs * blocks.length + T.dispatchMs
  add('route', t, t + routeMs, disagg ? `PrefillRouter: ${prefillWorker.name}` : `KV-aware роутер: ${prefillWorker.name}`, {
    simulated: true,
    detail: disagg
      ? 'Хэши блоков, поиск префикса, стоимость prefill-воркеров; запрос-клон с max_tokens = 1 уходит на prefill.'
      : 'Хэши блоков, поиск префикса в индексе, стоимость каждого воркера; запрос уходит на воркер.',
  })
  t += routeMs
  const prefillStart = t
  add('prefill', t, t + prefill.ms, disagg ? `Prefill-воркер ${prefillWorker.name.replace(/^\D+/, '')}` : `Prefill: ${prefillWorker.name}`, {
    detail: `Посчитано ${tokensWord(computedTokens)}, из кэша ${tokensWord(cachedTokens)}; ${pb.bound === 'memory' ? 'упирается в чтение весов' : 'упирается в вычисления'}.`,
  })
  t += prefill.ms

  let transfer: FlowRun['transfer']
  if (disagg) {
    const r2 = T.prefillReturnMs + T.decodeRouteMs
    add('route', t, t + r2, `Decode-воркер: ${decodeWorker.name}`, {
      simulated: true,
      detail: 'Prefill-воркер вернул метаданные KV во frontend; decode-воркер выбран по нагрузке.',
    })
    t += r2
    const link = getInterconnect('ib-ndr')
    const rep = tpKvReplication(model, tp)
    const bytes = kvBytesForSeq(model, total) * rep
    const links = tp
    const ms = T.nixlSetupMs + transferMs(bytes / links, link.GBs)
    const nBlocks = Math.ceil(total / B)
    const firstBlock = 17 + (seed % 400)
    transfer = {
      bytes,
      ms,
      link: link.name,
      linkId: link.id,
      GBs: link.GBs,
      links,
      setupMs: T.nixlSetupMs,
      from: prefillWorker.id,
      to: decodeWorker.id,
      fromName: prefillWorker.name,
      toName: decodeWorker.name,
      params: {
        do_remote_prefill: true,
        do_remote_decode: false,
        remote_block_ids: Array.from({ length: nBlocks }, (_, i) => firstBlock + i),
        remote_engine_id: shortHash(`${prefillWorker.id}|engine`),
        remote_request_id: `req-${shortHash(seed)}`,
        remote_host: `10.0.3.${11 + cluster.workers.indexOf(prefillWorker)}`,
        remote_port: 5600,
        tp_size: tp,
      },
    }
    add('transfer', t, t + ms, `NIXL: KV ${ruBytes(bytes)} → ${decodeWorker.name}`, {
      detail: `RDMA READ по ${links} ${plural(links, ['каналу', 'каналам', 'каналам'])} ${link.name} (${ruNum(link.GBs, 0)} ГБ/с каждый) + ${ruMs(T.nixlSetupMs)} на регистрацию и уведомление.`,
    })
    t += ms
    notes.push(
      'На prefill ушёл клон запроса с max_tokens = 1: prefill-воркер тоже выбрал токен, но в связке Dynamo и vLLM он отброшен, первый токен выбирает decode-воркер.',
    )
    notes.push('Цена дезагрегации: второй шаг роутера, передача KV и лишний шаг decode, поэтому TTFT выше, чем в агрегированном режиме.')
    if (model.experts) {
      const b = (decodeWorker.activeRequests ?? 0) + 1
      const per = (n: number) => Math.round(moeExpertsTouched(model, n))
      notes.push(
        `У MoE шаг decode дорожает с ростом batch: каждый запрос идёт к своим ${model.experts.active} из ${model.experts.total} экспертов, ` +
          `и batch из ${b} запросов задевает около ${per(b)} экспертов на слой, а batch из ${AGG_DECODE_BATCH} около ${per(AGG_DECODE_BATCH)}. ` +
          'Поэтому ITL здесь выше, чем в агрегированном режиме.',
      )
    }
  }

  /* ---------- 8. шаги decode ---------- */
  const temperature = settings.temperature
  const batch = (decodeWorker.activeRequests ?? 0) + 1
  const othersBase = decodeWorker.activeBlocks * B
  const decode: DecodeStepEx[] = []
  let decodeMs = 0
  // (X-sim-7) длинный промпт соседа: в agg его куски встают в шаги decode этого воркера, в disagg его считает prefill-воркер
  const nbBudget = servingTokenBudget(gpu.id)
  const nbAt = neighborArrivalStep(N)
  const nbOn = N >= NEIGHBOR_MIN_COMPLETION_TOKENS && N > nbAt + 1
  let nbDone = 0
  const nbSteps: number[] = []
  let nbStepMs = 0
  let nbPlainMs = 0
  for (let k = 0; k < N; k++) {
    const tok = outTokens[k]
    const range = rangeOf(k)
    let startMs: number
    let durationMs: number
    let phase: 'prefill' | 'decode'
    let breakdown: DecodeStepEx['breakdown']
    let neighborStep: DecodeStepEx['neighborPrefill']
    if (!disagg && k === 0) {
      // agg: первый токен выбирается по логитам последней позиции промпта в конце prefill
      phase = 'prefill'
      startMs = prefillStart
      durationMs = prefill.ms
    } else {
      phase = 'decode'
      const ours = total + k
      const others = othersBase + (batch - 1) * k
      const avgContext = (others + ours) / batch
      let bd = decodeStepBreakdown(model, gpu, { batch, avgContext, gpus: tp })
      if (nbOn && k >= nbAt && nbDone < AGG_NEIGHBOR_PREFILL_TOKENS) {
        // кусок prefill соседа в том же шаге: бюджет шага минус decode-токены batch
        const chunk = Math.min(AGG_NEIGHBOR_PREFILL_TOKENS - nbDone, Math.max(1, nbBudget - batch))
        const mixed = stepTime(
          model,
          gpu,
          { prefill: [{ tokens: chunk, context: nbDone, last: nbDone + chunk >= AGG_NEIGHBOR_PREFILL_TOKENS }], decodeUniform: { batch, context: avgContext } },
          { gpus: tp },
        )
        if (!disagg) {
          neighborStep = { tokens: chunk, doneBefore: nbDone, promptTokens: AGG_NEIGHBOR_PREFILL_TOKENS, plainMs: bd.ms, budget: nbBudget }
          nbSteps.push(k)
          nbStepMs = Math.max(nbStepMs, mixed.ms)
          nbPlainMs = Math.max(nbPlainMs, bd.ms)
          bd = mixed
        } else if (nbStepMs === 0) {
          // disagg: шаг ровный; запоминаем, сколько длился бы такой шаг, если бы prefill соседа пришёл сюда
          nbStepMs = mixed.ms
          nbPlainMs = bd.ms
        }
        nbDone += chunk
      }
      durationMs = bd.ms
      breakdown = {
        memoryMs: bd.memoryMs,
        computeMs: bd.computeMs,
        overheadMs: bd.overheadMs,
        weightBytes: bd.weightBytes,
        kvReadBytes: bd.kvBytes,
        bound: bd.bound,
      }
      startMs = t
      t += durationMs
      decodeMs += durationMs
    }
    const producedMs = startMs + durationMs
    const emitMs = producedMs + T.postprocessMs
    const clientMs = emitMs + T.networkMs
    const p = plan[k]
    if (phase === 'decode') {
      add('decode', startMs, producedMs, neighborStep ? `Токен ${k + 1} + prefill соседа` : `Токен ${k + 1}`, {
        step: k,
        detail: neighborStep
          ? `Шаг decode, batch ${batch}, и в нём же chunked prefill: ${ruNum(neighborStep.tokens, 0)} ${plural(neighborStep.tokens, ['токен', 'токена', 'токенов'])} промпта соседа (условно). Шаг ${ruMs(durationMs)} вместо ${ruMs(neighborStep.plainMs)}.`
          : `Шаг decode, batch ${batch}: наш запрос и ещё ${batch - 1} ${plural(batch - 1, ['запрос', 'запроса', 'запросов'])} этого воркера.`,
      })
    }
    add('postprocess', producedMs, emitMs, 'Детокенизация и парсеры', { step: k, simulated: true })
    add('stream', emitMs, clientMs, 'SSE-чанк к клиенту', { step: k, simulated: true })

    // Кандидаты
    const rng = mulberry32((hashString(`${seed}|${k}|${tok.id}`) ^ 0x9e3779b9) >>> 0)
    const raw: RawCandidate[] = buildCandidates({
      tokenizer,
      chosen: tokenizer.token(tok.id),
      prev: outTokens[k - 1],
      next: outTokens[k + 1],
      inJson: range.inJson,
      curated: p.alts,
      confidence: p.confidence,
      context,
      lang,
      rng,
    })
    for (const c of raw) c.logit = round2(c.logit)
    let allowed: boolean[] = raw.map(() => true)
    let grammarExpecting: string | undefined
    if (range.grammar && schema && grammarStart >= 0) {
      const prefix = outTokens.slice(grammarStart, k)
      allowed = allowedMaskAfterTokens(
        schema,
        prefix,
        raw.map((c) => c.token),
        { whitespace: 'xgrammar' },
      )
      grammarExpecting = grammarState(schema, tokenizer.decode(prefix.map((x) => x.id)), { whitespace: 'xgrammar' }).expecting
    }
    // Выбранный (заскриптованный) токен должен быть самым вероятным среди разрешённых.
    for (let i = 1; i < raw.length; i++) {
      if (allowed[i] && raw[i].logit >= raw[0].logit) raw[i].logit = round2(raw[0].logit - 0.3 - 0.15 * i)
    }
    const order = raw.map((_, i) => i).sort((a, b) => raw[b].logit - raw[a].logit || a - b)
    const sorted = order.map((i) => raw[i])
    const sortedAllowed = order.map((i) => allowed[i])
    const cands: Candidate[] = sorted.map((c) => ({ token: c.token, logit: c.logit }))
    const d = distribution(cands, { temperature, topK: 0, topP: 1 }, sortedAllowed)
    const candidates: DecodeCandidate[] = sorted.map((c, i) => {
      const dc: DecodeCandidate = { token: c.token, logit: c.logit, prob: d.finalProbs[i], allowed: sortedAllowed[i], rawProb: d.probs[i] }
      if (c.preview) dc.preview = c.preview
      return dc
    })
    const chosenIndex = order.indexOf(0)
    const kvTokens = total + k
    const step: DecodeStepEx = {
      step: k,
      token: tok,
      channel: range.channel,
      candidates,
      masked: range.grammar,
      startMs,
      durationMs,
      phase,
      chosenIndex,
      entropyBits: d.entropyBits,
      maskedCount: sortedAllowed.filter((a) => !a).length,
      kvTokens,
      kvBytes: kvBytesForSeq(model, kvTokens),
      batch: phase === 'decode' ? batch : 1,
      emitMs,
      clientMs,
    }
    if (grammarExpecting !== undefined) step.grammarExpecting = grammarExpecting
    if (breakdown) step.breakdown = breakdown
    if (neighborStep) step.neighborPrefill = neighborStep
    if (tok.id === EOS_ID) step.stop = true
    let note = p.note
    if (k === 0) {
      note = disagg
        ? 'Decode-воркер получил KV через NIXL, пересчитал последний токен промпта и сам выбрал первый токен (так делает vLLM).'
        : 'Первый токен выбран в конце prefill: логиты последней позиции промпта.'
      if (p.note) note += ' ' + p.note
    }
    if (note) step.note = note
    decode.push(step)
  }

  // (X-sim-7) что сделал длинный промпт соседа: в agg растянул шаги decode, в disagg ушёл на prefill-воркер
  let neighbor: FlowRunEx['neighbor']
  if (nbOn && nbStepMs > 0) {
    const nb = ruNum(AGG_NEIGHBOR_PREFILL_TOKENS, 0)
    neighbor = { promptTokens: AGG_NEIGHBOR_PREFILL_TOKENS, atStep: nbAt, steps: nbSteps, stepMs: nbStepMs, plainMs: nbPlainMs, applied: !disagg }
    if (!disagg) {
      const n = nbSteps.length
      notes.push(
        `Под нагрузкой (условно): на шаге ${nbAt + 1} на этот же воркер пришёл промпт соседа на ${nb} токенов. ` +
          `Chunked prefill поставил его в ${n === 1 ? 'тот же шаг' : `${n} ${plural(n, ['шаг', 'шага', 'шагов'])}`}, что и decode вашего запроса, ` +
          `и шаг длился ${ruMs(nbStepMs)} вместо ${ruMs(nbPlainMs)}: так в агрегированном режиме скачет ITL. ` +
          'В дезагрегированном режиме этот prefill считает prefill-воркер, а шаги decode остаются ровными.',
      )
    } else {
      notes.push(
        `Выигрыш дезагрегации: длинный промпт соседа (условно ${nb} токенов) считает prefill-воркер, и шаги decode здесь ровные, около ${ruMs(nbPlainMs)}. ` +
          `В агрегированном режиме chunked prefill поставил бы его в шаг decode, и тот длился бы около ${ruMs(nbStepMs)}.`,
      )
    }
  }

  /* ---------- 9. детокенизация, парсеры, SSE ---------- */
  const detok: DetokStepEx[] = incrementalDetokenize(outTokens, tokenizer, {
    skipSpecial: true,
    promptTokens,
    stopTokenIds: [...QWEN3_STOP_TOKEN_IDS],
  })
  const parser: ParserEventEx[] = runStreamingParser(
    detok.map((s) => s.emitted),
    { reasoning: 'qwen3', tools: settings.tools ? 'hermes' : 'none', idSeed: seed.toString(16) },
  )
  const toolCalls: ToolCall[] = []
  let reasoningOut = ''
  let contentOut = ''
  for (const ev of parser) {
    for (const dlt of ev.deltas) {
      if (dlt.channel === 'reasoning') reasoningOut += dlt.text
      else if (dlt.channel === 'content') contentOut += dlt.text
    }
    if (ev.toolCalls) toolCalls.push(...ev.toolCalls)
    else if (ev.toolCall) toolCalls.push(ev.toolCall)
  }
  const anyToolDelta = parser.some((ev) => ev.toolDeltas.length > 0)
  const finishReason: FlowRun['metrics']['finishReason'] = anyToolDelta || toolCalls.length > 0 ? 'tool_calls' : truncated ? 'length' : 'stop'
  const completionTokens = N
  const sse = toSseChunks(parser, {
    id: `chatcmpl-${shortHash(`${seed}|id`)}`,
    created: CREATED_BASE + (seed % 86400),
    model: model.hfId,
    finishReason: truncated ? 'length' : 'stop',
    includeUsage: true,
    usage: { promptTokens: total, completionTokens, cachedTokens },
    reasoningField: 'reasoning_content',
  })

  // Какой шаг отправил каждый SSE-чанк: роль приходит с первым токеном, финал после последнего (шаг N).
  const sseStep: number[] = [0]
  for (const ev of parser) {
    const n = ev.out.filter((it) => it.kind === 'tool' || it.text.length > 0).length
    const s = Math.min(ev.step, N - 1)
    for (let i = 0; i < n; i++) sseStep.push(s)
  }
  while (sseStep.length < sse.length) sseStep.push(N)
  const lastClient = decode[N - 1].clientMs!
  const sseMs = sseStep.map((s) => (s < N ? decode[s].clientMs! : lastClient))

  /* ---------- 10. метрики ---------- */
  const ttftMs = decode[0].clientMs!
  const e2eMs = lastClient
  const itlMs = N > 1 ? (e2eMs - ttftMs) / (N - 1) : 0
  let ttftVisibleMs: number | undefined
  let firstAnswerMs: number | undefined
  for (let i = 1; i < sse.length; i++) {
    const delta = (sse[i].json?.choices as Array<{ delta?: Record<string, unknown> }> | undefined)?.[0]?.delta
    if (!delta) continue
    const visible = Object.entries(delta).some(([key, v]) => key !== 'role' && (Array.isArray(v) ? v.length > 0 : typeof v === 'string' && v.length > 0))
    if (visible && ttftVisibleMs === undefined) ttftVisibleMs = sseMs[i]
    // ответ, а не рассуждение: непустой content (не одни переводы строк после </think>) или tool_calls
    const answer =
      (typeof delta.content === 'string' && delta.content.trim().length > 0) || (Array.isArray(delta.tool_calls) && delta.tool_calls.length > 0)
    if (answer) {
      firstAnswerMs = sseMs[i]
      break
    }
  }
  const metrics: FlowRunEx['metrics'] = {
    ttftMs,
    itlMs,
    e2eMs,
    promptTokens: total,
    completionTokens,
    cachedTokens,
    tokensPerSec: itlMs > 0 ? 1000 / itlMs : 0,
    finishReason,
    perUserTps: e2eMs > 0 ? completionTokens / (e2eMs / 1000) : 0,
    prefillMs: prefill.ms,
    decodeMs,
  }
  if (ttftVisibleMs !== undefined) metrics.ttftVisibleMs = ttftVisibleMs
  if (firstAnswerMs !== undefined) metrics.firstAnswerMs = firstAnswerMs

  /* ---------- 11. сводки для UI ---------- */
  const stages = summarizeStages(timeline)
  const playback = buildPlayback(timeline, decode, sseStep, sse.length, finishReason, N)

  // Сколько GPU на воркер у этой модели на других GPU симулятора: если TP разный, поясняем, откуда он.
  const tpOn = SIM_GPUS.slice(0, 3).map((g) => ({ g, tp: simTensorParallel(model, g).tp }))
  if (tp > 1 || tpOn.some((x) => x.tp !== tp)) {
    const wBytes = weightBytes(model, 'bf16')
    const wGB = ruNum(wBytes / 1e9, 1)
    const memGB = ruNum(gpu.memGB, 0)
    const req = `KV cache для ${SIM_KV_REQUIREMENT.seqs} запросов по ${tokensWord(SIM_KV_REQUIREMENT.contextLen)} (условный запас)`
    if (tp > 1 && wBytes > gpu.memGB * 1e9) {
      // (G0-r2-10) веса больше памяти одного GPU: причина в весах, а не в запасе под KV
      const prevTp = SIM_TP_OPTIONS[SIM_TP_OPTIONS.indexOf(tp) - 1] ?? 1
      const prevWeightsFit = fitsCheck(model, gpu, { ...SIM_KV_REQUIREMENT, gpus: prevTp }).weightsFit
      notes.push(
        `${model.name} в BF16 весит около ${wGB} ГБ: веса не помещаются на один ${gpu.name} с ${memGB} ГБ памяти, ` +
          `поэтому воркер делит их на ${tp} GPU (tensor parallel = ${tp}; на ${prevTp} GPU ` +
          `${prevWeightsFit ? `не остаётся места под ${req}` : 'не помещаются даже веса'}). ` +
          `Каждый GPU хранит и читает свою часть весов, около ${ruNum(wBytes / tp / 1e9, 1)} ГБ, и шаг идёт быстрее.`,
      )
    } else if (tp > 1) {
      notes.push(
        `${model.name} в BF16 весит около ${wGB} ГБ: на одном ${gpu.name} с ${memGB} ГБ памяти вместе с весами не помещается ${req}, ` +
          `поэтому воркер берёт ${tp} GPU (tensor parallel = ${tp}): каждый читает свою часть весов, и шаг идёт быстрее.`,
      )
    } else {
      const other = tpOn.find((x) => x.tp > tp)
      notes.push(
        `${model.name} в BF16 весит около ${wGB} ГБ и вместе с ${req} помещается на один ${gpu.name}.` +
          (other ? ` На ${other.g.name} воркер взял бы ${other.tp} GPU: шаг быстрее, но GPU на воркер больше.` : ''),
      )
    }
  }

  const gen = { ...QWEN3_GENERATION_CONFIG }
  const frontend: FlowFrontendInfoEx = {
    checks: frontendChecks(settings, model, total, maxTokens, !!tools, !!schema),
    // top_p и top_k в запросе нет: воркер vLLM возьмёт их из generation_config.json модели
    sampling: { temperature, topP: gen.topP, topK: gen.topK, seed: null },
    samplingFromModel: ['top_p', 'top_k'],
    generationConfig: gen,
    stop: { maxTokens, stopTokenIds: [...QWEN3_STOP_TOKEN_IDS], stopTokenTexts: ['<|im_end|>', '<|endoftext|>'], stopStrings: [] },
    parsers: { reasoning: 'qwen3', tool: 'hermes' },
    chatTemplateKwargs: { enable_thinking: settings.thinking },
  }
  if (schema) frontend.guidedJson = schema

  // Dynamo передаёт незаданные поля сэмплинга пустыми (их подставит воркер), схема structured output едет внутри sampling_options.
  const samplingOptions: Record<string, unknown> = { temperature }
  if (schema) samplingOptions.guided_decoding = { json: schema }
  const preprocessedRequest: Record<string, unknown> = {
    model: model.hfId,
    token_ids: promptIds,
    stop_conditions: { max_tokens: maxTokens, stop: [], stop_token_ids_hidden: [...QWEN3_STOP_TOKEN_IDS], min_tokens: 0, ignore_eos: false },
    sampling_options: samplingOptions,
    output_options: { skip_special_tokens: true },
    eos_token_ids: [...QWEN3_STOP_TOKEN_IDS],
  }
  const preprocess: FlowPreprocessInfo = {
    templateMs,
    tokenizeMs,
    chars: template.text.length,
    segmentOfToken,
    toolsTokens,
    contextLimit: model.maxContext,
    preprocessedRequest,
  }

  let outJson: unknown
  if (bodyJson) {
    try {
      outJson = JSON.parse(contentOut.trim())
    } catch {
      outJson = undefined
    }
  }
  const lastDetok = detok[detok.length - 1]

  const run: FlowRunEx = {
    settings,
    request,
    template,
    promptTokens,
    blockSize: B,
    blocks,
    workers: cluster.workers,
    routing: cluster.routing,
    prefill,
    ...(transfer ? { transfer } : {}),
    decode,
    detok,
    parser,
    sse,
    timeline,
    metrics,
    effectiveScenarioId: script.scenarioId,
    answerSource: script.answerSource,
    ...(script.topic ? { topic: script.topic.label } : {}),
    simulatedNote: script.simulatedNote,
    modelId: model.id,
    gpuId: gpu.id,
    gpus: tp,
    fits: tpPick.fits,
    fitsMessage: tpPick.message,
    maxTokens,
    seed,
    notes,
    frontend,
    preprocess,
    cluster: cluster.info,
    output: {
      text: lastDetok ? lastDetok.output : '',
      reasoning: reasoningOut,
      content: contentOut,
      toolCalls,
      ...(outJson !== undefined ? { json: outJson } : {}),
    },
    sseStep,
    sseMs,
    stages,
    playback,
    injection,
    ...(neighbor ? { neighbor } : {}),
    prefillThreshold,
  }
  if (disagg) {
    run.decodeWorkers = cluster.decodeWorkers
    run.decodeRouting = cluster.decodeRouting
  }
  return run
}

/* ============================================================================
 * Сводки
 * ========================================================================== */

const STAGE_ORDER: readonly StageId[] = ['client', 'frontend', 'preprocess', 'route', 'prefill', 'transfer', 'decode', 'postprocess', 'stream']
const STAGE_LABELS: Record<StageId, string> = {
  client: 'Клиент',
  frontend: 'Frontend',
  preprocess: 'Preprocessor: шаблон и токены',
  route: 'Router',
  prefill: 'Prefill',
  transfer: 'Передача KV (NIXL)',
  decode: 'Decode',
  postprocess: 'Postprocessor: детокенизация и парсеры',
  stream: 'Стрим SSE',
}

function summarizeStages(timeline: readonly TimelineItem[]): FlowStageSummary[] {
  const out: FlowStageSummary[] = []
  for (const stage of STAGE_ORDER) {
    const items = timeline.filter((x) => x.stage === stage)
    if (items.length === 0) continue
    out.push({
      stage,
      startMs: Math.min(...items.map((x) => x.startMs)),
      endMs: Math.max(...items.map((x) => x.endMs)),
      ms: items.reduce((s, x) => s + (x.endMs - x.startMs), 0),
      count: items.length,
      label: STAGE_LABELS[stage],
      simulated: items.every((x) => x.simulated === true),
    })
  }
  return out
}

function buildPlayback(
  timeline: readonly TimelineItem[],
  decode: readonly DecodeStep[],
  sseStep: readonly number[],
  sseTotal: number,
  finishReason: string,
  N: number,
): FlowPlaybackStep[] {
  const out: FlowPlaybackStep[] = []
  for (const item of timeline) {
    if (item.step !== undefined) continue
    out.push({ kind: 'stage', stage: item.stage, label: item.label, tMs: item.endMs, sseCount: 0 })
  }
  decode.forEach((d, k) => {
    out.push({
      kind: 'token',
      stage: 'decode',
      label: `Токен ${k + 1} из ${N}`,
      tMs: d.clientMs ?? d.startMs + d.durationMs,
      token: k,
      sseCount: sseStep.filter((s) => s <= k).length,
    })
  })
  out.push({
    kind: 'done',
    stage: 'stream',
    label: `Ответ завершён: finish_reason ${finishReason}, usage и [DONE]`,
    tMs: decode[N - 1]?.clientMs ?? 0,
    sseCount: sseTotal,
  })
  return out
}

/** Что frontend проверил (подписи по-русски). */
function frontendChecks(
  settings: FlowSettings,
  model: ModelSpecExt,
  promptTokens: number,
  maxTokens: number,
  tools: boolean,
  json: boolean,
): FlowFrontendInfo['checks'] {
  const checks: FlowFrontendInfo['checks'] = [
    { label: 'Тело запроса', ok: true, detail: 'JSON разобран, поля OpenAI chat.completions распознаны.' },
    { label: 'Модель', ok: true, detail: `${model.hfId} обслуживается: воркеры с этой моделью зарегистрированы.` },
    {
      label: 'Сообщения',
      ok: true,
      detail: settings.systemText.trim() ? 'Два сообщения: system и user.' : 'Одно сообщение user: пустой system prompt клиент не отправил.',
    },
    { label: 'Сэмплинг', ok: true, detail: `temperature ${ruNum(settings.temperature, 2)} в допустимом диапазоне от 0 до 2.` },
    { label: 'Стрим', ok: true, detail: 'stream: true, ответ пойдёт SSE-чанками; include_usage добавит чанк с usage в конце.' },
    {
      label: 'Бюджет контекста',
      ok: promptTokens + maxTokens <= model.maxContext,
      detail: `prompt_tokens + max_tokens = ${ruNum(promptTokens, 0)} + ${ruNum(maxTokens, 0)} ≤ ${ruNum(model.maxContext, 0)} (проверяется после токенизации).`,
    },
  ]
  if (tools) checks.push({ label: 'Инструменты', ok: true, detail: 'Один инструмент get_weather, tool_choice auto: вызовы разберёт парсер hermes.' })
  if (json) {
    checks.push({ label: 'response_format', ok: true, detail: 'JSON-схема проверена заранее и уйдёт в движок как sampling_options.guided_decoding.json.' })
  }
  return checks
}

/* ============================================================================
 * Помощники для UI
 * ========================================================================== */

/** SSE-чанки, которые клиент получил к шагу плеера (префикс run.sse). */
export function sseUpTo(run: FlowRun, sseCount: number): FlowRun['sse'] {
  return run.sse.slice(0, Math.max(0, Math.min(run.sse.length, sseCount)))
}

/** Сколько SSE-чанков клиент получил к моменту tMs реального таймлайна. */
export function sseCountAt(run: FlowRun, tMs: number): number {
  const ms = run.sseMs ?? []
  let n = 0
  while (n < ms.length && ms[n] <= tMs) n++
  return n
}

/** Индекс последнего выходного токена, дошедшего до клиента к моменту tMs (−1, если ещё ни одного). */
export function tokenIndexAt(run: FlowRun, tMs: number): number {
  let k = -1
  for (const d of run.decode) {
    if ((d.clientMs ?? Infinity) <= tMs) k = d.step
    else break
  }
  return k
}

/** Блоки промпта, лежащие в кэше выбранного воркера (для подсветки в инспекторе route / prefill). */
export function cachedBlockFlags(run: FlowRun): boolean[] {
  const w = run.workers.find((x) => x.id === run.prefill.worker)
  let hit = true
  return run.blocks.map((b: BlockInfo) => {
    hit = hit && b.full && !!w?.cachedBlocks.has(b.hash)
    return hit
  })
}
