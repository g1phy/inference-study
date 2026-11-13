/**
 * ОБЩИЕ КОНТРАКТЫ ДВИЖКА.
 *
 * Этот файл написан оркестратором и является источником правды для всех агентов.
 * Реализации живут в соседних модулях (см. SPEC.md, раздел «Модули движка»).
 * Менять сигнатуры можно только добавлением НЕобязательных полей. Если нужно
 * что-то несовместимое, опишите это в отчёте, не ломайте контракт молча.
 *
 * Всё в движке чистое и детерминированное (никакого Math.random без seed),
 * без React и без DOM, чтобы покрываться unit-тестами (vitest).
 */

/* ============================================================================
 * 1. ТОКЕНЫ И ТОКЕНИЗАТОР  (src/engine/tokenizer/*)
 * ========================================================================== */

export interface Token {
  /** ID в словаре. Для Qwen3: 0..151642 обычные, 151643..151668 спецтокены. */
  id: number
  /** Форма из словаря byte-level BPE (пробел виден как 'Ġ', перевод строки как 'Ċ'). У спецтокена равна литералу. */
  raw: string
  /** Сырые UTF-8 байты, которые обозначает токен (у спецтокена это байты литерала). */
  bytes: number[]
  /** Текст токена отдельно: UTF-8 декодирование bytes, неполная последовательность даёт U+FFFD. */
  text: string
  /**
   * true для всех added tokens (<|im_start|>, <|im_end|>, <think>, <tool_call> и т.п.):
   * они не получаются слиянием BPE и в UI рисуются пунктирной рамкой.
   */
  special: boolean
  /**
   * Флаг special из HF tokenizer_config. true значит, что токен выкидывается при
   * skip_special_tokens. У Qwen3 <think>, </think>, <tool_call> он false (парсеры их видят),
   * у <|im_start|>, <|im_end|>, <|endoftext|> true.
   */
  hfSpecial?: boolean
  /** Смещения в исходной строке (в UTF-16 code units), если известны. */
  start?: number
  end?: number
}

export interface BpeMergeStep {
  /** Текущее разбиение pre-token на куски, в человекочитаемом виде (байты декодируются мягко, неполные как «\xNN»). */
  pieces: string[]
  /** То же разбиение в форме словаря byte-level (Ġ, Ċ, ...). */
  rawPieces: string[]
  /** Какую пару сливаем, чтобы получить СЛЕДУЮЩИЙ шаг. На последнем шаге undefined. */
  merge?: { left: string; right: string; rank: number; index: number }
}

export interface BpeTrace {
  /** Pre-token как текст, например " привет". */
  pretoken: string
  bytes: number[]
  /** steps[0]: отдельные байты; последний шаг: итоговые токены. */
  steps: BpeMergeStep[]
  tokens: Token[]
}

export interface Tokenizer {
  readonly name: string // "Qwen3"
  /** Размер словаря вместе со спецтокенами (Qwen3: 151669). */
  readonly vocabSize: number
  readonly specialTokens: ReadonlyArray<{ id: number; text: string }>
  /** allowSpecial (по умолчанию true): литералы спецтокенов в тексте превращаются в один спецтокен. */
  encode(text: string, opts?: { allowSpecial?: boolean }): Token[]
  decode(ids: number[]): string
  /** Сырые байты последовательности без исправления UTF-8. */
  decodeBytes(ids: number[]): Uint8Array
  token(id: number): Token
  idOf(raw: string): number | undefined
  /** Разбиение регуляркой модели на pre-tokens (спецтокены остаются целыми). */
  preTokenize(text: string): string[]
  /** Пошаговое BPE-слияние одного pre-token (без спецтокенов). */
  trace(pretoken: string): BpeTrace
}

export type TokenizerStatus = 'idle' | 'loading' | 'ready' | 'error'

/* ============================================================================
 * 2. ШАБЛОНЫ ЧАТА  (src/engine/templates.ts)
 * ========================================================================== */

export type ModelFamily = 'qwen3' | 'llama3' | 'deepseek-r1' | 'gpt-oss'

export interface ToolCall {
  id?: string
  name: string
  arguments: Record<string, unknown>
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  /** Для assistant: рассуждение (если шаблон его рендерит). */
  reasoning?: string
  tool_calls?: ToolCall[]
  /** Для tool: имя инструмента / id вызова. */
  name?: string
  tool_call_id?: string
}

/** Подмножество JSON Schema, которого хватает для демо. */
export interface JsonSchema {
  type: 'object' | 'string' | 'integer' | 'number' | 'boolean' | 'array'
  description?: string
  properties?: Record<string, JsonSchema>
  required?: string[]
  enum?: Array<string | number>
  items?: JsonSchema
  additionalProperties?: boolean
}

export interface ToolDef {
  type: 'function'
  function: { name: string; description: string; parameters: JsonSchema }
}

export interface TemplateInput {
  family: ModelFamily
  messages: ChatMessage[]
  tools?: ToolDef[]
  /** Qwen3: enable_thinking. Для других семейств игнорируется или мапится на их механизм. */
  enableThinking?: boolean
  /** Добавить «приглашение» ассистента в конце (add_generation_prompt). По умолчанию true. */
  addGenerationPrompt?: boolean
}

export type SegmentKind =
  | 'special' // спецтокен целиком: <|im_start|>, <|eot_id|>, <think>
  | 'role' // имя роли: system / user / assistant
  | 'content' // текст сообщения
  | 'tools' // сериализованное описание инструментов
  | 'gen-prompt' // приглашение ассистента в конце
  | 'format' // служебные переводы строк, разделители, обвязка шаблона

export interface TemplateSegment {
  text: string
  kind: SegmentKind
  /** Индекс сообщения, к которому относится кусок (если относится). */
  messageIndex?: number
}

export interface RenderedTemplate {
  family: ModelFamily
  text: string
  segments: TemplateSegment[]
}

export interface FamilyInfo {
  family: ModelFamily
  displayName: string // "Qwen3", "Llama 3.1", "DeepSeek-R1", "gpt-oss (Harmony)"
  exampleModel: string // "Qwen/Qwen3-8B"
  /** Имя reasoning-парсера в vLLM/SGLang/Dynamo, если есть. */
  reasoningParser?: string
  /** Имя tool-call парсера, если есть. */
  toolParser?: string
  notes: string // 1-2 предложения по-русски: чем формат примечателен
}

/* ============================================================================
 * 3. СЭМПЛИНГ  (src/engine/sampling.ts)
 * ========================================================================== */

export interface Candidate {
  token: Token
  logit: number
}

export interface SamplingParams {
  /** 0 означает greedy. */
  temperature: number
  /** 0 или отрицательное: выключено. */
  topK: number
  /** 1: выключено. */
  topP: number
  seed?: number
}

export interface SamplingResult {
  /** Вероятности после temperature и softmax (до отсечений), по кандидатам. */
  probs: number[]
  /** Остался ли кандидат после top-k / top-p / маски. */
  kept: boolean[]
  /** Перенормированные вероятности среди оставшихся (0 у выкинутых). */
  finalProbs: number[]
  /** Индекс выбранного кандидата. */
  chosen: number
}

/* ============================================================================
 * 4. ЖЕЛЕЗО, МОДЕЛИ, ПРОИЗВОДИТЕЛЬНОСТЬ  (src/engine/data/*, src/engine/perf.ts)
 * ========================================================================== */

export type DType = 'bf16' | 'fp8' | 'fp4'

export interface GpuSpec {
  id: string // 'h100-sxm'
  name: string // 'H100 SXM'
  memGB: number
  bwTBs: number // HBM, ТБ/с
  /** Dense TFLOPS (без sparsity). */
  tflops: { bf16: number; fp8?: number; fp4?: number }
  nvlinkGBs?: number // на GPU, суммарно в обе стороны, как в даташите
  year?: number
  source: string
}

export interface InterconnectSpec {
  id: string // 'pcie5', 'nvlink4', 'ib-ndr', 'nvme5'
  name: string
  GBs: number // практическая/номинальная полоса в ГБ/с в одну сторону
  notes?: string
  source: string
}

export interface ModelSpec {
  id: string // 'qwen3-8b'
  name: string // 'Qwen3-8B'
  layers: number
  hidden: number
  heads: number
  kvHeads: number
  headDim: number
  vocab: number
  paramsB: number
  activeParamsB: number
  attention: 'mha' | 'gqa' | 'mla' | 'gqa-swa'
  /** Для MLA: kv_lora_rank + qk_rope_head_dim на слой. */
  mlaDim?: number
  /** Для моделей со sliding window: сколько слоёв полные и размер окна. */
  slidingWindow?: { window: number; fullLayers: number }
  experts?: { total: number; active: number }
  maxContext: number
  source: string
}

/* ============================================================================
 * 5. KV CACHE, PREFIX CACHING  (src/engine/kv.ts)
 * ========================================================================== */

export interface BlockInfo {
  index: number
  /** Короткий хэш для показа (8 hex). Цепочка: hash(parentHash, tokenIds блока). */
  hash: string
  tokenIds: number[]
  full: boolean
}

export interface RadixNodeView {
  id: string
  /** Токены ребра, ведущего в узел. */
  tokens: number[]
  children: RadixNodeView[]
  /** Сколько активных запросов держат узел (ref count). */
  refs: number
  lastAccess: number
  /** Подсветка: участвовал в последнем совпадении. */
  hit?: boolean
}

export interface PrefixMatch {
  matchedTokens: number
  totalTokens: number
  /** id узлов на пути совпадения. */
  path: string[]
}

/* ============================================================================
 * 6. РОУТИНГ  (src/engine/router.ts)
 * ========================================================================== */

export interface WorkerState {
  id: string // 'worker-0'
  name: string
  /** Хэши блоков, которые лежат в KV cache этого воркера на GPU (по KV events). */
  cachedBlocks: Set<string>
  /** Блоки, вытесненные в host RAM (кредит 0.75 в tier-aware формуле Dynamo). Необязательно. */
  hostBlocks?: Set<string>
  /** Блоки на локальном диске (кредит 0.25). Необязательно. */
  diskBlocks?: Set<string>
  /** Нагрузка decode: сколько блоков KV сейчас занято активными запросами (active_decode_blocks). */
  activeBlocks: number
  /** Токены, которые сейчас в prefill / ждут prefill (active_prefill_tokens). */
  queuedPrefillTokens?: number
  /** Число активных запросов (для члена w·active_requests, по умолчанию w = 0). */
  activeRequests?: number
  /** Ёмкость KV cache в блоках (для визуализации заполненности). */
  capacityBlocks?: number
}

export interface WorkerScore {
  workerId: string
  /** Сколько блоков запроса уже лежит у воркера (с начала, по цепочке хэшей). */
  overlapBlocks: number
  /** Сколько блоков придётся считать в prefill (новые блоки запроса). */
  prefillBlocks: number
  /** Нагрузка decode после назначения (active_decode_blocks + новые блоки). */
  decodeBlocks: number
  /** Кредит за кэш: c·device + 0.75·host + 0.25·disk (+ 0.5·shared). */
  credit?: number
  /** Слагаемое prefill в стоимости (после max(0, ...) и масштаба). */
  prefillTerm?: number
  /** Слагаемое decode в стоимости. */
  decodeTerm?: number
  /** Итоговая стоимость (меньше лучше) по формуле Dynamo. */
  cost: number
  /** Вероятность выбора (softmax по min-max нормированным стоимостям; при temperature=0 one-hot у минимума). */
  prob: number
}

export interface RouteDecision {
  mode: 'kv' | 'round-robin' | 'random'
  requestBlocks: number
  scores: WorkerScore[]
  chosen: string
  /** Человекочитаемое объяснение выбора по-русски. */
  explanation: string
}

/**
 * Формула Dynamo ≥ 1.2 (сверено исследователем с кодом v1.6.2, см. research/notes-dynamo.md):
 *   cost = prefillLoadScale · max(0, (active_prefill_tokens + ISL)/B − credit)
 *        + (active_decode_blocks + new_blocks)
 *        + activeRequestsWeight · active_requests
 *   credit = overlapCredit·device_blocks + 0.75·host_blocks + 0.25·disk_blocks (+ 0.5·shared_beyond)
 * Выбор: temperature = 0 → argmin; иначе p_i ∝ exp(−(c_i − c_min) / (T·(c_max − c_min))).
 * 'legacy' (до v1.1): cost = overlapWeight·prefill_blocks + decode_blocks.
 */
export interface RouterOptions {
  /** Dynamo: overlap_score_credit (c), по умолчанию 1.0. */
  overlapCredit: number
  /** Температура выбора (Dynamo: router-temperature). 0 = детерминированный argmin. */
  temperature: number
  /** По умолчанию 1.0. */
  prefillLoadScale?: number
  /** Вес w для active_requests, по умолчанию 0. */
  activeRequestsWeight?: number
  /** По умолчанию 'v1.2'. */
  formula?: 'v1.2' | 'legacy'
  /** Для formula = 'legacy': overlap_score_weight, по умолчанию 1.0. */
  legacyOverlapWeight?: number
  seed?: number
}

/* ============================================================================
 * 7. ПЛАНИРОВЩИК И СЕРВИНГ  (src/engine/sched.ts)
 * ========================================================================== */

export interface SimRequest {
  id: string
  arrivalMs: number
  promptTokens: number
  outputTokens: number
  /** Сколько токенов промпта уже лежат в prefix cache. */
  cachedTokens?: number
}

export interface StepRecord {
  /** Начало и длительность шага движка (одна итерация forward). */
  tMs: number
  durationMs: number
  /** Куски prefill в этом шаге. */
  prefill: Array<{ reqId: string; tokens: number }>
  /** Запросы, которые получили по одному decode-токену в этом шаге. */
  decode: string[]
  worker?: string
}

export interface RequestStats {
  id: string
  arrivalMs: number
  firstTokenMs: number
  finishMs: number
  ttftMs: number
  /** Межтокенные интервалы (длина = outputTokens - 1). */
  itlMs: number[]
}

export interface SimResult {
  steps: StepRecord[]
  requests: RequestStats[]
  /** Сводка для подписи графиков. */
  summary: {
    ttftP50: number
    ttftP99: number
    itlP50: number
    itlP99: number
    throughputTokS: number
    makespanMs: number
  }
}

/* ============================================================================
 * 8. ИГРУШЕЧНЫЙ ATTENTION  (src/engine/attention.ts)
 * ========================================================================== */

export type Matrix = number[][]

export interface AttentionTrace {
  Q: Matrix
  K: Matrix
  V: Matrix
  /** Q·Kᵀ до масштабирования. */
  rawScores: Matrix
  /** После деления на √d_k. */
  scaled: Matrix
  /** После causal mask (замаскированные = -Infinity). */
  masked: Matrix
  /** Softmax по строкам. */
  weights: Matrix
  /** weights · V. */
  out: Matrix
}

export interface HeadPattern {
  id: string
  name: string // по-русски: «Предыдущий токен»
  description: string // по-русски, 1-2 предложения
  /** n×n, строки суммируются в 1, causal (верхний треугольник = 0). */
  weights: Matrix
}

/* ============================================================================
 * 9. ДЕТОКЕНИЗАЦИЯ И ПАРСЕРЫ  (src/engine/detok.ts, src/engine/parsers.ts)
 * ========================================================================== */

export interface DetokStep {
  index: number
  token: Token
  /** Что вышло бы при наивном декодировании этого токена отдельно. */
  naiveText: string
  /** Что инкрементальный детокенизатор отдал клиенту на этом шаге (может быть ''). */
  emitted: string
  /** Что придержано (неполный UTF-8 или возможное начало stop-строки). */
  held: string
  prefixOffset: number
  readOffset: number
  /** Сработал stop. */
  stopped?: boolean
}

export type Channel = 'content' | 'reasoning' | 'tool_call'

export type ParserState = 'content' | 'reasoning' | 'tool_call' | 'maybe_tag'

export interface ParserDelta {
  channel: Channel
  text: string
}

export interface ParserEvent {
  /** Номер входного куска (обычно один токен = один кусок). */
  step: number
  input: string
  stateBefore: ParserState
  stateAfter: ParserState
  /** Буфер: текст, который парсер придержал, пока не понял, тег это или нет. */
  buffer: string
  deltas: ParserDelta[]
  /** Полностью разобранный tool call (появляется на шаге, где пришёл закрывающий тег). */
  toolCall?: ToolCall
  note?: string // по-русски, что произошло
}

export interface ParserOptions {
  reasoning: 'qwen3' | 'deepseek-r1' | 'none'
  tools: 'hermes' | 'none'
  /** Модель начинает сразу «внутри» рассуждения (шаблон уже вставил <think>). */
  startInReasoning?: boolean
}

/** Один SSE-чанк OpenAI-совместимого стрима. */
export interface SseChunk {
  /** Готовая строка «data: {...}» или «data: [DONE]». */
  line: string
  /** Тот же объект до сериализации (null у [DONE]). */
  json: Record<string, unknown> | null
}

/* ============================================================================
 * 10. STRUCTURED OUTPUT  (src/engine/structured.ts)
 * ========================================================================== */

export interface GrammarState {
  /** Валиден ли текст как префикс какого-то JSON, удовлетворяющего схеме. */
  validPrefix: boolean
  /** Полный ли это валидный документ. */
  complete: boolean
  /** По-русски, чего грамматика ждёт дальше: «ключ "name"», «целое число», «запятая или }». */
  expecting: string
}

/* ============================================================================
 * 11. ПОЛНЫЙ ПРОГОН ЗАПРОСА  (src/engine/flow.ts, src/engine/scenarios.ts)
 * ========================================================================== */

export type ScenarioId = 'simple' | 'thinking' | 'tool' | 'json' | 'custom'

/** Глобальные настройки, которые пользователь задаёт в симуляторе. Хранятся в src/state/settings.ts. */
export interface FlowSettings {
  scenarioId: ScenarioId
  systemText: string
  userText: string
  thinking: boolean
  tools: boolean
  json: boolean
  deployment: 'agg' | 'disagg'
  /** Тёплый prefix cache: системный промпт уже лежит на одном из воркеров. */
  cacheWarm: boolean
  modelId: string // 'qwen3-8b'
  gpuId: string // 'h100-sxm'
  temperature: number
}

export type StageId =
  | 'client'
  | 'frontend'
  | 'preprocess'
  | 'route'
  | 'prefill'
  | 'transfer'
  | 'decode'
  | 'postprocess'
  | 'stream'

export interface TimelineItem {
  stage: StageId
  startMs: number
  endMs: number
  label: string
  /** Для decode, postprocess, stream: номер шага (индекс выходного токена). */
  step?: number
  /** (S1a) Пояснение по-русски, одна фраза. */
  detail?: string
  /** (S1a) Длительность условная (константа из README.flow.md), а не из perf.ts. */
  simulated?: boolean
}

/** (S1a) Кандидат на шаге decode. Поля token/logit/prob/allowed были в контракте изначально. */
export interface DecodeCandidate {
  token: Token
  /** Синтетический логит (детерминированный по seed). */
  logit: number
  /** Вероятность после маски structured output и перенормировки: из этого выбирает сэмплер (0 у замаскированных). */
  prob: number
  allowed: boolean
  /** (S1a) Вероятность до маски (после temperature): что «хотела» модель. Без маски равна prob. */
  rawProb?: number
  /** (S1a) Для ручных альтернатив: как модель продолжила бы, выбрав этот токен («Конечно, вот JSON:»). */
  preview?: string
}

export interface DecodeStep {
  step: number
  token: Token
  channel: Channel
  candidates: DecodeCandidate[]
  /** Был ли выбор ограничен structured output маской (грамматика активна на этом шаге). */
  masked: boolean
  startMs: number
  durationMs: number
  /** (S1a) 'prefill': первый токен выбран в конце prefill (agg); 'decode': отдельный шаг decode. */
  phase?: 'prefill' | 'decode'
  /** (S1a) Индекс выбранного токена в candidates (кандидаты отсортированы по логиту). */
  chosenIndex?: number
  /** (S1a) Энтропия распределения, из которого выбирали, в битах. */
  entropyBits?: number
  /** (S1a) Сколько показанных кандидатов отрезала маска. */
  maskedCount?: number
  /** (S1a) Что грамматика ждёт перед этим шагом, по-русски («ключ "name"»). Только в json-режиме. */
  grammarExpecting?: string
  /** (S1a) Длина KV cache нашей последовательности после шага, токенов, и её размер в байтах. */
  kvTokens?: number
  kvBytes?: number
  /** (S1a) Сколько последовательностей в batch на этом шаге (наша + соседи по воркеру). */
  batch?: number
  /** (S1a) Когда frontend отправил чанк с этим токеном (после детокенизации и парсеров) и когда он дошёл до клиента. */
  emitMs?: number
  clientMs?: number
  /** (S1a) Это стоп-токен (<|im_end|>). */
  stop?: boolean
  /** (S1a) Пояснение к шагу по-русски. */
  note?: string
}

/** (S1a) Сводка по стадии пайплайна: одна запись на StageId, который был в прогоне. */
export interface FlowStageSummary {
  stage: StageId
  /** Начало первого и конец последнего отрезка стадии на таймлайне. */
  startMs: number
  endMs: number
  /** Суммарная длительность отрезков стадии (у decode сумма шагов, у postprocess и stream сумма по токенам). */
  ms: number
  /** Сколько отрезков у стадии на таймлайне (у decode, postprocess, stream по одному на токен). */
  count: number
  label: string
  /** Все отрезки стадии условные (константы), а не из perf.ts. */
  simulated: boolean
}

/** (S1a) Шаг плеера симулятора: стадия до первого токена, один выходной токен или финал. */
export interface FlowPlaybackStep {
  kind: 'stage' | 'token' | 'done'
  /** Узел пайплайна, который подсвечивается на этом шаге. */
  stage: StageId
  label: string
  /** Момент реального таймлайна (мс) в конце шага: позиция плейхеда. */
  tMs: number
  /** Для kind = 'token': индекс в decode / detok / parser. */
  token?: number
  /** Сколько SSE-чанков клиент получил к концу шага (префикс run.sse). */
  sseCount: number
}

/** (S1a) Воркер кластера симулятора (для подписи и схемы). */
export interface FlowWorkerInfo {
  id: string
  name: string
  role: 'aggregated' | 'prefill' | 'decode'
  /** GPU на воркер (tensor parallel). */
  gpus: number
  /** Ёмкость KV cache воркера, блоков. */
  capacityBlocks: number
  activeRequests: number
  activeBlocks: number
  queuedPrefillTokens: number
  /** Что лежит в его prefix cache: чьи префиксы и сколько полных блоков. */
  cached: Array<{ label: string; blocks: number; matchesPrompt: boolean }>
}

/** (S1a) Кластер симулятора. */
export interface FlowClusterInfo {
  deployment: 'agg' | 'disagg'
  /** GPU на воркер (TP) и всего GPU. */
  tp: number
  gpuCount: number
  workers: FlowWorkerInfo[]
  /** Воркер, у которого в тёплом режиме лежит префикс промпта (если есть). */
  warmWorkerId?: string
  /** Сколько полных блоков промпта лежит на тёплом воркере (0 в холодном режиме). */
  warmBlocks: number
}

/** (S1a) Что frontend проверил и во что превратил запрос (стадия frontend). */
export interface FlowFrontendInfo {
  checks: Array<{ label: string; ok: boolean; detail: string }>
  /** Параметры, которые уедут в сэмплер. Симулятор учитывает только temperature. */
  sampling: { temperature: number; topP: number; topK: number; seed: number | null }
  stop: { maxTokens: number; stopTokenIds: number[]; stopTokenTexts: string[]; stopStrings: string[] }
  /** Схема для structured output (guided_decoding.json), если включён json. */
  guidedJson?: JsonSchema
  /** Флаги парсеров на сервере. */
  parsers: { reasoning: string; tool: string }
  chatTemplateKwargs: { enable_thinking: boolean }
}

/** (S1a) Стадия preprocess: шаблон, токенизация, что уходит на воркер. */
export interface FlowPreprocessInfo {
  templateMs: number
  tokenizeMs: number
  chars: number
  /** Индекс сегмента template.segments для каждого токена промпта (по смещению начала токена). */
  segmentOfToken: number[]
  /** Сколько токенов промпта добавило описание tools (0, если tools выключены). */
  toolsTokens: number
  /** prompt_tokens + max_tokens должно влезть в контекст модели. */
  contextLimit: number
  /** Упрощённый PreprocessedRequest Dynamo (то, что уходит роутеру и воркеру). */
  preprocessedRequest: Record<string, unknown>
}

export interface FlowRun {
  settings: FlowSettings
  request: Record<string, unknown>
  template: RenderedTemplate
  promptTokens: Token[]
  blockSize: number
  blocks: BlockInfo[]
  /** agg: 4 агрегированных воркера; disagg: пул prefill-воркеров (KV-aware выбор). */
  workers: WorkerState[]
  /** agg: выбор воркера; disagg: выбор prefill-воркера (PrefillRouter, KV-aware). */
  routing: RouteDecision
  prefill: {
    totalTokens: number
    cachedTokens: number
    computedTokens: number
    ms: number
    worker: string
    /** KV, записанный prefill (посчитанные токены), байт. */
    kvBytes: number
    /** (S1a) Имя воркера для подписи. */
    workerName?: string
    /** (S1a) Сколько полных блоков взято из prefix cache. */
    cachedBlocks?: number
    /** (S1a) KV всего промпта (из кэша + посчитанный), байт. */
    kvTotalBytes?: number
    /** (S1a) Для прогресса по слоям: число слоёв модели и время на слой (поровну). */
    layers?: number
    layerMs?: number
    /** (S1a) Разбивка из perf.prefillBreakdown. */
    bound?: 'memory' | 'compute'
    computeMs?: number
    memoryMs?: number
    flops?: number
  }
  transfer?: {
    bytes: number
    ms: number
    link: string
    from: string
    to: string
    /** (S1a) id канала из data/interconnects, полоса одного канала (ГБ/с), сколько каналов параллельно. */
    linkId?: string
    GBs?: number
    links?: number
    /** (S1a) Условная константа: регистрация передачи и уведомление о завершении, мс. */
    setupMs?: number
    /** (S1a) Иллюстративные kv_transfer_params, которые prefill-воркер вернул во frontend (vLLM NixlConnector). */
    params?: Record<string, unknown>
    fromName?: string
    toName?: string
  }
  decode: DecodeStep[]
  detok: DetokStep[]
  parser: ParserEvent[]
  sse: SseChunk[]
  timeline: TimelineItem[]
  metrics: {
    ttftMs: number
    itlMs: number
    e2eMs: number
    promptTokens: number
    completionTokens: number
    cachedTokens: number
    tokensPerSec: number
    finishReason: 'stop' | 'length' | 'tool_calls'
    /** (S1a) Когда до клиента дошёл первый чанк с непустой дельтой (reasoning / content / tool_calls). */
    ttftVisibleMs?: number
    /** (S1a) Ток/с на пользователя по NVIDIA: completionTokens / E2E. */
    perUserTps?: number
    /** (S1a) Время prefill и сумма шагов decode, мс. */
    prefillMs?: number
    decodeMs?: number
  }

  /* ---------- (S1a) необязательные поля, все заполняет buildRun ---------- */

  /** Сценарий, по которому реально построен ответ (пресет только пока его текст не изменён). */
  effectiveScenarioId?: ScenarioId
  /** Откуда текст ответа: пресет, заготовка по теме своего текста, общий ответ. */
  answerSource?: 'preset' | 'topic' | 'generic'
  /** Тема своего текста (для answerSource 'topic' | 'generic'). */
  topic?: string
  /** Честная подпись: что в прогоне симулировано, а что посчитано. */
  simulatedNote?: string
  /** Модель и GPU, которые реально использованы (после замены неподдержанных), и TP. */
  modelId?: string
  gpuId?: string
  gpus?: number
  /** Влезают ли веса и бюджет KV при выбранном TP (fitsCheck) и вывод по-русски. */
  fits?: boolean
  fitsMessage?: string
  /** max_tokens запроса. */
  maxTokens?: number
  /** Seed синтетических логитов. */
  seed?: number
  /** Заметки по-русски к этому прогону (упрощения, особые случаи). */
  notes?: string[]
  frontend?: FlowFrontendInfo
  preprocess?: FlowPreprocessInfo
  cluster?: FlowClusterInfo
  /** disagg: пул decode-воркеров и выбор decode-воркера (только по нагрузке). */
  decodeWorkers?: WorkerState[]
  decodeRouting?: RouteDecision
  /** Итог глазами клиента: склейка дорожек парсера. */
  output?: { text: string; reasoning: string; content: string; toolCalls: ToolCall[]; json?: unknown }
  /** Для каждого SSE-чанка: после какого шага decode он отправлен и когда дошёл до клиента. */
  sseStep?: number[]
  sseMs?: number[]
  /** Сводка по стадиям (по одной записи на StageId). */
  stages?: FlowStageSummary[]
  /** Шаги плеера: стадии до первого токена, по шагу на токен, финал. */
  playback?: FlowPlaybackStep[]
}
