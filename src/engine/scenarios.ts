/**
 * Сценарии симулятора «Путь запроса» (агент S1a): вкладки-пресеты, заскриптованные ответы модели
 * и тематические заготовки для «Своего текста».
 *
 * Живой LLM в браузере не запускается: текст ответа здесь написан заранее (пресеты) или подобран по
 * ключевым словам (свой текст). Всё остальное flow.ts считает по-настоящему поверх этого текста:
 * настоящий шаблон Qwen3, токены словаря Qwen3, хэши блоков, решение роутера, время из perf.ts,
 * маски structured output, детокенизация, парсеры, SSE.
 *
 * Как устроен скрипт ответа: рассуждение (без тегов <think>) и тело (текст, вызов инструмента или
 * JSON) нарезаны на куски ScriptPiece. Кусок может нести ручные альтернативы (AltSpec) для своего
 * первого токена: так в ключевых местах показываются осмысленные кандидаты («Конечно» на первом шаге
 * JSON, «Казани» вместо «Казань»). Остальные кандидаты flow-candidates.ts подбирает сам.
 */
import type { FlowSettings, JsonSchema, ScenarioId, ToolDef } from './types'
import { FAMILY_SPECIAL_TOKENS, pyJson } from './templates'

/* ============================================================================
 * Типы
 * ========================================================================== */

/** Пресеты (все сценарии, кроме своего текста). */
export type PresetId = Exclude<ScenarioId, 'custom'>

/** Переключатели запроса. */
export interface ScenarioToggles {
  thinking: boolean
  tools: boolean
  json: boolean
}

/** Вкладка сценария. */
export interface ScenarioInfo {
  id: ScenarioId
  /** Подпись вкладки. */
  label: string
  /** Что показывает сценарий, одна фраза. */
  description: string
  /** Текст пользователя пресета ('' у своего текста: там остаётся текст читателя). */
  userText: string
  systemText: string
  /** Переключатели по умолчанию. */
  toggles: ScenarioToggles
}

/** Ручная альтернатива на ключевом шаге: текст кодируется настоящим токенизатором, берётся первый токен. */
export interface AltSpec {
  text: string
  /** Как модель продолжила бы, если бы выбрала этот токен (подпись в инспекторе). */
  preview?: string
  /**
   * Логит относительно выбранного токена. Отрицательный: менее вероятен. Положительный допустим только
   * там, где кандидата отрежет маска structured output (модель «хотела» его сильнее).
   */
  delta?: number
}

/** Кусок заскриптованного текста; alts и confidence относятся к токену, на который попадает начало куска. */
export interface ScriptPiece {
  text: string
  alts?: AltSpec[]
  /** Уверенность модели в этом токене: задаёт разрыв логитов до автоматических альтернатив. */
  confidence?: 'high' | 'medium' | 'low'
}

export type BodyKind = 'text' | 'tool_call' | 'json'

/** Схема structured output сценария и значение, которое модель выдаст. */
export interface ScriptJson {
  /** response_format.json_schema.name */
  name: string
  schema: JsonSchema
  value: Record<string, unknown>
}

/** Заскриптованный ответ модели для конкретных настроек. */
export interface Script {
  /** Сценарий, по которому построен ответ (см. resolveScenario). */
  scenarioId: ScenarioId
  answerSource: 'preset' | 'topic' | 'generic'
  /** Тема своего текста. */
  topic?: { id: string; label: string }
  /** Рассуждение без тегов <think>; null, если thinking выключен. */
  reasoning: ScriptPiece[] | null
  bodyKind: BodyKind
  body: ScriptPiece[]
  /** Вызов инструмента (bodyKind = 'tool_call'). */
  toolCall?: { name: string; arguments: Record<string, unknown> }
  /** Схема и значение (если включён json; тогда bodyKind = 'json'). */
  json?: ScriptJson
  /** Заметки по-русски: особые случаи этого сочетания переключателей. */
  notes: string[]
  /** Честная подпись: что симулировано, а что посчитано. */
  simulatedNote: string
}

/* ============================================================================
 * Инструмент и вкладки
 * ========================================================================== */

/** Инструмент, который получает модель, когда включены tools (сценарий «Вызов инструмента» и свой текст). */
export const WEATHER_TOOL: ToolDef = {
  type: 'function',
  function: {
    name: 'get_weather',
    description: 'Текущая погода в указанном городе',
    parameters: {
      type: 'object',
      properties: { city: { type: 'string', description: 'Название города, например Берлин' } },
      required: ['city'],
    },
  },
}

const SYSTEM_TEXT = 'Ты полезный ассистент.'

/** Вкладки в порядке показа. Тексты thinking-пресета совпадают с DEFAULT_SETTINGS (src/state/settings.ts). */
export const SCENARIOS: readonly ScenarioInfo[] = [
  {
    id: 'simple',
    label: 'Простой вопрос',
    description: 'Короткий вопрос без рассуждения: самый короткий путь запроса.',
    userText: 'Сколько будет 2+2?',
    systemText: SYSTEM_TEXT,
    toggles: { thinking: false, tools: false, json: false },
  },
  {
    id: 'thinking',
    label: 'С размышлением',
    description: 'Модель сначала рассуждает внутри <think>, парсер уводит мысли в reasoning_content.',
    userText: 'Объясни простыми словами, что такое KV cache и зачем он нужен.',
    systemText: SYSTEM_TEXT,
    toggles: { thinking: true, tools: false, json: false },
  },
  {
    id: 'tool',
    label: 'Вызов инструмента',
    description: 'Модель вызывает get_weather: tool parser собирает tool_calls, finish_reason равен tool_calls.',
    userText: 'Какая погода сейчас в Берлине?',
    systemText: SYSTEM_TEXT,
    toggles: { thinking: false, tools: true, json: false },
  },
  {
    id: 'json',
    label: 'JSON по схеме',
    description: 'Ответ по JSON-схеме: маска structured output отсекает «Конечно, вот JSON:».',
    userText: 'Извлеки данные: Анна, 29 лет, живёт в Казани',
    systemText: SYSTEM_TEXT,
    toggles: { thinking: false, tools: false, json: true },
  },
  {
    id: 'custom',
    label: 'Свой текст',
    description: 'Ваш собственный вопрос: ответ модели симулирован, весь путь запроса посчитан по-настоящему.',
    userText: '',
    systemText: SYSTEM_TEXT,
    toggles: { thinking: true, tools: false, json: false },
  },
]

/**
 * Длинный system prompt: регламент дежурного инженера (больше тысячи токенов). На нём prefill упирается в вычисления,
 * а тёплый prefix cache заметно сокращает TTFT. Начало не совпадает с промптами тенантов flow-cluster.ts:
 * в холодном режиме совпадений в кэшах нет.
 */
export const LONG_DOC_SYSTEM = [
  'Ты помогаешь дежурным инженерам платформы инференса. Отвечай коротко и только по регламенту ниже. Если ответа в регламенте нет, так и скажи.',
  '',
  'Регламент эксплуатации кластера инференса',
  '',
  '1. Состав кластера. В кластере восемь узлов, на каждом восемь GPU H100 SXM по 80 ГБ. На узлах работают воркеры vLLM с моделью Qwen3-8B, ' +
    'по одному воркеру на GPU. Перед воркерами стоит frontend NVIDIA Dynamo: он принимает запросы OpenAI API, применяет шаблон чата, ' +
    'токенизирует текст и через KV-aware роутер выбирает воркер. На каждом узле запущен DaemonSet с общим кэшем KV в оперативной памяти: ' +
    'туда воркеры выгружают блоки, вытесненные из памяти GPU, и оттуда же их забирают обратно.',
  '',
  '2. Метрики и пороги. Главные метрики: TTFT (время до первого токена), ITL (пауза между токенами), длина очереди prefill и доля ' +
    'попаданий в prefix cache. Тревога поднимается, если TTFT p95 дольше 800 мс пять минут подряд, если ITL p95 дольше 60 мс или если ' +
    'очередь на воркере превышает 20 запросов. Доля попаданий в prefix cache ниже 30 % обычно значит, что роутер переключили в режим ' +
    'round-robin или что клиенты начали менять начало системного промпта.',
  '',
  '3. Перезапуск узла. Узлы перезапускают только по одному. Сначала выведите узел из балансировки: роутер перестаёт отправлять на его ' +
    'воркеры новые запросы. Затем дождитесь дренирования: воркеры дописывают активные ответы, обычно за одну-две минуты, но не дольше ' +
    'десяти минут. Если за десять минут запросы не закончились, оставшиеся можно прервать: клиент получит ошибку и повторит запрос. ' +
    'После этого перезапустите узел, дождитесь, пока все воркеры пройдут проверку готовности, и верните узел в балансировку. ' +
    'Первые минуты prefix cache на узле холодный, поэтому TTFT на нём выше обычного.',
  '',
  '4. Обновление модели. Новую версию модели сначала поднимают на одном узле и пускают туда пять процентов трафика. Если за час ' +
    'метрики не ухудшились, обновляют остальные узлы по одному, по правилам раздела 3. Веса заранее скачивают в локальный кэш узла, ' +
    'чтобы старт воркера не ждал сети.',
  '',
  '5. Нехватка памяти под KV cache. Если воркер часто вытесняет запросы (растёт счётчик preemption), сначала проверьте длину контекстов: ' +
    'один запрос на 32 тысячи токенов занимает столько же KV, сколько сорок запросов по 800 токенов. Временное решение: снизить ' +
    'max_num_seqs на воркере. Постоянное: хранить KV cache в FP8 или добавить узлы.',
  '',
  '6. Системный промпт. Системный промпт сервиса меняют целиком и только через ревью. Не добавляйте в его начало дату, имя ' +
    'пользователя или другие меняющиеся данные: prefix cache совпадает только по началу промпта, и такая правка обнуляет попадания ' +
    'во всём кластере. Меняющиеся данные ставьте в конец, после постоянной части.',
  '',
  '7. Масштабирование. Если TTFT растёт при нормальной доле попаданий в кэш, а очередь prefill держится выше 20 запросов дольше ' +
    'десяти минут, добавьте узел. Если растёт ITL, а TTFT в норме, проверьте размер batch на воркерах: чем больше запросов в batch, ' +
    'тем больше KV читает каждый шаг decode. Помогает меньший max_num_seqs или ещё один узел.',
  '',
  '8. Инциденты. О любом перезапуске и откате пишите в канал дежурных: что сделали, когда и почему. Если недоступны больше двух узлов ' +
    'сразу, вызывайте старшего дежурного.',
].join('\n')

/** Вкладка «Длинный документ» (не отдельный ScenarioId: в настройках это свой текст с длинным system prompt). */
export const LONG_DOC_TAB = {
  id: 'long' as const,
  label: 'Длинный документ',
  description:
    'Длинный system prompt (регламент на тысячу с лишним токенов). У плотных моделей (Qwen3-8B, Qwen3-32B) холодный prefill здесь упирается в ' +
    'вычисления, и тёплый prefix cache заметно сокращает TTFT. У MoE-моделей порог выше: prefill и здесь упирается в чтение весов, и кэш почти ' +
    'не сокращает TTFT.',
  userText: 'По регламенту: как безопасно перезапустить узел?',
  systemText: LONG_DOC_SYSTEM,
  toggles: { thinking: false, tools: false, json: false } satisfies ScenarioToggles,
}

/**
 * Токенов в промпте вкладки «Длинный документ» с её переключателями (шаблон Qwen3, без рассуждения и tools; закреплено тестом).
 * С ним сравнивают порог упора prefill в вычисления: если порог выше, вкладка упор не сменит (так у MoE-моделей).
 */
export const LONG_DOC_PROMPT_TOKENS = 1167

/** Что записать в настройки при клике по вкладке «Длинный документ». */
export function longDocPatch(): Partial<FlowSettings> {
  return { scenarioId: 'custom', userText: LONG_DOC_TAB.userText, systemText: LONG_DOC_TAB.systemText, ...LONG_DOC_TAB.toggles }
}

/** Это вопрос вкладки «Длинный документ» с её system prompt (тогда ответ заготовлен по регламенту). */
export function isLongDoc(settings: Pick<FlowSettings, 'userText' | 'systemText'>): boolean {
  return normText(settings.userText) === normText(LONG_DOC_TAB.userText) && settings.systemText.trim() === LONG_DOC_SYSTEM.trim()
}

/** Вкладка по id. */
export function getScenario(id: ScenarioId): ScenarioInfo {
  const s = SCENARIOS.find((x) => x.id === id)
  if (!s) throw new Error(`Неизвестный сценарий: ${id}`)
  return s
}

/**
 * Что записать в настройки при клике по вкладке: у пресета его текст, system prompt и переключатели;
 * у своего текста только scenarioId (текст читателя остаётся).
 */
export function scenarioPatch(id: ScenarioId): Partial<FlowSettings> {
  const s = getScenario(id)
  if (id === 'custom') return { scenarioId: 'custom' }
  return { scenarioId: id, userText: s.userText, systemText: s.systemText, ...s.toggles }
}

/** Нормализация для сравнения текста с пресетом: регистр, пробелы, финальная пунктуация. */
function normText(s: string): string {
  return s
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/[\s?!.…]+$/u, '')
}

/**
 * Сценарий, по которому строится ответ. Пресет действует, пока текст пользователя совпадает с его текстом
 * (без учёта регистра, лишних пробелов и финальной пунктуации); изменённый текст даёт 'custom'.
 * Если на вкладке «Свой текст» набран текст пресета, берётся пресет: ответ на него уже написан.
 */
export function resolveScenario(settings: Pick<FlowSettings, 'scenarioId' | 'userText'>): ScenarioId {
  const text = normText(settings.userText)
  const own = SCENARIOS.find((s) => s.id === settings.scenarioId && s.id !== 'custom')
  if (own && normText(own.userText) === text) return own.id
  const any = SCENARIOS.find((s) => s.id !== 'custom' && normText(s.userText) === text)
  return any ? any.id : 'custom'
}

/* ============================================================================
 * Помощники скрипта
 * ========================================================================== */

/** Склейка кусков в текст. */
export function piecesText(pieces: readonly ScriptPiece[] | null | undefined): string {
  return (pieces ?? []).map((p) => p.text).join('')
}

const p = (text: string, alts?: AltSpec[], confidence?: ScriptPiece['confidence']): ScriptPiece => {
  const piece: ScriptPiece = { text }
  if (alts) piece.alts = alts
  if (confidence) piece.confidence = confidence
  return piece
}

/** «Болтливые» начала, которые маска JSON отсекает на первом шаге грамматики. */
function chattyJsonAlts(intro: string): AltSpec[] {
  return [
    { text: 'Конечно', preview: intro, delta: 0.6 },
    { text: '```', preview: '```json', delta: -0.3 },
    { text: 'Вот', preview: 'Вот JSON по схеме:', delta: -0.8 },
    { text: '{', delta: -1.1 },
    { text: '\n', delta: -2.4 },
  ]
}

/** JSON-тело: pyJson (разделители ', ' и ': ', как XGrammar без лишних пробелов), болтливые альтернативы на первом токене. */
function jsonBody(value: Record<string, unknown>, intro: string): ScriptPiece[] {
  const text = pyJson(value)
  return [p(text.slice(0, 1), chattyJsonAlts(intro)), p(text.slice(1))]
}

/** Вызов инструмента в формате hermes, как его пишет Qwen3 (и как его рендерит шаблон в истории). */
function toolCallText(name: string, args: Record<string, unknown>): string {
  return `<tool_call>\n{"name": "${name}", "arguments": ${pyJson(args)}}\n</tool_call>`
}

function toolCallAlts(city: string): AltSpec[] {
  return [
    { text: 'Сейчас', preview: `Сейчас посмотрю погоду: ${city}.`, delta: -1.4 },
    { text: 'К сожалению', preview: 'К сожалению, у меня нет доступа к погоде…', delta: -2.0 },
    { text: 'В', preview: 'В городе сейчас солнечно, около 18 градусов. (выдумка без инструмента)', delta: -2.6 },
    { text: '\n', delta: -3.6 },
  ]
}

/** Убирает из текста литералы спецтокенов и переводы строк: эхо пользовательского текста в ответе не должно их содержать. */
export function sanitizeEcho(s: string): string {
  let out = s
  for (const t of FAMILY_SPECIAL_TOKENS.qwen3) out = out.split(t).join(' ')
  return out.replace(/\s+/g, ' ').trim()
}

const TOOLS_NOT_NEEDED = 'Инструмент get_weather здесь не нужен. '

/** Что пишется в зависимости от переключателей. */
interface ScriptSpec {
  scenarioId: ScenarioId
  answerSource: Script['answerSource']
  topic?: Script['topic']
  simulatedNote: string
  text: { pieces: ScriptPiece[]; reasoning: string }
  json: { name: string; schema: JsonSchema; value: Record<string, unknown>; pieces?: ScriptPiece[]; reasoning: string; intro?: string }
  /** Вопросу нужен инструмент: вызов, если tools включены. */
  tool?: { name: string; arguments: Record<string, unknown>; pieces?: ScriptPiece[]; reasoning: string; city: string; note?: string }
  /** Ответ, если инструмент нужен, но tools выключены. */
  noTool?: { pieces: ScriptPiece[]; reasoning: string }
  /** Вопрос про погоду без города: текст уточняет город, а не «инструмент не нужен». */
  toolRelevant?: boolean
}

function assemble(spec: ScriptSpec, t: ScenarioToggles): Script {
  const notes: string[] = []
  let bodyKind: BodyKind
  let body: ScriptPiece[]
  let reasoning: string
  let json: ScriptJson | undefined
  let toolCall: Script['toolCall']
  if (t.json) {
    bodyKind = 'json'
    json = { name: spec.json.name, schema: spec.json.schema, value: spec.json.value }
    body = spec.json.pieces ?? jsonBody(spec.json.value, spec.json.intro ?? 'Конечно, вот ответ в JSON:')
    reasoning = spec.json.reasoning
    if (t.tools && spec.tool) {
      reasoning += ' Инструмент есть, но ответ ограничен JSON-схемой.'
      notes.push('JSON-схема ограничивает весь ответ, поэтому <tool_call> не проходит маску: вместо вызова инструмента модель отвечает JSON.')
    } else if (t.tools) {
      notes.push('Описание get_weather попало в промпт, но модель не стала вызывать инструмент.')
    }
    if (t.thinking) notes.push('Грамматика JSON включается только после </think>: рассуждение идёт без маски, а сразу после него маска не пускает обычные «\\n\\n».')
  } else if (t.tools && spec.tool) {
    bodyKind = 'tool_call'
    toolCall = { name: spec.tool.name, arguments: spec.tool.arguments }
    body = spec.tool.pieces ?? [p('<tool_call>', toolCallAlts(spec.tool.city)), p(toolCallText(spec.tool.name, spec.tool.arguments).slice('<tool_call>'.length))]
    reasoning = spec.tool.reasoning
    if (spec.tool.note) notes.push(spec.tool.note)
  } else if (!t.tools && spec.noTool) {
    bodyKind = 'text'
    body = spec.noTool.pieces
    reasoning = spec.noTool.reasoning
  } else {
    bodyKind = 'text'
    body = spec.text.pieces
    reasoning = spec.text.reasoning
    if (t.tools && spec.toolRelevant) {
      notes.push('Без города get_weather вызвать нельзя: модель сначала уточняет город.')
    } else if (t.tools) {
      reasoning = TOOLS_NOT_NEEDED + reasoning
      notes.push('Описание get_weather попало в промпт, но модель не стала вызывать инструмент.')
    }
  }
  return {
    scenarioId: spec.scenarioId,
    answerSource: spec.answerSource,
    ...(spec.topic ? { topic: spec.topic } : {}),
    reasoning: t.thinking ? [p(reasoning)] : null,
    bodyKind,
    body,
    ...(toolCall ? { toolCall } : {}),
    ...(json ? { json } : {}),
    notes,
    simulatedNote: spec.simulatedNote,
  }
}

function presetNote(label: string): string {
  return (
    `Ответ модели заранее написан для сценария «${label}»: живая модель в браузере не запускается. ` +
    'Шаблон, токены, хэши блоков, решение роутера, время prefill и decode, маски, детокенизация, парсеры и SSE посчитаны по-настоящему.'
  )
}

const objectSchema = (props: Record<string, JsonSchema>): JsonSchema => ({
  type: 'object',
  properties: props,
  required: Object.keys(props),
  additionalProperties: false,
})

/* ============================================================================
 * Пресеты
 * ========================================================================== */

function simpleSpec(): ScriptSpec {
  return {
    scenarioId: 'simple',
    answerSource: 'preset',
    simulatedNote: presetNote('Простой вопрос'),
    text: {
      reasoning: 'Простое сложение: 2 + 2 = 4. Отвечу коротко.',
      pieces: [
        p('2', [
          { text: 'Четыре', preview: 'Четыре.', delta: -1.2 },
          { text: '4', preview: '4', delta: -1.6 },
          { text: 'Ответ', preview: 'Ответ: 4.', delta: -2.1 },
          { text: 'Будет', preview: 'Будет 4.', delta: -2.6 },
        ]),
        p(' + 2 = '),
        p('4', [{ text: '5', delta: -7 }, { text: '3', delta: -7.6 }, { text: '2', delta: -8.1 }, { text: '8', delta: -8.7 }], 'high'),
        p('.'),
      ],
    },
    json: {
      name: 'sum',
      schema: objectSchema({ answer: { type: 'integer' } }),
      value: { answer: 4 },
      reasoning: 'Простое сложение: 2 + 2 = 4. Ответ нужен в JSON с полем answer.',
      pieces: [
        p('{', chattyJsonAlts('Конечно! 2 + 2 = 4.')),
        p('"answer": '),
        p('4', [{ text: '"', preview: '"4" строкой вместо числа', delta: 0.3 }, { text: '5', delta: -6.5 }, { text: '3', delta: -7 }]),
        p('}'),
      ],
    },
  }
}

const KV_ANSWER: ScriptPiece[] = [
  p('KV', [
    { text: 'Представьте', preview: 'Представьте, что модель ведёт конспект…', delta: -0.7 },
    { text: 'Это', preview: 'Это память модели о прочитанном тексте…', delta: -1.1 },
    { text: 'Кэш', preview: 'Кэш ключей и значений attention…', delta: -1.4 },
    { text: 'В', preview: 'В трансформере каждый токен…', delta: -1.9 },
  ]),
  p(' cache хранит ключи (K) и значения (V) из attention для всех токенов, которые модель уже обработала.\n\n'),
  p('Ответ', [
    { text: 'Модель', preview: 'Модель пишет ответ по одному токену…', delta: -0.4 },
    { text: 'Когда', preview: 'Когда модель пишет ответ…', delta: -0.9 },
    { text: 'Генерация', preview: 'Генерация идёт по одному токену…', delta: -1.3 },
  ]),
  p(
    ' пишется по одному токену, и каждый новый токен сверяется со всеми предыдущими. Их K и V при этом не меняются, ' +
      'поэтому выгоднее посчитать их один раз и держать в памяти, а не пересчитывать весь текст на каждом шаге.\n\n',
  ),
  p('Цена', [
    { text: 'Но', preview: 'Но у кэша есть цена…', delta: -0.5 },
    { text: 'Главный', preview: 'Главный минус: память GPU…', delta: -0.9 },
    { text: 'Минус', preview: 'Минус: кэш занимает память GPU…', delta: -1.2 },
  ]),
  p(' за скорость: память GPU. Кэш растёт с длиной контекста и с числом одновременных запросов.'),
]

function thinkingSpec(): ScriptSpec {
  return {
    scenarioId: 'thinking',
    answerSource: 'preset',
    simulatedNote: presetNote('С размышлением'),
    text: {
      reasoning:
        'Нужно объяснить просто. Модель пишет ответ по одному токену, и на каждом шаге attention смотрит на все предыдущие токены. ' +
        'Их ключи и значения не меняются, значит, их можно сохранить и не считать заново. Ещё стоит сказать о цене: кэш занимает память GPU.',
      pieces: KV_ANSWER,
    },
    json: {
      name: 'explanation',
      schema: objectSchema({ term: { type: 'string' }, what: { type: 'string' }, why: { type: 'string' } }),
      value: {
        term: 'KV cache',
        what: 'Ключи и значения attention для уже обработанных токенов',
        why: 'Чтобы decode не пересчитывал весь текст на каждом шаге',
      },
      reasoning: 'Объяснение нужно уложить в три поля схемы: термин, что хранится и зачем. Пишу коротко.',
      intro: 'Конечно, вот объяснение в JSON:',
    },
  }
}

function toolSpec(): ScriptSpec {
  const city = 'Берлин'
  return {
    scenarioId: 'tool',
    answerSource: 'preset',
    simulatedNote: presetNote('Вызов инструмента'),
    text: {
      // tools включены всегда, когда нужен этот вариант, поэтому он не используется (см. tool и noTool)
      reasoning: '',
      pieces: [],
    },
    tool: {
      name: 'get_weather',
      arguments: { city },
      city,
      reasoning:
        'Пользователь спрашивает, какая погода в Берлине прямо сейчас. Свежих данных у меня нет, но есть инструмент get_weather. Вызову его для Берлина.',
      pieces: [
        p('<tool_call>', [
          { text: 'Сейчас', preview: 'Сейчас посмотрю погоду в Берлине.', delta: -1.4 },
          { text: 'К сожалению', preview: 'К сожалению, у меня нет доступа к погоде…', delta: -2.0 },
          { text: 'В', preview: 'В Берлине сейчас солнечно, около 18 градусов. (выдумка без инструмента)', delta: -2.6 },
          { text: '\n', delta: -3.6 },
        ]),
        p('\n{"name": "get_weather", "arguments": {"city": "'),
        p('Берлин', [{ text: 'Berlin', preview: 'Berlin: город по-английски', delta: -1.6 }]),
        p('"}}\n</tool_call>'),
      ],
    },
    noTool: {
      reasoning: 'Пользователь спрашивает о погоде прямо сейчас. Инструментов в запросе нет, а свежих данных у меня нет. Честно скажу об этом.',
      pieces: [
        p('У', [
          { text: 'К сожалению', preview: 'К сожалению, я не знаю, какая сейчас погода…', delta: -0.6 },
          { text: 'Я', preview: 'Я не могу проверить погоду…', delta: -0.9 },
          { text: 'В', preview: 'В Берлине сейчас солнечно… (выдумка без данных)', delta: -1.5 },
        ]),
        p(' меня нет доступа к данным о погоде в реальном времени, поэтому не могу сказать, что сейчас в Берлине. Посмотрите прогноз в погодном сервисе или приложении.'),
      ],
    },
    json: {
      name: 'weather_answer',
      schema: objectSchema({ city: { type: 'string' }, answer: { type: 'string' } }),
      value: { city, answer: 'Нет доступа к данным о погоде в реальном времени' },
      reasoning: 'Ответ нужен в JSON. Город: Берлин. Свежих данных о погоде у меня нет, так и напишу.',
    },
  }
}

function jsonSpec(): ScriptSpec {
  const value = { name: 'Анна', age: 29, city: 'Казань' }
  return {
    scenarioId: 'json',
    answerSource: 'preset',
    simulatedNote: presetNote('JSON по схеме'),
    text: {
      reasoning: 'Нужно извлечь имя, возраст и город и показать их в виде JSON. Город привожу к именительному падежу: Казань.',
      // без маски модель болтает и оборачивает JSON в блок кода: ровно то, что отсекает structured output
      pieces: [
        p('Конечно', [
          { text: '{', preview: 'сразу JSON', delta: -0.6 },
          { text: '```', preview: '```json', delta: -0.9 },
          { text: 'Вот', preview: 'Вот данные:', delta: -1.1 },
        ]),
        p(', вот JSON:\n```json\n' + pyJson(value) + '\n```'),
      ],
    },
    json: {
      name: 'person',
      schema: objectSchema({ name: { type: 'string' }, age: { type: 'integer' }, city: { type: 'string' } }),
      value,
      reasoning:
        'Нужно извлечь три поля: имя Анна, возраст 29 и город. В тексте написано «в Казани», а в JSON город ставлю в именительном падеже: Казань.',
      pieces: [
        p('{', chattyJsonAlts('Конечно, вот JSON:')),
        p('"name": "Анна", "age": '),
        p('29', [
          { text: '"', preview: '"29" строкой вместо числа', delta: 0.4 },
          { text: '3', delta: -4.2 },
          { text: '1', delta: -4.8 },
        ]),
        p(', "city": "Казан'),
        p('ь', [{ text: 'и', preview: '«Казани», как в тексте запроса', delta: -0.7 }]),
        p('"}'),
      ],
    },
  }
}

function longDocSpec(): ScriptSpec {
  const steps = [
    'Вывести узел из балансировки',
    'Дождаться дренирования, не дольше 10 минут',
    'Перезапустить узел и дождаться проверки готовности',
    'Вернуть узел в балансировку',
  ]
  return {
    scenarioId: 'custom',
    answerSource: 'preset',
    simulatedNote: presetNote(LONG_DOC_TAB.label),
    text: {
      reasoning:
        'Ответ есть в разделе 3 регламента. Перечислю шаги по порядку: вывод из балансировки, дренирование, перезапуск с проверкой готовности, возврат в балансировку.',
      pieces: [
        p('По', [
          { text: 'Сначала', preview: 'Сначала выведите узел из балансировки…', delta: -0.8 },
          { text: 'Узлы', preview: 'Узлы перезапускают по одному…', delta: -1.1 },
          { text: 'Чтобы', preview: 'Чтобы безопасно перезапустить узел…', delta: -1.5 },
        ]),
        p(
          ' регламенту узлы перезапускают по одному:\n' +
            '1. Выведите узел из балансировки, чтобы роутер перестал слать на него новые запросы.\n' +
            '2. Дождитесь дренирования: воркеры допишут активные ответы, обычно за минуту-две, максимум за десять минут.\n' +
            '3. Перезапустите узел и дождитесь, пока все воркеры пройдут проверку готовности.\n' +
            '4. Верните узел в балансировку. Первые минуты TTFT на нём будет выше: prefix cache ещё холодный.',
        ),
      ],
    },
    json: {
      name: 'restart_steps',
      schema: objectSchema({ steps: { type: 'array', items: { type: 'string' } } }),
      value: { steps },
      reasoning: 'Шаги перезапуска есть в разделе 3 регламента. Ответ нужен в JSON: массив шагов по порядку.',
      intro: 'Конечно, вот шаги в JSON:',
    },
  }
}

/* ============================================================================
 * Свой текст: темы по ключевым словам
 * ========================================================================== */

interface Topic {
  id: string
  label: string
  re: RegExp
  reasoning: string
  answer: string
  /** Короткий ответ для поля answer в JSON. */
  short: string
}

/** Темы в порядке проверки (первое совпадение). */
const TOPICS: readonly Topic[] = [
  {
    id: 'detokenization',
    label: 'детокенизация',
    re: /детокен|detoken/iu,
    reasoning: 'Вопрос про детокенизацию. Главное: текст отдаётся, только когда байты сложились в целые символы.',
    answer:
      'Детокенизация превращает номера сгенерированных токенов обратно в текст. Один символ, например эмодзи или буква кириллицы, может растянуться ' +
      'на байты нескольких токенов, поэтому детокенизатор придерживает неполные байты и отдаёт текст, только когда символ сложился. ' +
      'Служебные токены вроде конца реплики в текст не попадают.',
    short: 'Номера токенов снова становятся текстом; неполные символы придерживаются',
  },
  {
    id: 'paged-attention',
    label: 'PagedAttention',
    re: /paged[\s-]*attention|pagedattention|страничн/iu,
    reasoning: 'Вопрос про PagedAttention. Важно не спутать его с самим attention: меняется раскладка KV cache в памяти.',
    answer:
      'PagedAttention хранит KV cache не одним куском на запрос, а блоками (в vLLM по 16 токенов), как страницы виртуальной памяти. ' +
      'Блоки выделяются по мере роста последовательности, поэтому память почти не простаивает, а одинаковые начала промптов можно делить ' +
      'между запросами. Результат attention тот же, меняется только раскладка KV в памяти.',
    short: 'KV cache хранится блоками, как страницы памяти; сам attention не меняется',
  },
  {
    id: 'kv-tiers',
    label: 'уровни KV cache',
    re: /daemonset|(^|[^a-z0-9])l3([^a-z0-9]|$)|lmcache|mooncake|hicache|kvbm|offload|оффлоад|выгруз\S*\s+(kv|кэш|кеш)|кэш\S*\s+нод|уровн\S*\s+(kv|кэш|кеш)/iu,
    reasoning: 'Вопрос про уровни KV cache. Объясню выгрузку из GPU в память узла и общий кэш ноды как DaemonSet.',
    answer:
      'KV cache можно хранить не только в памяти GPU. Блоки, которые не помещаются или давно не нужны, выгружают уровнем ниже: в оперативную память узла, ' +
      'на локальный SSD или в общий кэш кластера. В Kubernetes такой кэш часто запускают как DaemonSet, по одному серверу на ноду (например, LMCache или Mooncake): ' +
      'поды движка подключаются к нему как клиенты, поэтому кэш переживает перезапуск движка. Достать блок из памяти узла дольше, чем из GPU, но обычно быстрее, ' +
      'чем заново посчитать его в prefill.',
    short: 'Выгрузка KV из GPU в RAM, на SSD и в общий кэш ноды (DaemonSet)',
  },
  {
    id: 'prefix-caching',
    label: 'prefix caching',
    re: /prefix|префикс|radix[\s-]*attention|radixattention/iu,
    reasoning: 'Вопрос про prefix caching. Главное: переиспользуется KV общего начала промптов, совпадение ищется строго от начала.',
    answer:
      'Prefix caching переиспользует KV cache общего начала промптов: если системный промпт уже посчитан, prefill пропускает его и считает только новое. ' +
      'Совпадение ищется строго от начала: vLLM сравнивает цепочку хэшей блоков, SGLang (RadixAttention) ищет префикс в radix-дереве токенов. ' +
      'Поэтому одна изменённая буква в начале промпта ломает весь кэш.',
    short: 'Переиспользует KV общего начала промптов; совпадение только от начала',
  },
  {
    id: 'kv-cache',
    label: 'KV cache',
    re: /kv[\s-]*(cache|кэш|кеш)|кэш\S*\s+ключ|кеш\S*\s+ключ/iu,
    reasoning: 'Вопрос про KV cache. Нужно сказать, что в нём хранится, зачем он decode и чем за него платят.',
    answer:
      'KV cache хранит ключи и значения attention для уже обработанных токенов. Благодаря ему каждый шаг decode считает только новый токен, а не весь текст заново. ' +
      'Платить приходится памятью GPU: кэш растёт с длиной контекста и с числом одновременных запросов.',
    short: 'Хранит ключи и значения attention, чтобы decode не пересчитывал весь текст',
  },
  {
    id: 'disaggregation',
    label: 'дезагрегация',
    re: /дезагрег|disagg|nixl/iu,
    reasoning: 'Вопрос про дезагрегированный serving. Главное: prefill и decode мешают друг другу на одних GPU, поэтому их разводят по пулам.',
    answer:
      'При дезагрегации prefill и decode работают на разных GPU. Prefill-воркер считает KV cache промпта и передаёт его decode-воркеру по быстрому каналу через NIXL. ' +
      'Так длинные промпты не тормозят генерацию остальных запросов, но приходится платить передачей KV и более сложной схемой.',
    short: 'Prefill и decode на разных GPU, KV передаётся через NIXL',
  },
  {
    id: 'roofline',
    label: 'compute- и memory-bound',
    re: /memory[\s-]*bound|compute[\s-]*bound|roofline|упира\S*\s+в\s+(памят|вычислен)/iu,
    reasoning: 'Вопрос про memory-bound и compute-bound. Сравню, сколько вычислений приходится на прочитанный байт в prefill и в decode.',
    answer:
      'Memory-bound значит, что GPU считает быстрее, чем успевает читать данные из памяти. Decode на каждом шаге читает все веса модели ради одного токена ' +
      'на запрос, поэтому упирается в полосу памяти. Prefill обрабатывает сотни токенов за то же чтение весов и на длинных промптах упирается в вычисления.',
    short: 'Decode упирается в чтение памяти, prefill на длинных промптах в вычисления',
  },
  {
    id: 'speculative',
    label: 'speculative decoding',
    re: /спекулятив|speculative|eagle/iu,
    reasoning: 'Вопрос про speculative decoding. Объясню идею черновика и проверки за один проход.',
    answer:
      'Speculative decoding ускоряет decode: черновик (маленькая модель или дополнительная голова) угадывает несколько токенов вперёд, а большая модель проверяет их за один проход. ' +
      'Принятые токены почти бесплатны, потому что decode упирается в память, а вычислители простаивают.',
    short: 'Черновик угадывает токены вперёд, большая модель проверяет их за один проход',
  },
  {
    id: 'prefill-decode',
    label: 'prefill и decode',
    re: /prefill|decode|префил|декод/iu,
    reasoning: 'Вопрос про две фазы генерации. Сравню, что делает каждая и во что упирается.',
    answer:
      'Генерация идёт в две фазы. Prefill обрабатывает весь промпт за один проход и выдаёт первый токен: на каждый прочитанный байт весов здесь много вычислений. ' +
      'Decode пишет ответ по одному токену и на каждом шаге читает все веса и KV cache, поэтому упирается в пропускную способность памяти.',
    short: 'Prefill читает промпт целиком, decode пишет ответ по одному токену',
  },
  {
    id: 'metrics',
    label: 'метрики латентности',
    re: /ttft|(^|[^a-z])itl([^a-z]|$)|латентност|задержк|метрик|перв\S*\s+токен/iu,
    reasoning: 'Вопрос про метрики. Разведу TTFT и ITL и скажу, от чего зависит каждая.',
    answer:
      'TTFT показывает, сколько ждать первого токена: его определяют сеть, очередь и prefill. ITL показывает паузу между соседними токенами, её задаёт decode. ' +
      'Для чата важны обе: TTFT отвечает за первое впечатление, ITL за скорость печати.',
    short: 'TTFT зависит от очереди и prefill, ITL от скорости decode',
  },
  {
    id: 'sampling',
    label: 'сэмплинг',
    re: /температур|temperature|сэмпл|семпл|sampling|top[\s_-]?[pk](?![a-z])|следующ\S*\s+токен/iu,
    reasoning: 'Вопрос про выбор следующего токена. Объясню логиты, softmax и temperature.',
    answer:
      'На каждом шаге модель выдаёт оценку (логит) для каждого токена словаря. Сэмплер делит логиты на temperature, превращает их в вероятности через softmax ' +
      'и выбирает один токен: при temperature 0 всегда самый вероятный, при большей ответ разнообразнее. Top-p и top-k заранее отрезают маловероятный хвост.',
    short: 'Логиты делятся на temperature, softmax даёт вероятности, сэмплер выбирает токен',
  },
  {
    id: 'attention',
    label: 'attention',
    re: /attention|внимани|q\s*\/\s*k\s*\/\s*v|qkv|(^|[^a-z])q\s*,\s*k\s*(и|,|and)\s*v([^a-z]|$)/iu,
    reasoning: 'Вопрос про attention. Объясню через запросы, ключи и значения, без формул.',
    answer:
      'Attention позволяет каждому токену собрать информацию из предыдущих. Токен строит запрос (Q), сравнивает его с ключами (K) прошлых токенов ' +
      'и по полученным весам смешивает их значения (V). Так модель понимает, к чему относится каждое слово.',
    short: 'Каждый токен взвешенно собирает информацию из предыдущих через Q, K и V',
  },
  {
    id: 'template',
    label: 'шаблон чата',
    re: /шаблон|template|jinja/iu,
    reasoning: 'Вопрос про шаблон чата. Главное: модель видит один текст со служебными маркерами, а не JSON.',
    answer:
      'Модель видит не JSON с сообщениями, а один длинный текст. Шаблон чата склеивает сообщения в этот текст и расставляет служебные маркеры начала и конца ' +
      'каждой реплики с ролью. Шаблон берётся из файлов модели: с чужим шаблоном модель отвечает хуже и может не остановиться вовремя.',
    short: 'Склеивает сообщения в один текст со служебными маркерами ролей',
  },
  {
    id: 'parsers',
    label: 'парсеры',
    re: /парсер|parser|tool[\s_-]?call|reasoning_content|вызов\S*\s+(инструмент|функци)/iu,
    reasoning: 'Вопрос про парсеры ответа. Объясню, как один поток текста раскладывается по полям.',
    answer:
      'Модель пишет один поток текста, а клиенту нужны отдельные поля. Парсеры на сервере режут этот поток: рассуждение уходит в reasoning_content ' +
      '(в vLLM это поле называется reasoning), вызов инструмента разбирается в tool_calls с именем и JSON-аргументами, остальное идёт в content. ' +
      'Если парсер не включён или выбран не тот, служебные теги попадут клиенту прямо в текст.',
    short: 'Раскладывают поток текста модели по полям reasoning, tool_calls и content',
  },
  {
    id: 'structured',
    label: 'structured output',
    re: /json|structured|структурирован|грамматик|grammar/iu,
    reasoning: 'Вопрос про structured output. Главное: маска на логиты перед сэмплингом.',
    answer:
      'Structured output заставляет модель отвечать строго по схеме, например валидным JSON. На каждом шаге decode сервер вычисляет, какие токены допускает грамматика, ' +
      'и запрещает остальные до сэмплинга. Содержание выбирает модель, а форму гарантирует маска.',
    short: 'Маска грамматики запрещает токены, которые сломали бы JSON по схеме',
  },
  {
    id: 'tokenization',
    label: 'токенизация',
    re: /токениз|tokeniz|(^|[^a-z])bpe([^a-z]|$)|словар[ьяеюи]|что\s+такое\s+токен|what\s+is\s+a\s+token/iu,
    reasoning: 'Вопрос про токенизацию. Расскажу, что такое токен и почему русский текст длиннее в токенах.',
    answer:
      'Модель работает с числами, поэтому текст сначала режется на токены: частые слова и куски слов из словаря. У Qwen3 в словаре 151 669 токенов, у каждого свой номер. ' +
      'Русский текст обычно занимает больше токенов, чем такой же по смыслу английский.',
    short: 'Текст режется на куски из словаря, и каждый кусок получает свой номер',
  },
  {
    id: 'batching',
    label: 'batching',
    re: /батч|batch/iu,
    reasoning: 'Вопрос про batching. Главное: веса читаются один раз на весь batch, а continuous batching не ждёт самый длинный запрос.',
    answer:
      'GPU выгоднее обрабатывать много запросов сразу: за один шаг веса читаются из памяти один раз для всего batch. ' +
      'Continuous batching добавляет и убирает запросы на каждом шаге decode, поэтому освободившееся место сразу занимает следующий запрос.',
    short: 'Много запросов в одном шаге: веса читаются один раз на весь batch',
  },
  {
    id: 'quantization',
    label: 'квантизация',
    re: /квантиз|квантован|quantiz|(^|[^a-z])(fp8|fp4|int8|int4|bf16)([^a-z0-9]|$)/iu,
    reasoning: 'Вопрос про квантизацию. Скажу, что меняется в памяти и скорости и чем рискуем.',
    answer:
      'Квантизация хранит веса, а иногда и KV cache, в меньшем числе бит: FP8 или FP4 вместо BF16. Модель занимает меньше памяти, а decode, который упирается в чтение весов, ускоряется. ' +
      'Качество стоит проверить на своих задачах: небольшие потери возможны.',
    short: 'Меньше бит на вес: меньше памяти и быстрее decode, но качество надо проверять',
  },
  {
    id: 'routing',
    label: 'роутинг',
    re: /роут|маршрутиз|routing|router|балансир/iu,
    reasoning: 'Вопрос про роутинг запросов. Главное: KV-aware роутер взвешивает совпадение префикса и нагрузку.',
    answer:
      'KV-aware роутер отправляет запрос туда, где уже лежит KV cache его префикса: тогда prefill можно сократить. ' +
      'Но он учитывает и нагрузку, поэтому занятый воркер с кэшем может проиграть свободному. В Dynamo оба фактора сведены в одну формулу стоимости.',
    short: 'Выбирает воркер по совпадению префикса в KV cache и по нагрузке',
  },
  {
    id: 'dynamo',
    label: 'NVIDIA Dynamo',
    re: /dynamo|динамо/iu,
    reasoning: 'Вопрос про Dynamo. Опишу его как слой над движками и перечислю главные части.',
    answer:
      'NVIDIA Dynamo работает поверх движков vLLM, SGLang и TensorRT-LLM. Он принимает запросы через OpenAI-совместимый frontend, сам применяет шаблон и токенизирует текст, ' +
      'выбирает воркер с учётом KV cache и умеет разносить prefill и decode по разным GPU, передавая KV через NIXL.',
    short: 'Распределённый слой над движками: frontend, KV-aware роутер, дезагрегация',
  },
]

const GREETING_RE = /^\s*(привет|здравствуй(те)?|добрый\s+(день|вечер)|доброе\s+утро|hi|hello|hey)(?=$|[\s,.!?])/iu
const WEATHER_RE = /погод|weather/iu
/**
 * Одно действие с двумя числами. Слева от первого числа и справа от второго не должно быть ещё одного действия:
 * «2 + 2 * 3» заготовка не считает (там два действия), а двоеточие после слова («Сколько будет: 2 + 2?») не мешает.
 */
const ARITH_RE =
  /(?<![\p{L}\d.,])(?<!\d\s*[+\-−*×·/:]\s*)(-?\d{1,9})\s*([+\-−*×·/:])\s*(-?\d{1,9})(?![\p{L}\d])(?!\s*[+\-−*×·/:]\s*-?\d)/u
/** Пример из нескольких действий подряд («10 - 3 - 2»). */
const ARITH_MULTI_RE = /\d\s*[+\-−*×·/:]\s*-?\d+\s*[+\-−*×·/:]\s*-?\d/u
const ARITH_INTENT_RE = /сколько|посчитай|вычисли|чему\s+равн|=|^[\s\d+\-−*×·/:().=?]+$/iu

/** Город для get_weather: основа слова → именительный падеж (простой словарь; падежи по основе). */
const CITIES: ReadonlyArray<[RegExp, string]> = [
  [/москв|moscow/iu, 'Москва'],
  [/петербург|питер/iu, 'Санкт-Петербург'],
  [/казан/iu, 'Казань'],
  [/новосибирск/iu, 'Новосибирск'],
  [/екатеринбург/iu, 'Екатеринбург'],
  [/нижн\S*\s+новгород/iu, 'Нижний Новгород'],
  [/самар/iu, 'Самара'],
  [/(^|[^\p{L}])перм/iu, 'Пермь'],
  [/(^|[^\p{L}])уф(а|е|у|ы|ой)(?=$|[^\p{L}])/iu, 'Уфа'],
  [/челябинск/iu, 'Челябинск'],
  [/(^|[^\p{L}])омск/iu, 'Омск'],
  [/(^|[^\p{L}])томск/iu, 'Томск'],
  [/ростов/iu, 'Ростов-на-Дону'],
  [/красноярск/iu, 'Красноярск'],
  [/воронеж/iu, 'Воронеж'],
  [/волгоград/iu, 'Волгоград'],
  [/краснодар/iu, 'Краснодар'],
  [/владивосток/iu, 'Владивосток'],
  [/калининград/iu, 'Калининград'],
  [/сочи/iu, 'Сочи'],
  [/минск/iu, 'Минск'],
  [/берлин|berlin/iu, 'Берлин'],
  [/париж|paris/iu, 'Париж'],
  [/лондон|london/iu, 'Лондон'],
  [/мадрид|madrid/iu, 'Мадрид'],
  [/(^|[^\p{L}])праг(а|е|у|и|ой)(?=$|[^\p{L}])|prague/iu, 'Прага'],
  [/(^|[^\p{L}])риг(а|е|у|и|ой)(?=$|[^\p{L}])|(^|[^\p{L}])riga(?=$|[^\p{L}])/iu, 'Рига'],
  [/тбилиси|tbilisi/iu, 'Тбилиси'],
  [/ереван|yerevan/iu, 'Ереван'],
  [/алмат|almaty/iu, 'Алматы'],
  [/ташкент|tashkent/iu, 'Ташкент'],
  [/киев|kyiv|kiev/iu, 'Киев'],
  [/варшав|warsaw/iu, 'Варшава'],
  [/стамбул|istanbul/iu, 'Стамбул'],
  [/пекин|beijing/iu, 'Пекин'],
  [/сеул|seoul/iu, 'Сеул'],
  [/токио|tokyo/iu, 'Токио'],
  [/нью[\s-]?йорк|new york/iu, 'Нью-Йорк'],
  [/(^|[^\p{L}])рим(е|а)?(?=$|[^\p{L}])/iu, 'Рим'],
]

/** Город в тексте (именительный падеж) или null. */
export function findCity(text: string): string | null {
  for (const [re, city] of CITIES) if (re.test(text)) return city
  return null
}

/** Слова с большой буквы после «в», которые не бывают городом («погода в Понедельник»). */
const NOT_CITY_RE = /^(понедельник|вторник|сред[ау]|четверг|пятниц[ау]|суббот[ау]|воскресенье|выходные|январ|феврал|март|апрел|ма[йея]|июн|июл|август|сентябр|октябр|ноябр|декабр|monday|tuesday|wednesday|thursday|friday|saturday|sunday)/iu

/**
 * Город, которого нет в словаре: слово (или два) с большой буквы после «в», «во», «in», «for», ровно как в тексте.
 * Падеж не угадываем: «в Самаре» дало бы «Самаре», поэтому частые города лежат в словаре CITIES.
 */
export function guessCity(text: string): string | null {
  const m = /(?:^|[^\p{L}])(?:[вВ]|[вВ]о|[iI]n|[fF]or)\s+([A-ZА-ЯЁ][\p{L}-]+(?:\s+[A-ZА-ЯЁ][\p{L}-]+)?)/u.exec(text)
  if (!m) return null
  const city = m[1].trim()
  return NOT_CITY_RE.test(city) ? null : city
}

/** Короткая фраза темы для общего ответа: «что такое X» → X, иначе начало вопроса. */
export function extractTopicPhrase(text: string): string {
  let s = sanitizeEcho(text).replace(/[?!.…\s]+$/u, '')
  const patterns: RegExp[] = [
    /(?:что\s+такое|что\s+значит|что\s+означает)\s+(.+)/iu,
    /(?:как\s+работа\S*|как\s+устроен\S*)\s+(.+)/iu,
    /(?:зачем\s+нуж(?:ен|на|но|ны)\S*|зачем)\s+(.+)/iu,
    /(?:расскажи|объясни|опиши)(?:\s+(?:мне|пожалуйста|простыми\s+словами|подробно|кратко|коротко))*\s+(?:про|о|об)\s+(.+)/iu,
  ]
  for (const re of patterns) {
    const m = re.exec(s)
    if (m && m[1].trim()) {
      s = m[1].trim()
      break
    }
  }
  s = s.replace(/[«»"]/g, '').trim()
  if (s.length > 60) {
    const cut = s.slice(0, 60)
    const sp = cut.lastIndexOf(' ')
    s = (sp > 20 ? cut.slice(0, sp) : cut) + '…'
  }
  return s || 'ваш вопрос'
}

function topicNote(label: string): string {
  return (
    `Ответ симулирован: это заготовка по теме «${label}», подобранная по ключевым словам вашего сообщения. ` +
    'Шаблон, токены, хэши блоков, решение роутера и время prefill и decode посчитаны для вашего текста по-настоящему.'
  )
}

/** Число по-русски: десятичная запятая, без группировки (так пишет модель). */
function ruNumber(x: number): string {
  const r = Math.round(x * 100) / 100
  return String(r).replace('.', ',')
}

const customJson = (topic: string, answer: string): Pick<ScriptSpec['json'], 'name' | 'schema' | 'value'> => ({
  name: 'answer',
  schema: objectSchema({ topic: { type: 'string' }, answer: { type: 'string' } }),
  value: { topic, answer },
})

/** Тема без приветствия: погода, арифметика, темы по ключевым словам или общий ответ. */
function detectTopicBody(text: string): { id: string; label: string } {
  if (WEATHER_RE.test(text)) return { id: 'weather', label: 'погода' }
  if (ARITH_INTENT_RE.test(text)) {
    if (ARITH_RE.test(text)) return { id: 'arithmetic', label: 'арифметика' }
    if (ARITH_MULTI_RE.test(text)) return { id: 'arithmetic-multi', label: 'арифметика' }
  }
  for (const t of TOPICS) if (t.re.test(text)) return { id: t.id, label: t.label }
  return { id: 'generic', label: extractTopicPhrase(text) }
}

/** Светская беседа после приветствия («как дела?», «меня зовут…»): на неё отвечает заготовка приветствия. */
const SMALL_TALK_RE =
  /^[^\p{L}]*(как\s+(у\s+(тебя|вас)\s+)?(дела|ты|вы|жизнь|поживаешь|поживаете|настроение)|что\s+нового|меня\s+зовут|спасибо|благодарю|рада?\s+(тебя\s+|вас\s+)?(видеть|познакомиться)|how\s+are\s+you|what'?s\s+up|thanks|thank\s+you|my\s+name\s+is)/iu

/** Тема своего текста (для подписи и тестов): id и подпись. */
export function detectTopic(text: string): { id: string; label: string } {
  if (!sanitizeEcho(text)) return { id: 'empty', label: 'пустое сообщение' }
  const g = GREETING_RE.exec(text)
  if (g) {
    // «Привет! Что такое KV cache?»: тема по тексту после приветствия. Вопрос без заготовки получает общий ответ
    // («Привет! Что такое vLLM?»), иначе модель ответила бы «Привет! Чем могу помочь?» и не заметила вопроса.
    // Приветствие остаётся приветствием, если после него ничего нет или там светская беседа («как дела?»).
    const rest = text.slice(g.index + g[0].length)
    if (/\p{L}{3,}/u.test(rest)) {
      const t = detectTopicBody(rest)
      if (t.id !== 'generic') return t
      if (!SMALL_TALK_RE.test(rest)) return { id: 'generic', label: extractTopicPhrase(rest.replace(/^[\s\p{P}]+/u, '')) }
    }
    return { id: 'greeting', label: 'приветствие' }
  }
  return detectTopicBody(text)
}

/** Вопрос в основном латиницей: заготовки только русские, это стоит сказать. */
function mostlyLatin(text: string): boolean {
  const lat = (text.match(/[A-Za-z]/g) ?? []).length
  const cyr = (text.match(/[А-Яа-яЁё]/g) ?? []).length
  return lat >= 3 && lat > cyr * 2
}

function customSpec(userText: string): ScriptSpec {
  const spec = customSpecBase(userText)
  if (mostlyLatin(userText)) {
    spec.simulatedNote += ' Заготовки ответов написаны только на русском, поэтому на английский вопрос «модель» отвечает по-русски; живая модель ответила бы на языке вопроса.'
  }
  return spec
}

function customSpecBase(userText: string): ScriptSpec {
  const topic = detectTopic(userText)
  const textSpec = (reasoning: string, answer: string, alts?: AltSpec[]) => ({
    reasoning,
    pieces: [p(answer.slice(0, 1), alts), p(answer.slice(1))].filter((x) => x.text.length > 0),
  })

  if (topic.id === 'empty') {
    const answer = 'Сообщение пустое: напишите вопрос, и я отвечу.'
    return {
      scenarioId: 'custom',
      answerSource: 'topic',
      topic,
      simulatedNote: topicNote(topic.label),
      text: textSpec('Сообщение пустое. Попрошу задать вопрос.', answer),
      json: { ...customJson('пустое сообщение', answer), reasoning: 'Сообщение пустое. Ответ нужен в JSON: тема и короткий ответ.' },
    }
  }

  if (topic.id === 'greeting') {
    const answer = 'Привет! Чем могу помочь?'
    return {
      scenarioId: 'custom',
      answerSource: 'topic',
      topic,
      simulatedNote: topicNote(topic.label),
      text: textSpec('Пользователь поздоровался. Отвечу коротко и предложу помощь.', answer, [
        { text: 'Здравствуйте', preview: 'Здравствуйте! Чем могу помочь?', delta: -0.6 },
        { text: 'Добрый', preview: 'Добрый день!', delta: -1.4 },
      ]),
      json: { ...customJson('приветствие', answer), reasoning: 'Пользователь поздоровался. Ответ нужен в JSON: тема и короткий ответ.' },
    }
  }

  if (topic.id === 'weather') {
    const known = findCity(userText)
    const guessed = known ? null : guessCity(userText)
    const city = known ?? guessed
    const noData = 'У меня нет доступа к данным о погоде в реальном времени. Посмотрите прогноз в погодном сервисе.'
    const spec: ScriptSpec = {
      scenarioId: 'custom',
      answerSource: 'topic',
      topic,
      toolRelevant: true,
      simulatedNote: topicNote(topic.label),
      text: city
        ? textSpec('Вопрос о погоде, но инструментов в запросе нет. Честно скажу, что свежих данных нет.', noData)
        : textSpec('Пользователь спрашивает о погоде, но не назвал город. Уточню город, прежде чем вызывать get_weather.', 'Уточните, пожалуйста, город, и я узнаю погоду.'),
      noTool: textSpec('Вопрос о погоде, но инструментов в запросе нет. Честно скажу, что свежих данных нет.', noData),
      json: {
        ...customJson('погода', 'Нет доступа к данным о погоде в реальном времени'),
        reasoning: 'Вопрос о погоде. Ответ нужен в JSON: тема и короткий ответ.',
      },
    }
    if (city) {
      spec.tool = {
        name: 'get_weather',
        arguments: { city },
        city,
        reasoning: `Пользователь спрашивает о погоде, город: ${city}. Нужны свежие данные, поэтому вызову get_weather.`,
        ...(guessed
          ? { note: 'Города нет в словаре симулятора: аргумент взят как в тексте, живая модель привела бы его к именительному падежу.' }
          : {}),
      }
    }
    return spec
  }

  if (topic.id === 'arithmetic-multi') {
    const answer = 'Заготовка считает только одно действие, например 17 × 23.'
    return {
      scenarioId: 'custom',
      answerSource: 'topic',
      topic,
      simulatedNote:
        'Ответ симулирован: заготовка распознаёт пример только с одним действием. Шаблон, токены, хэши блоков, решение роутера и время prefill и decode посчитаны для вашего текста по-настоящему.',
      text: textSpec('В примере несколько действий, а заготовка симулятора считает только одно. Так и скажу.', answer),
      json: { ...customJson('арифметика', answer.replace(/\.$/, '')), reasoning: 'В примере несколько действий. Ответ нужен в JSON: тема и короткий ответ.' },
    }
  }

  if (topic.id === 'arithmetic') {
    const m = ARITH_RE.exec(userText)!
    const a = Number(m[1])
    const b = Number(m[3])
    const opRaw = m[2]
    const op = opRaw === '−' ? '-' : opRaw === '×' || opRaw === '·' || opRaw === '*' ? '×' : opRaw === ':' ? '/' : opRaw
    let result: string
    if (op === '/' && b === 0) result = ''
    else if (op === '+') result = ruNumber(a + b)
    else if (op === '-') result = ruNumber(a - b)
    else if (op === '×') result = ruNumber(a * b)
    else result = ruNumber(a / b)
    const expr = `${a} ${op === '×' ? '×' : op === '/' ? ':' : op} ${b}`
    const answer = result ? `${expr} = ${result}.` : 'На ноль делить нельзя.'
    return {
      scenarioId: 'custom',
      answerSource: 'topic',
      topic,
      simulatedNote:
        'Ответ симулирован, но пример посчитан по-настоящему: такие вопросы распознаются отдельно. Шаблон, токены, хэши, роутинг и время prefill и decode посчитаны для вашего текста.',
      text: textSpec(result ? `Нужно посчитать: ${expr} = ${result}. Отвечу одной строкой.` : 'Деление на ноль не определено, так и скажу.', answer),
      json: { ...customJson('арифметика', answer.replace(/\.$/, '')), reasoning: 'Нужно посчитать пример и вернуть JSON: тема и ответ.' },
    }
  }

  const t = TOPICS.find((x) => x.id === topic.id)
  if (t) {
    return {
      scenarioId: 'custom',
      answerSource: 'topic',
      topic,
      simulatedNote: topicNote(t.label),
      text: textSpec(t.reasoning, t.answer),
      json: { ...customJson(t.label, t.short), reasoning: `${t.reasoning} Ответ нужен в JSON: тема и короткий ответ.` },
    }
  }

  const phrase = topic.label
  const answer =
    `Вы спрашиваете про «${phrase}». Это симулированный ответ: живая модель в браузере не запускается, поэтому содержание здесь условное. ` +
    'Зато путь запроса посчитан для вашего текста по-настоящему: шаблон, токены, блоки KV cache, выбор воркера и время каждого шага.'
  return {
    scenarioId: 'custom',
    answerSource: 'generic',
    topic: { id: 'generic', label: phrase },
    simulatedNote:
      'Ответ симулирован: готовой заготовки для такой темы нет, поэтому модель «отвечает» общим текстом. ' +
      'Шаблон, токены, хэши блоков, решение роутера и время prefill и decode посчитаны для вашего текста по-настоящему.',
    text: textSpec(`Вопрос про «${phrase}». Отвечу коротко.`, answer),
    json: { ...customJson(phrase, 'Симулированный ответ: живая модель здесь не запускается'), reasoning: `Вопрос про «${phrase}». Ответ нужен в JSON: тема и короткий ответ.` },
  }
}

/* ============================================================================
 * Главная функция
 * ========================================================================== */

/**
 * Заскриптованный ответ модели для настроек: сценарий (resolveScenario) и переключатели thinking / tools / json.
 * Любое сочетание переключателей даёт осмысленный ответ:
 *  - json: тело это JSON по схеме сценария (вызов инструмента невозможен, его отрежет маска);
 *  - tools и вопрос, которому нужен инструмент: вызов get_weather;
 *  - иначе текст (при tools модель видит описание инструмента, но не вызывает его);
 *  - thinking: перед телом рассуждение (в flow.ts оно оборачивается в <think>…</think>).
 */
export function scriptFor(settings: FlowSettings): Script {
  const id = resolveScenario(settings)
  const t: ScenarioToggles = { thinking: settings.thinking, tools: settings.tools, json: settings.json }
  switch (id) {
    case 'simple':
      return assemble(simpleSpec(), t)
    case 'thinking':
      return assemble(thinkingSpec(), t)
    case 'tool': {
      // У пресета с инструментом текстовый вариант есть только «без tools» (noTool); с tools всегда вызов.
      return assemble(toolSpec(), t)
    }
    case 'json':
      return assemble(jsonSpec(), t)
    default:
      // вкладка «Длинный документ»: свой текст с длинным system prompt, ответ заготовлен по регламенту
      if (isLongDoc(settings)) return assemble(longDocSpec(), t)
      return assemble(customSpec(settings.userText), t)
  }
}
