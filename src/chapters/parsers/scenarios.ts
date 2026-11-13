/**
 * Данные и чистые помощники главы «Парсеры»: сценарии вывода модели, разбиение потока на куски,
 * разбор вывода разных семейств в поля OpenAI-ответа. Тексты ответов условные, токены настоящие.
 */
import { incrementalDetokenize } from '@/engine/detok'
import { harmonyToChannels, runStreamingParser, type ParserEventEx } from '@/engine/parsers'
import { FAMILY_SPECIAL_TOKENS, familyInfo, renderTemplate } from '@/engine/templates'
import type { ModelFamily, Token, Tokenizer, ToolDef } from '@/engine/types'

/* ---------------------------------------------------------------------------------------------
 * Сценарии Qwen3 для фигуры с машиной состояний
 * ------------------------------------------------------------------------------------------- */

export type ScenarioId = 'answer' | 'tool'

export const SCENARIOS: ReadonlyArray<{ id: ScenarioId; label: string; text: string; question: string }> = [
  {
    id: 'answer',
    label: 'Мысли и ответ',
    text: '<think>\nНужна аналогия.\n</think>\n\nKV cache хранит ключи и значения прошлых токенов.<|im_end|>',
    question: 'Что хранит KV cache? Объясни просто.',
  },
  {
    id: 'tool',
    label: 'Мысли и вызов инструмента',
    text: '<think>\nНужен точный расчёт.\n</think>\n\n<tool_call>\n{"name": "kv_cache_size", "arguments": {"context": 8192}}\n</tool_call><|im_end|>',
    question: 'Сколько памяти займёт KV cache на контекст 8192 токена?',
  },
]

/** Инструмент из сценария с вызовом: без него в запросе модель не смогла бы вызвать kv_cache_size (описание условное). */
export const KV_CACHE_TOOL: ToolDef = {
  type: 'function',
  function: {
    name: 'kv_cache_size',
    description: 'Считает объём KV cache в байтах для заданной длины контекста.',
    parameters: {
      type: 'object',
      properties: { context: { type: 'integer', description: 'Длина контекста в токенах' } },
      required: ['context'],
    },
  },
}

/**
 * Длина промпта сценария в токенах по настоящему шаблону Qwen3 и словарю Qwen3: вопрос пользователя,
 * а в сценарии с вызовом ещё и описание kv_cache_size в tools. tools: сколько токенов добавило описание инструмента.
 */
export function promptTokens(tokenizer: Tokenizer, id: ScenarioId): { total: number; tools: number } {
  const sc = SCENARIOS.find((s) => s.id === id)!
  const count = (tools?: ToolDef[]) =>
    tokenizer.encode(renderTemplate({ family: 'qwen3', messages: [{ role: 'user', content: sc.question }], tools }).text).length
  const total = count(id === 'tool' ? [KV_CACHE_TOOL] : undefined)
  return { total, tools: id === 'tool' ? total - count() : 0 }
}

export const EOS_LITERAL = '<|im_end|>'
export const QWEN3_EOS_ID = 151645

export type ChunkMode = 'tokens' | 'chars'

export interface StreamInput {
  /** Куски текста, которые получает парсер (после детокенизации). */
  chunks: string[]
  /** В режиме «по токенам»: токен, давший кусок (того же индекса). */
  tokens: Token[] | null
  /** Токен EOS в конце (в режиме токенов). */
  eos: Token | null
}

/** Делит строку на куски по n code point. */
export function splitByChars(text: string, n: number): string[] {
  const cps = [...text]
  const out: string[] = []
  for (let i = 0; i < cps.length; i += n) out.push(cps.slice(i, i + n).join(''))
  return out
}

/**
 * Поток для парсера. «По токенам»: настоящие токены Qwen3 через инкрементальный детокенизатор
 * (EOS заканчивает поток без текста). «Кусками»: текст режется по 3 символа, теги приходят частями.
 */
export function buildStream(tokenizer: Tokenizer, text: string, mode: ChunkMode): StreamInput {
  const body = text.endsWith(EOS_LITERAL) ? text.slice(0, -EOS_LITERAL.length) : text
  if (mode === 'chars') return { chunks: splitByChars(body, 3), tokens: null, eos: null }
  const all = tokenizer.encode(text)
  const steps = incrementalDetokenize(all, tokenizer, { stopTokenIds: [QWEN3_EOS_ID] })
  const chunks: string[] = []
  const tokens: Token[] = []
  let eos: Token | null = null
  for (const s of steps) {
    if (s.stopped && s.token.id === QWEN3_EOS_ID) {
      eos = s.token
      break
    }
    chunks.push(s.emitted)
    tokens.push(s.token)
  }
  return { chunks, tokens, eos }
}

/** Что накопилось в полях ответа к концу шага (включительно) и что добавил сам шаг. */
export interface LaneState {
  reasoning: string
  content: string
  toolName: string | null
  toolArgs: string
  added: { reasoning: string; content: string; toolName: boolean; toolArgs: string }
}

export function lanesAt(events: ParserEventEx[], step: number): LaneState {
  const s: LaneState = { reasoning: '', content: '', toolName: null, toolArgs: '', added: { reasoning: '', content: '', toolName: false, toolArgs: '' } }
  events.slice(0, step + 1).forEach((ev, i) => {
    const last = i === step
    for (const it of ev.out) {
      if (it.kind === 'tool') {
        if (it.delta.name !== undefined) {
          s.toolName = it.delta.name
          if (last) s.added.toolName = true
        }
        if (it.delta.arguments !== undefined) {
          s.toolArgs += it.delta.arguments
          if (last) s.added.toolArgs += it.delta.arguments
        }
      } else if (it.kind === 'reasoning') {
        s.reasoning += it.text
        if (last) s.added.reasoning += it.text
      } else {
        s.content += it.text
        if (last) s.added.content += it.text
      }
    }
  })
  return s
}

/* ---------------------------------------------------------------------------------------------
 * Семейства моделей: что печатает модель и что получает клиент
 * ------------------------------------------------------------------------------------------- */

export type OutputKind = 'answer' | 'tool'

export interface ParsedOutput {
  reasoning: string
  content: string
  toolCalls: Array<{ name: string; arguments: string }>
}

export interface FamilyCase {
  family: ModelFamily
  raw: Record<OutputKind, string>
  /** Как разбирает сервер (коротко, по-русски). */
  how: string
  /** Подпись варианта «вызов инструмента», если вызовы умеет только отдельная версия модели. */
  toolLabel?: string
  /** Флаги vLLM и Dynamo для варианта «Ответ», если общие флаги семейства из движка к нему не подходят. */
  answerFlags?: { vllm: string; dynamo: string }
  /** Модель и флаги vLLM и Dynamo для варианта с вызовом, если они не совпадают с общими флагами семейства. */
  toolFlags?: { model: string; vllm: string; dynamo: string }
}

/** Примеры вывода (иллюстрация; форматы по notes-tokens §1.3, §3.2, §3.9). */
export const FAMILY_CASES: Record<ModelFamily, FamilyCase> = {
  qwen3: {
    family: 'qwen3',
    raw: {
      answer: SCENARIOS[0].text,
      tool: SCENARIOS[1].text,
    },
    how: 'Мысли идут в <think>…</think>, вызов записан JSON внутри <tool_call>. Оба тега являются отдельными токенами словаря с special=false, поэтому доходят до парсера в тексте.',
  },
  'deepseek-r1': {
    family: 'deepseek-r1',
    raw: {
      answer: 'Нужна аналогия.\n</think>\n\nKV cache хранит ключи и значения прошлых токенов.<｜end▁of▁sentence｜>',
      // R1-0528: шаблон не дописывает <think>, модель печатает его сама
      tool: '<think>\nНужен расчёт.\n</think>\n\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>kv_cache_size\n```json\n{"context": 8192}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>',
    },
    how: 'Шаблон DeepSeek-R1 сам дописывает <think>\\n в конец промпта, поэтому вывод начинается сразу с мыслей и содержит только закрывающий </think>: парсеру нужен режим forced reasoning. Вызовы инструментов официально умеет версия R1-0528: формат тот же, что у DeepSeek-V3, со своими спецтокенами с полноширинной чертой «｜» (парсер deepseek_v3, в vLLM ещё нужен --chat-template examples/tool_chat_template_deepseekr1.jinja). Шаблон R1-0528 уже не дописывает <think>: модель печатает его сама.',
    toolLabel: 'Вызов инструмента (R1-0528)',
    // исходная R1 вызовы официально не поддерживает (парсер deepseek_v3 в vLLM и Dynamo рассчитан на V3-0324 и R1-0528):
    // для «Ответа» только парсер рассуждений
    answerFlags: { vllm: '--reasoning-parser deepseek_r1', dynamo: '--dyn-reasoning-parser deepseek_r1' },
    toolFlags: {
      model: 'deepseek-ai/DeepSeek-R1-0528',
      vllm: '--reasoning-parser deepseek_r1 --enable-auto-tool-choice --tool-call-parser deepseek_v3 --chat-template examples/tool_chat_template_deepseekr1.jinja',
      dynamo: '--dyn-reasoning-parser deepseek_r1 --dyn-tool-call-parser deepseek_v3',
    },
  },
  'gpt-oss': {
    family: 'gpt-oss',
    raw: {
      answer:
        '<|channel|>analysis<|message|>Нужна аналогия.<|end|><|start|>assistant<|channel|>final<|message|>KV cache хранит ключи и значения прошлых токенов.<|return|>',
      tool: '<|channel|>analysis<|message|>Нужен расчёт.<|end|><|start|>assistant to=functions.kv_cache_size<|channel|>commentary json<|message|>{"context": 8192}<|call|>',
    },
    how: 'Формат Harmony: у каждого сообщения есть канал. analysis уходит в рассуждения, final в ответ, commentary с получателем to=functions.… становится вызовом. Вся разметка состоит из спецтокенов с special=true, поэтому vLLM разбирает её по номерам токенов, а не по тексту.',
  },
  llama3: {
    family: 'llama3',
    raw: {
      answer: 'KV cache хранит ключи и значения прошлых токенов.<|eot_id|>',
      tool: '{"name": "kv_cache_size", "parameters": {"context": 8192}}<|eot_id|>',
    },
    how: 'Рассуждений нет. Вызов модель печатает голым JSON с ключом parameters, а парсер llama3_json отдаёт его клиенту как arguments. Параллельных вызовов Llama 3 не поддерживает.',
  },
}

export const FAMILY_ORDER: readonly ModelFamily[] = ['qwen3', 'deepseek-r1', 'gpt-oss', 'llama3']

const STOP_LITERALS: Record<ModelFamily, string[]> = {
  qwen3: ['<|im_end|>'],
  'deepseek-r1': ['<｜end▁of▁sentence｜>'],
  'gpt-oss': ['<|return|>', '<|call|>'],
  llama3: ['<|eot_id|>'],
}

function stripStop(family: ModelFamily, raw: string): string {
  for (const s of STOP_LITERALS[family]) if (raw.endsWith(s)) return raw.slice(0, -s.length)
  return raw
}

function collect(events: ParserEventEx[]): ParsedOutput {
  const r: ParsedOutput = { reasoning: '', content: '', toolCalls: [] }
  for (const ev of events) {
    for (const it of ev.out) {
      if (it.kind === 'reasoning') r.reasoning += it.text
      else if (it.kind === 'content') r.content += it.text
    }
    for (const call of ev.toolCalls ?? []) r.toolCalls.push({ name: call.name, arguments: JSON.stringify(call.arguments) })
  }
  return r
}

const DS_CALL = /<｜tool▁call▁begin｜>function<｜tool▁sep｜>([\w.-]+)\n```json\n([\s\S]*?)\n```<｜tool▁call▁end｜>/g
const DS_BLOCK = /<｜tool▁calls▁begin｜>[\s\S]*?<｜tool▁calls▁end｜>/

/** Разбор вывода семейства в поля OpenAI-ответа (упрощённо; Qwen3 и R1 через движок, остальные локально). */
export function parseFamilyOutput(family: ModelFamily, raw: string): ParsedOutput {
  const text = stripStop(family, raw)
  if (family === 'qwen3') return collect(runStreamingParser([text], { reasoning: 'qwen3', tools: 'hermes', idSeed: 'families' }))
  if (family === 'deepseek-r1') {
    const base = collect(runStreamingParser([text], { reasoning: 'deepseek-r1', tools: 'none' }))
    const calls: ParsedOutput['toolCalls'] = []
    for (const m of base.content.matchAll(DS_CALL)) calls.push({ name: m[1], arguments: normalizeJson(m[2]) })
    if (calls.length === 0) return base
    return { reasoning: base.reasoning, content: base.content.replace(DS_BLOCK, '').trim(), toolCalls: calls }
  }
  if (family === 'gpt-oss') {
    const r: ParsedOutput = { reasoning: '', content: '', toolCalls: [] }
    for (const m of harmonyToChannels(text)) {
      if (m.lane === 'reasoning') r.reasoning += m.content
      else if (m.lane === 'content') r.content += m.content
      else r.toolCalls.push({ name: (m.recipient ?? '').replace(/^functions\./, ''), arguments: normalizeJson(m.content) })
    }
    return r
  }
  // llama3: голый JSON {"name", "parameters"} считается вызовом, остальное текстом
  const trimmed = text.trim()
  if (trimmed.startsWith('{')) {
    try {
      const obj = JSON.parse(trimmed) as { name?: unknown; parameters?: unknown }
      if (typeof obj.name === 'string' && obj.parameters && typeof obj.parameters === 'object') {
        return { reasoning: '', content: '', toolCalls: [{ name: obj.name, arguments: JSON.stringify(obj.parameters) }] }
      }
    } catch {
      /* не JSON: обычный текст */
    }
  }
  return { reasoning: '', content: text, toolCalls: [] }
}

/** JSON-строка аргументов в компактном виде (как её отдаёт сервер); не JSON остаётся как есть. */
function normalizeJson(s: string): string {
  try {
    return JSON.stringify(JSON.parse(s))
  } catch {
    return s
  }
}

/** Литералы спецтокенов семейства (для подсветки пунктиром). */
export function familySpecials(family: ModelFamily): readonly string[] {
  return FAMILY_SPECIAL_TOKENS[family]
}

/** Флаги парсеров из движка (templates.ts FAMILIES). */
export function familyFlags(family: ModelFamily): { model: string; vllm?: string; dynamo?: string; name: string } {
  const f = familyInfo(family)
  return { model: f.exampleModel, vllm: f.vllmFlags, dynamo: f.dynamoFlags, name: f.displayName }
}

/** Разбивает текст на куски: литералы спецтокенов отдельно. */
export function splitSpecial(text: string, literals: readonly string[]): Array<{ text: string; special: boolean }> {
  const sorted = [...literals].sort((a, b) => b.length - a.length)
  const out: Array<{ text: string; special: boolean }> = []
  let buf = ''
  let i = 0
  while (i < text.length) {
    const lit = sorted.find((l) => text.startsWith(l, i))
    if (lit) {
      if (buf) out.push({ text: buf, special: false })
      buf = ''
      out.push({ text: lit, special: true })
      i += lit.length
    } else {
      buf += text[i]
      i++
    }
  }
  if (buf) out.push({ text: buf, special: false })
  return out
}

/* ---------------------------------------------------------------------------------------------
 * SSE
 * ------------------------------------------------------------------------------------------- */

export type SseKind = 'role' | 'reasoning' | 'content' | 'tool-head' | 'tool-args' | 'finish' | 'usage' | 'done'

/** Вид чанка по его JSON. */
export function sseKind(json: Record<string, unknown> | null): SseKind {
  if (json === null) return 'done'
  const choices = json.choices as Array<Record<string, unknown>> | undefined
  if (!choices || choices.length === 0) return 'usage'
  const c = choices[0]
  if (c.finish_reason) return 'finish'
  const delta = (c.delta ?? {}) as Record<string, unknown>
  if ('role' in delta) return 'role'
  if ('reasoning_content' in delta || 'reasoning' in delta) return 'reasoning'
  if ('tool_calls' in delta) {
    const tc = (delta.tool_calls as Array<Record<string, unknown>>)[0]
    return tc && ('id' in tc || 'type' in tc) ? 'tool-head' : 'tool-args'
  }
  return 'content'
}

/** Короткая запись строки: служебные поля (id, object, created, model) свёрнуты в «…». */
export function compactLine(json: Record<string, unknown> | null): string {
  if (json === null) return 'data: [DONE]'
  const rest: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(json)) if (!['id', 'object', 'created', 'model'].includes(k)) rest[k] = v
  const choices = rest.choices as Array<Record<string, unknown>> | undefined
  if (choices && choices.length > 0) {
    const c: Record<string, unknown> = { ...choices[0] }
    delete c.index
    delete c.logprobs
    if (c.finish_reason === null) delete c.finish_reason
    return 'data: {…,"choices":[{' + Object.entries(c).map(([k, v]) => `${JSON.stringify(k)}:${JSON.stringify(v)}`).join(',') + ',…}]}'
  }
  return 'data: {…,' + Object.entries(rest).map(([k, v]) => `${JSON.stringify(k)}:${JSON.stringify(v)}`).join(',') + '}'
}
