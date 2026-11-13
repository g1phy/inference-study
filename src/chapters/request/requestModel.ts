/**
 * Модель запроса для главы «Запрос»: тело POST /v1/chat/completions из настроек читателя
 * и его JSON-текст, разбитый на блоки по полям верхнего уровня (каждый блок подсвечивается в фигуре).
 * Чистые функции без React: покрыты тестом requestModel.test.ts (склейка блоков = валидный JSON = тело).
 */
import { pyJson } from '@/engine/templates'
import type { ChatMessage, JsonSchema, ToolDef } from '@/engine/types'

export type FieldId =
  | 'model'
  | 'messages'
  | 'tools'
  | 'chat_template_kwargs'
  | 'temperature'
  | 'top_p'
  | 'max_tokens'
  | 'stop'
  | 'response_format'
  | 'stream'

export interface RequestOptions {
  /** HF id модели: "Qwen/Qwen3-8B". */
  model: string
  systemText: string
  userText: string
  temperature: number
  topP: number
  maxTokens: number
  tools: boolean
  json: boolean
  stream: boolean
  /** false: в запрос добавляется chat_template_kwargs.enable_thinking = false. */
  thinking: boolean
}

/** Инструмент из сквозного примера сайта (тот же, что в заметках: notes-tokens §3.2). */
export const WEATHER_TOOL: ToolDef = {
  type: 'function',
  function: {
    name: 'get_weather',
    description: 'Получить текущую погоду в городе',
    parameters: {
      type: 'object',
      properties: { city: { type: 'string', description: 'Название города' } },
      required: ['city'],
    },
  },
}

/** Схема для response_format: ответ одним полем answer. */
export const ANSWER_SCHEMA: JsonSchema = {
  type: 'object',
  properties: { answer: { type: 'string' } },
  required: ['answer'],
}

export const STOP_EXAMPLE = ['</answer>']

/** top_p сквозного примера (тот же в главах «Запрос» и «Шаблон чата»). */
export const DEFAULT_TOP_P = 0.95

/** messages из настроек: system (если не пустой) + user. */
export function requestMessages(o: Pick<RequestOptions, 'systemText' | 'userText'>): ChatMessage[] {
  const out: ChatMessage[] = []
  if (o.systemText.trim()) out.push({ role: 'system', content: o.systemText })
  out.push({ role: 'user', content: o.userText })
  return out
}

/** Тело запроса как объект (порядок ключей как в показанном JSON). */
export function buildRequestBody(o: RequestOptions): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: o.model,
    messages: requestMessages(o).map((m) => ({ role: m.role, content: m.content })),
  }
  if (o.tools) body.tools = [WEATHER_TOOL]
  if (!o.thinking) body.chat_template_kwargs = { enable_thinking: false }
  body.temperature = o.temperature
  body.top_p = o.topP
  body.max_tokens = o.maxTokens
  body.stop = STOP_EXAMPLE
  if (o.json) body.response_format = { type: 'json_schema', json_schema: { name: 'answer', schema: ANSWER_SCHEMA } }
  body.stream = o.stream
  if (o.stream) body.stream_options = { include_usage: true }
  return body
}

export interface FieldBlock {
  id: FieldId
  /** Строки блока с отступами (2 пробела на уровень), последняя строка с запятой, если блок не последний. */
  lines: string[]
}

const I1 = '  '
const I2 = '    '
const I3 = '      '
const I4 = '        '

/**
 * JSON-текст запроса, разбитый на блоки по полям. Вложенные объекты местами записаны в одну строку
 * (как их обычно пишут руками), чтобы запрос помещался на экран; JSON при этом тот же.
 */
export function requestBlocks(o: RequestOptions): FieldBlock[] {
  const blocks: FieldBlock[] = []
  const msgs = requestMessages(o)
  blocks.push({ id: 'model', lines: [`${I1}"model": ${JSON.stringify(o.model)}`] })
  blocks.push({
    id: 'messages',
    lines: [
      `${I1}"messages": [`,
      ...msgs.map((m, i) => `${I2}${pyJson({ role: m.role, content: m.content })}${i < msgs.length - 1 ? ',' : ''}`),
      `${I1}]`,
    ],
  })
  if (o.tools) {
    const f = WEATHER_TOOL.function
    const p = f.parameters
    blocks.push({
      id: 'tools',
      lines: [
        `${I1}"tools": [{`,
        `${I2}"type": "function",`,
        `${I2}"function": {`,
        `${I3}"name": ${JSON.stringify(f.name)},`,
        `${I3}"description": ${JSON.stringify(f.description)},`,
        `${I3}"parameters": {`,
        `${I4}"type": "object",`,
        `${I4}"properties": ${pyJson(p.properties)},`,
        `${I4}"required": ${pyJson(p.required)}`,
        `${I3}}`,
        `${I2}}`,
        `${I1}}]`,
      ],
    })
  }
  if (!o.thinking) blocks.push({ id: 'chat_template_kwargs', lines: [`${I1}"chat_template_kwargs": {"enable_thinking": false}`] })
  blocks.push({ id: 'temperature', lines: [`${I1}"temperature": ${JSON.stringify(o.temperature)}`] })
  blocks.push({ id: 'top_p', lines: [`${I1}"top_p": ${JSON.stringify(o.topP)}`] })
  blocks.push({ id: 'max_tokens', lines: [`${I1}"max_tokens": ${JSON.stringify(o.maxTokens)}`] })
  blocks.push({ id: 'stop', lines: [`${I1}"stop": ${pyJson(STOP_EXAMPLE)}`] })
  if (o.json) {
    blocks.push({
      id: 'response_format',
      lines: [
        `${I1}"response_format": {`,
        `${I2}"type": "json_schema",`,
        `${I2}"json_schema": {`,
        `${I3}"name": "answer",`,
        `${I3}"schema": ${pyJson(ANSWER_SCHEMA)}`,
        `${I2}}`,
        `${I1}}`,
      ],
    })
  }
  blocks.push({
    id: 'stream',
    lines: o.stream ? [`${I1}"stream": true,`, `${I1}"stream_options": {"include_usage": true}`] : [`${I1}"stream": false`],
  })
  // запятые между полями верхнего уровня
  blocks.forEach((b, i) => {
    if (i < blocks.length - 1) b.lines[b.lines.length - 1] += ','
  })
  return blocks
}

/** Весь JSON одной строкой с переводами строк (то, что уходит в теле POST). */
export function requestText(blocks: FieldBlock[]): string {
  return ['{', ...blocks.flatMap((b) => b.lines), '}'].join('\n')
}
