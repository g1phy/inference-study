/**
 * Чат-шаблоны четырёх семейств: messages → одна строка промпта, размеченная сегментами для подсветки.
 *
 * Каждый рендерер повторяет логику официального Jinja-шаблона шаг за шагом (включая его странности),
 * и результат сверен побайтно с jinja2 в окружении transformers (templates.fixture.json, 24 кейса):
 *   qwen3       — research/vendor/qwen3/tokenizer_config.json (Qwen/Qwen3-8B);
 *   llama3      — chat_template.jinja Llama 3.1 Instruct (зеркало unsloth, совпадает с RedHatAI и 3.3, notes-tokens §1.3);
 *   deepseek-r1 — tokenizer_config.json deepseek-ai/DeepSeek-R1 (обновлённый, с `<think>\n` в generation prompt);
 *   gpt-oss     — chat_template.jinja openai/gpt-oss-20b (Harmony).
 * Строки и поведение описаны в research/notes-tokens-templates-structured.md §1.2–1.3, §3.1–3.3.
 *
 * Фиксированные даты (в настоящих шаблонах дата подставляется при рендере):
 *   Llama 3.1: date_string по умолчанию из самого шаблона — "26 Jul 2024";
 *   gpt-oss: strftime_now("%Y-%m-%d") → "2026-11-12" (день замеров исследования), можно переопределить.
 */
import type {
  ChatMessage,
  FamilyInfo,
  JsonSchema,
  ModelFamily,
  RenderedTemplate,
  SegmentKind,
  TemplateInput,
  TemplateSegment,
  ToolDef,
} from './types'

/** Дополнительные (необязательные) параметры рендера сверх TemplateInput. */
export interface TemplateInputEx extends TemplateInput {
  /** Llama 3.1: date_string в строке «Today Date: …». По умолчанию "26 Jul 2024" (дефолт шаблона). */
  llamaDateString?: string
  /** gpt-oss: «Current date: …» в формате YYYY-MM-DD. По умолчанию "2026-11-12". */
  harmonyDate?: string
  /** gpt-oss: reasoning_effort. По умолчанию "medium"; enableThinking=false даёт "low". */
  reasoningEffort?: 'low' | 'medium' | 'high'
}

/** Результат рендера с предупреждениями (там, где настоящий шаблон бросил бы исключение). */
export interface RenderedTemplateEx extends RenderedTemplate {
  /** По-русски: где вход не подходит шаблону и что мы сделали вместо raise_exception. */
  warnings: string[]
}

/** Метаданные семейства + стоп-токены и флаги серверов (notes-tokens §1.3, §1.12; notes-dynamo §1.12). */
export interface FamilyInfoEx extends FamilyInfo {
  /** Литералы стоп-токенов генерации (generation_config.json). */
  stopTokens: string[]
  /** Начало ответа ассистента (хвост add_generation_prompt). */
  generationPrompt: string
  /** Флаги vLLM для парсеров. */
  vllmFlags?: string
  /** Флаги воркера Dynamo для парсеров. */
  dynamoFlags?: string
}

/** Добавленные (цельные) токены каждого семейства: в сегментах их литералы помечаются как 'special'. */
export const FAMILY_SPECIAL_TOKENS: Record<ModelFamily, readonly string[]> = {
  // research/vendor/qwen3/tokenizer.json, added_tokens (151643…151668)
  qwen3: [
    '<|endoftext|>', '<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>',
    '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>',
    '<|image_pad|>', '<|video_pad|>', '<tool_call>', '</tool_call>', '<|fim_prefix|>', '<|fim_middle|>',
    '<|fim_suffix|>', '<|fim_pad|>', '<|repo_name|>', '<|file_sep|>', '<tool_response>', '</tool_response>',
    '<think>', '</think>',
  ],
  // notes-tokens §2.1 «Llama 3.1 ключевые id»
  llama3: [
    '<|begin_of_text|>', '<|end_of_text|>', '<|finetune_right_pad_id|>', '<|start_header_id|>', '<|end_header_id|>',
    '<|eom_id|>', '<|eot_id|>', '<|python_tag|>',
  ],
  // notes-tokens §2.1 «DeepSeek ключевые id» (U+FF5C «｜» и U+2581 «▁», не ASCII)
  'deepseek-r1': [
    '<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<｜User｜>', '<｜Assistant｜>', '<think>', '</think>',
    '<｜tool▁calls▁begin｜>', '<｜tool▁calls▁end｜>', '<｜tool▁call▁begin｜>', '<｜tool▁call▁end｜>',
    '<｜tool▁outputs▁begin｜>', '<｜tool▁outputs▁end｜>', '<｜tool▁output▁begin｜>', '<｜tool▁output▁end｜>', '<｜tool▁sep｜>',
  ],
  // notes-tokens §2.1 «Harmony id»
  'gpt-oss': [
    '<|startoftext|>', '<|endoftext|>', '<|return|>', '<|constrain|>', '<|channel|>', '<|start|>', '<|end|>',
    '<|message|>', '<|call|>', '<|endofprompt|>',
  ],
}

/** Семейства шаблонов для переключателя в UI. */
export const FAMILIES: FamilyInfoEx[] = [
  {
    family: 'qwen3',
    displayName: 'Qwen3',
    exampleModel: 'Qwen/Qwen3-8B',
    reasoningParser: 'qwen3',
    toolParser: 'hermes',
    notes:
      'ChatML: реплики обёрнуты в <|im_start|>роль … <|im_end|>, BOS нет. Мысли идут в <think>…</think>, вызовы инструментов — JSON внутри <tool_call>; enable_thinking=false вставляет в промпт пустой блок <think></think>.',
    stopTokens: ['<|im_end|>', '<|endoftext|>'],
    generationPrompt: '<|im_start|>assistant\n',
    vllmFlags: '--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes',
    dynamoFlags: '--dyn-reasoning-parser qwen3 --dyn-tool-call-parser hermes',
  },
  {
    family: 'llama3',
    displayName: 'Llama 3.1',
    exampleModel: 'meta-llama/Llama-3.1-8B-Instruct',
    toolParser: 'llama3_json',
    notes:
      'Шаблон сам ставит BOS <|begin_of_text|> и дописывает строки «Cutting Knowledge Date» и «Today Date», которых пользователь не писал. Инструменты по умолчанию уходят в первое сообщение user, вызов — голый JSON с ключом parameters, мышления нет.',
    stopTokens: ['<|end_of_text|>', '<|eom_id|>', '<|eot_id|>'],
    generationPrompt: '<|start_header_id|>assistant<|end_header_id|>\n\n',
    vllmFlags: '--enable-auto-tool-choice --tool-call-parser llama3_json',
    dynamoFlags: '--dyn-tool-call-parser llama3_json',
  },
  {
    family: 'deepseek-r1',
    displayName: 'DeepSeek-R1',
    exampleModel: 'deepseek-ai/DeepSeek-R1',
    reasoningParser: 'deepseek_r1',
    toolParser: 'deepseek_v3',
    notes:
      'Шаблон сам открывает <think>\\n в промпте, поэтому ответ модели начинается сразу с рассуждения и содержит только </think> (forced reasoning). Спецтокены пишутся с полноширинной чертой ｜ и ▁; описание tools шаблон не рендерит. Вызовы инструментов vLLM и Dynamo официально поддерживают только у DeepSeek-R1-0528 (к флагам ниже добавляют --enable-auto-tool-choice --tool-call-parser deepseek_v3 и --chat-template examples/tool_chat_template_deepseekr1.jinja, в Dynamo --dyn-tool-call-parser deepseek_v3), а её шаблон <think> уже не дописывает.',
    stopTokens: ['<｜end▁of▁sentence｜>'],
    generationPrompt: '<｜Assistant｜><think>\n',
    // у исходной R1 только парсер рассуждений: deepseek_v3 в vLLM и Dynamo рассчитан на V3-0324 и R1-0528 (см. notes)
    vllmFlags: '--reasoning-parser deepseek_r1',
    dynamoFlags: '--dyn-reasoning-parser deepseek_r1',
  },
  {
    family: 'gpt-oss',
    displayName: 'gpt-oss (Harmony)',
    exampleModel: 'openai/gpt-oss-120b',
    reasoningParser: 'gpt_oss',
    toolParser: 'harmony',
    notes:
      'Harmony: сообщения <|start|>роль<|channel|>канал<|message|>…<|end|>, мысли в канале analysis, ответ в final, вызовы в commentary. Шаблон сам пишет system-блок с датой и «Reasoning: medium», а ваш system переносит в developer; разметку разбирают по id токенов (в vLLM tool-парсер называется openai).',
    stopTokens: ['<|return|>', '<|endoftext|>', '<|call|>'],
    generationPrompt: '<|start|>assistant',
    vllmFlags: '--enable-auto-tool-choice --tool-call-parser openai',
    dynamoFlags: '--dyn-reasoning-parser gpt_oss --dyn-tool-call-parser harmony',
  },
]

/* ---------------------------------------------------------------------------------------------
 * Помощники, повторяющие Python/Jinja
 * ------------------------------------------------------------------------------------------- */

/** Число как в Python json.dumps (repr float: экспонента при |x| < 1e-4 в виде 1e-05). */
function pyNumber(v: number): string {
  if (Number.isNaN(v)) return 'NaN'
  if (!Number.isFinite(v)) return v > 0 ? 'Infinity' : '-Infinity'
  if (Number.isInteger(v) || v === 0 || Math.abs(v) >= 1e-4) return String(v)
  const [mant, exp] = v.toExponential().split('e')
  const e = Number(exp)
  return `${mant}e${e < 0 ? '-' : '+'}${String(Math.abs(e)).padStart(2, '0')}`
}

function pyJsonInner(v: unknown, indent: number | undefined, cur: string): string {
  if (v === null || v === undefined) return 'null'
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number') return pyNumber(v)
  if (typeof v === 'string') return JSON.stringify(v)
  if (Array.isArray(v)) {
    if (v.length === 0) return '[]'
    if (indent === undefined) return '[' + v.map((x) => pyJsonInner(x, indent, cur)).join(', ') + ']'
    const inner = cur + ' '.repeat(indent)
    return '[\n' + v.map((x) => inner + pyJsonInner(x, indent, inner)).join(',\n') + '\n' + cur + ']'
  }
  if (typeof v === 'object') {
    const entries = Object.entries(v as Record<string, unknown>).filter(([, x]) => x !== undefined)
    if (entries.length === 0) return '{}'
    if (indent === undefined) {
      return '{' + entries.map(([k, x]) => JSON.stringify(k) + ': ' + pyJsonInner(x, indent, cur)).join(', ') + '}'
    }
    const inner = cur + ' '.repeat(indent)
    return '{\n' + entries.map(([k, x]) => inner + JSON.stringify(k) + ': ' + pyJsonInner(x, indent, inner)).join(',\n') + '\n' + cur + '}'
  }
  return JSON.stringify(String(v))
}

/**
 * JSON как `tojson` в transformers = Python json.dumps(x, ensure_ascii=False[, indent]):
 * разделители ', ' и ': ' (с indent — ',' и переносы), кириллица не экранируется, порядок ключей сохраняется.
 */
export function pyJson(value: unknown, indent?: number): string {
  return pyJsonInner(value, indent, '')
}

/** Пробельные символы Python str.strip() (str.isspace): шире, чем у JS trim (там нет \x1c-\x1f и \x85, но есть ﻿). */
const PY_WS = '\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000'
const PY_STRIP_RE = new RegExp(`^[${PY_WS}]+|[${PY_WS}]+$`, 'g')

/** Jinja-фильтр trim = Python str.strip(). */
function pyStrip(s: string): string {
  return s.replace(PY_STRIP_RE, '')
}

/** Python s.lstrip(ch) / rstrip(ch) / strip(ch) для одного символа. */
function lstripCh(s: string, ch: string): string {
  let i = 0
  while (i < s.length && s[i] === ch) i++
  return s.slice(i)
}
function rstripCh(s: string, ch: string): string {
  let j = s.length
  while (j > 0 && s[j - 1] === ch) j--
  return s.slice(0, j)
}
function stripCh(s: string, ch: string): string {
  return lstripCh(rstripCh(s, ch), ch)
}

const hasCalls = (m: ChatMessage | undefined): m is ChatMessage & { tool_calls: NonNullable<ChatMessage['tool_calls']> } =>
  !!m && Array.isArray(m.tool_calls) && m.tool_calls.length > 0
const hasTools = (tools: ToolDef[] | undefined): tools is ToolDef[] => Array.isArray(tools) && tools.length > 0

/** Накопитель сегментов: соседние куски одного вида и одного сообщения склеиваются (кроме 'special'). */
class SegmentWriter {
  readonly segments: TemplateSegment[] = []
  readonly warnings: string[] = []
  push(text: string, kind: SegmentKind, messageIndex?: number): void {
    if (!text) return
    const last = this.segments[this.segments.length - 1]
    if (last && kind !== 'special' && last.kind === kind && last.messageIndex === messageIndex) {
      last.text += text
      return
    }
    this.segments.push(messageIndex === undefined ? { text, kind } : { text, kind, messageIndex })
  }
  special(text: string, mi?: number): void {
    this.push(text, 'special', mi)
  }
  role(text: string, mi?: number): void {
    this.push(text, 'role', mi)
  }
  content(text: string, mi?: number): void {
    this.push(text, 'content', mi)
  }
  format(text: string, mi?: number): void {
    this.push(text, 'format', mi)
  }
  tools(text: string, mi?: number): void {
    this.push(text, 'tools', mi)
  }
  gen(text: string): void {
    this.push(text, 'gen-prompt')
  }
}

/* ---------------------------------------------------------------------------------------------
 * Qwen3 (ChatML), повтор tokenizer_config.json → chat_template
 * ------------------------------------------------------------------------------------------- */

const QWEN_TOOLS_HEAD =
  '# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>'
const QWEN_TOOLS_TAIL =
  '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'

function renderQwen3(input: TemplateInputEx, w: SegmentWriter): void {
  const msgs = input.messages
  const first = msgs[0]
  if (hasTools(input.tools)) {
    const si = first?.role === 'system' ? 0 : undefined
    w.special('<|im_start|>', si)
    w.role('system', si)
    w.format('\n', si)
    if (first?.role === 'system') {
      w.content(first.content, 0)
      w.format('\n\n')
    }
    w.tools(QWEN_TOOLS_HEAD)
    for (const tool of input.tools) w.tools('\n' + pyJson(tool))
    w.tools(QWEN_TOOLS_TAIL)
    w.special('<|im_end|>')
    w.format('\n')
  } else if (first?.role === 'system') {
    w.special('<|im_start|>', 0)
    w.role('system', 0)
    w.format('\n', 0)
    w.content(first.content, 0)
    w.special('<|im_end|>', 0)
    w.format('\n', 0)
  }

  // ns.last_query_index: последний user-запрос, который не является ответом инструмента
  let multiStep = true
  let lastQuery = msgs.length - 1
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i]
    if (multiStep && m.role === 'user' && !(m.content.startsWith('<tool_response>') && m.content.endsWith('</tool_response>'))) {
      multiStep = false
      lastQuery = i
    }
  }

  msgs.forEach((m, i) => {
    let content = m.content
    const isLast = i === msgs.length - 1
    if (m.role === 'user' || (m.role === 'system' && i !== 0)) {
      w.special('<|im_start|>', i)
      w.role(m.role, i)
      w.format('\n', i)
      w.content(content, i)
      w.special('<|im_end|>', i)
      w.format('\n', i)
    } else if (m.role === 'assistant') {
      let reasoning = ''
      if (typeof m.reasoning === 'string') {
        reasoning = m.reasoning
      } else if (content.includes('</think>')) {
        const parts = content.split('</think>')
        reasoning = lstripCh(rstripCh(parts[0], '\n').split('<think>').at(-1)!, '\n')
        content = lstripCh(parts.at(-1)!, '\n')
      }
      w.special('<|im_start|>', i)
      w.role('assistant', i)
      if (i > lastQuery && (isLast || reasoning)) {
        // мысли сохраняются только у ходов после последнего запроса пользователя
        w.format('\n', i)
        w.special('<think>', i)
        w.format('\n', i)
        w.content(stripCh(reasoning, '\n'), i)
        w.format('\n', i)
        w.special('</think>', i)
        w.format('\n\n', i)
        w.content(lstripCh(content, '\n'), i)
      } else {
        w.format('\n', i)
        w.content(content, i)
      }
      if (hasCalls(m)) {
        m.tool_calls.forEach((tc, j) => {
          if ((j === 0 && content) || j > 0) w.format('\n', i)
          w.special('<tool_call>', i)
          w.format('\n{"name": "', i)
          w.content(tc.name, i)
          w.format('", "arguments": ', i)
          w.content(typeof tc.arguments === 'string' ? tc.arguments : pyJson(tc.arguments), i)
          w.format('}\n', i)
          w.special('</tool_call>', i)
        })
      }
      w.special('<|im_end|>', i)
      w.format('\n', i)
    } else if (m.role === 'tool') {
      if (i === 0 || msgs[i - 1].role !== 'tool') {
        w.special('<|im_start|>', i)
        w.role('user', i)
      }
      w.format('\n', i)
      w.special('<tool_response>', i)
      w.format('\n', i)
      w.content(content, i)
      w.format('\n', i)
      w.special('</tool_response>', i)
      if (isLast || msgs[i + 1].role !== 'tool') {
        w.special('<|im_end|>', i)
        w.format('\n', i)
      }
    }
    // system на позиции 0 уже выведен выше
  })

  if (input.addGenerationPrompt ?? true) {
    w.gen('<|im_start|>assistant\n')
    if (input.enableThinking === false) w.gen('<think>\n\n</think>\n\n')
  }
}

/* ---------------------------------------------------------------------------------------------
 * Llama 3.1, повтор chat_template.jinja (без builtin_tools, tools_in_user_message=true)
 * ------------------------------------------------------------------------------------------- */

function renderLlama3(input: TemplateInputEx, w: SegmentWriter): void {
  const all = input.messages.map((m, i) => ({ m, i }))
  const tools = hasTools(input.tools) ? input.tools : null
  const date = input.llamaDateString ?? '26 Jul 2024'
  let rest = all
  let system = ''
  let systemIndex: number | undefined
  if (all[0]?.m.role === 'system') {
    system = pyStrip(all[0].m.content)
    systemIndex = 0
    rest = all.slice(1)
  }
  w.special('<|begin_of_text|>')
  w.special('<|start_header_id|>')
  w.role('system')
  w.special('<|end_header_id|>')
  w.format('\n\n')
  if (tools) w.tools('Environment: ipython\n')
  w.format('Cutting Knowledge Date: December 2023\n')
  w.format('Today Date: ' + date + '\n\n')
  w.content(system, systemIndex)
  w.special('<|eot_id|>')

  if (tools) {
    let firstUser = ''
    let firstIndex: number | undefined
    if (rest.length > 0) {
      firstUser = pyStrip(rest[0].m.content)
      firstIndex = rest[0].i
      rest = rest.slice(1)
    } else {
      w.warnings.push('Llama 3.1: инструменты кладутся в первое сообщение user, а его нет (настоящий шаблон бросает исключение).')
    }
    w.special('<|start_header_id|>')
    w.role('user', firstIndex)
    w.special('<|end_header_id|>')
    w.format('\n\n')
    w.tools(
      'Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\n' +
        'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.\n\n',
    )
    for (const t of tools) w.tools(pyJson(t, 4) + '\n\n')
    w.content(firstUser, firstIndex)
    w.special('<|eot_id|>', firstIndex)
  }

  for (const { m, i } of rest) {
    if (!(m.role === 'tool' || hasCalls(m))) {
      w.special('<|start_header_id|>', i)
      w.role(m.role, i)
      w.special('<|end_header_id|>', i)
      w.format('\n\n', i)
      w.content(pyStrip(m.content), i)
      w.special('<|eot_id|>', i)
    } else if (hasCalls(m)) {
      if (m.tool_calls.length !== 1) {
        w.warnings.push('Llama 3.1 поддерживает только один вызов инструмента за ход: показан первый (настоящий шаблон бросает исключение).')
      }
      const tc = m.tool_calls[0]
      w.special('<|start_header_id|>', i)
      w.role('assistant', i)
      w.special('<|end_header_id|>', i)
      w.format('\n\n{"name": "', i)
      w.content(tc.name, i)
      w.format('", "parameters": ', i)
      w.content(pyJson(tc.arguments), i)
      w.format('}', i)
      w.special('<|eot_id|>', i)
    } else {
      // tool: строка в Jinja iterable, поэтому content проходит через tojson и попадает в кавычках
      w.special('<|start_header_id|>', i)
      w.role('ipython', i)
      w.special('<|end_header_id|>', i)
      w.format('\n\n', i)
      w.content(pyJson(m.content), i)
      w.special('<|eot_id|>', i)
    }
  }
  if (input.addGenerationPrompt ?? true) w.gen('<|start_header_id|>assistant<|end_header_id|>\n\n')
}

/* ---------------------------------------------------------------------------------------------
 * DeepSeek-R1, повтор tokenizer_config.json → chat_template (с его багами is_first / is_output_first)
 * ------------------------------------------------------------------------------------------- */

function renderDeepSeekR1(input: TemplateInputEx, w: SegmentWriter): void {
  const msgs = input.messages
  w.special('<｜begin▁of▁sentence｜>')
  // все system-сообщения склеиваются через \n\n и идут сразу после BOS, без тега
  let firstSp = true
  msgs.forEach((m, i) => {
    if (m.role !== 'system') return
    if (!firstSp) w.format('\n\n')
    w.content(m.content, i)
    firstSp = false
  })
  if (hasTools(input.tools)) w.warnings.push('DeepSeek-R1: шаблон не рендерит описание tools (notes-tokens §1.3).')

  let isFirst = false // в шаблоне не сбрасывается между сообщениями — так и есть в оригинале
  let isTool = false
  let isOutputFirst = true
  msgs.forEach((m, i) => {
    if (m.role === 'user') {
      isTool = false
      w.special('<｜User｜>', i)
      w.content(m.content, i)
    }
    if (m.role === 'assistant' && hasCalls(m)) {
      isTool = false
      m.tool_calls.forEach((tc) => {
        if (!isFirst) {
          w.special('<｜Assistant｜>', i)
          w.content(m.content, i)
          w.special('<｜tool▁calls▁begin｜>', i)
          isFirst = true
        } else {
          w.format('\n', i)
        }
        w.special('<｜tool▁call▁begin｜>', i)
        w.format('function', i)
        w.special('<｜tool▁sep｜>', i)
        w.content(tc.name, i)
        w.format('\n```json\n', i)
        w.content(pyJson(tc.arguments), i)
        w.format('\n```', i)
        w.special('<｜tool▁call▁end｜>', i)
      })
      w.special('<｜tool▁calls▁end｜>', i)
      w.special('<｜end▁of▁sentence｜>', i)
    }
    if (m.role === 'assistant' && !hasCalls(m)) {
      if (isTool) {
        w.special('<｜tool▁outputs▁end｜>', i)
        w.content(m.content, i)
        w.special('<｜end▁of▁sentence｜>', i)
        isTool = false
      } else {
        const content = m.content.includes('</think>') ? m.content.split('</think>').at(-1)! : m.content
        w.special('<｜Assistant｜>', i)
        w.content(content, i)
        w.special('<｜end▁of▁sentence｜>', i)
      }
    }
    if (m.role === 'tool') {
      isTool = true
      if (isOutputFirst) {
        w.special('<｜tool▁outputs▁begin｜>', i)
        isOutputFirst = false
      }
      w.special('<｜tool▁output▁begin｜>', i)
      w.content(m.content, i)
      w.special('<｜tool▁output▁end｜>', i)
    }
  })
  if (isTool) w.special('<｜tool▁outputs▁end｜>')
  if ((input.addGenerationPrompt ?? true) && !isTool) w.gen('<｜Assistant｜><think>\n')
}

/* ---------------------------------------------------------------------------------------------
 * gpt-oss (Harmony), повтор chat_template.jinja
 * ------------------------------------------------------------------------------------------- */

/** render_typescript_type для нашего подмножества JSON Schema (с пробелами ровно как у Jinja-макроса). */
function harmonyType(spec: JsonSchema): string {
  if (spec.type === 'array') {
    if (!spec.items) return 'any[]'
    const it = spec.items.type
    if (it === 'string') return 'string[]'
    if (it === 'number' || it === 'integer') return 'number[]'
    if (it === 'boolean') return 'boolean[]'
    const inner = harmonyType(spec.items)
    return inner === 'object | object' || inner.length > 50 ? 'any[]' : inner + '[]'
  }
  if (spec.type === 'string') return spec.enum && spec.enum.length > 0 ? '"' + spec.enum.join('" | "') + '"' : 'string'
  if (spec.type === 'number' || spec.type === 'integer') return 'number'
  if (spec.type === 'boolean') return 'boolean'
  if (spec.type === 'object') {
    const props = Object.entries(spec.properties ?? {})
    if (props.length === 0) return 'object'
    const req = spec.required ?? []
    let s = '{\n'
    props.forEach(([name, ps], k) => {
      // в макросе перед вложенным типом остаётся перевод строки и 16 пробелов отступа шаблона
      s += name + (req.includes(name) ? '' : '?') + ': ' + '\n                ' + harmonyType(ps)
      if (k < props.length - 1) s += ', '
    })
    return s + '}'
  }
  return 'any'
}

function harmonyNamespace(tools: ToolDef[]): string {
  let s = '## functions\n\nnamespace functions {\n\n'
  for (const { function: f } of tools) {
    s += '// ' + f.description + '\n'
    s += 'type ' + f.name + ' = '
    const props = Object.entries(f.parameters?.properties ?? {})
    if (props.length > 0) {
      const req = f.parameters.required ?? []
      s += '(_: {\n'
      for (const [name, ps] of props) {
        if (ps.description) s += '// ' + ps.description + '\n'
        s += name + (req.includes(name) ? '' : '?') + ': ' + harmonyType(ps) + ',\n'
      }
      s += '}) => any;\n\n'
    } else {
      s += '() => any;\n\n'
    }
  }
  return s + '} // namespace functions'
}

function renderGptOss(input: TemplateInputEx, w: SegmentWriter): void {
  const tools = hasTools(input.tools) ? input.tools : null
  const effort = input.reasoningEffort ?? (input.enableThinking === false ? 'low' : 'medium')
  const date = input.harmonyDate ?? '2026-11-12'

  w.special('<|start|>')
  w.role('system')
  w.special('<|message|>')
  w.format(
    'You are ChatGPT, a large language model trained by OpenAI.\n' +
      'Knowledge cutoff: 2024-06\n' +
      'Current date: ' + date + '\n\n' +
      'Reasoning: ' + effort + '\n\n' +
      '# Valid channels: analysis, commentary, final. Channel must be included for every message.' +
      (tools ? "\nCalls to these tools must go to the commentary channel: 'functions'." : ''),
  )
  w.special('<|end|>')

  const msgs = input.messages.map((m, i) => ({ m, i }))
  let loop = msgs
  let dev = ''
  let devIndex: number | undefined
  if (msgs[0] && (msgs[0].m.role === 'system' || (msgs[0].m.role as string) === 'developer')) {
    dev = msgs[0].m.content
    devIndex = 0
    loop = msgs.slice(1)
  }
  if (dev || tools) {
    w.special('<|start|>', devIndex)
    w.role('developer', devIndex)
    w.special('<|message|>', devIndex)
    if (dev) {
      w.format('# Instructions\n\n', devIndex)
      w.content(dev, devIndex)
      w.format('\n\n', devIndex)
    }
    if (tools) w.tools('# Tools\n\n' + harmonyNamespace(tools))
    w.special('<|end|>', devIndex)
  }

  const addGen = input.addGenerationPrompt ?? true
  let lastToolName: string | null = null
  loop.forEach(({ m, i }, k) => {
    const isLast = k === loop.length - 1
    const thinking = m.reasoning
    if (m.role === 'assistant') {
      if (hasCalls(m)) {
        const futureFinal = loop.slice(k + 1).some(({ m: f }) => f.role === 'assistant' && !hasCalls(f))
        const tc = m.tool_calls[0]
        if (m.tool_calls.length > 1) w.warnings.push('gpt-oss: шаблон рендерит только первый вызов инструмента в сообщении.')
        if (m.content && thinking) {
          w.warnings.push('gpt-oss: нельзя передать и content, и reasoning в сообщении с вызовом (настоящий шаблон бросает исключение); показан reasoning.')
        }
        const analysis = m.content && !thinking ? m.content : thinking
        if (analysis && !futureFinal) {
          w.special('<|start|>', i)
          w.role('assistant', i)
          w.special('<|channel|>', i)
          w.role('analysis', i)
          w.special('<|message|>', i)
          w.content(analysis, i)
          w.special('<|end|>', i)
        }
        w.special('<|start|>', i)
        w.role('assistant to=functions.' + tc.name, i)
        w.special('<|channel|>', i)
        w.role('commentary json', i)
        w.special('<|message|>', i)
        w.content(pyJson(tc.arguments), i)
        w.special('<|call|>', i)
        lastToolName = tc.name
      } else if (isLast && !addGen) {
        if (thinking !== undefined) {
          w.special('<|start|>', i)
          w.role('assistant', i)
          w.special('<|channel|>', i)
          w.role('analysis', i)
          w.special('<|message|>', i)
          w.content(thinking, i)
          w.special('<|end|>', i)
        }
        w.special('<|start|>', i)
        w.role('assistant', i)
        w.special('<|channel|>', i)
        w.role('final', i)
        w.special('<|message|>', i)
        w.content(m.content, i)
        w.special('<|return|>', i)
      } else {
        // мысли прошлых ходов выбрасываются (Harmony: CoT не хранится в истории)
        w.special('<|start|>', i)
        w.role('assistant', i)
        w.special('<|channel|>', i)
        w.role('final', i)
        w.special('<|message|>', i)
        w.content(m.content, i)
        w.special('<|end|>', i)
        lastToolName = null
      }
    } else if (m.role === 'tool') {
      let name = lastToolName
      if (name === null) {
        w.warnings.push('gpt-oss: сообщение tool без предшествующего вызова (настоящий шаблон бросает исключение).')
        name = m.name ?? 'unknown'
      }
      w.special('<|start|>', i)
      w.role('functions.' + name + ' to=assistant', i)
      w.special('<|channel|>', i)
      w.role('commentary', i)
      w.special('<|message|>', i)
      w.content(pyJson(m.content), i)
      w.special('<|end|>', i)
    } else if (m.role === 'user') {
      w.special('<|start|>', i)
      w.role('user', i)
      w.special('<|message|>', i)
      w.content(m.content, i)
      w.special('<|end|>', i)
    }
    // system не на первом месте шаблон молча пропускает
  })
  if (addGen) w.gen('<|start|>assistant')
}

/* ---------------------------------------------------------------------------------------------
 * Публичный API
 * ------------------------------------------------------------------------------------------- */

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

const SPECIAL_RE: Record<ModelFamily, RegExp> = Object.fromEntries(
  (Object.keys(FAMILY_SPECIAL_TOKENS) as ModelFamily[]).map((f) => [
    f,
    new RegExp(
      [...FAMILY_SPECIAL_TOKENS[f]]
        .sort((a, b) => b.length - a.length)
        .map(escapeRegExp)
        .join('|'),
      'g',
    ),
  ]),
) as Record<ModelFamily, RegExp>

/** Литералы спецтокенов внутри обычных сегментов (например, <tool_call> в инструкции tools) выделяем в 'special'. */
function splitSpecials(segments: TemplateSegment[], family: ModelFamily): TemplateSegment[] {
  const re = SPECIAL_RE[family]
  const out: TemplateSegment[] = []
  for (const s of segments) {
    if (s.kind === 'special' || s.kind === 'gen-prompt') {
      out.push(s)
      continue
    }
    re.lastIndex = 0
    let pos = 0
    let m: RegExpExecArray | null
    const withIdx = (text: string, kind: SegmentKind): TemplateSegment =>
      s.messageIndex === undefined ? { text, kind } : { text, kind, messageIndex: s.messageIndex }
    while ((m = re.exec(s.text))) {
      if (m.index > pos) out.push(withIdx(s.text.slice(pos, m.index), s.kind))
      out.push(withIdx(m[0], 'special'))
      pos = m.index + m[0].length
    }
    if (pos === 0) out.push(s)
    else if (pos < s.text.length) out.push(withIdx(s.text.slice(pos), s.kind))
  }
  return out
}

/**
 * Рендер чат-шаблона семейства: строка промпта, которую увидит модель, и сегменты для раскраски
 * (special / role / content / tools / gen-prompt / format). text === segments.map(s => s.text).join('').
 */
export function renderTemplate(rawInput: TemplateInputEx): RenderedTemplateEx {
  // content: null (так OpenAI API присылает ответ ассистента с tool_calls) шаблон Qwen3 читает как ''
  // ({% if message.content is string %} … {% else %} '' ); без этого рендер падал на .startsWith.
  const input: TemplateInputEx = {
    ...rawInput,
    messages: (rawInput.messages ?? []).map((m) => (typeof m.content === 'string' ? m : { ...m, content: '' })),
  }
  const w = new SegmentWriter()
  switch (input.family) {
    case 'qwen3':
      renderQwen3(input, w)
      break
    case 'llama3':
      renderLlama3(input, w)
      break
    case 'deepseek-r1':
      renderDeepSeekR1(input, w)
      break
    case 'gpt-oss':
      renderGptOss(input, w)
      break
    default:
      throw new Error(`Неизвестное семейство шаблона: ${String((input as TemplateInput).family)}`)
  }
  const segments = splitSpecials(w.segments, input.family)
  return { family: input.family, text: segments.map((s) => s.text).join(''), segments, warnings: w.warnings }
}

/** Информация о семействе по id. */
export function familyInfo(family: ModelFamily): FamilyInfoEx {
  const f = FAMILIES.find((x) => x.family === family)
  if (!f) throw new Error(`Неизвестное семейство: ${family}`)
  return f
}
