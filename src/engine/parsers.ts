/**
 * Стриминговые парсеры postprocessor: текст модели по кускам → reasoning_content / content / tool_calls,
 * и сборка OpenAI-совместимых SSE-чанков chat.completion.chunk.
 *
 * Модель поведения (research/notes-tokens-templates-structured.md §1.12, §1.13, §3.11; notes-dynamo §1.12, §3.12):
 *  - сначала reasoning-парсер, потом tool-парсер по content (tool calls внутри рассуждения не ищутся);
 *  - qwen3: рассуждение в <think>…</think>; deepseek-r1: шаблон уже открыл <think>, поэтому поток
 *    начинается внутри рассуждения (forced reasoning, по умолчанию startInReasoning = true);
 *  - hermes: <tool_call>{"name": …, "arguments": …}</tool_call>, вызовов может быть несколько;
 *  - частичные теги придерживаются, пока не станет ясно, тег это или текст (как _buffer в SGLang
 *    BaseReasoningFormatDetector и partial_tag_overlap в vLLM hermes: «<thi» ждёт следующего куска);
 *  - аргументы вызова стримятся как в vLLM hermes_tool_parser: один раз {index, id, type, name},
 *    потом приращения строки после "arguments":, закрывающая } внешнего объекта не отправляется;
 *  - finish_reason = "tool_calls", если был хотя бы один вызов.
 * Отличия от реальных серверов (упрощения): теги ищутся по тексту, а не по id токенов; после каждого вызова
 * ведущие пробелы до следующего текста отбрасываются, а содержательный текст идёт в content (так делают SGLang
 * и Dynamo; vLLM hermes отбрасывает любой текст после первого <tool_call>).
 * Результат (склейка дорожек, аргументы, вызовы) не зависит от того, как поток порезан на куски:
 * это проверяет parsers.adversarial.test.ts на каждом офсете.
 */
import type { Channel, ParserDelta, ParserEvent, ParserOptions, ParserState, SseChunk, ToolCall } from './types'

/** Опции парсера сверх ParserOptions. */
export interface ParserOptionsEx extends ParserOptions {
  /** Префикс id вызовов (по умолчанию 'chatcmpl-tool-', как в vLLM). */
  toolCallIdPrefix?: string
  /** Соль для детерминированных id вызовов. */
  idSeed?: string
  /** Добавить событие сброса буфера в конце потока (step = chunks.length). По умолчанию true. */
  final?: boolean
}

/** Фрагмент tool_calls для SSE: сначала index+id+type+name, потом куски arguments. */
export interface ToolCallDelta {
  index: number
  id?: string
  type?: 'function'
  name?: string
  arguments?: string
}

/** Что уходит наружу, в порядке появления: текст дорожки или фрагмент tool_calls. */
export type ParserOutItem = { kind: 'content' | 'reasoning'; text: string } | { kind: 'tool'; delta: ToolCallDelta }

/** ParserEvent с данными для SSE и настоящей дорожкой после шага. */
export interface ParserEventEx extends ParserEvent {
  /** Фрагменты tool_calls этого шага (vLLM-стиль). */
  toolDeltas: ToolCallDelta[]
  /** Всё, что уходит клиенту на этом шаге, по порядку. */
  out: ParserOutItem[]
  /** Дорожка после шага (stateAfter может быть 'maybe_tag', а это — реальное состояние под ним). */
  channelAfter: Channel
  /** Все вызовы, закрытые на этом шаге (toolCall — первый из них). */
  toolCalls?: ToolCall[]
}

const THINK_OPEN = '<think>'
const THINK_CLOSE = '</think>'
const TOOL_OPEN = '<tool_call>'
const TOOL_CLOSE = '</tool_call>'

function fnv1a(s: string): string {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193) >>> 0
  }
  return h.toString(16).padStart(8, '0')
}

/** Длина самого длинного суффикса text, который является собственным префиксом одного из тегов. */
export function partialTagOverlap(text: string, tags: readonly string[]): number {
  let best = 0
  for (const tag of tags) {
    for (let len = Math.min(tag.length - 1, text.length); len > best; len--) {
      if (text.endsWith(tag.slice(0, len))) {
        best = len
        break
      }
    }
  }
  return best
}

/**
 * Часть строки аргументов, которую уже можно отправить: всё после "arguments": до конца значения
 * (если значение закончено) или всё, что есть (если ещё нет). null, если ключ "arguments" ещё не встретился.
 */
export function streamableArguments(body: string): string | null {
  const m = /"arguments"\s*:\s*/.exec(body)
  if (!m) return null
  const rest = body.slice(m.index + m[0].length)
  if (rest === '') return ''
  const c0 = rest[0]
  if (c0 === '{' || c0 === '[') {
    let depth = 0
    let inStr = false
    let esc = false
    for (let i = 0; i < rest.length; i++) {
      const c = rest[i]
      if (inStr) {
        if (esc) esc = false
        else if (c === '\\') esc = true
        else if (c === '"') inStr = false
      } else if (c === '"') inStr = true
      else if (c === '{' || c === '[') depth++
      else if (c === '}' || c === ']') {
        depth--
        if (depth === 0) return rest.slice(0, i + 1)
      }
    }
    return rest
  }
  if (c0 === '"') {
    let esc = false
    for (let i = 1; i < rest.length; i++) {
      const c = rest[i]
      if (esc) esc = false
      else if (c === '\\') esc = true
      else if (c === '"') return rest.slice(0, i + 1)
    }
    return rest
  }
  const lit = /^[^,}\]\s]*/.exec(rest)![0]
  return lit
}

/**
 * Прогоняет куски текста (обычно по одному детокенизированному токену) через машину состояний
 * reasoning → content → tool_call и возвращает событие на каждый кусок (+ финальный сброс буфера).
 */
export function runStreamingParser(chunks: string[], opts: ParserOptionsEx): ParserEventEx[] {
  const reasoningOn = opts.reasoning !== 'none'
  const toolsOn = opts.tools === 'hermes'
  const forced = opts.startInReasoning ?? opts.reasoning === 'deepseek-r1'
  const idPrefix = opts.toolCallIdPrefix ?? 'chatcmpl-tool-'

  let mode: Channel = reasoningOn && forced ? 'reasoning' : 'content'
  let buffer = ''
  let reasoningDone = !reasoningOn
  // Forced reasoning: шаблон уже открыл <think>, но модель может напечатать его ещё раз. Первый такой <think>
  // пропускаем, где бы в рассуждении он ни стоял (как SGLang: stripped_think_start), — признак зависит только
  // от текста, а не от того, как поток порезан на куски.
  let thinkOpenPending = mode === 'reasoning'
  // После вызова инструмента пробелы до следующего содержательного текста не отправляются (\n между вызовами,
  // хвостовой \n). Срезаем ведущие пробелы, пока не придёт непробельный символ: так результат не зависит
  // от разбиения на куски (раньше выбрасывались только куски целиком из пробелов).
  let skipLeadingWs = false
  let toolIndex = -1
  let body = ''
  let nameSent = false
  let argsSent = 0
  let toolId = ''
  let anySentForCall = false

  const events: ParserEventEx[] = []

  const tagsFor = (m: Channel): string[] => {
    if (m === 'reasoning') return thinkOpenPending ? [THINK_CLOSE, THINK_OPEN] : [THINK_CLOSE]
    if (m === 'tool_call') return [TOOL_CLOSE]
    const t: string[] = []
    if (reasoningOn && !reasoningDone) t.push(THINK_OPEN)
    if (toolsOn) t.push(TOOL_OPEN)
    return t
  }

  const process = (step: number, input: string, isFinal: boolean): ParserEventEx => {
    const stateBefore: ParserState = buffer ? 'maybe_tag' : mode
    const hadBuffer = buffer
    let text = buffer + input
    buffer = ''
    const out: ParserOutItem[] = []
    const deltas: ParserDelta[] = []
    const notes: string[] = []
    const closedCalls: ToolCall[] = []
    let tagSeen = false

    const pushDelta = (channel: Channel, t: string) => {
      const last = deltas[deltas.length - 1]
      if (last && last.channel === channel) last.text += t
      else deltas.push({ channel, text: t })
    }
    const pushOut = (item: ParserOutItem) => {
      const last = out[out.length - 1]
      if (item.kind !== 'tool' && last && last.kind === item.kind) last.text += item.text
      else out.push(item)
    }
    const toolDeltas: ToolCallDelta[] = []
    const pushTool = (d: ToolCallDelta) => {
      toolDeltas.push(d)
      pushOut({ kind: 'tool', delta: d })
      anySentForCall = true
    }

    const updateToolStream = () => {
      if (!nameSent) {
        const m = /"name"\s*:\s*"((?:[^"\\]|\\.)*)"/.exec(body)
        if (m) {
          let name = m[1]
          try {
            name = JSON.parse(`"${m[1]}"`) as string
          } catch {
            /* оставляем как есть */
          }
          pushTool({ index: toolIndex, id: toolId, type: 'function', name })
          nameSent = true
          notes.push(`имя функции готово (${name}): первый чанк tool_calls с id и name`)
        }
      }
      if (nameSent) {
        const args = streamableArguments(body)
        if (args !== null && args.length > argsSent) {
          pushTool({ index: toolIndex, arguments: args.slice(argsSent) })
          argsSent = args.length
          notes.push('кусок аргументов → function.arguments')
        }
      }
    }

    const emit = (t: string) => {
      if (!t) return
      if (mode === 'content') {
        if (skipLeadingWs) {
          const rest = t.replace(/^\s+/, '')
          if (rest !== t) notes.push('пробелы после вызова инструмента клиенту не отправляются')
          if (!rest) return
          skipLeadingWs = false
          t = rest
        }
        pushDelta('content', t)
        pushOut({ kind: 'content', text: t })
        notes.push('обычный текст → content')
      } else if (mode === 'reasoning') {
        pushDelta('reasoning', t)
        pushOut({ kind: 'reasoning', text: t })
        notes.push('текст рассуждения → reasoning_content')
      } else {
        body += t
        pushDelta('tool_call', t)
        updateToolStream()
      }
    }

    const finishTool = (closed: boolean): void => {
      const n = toolIndex + 1
      let parsed: unknown
      try {
        parsed = JSON.parse(body.trim())
      } catch {
        parsed = undefined
      }
      const obj = parsed as { name?: unknown; arguments?: unknown } | undefined
      if (obj && typeof obj === 'object' && typeof obj.name === 'string') {
        if (!nameSent) {
          pushTool({ index: toolIndex, id: toolId, type: 'function', name: obj.name })
          nameSent = true
        }
        const args = streamableArguments(body) ?? ''
        if (args.length > argsSent) pushTool({ index: toolIndex, arguments: args.slice(argsSent) })
        argsSent = Math.max(argsSent, args.length)
        let a = obj.arguments
        // Некоторые модели пишут arguments строкой с JSON внутри ("{\"city\": …}"): разворачиваем её.
        if (typeof a === 'string') {
          try {
            a = JSON.parse(a)
          } catch {
            /* не JSON — аргументов-объекта нет */
          }
        }
        closedCalls.push({
          id: toolId,
          name: obj.name,
          arguments: a && typeof a === 'object' && !Array.isArray(a) ? (a as Record<string, unknown>) : {},
        })
        notes.push(
          closed
            ? `${TOOL_CLOSE}: JSON разобран, вызов №${n} (${obj.name}) готов`
            : `поток кончился без ${TOOL_CLOSE}, но JSON валиден: вызов №${n} (${obj.name}) готов`,
        )
        skipLeadingWs = true
      } else if (!anySentForCall) {
        // Ничего не ушло клиенту: возвращаем исходный текст в content (как jail Dynamo при неудачном разборе).
        notes.push('JSON вызова не разобрался: исходный текст отдан как content')
        toolIndex--
        mode = 'content'
        const raw = TOOL_OPEN + body + (closed ? TOOL_CLOSE : '')
        pushDelta('content', raw)
        pushOut({ kind: 'content', text: raw })
        skipLeadingWs = false
      } else {
        notes.push('битый JSON в вызове: клиент получил только то, что успело уйти')
        skipLeadingWs = true
      }
      mode = 'content'
      body = ''
    }

    const handleTag = (tag: string) => {
      tagSeen = true
      if (tag === THINK_OPEN && mode === 'reasoning') {
        notes.push('модель напечатала <think>, хотя шаблон уже открыл рассуждение: тег пропускаем')
        thinkOpenPending = false
      } else if (tag === THINK_OPEN) {
        mode = 'reasoning'
        thinkOpenPending = false
        notes.push('<think>: начало рассуждения, сам тег клиенту не уходит')
      } else if (tag === THINK_CLOSE) {
        mode = 'content'
        reasoningDone = true
        thinkOpenPending = false
        notes.push('</think>: рассуждение закончилось, дальше content')
      } else if (tag === TOOL_OPEN) {
        mode = 'tool_call'
        toolIndex++
        body = ''
        nameSent = false
        argsSent = 0
        anySentForCall = false
        toolId = idPrefix + fnv1a(`${opts.idSeed ?? ''}#${toolIndex}`)
        notes.push('<tool_call>: начался вызов инструмента, текст копится и в content не идёт')
      } else if (tag === TOOL_CLOSE) {
        finishTool(true)
      }
    }

    if (step === 0 && mode === 'reasoning' && forced && !input.startsWith(THINK_OPEN)) {
      notes.push('шаблон уже открыл <think>: поток начинается сразу с рассуждения')
    }

    for (;;) {
      const tags = tagsFor(mode)
      let at = -1
      let tag = ''
      for (const t of tags) {
        const i = text.indexOf(t)
        if (i >= 0 && (at < 0 || i < at)) {
          at = i
          tag = t
        }
      }
      if (at >= 0) {
        emit(text.slice(0, at))
        handleTag(tag)
        text = text.slice(at + tag.length)
        continue
      }
      const hold = isFinal ? 0 : partialTagOverlap(text, tags)
      emit(text.slice(0, text.length - hold))
      buffer = text.slice(text.length - hold)
      break
    }

    if (isFinal && mode === 'tool_call') finishTool(false)
    if (buffer) {
      const tagsNow = tagsFor(mode)
      const candidates = tagsNow.filter((t) => t.startsWith(buffer)).join(' или ')
      notes.push(`«${buffer}» может оказаться началом ${candidates}: придерживаем`)
    } else if (hadBuffer && !tagSeen && !isFinal) {
      notes.push(`придержанное «${hadBuffer}» оказалось обычным текстом`)
    }
    if (isFinal && hadBuffer && !tagSeen) notes.push(`конец потока: придержанный хвост «${hadBuffer}» отдан как текст`)

    const ev: ParserEventEx = {
      step,
      input,
      stateBefore,
      stateAfter: buffer ? 'maybe_tag' : mode,
      buffer,
      deltas,
      toolDeltas,
      out,
      channelAfter: mode,
      note: [...new Set(notes)].join('; ') || (input ? 'без изменений' : undefined),
    }
    if (closedCalls.length > 0) {
      ev.toolCall = closedCalls[0]
      ev.toolCalls = closedCalls
    }
    return ev
  }

  chunks.forEach((c, i) => events.push(process(i, c, false)))
  // mode меняется внутри process (замыкание), TS этого не видит — сравниваем по объявленному типу
  if ((opts.final ?? true) && (buffer || (mode as Channel) === 'tool_call')) events.push(process(chunks.length, '', true))
  return events
}

/** Параметры SSE-обёртки. */
export interface SseMeta {
  /** id ответа, одинаковый у всех чанков (например, 'chatcmpl-7f3a'). */
  id: string
  /** Unix-секунды. */
  created: number
  model: string
  /** Причина остановки без вызовов инструментов: 'stop' (EOS / stop-строка) или 'length'. */
  finishReason?: 'stop' | 'length'
  /** stream_options.include_usage: добавить чанк с usage и choices: []. */
  includeUsage?: boolean
  usage?: { promptTokens: number; completionTokens: number; cachedTokens?: number }
  /** Поле для мыслей: 'reasoning_content' (Dynamo, SGLang, DeepSeek API; по умолчанию) или 'reasoning' (vLLM ≥ 0.16). */
  reasoningField?: 'reasoning_content' | 'reasoning'
  /** Расширение vLLM stop_reason в финальном чанке (добавляется, только если задано, null тоже). */
  stopReason?: string | number | null
}

/**
 * События парсера → строки «data: {...}» OpenAI chat.completion.chunk: чанк с ролью, дельты
 * reasoning/content/tool_calls, финальный чанк с finish_reason, usage-чанк (если include_usage), «data: [DONE]».
 */
export function toSseChunks(events: ParserEvent[], meta: SseMeta): SseChunk[] {
  const field = meta.reasoningField ?? 'reasoning_content'
  const head = { id: meta.id, object: 'chat.completion.chunk', created: meta.created, model: meta.model }
  const chunks: SseChunk[] = []
  const add = (json: Record<string, unknown>) => chunks.push({ line: 'data: ' + JSON.stringify(json), json })
  const choice = (delta: Record<string, unknown>, finish: string | null = null, extra?: Record<string, unknown>) => ({
    ...head,
    choices: [{ index: 0, delta, logprobs: null, finish_reason: finish, ...extra }],
  })

  add(choice({ role: 'assistant', content: '' }))
  let anyTool = false
  for (const ev of events) {
    const items: ParserOutItem[] =
      (ev as ParserEventEx).out ??
      ev.deltas.filter((d) => d.channel !== 'tool_call').map((d) => ({ kind: d.channel as 'content' | 'reasoning', text: d.text }))
    for (const it of items) {
      if (it.kind === 'tool') {
        anyTool = true
        const d = it.delta
        const fn: Record<string, unknown> = {}
        if (d.name !== undefined) fn.name = d.name
        if (d.arguments !== undefined) fn.arguments = d.arguments
        const tc: Record<string, unknown> = { index: d.index }
        if (d.id !== undefined) tc.id = d.id
        if (d.type !== undefined) tc.type = d.type
        tc.function = fn
        add(choice({ tool_calls: [tc] }))
      } else if (it.text) {
        add(choice(it.kind === 'reasoning' ? { [field]: it.text } : { content: it.text }))
      }
    }
    if (ev.toolCall) anyTool = true
  }
  const finish = anyTool ? 'tool_calls' : (meta.finishReason ?? 'stop')
  add(choice({}, finish, meta.stopReason !== undefined ? { stop_reason: meta.stopReason } : undefined))
  if (meta.includeUsage && meta.usage) {
    const u = meta.usage
    const usage: Record<string, unknown> = {
      prompt_tokens: u.promptTokens,
      completion_tokens: u.completionTokens,
      total_tokens: u.promptTokens + u.completionTokens,
    }
    if (u.cachedTokens !== undefined) usage.prompt_tokens_details = { cached_tokens: u.cachedTokens }
    add({ ...head, choices: [], usage })
  }
  chunks.push({ line: 'data: [DONE]', json: null })
  return chunks
}

/** Как поток выглядит на проводе: каждая строка «data: …» и пустая строка после неё. */
export function toSseText(chunks: SseChunk[]): string {
  return chunks.map((c) => c.line + '\n\n').join('')
}

/** Сообщение Harmony (gpt-oss), разобранное из текста. */
export interface HarmonyMessage {
  role: string
  channel?: string
  /** Получатель вызова: functions.NAME (to=…). */
  recipient?: string
  /** Тип содержимого после канала: json (или <|constrain|>json). */
  contentType?: string
  content: string
  terminator: '<|end|>' | '<|return|>' | '<|call|>' | null
  /** Куда это попадёт в OpenAI-ответе: analysis → reasoning, final → content, вызов → tool_call. */
  lane: Channel
}

/**
 * Упрощённый разбор вывода gpt-oss (Harmony) по каналам (notes-tokens §1.3, §1.12, §3.9).
 * Настоящие серверы разбирают Harmony по id токенов (openai_harmony StreamableParser), здесь — по тексту.
 * Текст может начинаться с <|channel|> (генерация идёт после «<|start|>assistant» из промпта).
 */
export function harmonyToChannels(text: string): HarmonyMessage[] {
  const out: HarmonyMessage[] = []
  const parts = text.split('<|start|>')
  parts.forEach((part, k) => {
    if (part === '' && k === 0) return
    const mi = part.indexOf('<|message|>')
    if (mi < 0) return
    let header = part.slice(0, mi)
    let rest = part.slice(mi + '<|message|>'.length)
    let terminator: HarmonyMessage['terminator'] = null
    let cut = rest.length
    for (const t of ['<|end|>', '<|return|>', '<|call|>'] as const) {
      const i = rest.indexOf(t)
      if (i >= 0 && i < cut) {
        cut = i
        terminator = t
      }
    }
    rest = rest.slice(0, cut)
    if (k === 0) header = 'assistant' + header
    const [rolePart, channelPart = ''] = header.split('<|channel|>')
    const roleTokens = rolePart.trim().split(/\s+/)
    const role = roleTokens[0] || 'assistant'
    let recipient = roleTokens.find((x) => x.startsWith('to='))?.slice(3)
    const chTokens = channelPart.replace('<|constrain|>', ' ').trim().split(/\s+/).filter(Boolean)
    const channel = chTokens[0]
    const chRecipient = chTokens.find((x) => x.startsWith('to='))?.slice(3)
    recipient = recipient ?? chRecipient
    const contentType = chTokens.slice(1).find((x) => !x.startsWith('to='))
    const lane: Channel = recipient ? 'tool_call' : channel === 'analysis' ? 'reasoning' : 'content'
    const msg: HarmonyMessage = { role, content: rest, terminator, lane }
    if (channel) msg.channel = channel
    if (recipient) msg.recipient = recipient
    if (contentType) msg.contentType = contentType
    out.push(msg)
  })
  return out
}
