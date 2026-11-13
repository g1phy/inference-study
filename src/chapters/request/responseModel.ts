/**
 * Ответ сервера для фигуры «Ответ: целиком или потоком».
 *
 * Текст ответа условный (живой модели в браузере нет), всё остальное настоящее:
 * токены считает словарь Qwen3, текст по токенам отдаёт инкрементальный детокенизатор движка,
 * рассуждение отделяет стриминговый парсер qwen3, SSE-строки собирает toSseChunks в формате
 * chat.completion.chunk, тайминги считает perf.ts для модели и GPU из настроек.
 */
import { incrementalDetokenize } from '@/engine/detok'
import { runStreamingParser, toSseChunks, type ParserEventEx } from '@/engine/parsers'
import { decodeStepMs, prefillMs, usableMemBytes, weightBytes } from '@/engine/perf'
import { findGpu, getGpu } from '@/engine/data/gpus'
import { findModel, getModel } from '@/engine/data/models'
import { shortHash } from '@/engine/rng'
import { QWEN3_STOP_TOKEN_IDS } from '@/engine/tokenizer'
import type { GpuSpec, ModelSpec, SseChunk, Token, Tokenizer } from '@/engine/types'

export const DEFAULT_USER_TEXT = 'Объясни простыми словами, что такое KV cache и зачем он нужен.'

/** Блок prefix cache в vLLM по умолчанию (notes-engines §1.2; SPEC §8.8). */
export const CACHE_BLOCK = 16

export interface SimAnswer {
  reasoning?: string
  content: string
}

/** Условный ответ модели: под вопрос по умолчанию свой, под любой другой честная заглушка. */
export function simulatedAnswer(userText: string, thinking: boolean, json: boolean): SimAnswer {
  const isDefault = userText.trim() === DEFAULT_USER_TEXT
  const reasoning = isDefault ? 'Нужно объяснить без формул и коротко.' : 'Отвечу коротко и по делу.'
  const text = isDefault
    ? 'KV cache хранит ключи и значения уже прочитанных токенов. На каждом шаге модель берёт их из памяти, а не считает заново, поэтому ответ печатается быстрее.'
    : 'Это условный ответ: настоящая модель в браузере не запускается, а формат ответа и числа в usage настоящие.'
  const content = json ? JSON.stringify({ answer: text }) : text
  return thinking ? { reasoning, content } : { content }
}

/** Сколько GPU нужно, чтобы веса BF16 поместились с запасом под KV (тензорный параллелизм, степень двойки). */
export function gpusForModel(model: ModelSpec, gpu: GpuSpec): number {
  const w = weightBytes(model)
  let n = 1
  while (n < 16 && w > 0.8 * usableMemBytes(gpu) * n) n *= 2
  return n
}

export function resolveHardware(modelId: string, gpuId: string): { model: ModelSpec & { hfId?: string }; gpu: GpuSpec; gpus: number } {
  const model = findModel(modelId) ?? getModel('qwen3-8b')
  const gpu = findGpu(gpuId) ?? getGpu('h100-sxm')
  return { model, gpu, gpus: gpusForModel(model, gpu) }
}

/** Сколько токенов промпта попадёт в prefix cache у повторного запроса: целые блоки, последний токен считается всегда. */
export function warmCachedTokens(promptTokens: number, block = CACHE_BLOCK): number {
  if (promptTokens <= 1) return 0
  return Math.floor((promptTokens - 1) / block) * block
}

export interface TimedChunk {
  chunk: SseChunk
  /** Кадр (1..n), на котором чанк приходит к клиенту. */
  frame: number
}

export interface ConversationSim {
  promptTokens: number
  cachedTokens: number
  answerTokens: Token[]
  chunks: TimedChunk[]
  /** frameTimes[0] = 0 (запрос отправлен), frameTimes[i] = момент i-го токена. */
  frameTimes: number[]
  /** Текст на экране у пользователя к кадру i (рассуждение и ответ отдельно). */
  textAt: Array<{ reasoning: string; content: string }>
  ttftMs: number
  itlMs: number
  e2eMs: number
  /** Ответ при stream: false (chat.completion). */
  nonStream: Record<string, unknown>
  gpus: number
  modelName: string
  gpuName: string
}

export interface SimInput {
  tokenizer: Tokenizer
  promptTokens: number
  warm: boolean
  answer: SimAnswer
  modelId: string
  gpuId: string
  modelLabel: string
}

/** Фиксированный момент создания ответа (unix-секунды), чтобы примеры были детерминированными. */
const CREATED = Math.floor(Date.UTC(2026, 10, 12, 12, 0, 0) / 1000)

export function simulateConversation(inp: SimInput): ConversationSim {
  const { tokenizer, promptTokens, answer } = inp
  const { model, gpu, gpus } = resolveHardware(inp.modelId, inp.gpuId)
  const cachedTokens = inp.warm ? warmCachedTokens(promptTokens) : 0

  // То, что генерирует Qwen3: <think>\n…\n</think>\n\n ответ <|im_end|> (notes-tokens §1.3, §3.11)
  const outText = (answer.reasoning !== undefined ? `<think>\n${answer.reasoning}\n</think>\n\n` : '') + answer.content + '<|im_end|>'
  const answerTokens = tokenizer.encode(outText)
  const n = answerTokens.length

  // текст по токенам, как его отдаёт сервер (skip_special_tokens, EOS без текста)
  const detok = incrementalDetokenize(answerTokens, tokenizer, { stopTokenIds: [...QWEN3_STOP_TOKEN_IDS] })
  const pieces = answerTokens.map((_, i) => detok[i]?.emitted ?? '')
  const events = runStreamingParser(pieces, { reasoning: 'qwen3', tools: 'none' }) as ParserEventEx[]

  const id = `chatcmpl-${shortHash(answer.content + promptTokens)}`
  const usage = { promptTokens, completionTokens: n, cachedTokens }
  const sse = toSseChunks(events, {
    id,
    created: CREATED,
    model: inp.modelLabel,
    finishReason: 'stop',
    includeUsage: true,
    usage,
    reasoningField: 'reasoning_content',
  })

  // раскладываем чанки по кадрам: роль приходит вместе с первым токеном, хвост (finish, usage, [DONE]) с последним
  const chunks: TimedChunk[] = []
  let k = 0
  chunks.push({ chunk: sse[k++], frame: 1 })
  for (const ev of events) {
    const cnt = ev.out.filter((it) => it.kind === 'tool' || it.text).length
    const frame = Math.min(n, ev.step + 1)
    for (let j = 0; j < cnt && k < sse.length; j++) chunks.push({ chunk: sse[k++], frame })
  }
  while (k < sse.length) chunks.push({ chunk: sse[k++], frame: n })

  const textAt: Array<{ reasoning: string; content: string }> = [{ reasoning: '', content: '' }]
  let reasoning = ''
  let content = ''
  for (let f = 1; f <= n; f++) {
    for (const ev of events) {
      if (Math.min(n, ev.step + 1) !== f) continue
      for (const it of ev.out) {
        if (it.kind === 'reasoning') reasoning += it.text
        else if (it.kind === 'content') content += it.text
      }
    }
    textAt.push({ reasoning, content })
  }

  const ttftMs = prefillMs(model, gpu, promptTokens, { cachedTokens, gpus })
  const itlMs = decodeStepMs(model, gpu, { batch: 1, avgContext: promptTokens + Math.round(n / 2), gpus })
  const frameTimes = [0]
  for (let f = 1; f <= n; f++) frameTimes.push(ttftMs + (f - 1) * itlMs)
  const e2eMs = frameTimes[n]

  const message: Record<string, unknown> = { role: 'assistant' }
  if (reasoning.trim()) message.reasoning_content = reasoning.trim()
  // в непотоковом ответе переводы строк после </think> срезаем: клиенту нужен только текст ответа
  message.content = content.replace(/^\n+/, '')
  const nonStream = {
    id,
    object: 'chat.completion',
    created: CREATED,
    model: inp.modelLabel,
    choices: [{ index: 0, message, finish_reason: 'stop' }],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: n,
      total_tokens: promptTokens + n,
      prompt_tokens_details: { cached_tokens: cachedTokens },
    },
  }

  return {
    promptTokens,
    cachedTokens,
    answerTokens,
    chunks,
    frameTimes,
    textAt,
    ttftMs,
    itlMs,
    e2eMs,
    nonStream,
    gpus,
    modelName: model.name,
    gpuName: gpu.name,
  }
}

/**
 * Ответ stream: false текстом для фигуры: тот же JSON, но плотнее, чем JSON.stringify(…, 2).
 * Шапка и usage записаны в пару строк, чтобы конец ответа с finish_reason и usage помещался в панель.
 */
export function completionJson(sim: Pick<ConversationSim, 'nonStream'>): string {
  const n = sim.nonStream as {
    id: string
    object: string
    created: number
    model: string
    choices: Array<{ index: number; message: Record<string, unknown>; finish_reason: string }>
    usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number; prompt_tokens_details: { cached_tokens: number } }
  }
  const J = (v: unknown) => JSON.stringify(v)
  const ch = n.choices[0]
  const msg = Object.entries(ch.message).map(([k, v], i, all) => `      ${J(k)}: ${J(v)}${i < all.length - 1 ? ',' : ''}`)
  const u = n.usage
  return [
    '{',
    `  "id": ${J(n.id)}, "object": ${J(n.object)},`,
    `  "created": ${n.created}, "model": ${J(n.model)},`,
    '  "choices": [{',
    `    "index": ${ch.index},`,
    '    "message": {',
    ...msg,
    '    },',
    `    "finish_reason": ${J(ch.finish_reason)}`,
    '  }],',
    `  "usage": {"prompt_tokens": ${u.prompt_tokens}, "completion_tokens": ${u.completion_tokens}, "total_tokens": ${u.total_tokens},`,
    `    "prompt_tokens_details": {"cached_tokens": ${u.prompt_tokens_details.cached_tokens}}}`,
    '}',
  ].join('\n')
}

/** Короткая запись чанка для ленты: без повторяющейся «шапки» (id, object, created, model). */
export function compactChunk(c: SseChunk): string {
  if (!c.json) return 'data: [DONE]'
  const j = c.json as { choices?: Array<Record<string, unknown>>; usage?: unknown }
  const ch = j.choices?.[0]
  if (!ch) return `data: {…, "choices": [], "usage": ${JSON.stringify(j.usage)}}`
  const parts = [`"delta": ${JSON.stringify(ch.delta)}`]
  if (ch.finish_reason !== null && ch.finish_reason !== undefined) parts.push(`"finish_reason": ${JSON.stringify(ch.finish_reason)}`)
  return `data: {…, ${parts.join(', ')}}`
}
