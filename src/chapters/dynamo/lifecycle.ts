/**
 * Жизненный цикл запроса читателя в Dynamo (фигура 2 главы 19). Всё, что можно посчитать, настоящее:
 * шаблон Qwen3 и токены, блоки и хэши, решение роутера (engine/router), время prefill и decode и объём KV
 * (engine/perf), детокенизация, парсер и SSE (engine/detok, engine/parsers). Ответ модели условный и тот же,
 * что в симуляторе (engine/scenarios.scriptFor): с tools это вызов get_weather, с json ответ по схеме.
 */
import type { FlowSettings, SseChunk, Token, Tokenizer, ToolCall } from '@/engine/types'
import { findGpu, getGpu, type GpuSpecExt } from '@/engine/data/gpus'
import { findModel, getModel, type ModelSpecExt } from '@/engine/data/models'
import { getInterconnect } from '@/engine/data/interconnects'
import { decodeStepMs, kvBytesForSeq, prefillMs, transferMs } from '@/engine/perf'
import { routeKV, type KvRouteDecision, type RouterWorker } from '@/engine/router'
import { incrementalDetokenize, type DetokStepEx } from '@/engine/detok'
import { runStreamingParser, toSseChunks, type ParserEventEx, type ParserOutItem, type ToolCallDelta } from '@/engine/parsers'
import { piecesText, scriptFor, WEATHER_TOOL } from '@/engine/scenarios'
import { buildPrompt, readerParts, type BuiltPrompt } from '../routing/prompt'

export type Mode = 'disagg' | 'agg'

export interface LifecycleRun {
  model: ModelSpecExt
  gpu: GpuSpecExt
  request: Record<string, unknown>
  prompt: BuiltPrompt
  /** Prefill-роутер (disagg) или KV-роутер по агрегированным воркерам (agg). */
  prefillRoute: KvRouteDecision
  aggRoute: KvRouteDecision
  decodeRoute: KvRouteDecision
  cachedTokens: number
  aggCachedTokens: number
  prefillMs: number
  aggPrefillMs: number
  kvBytes: number
  kvBlocks: number
  transfer: { link: string; ms: number }[]
  itlMs: number
  answer: {
    tokens: Token[]
    detok: DetokStepEx[]
    /** События парсера после jail Dynamo: вызов инструмента уходит одним куском, когда закрыт. */
    parser: ParserEventEx[]
    /** SSE дезагрегированного режима: cached_tokens берётся у prefill-воркера. */
    sse: SseChunk[]
    /** SSE агрегированного режима: cached_tokens того воркера, который обслужил запрос. */
    sseAgg: SseChunk[]
    thinking: boolean
    bodyKind: 'text' | 'tool_call' | 'json'
    toolCalls: ToolCall[]
    truncated: boolean
  }
  /** Схема structured output, если в запросе есть response_format. */
  jsonSchemaName: string | null
  maxTokens: number
}

/** Как в запросе симулятора (DEFAULT_MAX_TOKENS в engine/flow.ts). */
const MAX_TOKENS = 512
const STOP_IDS = [151645, 151643]
const THINK_OPEN = '<think>'
const THINK_CLOSE = '</think>'

/**
 * Текст ответа модели так же, как его собирает симулятор (engine/flow.ts, раздел 6): рассуждение в <think>,
 * после него «\n\n» (кроме JSON под маской), тело, EOS.
 */
function answerText(s: FlowSettings): { text: string; script: ReturnType<typeof scriptFor> } {
  const script = scriptFor(s)
  let text = ''
  if (script.reasoning) {
    text += THINK_OPEN + '\n' + piecesText(script.reasoning) + '\n' + THINK_CLOSE
    if (script.bodyKind !== 'json') text += '\n\n'
  }
  text += piecesText(script.body) + '<|im_end|>'
  return { text, script }
}

/**
 * Jail Dynamo для вызовов инструментов: пока вызов открыт, наружу ничего не уходит, а когда парсер его закрыл,
 * клиент получает один кусок tool_calls с готовыми аргументами (notes-dynamo.md §1.12). Парсер движка
 * (engine/parsers) стримит аргументы по кускам, как vLLM; здесь куски собираются обратно.
 */
export function jailToolCalls(events: readonly ParserEventEx[]): ParserEventEx[] {
  const pending: ToolCallDelta[] = []
  return events.map((ev) => {
    const out: ParserOutItem[] = []
    for (const it of ev.out) {
      if (it.kind !== 'tool') {
        out.push(it)
        continue
      }
      const d = it.delta
      let cur = pending.find((p) => p.index === d.index)
      if (!cur) {
        cur = { index: d.index }
        pending.push(cur)
      }
      if (d.id !== undefined) cur.id = d.id
      if (d.type !== undefined) cur.type = d.type
      if (d.name !== undefined) cur.name = d.name
    }
    const closed = ev.toolCalls ?? (ev.toolCall ? [ev.toolCall] : [])
    const emitted: ToolCallDelta[] = []
    for (const call of closed) {
      const base = pending.shift() ?? { index: emitted.length }
      emitted.push({ index: base.index, id: base.id ?? call.id, type: 'function', name: call.name, arguments: JSON.stringify(call.arguments) })
    }
    for (const d of emitted) out.push({ kind: 'tool', delta: d })
    return { ...ev, out, toolDeltas: emitted }
  })
}

function w(id: string, name: string, cached: string[], activeBlocks: number, queued = 0): RouterWorker {
  return {
    id,
    name,
    cachedBlocks: new Set(cached),
    activeBlocks,
    queuedPrefillTokens: queued,
    activeRequests: 0,
    activeBlockHashes: new Set(),
    activeBlockRefs: new Map(),
  }
}

export function computeRun(s: FlowSettings, tokenizer: Tokenizer): LifecycleRun {
  const model = findModel(s.modelId) ?? getModel('qwen3-8b')
  const gpu = findGpu(s.gpuId) ?? getGpu('h100-sxm')
  const parts = readerParts(s)
  const prompt = buildPrompt(parts, tokenizer)
  // Воркер уже обслуживал другого пользователя с тем же системным промптом (настоящие хэши).
  const neighbour = buildPrompt({ ...parts, userText: 'Чем prefill отличается от decode?' }, tokenizer)
  const shared = neighbour.blocks.filter((b) => b.full).map((b) => b.hash)

  // Prefill-роутер: учёт активных блоков выключен (prefill-воркеры не делают decode), решают кэш и очередь prefill.
  const prefillRoute = routeKV(prompt.blocks, [w('worker-0', 'Prefill 1', shared, 0, 0), w('worker-1', 'Prefill 2', [], 0, 512)], {
    overlapCredit: 1,
    temperature: 0,
    blockSize: 16,
  })
  // Decode-роутер в disagg: только нагрузка (overlap_score_credit = 0, без assume_kv_reuse).
  const decodeRoute = routeKV(prompt.blocks, [w('worker-0', 'Decode 1', [], 120), w('worker-1', 'Decode 2', [], 64)], {
    overlapCredit: 0,
    temperature: 0,
    assumeKvReuse: false,
    blockSize: 16,
  })
  const aggRoute = routeKV(prompt.blocks, [w('worker-0', 'Воркер 1', shared, 48), w('worker-1', 'Воркер 2', [], 24)], {
    overlapCredit: 1,
    temperature: 0,
    blockSize: 16,
  })
  const scoreOf = (d: KvRouteDecision) => d.scores.find((x) => x.workerId === d.chosen)!
  const cachedTokens = scoreOf(prefillRoute).cachedTokens
  const aggCachedTokens = scoreOf(aggRoute).cachedTokens
  const n = prompt.ids.length
  const kvBytes = kvBytesForSeq(model, n)
  const transfer = ['ib-ndr', 'nvlink4'].map((id) => {
    const link = getInterconnect(id)
    return { link: link.name, ms: transferMs(kvBytes, link.GBs) }
  })

  const { text, script } = answerText(s)
  const encoded = tokenizer.encode(text)
  const truncated = encoded.length > MAX_TOKENS
  const tokens = truncated ? encoded.slice(0, MAX_TOKENS) : encoded
  const detok = incrementalDetokenize(tokens, tokenizer, { skipSpecial: true, stopTokenIds: STOP_IDS, promptTokens: prompt.tokens })
  const parser = jailToolCalls(
    runStreamingParser(
      detok.map((d) => d.emitted),
      { reasoning: 'qwen3', tools: s.tools ? 'hermes' : 'none', idSeed: '7f3a' },
    ),
  )
  const toolCalls = parser.flatMap((e) => e.toolCalls ?? (e.toolCall ? [e.toolCall] : []))
  const meta = {
    id: 'chatcmpl-dyn-7f3a',
    created: Math.floor(Date.UTC(2026, 10, 12, 12) / 1000),
    model: model.hfId ?? model.name,
    finishReason: truncated ? ('length' as const) : ('stop' as const),
    includeUsage: true,
    reasoningField: 'reasoning_content' as const,
  }
  const sse = toSseChunks(parser, { ...meta, usage: { promptTokens: n, completionTokens: tokens.length, cachedTokens } })
  const sseAgg = toSseChunks(parser, { ...meta, usage: { promptTokens: n, completionTokens: tokens.length, cachedTokens: aggCachedTokens } })

  // Тело запроса в том же порядке полей, что у симулятора (engine/flow.ts, раздел 1).
  const json = s.json && script.json ? script.json : null
  const request: Record<string, unknown> = {
    model: model.hfId ?? model.name,
    messages: [...(parts.systemText.trim() ? [{ role: 'system', content: parts.systemText }] : []), { role: 'user', content: parts.userText }],
    ...(s.tools ? { tools: [WEATHER_TOOL], tool_choice: 'auto' } : {}),
    ...(json ? { response_format: { type: 'json_schema', json_schema: { name: json.name, schema: json.schema, strict: true } } } : {}),
    temperature: s.temperature,
    max_tokens: MAX_TOKENS,
    stream: true,
    stream_options: { include_usage: true },
    chat_template_kwargs: { enable_thinking: s.thinking },
  }

  return {
    model,
    gpu,
    request,
    prompt,
    prefillRoute,
    aggRoute,
    decodeRoute,
    cachedTokens,
    aggCachedTokens,
    prefillMs: prefillMs(model, gpu, n, { cachedTokens }),
    aggPrefillMs: prefillMs(model, gpu, n, { cachedTokens: aggCachedTokens }),
    kvBytes,
    kvBlocks: Math.ceil(n / 16),
    transfer,
    itlMs: decodeStepMs(model, gpu, { batch: 1, avgContext: n + tokens.length / 2 }),
    answer: { tokens, detok, parser, sse, sseAgg, thinking: s.thinking, bodyKind: script.bodyKind, toolCalls, truncated },
    jsonSchemaName: json ? json.name : null,
    maxTokens: MAX_TOKENS,
  }
}

/** Узел схемы для выбранного воркера: worker-0 → p1 / d1 / w1. */
export function chosenNode(prefix: 'p' | 'd' | 'w', d: KvRouteDecision): string {
  return `${prefix}${d.chosen === 'worker-1' ? 2 : 1}`
}

/**
 * Событие BlockStored после prefill (как vLLM block_pool.cache_full_blocks): только новые полные блоки, то есть
 * те, что лежат за уже закэшированным префиксом; parent_hash указывает на последний закэшированный блок.
 * Возвращает null, если новых полных блоков нет (тогда событие не публикуется).
 */
export function blockStored(prompt: BuiltPrompt, overlapBlocks: number) {
  const full = prompt.blocks.filter((b) => b.full)
  const fresh = full.slice(overlapBlocks)
  if (fresh.length === 0) return null
  const B = 16
  const hashes = fresh.map((b) => b.hash)
  const tokenIds = prompt.ids.slice(overlapBlocks * B, full.length * B)
  return {
    newBlocks: fresh.length,
    event: {
      event_id: 4812,
      type: 'BlockStored',
      block_hashes: hashes.length > 3 ? [...hashes.slice(0, 3), '…'] : hashes,
      parent_hash: overlapBlocks > 0 ? full[overlapBlocks - 1].hash : null,
      num_block_tokens: B,
      token_ids: tokenIds.length > 6 ? [...tokenIds.slice(0, 6), '…'] : tokenIds,
    },
  }
}
