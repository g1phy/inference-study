/**
 * Чистые помощники симулятора «Путь запроса»: длительности шагов плеера, статусы узлов схемы,
 * что видит клиент по SSE, разрывная ось таймлайна. Без React, покрыто model.test.ts.
 */
import type { FlowRunEx } from '@/engine/flow'
import type { FlowPlaybackStep, SseChunk, StageId } from '@/engine/types'
import { fmt } from '@/ui/fmt'

/* ============================================================================
 * Темп проигрывания (анимация замедлена, реальные мс подписываются отдельно)
 * ========================================================================== */

/** Шаг-стадия при скорости 1×: пакет успевает доехать, подпись успевает прочитаться. */
export const STAGE_STEP_MS = 1100
/** Первые токены идут медленно: видно, как токен проходит decode → postprocessor → клиент. */
export const SLOW_TOKENS = 3
export const SLOW_TOKEN_MS = 650
/** Финальный шаг (finish_reason, usage, [DONE]). */
export const DONE_STEP_MS = 1400

/** Длительность шага-токена: вся генерация занимает около 9 с при 1×, но не быстрее 50 мс на токен. */
export function tokenStepMs(tokens: number): number {
  return Math.round(Math.min(220, Math.max(50, 9000 / Math.max(1, tokens))))
}

/** Длительность каждого шага плеера при скорости 1×, мс. */
export function stepDurations(run: Pick<FlowRunEx, 'playback' | 'decode'>): number[] {
  const pb = run.playback ?? []
  const fast = tokenStepMs(run.decode.length)
  return pb.map((p) => {
    if (p.kind === 'stage') return STAGE_STEP_MS
    if (p.kind === 'token') return (p.token ?? 0) < SLOW_TOKENS ? SLOW_TOKEN_MS : fast
    return DONE_STEP_MS
  })
}

/* ============================================================================
 * Шаги плеера и стадии
 * ========================================================================== */

/** id колонки инспектора и его заголовка: к ним ведёт ссылка «Где это в симуляторе» (прокрутка и фокус). */
export const INSPECTOR_ID = 'simulator-inspector'
export const INSPECTOR_HEADING_ID = 'simulator-inspector-title'

const TOKEN_STAGES: ReadonlySet<StageId> = new Set<StageId>(['decode', 'postprocess', 'stream'])

/** Индекс первого шага плеера, на котором работает стадия (−1, если стадии нет в прогоне). */
export function firstStepOf(playback: readonly FlowPlaybackStep[], stage: StageId): number {
  if (TOKEN_STAGES.has(stage)) {
    const i = playback.findIndex((p) => p.kind === 'token')
    return i >= 0 ? i : playback.findIndex((p) => p.stage === stage)
  }
  return playback.findIndex((p) => p.kind === 'stage' && p.stage === stage)
}

/**
 * (X-sim-r2-6) Шаг плеера для ссылки «Где это в симуляторе» на decode. С JSON-схемой: первый шаг под маской (там видно,
 * что маска отрезала). Иначе первый настоящий шаг decode: в agg первый токен выбран в конце prefill, у него нет ни
 * полос «Время и упор», ни роста KV, поэтому берётся второй; в disagg первый.
 */
export function firstDecodeStep(run: Pick<FlowRunEx, 'playback' | 'decode'>): number {
  const pb = run.playback ?? []
  const masked = run.decode.findIndex((d) => d.masked)
  const k = masked >= 0 ? masked : run.decode.findIndex((d) => d.phase === 'decode')
  const i = k >= 0 ? pb.findIndex((p) => p.kind === 'token' && p.token === k) : -1
  return i >= 0 ? i : firstStepOf(pb, 'decode')
}

/**
 * (X-sim-r2-5) Тот же ли у прогонов порядок шагов плеера: та же длина, те же виды и стадии. Так бывает, когда поменялись
 * temperature, модель, GPU или prefix cache: шаг прошлого прогона в новом значит то же самое.
 */
export function samePlaybackShape(a: readonly FlowPlaybackStep[], b: readonly FlowPlaybackStep[]): boolean {
  return a.length === b.length && a.every((x, i) => x.kind === b[i].kind && x.stage === b[i].stage && x.token === b[i].token)
}

/**
 * (X-sim-r2-4) Обычный шаг decode, если в прогон попал prefill соседа: среднее ITL тогда тянет вверх один долгий шаг,
 * и рядом со средним надо показывать обычный. null, если соседа в шагах decode нет (agg без соседа, disagg).
 */
export function neighborSteps(run: Pick<FlowRunEx, 'neighbor'>): { plainMs: number; stepMs: number; count: number } | null {
  const nb = run.neighbor
  if (!nb?.applied || nb.steps.length === 0) return null
  return { plainMs: nb.plainMs, stepMs: nb.stepMs, count: nb.steps.length }
}

/** «шаг с prefill соседа 190 мс» или «3 шага с prefill соседа, до 207 мс». */
export function neighborStepsText(nb: { stepMs: number; count: number }): string {
  return nb.count === 1
    ? `шаг с prefill соседа ${fmt.ms(nb.stepMs)}`
    : `${fmt.int(nb.count)} ${fmt.plural(nb.count, ['шаг', 'шага', 'шагов'])} с prefill соседа, до ${fmt.ms(nb.stepMs)}`
}

/** Токены в родительном падеже: «из 43 токенов», «около 21 токена», «всех 1 167 токенов». */
export function tokensGen(n: number): string {
  return fmt.tokensGen(n)
}

export type NodeStatus = 'pending' | 'active' | 'done'

/** Состояние узла схемы на шаге плеера: ещё не дошли, работает сейчас, уже пройден. */
export function nodeStatus(playback: readonly FlowPlaybackStep[], step: number, stage: StageId): NodeStatus {
  const cur = playback[step]
  if (!cur) return 'pending'
  if (cur.kind === 'token') {
    if (TOKEN_STAGES.has(stage)) return 'active'
  } else if (cur.kind === 'done') {
    if (stage === 'stream') return 'active'
    if (TOKEN_STAGES.has(stage)) return 'done'
  } else if (cur.stage === stage) return 'active'
  const first = firstStepOf(playback, stage)
  return first >= 0 && first < step ? 'done' : 'pending'
}

/** Индекс выходного токена на шаге плеера: −1 до генерации, последний на финальном шаге. */
export function tokenAtStep(playback: readonly FlowPlaybackStep[], step: number, tokens: number): number {
  const cur = playback[step]
  if (!cur) return -1
  if (cur.kind === 'token') return cur.token ?? -1
  if (cur.kind === 'done') return tokens - 1
  return -1
}

/** Стадия, к которой «едет» пакет на шаге, и откуда он выехал (для анимации по рельсу схемы). */
export interface PacketMove {
  from: StageId
  to: StageId
  label: string
  kind: 'request' | 'tokens' | 'kv' | 'meta'
}

/** Движение пакета на шаге-стадии; на шагах-токенах пакетов много (поток точек), их рисует схема. */
export function packetMove(run: Pick<FlowRunEx, 'playback' | 'promptTokens' | 'transfer'>, step: number, labels: {
  json: string
  tokens: (n: number) => string
  kv: (bytes: number) => string
  meta: string
}): PacketMove | null {
  const pb = run.playback ?? []
  const cur = pb[step]
  if (!cur || cur.kind !== 'stage') return null
  const prev = step > 0 ? pb[step - 1] : null
  const from: StageId = prev && prev.kind === 'stage' ? prev.stage : cur.stage
  const n = run.promptTokens.length
  switch (cur.stage) {
    case 'client':
    case 'frontend':
      return { from, to: cur.stage, label: labels.json, kind: 'request' }
    case 'preprocess':
    case 'prefill':
      return { from, to: cur.stage, label: labels.tokens(n), kind: 'request' }
    case 'route':
      // в disagg второй шаг route: prefill-воркер вернул метаданные KV, роутер выбирает decode-воркер
      return from === 'prefill'
        ? { from, to: 'route', label: labels.meta, kind: 'meta' }
        : { from, to: 'route', label: labels.tokens(n), kind: 'request' }
    case 'transfer':
      return { from: 'prefill', to: 'decode', label: labels.kv(run.transfer?.bytes ?? 0), kind: 'kv' }
    default:
      return { from, to: cur.stage, label: labels.json, kind: 'request' }
  }
}

/* ============================================================================
 * Что видит клиент: склейка SSE-чанков
 * ========================================================================== */

export interface ClientToolCall {
  index: number
  id?: string
  name?: string
  args: string
}

export interface ClientView {
  /** Сколько чанков с данными (без [DONE]). */
  chunks: number
  role: boolean
  reasoning: string
  content: string
  tools: ClientToolCall[]
  finishReason?: string
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number; prompt_tokens_details?: { cached_tokens?: number } }
  done: boolean
}

type Delta = {
  role?: string
  content?: string
  reasoning_content?: string
  tool_calls?: Array<{ index: number; id?: string; function?: { name?: string; arguments?: string } }>
}

function choiceOf(chunk: SseChunk): { delta?: Delta; finish_reason?: string | null } | undefined {
  const choices = chunk.json?.choices as Array<{ delta?: Delta; finish_reason?: string | null }> | undefined
  return choices?.[0]
}

/** Всё, что клиент собрал из первых sseCount чанков (как это делает OpenAI SDK). */
export function clientView(sse: readonly SseChunk[], sseCount: number): ClientView {
  const out: ClientView = { chunks: 0, role: false, reasoning: '', content: '', tools: [], done: false }
  const n = Math.max(0, Math.min(sse.length, sseCount))
  for (let i = 0; i < n; i++) {
    const ch = sse[i]
    if (!ch.json) {
      out.done = true
      continue
    }
    out.chunks++
    const usage = ch.json.usage as ClientView['usage'] | undefined
    if (usage) out.usage = usage
    const c = choiceOf(ch)
    if (!c) continue
    const d = c.delta ?? {}
    if (d.role) out.role = true
    if (d.reasoning_content) out.reasoning += d.reasoning_content
    if (d.content) out.content += d.content
    for (const tc of d.tool_calls ?? []) {
      let t = out.tools.find((x) => x.index === tc.index)
      if (!t) {
        t = { index: tc.index, args: '' }
        out.tools.push(t)
      }
      if (tc.id) t.id = tc.id
      if (tc.function?.name) t.name = tc.function.name
      if (tc.function?.arguments) t.args += tc.function.arguments
    }
    if (c.finish_reason) out.finishReason = c.finish_reason
  }
  return out
}

/** Короткая запись чанка для ленты: что в нём нового (полная строка показывается отдельно). */
export function sseSummary(chunk: SseChunk): string {
  if (!chunk.json) return 'data: [DONE]'
  const usage = chunk.json.usage as ClientView['usage'] | undefined
  if (usage) {
    const cached = usage.prompt_tokens_details?.cached_tokens ?? 0
    return `usage: prompt ${usage.prompt_tokens}, completion ${usage.completion_tokens}, cached ${cached}`
  }
  const c = choiceOf(chunk)
  if (!c) return '{}'
  const d = c.delta ?? {}
  const parts: string[] = []
  if (d.role) parts.push(`role: "${d.role}"`)
  if (d.reasoning_content !== undefined && d.reasoning_content !== '') parts.push(`reasoning_content: ${JSON.stringify(d.reasoning_content)}`)
  if (d.content !== undefined && d.content !== '' ) parts.push(`content: ${JSON.stringify(d.content)}`)
  for (const tc of d.tool_calls ?? []) {
    if (tc.function?.name) parts.push(`tool_calls[${tc.index}]: ${tc.function.name}`)
    if (tc.function?.arguments) parts.push(`tool_calls[${tc.index}].arguments: ${JSON.stringify(tc.function.arguments)}`)
  }
  if (c.finish_reason) parts.push(`finish_reason: "${c.finish_reason}"`)
  return parts.length ? parts.join(', ') : 'delta: {}'
}

/** Канал чанка (для цветной метки в ленте SSE). */
export function sseChannel(chunk: SseChunk): 'reasoning' | 'content' | 'tool' | 'meta' {
  const d = choiceOf(chunk)?.delta
  if (!d) return 'meta'
  if (d.tool_calls?.length) return 'tool'
  if (d.reasoning_content) return 'reasoning'
  if (d.content) return 'content'
  return 'meta'
}

/* ============================================================================
 * Дорожки парсера до токена k включительно
 * ========================================================================== */

export interface Lanes {
  reasoning: string
  content: string
  tool: string
}

/** Что парсер разложил по дорожкам к концу шага k (сырые дельты, как их видит postprocessor). */
export function lanesUpTo(parser: FlowRunEx['parser'], k: number): Lanes {
  const out: Lanes = { reasoning: '', content: '', tool: '' }
  for (const ev of parser) {
    if (ev.step > k) break
    for (const d of ev.deltas) {
      if (d.channel === 'reasoning') out.reasoning += d.text
      else if (d.channel === 'content') out.content += d.text
      else out.tool += d.text
    }
  }
  return out
}

/* ============================================================================
 * Таймлайн: разрывная ось (до TTFT и генерация)
 * ========================================================================== */

/** Шаг «круглых» делений: 1, 2, 2,5, 5 × 10^k. */
export function niceStep(span: number, target: number): number {
  const raw = span / Math.max(1, target)
  const mag = 10 ** Math.floor(Math.log10(raw))
  const norm = raw / mag
  const pick = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10
  return pick * mag
}

/** Круглые деления внутри [t0, t1]. */
export function ticks(t0: number, t1: number, target = 4): number[] {
  if (!(t1 > t0)) return [t0]
  const step = niceStep(t1 - t0, target)
  const out: number[] = []
  for (let v = Math.ceil(t0 / step - 1e-9) * step; v <= t1 + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6 + 0)
  return out
}

export interface BrokenAxis {
  /** Время → x. */
  x: (t: number) => number
  /** x → время (для клика по таймлайну). */
  t: (x: number) => number
  a: { x0: number; x1: number; t0: number; t1: number }
  b: { x0: number; x1: number; t0: number; t1: number }
}

/**
 * Две линейные шкалы встык с разрывом: слева [0, ttft] на ширину wA, справа [ttft, e2e] на ширину wB.
 * Так видны и доли миллисекунды до первого токена, и секунды генерации.
 */
export function brokenAxis(ttft: number, e2e: number, x0: number, wA: number, gap: number, wB: number): BrokenAxis {
  const T = Math.max(ttft, 1e-6)
  const E = Math.max(e2e, T + 1e-3)
  const a = { x0, x1: x0 + wA, t0: 0, t1: T }
  const b = { x0: x0 + wA + gap, x1: x0 + wA + gap + wB, t0: T, t1: E }
  const x = (t: number) => {
    if (t <= T) return a.x0 + (Math.max(0, t) / T) * wA
    return b.x0 + ((Math.min(t, E) - T) / (E - T)) * wB
  }
  const t = (px: number) => {
    if (px <= a.x1 + gap / 2) return (Math.min(Math.max(px, a.x0), a.x1) - a.x0) / wA * T
    return T + ((Math.min(Math.max(px, b.x0), b.x1) - b.x0) / wB) * (E - T)
  }
  return { x, t, a, b }
}

/** Шаг плеера, ближайший к моменту времени (первый, чей tMs не меньше t). */
export function stepAtTime(playback: readonly FlowPlaybackStep[], t: number): number {
  for (let i = 0; i < playback.length; i++) if (playback[i].tMs >= t - 1e-9) return i
  return Math.max(0, playback.length - 1)
}

/* ============================================================================
 * Мелочи для подписей
 * ========================================================================== */

/** Первые слова текста для подписи (по границе слова). */
export function excerpt(text: string, max = 48): string {
  const s = text.replace(/\s+/g, ' ').trim()
  if (s.length <= max) return s
  const cut = s.slice(0, max)
  const sp = cut.lastIndexOf(' ')
  return `${(sp > max * 0.6 ? cut.slice(0, sp) : cut).replace(/[\s,.;:!?—-]+$/u, '')}…`
}

/** JSON с отступами, но массивы чисел и строк в одну строку (token_ids, required, remote_block_ids). */
export function prettyJson(v: unknown, indent = 0): string {
  const pad = '  '.repeat(indent)
  if (Array.isArray(v)) {
    if (v.length === 0) return '[]'
    if (v.every((x) => x === null || typeof x !== 'object')) return `[${v.map((x) => JSON.stringify(x)).join(', ')}]`
    return `[\n${v.map((x) => `${pad}  ${prettyJson(x, indent + 1)}`).join(',\n')}\n${pad}]`
  }
  if (v && typeof v === 'object') {
    const entries = Object.entries(v as Record<string, unknown>).filter(([, x]) => x !== undefined)
    if (entries.length === 0) return '{}'
    return `{\n${entries.map(([k, x]) => `${pad}  ${JSON.stringify(k)}: ${prettyJson(x, indent + 1)}`).join(',\n')}\n${pad}}`
  }
  return JSON.stringify(v) ?? 'null'
}

/** Видимые пробелы и переводы строк (как в ленте токенов). */
export function visibleWs(s: string): string {
  return s.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '→')
}
