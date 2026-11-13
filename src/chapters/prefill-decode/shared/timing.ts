import { decodeStepMs, prefillBreakdown } from '@/engine/perf'
import type { Hardware } from './hw-core'

export interface RequestTiming {
  promptLen: number
  outLen: number
  queueMs: number
  prefillMs: number
  prefillBound: 'memory' | 'compute'
  ttftMs: number
  /** Средний шаг decode (= ITL) по ответу. */
  itlMs: number
  firstItlMs: number
  lastItlMs: number
  e2eMs: number
}

/**
 * Тайминг одного запроса на свободном GPU (batch 1): TTFT = очередь + prefill,
 * ITL = шаг decode при растущем контексте, E2E = TTFT + Σ ITL. Шаг decode при batch 1
 * линейно растёт с контекстом (читается весь KV), поэтому среднее = полусумма крайних.
 */
export function requestTiming(state: Hardware, promptLen: number, outLen: number, queueMs: number): RequestTiming {
  const { model, gpu, perf } = state
  const pre = prefillBreakdown(model, gpu, promptLen, perf)
  const first = decodeStepMs(model, gpu, { ...perf, batch: 1, avgContext: promptLen + 1 })
  const last = decodeStepMs(model, gpu, { ...perf, batch: 1, avgContext: promptLen + Math.max(1, outLen - 1) })
  const itl = (first + last) / 2
  const ttft = queueMs + pre.ms
  return {
    promptLen,
    outLen,
    queueMs,
    prefillMs: pre.ms,
    prefillBound: pre.bound,
    ttftMs: ttft,
    itlMs: itl,
    firstItlMs: first,
    lastItlMs: last,
    e2eMs: ttft + itl * Math.max(0, outLen - 1),
  }
}

