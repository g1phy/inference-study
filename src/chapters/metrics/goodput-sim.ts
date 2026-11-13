import { makeWorkload, mean, percentile, simulateContinuous, type Dist } from '@/engine/sched'
import { VLLM_MEM_UTIL, type Hardware } from '@/chapters/prefill-decode/shared/hw-core'

/**
 * Throughput против латентности (глава «Метрики», фигура 3): замкнутый цикл с C запросами в работе
 * поверх engine/sched.simulateContinuous (continuous batching, chunked prefill, время шага из perf.ts).
 */

const PROMPT_DIST: Dist = { kind: 'lognormal', median: 1000, sigma: 0.7, min: 32, max: 8000 }
const OUTPUT_DIST: Dist = { kind: 'lognormal', median: 200, sigma: 0.6, min: 8, max: 1500 }
/** Сколько запросов одновременно держит клиент (как vllm bench serve --max-concurrency). */
export const CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384]

export interface LevelResult {
  concurrency: number
  reqPerSec: number
  tokPerSec: number
  ttftP90: number
  tpotP90: number
  /**
   * Пары (TTFT, TPOT) запросов окна, получивших слот в устойчивом режиме: по ним пересчитываем goodput
   * при смене SLO (доля в SLO × завершённые запросы в секунду).
   */
  ttft: number[]
  tpot: number[]
  windowMs: number
}

/**
 * Замкнутый цикл: в работе всегда C запросов (новый уходит, когда закончился предыдущий).
 * Моделируем очередью из C + max(64, 1,5·C) запросов с max_num_seqs = C; ожидание в очереди в TTFT
 * не входит (у клиента с ограничением concurrency запрос ещё не отправлен). Throughput меряем в окне,
 * пока очередь не опустела (в работе ровно C): от 15 % завершений до (n − C)-го. TTFT берём только у
 * запросов, получивших слот после первого завершения: стартовая волна заполняет все слоты разом,
 * и её prefill идёт огромными шагами, чего в устойчивом режиме не бывает.
 */
export function simulateLevel(hw: Hardware, C: number, seed: number): LevelResult {
  const n = C + Math.max(64, Math.round(1.5 * C))
  const reqs = makeWorkload({ rate: Infinity, promptDist: PROMPT_DIST, outputDist: OUTPUT_DIST, seed, count: n })
  const r = simulateContinuous(reqs, { model: hw.model, gpu: hw.gpu, ...hw.perf, memUtil: VLLM_MEM_UTIL, maxSeqs: C })
  const byFinish = r.requests.slice().sort((a, b) => a.finishMs - b.finishMs)
  const i0 = Math.floor(0.15 * n)
  let i1 = n - C - 1
  if (i1 < i0 + 5) i1 = Math.floor(0.8 * n)
  const t0 = byFinish[i0].finishMs
  const t1 = byFinish[i1].finishMs
  const W = Math.max(1e-6, t1 - t0)
  let tok = 0
  for (const st of r.steps) if (st.tMs >= t0 && st.tMs < t1) tok += st.decode.length
  const inWindow = byFinish.slice(i0 + 1, i1 + 1)
  const firstFinish = byFinish[0].finishMs
  // завершились, пока в работе было ровно C (окно), и получили слот уже в устойчивом режиме
  const steady = inWindow.filter((q) => q.arrivalMs + q.queueMs >= firstFinish)
  const pairs = steady.length >= 8 ? steady : inWindow
  const ttft = pairs.map((q) => q.ttftMs - q.queueMs)
  const tpot = pairs.map((q) => (q.itlMs.length ? mean(q.itlMs) : 0))
  return {
    concurrency: C,
    reqPerSec: (inWindow.length * 1000) / W,
    tokPerSec: (tok * 1000) / W,
    ttftP90: percentile(ttft, 90),
    tpotP90: percentile(tpot, 90),
    ttft,
    tpot,
    windowMs: W,
  }
}
