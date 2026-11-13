/**
 * Фигура 2 главы 20: поток запросов от нескольких «тенантов» (у каждого свой системный промпт)
 * через четыре воркера при трёх стратегиях. Всё считает engine/router.ts:
 * makeTenantWorkload (настоящие блоки и хэши) + simulateRouting (prefill-очередь FIFO, TTFT из perf.ts).
 */
import type { GpuSpec, ModelSpec } from '@/engine/types'
import { makeTenantWorkload, makeWorkers, simulateRouting, type RouteMode, type RoutingSimResult } from '@/engine/router'

export interface FleetConfig {
  tenants: number
  systemTokens: number
  /** Ёмкость GPU-кэша одного воркера, в токенах (в блоках = / 16). */
  capacityTokens: number
  ratePerSec: number
}

export const FLEET_WORKERS = 4
export const FLEET_REQUESTS = 240
export const FLEET_MODES: RouteMode[] = ['kv', 'round-robin', 'random']
/** Точки графика «TTFT против потока». */
export const FLEET_RATES = [10, 25, 50, 75, 100, 150, 200]

export const MODE_LABEL: Record<RouteMode, string> = { kv: 'KV-aware', 'round-robin': 'Round-robin', random: 'Случайно' }

/** modes: какие стратегии считать (по умолчанию все три; стратегии считаются независимо, результат тот же). */
export function runFleet(cfg: FleetConfig, model: ModelSpec, gpu: GpuSpec, rate = cfg.ratePerSec, modes: readonly RouteMode[] = FLEET_MODES): RoutingSimResult {
  const requests = makeTenantWorkload({
    tenants: cfg.tenants,
    systemTokens: cfg.systemTokens,
    userTokens: [64, 320],
    requests: FLEET_REQUESTS,
    ratePerSec: rate,
    outputTokens: 200,
    seed: 7,
  })
  return simulateRouting(requests, makeWorkers(FLEET_WORKERS), {
    model,
    gpu,
    capacityBlocks: Math.max(16, Math.round(cfg.capacityTokens / 16)),
    router: { overlapCredit: 1, temperature: 0 },
    strategies: [...modes],
    seed: 3,
  })
}
