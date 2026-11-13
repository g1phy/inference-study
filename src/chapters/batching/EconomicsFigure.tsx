import { useMemo } from 'react'
import { Badge, Figure, ScrollX, fmt } from '@/ui'
import { useSettings } from '@/state/settings'
import { decodeBatchSweep, type DecodeSweepPoint } from '@/engine/perf'
import { configLabel, simConfig } from './config'

export const ECON_BATCHES = [1, 8, 32, 128] as const
export const ECON_CONTEXT = 1024

/** Таблица «batch → шаг decode → скорость» по roofline-модели perf.ts. */
export function useEconomics() {
  const [settings] = useSettings()
  const cfg = simConfig(settings.modelId, settings.gpuId)
  const rows = useMemo(
    () => decodeBatchSweep(cfg.model, cfg.gpu, [...ECON_BATCHES], { avgContext: ECON_CONTEXT, gpus: cfg.tp, dtype: cfg.dtype }),
    [cfg],
  )
  return { cfg, rows }
}

export function EconomicsFigure({ rows, label, moe = false }: { rows: DecodeSweepPoint[]; label: string; moe?: boolean }) {
  const base = rows[0]
  const growth = moe
    ? 'У MoE-модели шаг растёт быстро: чем больше запросов, тем больше разных экспертов читается за шаг. Но токенов за шаг столько же, сколько запросов, и всего токенов в секунду всё равно в разы больше.'
    : 'Шаг растёт медленно, а токенов за шаг столько же, сколько запросов.'
  return (
    <Figure
      id="batching-figure-economics"
      width="narrow"
      title="Цена одного шага decode при разном batch"
      actions={<Badge>{label}</Badge>}
      caption={`Расчёт по roofline-модели (MBU 70 %, MFU 50 %), у каждой последовательности ${fmt.tokens(ECON_CONTEXT)} контекста. ${growth}`}
    >
      <ScrollX label="Цена шага decode">
        <table className="w-full border-collapse text-sm @md:text-[15px]">
          <thead>
            <tr className="border-b border-line text-left text-sm text-ink-3">
              <th scope="col" className="py-2 pr-2 font-normal @md:pr-3">
                Запросов<span className="hidden @md:inline"> в шаге</span>
              </th>
              <th scope="col" className="py-2 pr-2 text-right font-normal @md:pr-3">
                Шаг<span className="hidden @md:inline"> decode</span>
              </th>
              <th scope="col" className="hidden py-2 pr-3 text-right font-normal @md:table-cell">
                Скорость одного
              </th>
              <th scope="col" className="py-2 text-right font-normal">
                Всего<span className="@md:hidden">, ток/с</span>
                <span className="hidden @md:inline"> токенов/с</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.batch} className="border-b border-line align-top last:border-b-0">
                <td className="py-2 pr-2 font-medium tabular-nums @md:pr-3">{r.batch}</td>
                <td className="py-2 pr-2 text-right tabular-nums @md:pr-3">
                  {fmt.ms(r.stepMs)}
                  {r.batch > 1 && <span className="ml-1.5 hidden text-sm text-ink-3 @md:inline">{fmt.multiplier(r.stepMs / base.stepMs, 2)}</span>}
                  <span className="block text-xs text-ink-3 @md:hidden">{fmt.rate(r.perUserTokS)} на одного</span>
                </td>
                <td className="hidden py-2 pr-3 text-right tabular-nums @md:table-cell">{fmt.rate(r.perUserTokS)}</td>
                <td className="py-2 text-right font-semibold whitespace-nowrap tabular-nums">
                  {fmt.int(r.totalTokS)}
                  {r.batch > 1 && <span className="ml-1.5 text-sm font-normal text-ink-3">{fmt.multiplier(r.totalTokS / base.totalTokS, 0)}</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollX>
    </Figure>
  )
}
