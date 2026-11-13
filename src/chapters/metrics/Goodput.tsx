import { useEffect, useMemo, useState } from 'react'
import { CheckCircleIcon } from '@phosphor-icons/react'
import { maxConcurrentSeqs } from '@/engine/perf'
import { Figure, LineChart, Skeleton, Slider, Stat, fmt, useInView } from '@/ui'
import { VLLM_MEM_UTIL, hwLabel, type Hardware } from '@/chapters/prefill-decode/shared/hardware'
import { CONCURRENCY, simulateLevel, type LevelResult } from './goodput-sim'

/** Симуляция по уровням concurrency, по одному уровню за раз (без долгих задач в главном потоке). */
function useLevels(hw: Hardware, enabled: boolean): { levels: LevelResult[]; done: boolean; total: number } {
  const plan = useMemo(() => {
    const maxSeqs = maxConcurrentSeqs(hw.model, hw.gpu, { contextLen: 1500, dtype: hw.dtype, gpus: hw.tp, memUtil: VLLM_MEM_UTIL })
    return CONCURRENCY.filter((c) => c <= Math.max(16, 2 * maxSeqs))
  }, [hw])
  const [state, setState] = useState<{ key: Hardware | null; levels: LevelResult[]; done: boolean }>({ key: null, levels: [], done: false })
  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    let timer = 0
    setState({ key: hw, levels: [], done: false })
    let i = 0
    const step = () => {
      if (cancelled) return
      const res = simulateLevel(hw, plan[i], 11 + i)
      i += 1
      const finished = i >= plan.length
      setState((s) => (s.key === hw ? { key: hw, levels: [...s.levels, res], done: finished } : s))
      if (!finished) timer = window.setTimeout(step, 16)
    }
    timer = window.setTimeout(step, 30)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [hw, plan, enabled])
  return { levels: state.key === hw ? state.levels : [], done: state.key === hw && state.done, total: plan.length }
}

export function GoodputFigure({ hw }: { hw: Hardware }) {
  // по умолчанию пороги MLPerf Inference для интерактивного режима: TTFT 450 мс, TPOT 40 мс
  const [sloTtft, setSloTtft] = useState(450)
  const [sloTpot, setSloTpot] = useState(40)
  // симуляцию запускаем, когда фигура подъехала к экрану (глава монтируется заранее)
  const [viewRef, near] = useInView<HTMLDivElement>({ rootMargin: '600px 0px', once: true })
  const { levels, done, total } = useLevels(hw, near)

  const rows = useMemo(
    () =>
      levels.map((l) => {
        let ok = 0
        for (let k = 0; k < l.ttft.length; k++) if (l.ttft[k] <= sloTtft && l.tpot[k] <= sloTpot) ok++
        // goodput: доля запросов в SLO × завершённые запросы в секунду
        const good = l.ttft.length ? (ok / l.ttft.length) * l.reqPerSec : 0
        return { ...l, good, fraction: l.ttft.length ? ok / l.ttft.length : 0 }
      }),
    [levels, sloTtft, sloTpot],
  )
  const best = rows.length ? rows.reduce((a, b) => (b.good > a.good ? b : a)) : null
  // слайдеры SLO стоят между итогами и графиками: на телефоне порог и то, что он двигает, видны вместе
  const sliders = (
    <div className="grid gap-x-6 gap-y-4 border-y border-line py-4 @xl:grid-cols-2">
      <Slider
        label="SLO по TTFT"
        value={sloTtft}
        min={100}
        max={5000}
        scale="log"
        step={10}
        onChange={setSloTtft}
        format={(v) => fmt.ms(v)}
        hint="каждый запрос должен уложиться"
      />
      <Slider
        label="SLO по TPOT (≈ ITL)"
        value={sloTpot}
        min={10}
        max={200}
        step={1}
        onChange={setSloTpot}
        format={(v) => fmt.ms(v)}
        hint="средний интервал между токенами одного запроса"
      />
    </div>
  )
  const peak = rows.length ? rows.reduce((a, b) => (b.reqPerSec > a.reqPerSec ? b : a)) : null

  return (
    <Figure
      id="metrics-figure-goodput"
      title="Throughput против латентности и goodput"
      width="wide"
      actions={<span className="block max-w-[18rem] text-[13px] sm:max-w-none leading-snug text-ink-3">{hwLabel(hw)}, continuous batching</span>}
      caption="Попробуйте: ужесточите SLO по TPOT до 20 мс и посмотрите, как пик goodput уезжает к меньшей concurrency, а потом ослабьте его до 100 мс. По умолчанию пороги как в MLPerf для интерактивного режима (TTFT 450 мс, TPOT 40 мс)."
      source="Как считаем: клиент держит в работе C запросов (новый уходит, когда закончился предыдущий), длины случайные (промпт около 1 000 токенов, ответ около 200), шаги считает симуляция continuous batching с тем же расчётом времени шага, что в главе «Compute-bound и memory-bound». Ожидание у клиента в TTFT не входит."
    >
      {rows.length === 0 ? (
        <div ref={viewRef} className="grid gap-3">
          <Skeleton height={64} />
          {sliders}
          <Skeleton height={220} />
          <span className="text-[13px] text-ink-3">Считаем симуляцию…</span>
        </div>
      ) : (
        <div className="grid gap-5">
          <div className="grid grid-cols-2 gap-x-6 gap-y-4 @xl:grid-cols-4">
            <Stat label="Пик throughput" value={fmt.num(peak!.reqPerSec, 1)} unit="запр/с" hint={`${fmt.int(peak!.tokPerSec)} ток/с при concurrency ${fmt.int(peak!.concurrency)}`} />
            <Stat
              label="Лучший goodput"
              value={fmt.num(best!.good, 1)}
              unit="запр/с"
              hint={best!.good > 0 ? `при concurrency ${fmt.int(best!.concurrency)}` : 'ни один уровень не держит SLO'}
              swatch="var(--good)"
            />
            <Stat label="В SLO на лучшей точке" value={fmt.pct(best!.fraction)} hint="доля запросов" />
            <Stat label="TTFT p90 там же" value={fmt.ms(best!.ttftP90)} hint={`TPOT p90 ${fmt.ms(best!.tpotP90)}`} />
          </div>
          {sliders}
          <div className="grid gap-6 @3xl:grid-cols-2">
            <div className="min-w-0">
              <div className="mb-1 text-[14px] font-medium text-ink">Сколько выдаёт сервис</div>
              <LineChart
                series={[
                  { id: 'thr', label: 'Throughput', color: 'ink-3', points: rows.map((r) => [r.concurrency, r.reqPerSec]), markers: true },
                  { id: 'good', label: 'Goodput', color: 'good', points: rows.map((r) => [r.concurrency, r.good]), markers: true },
                ]}
                xLabel="Одновременных запросов (concurrency)"
                yLabel="Запросов в секунду"
                xScale="log"
                height={220}
                xFormat={(v) => fmt.int(v)}
                yFormat={(v) => fmt.num(v)}
                aria-label="Throughput и goodput в зависимости от concurrency"
              />
            </div>
            <div className="min-w-0">
              <div className="mb-1 text-[14px] font-medium text-ink">Сколько ждёт пользователь (p90)</div>
              <LineChart
                series={[
                  { id: 'ttft', label: 'TTFT p90', color: 'prefill', points: rows.map((r) => [r.concurrency, r.ttftP90]), markers: true },
                  { id: 'tpot', label: 'TPOT p90', color: 'decode', points: rows.map((r) => [r.concurrency, r.tpotP90]), markers: true },
                ]}
                xLabel="Одновременных запросов (concurrency)"
                yLabel="Миллисекунды, логарифмическая шкала"
                xScale="log"
                yScale="log"
                annotations={[
                  { kind: 'y', y: sloTtft, label: `SLO TTFT ${fmt.ms(sloTtft)}`, labelAlign: 'start' },
                  { kind: 'y', y: sloTpot, label: `SLO TPOT ${fmt.ms(sloTpot)}`, labelAlign: 'start' },
                ]}
                height={220}
                xFormat={(v) => fmt.int(v)}
                yFormat={(v) => fmt.compact(v)}
                aria-label="TTFT p90 и TPOT p90 в зависимости от concurrency, с порогами SLO"
              />
            </div>
          </div>
          {!done && <div className="text-[13px] text-ink-3">Досчитываем уровни concurrency: {rows.length} из {total}…</div>}
          {done && best && best.good > 0 && (
            <div className="flex items-start gap-2 text-[14px] leading-snug text-ink-2">
              <CheckCircleIcon size={18} className="mt-px shrink-0 text-good" aria-hidden />
              <span>
                <span className="font-medium text-ink">Рабочая точка:</span> около {fmt.int(best.concurrency)} одновременных запросов.
                {peak!.reqPerSec > best.reqPerSec * 1.02
                  ? ` Дальше throughput ещё растёт на ${fmt.pct(peak!.reqPerSec / best.reqPerSec - 1)}, но всё больше запросов выпадает из SLO.`
                  : ' Здесь же и пик throughput: SLO пока не ограничивает.'}
              </span>
            </div>
          )}
        </div>
      )}
    </Figure>
  )
}
