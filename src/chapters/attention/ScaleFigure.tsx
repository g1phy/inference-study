import { useMemo, useState } from 'react'
import { BarChart, Figure, fmt, Segmented, Stat, Toggle } from '@/ui'
import { scaleDemo, softmax } from './cost'

const DIMS = [2, 8, 32, 128] as const
type Dim = (typeof DIMS)[number]

/**
 * «Зачем делить на √d_k» на случайных векторах: один запрос против 8 ключей с компонентами N(0, 1).
 * Без деления разброс оценок растёт как √d_k, и softmax отдаёт почти всё одному ключу.
 */
export function ScaleFigure({ id, headDim }: { id: string; headDim: number }) {
  const [dK, setDK] = useState<Dim>(128)
  const [scaled, setScaled] = useState(false)
  const demo = useMemo(() => scaleDemo(dK), [dK])
  const weights = useMemo(() => softmax(scaled ? demo.scores.map((x) => x / Math.sqrt(dK)) : demo.scores), [demo, scaled, dK])
  const maxW = Math.max(...weights)
  const sigma = scaled ? demo.sigma / Math.sqrt(dK) : demo.sigma
  const headMax = useMemo(() => Math.max(...softmax(scaleDemo(headDim).scores)), [headDim])

  return (
    <Figure
      id={id}
      title={
        <>
          Один запрос против 8 ключей: что делает деление на √d<sub>k</sub>
        </>
      }
      width="narrow"
      controls={
        <>
          <Segmented<Dim>
            label={
              <>
                Размерность d<sub>k</sub>
              </>
            }
            value={dK}
            onChange={setDK}
            options={DIMS.map((d) => ({ value: d, label: String(d), title: d === headDim ? 'как у головы Qwen3' : undefined }))}
          />
          <Toggle
            label={
              <>
                Делить на √d<sub>k</sub>
              </>
            }
            checked={scaled}
            onChange={setScaled}
          />
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> при d<sub>k</sub> = {headDim} (как у головы Qwen3) включите и выключите деление. Без него один
          ключ забирает {fmt.pct(headMax)} веса, и остальные слова для этой головы почти исчезают. Векторы
          случайные, но с фиксированным seed.
        </>
      }
    >
      <div className="grid min-w-0 gap-4">
        <div className="grid grid-cols-2 gap-x-6 gap-y-3">
          <Stat
            label="Разброс оценок q·k"
            value={fmt.num(sigma, sigma < 10 ? 2 : 1)}
            hint={scaled ? 'после деления около 1 при любой размерности' : `растёт как корень из размерности: √${dK} ≈ ${fmt.num(Math.sqrt(dK), 1)}`}
          />
          <Stat label="Самый большой вес" value={fmt.pct(maxW)} hint={maxW > 0.9 ? 'softmax почти выбрал один ключ' : 'вес распределён между ключами'} />
        </div>
        <BarChart
          data={weights.map((w, i) => ({ id: `k${i}`, label: `ключ ${i + 1}`, value: w, note: `оценка ${fmt.num(demo.scores[i] / (scaled ? Math.sqrt(dK) : 1), 2)}` }))}
          max={1}
          valueFormat={(v) => fmt.pct(v)}
          color="prefill"
          aria-label="Веса softmax для восьми ключей"
        />
      </div>
    </Figure>
  )
}
