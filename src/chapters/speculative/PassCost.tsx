import { useMemo } from 'react'
import { BarChart, Figure, fmt } from '@/ui'
import { useRunningExample } from './runningExample'
import { passMs } from './spec'
import { LegendItem } from './TokenBits'

const TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

export function PassCost() {
  const { model, gpu, fallback } = useRunningExample()
  const rows = useMemo(
    () =>
      TOKENS.map((n) => {
        const r = passMs(model, gpu, n, 1000)
        return {
          id: String(n),
          label: fmt.count(n, ['токен', 'токена', 'токенов']),
          value: r.ms,
          color: r.bound === 'memory' ? 'decode' : 'prefill',
          note: r.bound === 'memory' ? 'упор в память' : 'упор в вычисления',
        }
      }),
    [model, gpu],
  )
  const one = rows[0].value
  const five = passMs(model, gpu, 5, 1000).ms
  const firstCompute = rows.find((r) => r.color === 'prefill')
  // вторая плотная модель симулятора: с ней видно, что время меняется, а граница нет
  const other = model.id === 'qwen3-32b' ? 'Qwen3-8B' : 'Qwen3-32B'
  return (
    <Figure
      id="speculative-figure-pass"
      title={`Сколько стоит один проход ${model.name} на ${gpu.name}, если в нём больше токенов`}
      caption={
        <>
          Попробуйте: выберите в симуляторе (раздел «Дополнительно») {other} или другой GPU. Время прохода изменится, а граница, с которой
          проход дорожает, почти не сдвинется: у плотной модели её задаёт не размер модели, а отношение вычислительной мощности GPU к полосе
          его памяти. Сейчас один токен стоит {fmt.ms(one)}, пять токенов {fmt.ms(five)}: пока проход упирается в чтение весов, лишние токены
          почти ничего не стоят{firstCompute ? `, а дороже он становится примерно с ${firstCompute.label}` : ''}.
          {fallback === 'moe'
            ? ' В симуляторе выбрана MoE-модель: каждый токен идёт к своим экспертам, поэтому проход из нескольких токенов задевает больше экспертов и читает больше весов. Из-за этого при малом batch спекуляция на MoE обычно даёт заметно меньше, чем на плотной модели. Чтобы показать сам принцип, здесь плотная Qwen3-8B.'
            : fallback === 'fit'
              ? ' Модель из симулятора не помещается в одну GPU, поэтому здесь Qwen3-8B.'
              : ''}
        </>
      }
      source="Время одного forward-прохода по n новым токенам при контексте 1 000 токенов: roofline-модель движка, MBU 0,7 и MFU 0,5, BF16."

    >
      <div className="grid min-w-0 gap-3">
        <BarChart data={rows} valueFormat={(v) => fmt.ms(v)} labelFont="mono" height={300} aria-label="Время прохода в зависимости от числа токенов" />
        <div className="flex flex-wrap gap-x-4 gap-y-1.5">
          <LegendItem swatch={<span className="h-3 w-3 rounded-[3px] bg-decode" />}>упор в память: время задаёт чтение весов и KV</LegendItem>
          <LegendItem swatch={<span className="h-3 w-3 rounded-[3px] bg-prefill" />}>упор в вычисления: время растёт с числом токенов</LegendItem>
        </div>
      </div>
    </Figure>
  )
}
