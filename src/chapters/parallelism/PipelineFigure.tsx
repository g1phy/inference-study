import { useState } from 'react'
import { Figure, ScrollX, Segmented, Slider, fmt } from '@/ui'
import { cn } from '@/ui/cn'
import { bubbleFraction, decodeBubbleFraction, decodeSchedule, pipelineSchedule } from './plan'

type Mode = 'pass' | 'decode'

/** Работа микробатча на стадии: нейтральная заливка (это любой forward, не только decode). */
const WORK_FILL = 'color-mix(in oklab, var(--ink-2) 24%, var(--surface))'

/** Пузыри конвейера: S стадий (GPU), M микробатчей; один проход или непрерывный decode. */
export function PipelineFigure() {
  const [stages, setStages] = useState(4)
  const [micro, setMicro] = useState(1)
  const [mode, setMode] = useState<Mode>('pass')
  const sch = mode === 'pass' ? pipelineSchedule(stages, micro) : decodeSchedule(stages, micro)
  const slots = sch[0].length
  const bubble = mode === 'pass' ? bubbleFraction(stages, micro) : decodeBubbleFraction(stages, micro)

  return (
    <Figure
      id="parallelism-figure-pipeline"
      width="narrow"
      title="Пузыри конвейера при pipeline parallelism"
      controls={
        <>
          <Segmented<Mode>
            className="col-span-full"
            label="Режим"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'pass', label: 'Один проход' },
              { value: 'decode', label: 'Непрерывный decode' },
            ]}
          />
          <Slider label="Стадий (GPU)" value={stages} min={2} max={8} onChange={setStages} />
          <Slider label="Групп (microbatch) в конвейере" value={micro} min={1} max={16} onChange={setMicro} />
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> при четырёх стадиях и одной группе запросов каждый GPU работает четверть времени.
          Добавьте групп: пузыри сжимаются. «Один проход» показывает, например, prefill одного batch: конвейер наполняется и опустошается.
          Переключите на «Непрерывный decode»: группа, вышедшая из последней стадии, со следующим токеном снова входит в первую, поэтому при числе
          групп не меньше числа стадий GPU почти не простаивают. Отдельная группа всё так же проходит все стадии по очереди. Клетка: шаг
          одной группы на одной стадии; схема условная, все стадии одинаково долгие.
        </>
      }
    >
      <ScrollX label="Расписание конвейера" scrollerClassName="pb-1">
        <div
          className="grid gap-[3px]"
          style={{ gridTemplateColumns: `4.5rem repeat(${slots}, minmax(1.4rem, 3.5rem))`, minWidth: `${4.5 * 16 + slots * 26}px` }}
          role="table"
          aria-label={`Расписание: ${stages} ${fmt.plural(stages, ['стадия', 'стадии', 'стадий'])}, ${micro} ${fmt.plural(micro, ['группа', 'группы', 'групп'])}, ${mode === 'pass' ? 'один проход' : 'непрерывный decode'}: простой ${fmt.pct(bubble)}`}
        >
          <div role="row" className="contents">
            <span role="columnheader" className="text-xs text-ink-3">
              время →
            </span>
            {Array.from({ length: slots }, (_, t) => (
              <span key={t} role="columnheader" className="text-center text-xs text-ink-3 tabular-nums">
                {t + 1}
              </span>
            ))}
          </div>
          {sch.map((row, s) => (
            <div role="row" key={s} className="contents">
              <span role="rowheader" className="self-center text-sm text-ink-2">
                GPU {s + 1}
              </span>
              {row.map((m, t) => (
                <span
                  key={t}
                  role="cell"
                  aria-label={m >= 0 ? `группа ${m + 1}` : 'простой'}
                  className={cn('flex h-7 items-center justify-center rounded-[5px] font-mono text-[11px] font-semibold', m < 0 && 'bg-hatch shadow-[inset_0_0_0_1px_var(--line)]')}
                  style={m >= 0 ? { background: WORK_FILL, color: 'var(--ink)' } : undefined}
                >
                  {m >= 0 ? m + 1 : ''}
                </span>
              ))}
            </div>
          ))}
        </div>
      </ScrollX>
      {mode === 'decode' && (
        <p className="mt-2 text-[13px] leading-snug text-ink-3">
          Показан установившийся режим: конвейер уже заполнен, поэтому в первых слотах последние стадии ещё досчитывают предыдущий токен.
        </p>
      )}
      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-[15px]">
        <span>
          <span className="text-ink-2">Простой GPU </span>
          <span className="font-semibold text-ink tabular-nums">{fmt.pct(bubble)}</span>
        </span>
        <span>
          <span className="text-ink-2">Загрузка </span>
          <span className="font-semibold text-ink tabular-nums">{fmt.pct(1 - bubble)}</span>
        </span>
        <span className="text-ink-2">
          {mode === 'pass' ? (
            <>
              формула простоя: <span className="whitespace-nowrap">(S − 1) / (M + S − 1) = {stages - 1} / {micro + stages - 1}</span>
            </>
          ) : micro < stages ? (
            <>
              простой: <span className="whitespace-nowrap">1 − M / S = 1 − {micro} / {stages}</span>
            </>
          ) : (
            'групп не меньше, чем стадий: простоя нет'
          )}
        </span>
      </div>
    </Figure>
  )
}
