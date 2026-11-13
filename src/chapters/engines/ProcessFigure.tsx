import { useState, type ReactNode } from 'react'
import { Button, FieldLabel, Figure, Formula, Segmented, Stat, cn, fmt } from '@/ui'

type Opt = '1' | '2' | '4' | '8'

const PRESETS: Array<{ value: string; label: string; tp: Opt; pp: Opt; dp: Opt }> = [
  { value: 'tp4', label: '-tp=4', tp: '4', pp: '1', dp: '1' },
  { value: 'tp2dp4', label: '-tp=2 -dp=4', tp: '2', pp: '1', dp: '4' },
]

/**
 * Фигура 2 главы 18: какие процессы запускает vLLM V1 при заданных TP, PP и DP.
 * Формула из документации: A + DP + N (+1 координатор при DP > 1), N = DP · PP · TP, A = DP (notes-engines.md §1.3).
 */
export function ProcessFigure() {
  const [tp, setTp] = useState<Opt>('4')
  const [pp, setPp] = useState<Opt>('1')
  const [dp, setDp] = useState<Opt>('1')
  const T = Number(tp)
  const P = Number(pp)
  const D = Number(dp)
  const A = D
  const N = D * P * T
  const coord = D > 1 ? 1 : 0
  const total = A + D + N + coord
  const preset = PRESETS.find((x) => x.tp === tp && x.pp === pp && x.dp === dp)?.value ?? 'own'

  return (
    <Figure
      id="engines-figure-processes"
      title="Сколько процессов запускает vLLM"
      width="wide"
      controls={
        <>
          <div className="grid content-start gap-1.5">
            <FieldLabel>Примеры из документации</FieldLabel>
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((x) => (
                <Button
                  key={x.value}
                  size="sm"
                  variant={preset === x.value ? 'primary' : 'secondary'}
                  aria-pressed={preset === x.value}
                  onClick={() => {
                    setTp(x.tp)
                    setPp(x.pp)
                    setDp(x.dp)
                  }}
                >
                  {x.label}
                </Button>
              ))}
            </div>
          </div>
          <Segmented label="Tensor parallel, TP" size="sm" value={tp} onChange={setTp} options={(['1', '2', '4', '8'] as Opt[]).map((v) => ({ value: v, label: v }))} />
          <Segmented label="Pipeline parallel, PP" size="sm" value={pp} onChange={setPp} options={(['1', '2'] as Opt[]).map((v) => ({ value: v, label: v }))} />
          <Segmented label="Data parallel, DP" size="sm" value={dp} onChange={setDp} options={(['1', '2', '4'] as Opt[]).map((v) => ({ value: v, label: v }))} />
        </>
      }
      caption="Попробуйте: выберите пример «-tp=2 -dp=4» и проверьте формулу: 4 API server, 4 EngineCore, 8 GPU-воркеров и координатор, всего 17 процессов. Каждый API server говорит с каждым EngineCore по ZMQ."
    >
      <div className="grid min-w-0 gap-5">
        <Row label={`API server × ${A}`} hint="HTTP, токенизация, детокенизация">
          {Array.from({ length: A }, (_, i) => (
            <Box key={i} title={`API server ${i + 1}`} />
          ))}
        </Row>
        {coord > 0 && (
          <Row label="DP-координатор" hint="балансирует запросы между EngineCore">
            <Box title="координатор" dashed />
          </Row>
        )}
        <div className="grid min-w-0 gap-3">
          <div className="flex flex-wrap items-baseline gap-x-2 text-[13px]">
            <span className="font-medium text-ink">EngineCore × {D}</span>
            <span className="text-ink-3">по одному на DP-ранг, у каждого {fmt.count(T * P, ['GPU-воркер', 'GPU-воркера', 'GPU-воркеров'])}</span>
          </div>
          <div className={cn('grid min-w-0 gap-3', D === 2 && '@xl:grid-cols-2', D === 4 && '@xl:grid-cols-2 @4xl:grid-cols-4')}>
            {Array.from({ length: D }, (_, d) => (
              <div key={d} className="grid min-w-0 content-start gap-2 rounded-panel border border-line bg-surface-2 p-3">
                <Box title={`EngineCore ${d + 1}`} hint="scheduler, KV cache manager" />
                <div className="grid gap-1.5">
                  {Array.from({ length: P }, (_, s) => (
                    <div key={s} className="grid gap-1">
                      {P > 1 && <span className="text-[12px] text-ink-3">стадия PP {s + 1}: слои {s === 0 ? 'первой' : 'второй'} половины</span>}
                      <div className="flex flex-wrap gap-1.5">
                        {Array.from({ length: T }, (_, t) => (
                          <span
                            key={t}
                            className="inline-flex h-8 min-w-[3.25rem] items-center justify-center rounded-control border border-line-strong bg-surface px-2 font-mono text-[12px] text-ink tabular-nums"
                            title={`GPU ${d * P * T + s * T + t}: 1/${T} каждого слоя${P > 1 ? `, стадия ${s + 1}` : ''}`}
                          >
                            GPU {d * P * T + s * T + t}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="grid gap-3 border-t border-line pt-4">
          <Stat
            label="Всего процессов"
            value={fmt.int(total)}
            size="lg"
            hint={`${fmt.count(A, ['API server', 'API server', 'API server'])} + ${fmt.int(D)} EngineCore + ${fmt.count(N, ['GPU-воркер', 'GPU-воркера', 'GPU-воркеров'])}${coord ? ' + координатор' : ''}`}
          />
          {/* формула в две-три короткие строки: на телефоне итог не уезжает за край */}
          <Formula
            block
            className="!mt-0"
            tex={
              `\\begin{aligned}\\text{всего} &= A + DP + N${coord ? ' + 1' : ''}\\\\ &= ${A} + ${D} + ${N}${coord ? ' + 1' : ''} = ${total}\\\\ ` +
              `N &= DP\\cdot PP\\cdot TP\\\\ &= ${D}\\cdot ${P}\\cdot ${T} = ${N}\\ \\text{GPU}\\end{aligned}`
            }
          />
          <p className="text-[14px] leading-snug text-ink-2">
            A означает число API server (по умолчанию равно DP), N означает число GPU-воркеров, по одному процессу на GPU. Каждый воркер TP держит 1/{T} матриц каждого слоя и
            на каждом слое обменивается результатами с соседями (all-reduce), поэтому TP живёт внутри одной ноды с NVLink.
          </p>
        </div>
      </div>
    </Figure>
  )
}

function Row({ label, hint, children }: { label: string; hint: string; children: ReactNode }) {
  return (
    <div className="grid min-w-0 gap-2">
      <div className="flex flex-wrap items-baseline gap-x-2 text-[13px]">
        <span className="font-medium text-ink">{label}</span>
        <span className="text-ink-3">{hint}</span>
      </div>
      <div className="flex flex-wrap gap-2">{children}</div>
    </div>
  )
}

function Box({ title, hint, dashed }: { title: string; hint?: string; dashed?: boolean }) {
  return (
    <span
      className={cn(
        'inline-grid min-w-[8rem] gap-0.5 rounded-control border bg-surface px-3 py-2 text-[13px]',
        dashed ? 'border-dashed border-line-strong' : 'border-line-strong',
      )}
    >
      <span className="font-medium text-ink">{title}</span>
      {hint && <span className="text-[12px] text-ink-3">{hint}</span>}
    </span>
  )
}
