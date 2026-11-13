import { useState, type ReactNode } from 'react'
import { ArrowRightIcon, CheckCircleIcon, MapPinIcon, MinusCircleIcon, XCircleIcon } from '@phosphor-icons/react'
import { Figure, Segmented, Slider, Toggle, fmt, mix } from '@/ui'
import { cn } from '@/ui/cn'
import { COND_EXAMPLES, EFF_ISL_RATIO_THRESHOLD, EFF_ISL_THRESHOLD, decideConditional, type CondPolicy } from './logic'

export function CondFigure() {
  const [isl, setIsl] = useState(10000)
  const [cached, setCached] = useState(8500)
  const [policy, setPolicy] = useState<CondPolicy>('isl_bounding')
  const [busy, setBusy] = useState(false)
  const c = Math.min(cached, isl)
  const d = decideConditional({ isl, cached: c, policy, prefillBusy: busy })
  const usesIsl = policy !== 'prefill_load'
  const usesLoad = policy !== 'isl_bounding'

  return (
    <Figure
      id="disaggregation-figure-conditional"
      width="narrow"
      title="Conditional disaggregation: где делать prefill"
      controls={
        <>
          <Slider className="col-span-full" label="Длина промпта (ISL)" value={isl} min={256} max={32768} scale="log" step={1} onChange={setIsl} format={(v) => fmt.tokens(v)} />
          <Slider
            className="col-span-full"
            label="Уже в кэше decode-воркера"
            value={c}
            min={0}
            max={isl}
            step={Math.max(1, Math.round(isl / 200))}
            onChange={setCached}
            format={(v) => fmt.tokens(v)}
          />
          <Segmented<CondPolicy>
            className="col-span-full"
            label="Политика"
            size="sm"
            value={policy}
            onChange={setPolicy}
            options={[
              { value: 'isl_bounding', label: 'isl_bounding' },
              { value: 'prefill_load', label: 'prefill_load' },
              { value: 'isl_or_load', label: 'isl_or_load' },
            ]}
          />
          <Toggle className="col-span-full" label="Prefill-воркер перегружен" checked={busy} onChange={setBusy} hint="выше prefill_busy_threshold" />
          <div className="col-span-full grid gap-1.5">
            <div className="text-sm font-medium text-ink">Примеры из разбора</div>
            <div className="flex flex-wrap gap-1.5">
              {COND_EXAMPLES.map((e) => {
                const on = isl === e.isl && c === e.cached
                return (
                  <button
                    key={e.label}
                    type="button"
                    aria-pressed={on}
                    onClick={() => {
                      setIsl(e.isl)
                      setCached(e.cached)
                    }}
                    className={cn('h-8 rounded-control px-2.5 text-sm transition-colors', on ? 'bg-accent-soft text-accent-hover' : 'bg-surface-2 text-ink-2 hover:bg-surface-3')}
                  >
                    {e.label}
                  </button>
                )
              })}
            </div>
          </div>
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> выберите «Короткий без кэша»: промпт всего {fmt.tokens(1800)}, но по умолчанию он
          всё равно уходит на prefill-воркер. Локально его оставит только политика с учётом нагрузки, когда prefill-пул перегружен. Пороги{' '}
          {fmt.int(EFF_ISL_THRESHOLD)} и {fmt.num(EFF_ISL_RATIO_THRESHOLD, 1)} это значения по умолчанию в Dynamo.
        </>
      }
    >
      <div className="grid gap-3">
        <div className="grid gap-2 text-[15px]">
          <Check on={usesIsl} ok={d.islOk}>
            Досчитать осталось <b className="tabular-nums">{fmt.tokens(d.effIsl)}</b>, меньше {fmt.int(EFF_ISL_THRESHOLD)}
          </Check>
          <Check on={usesIsl} ok={d.ratioOk}>
            Это <b className="tabular-nums">{fmt.pct(d.ratio)}</b> промпта, меньше {fmt.pct(EFF_ISL_RATIO_THRESHOLD)}
          </Check>
          <Check on={usesLoad} ok={d.loadOk}>
            Выбранный prefill-воркер перегружен
          </Check>
        </div>
        <div className="grid gap-3 rounded-panel px-4 py-3 text-[15px]" style={{ background: mix(d.local ? 'decode' : 'prefill', 14) }} aria-live="polite">
          <div className="flex items-start gap-3">
            {d.local ? (
              <MapPinIcon size={20} aria-hidden className="mt-0.5 shrink-0 text-ink" />
            ) : (
              <ArrowRightIcon size={20} aria-hidden className="mt-0.5 shrink-0 text-ink" />
            )}
            <span className="text-ink">
              {d.local ? (
                d.effIsl === 0 ? (
                  <>
                    <b>Prefill на месте:</b> весь промпт уже в кэше decode-воркера, досчитать остаётся только последний токен, KV никуда не едет.
                  </>
                ) : (
                  <>
                    <b>Prefill на месте:</b> decode-воркер досчитает {fmt.tokens(d.effIsl)} сам, KV никуда не едет.
                  </>
                )
              ) : (
                <>
                  <b>Удалённый prefill:</b> запрос идёт на prefill-воркер, KV потом переедет на decode-воркер.
                </>
              )}
            </span>
          </div>
          <Route local={d.local} />
        </div>
      </div>
    </Figure>
  )
}

/** Мини-схема: кто считает prefill и едет ли KV. */
function Route({ local }: { local: boolean }) {
  const box = (role: 'prefill' | 'decode', label: string, active: boolean, note: string) => (
    <span
      className={cn('grid min-w-0 gap-0.5 rounded-control px-2.5 py-1.5 text-sm', active ? 'text-ink shadow-[inset_0_0_0_1.5px_var(--ink-2)]' : 'text-ink-2 shadow-[inset_0_0_0_1px_var(--line-strong)]')}
      style={{ background: active ? 'var(--surface)' : 'transparent' }}
    >
      <span className="inline-flex items-center gap-1.5 font-medium">
        <span aria-hidden className="size-2.5 shrink-0 rounded-[3px]" style={{ background: `var(--c-${role})`, opacity: active ? 1 : 0.45 }} />
        {label}
      </span>
      <span className="text-xs">{note}</span>
    </span>
  )
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-2" aria-hidden>
      {box('prefill', 'Prefill-воркер', !local, local ? 'не участвует' : 'считает промпт')}
      <span className={cn('grid justify-items-center text-xs', local ? 'text-ink-2' : 'text-ink')}>
        <span>KV</span>
        <ArrowRightIcon size={16} className={local ? 'opacity-40' : ''} />
      </span>
      {box('decode', 'Decode-воркер', true, local ? 'досчитывает сам и генерирует' : 'получает KV и генерирует')}
    </div>
  )
}

function Check({ on, ok, children }: { on: boolean; ok: boolean; children: ReactNode }) {
  return (
    <div className="flex items-start gap-2.5">
      {!on ? (
        <MinusCircleIcon size={18} aria-hidden className="mt-[3px] shrink-0 text-ink-3" />
      ) : ok ? (
        <CheckCircleIcon size={18} weight="fill" aria-hidden className="mt-[3px] shrink-0 text-good" />
      ) : (
        <XCircleIcon size={18} weight="fill" aria-hidden className="mt-[3px] shrink-0 text-bad" />
      )}
      <span className={on ? 'text-ink' : 'text-ink-3'}>
        {on && <span className="sr-only">{ok ? 'да: ' : 'нет: '}</span>}
        {children}
        {!on && <span className="ml-1 text-sm">(эта политика не проверяет)</span>}
        {on && (
          <span aria-hidden className="ml-2 text-sm font-medium text-ink-2">
            {ok ? 'да' : 'нет'}
          </span>
        )}
      </span>
    </div>
  )
}
