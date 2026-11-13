import { useState } from 'react'
import { ChapterLink, CodeBlock, Figure, Segmented, cn, useElementSize } from '@/ui'
import { Diagram, Swatch } from './diagram'
import { ARCH_V_BREAK, archLayout, detailKey } from './archLayout'
import { ARCH } from './archData'
import { useFocusWithin } from './useFocusWithin'

const FIGURE_ID = 'dynamo-figure-architecture'

type Mode = 'disagg' | 'agg'

/** Фигура 1 главы 19: интерактивная архитектура Dynamo, клик по компоненту раскрывает роль и конфиг. */
export function ArchitectureFigure() {
  const [mode, setMode] = useState<Mode>('disagg')
  const [selected, setSelected] = useState<string>('router')
  // на узком экране схема встаёт вертикально: широкая показывала бы только клиента и frontend
  const [boxRef, box] = useElementSize<HTMLDivElement>()
  const narrow = box.width > 0 && box.width < ARCH_V_BREAK
  const layout = archLayout(mode, narrow)
  const nodes = layout.nodes
  const inside = useFocusWithin(FIGURE_ID)
  // при смене режима выбранный узел может исчезнуть (prefill ↔ workers): переводим на аналог
  const key = detailKey(selected)
  const shownKey = mode === 'agg' && (key === 'prefill' || key === 'decode' || key === 'nixl') ? 'workers' : mode === 'disagg' && key === 'workers' ? 'prefill' : key
  const item = ARCH[shownKey] ?? ARCH.router
  const selNode = nodes.some((n) => n.id === selected) ? selected : (nodes.find((n) => n.variant !== 'group' && detailKey(n.id) === shownKey)?.id ?? null)

  return (
    <Figure
      id={FIGURE_ID}
      title="Из чего состоит Dynamo"
      width="full"
      actions={
        <Segmented
          aria-label="Режим"
          size="sm"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'disagg', label: 'Дезагрегированный' },
            { value: 'agg', label: 'Агрегированный' },
          ]}
        />
      }
      caption="Попробуйте: нажмите на любой блок схемы, чтобы увидеть его роль и то, как он выглядит в конфиге. Переключите режим: в агрегированном нет NIXL и отдельных пулов prefill и decode."
    >
      <div ref={boxRef} className="grid min-w-0 gap-5">
        <div className="flex flex-wrap gap-x-5 gap-y-1">
          {mode === 'disagg' ? (
            <>
              <Swatch role="prefill" label="prefill-воркеры" />
              <Swatch role="decode" label="decode-воркеры" />
            </>
          ) : (
            <span className="text-[13px] text-ink-2">каждый воркер делает и prefill, и decode</span>
          )}
          <span className="text-[13px] text-ink-2">сплошная стрелка: запрос и ответ; пунктир: события и управление</span>
        </div>
        <Diagram
          width={layout.width}
          height={layout.height}
          nodes={nodes}
          edges={layout.edges}
          selected={selNode}
          onSelect={setSelected}
          minWidth={layout.minWidth}
          className={cn(narrow && 'mx-auto w-full max-w-[420px]')}
          aria-label="Архитектура Dynamo: клиент, frontend с preprocessor, роутером и postprocessor, пулы воркеров, NIXL, event plane и Planner"
        />
        <div className="grid min-w-0 gap-5 border-t border-line pt-5 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="grid min-w-0 content-start gap-3">
            <div aria-live={inside ? 'polite' : 'off'}>
              <div className="text-[15px] font-semibold text-ink">{item.title}</div>
              <div className="text-[13px] text-ink-3">{item.where}</div>
            </div>
            <p className="text-[15px] leading-relaxed text-ink [&_code]:whitespace-nowrap">{item.what}</p>
            <ul className="grid list-disc gap-1.5 pl-5 text-[14px] leading-snug text-ink-2 marker:text-ink-3 [&_code]:whitespace-nowrap">
              {item.facts.map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
            {item.links && (
              <p className="text-[14px] text-ink-2">
                Подробнее:{' '}
                {item.links.map((l, i) => (
                  <span key={l}>
                    {i > 0 && ', '}
                    <ChapterLink to={l} />
                  </span>
                ))}
              </p>
            )}
          </div>
          <div className="min-w-0">
            <CodeBlock lang={item.config.lang} title={item.config.title} code={item.config.code} wrap />
          </div>
        </div>
      </div>
    </Figure>
  )
}
