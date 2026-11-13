import { useState, type ReactNode } from 'react'
import { ArrowRightIcon } from '@phosphor-icons/react'
import { CodeBlock, Figure, Tabs } from '@/ui'
import { ENGINES, type EngineCard } from './engineData'

/** Подписи вкладок: на узкой фигуре короткие, чтобы все четыре помещались без прокрутки. */
const TAB_LABEL: Record<EngineCard['id'], ReactNode> = {
  vllm: 'vLLM',
  sglang: 'SGLang',
  trtllm: (
    <>
      <span className="@md:hidden">TRT-LLM</span>
      <span className="hidden @md:inline">TensorRT-LLM</span>
    </>
  ),
  local: (
    <>
      <span className="@md:hidden">llama.cpp</span>
      <span className="hidden @md:inline">llama.cpp и Ollama</span>
    </>
  ),
}

/** Фигура 3 главы 18: vLLM, SGLang, TensorRT-LLM и локальные движки, вкладками. */
export function CompareFigure() {
  const [tab, setTab] = useState<EngineCard['id']>('vllm')
  return (
    <Figure
      id="engines-figure-compare"
      title="Три серверных движка и локальный запуск"
      width="wide"
      caption="Попробуйте: переключайте вкладки и сравните нижние строки. vLLM режет KV на блоки по 16 токенов и при нехватке памяти вытесняет запрос, чтобы потом пересчитать его; SGLang хранит KV по одному токену и отзывает decode-запросы в очередь; TensorRT-LLM берёт блоки по 32 токена и не пускает запрос, пока KV не хватит до самого конца. Значения по умолчанию на ноябрь 2026, они меняются каждые несколько недель."
    >
      <Tabs
        aria-label="Движок"
        value={tab}
        onChange={setTab}
        wrap
        tabs={ENGINES.map((e) => ({ id: e.id, label: TAB_LABEL[e.id] }))}
        panelClassName="@3xl:min-h-[31rem]"
      >
        {(active) => <EnginePanel card={ENGINES.find((e) => e.id === active) ?? ENGINES[0]} />}
      </Tabs>
    </Figure>
  )
}

function EnginePanel({ card }: { card: EngineCard }) {
  return (
    <div className="grid min-w-0 gap-5">
      <div className="grid gap-1">
        <span className="text-[13px] text-ink-3">{card.origin}</span>
        <p className="text-[15.5px] leading-relaxed text-ink">{card.idea}</p>
      </div>
      {card.book && (
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5 text-[13px]">
          <span className="text-ink-3">Оценка книги:</span>
          {card.book.map(([k, v]) => (
            <span key={k} className="rounded-chip bg-surface-2 px-2 py-0.5 text-ink-2">
              {k} <span className="font-medium text-ink">{v}</span>
            </span>
          ))}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-1.5" aria-label="Процессы">
        {card.procs.map((pr, i) => (
          <span key={pr} className="inline-flex items-center gap-1.5">
            {i > 0 && <ArrowRightIcon size={14} aria-hidden className="text-ink-3" />}
            <span className="rounded-control border border-line-strong bg-surface-2 px-2.5 py-1 text-[13px] text-ink">{pr}</span>
          </span>
        ))}
      </div>
      <div className="grid min-w-0 gap-5 @3xl:grid-cols-2">
        <Section title="Сильные стороны" items={card.strengths} />
        <Section title="Когда выбирать" items={card.choose} />
      </div>
      <div className="grid min-w-0 gap-5 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] content-start gap-x-4 gap-y-1.5 text-[14px]">
          {card.defaults.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-ink-3">{k[0].toUpperCase() + k.slice(1)}</dt>
              <dd className="text-ink">{v}</dd>
            </div>
          ))}
        </dl>
        <CodeBlock lang={card.run.lang} title="запуск" code={card.run.code} wrap />
      </div>
    </div>
  )
}

function Section({ title, items }: { title: string; items: ReactNode[] }) {
  return (
    <div className="grid content-start gap-2">
      <div className="text-[14px] font-semibold text-ink">{title}</div>
      <ul className="grid list-disc gap-1.5 pl-5 text-[14.5px] leading-snug text-ink-2 marker:text-ink-3">
        {items.map((it, i) => (
          <li key={i}>{it}</li>
        ))}
      </ul>
    </div>
  )
}
