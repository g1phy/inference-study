import { Fragment } from 'react'
import { ArrowRightIcon } from '@phosphor-icons/react'
import { Figure, cn } from '@/ui'

interface Box {
  title: string
  body: string
  /** Здесь работает детокенизатор. */
  detok?: boolean
  gpu?: boolean
}

interface Row {
  engine: string
  boxes: Box[]
  links: string[]
}

/** Где живёт детокенизация (notes-engines §1.0, notes-tokens §1.10, notes-dynamo §1.11; SPEC §8.11). */
const ROWS: Row[] = [
  {
    engine: 'vLLM',
    boxes: [
      { title: 'GPU-воркеры', body: 'forward и сэмплинг', gpu: true },
      { title: 'EngineCore', body: 'планировщик; стоп по EOS, stop_token_ids и max_tokens' },
      { title: 'API-сервер', body: 'детокенизатор, стоп-строки, парсеры, SSE', detok: true },
    ],
    links: ['', 'ZMQ'],
  },
  {
    engine: 'SGLang',
    boxes: [
      { title: 'Scheduler', body: 'планировщик, forward, сэмплинг; стоп по EOS, стоп-строкам и max_tokens', gpu: true },
      // \u200b: место переноса в длинном имени класса, иначе на узкой схеме слово вылезает из рамки
      { title: 'Detokenizer\u200bManager', body: 'отдельный процесс: id → текст', detok: true },
      { title: 'Tokenizer\u200bManager', body: 'HTTP, парсеры, SSE' },
    ],
    links: ['ZMQ', 'ZMQ'],
  },
  {
    engine: 'Dynamo',
    boxes: [
      { title: 'Воркер', body: 'vLLM, SGLang или TRT-LLM: возвращает id токенов', gpu: true },
      { title: 'Frontend: детокенизатор', body: 'Rust, оператор Backend: DecodeStream и jail для стоп-строк', detok: true },
      { title: 'Frontend: обратный путь', body: 'Preprocessor на обратном пути: парсеры, чанки SSE, usage, метрики TTFT и ITL' },
    ],
    links: ['сеть', ''],
  },
]

export function WhereDiagram() {
  return (
    <Figure
      id="detokenization-figure-where"
      title="Где работает детокенизатор"
      width="wide"
      caption="Во всех трёх системах детокенизация вынесена из GPU-цикла в отдельный процесс или сервис на CPU. Рамкой отмечено место, где id превращаются в текст."
    >
      <div className="grid gap-4" role="img" aria-label="Схема: в vLLM детокенизация идёт в процессе API-сервера, в SGLang в отдельном процессе DetokenizerManager, в Dynamo во frontend на Rust">
        {ROWS.map((row) => (
          <div key={row.engine} className="grid gap-2 @lg:grid-cols-[88px_minmax(0,1fr)] @lg:items-center">
            <div className="text-[15px] font-semibold text-ink">{row.engine}</div>
            <div className="grid gap-1.5 @md:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)_auto_minmax(0,1fr)] @md:items-stretch">
              {row.boxes.map((b, i) => (
                <Fragment key={i}>
                  <div
                    className={cn(
                      'grid min-w-0 grid-cols-[minmax(0,1fr)] content-start gap-0.5 rounded-control px-3 py-2',
                      b.detok ? 'bg-surface shadow-[inset_0_0_0_2px_var(--ink-2)]' : 'bg-surface-2 shadow-[inset_0_0_0_1px_var(--line)]',
                    )}
                  >
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                      <span className="rounded-chip bg-surface-3 px-1.5 text-[11px] font-medium text-ink-2">{b.gpu ? 'GPU' : 'CPU'}</span>
                      <span className="text-[14px] font-semibold text-ink">{b.title}</span>
                    </div>
                    <div className="text-[13px] leading-snug text-ink-2">{b.body}</div>
                  </div>
                  {i < row.boxes.length - 1 && (
                    <div className="flex items-center justify-center gap-1 py-0.5 text-[11px] text-ink-3 @md:flex-col @md:px-1">
                      <ArrowRightIcon size={16} aria-hidden className="rotate-90 @md:rotate-0" />
                      {row.links[i] && <span className="whitespace-nowrap">{row.links[i]}</span>}
                    </div>
                  )}
                </Fragment>
              ))}
            </div>
          </div>
        ))}
      </div>
    </Figure>
  )
}
