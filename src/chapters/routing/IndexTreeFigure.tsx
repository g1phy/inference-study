import { useMemo, type ReactNode } from 'react'
import { ArrowRightIcon } from '@phosphor-icons/react'
import { useSettings } from '@/state/settings'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { Figure, Skeleton, cn, cssVar, fmt, mix } from '@/ui'
import { OTHER_QUESTIONS, readerParts } from './prompt'
import { initialWorkers, readerRequest } from './routerModel'

const BLOCK_FORMS = ['блок', 'блока', 'блоков'] as const

/** Сколько первых хэшей двух цепочек совпадает подряд. */
function sharedPrefix(a: readonly string[], b: readonly string[]): number {
  let k = 0
  while (k < a.length && k < b.length && a[k] === b[k]) k++
  return k
}

/**
 * Индекс роутера на тех же данных, что главная фигура главы (сценарий «Агент с tools»): цепочки хэшей блоков,
 * которые воркеры сообщили в BlockStored, склеены в дерево префиксов, а запрос читателя спускается по нему.
 */
export function IndexTreeFigure() {
  const [settings] = useSettings()
  const { tokenizer } = useTokenizer()
  const data = useMemo(() => {
    if (!tokenizer) return null
    const base = readerParts(settings)
    const workers = initialWorkers('agent', base, tokenizer)
    const req = readerRequest('agent', base, tokenizer)
    const chain = (i: number) => [...workers[i].cachedBlocks]
    const a = chain(0)
    const b = chain(1)
    const foreign = chain(2)
    const mine = req.blocks.filter((x) => x.full).map((x) => x.hash)
    const shared = sharedPrefix(a, b)
    const depth = workers.map((w) => {
      let k = 0
      while (k < mine.length && w.cachedBlocks.has(mine[k])) k++
      return { name: w.name, depth: k }
    })
    const mineShared = Math.max(sharedPrefix(mine, a), sharedPrefix(mine, b))
    return { workers, shared, a, b, foreign, mine, mineShared, tail: req.tailTokens, depth, total: req.blocks.length }
  }, [tokenizer, settings.systemText, settings.userText, settings.thinking, settings.tools])

  return (
    <Figure
      id="routing-figure-index"
      title="Индекс роутера: дерево префиксов из KV events"
      width="wide"
      caption="Хэши настоящие: это те же цепочки блоков, что лежат у воркеров в сценарии «Агент с tools» фигуры ниже. Цепочки без ветвлений склеены в один узел дерева. Числа совпадения справа роутер подставит в функцию стоимости."
    >
      {!data ? (
        <Skeleton height={240} radius={12} label="Загружается токенизатор" />
      ) : (
        <div className="grid min-w-0 gap-5 @3xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
          <div className="grid min-w-0 content-start gap-3">
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px] text-ink-2">
              {data.workers.map((w) => (
                <WorkerChip key={w.id} name={w.name} />
              ))}
              <ArrowRightIcon size={14} aria-hidden className="text-ink-3" />
              <code className="font-mono text-[12.5px] text-ink">BlockStored</code>
              <ArrowRightIcon size={14} aria-hidden className="text-ink-3" />
              <span>индекс роутера</span>
            </div>
            <div className="grid min-w-0 gap-1 rounded-panel bg-surface-2 p-3 @md:p-4">
              <span className="text-[13px] text-ink-3">корень</span>
              <ul className="grid min-w-0 gap-2">
                <TreeNode
                  blocks={data.shared}
                  label="системный промпт и 4 инструмента агента"
                  who={[data.workers[0].name, data.workers[1].name]}
                  hit
                >
                  <ul className="grid min-w-0 gap-2">
                    <TreeNode blocks={data.a.length - data.shared} label={`«${OTHER_QUESTIONS[0]}»`} who={[data.workers[0].name]} />
                    <TreeNode blocks={data.b.length - data.shared} label={`«${OTHER_QUESTIONS[1]}»`} who={[data.workers[1].name]} />
                    <TreeNode
                      blocks={data.mine.length - data.mineShared}
                      label="ваш вопрос"
                      note={`в индексе нет${data.tail > 0 ? `, плюс хвост ${fmt.tokens(data.tail)}` : ''}`}
                      who={[]}
                      yours
                    />
                  </ul>
                </TreeNode>
                <TreeNode blocks={data.foreign.length} label="чужое приложение: переводчик" who={[data.workers[2].name]} />
              </ul>
            </div>
          </div>
          <div className="grid min-w-0 content-start gap-2">
            <div className="text-[14px] font-semibold text-ink">Ваш запрос спускается по дереву</div>
            <p className="text-[13.5px] leading-snug text-ink-2">
              Из {fmt.count(data.total, ['блока', 'блоков', 'блоков'])} запроса совпадение возможно только по общему началу. Глубина совпадения у каждого воркера:
            </p>
            <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1 text-[14px] tabular-nums">
              {data.depth.map((d) => (
                <div key={d.name} className="contents">
                  <dt className="text-ink-2">{d.name}</dt>
                  <dd className={cn('text-right', d.depth > 0 ? 'font-semibold text-ink' : 'text-ink-3')}>{fmt.count(d.depth, BLOCK_FORMS)}</dd>
                </div>
              ))}
            </dl>
            <p className="text-[13px] leading-snug text-ink-3">
              Индекс отвечает не «есть или нет», а сколько первых блоков подряд уже лежит у воркера.
            </p>
          </div>
        </div>
      )}
    </Figure>
  )
}

function WorkerChip({ name }: { name: string }) {
  return <span className="rounded-chip border border-line-strong bg-surface px-1.5 py-0.5 text-[12px] whitespace-nowrap text-ink">{name}</span>
}

/** Узел дерева: полоска блоков (до 12 квадратов), подпись, у каких воркеров лежит. */
function TreeNode({
  blocks,
  label,
  who,
  note,
  hit,
  yours,
  children,
}: {
  blocks: number
  label: string
  who: string[]
  note?: string
  hit?: boolean
  yours?: boolean
  children?: ReactNode
}) {
  const shown = Math.min(blocks, 12)
  return (
    <li className="relative grid min-w-0 gap-2 border-l border-line-strong pl-4">
      <div className="grid min-w-0 gap-1">
        <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
          <span aria-hidden className="flex flex-wrap gap-[2px]">
            {Array.from({ length: shown }, (_, i) => (
              <span
                key={i}
                className={cn('h-2.5 w-2.5 rounded-[2px]', yours ? 'border border-dashed border-ink-3' : '')}
                style={yours ? undefined : { background: hit ? cssVar('prefill') : mix('prefill', 40) }}
              />
            ))}
            {blocks > shown && <span className="text-[11px] leading-[10px] text-ink-3">…</span>}
          </span>
          <span className={cn('min-w-0 text-[14px]', yours ? 'text-ink-2 italic' : 'text-ink')}>{label}</span>
          <span className="text-[13px] text-ink-3 tabular-nums">{fmt.count(blocks, BLOCK_FORMS)}</span>
        </div>
        <div className="flex flex-wrap items-center gap-1.5 text-[12.5px] text-ink-3">
          {who.length > 0 ? (
            <>
              у кого: {who.map((n) => <WorkerChip key={n} name={n} />)}
            </>
          ) : (
            <span>{note}</span>
          )}
        </div>
      </div>
      {children}
    </li>
  )
}
