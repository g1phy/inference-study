import { useMemo, useRef, useState, type FocusEvent, type PointerEvent } from 'react'
import { ArrowCounterClockwiseIcon, CheckIcon, PaperPlaneRightIcon } from '@phosphor-icons/react'
import { useSettings } from '@/state/settings'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import type { KvRouteDecision, KvWorkerScore } from '@/engine/router'
import { Badge, Button, Figure, Formula, Segmented, Skeleton, Slider, cn, cssVar, fmt, useTip } from '@/ui'
import { readerParts } from './prompt'
import {
  BLOCK,
  COST_TEX,
  PROB_TEX,
  agentSwitchAt,
  commit,
  costTexFor,
  decide,
  initialWorkers,
  nextRequest,
  readerRequest,
  summarize,
  type DemoWorker,
  type RouterKnobs,
  type RouterRequest,
  type Scenario,
  type SentRecord,
  type Strategy,
} from './routerModel'

interface Session {
  key: string
  workers: DemoWorker[]
  req: RouterRequest
  sent: SentRecord[]
  /** Воркер, для которого раскрыта формула и работает слайдер нагрузки (null: выбранный роутером). */
  selected: string | null
}

const SCENARIOS: { value: Scenario; label: string }[] = [
  { value: 'agent', label: 'Агент с tools' },
  { value: 'plain', label: 'Ваш запрос' },
  { value: 'docs', label: 'Из доков' },
]

const STRATEGIES: { value: Strategy; label: string }[] = [
  { value: 'kv', label: 'KV-aware' },
  { value: 'round-robin', label: 'Round-robin' },
  { value: 'random', label: 'Случайно' },
]

const BLOCK_FORMS = ['блок', 'блока', 'блоков'] as const

/**
 * Главная фигура главы 20: запрос читателя (настоящие токены → блоки по 16 → хэши), четыре воркера со
 * своими кэшами и нагрузкой, стоимость по формуле Dynamo ≥ 1.2, вероятности при температуре, выбор.
 * «Отправить запрос» применяет решение (applyRoute) и приводит следующий запрос потока.
 */
export function RouterFigure() {
  const [settings] = useSettings()
  const { tokenizer } = useTokenizer()
  const [scenario, setScenario] = useState<Scenario>('agent')
  const [strategy, setStrategy] = useState<Strategy>('kv')
  const [credit, setCredit] = useState(1)
  const [temp, setTemp] = useState(0)
  const [nonce, setNonce] = useState(0)

  const base = useMemo(
    () => readerParts(settings),
    [settings.systemText, settings.userText, settings.thinking, settings.tools],
  )
  const key = `${scenario}|${nonce}|${base.systemText}|${base.userText}|${base.thinking}|${base.tools ? 1 : 0}|${tokenizer ? 1 : 0}`
  const fresh = useMemo(
    () => (tokenizer ? { workers: initialWorkers(scenario, base, tokenizer), req: readerRequest(scenario, base, tokenizer) } : null),
    [tokenizer, scenario, base],
  )
  // подпись фигуры называет номер запроса, на котором агент уходит на Воркер 4: он зависит от длины сообщения читателя
  const switchAt = useMemo(() => (tokenizer ? agentSwitchAt(base, tokenizer) : 0), [tokenizer, base])
  const [session, setSession] = useState<Session | null>(null)
  const cur: Session | null =
    session && session.key === key ? session : fresh ? { key, workers: fresh.workers, req: fresh.req, sent: [], selected: null } : null

  const knobs = useMemo<RouterKnobs>(() => ({ overlapCredit: credit, temperature: temp }), [credit, temp])
  const req = cur?.req
  const workers = cur?.workers
  const decision = useMemo(() => (req && workers ? decide(strategy, req, workers, knobs) : null), [req, workers, strategy, knobs])

  const selectedId = cur?.selected ?? decision?.chosen ?? null
  const selectedWorker = cur?.workers.find((w) => w.id === selectedId) ?? null

  const sendMany = (count: number) => {
    if (!cur || !tokenizer || !decision) return
    let s: Session = cur
    let d: KvRouteDecision = decision
    for (let i = 0; i < count; i++) {
      if (i > 0) d = decide(strategy, s.req, s.workers, knobs)
      const score = d.scores.find((x) => x.workerId === d.chosen) as KvWorkerScore
      const record: SentRecord = {
        n: s.req.n,
        label: s.req.label,
        worker: d.chosen,
        workerName: score.name,
        isl: s.req.isl,
        cachedTokens: score.cachedTokens,
        strategy,
      }
      s = {
        ...s,
        workers: commit(s.workers, d, s.req, scenario === 'agent' ? 'был пустым, теперь держит префикс агента' : 'был пустым, теперь держит этот промпт'),
        req: nextRequest(scenario, s.req.n + 1, base, tokenizer),
        sent: [...s.sent, record],
        selected: null,
      }
    }
    setSession(s)
  }

  const setLoad = (v: number) => {
    if (!cur || !selectedId) return
    setSession({ ...cur, selected: selectedId, workers: cur.workers.map((w) => (w.id === selectedId ? { ...w, activeBlocks: v } : w)) })
  }
  const select = (id: string) => cur && setSession({ ...cur, selected: id })
  const reset = () => {
    setSession(null)
    setNonce((n) => n + 1)
  }

  const sliders = (
    <div className="grid min-w-0 grid-cols-[repeat(auto-fit,minmax(min(200px,100%),1fr))] gap-x-6 gap-y-4 rounded-panel border border-line px-4 py-4">
      <Slider
        label="Кредит за кэш, c"
        value={credit}
        min={0}
        max={2}
        step={0.05}
        onChange={setCredit}
        format={(v) => fmt.num(v, 2)}
        hint="overlap_score_credit, по умолчанию 1"
        disabled={strategy !== 'kv'}
      />
      <Slider
        label="Температура, T"
        value={temp}
        min={0}
        max={2}
        step={0.05}
        onChange={setTemp}
        format={(v) => fmt.num(v, 2)}
        hint="router_temperature, по умолчанию 0"
        disabled={strategy !== 'kv'}
      />
      <Slider
        label={`Нагрузка: ${selectedWorker?.name ?? 'воркер'}`}
        value={selectedWorker?.activeBlocks ?? 0}
        min={0}
        max={160}
        step={1}
        onChange={setLoad}
        format={(v) => `${fmt.int(v)} бл.`}
        hint="активные KV-блоки; воркер выбирается кликом по карточке"
        disabled={!selectedWorker}
      />
    </div>
  )

  return (
    <Figure
      id="routing-figure-router"
      title="Куда отправить ваш запрос"
      width="full"
      caption={`Попробуйте: в сценарии «Агент с tools» нажмите «Отправить запрос» несколько раз подряд. Тёплый Воркер 1 копит активные блоки, и ${switchAt > 1 ? `на запросе №${switchAt}` : 'через несколько запросов'} роутер уходит на пустой Воркер 4, который дальше сам становится тёплым. Потом сбросьте, выберите стратегию round-robin, нажмите «Ещё 5» и сравните долю токенов из кэша в журнале отправленных запросов.`}
    >
      {!cur || !decision || !tokenizer ? (
        <RouterSkeleton />
      ) : (
        // На узкой фигуре стоимость и журнал поднимаются сразу под запрос, за ними ползунки: результат клика виден без прокрутки
        // на экран вниз. Порядок в DOM как на десктопе: карточки, стоимость, формула, ползунки, журнал.
        <div className="grid min-w-0 gap-6 @4xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
          <div className="order-1 flex flex-wrap items-end gap-x-6 gap-y-3 @4xl:col-span-2">
            <Segmented label="Сценарий" options={SCENARIOS} value={scenario} onChange={setScenario} size="sm" />
            <Segmented label="Стратегия роутера" options={STRATEGIES} value={strategy} onChange={setStrategy} size="sm" />
          </div>
          <div className="order-2 min-w-0 @4xl:col-span-2">
            <RequestPanel
              req={cur.req}
              userText={base.userText}
              tokenizer={tokenizer}
              scenario={scenario}
              onSend={() => sendMany(1)}
              onSendFive={() => sendMany(5)}
              onReset={reset}
            />
          </div>
          <div className="order-6 grid min-w-0 grid-cols-2 gap-2 @md:order-3 @md:gap-3 @4xl:col-span-2 @5xl:grid-cols-4">
            {decision.scores.map((s, i) => (
              <WorkerCard
                key={s.workerId}
                worker={cur.workers[i]}
                score={s}
                req={cur.req}
                chosen={s.workerId === decision.chosen}
                selected={s.workerId === selectedId}
                onSelect={() => select(s.workerId)}
              />
            ))}
          </div>
          <div className="order-3 min-w-0 @md:order-4">
            <CostChart decision={decision} strategy={strategy} selectedId={selectedId} onSelect={select} />
          </div>
          <div className="order-7 min-w-0 @md:order-6 @4xl:order-5">
            <FormulaPanel decision={decision} selectedId={selectedId} knobs={knobs} strategy={strategy} />
          </div>
          <div className="order-5 min-w-0 @4xl:order-6 @4xl:col-span-2">{sliders}</div>
          <div className="order-4 min-w-0 @md:order-7 @4xl:col-span-2">
            <SentLog sent={cur.sent} />
          </div>
        </div>
      )}
    </Figure>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Запрос: блоки по 16 и хэши
 * ------------------------------------------------------------------------------------------- */

function RequestPanel({
  req,
  userText,
  tokenizer,
  scenario,
  onSend,
  onSendFive,
  onReset,
}: {
  req: RouterRequest
  /** Сообщение читателя из симулятора: показываем его текст у «вашего» запроса. */
  userText: string
  tokenizer: { decode(ids: number[]): string }
  scenario: Scenario
  onSend: () => void
  onSendFive: () => void
  onReset: () => void
}) {
  const scope = useRef<HTMLDivElement>(null)
  const tip = useTip({ scopeEl: scope.current })
  const show = (el: Element, i: number) => {
    const b = req.blocks[i]
    const text = req.who === 'docs' ? `синтетические токены ${b.tokenIds[0]}…${b.tokenIds[b.tokenIds.length - 1]}` : tokenizer.decode(b.tokenIds)
    tip.show(
      el.getBoundingClientRect(),
      <div className="grid gap-1">
        <div className="font-medium text-ink">
          Блок {i}
          {b.full ? (
            <>
              {' '}
              · хэш <span className="font-mono">{b.hash}</span>
            </>
          ) : (
            ' · хвост, не хэшируется'
          )}
        </div>
        <div className="font-mono text-[12px] break-all whitespace-pre-wrap text-ink-2">{visible(text)}</div>
        <div className="text-ink-3">
          токены {b.start}…{b.start + b.tokenIds.length - 1}
        </div>
      </div>,
    )
  }
  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    const el = (e.target as HTMLElement).closest('[data-block]')
    if (el) show(el, Number(el.getAttribute('data-block')))
  }
  const onFocus = (e: FocusEvent<HTMLDivElement>) => {
    const el = (e.target as HTMLElement).closest('[data-block]')
    if (el) show(el, Number(el.getAttribute('data-block')))
  }

  return (
    <div className="grid min-w-0 gap-3 rounded-panel bg-surface-2 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="grid min-w-0 gap-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[15px] font-semibold text-ink">Запрос №{req.n + 1}</span>
            <Badge tone={req.who === 'reader' ? 'accent' : 'neutral'}>{whoLabel(req.who)}</Badge>
          </div>
          <div className={cn('max-w-[72ch] text-sm text-ink-2', req.who !== 'docs' && 'truncate')}>
            {req.who === 'docs'
              ? '160 синтетических токенов, как в примере документации Dynamo. Отправка здесь выключена: меняйте c, температуру и нагрузку W1–W3.'
              : `«${req.who === 'reader' ? userText.trim() || 'ваше сообщение из симулятора' : req.label}»`}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="primary"
            size="sm"
            icon={PaperPlaneRightIcon}
            onClick={onSend}
            disabled={scenario === 'docs'}
            title={scenario === 'docs' ? 'В примере из документации отправка выключена: меняйте c, температуру и нагрузку' : undefined}
          >
            Отправить запрос
          </Button>
          <Button
            size="sm"
            onClick={onSendFive}
            disabled={scenario === 'docs'}
            title={scenario === 'docs' ? 'В примере из документации отправка выключена: меняйте c, температуру и нагрузку' : undefined}
          >
            Ещё 5
          </Button>
          <Button size="sm" variant="ghost" icon={ArrowCounterClockwiseIcon} onClick={onReset}>
            Сбросить
          </Button>
        </div>
      </div>
      <div
        ref={scope}
        role="list"
        aria-label={`Блоки запроса: ${fmt.count(req.fullBlocks, ['полный', 'полных', 'полных'])} по ${BLOCK} токенов${req.tailTokens > 0 ? ` и хвост из ${fmt.tokensGen(req.tailTokens)}` : ''}`}
        className="flex flex-wrap gap-[3px]"
        onPointerMove={onMove}
        onPointerDown={onMove}
        onPointerLeave={(e) => e.pointerType === 'mouse' && tip.hide()}
        onFocus={onFocus}
        onBlur={tip.hide}
      >
        {req.blocks.map((b, i) => (
          <span
            key={i}
            role="listitem"
            data-block={i}
            tabIndex={i === 0 ? 0 : -1}
            onKeyDown={(e) => {
              const next = e.key === 'ArrowRight' ? i + 1 : e.key === 'ArrowLeft' ? i - 1 : -1
              if (next >= 0 && next < req.blocks.length) {
                e.preventDefault()
                ;(e.currentTarget.parentElement?.children[next] as HTMLElement | undefined)?.focus()
              }
            }}
            aria-label={b.full ? `блок ${i}, хэш ${b.hash}` : `хвост из ${fmt.tokensGen(b.tokenIds.length)}, не хэшируется`}
            className={cn('h-3.5 w-3.5 rounded-[3px] outline-offset-1', b.full ? '' : 'border border-dashed border-ink-3')}
            style={b.full ? { background: cssVar('prefill') } : undefined}
          />
        ))}
      </div>
      <div className="text-sm text-ink-2">
        {fmt.tokens(req.isl)} = {fmt.count(req.blocks.length, BLOCK_FORMS)}
        {req.tailTokens > 0 ? (
          <>
            : {fmt.count(req.fullBlocks, ['полный', 'полных', 'полных'])} по {BLOCK} токенов и неполный хвост из {fmt.tokensGen(req.tailTokens)}. Хвост
            не хэшируют, он всегда считается заново
          </>
        ) : (
          <> по {BLOCK} токенов</>
        )}
        . Хэш каждого квадрата зависит от его токенов и от хэша предыдущего, поэтому совпадение возможно только с начала.
      </div>
      {tip.node}
    </div>
  )
}

function whoLabel(who: RouterRequest['who']): string {
  return who === 'reader' ? 'ваш' : who === 'docs' ? 'синтетический' : 'другой пользователь'
}

/** Пробелы и переводы строк видимыми знаками, как в TokenStrip. */
function visible(s: string): string {
  return s.replace(/ /g, '·').replace(/\n/g, '↵\n')
}

/* ---------------------------------------------------------------------------------------------
 * Воркер: какие блоки запроса уже лежат у него, нагрузка
 * ------------------------------------------------------------------------------------------- */

function WorkerCard({
  worker,
  score,
  req,
  chosen,
  selected,
  onSelect,
}: {
  worker: DemoWorker
  score: KvWorkerScore
  req: RouterRequest
  chosen: boolean
  selected: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={cn(
        'grid min-w-0 content-start gap-2 rounded-panel border bg-surface p-2.5 text-left transition-colors @md:gap-2.5 @md:p-3.5',
        selected ? 'border-accent' : 'border-line hover:border-line-strong',
      )}
      style={selected ? { boxShadow: 'inset 0 0 0 1px var(--accent)' } : undefined}
    >
      <div className="flex flex-wrap items-start justify-between gap-x-2 gap-y-1">
        <div className="grid min-w-0 gap-0.5">
          <span className="text-[14px] font-semibold text-ink @md:text-[15px]">{worker.name}</span>
          <span className="line-clamp-2 text-[12px] leading-snug text-ink-3 @md:line-clamp-none @md:text-[13px]">{worker.role}</span>
        </div>
        {chosen && (
          <Badge tone="accent" icon={CheckIcon}>
            выбран
          </Badge>
        )}
      </div>
      <div aria-hidden className="flex flex-wrap gap-[2px]">
        {req.blocks.map((b, i) => {
          if (!b.full) return <span key={i} className="h-2.5 w-2.5 rounded-[2px] border border-dashed border-ink-3" />
          const hit = i < score.overlapBlocks
          return (
            <span
              key={i}
              className={cn('h-2.5 w-2.5 rounded-[2px]', hit ? '' : 'border border-ink-3')}
              style={hit ? { background: cssVar('prefill') } : undefined}
            />
          )
        })}
      </div>
      {/* узкая карточка: два числа; широкая: фраза и полная табличка */}
      <div className="grid gap-0.5 text-[12.5px] leading-snug text-ink-2 tabular-nums @md:hidden">
        <span>
          в кэше{' '}
          <span className="font-medium text-ink">
            {score.overlapBlocks} из {req.blocks.length}
          </span>
        </span>
        <span>
          активных блоков <span className="font-medium text-ink">{fmt.int(worker.activeBlocks)}</span>
        </span>
      </div>
      <div className="hidden text-[13px] leading-snug text-ink-2 @md:block">
        {score.overlapBlocks > 0 ? (
          <>
            <span className="font-medium text-ink">
              {score.overlapBlocks} из {req.blocks.length}
            </span>{' '}
            блоков запроса уже в его GPU-кэше
          </>
        ) : (
          'Префикса запроса в кэше нет'
        )}
      </div>
      <dl className="hidden grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[13px] tabular-nums @md:grid">
        <dt className="text-ink-3">посчитать в prefill</dt>
        <dd className="text-right text-ink">{fmt.tokens(score.prefillTokens)}</dd>
        <dt className="text-ink-3">активные блоки</dt>
        <dd className="text-right text-ink">{fmt.int(worker.activeBlocks)}</dd>
        <dt className="text-ink-3">очередь prefill</dt>
        <dd className="text-right text-ink">{fmt.tokens(worker.queuedPrefillTokens ?? 0)}</dd>
        <dt className="text-ink-3">всего в кэше</dt>
        <dd className="text-right text-ink">{fmt.count(worker.cachedBlocks.size, BLOCK_FORMS)}</dd>
      </dl>
    </button>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Стоимость: prefill + decode в одних единицах (блоки), общая шкала
 * ------------------------------------------------------------------------------------------- */

function CostChart({
  decision,
  strategy,
  selectedId,
  onSelect,
}: {
  decision: KvRouteDecision
  strategy: Strategy
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  const max = Math.max(1, ...decision.scores.map((s) => s.cost))
  return (
    <div className="grid min-w-0 content-start gap-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <div className="text-sm font-semibold text-ink">Стоимость по формуле Dynamo в блоках (меньше лучше) и вероятность выбора</div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-ink-2">
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-2.5 w-2.5 rounded-[3px]" style={{ background: cssVar('prefill') }} />
            prefill: что осталось посчитать
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-2.5 w-2.5 rounded-[3px]" style={{ background: cssVar('decode') }} />
            decode: нагрузка KV
          </span>
        </div>
      </div>
      <div role="list" className="grid gap-2">
        {decision.scores.map((s) => {
          const pre = s.prefillTerm ?? 0
          const dec = s.decodeTerm ?? 0
          const chosen = s.workerId === decision.chosen
          return (
            <button
              key={s.workerId}
              type="button"
              role="listitem"
              onClick={() => onSelect(s.workerId)}
              aria-label={`${s.name}: стоимость ${fmt.num(s.cost)} = prefill ${fmt.num(pre)} + decode ${fmt.num(dec)}${chosen ? ', выбран' : ''}`}
              className={cn(
                // узко: имя и число в строке, полоса на всю ширину под ними; от @md всё в одну строку
                'grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 rounded-control px-2 py-1.5 text-left transition-colors hover:bg-surface-2 @md:grid-cols-[5.5rem_minmax(0,1fr)_auto]',
                s.workerId === selectedId && 'bg-surface-2',
              )}
            >
              <span className={cn('flex items-center gap-1 truncate text-[13px]', chosen ? 'font-semibold text-ink' : 'text-ink-2')}>
                {chosen && <CheckIcon size={14} aria-hidden className="shrink-0 text-accent" />}
                {s.name}
              </span>
              <span className="order-3 col-span-2 flex h-4 min-w-0 items-stretch @md:order-none @md:col-span-1">
                <span className="rounded-l-[3px]" style={{ width: `${(pre / max) * 100}%`, background: cssVar('prefill') }} />
                <span
                  className={cn(pre <= 0 && 'rounded-l-[3px]', 'rounded-r-[3px]')}
                  style={{ width: `${(dec / max) * 100}%`, background: cssVar('decode') }}
                />
              </span>
              <span className="order-2 text-right text-[13px] whitespace-nowrap tabular-nums @md:order-none">
                <span className="font-semibold text-ink">{fmt.num(s.cost, 1)}</span>
                <span className="inline-block w-[3.25rem] text-ink-3">{fmt.pct(s.prob)}</span>
              </span>
            </button>
          )
        })}
      </div>
      <p className="text-[13px] leading-snug text-ink-3">
        {strategy === 'kv'
          ? decision.temperature > 0
            ? 'Справа стоимость и вероятность выбора: при T > 0 роутер бросает кубик, и дешёвые воркеры выпадают чаще.'
            : 'Справа вероятность выбора: при T = 0 роутер берёт самый дешёвый воркер наверняка. Обе части стоимости в одних единицах, KV-блоках, поэтому их можно складывать.'
          : 'Стоимость посчитана для сравнения: эта стратегия на неё не смотрит. Справа вероятность выбора.'}
      </p>
    </div>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Формула с подставленными числами и объяснение роутера
 * ------------------------------------------------------------------------------------------- */

function FormulaPanel({
  decision,
  selectedId,
  knobs,
  strategy,
}: {
  decision: KvRouteDecision
  selectedId: string | null
  knobs: RouterKnobs
  strategy: Strategy
}) {
  const s = decision.scores.find((x) => x.workerId === selectedId) ?? decision.scores[0]
  return (
    <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] content-start gap-2">
      <div className="text-sm font-semibold text-ink">Стоимость: {s.name}</div>
      <Formula block tex={COST_TEX} className="!mt-0 min-w-0" />
      <Formula block tex={costTexFor(s, knobs, decision.isl)} className="!mt-0 min-w-0" />
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[13px] text-ink-2">
        <span className="font-mono text-ink">P</span>
        <span>токены в очереди prefill воркера</span>
        <span className="font-mono text-ink">ISL, B</span>
        <span>длина запроса и размер блока ({BLOCK})</span>
        <span className="font-mono text-ink">c · overlap</span>
        <span>кредит за блоки, которые уже лежат в кэше</span>
        <span className="font-mono text-ink">D</span>
        <span>активные блоки воркера и новые блоки запроса, которых ещё не держат его запросы</span>
      </div>
      {strategy === 'kv' && knobs.temperature > 0 && <Formula block tex={PROB_TEX} className="!mt-1 min-w-0" />}
      <p className="mt-1 text-[14px] leading-normal text-ink">{decision.explanation}</p>
    </div>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Журнал отправленных запросов
 * ------------------------------------------------------------------------------------------- */

function SentLog({ sent }: { sent: SentRecord[] }) {
  if (sent.length === 0) {
    return <p className="text-[13px] text-ink-3">Запросы ещё не отправлялись: здесь появится, куда ушёл каждый и сколько токенов взято из кэша.</p>
  }
  const sum = summarize(sent)
  const last = sent.slice(-6).reverse()
  return (
    <div className="grid min-w-0 gap-3 border-t border-line pt-4">
      <div className="flex flex-wrap gap-x-8 gap-y-3">
        <SmallStat label="Отправлено" value={fmt.int(sum.count)} />
        <SmallStat label="Токенов промпта" value={fmt.int(sum.isl)} />
        <SmallStat label="Взято из кэша" value={fmt.int(sum.cached)} />
        <SmallStat label="Доля из кэша" value={fmt.pct(sum.hitRate)} />
      </div>
      <ol className="grid gap-1 text-[13px]">
        {last.map((r) => (
          <li key={r.n} className="grid min-w-0 grid-cols-[2.5rem_minmax(0,1fr)_auto] gap-3 tabular-nums">
            <span className="text-ink-3">№{r.n + 1}</span>
            <span className="truncate text-ink-2">
              {r.label} → <span className="text-ink">{r.workerName}</span>
              <span className="text-ink-3"> ({STRATEGIES.find((x) => x.value === r.strategy)?.label})</span>
            </span>
            <span className="text-right text-ink-2">из кэша {fmt.int(r.cachedTokens)} из {fmt.int(r.isl)}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}

function SmallStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid gap-0.5">
      <span className="text-[13px] text-ink-3">{label}</span>
      <span className="text-[17px] font-semibold text-ink tabular-nums">{value}</span>
    </div>
  )
}

function RouterSkeleton() {
  return (
    <div className="grid gap-6" aria-busy="true" aria-label="Загружается токенизатор">
      <Skeleton height={112} radius={12} />
      <div className="grid grid-cols-2 gap-3 @5xl:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} height={168} radius={12} />
        ))}
      </div>
      <Skeleton height={180} radius={12} />
    </div>
  )
}
