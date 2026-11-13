import { useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import {
  ArrowClockwiseIcon,
  ArrowCounterClockwiseIcon,
  CheckCircleIcon,
  CubeIcon,
  DatabaseIcon,
  HardDrivesIcon,
  HourglassMediumIcon,
  InfoIcon,
  LightningIcon,
  PaperPlaneRightIcon,
  PlusIcon,
  TrashIcon,
  WarningIcon,
  XCircleIcon,
} from '@phosphor-icons/react'
import { getInterconnect } from '@/engine/data/interconnects'
import { kvBytesForSeq, prefillMs, transferMs } from '@/engine/perf'
import { useSettings } from '@/state/settings'
import { Badge, Button, Figure, Segmented, Stat, Toggle, Tooltip, cn, cssVar, fmt, tint, useElementSize } from '@/ui'
import { gpuOr, minGpus, modelOr, shortGpuName } from '../kv-cache/lib'
import {
  CAP,
  MAX_NODES,
  PREFIXES,
  initialState,
  reduce,
  type Action,
  type CacheMode,
  type ClusterState,
  type NodeState,
  type PrefixId,
  type Source,
  type Timing,
  type Worker,
} from './cluster'
import { gpuHostLink } from './tiers'

const PREFIX_TOKENS = 6000
const TAIL_TOKENS = 400

const MODE_TEXT: Record<CacheMode, string> = {
  embedded: 'Каждый под vLLM держит кэш у себя: HBM и кусок RAM, выделенный поду. Ни с кем не делится и умирает вместе с подом.',
  daemonset:
    'DaemonSet ставит на каждую GPU-ноду ровно один под кэша. Он владеет RAM и NVMe ноды, а воркеры ходят к нему локально: управление через Service с internalTrafficPolicy: Local, данные через CUDA IPC и общую память.',
  cluster: 'Кэш вынесен в отдельный кэш-кластер (StatefulSet). Его видят все воркеры кластера, но каждый поход за KV идёт по сети.',
}

/** TTFT по источнику KV: загрузка и prefill хвоста перекрываются послойно, поэтому берём большее из двух. */
function computeTiming(modelId: string, gpuId: string, gpus: number, net: 'tcp' | 'rdma'): Timing & { kvBytes: number } {
  const m = modelOr(modelId)
  const g = gpuOr(gpuId)
  const miss = prefillMs(m, g, PREFIX_TOKENS + TAIL_TOKENS, { gpus })
  const tail = prefillMs(m, g, PREFIX_TOKENS + TAIL_TOKENS, { gpus, cachedTokens: PREFIX_TOKENS })
  const kv = kvBytesForSeq(m, PREFIX_TOKENS)
  // В HBM всё входит по PCIe самих GPU (A100 Gen4 ~25 ГБ/с, Hopper и Blackwell Gen5 ~50 ГБ/с, по линку на GPU реплики):
  // диск и сеть быстрее этого потолка не бывают
  const pcie = gpuHostLink(g).practicalGBs * gpus
  const cap = (GBs: number) => Math.min(GBs, pcie)
  const nvme = getInterconnect('nvme5').GBs
  const rdma = getInterconnect('ib-ndr').GBs
  const netGBs = net === 'tcp' ? getInterconnect('tcp-25g').GBs : rdma
  const over = (loadMs: number) => Math.max(loadMs, tail)
  return {
    l1: tail,
    local: over(transferMs(kv, pcie)),
    'node-dram': over(transferMs(kv, pcie)),
    'node-nvme': over(transferMs(kv, cap(nvme))),
    peer: over(transferMs(kv, cap(rdma), 0.05)),
    cluster: over(transferMs(kv, cap(netGBs), 0.05)),
    miss,
    kvBytes: kv,
  }
}

/* ============================================================================
 * Мелкие части
 * ========================================================================== */

type SlotTone = 'good' | 'prefill' | 'warn'

function PrefixSlots({ items, cap, hl, label }: { items: PrefixId[]; cap: number; hl?: { prefix: PrefixId; tone: SlotTone } | null; label: string }) {
  const shown = Math.min(cap, 4)
  return (
    <div className="flex min-w-0 items-center gap-2">
      <span className="w-[86px] shrink-0 text-[11.5px] leading-tight text-ink-2">{label}</span>
      <div className="flex flex-wrap gap-1" aria-label={`${label}: ${items.length ? items.join(', ') : 'пусто'}`}>
        {items.slice(-shown).map((p) => {
          const on = hl && hl.prefix === p
          return (
            <motion.span
              key={p}
              layout
              initial={{ opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex h-6 w-6 items-center justify-center rounded-[5px] border font-mono text-[12px] font-semibold text-ink"
              style={{
                background: on ? tint(hl!.tone, 24) : 'var(--surface)',
                borderColor: on ? cssVar(hl!.tone) : 'var(--line-strong)',
                borderWidth: on ? 2 : 1,
              }}
            >
              {p}
            </motion.span>
          )
        })}
        {Array.from({ length: Math.max(0, shown - Math.min(items.length, shown)) }, (_, i) => (
          <span key={`e${i}`} aria-hidden className="h-6 w-6 rounded-[5px] border border-dashed border-line-strong" />
        ))}
        {cap > 4 && items.length > 4 && <span className="self-center text-[11px] text-ink-3">+{items.length - 4}</span>}
      </div>
    </div>
  )
}

/** Фаза пода: иконка и слово (компактнее бейджа, чтобы имя пода помещалось на узких экранах). */
function PhaseBadge({ phase }: { phase: string }) {
  const ok = phase === 'Running'
  const Icon = ok ? CheckCircleIcon : HourglassMediumIcon
  return (
    <span className="inline-flex shrink-0 items-center gap-1 font-mono text-[11px] text-ink-2">
      <Icon size={14} aria-hidden style={{ color: cssVar(ok ? 'good' : 'warn') }} />
      {phase}
    </span>
  )
}

function WorkerCard({
  w,
  mode,
  state,
  onSend,
  onKill,
}: {
  w: Worker
  mode: CacheMode
  state: ClusterState
  onSend: () => void
  onKill: () => void
}) {
  const flow = state.flow
  const mine = flow?.worker === w.id
  const fresh = state.flash?.kind === 'kill' && state.flash.id === w.id
  const hlL1 = mine ? { prefix: flow!.prefix, tone: flow!.source === 'miss' || flow!.recomputed ? ('prefill' as const) : ('good' as const) } : null
  return (
    <div
      className={cn('grid min-w-0 gap-1.5 rounded-control border bg-surface p-2 transition-shadow', mine ? 'border-accent' : 'border-line')}
      style={mine ? { boxShadow: '0 0 0 1px var(--accent)' } : fresh ? { boxShadow: `0 0 0 2px ${cssVar('warn')}` } : undefined}
    >
      <div className="flex min-w-0 items-center gap-1.5">
        <CubeIcon size={16} aria-hidden className="shrink-0 text-ink-2" />
        <span className="min-w-0 truncate font-mono text-[12px] font-medium text-ink">{w.name}</span>
        <PhaseBadge phase={w.phase} />
        <span className="ml-auto flex shrink-0 gap-0.5">
          <Tooltip content="Отправить запрос в этот под">
            <Button size="sm" variant="ghost" iconOnly icon={PaperPlaneRightIcon} aria-label={`Отправить запрос в под ${w.name}`} disabled={w.phase !== 'Running'} onClick={onSend} />
          </Tooltip>
          <Tooltip content="Удалить под (kubectl delete pod)">
            <Button size="sm" variant="ghost" iconOnly icon={TrashIcon} aria-label={`Удалить под ${w.name}`} disabled={w.phase !== 'Running'} onClick={onKill} />
          </Tooltip>
        </span>
      </div>
      <PrefixSlots label="HBM (L1)" items={w.l1} cap={CAP.l1} hl={hlL1} />
      {mode === 'embedded' && (
        <PrefixSlots label="RAM пода (L2)" items={w.local} cap={CAP.local} hl={mine && flow!.source === 'local' ? { prefix: flow!.prefix, tone: 'good' } : null} />
      )}
    </div>
  )
}

function Connector({ active, label, reverse }: { active: boolean; label: ReactNode; reverse?: boolean }) {
  return (
    <div className="relative flex items-center gap-2 py-0.5 pl-4">
      <span aria-hidden className="relative block h-7 w-px" style={{ background: active ? 'var(--accent)' : 'var(--line-strong)' }}>
        <AnimatePresence>
          {active && (
            <motion.span
              key="dot"
              className="absolute -left-[3px] h-[7px] w-[7px] rounded-full bg-accent"
              initial={{ y: reverse ? 0 : 22, opacity: 0 }}
              animate={{ y: reverse ? 22 : 0, opacity: [0, 1, 1, 0] }}
              transition={{ duration: 0.9, repeat: 1 }}
            />
          )}
        </AnimatePresence>
      </span>
      <span className="text-[11px] leading-tight text-ink-3">{label}</span>
    </div>
  )
}

function NodeCard({
  node,
  state,
  dispatch,
  onSend,
}: {
  node: NodeState
  state: ClusterState
  dispatch: (a: Action) => void
  onSend: (workerId: string) => void
}) {
  const flow = state.flow
  const cache = node.cache
  const fromHere = flow && flow.node === node.id && (flow.source === 'node-dram' || flow.source === 'node-nvme')
  const peerHere = flow && flow.peerNode === node.id
  // нашлось, но загрузка вышла бы дольше пересчёта: подсвечиваем предупреждением, а не попаданием
  const foundTone: SlotTone = flow?.recomputed ? 'warn' : 'good'
  const storeHere = flow && flow.node === node.id && flow.source === 'miss' && state.mode === 'daemonset'
  const isNew = state.flash?.kind === 'node' && state.flash.id === node.id
  const restarted = state.flash?.kind === 'restart' && state.flash.id === node.id
  const canFail = state.nodes.length > 1
  const flowHere = flow && flow.node === node.id
  return (
    <motion.section
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.96 }}
      aria-label={`Нода ${node.name}`}
      className="grid min-w-0 content-start gap-2 rounded-panel border border-line bg-surface-2 p-2.5"
      // свойство передаём всегда: motion не снимает ключ стиля, если style стал undefined, и кольцо оставалось навсегда
      style={{ boxShadow: isNew ? '0 0 0 2px var(--accent)' : '0 0 0 0 transparent' }}
    >
      <div className="flex min-w-0 items-center gap-2">
        <HardDrivesIcon size={18} aria-hidden className="shrink-0 text-ink-2" />
        <div className="grid min-w-0 leading-tight">
          <span className="truncate font-mono text-[12.5px] font-semibold text-ink">{node.name}</span>
          <span className="truncate text-[11px] text-ink-3">8 GPU · RAM 2 ТБ · NVMe 30,7 ТБ</span>
        </div>
        <Tooltip content={canFail ? 'Нода выходит из строя целиком' : 'Последнюю ноду не роняем'}>
          <Button
            size="sm"
            variant="ghost"
            icon={LightningIcon}
            className="ml-auto shrink-0"
            disabled={!canFail}
            onClick={() => dispatch({ type: 'failNode', nodeId: node.id })}
            aria-label={`Отказ ноды ${node.name}`}
          >
            Сбой ноды
          </Button>
        </Tooltip>
      </div>
      <div className="grid gap-1.5">
        <AnimatePresence initial={false}>
          {node.workers.map((id) => {
            const w = state.workers[id]
            if (!w) return null
            return (
              <motion.div key={id} layout initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}>
                <WorkerCard
                  w={w}
                  mode={state.mode}
                  state={state}
                  onSend={() => onSend(w.id)}
                  onKill={() => dispatch({ type: 'kill', workerId: w.id })}
                />
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>
      {state.mode === 'daemonset' && cache && (
        <>
          <Connector
            active={Boolean(fromHere || storeHere)}
            reverse={Boolean(storeHere)}
            label={
              <>
                локально: Service <span className="font-mono">kv-cache:5555</span>, internalTrafficPolicy: Local; данные по CUDA IPC
              </>
            }
          />
          <div
            className={cn('grid min-w-0 gap-1.5 rounded-control border bg-surface p-2', fromHere || peerHere ? 'border-transparent' : 'border-line')}
            style={
              // источник KV: пунктир цветом текста (а не вторая зелёная рамка рядом с акцентной рамкой воркера);
              // если пересчитать оказалось быстрее, пунктир предупреждающий
              fromHere || peerHere
                ? { outline: `2px dashed ${foundTone === 'warn' ? cssVar('warn') : 'var(--ink-2)'}`, outlineOffset: -1 }
                : restarted
                  ? { boxShadow: `0 0 0 2px ${cssVar('warn')}` }
                  : undefined
            }
          >
            <div className="flex min-w-0 items-center gap-1.5">
              <DatabaseIcon size={16} aria-hidden className="shrink-0 text-ink-2" />
              <span className="min-w-0 truncate font-mono text-[12px] font-medium text-ink">{cache.name}</span>
              <PhaseBadge phase={cache.phase} />
              <Tooltip content="Перезапуск пода кэша, как при обновлении DaemonSet">
                <Button
                  size="sm"
                  variant="ghost"
                  iconOnly
                  icon={ArrowClockwiseIcon}
                  className="ml-auto shrink-0"
                  aria-label={`Перезапустить под кэша на ${node.name}`}
                  disabled={cache.phase !== 'Running'}
                  onClick={() => dispatch({ type: 'restartCache', nodeId: node.id })}
                />
              </Tooltip>
            </div>
            <div className="text-[11px] leading-tight text-ink-3">под DaemonSet: один на ноду, владеет её RAM и NVMe</div>
            {(fromHere || peerHere) && (
              <div className="text-[11px] leading-tight font-medium text-ink">
                {flow?.recomputed ? 'KV нашёлся здесь, но пересчитать быстрее' : 'KV этого запроса взят отсюда'}
              </div>
            )}
            <PrefixSlots
              label="RAM ноды"
              items={cache.dram}
              cap={CAP.dram}
              hl={
                flowHere && flow!.source === 'node-dram'
                  ? { prefix: flow!.prefix, tone: foundTone }
                  : peerHere
                    ? { prefix: flow!.prefix, tone: foundTone }
                    : storeHere
                      ? { prefix: flow!.prefix, tone: 'prefill' }
                      : null
              }
            />
            <PrefixSlots
              label="NVMe ноды"
              items={cache.nvme}
              cap={CAP.nvme}
              hl={flowHere && flow!.source === 'node-nvme' ? { prefix: flow!.prefix, tone: foundTone } : null}
            />
          </div>
        </>
      )}
      {state.mode === 'cluster' && (
        <Connector
          active={Boolean(flowHere && flow!.source === 'cluster')}
          label="по сети к кэш-кластеру"
        />
      )}
    </motion.section>
  )
}

const LOG_ICON = { good: CheckCircleIcon, warn: WarningIcon, bad: XCircleIcon, info: InfoIcon } as const

function EventLog({ state }: { state: ClusterState }) {
  return (
    <div className="grid min-w-0 content-start gap-2">
      <div className="text-sm font-medium text-ink">События</div>
      <ol className="grid max-h-[248px] gap-1.5 overflow-y-auto pr-1" aria-live="polite">
        <AnimatePresence initial={false}>
          {state.log.slice(0, 8).map((e, i) => {
            const Icon = LOG_ICON[e.tone]
            return (
              <motion.li
                key={e.id}
                layout
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: i === 0 ? 1 : 0.78, y: 0 }}
                className="flex items-start gap-2 rounded-control px-2 py-1.5 text-[13px] leading-snug text-ink"
                style={{ background: i === 0 ? (e.tone === 'info' ? 'var(--surface-2)' : tint(e.tone, 12)) : 'transparent' }}
              >
                <Icon size={16} aria-hidden className="mt-0.5 shrink-0" style={{ color: e.tone === 'info' ? 'var(--ink-3)' : cssVar(e.tone) }} />
                <span>{e.text}</span>
              </motion.li>
            )
          })}
        </AnimatePresence>
      </ol>
    </div>
  )
}

const COMPARE: Array<{ row: string; cells: Record<CacheMode, string> }> = [
  {
    row: 'Кто видит кэш',
    cells: { embedded: 'только этот под', daemonset: 'все поды ноды; с общим пулом весь кластер', cluster: 'все поды кластера' },
  },
  {
    row: 'Рестарт пода воркера',
    cells: { embedded: 'теряется всё: HBM и RAM пода', daemonset: 'теряется только HBM, кэш ноды цел', cluster: 'теряется только HBM' },
  },
  {
    row: 'Путь к данным',
    cells: { embedded: 'внутри процесса, по PCIe', daemonset: 'внутри ноды: CUDA IPC, общая память, PCIe', cluster: 'сеть: нужен RDMA, иначе медленнее пересчёта' },
  },
  {
    row: 'Сбой ноды',
    cells: { embedded: 'пропадает кэш её подов', daemonset: 'пропадает кэш этой ноды', cluster: 'кэш цел, пока жив сам кэш-кластер' },
  },
  {
    row: 'Радиус поражения',
    cells: { embedded: 'один под', daemonset: 'одна нода; обновление DaemonSet охлаждает ноды по очереди', cluster: 'весь кластер, если упадёт кэш-кластер или его мастер' },
  },
  {
    row: 'Новые реплики',
    cells: { embedded: 'стартуют холодными', daemonset: 'на тёплой ноде стартуют тёплыми; новая нода сама получает под кэша', cluster: 'тёплые сразу' },
  },
  {
    row: 'Реальные примеры',
    cells: {
      embedded: 'vLLM OffloadingConnector (CPU), SGLang HiCache L2, LMCacheConnectorV1',
      daemonset: 'LMCache Operator (LMCacheEngine → DaemonSet), llm-d mooncake-client',
      cluster: 'Mooncake Store как Deployment, кэш-сервер production-stack',
    },
  },
]

const MODE_NAMES: Record<CacheMode, string> = { embedded: 'Внутри воркера', daemonset: 'DaemonSet на ноде', cluster: 'Кэш-кластер' }

/** Узкая фигура: сравнение списком только для выбранного режима, полная таблица по кнопке. */
function CompareList({ mode, onMode }: { mode: CacheMode; onMode: (m: CacheMode) => void }) {
  const [all, setAll] = useState(false)
  return (
    <div className="grid min-w-0 gap-3 rounded-panel border border-line p-3">
      <div className="text-sm font-medium text-ink">Режим «{MODE_NAMES[mode]}» в сравнении</div>
      <dl className="grid gap-2.5">
        {COMPARE.map((r) => (
          <div key={r.row} className="grid gap-0.5">
            <dt className="text-xs text-ink-3">{r.row}</dt>
            <dd className="text-[13px] leading-snug text-ink">{r.cells[mode]}</dd>
          </div>
        ))}
      </dl>
      <div>
        <Button size="sm" variant="secondary" aria-expanded={all} onClick={() => setAll((v) => !v)}>
          {all ? 'Скрыть таблицу' : 'Сравнить все три'}
        </Button>
      </div>
      {all && <CompareTable mode={mode} onMode={onMode} />}
    </div>
  )
}

function CompareTable({ mode, onMode }: { mode: CacheMode; onMode: (m: CacheMode) => void }) {
  const modes: CacheMode[] = ['embedded', 'daemonset', 'cluster']
  return (
    <div className="overflow-x-auto rounded-panel border border-line">
      <table className="w-full min-w-[720px] border-collapse text-left text-[13px]">
        <caption className="sr-only">Сравнение трёх способов разместить кэш</caption>
        <thead>
          <tr className="bg-surface-2">
            <th scope="col" className="w-[150px] px-3 py-2 font-medium text-ink-2">
              &nbsp;
            </th>
            {modes.map((m) => (
              <th key={m} scope="col" className={cn('px-3 py-2', m === mode && 'bg-accent-soft')}>
                <button type="button" onClick={() => onMode(m)} className="font-semibold text-ink underline-offset-2 hover:underline">
                  {MODE_NAMES[m]}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {COMPARE.map((r) => (
            <tr key={r.row} className="border-t border-line">
              <th scope="row" className="px-3 py-2 align-top font-medium text-ink">
                {r.row}
              </th>
              {modes.map((m) => (
                <td key={m} className={cn('px-3 py-2 align-top', m === mode ? 'bg-accent-soft text-ink' : 'text-ink-2')}>
                  {r.cells[m]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ============================================================================
 * Фигура
 * ========================================================================== */

export function ClusterFigure() {
  const [settings] = useSettings()
  const model = modelOr(settings.modelId)
  const gpu = gpuOr(settings.gpuId)
  const gpus = useMemo(() => minGpus(model, gpu), [model, gpu])
  const [net, setNet] = useState<'tcp' | 'rdma'>('tcp')
  const timing = useMemo(() => computeTiming(model.id, gpu.id, gpus, net), [model.id, gpu.id, gpus, net])
  const timingRef = useRef(timing)
  timingRef.current = timing
  const [state, dispatch] = useReducer((s: ClusterState, a: Action) => reduce(s, a, timingRef.current), undefined, () => initialState('daemonset'))
  const [prefix, setPrefix] = useState<PrefixId>('A')
  const [boxRef, box] = useElementSize<HTMLDivElement>()
  const narrow = box.width > 0 && box.width < 640
  const latest = state.log[0]
  const LatestIcon = latest ? LOG_ICON[latest.tone] : null

  // ContainerCreating → Running и Pending → Running: по таймеру, один раз на под
  const scheduled = useRef(new Set<string>())
  const timers = useRef<number[]>([])
  useEffect(() => {
    for (const w of Object.values(state.workers)) {
      if (w.phase === 'ContainerCreating' && !scheduled.current.has(w.id)) {
        scheduled.current.add(w.id)
        timers.current.push(window.setTimeout(() => dispatch({ type: 'ready', workerId: w.id }), 1300))
      }
    }
    for (const n of state.nodes) {
      const key = `cache:${n.id}:${n.cache?.name}`
      if (n.cache?.phase === 'Pending' && !scheduled.current.has(key)) {
        scheduled.current.add(key)
        timers.current.push(window.setTimeout(() => dispatch({ type: 'cacheReady', nodeId: n.id }), 900))
      }
    }
  }, [state.workers, state.nodes])
  useEffect(
    () => () => {
      for (const t of timers.current) window.clearTimeout(t)
    },
    [],
  )

  const st = state.stats
  const hits = st.requests - st.by.miss
  const gpuLabel = `${gpus > 1 ? `${gpus} × ` : ''}${shortGpuName(gpu)}`
  const send = (workerId?: string) => dispatch({ type: 'send', prefix, workerId })
  const setMode = (m: CacheMode) => {
    scheduled.current.clear()
    dispatch({ type: 'setMode', mode: m })
  }
  const flowSourceCount = (s: Source[]) => s.reduce((a, k) => a + st.by[k], 0)

  return (
    <Figure
      id="kv-tiers-figure-cluster"
      title="Kubernetes: где живёт L3 KV cache"
      width="full"
      caption="Попробуйте: в режиме DaemonSet отправьте промпт A в один под, потом в соседний на той же ноде: второй возьмёт KV из кэша ноды. Удалите под и отправьте запрос в новый: кэш пережил рестарт. Добавьте ноду: под кэша появится на ней сам. Затем переключитесь на «В воркере» и повторите."
    >
      <div ref={boxRef} className="grid min-w-0 gap-4">
        <div className="grid min-w-0 gap-2">
          <Segmented
            label="Где живёт кэш"
            wrap
            value={state.mode}
            onChange={setMode}
            options={[
              { value: 'embedded', label: 'В воркере', title: 'Внутри воркера' },
              { value: 'daemonset', label: 'DaemonSet' },
              { value: 'cluster', label: 'Кэш-кластер' },
            ]}
          />
          <p className="max-w-[88ch] text-sm text-ink-2">{MODE_TEXT[state.mode]}</p>
        </div>

        <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
          <Segmented
            label="Промпт запроса"
            size="sm"
            value={prefix}
            onChange={setPrefix}
            options={PREFIXES.map((p) => ({ value: p, label: `Промпт ${p}` }))}
          />
          <Button variant="primary" icon={PaperPlaneRightIcon} onClick={() => send()}>
            Отправить запрос
          </Button>
          <Button icon={PlusIcon} onClick={() => dispatch({ type: 'addNode' })} disabled={state.nodes.length >= MAX_NODES}>
            Добавить ноду
          </Button>
          <Button variant="ghost" icon={ArrowCounterClockwiseIcon} onClick={() => setMode(state.mode)}>
            Сначала
          </Button>
          {state.mode === 'daemonset' && (
            <Toggle
              label="Общий пул между нодами"
              checked={state.pooled}
              onChange={(v) => dispatch({ type: 'setPooled', pooled: v })}
              hint="как llm-d с Mooncake: master знает, где какой блок"
            />
          )}
          {state.mode === 'cluster' && (
            <Segmented
              label="Сеть до кэш-кластера"
              size="sm"
              value={net}
              onChange={setNet}
              options={[
                { value: 'tcp', label: 'TCP 25 Гбит/с' },
                { value: 'rdma', label: 'RDMA 400 Гбит/с' },
              ]}
            />
          )}
        </div>
        {latest && LatestIcon && state.log.length > 1 && (
          <p
            aria-hidden
            className="flex items-start gap-2 rounded-control px-2.5 py-2 text-[13px] leading-snug text-ink"
            style={{ background: latest.tone === 'info' ? 'var(--surface-2)' : tint(latest.tone, 12) }}
          >
            <LatestIcon size={16} className="mt-0.5 shrink-0" style={{ color: latest.tone === 'info' ? 'var(--ink-3)' : cssVar(latest.tone) }} />
            <span>{latest.text}</span>
          </p>
        )}
        <p className="text-[13px] text-ink-3">
          Промпты A, B и C это три разных системных промпта по {fmt.int(PREFIX_TOKENS)} токенов ({fmt.bytes(timing.kvBytes)} KV у{' '}
          {model.name}), к каждому добавляется новый вопрос на {TAIL_TOKENS} токенов. Роутер раздаёт запросы по кругу, или отправьте в
          конкретный под кнопкой со стрелкой.
        </p>

        <div className="grid min-w-0 gap-3 rounded-panel border border-dashed border-line-strong p-3">
          <div className="flex flex-wrap items-baseline gap-x-2 text-[13px] font-medium text-ink-2">
            <span className="whitespace-nowrap">Kubernetes-кластер</span>
            <span className="font-normal text-ink-3">воркеры vLLM, {gpuLabel} на реплику</span>
          </div>
          {state.mode === 'daemonset' && state.pooled && (
            <div
              className="flex flex-wrap items-center gap-2 rounded-control border border-line bg-surface px-3 py-2 text-[12.5px] text-ink-2"
              style={state.flow?.source === 'peer' ? { boxShadow: `0 0 0 2px ${cssVar('good')}` } : undefined}
            >
              <DatabaseIcon size={16} aria-hidden />
              <span className="font-mono font-medium text-ink">mooncake-master</span>
              <Badge size="sm">Deployment</Badge>
              <span>метаданные: на какой ноде какой блок. Сами данные идут между нодами напрямую по RDMA.</span>
            </div>
          )}
          <div className="grid min-w-0 grid-cols-[repeat(auto-fit,minmax(min(100%,300px),1fr))] gap-3">
            <AnimatePresence initial={false}>
              {state.nodes.map((n) => (
                <NodeCard key={n.id} node={n} state={state} dispatch={dispatch} onSend={(id) => send(id)} />
              ))}
            </AnimatePresence>
          </div>
          {state.mode === 'cluster' && (
            <div
              className="grid min-w-0 gap-2 rounded-control border border-line bg-surface p-3"
              style={state.flow?.source === 'cluster' ? { boxShadow: `0 0 0 2px ${cssVar(state.flow.recomputed ? 'warn' : 'good')}` } : undefined}
            >
              <div className="flex flex-wrap items-center gap-2">
                <DatabaseIcon size={16} aria-hidden className="text-ink-2" />
                <span className="text-[13px] font-semibold text-ink">Кэш-кластер</span>
                <Badge size="sm">StatefulSet × 3</Badge>
                <span className="text-[12px] text-ink-3">kv-store-0, kv-store-1, kv-store-2 на CPU-нодах с большой RAM и NVMe</span>
              </div>
              <PrefixSlots
                label="Хранилище"
                items={state.store}
                cap={CAP.store}
                hl={
                  state.flow && (state.flow.source === 'cluster' || state.flow.source === 'miss')
                    ? { prefix: state.flow.prefix, tone: state.flow.source === 'cluster' ? (state.flow.recomputed ? 'warn' : 'good') : 'prefill' }
                    : null
                }
              />
            </div>
          )}
        </div>

        <div className="grid min-w-0 gap-5 @4xl:grid-cols-[minmax(0,1fr)_minmax(0,340px)]">
          <EventLog state={state} />
          <div className="grid min-w-0 content-start gap-4">
            <div className="grid grid-cols-2 gap-x-6 gap-y-4">
              <Stat label="Запросов" value={fmt.int(st.requests)} />
              <Stat label="Из кэша" value={fmt.pct(st.requests ? hits / st.requests : 0)} swatch={cssVar('good')} hint={st.requests ? undefined : 'запросов ещё не было'} />
              <Stat label="Средний TTFT" value={st.requests ? fmt.ms(st.ttftSum / st.requests) : 'пока нет'} hint={`${model.name}, оценка`} />
              <Stat label="Промахов" value={fmt.int(st.by.miss)} swatch={cssVar('prefill')} hint={`prefill ${fmt.ms(timing.miss)}`} />
            </div>
            <dl className="grid gap-1 text-[12.5px] text-ink-2">
              <div className="flex justify-between gap-3">
                <dt>из HBM воркера</dt>
                <dd className="tabular-nums text-ink">
                  {st.by.l1} · {fmt.ms(timing.l1)}
                </dd>
              </div>
              {state.mode === 'embedded' && (
                <div className="flex justify-between gap-3">
                  <dt>из RAM пода</dt>
                  <dd className="tabular-nums text-ink">
                    {st.by.local} · {fmt.ms(timing.local)}
                  </dd>
                </div>
              )}
              {state.mode === 'daemonset' && (
                <>
                  <div className="flex justify-between gap-3">
                    <dt>из кэша ноды: RAM / NVMe</dt>
                    <dd className="tabular-nums text-ink">
                      {flowSourceCount(['node-dram'])} / {flowSourceCount(['node-nvme'])} · {fmt.ms(timing['node-dram'])} / {fmt.ms(timing['node-nvme'])}
                    </dd>
                  </div>
                  {state.pooled && (
                    <div className="flex justify-between gap-3">
                      <dt>с соседней ноды</dt>
                      <dd className="tabular-nums text-ink">
                        {st.by.peer} · {fmt.ms(timing.peer)}
                      </dd>
                    </div>
                  )}
                </>
              )}
              {state.mode === 'cluster' && (
                <div className="flex justify-between gap-3">
                  <dt>из кэш-кластера</dt>
                  <dd className="tabular-nums text-ink">
                    {st.by.cluster} · {fmt.ms(Math.min(timing.cluster, timing.miss))}
                  </dd>
                </div>
              )}
            </dl>
            <p className="text-[12px] leading-snug text-ink-3">
              TTFT без очереди: загрузка KV и prefill нового хвоста идут слой за слоем параллельно, поэтому считаем большее из двух.
            </p>
          </div>
        </div>

        {narrow ? <CompareList mode={state.mode} onMode={setMode} /> : <CompareTable mode={state.mode} onMode={setMode} />}
      </div>
    </Figure>
  )
}
