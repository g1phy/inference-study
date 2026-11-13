import { useMemo, useRef, useState, type PointerEvent } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { BroomIcon } from '@phosphor-icons/react'
import { Figure, Player, Segmented, Slider, cn, cssVar, fmt, tint, truncateText, useElementSize, usePlayer, useTip } from '@/ui'
import {
  NODE_H,
  NODE_W,
  PRESETS,
  ROOT_W,
  getPreset,
  longestRequestTokens,
  radixStep,
  totalUniqueTokens,
  type LaidNode,
  type Preset,
} from './radix'

const LABEL_FONT = '600 12.5px "Onest Variable", sans-serif'
const META_FONT = '11.5px "Onest Variable", sans-serif'

/** Состояние узла словом (не только цветом): «из кэша», «посчитан», «вытеснен». */
const nodeStatus = (n: LaidNode) => (n.ghost ? 'вытеснен' : n.hit ? 'из кэша' : n.isNew ? 'посчитан' : '')

/** Вторая строка узла: токены, сколько запросов прошло через узел и состояние. */
const nodeMeta = (n: LaidNode) => {
  const st = nodeStatus(n)
  return `${fmt.int(n.tokens)} ток.${n.shared > 1 && !n.ghost ? ` · ${n.shared} запр.` : ''}${st ? ` · ${st}` : ''}`
}

function NodeBox({
  n,
  focused,
  onTip,
  onLeave,
  onFocusNode,
}: {
  n: LaidNode
  focused: boolean
  onTip: (n: LaidNode, el: Element) => void
  onLeave: () => void
  onFocusNode: (id: string | null) => void
}) {
  const stroke = n.ghost ? 'var(--line-strong)' : n.hit ? cssVar('good') : n.isNew ? cssVar('prefill') : 'var(--line-strong)'
  const fill = n.ghost ? 'none' : n.isNew ? tint('prefill', 14) : n.hit ? tint('good', 12) : 'var(--surface-2)'
  const label = truncateText(n.label, NODE_W - 20, LABEL_FONT)
  const status = nodeStatus(n)
  return (
    <motion.g
      initial={{ opacity: 0, x: n.x, y: n.y }}
      animate={{ opacity: 1, x: n.x, y: n.y }}
      exit={{ opacity: 0, transition: { duration: 0.25 } }}
      transition={{ type: 'spring', stiffness: 260, damping: 30 }}
      onPointerEnter={(e) => onTip(n, e.currentTarget)}
      onPointerDown={(e) => onTip(n, e.currentTarget)}
      onPointerLeave={(e: PointerEvent<SVGGElement>) => e.pointerType === 'mouse' && onLeave()}
      onFocus={(e) => {
        onTip(n, e.currentTarget)
        if (e.currentTarget.matches(':focus-visible')) onFocusNode(n.id)
      }}
      onBlur={() => {
        onLeave()
        onFocusNode(null)
      }}
      tabIndex={0}
      role="listitem"
      aria-label={`${n.label}: ${n.tokens} токенов, запросов через узел ${n.shared}${status ? `, ${status}` : ''}`}
      className="outline-none"
    >
      <rect
        width={NODE_W}
        height={NODE_H}
        rx={8}
        style={{ fill, stroke, transition: 'fill 200ms, stroke 200ms' }}
        strokeWidth={n.hit || n.isNew ? 2 : 1}
        strokeDasharray={n.ghost ? '5 4' : undefined}
      />
      {focused && <rect x={-4} y={-4} width={NODE_W + 8} height={NODE_H + 8} rx={11} fill="none" stroke="var(--accent)" strokeWidth={2} />}
      <text x={10} y={18} fontSize={12.5} fontWeight={600} fill={n.ghost ? 'var(--ink-2)' : 'var(--ink)'}>
        {label}
      </text>
      <text x={10} y={35} fontSize={11.5} fill="var(--ink-2)">
        {truncateText(nodeMeta(n), NODE_W - 16, META_FONT)}
      </text>
    </motion.g>
  )
}

function Tree({ step }: { step: ReturnType<typeof radixStep> }) {
  const ref = useRef<SVGSVGElement | null>(null)
  const tip = useTip({ scopeEl: ref.current, side: 'top', wide: true })
  const [focusId, setFocusId] = useState<string | null>(null)
  const byId = useMemo(() => new Map(step.nodes.map((n) => [n.id, n])), [step.nodes])
  const root = byId.get('root')!
  const pad = 6
  const W = Math.max(step.width, ROOT_W) + pad * 2
  const H = step.height + pad * 2
  const showTip = (n: LaidNode, el: Element) =>
    tip.show(
      el.getBoundingClientRect(),
      <div className="grid gap-1">
        <div className="font-medium text-ink">{n.parts.join(' + ')}</div>
        <div className="text-ink-2">
          {fmt.tokens(n.tokens)} на ребре, через узел {fmt.plural(n.shared, ['прошёл', 'прошли', 'прошли'])} {n.shared}{' '}
          {fmt.plural(n.shared, ['запрос', 'запроса', 'запросов'])}
        </div>
        {n.ghost && <div className="text-ink-2">Узел вытеснен по LRU на этом шаге: его KV больше нет, на следующем шаге он исчезнет.</div>}
        {!n.ghost && n.hit && <div className="text-ink-2">Этот кусок последний запрос взял из кэша.</div>}
        {!n.ghost && n.isNew && (
          <div className="text-ink-2">Этот кусок последний запрос посчитал сам (новую часть промпта в prefill, ответ в decode) и добавил в дерево.</div>
        )}
      </div>,
    )
  return (
    <div className="overflow-x-auto pb-1">
      <svg ref={ref} width={W} height={H} viewBox={`${-pad} ${-pad} ${W} ${H}`} role="list" aria-label="Дерево префиксов" className="block">
        <AnimatePresence initial={false}>
          {step.nodes
            .filter((n) => n.id !== 'root' && n.parent)
            .map((n) => {
              const par = byId.get(n.parent!)
              if (!par) return null
              const x1 = par.id === 'root' ? ROOT_W : par.x + NODE_W
              const y1 = par.y + NODE_H / 2
              const x2 = n.x
              const y2 = n.y + NODE_H / 2
              const mx = (x1 + x2) / 2
              const d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`
              return (
                <motion.path
                  key={`e-${n.id}`}
                  initial={{ opacity: 0, d }}
                  animate={{ opacity: 1, d }}
                  exit={{ opacity: 0 }}
                  transition={{ type: 'spring', stiffness: 260, damping: 30 }}
                  fill="none"
                  stroke={n.hit && !n.ghost ? cssVar('good') : 'var(--line-strong)'}
                  strokeWidth={n.hit && !n.ghost ? 2 : 1.5}
                  strokeDasharray={n.ghost ? '5 4' : undefined}
                />
              )
            })}
        </AnimatePresence>
        <g>
          <rect x={0} y={root.y + NODE_H / 2 - 13} width={ROOT_W} height={26} rx={13} fill="var(--surface)" stroke="var(--line-strong)" strokeWidth={1.5} />
          <text x={ROOT_W / 2} y={root.y + NODE_H / 2 + 4} textAnchor="middle" fontSize={11} fill="var(--ink-2)">
            корень
          </text>
        </g>
        <AnimatePresence initial={false}>
          {step.nodes
            .filter((n) => n.id !== 'root')
            .map((n) => (
              <NodeBox key={n.id} n={n} focused={focusId === n.id} onTip={showTip} onLeave={tip.hide} onFocusNode={setFocusId} />
            ))}
        </AnimatePresence>
      </svg>
      {tip.node}
    </div>
  )
}

/** Узкий экран: то же дерево списком с отступами (корень сверху, дети под родителем). */
function TreeList({ step }: { step: ReturnType<typeof radixStep> }) {
  const rows = useMemo(() => {
    const kids = new Map<string, LaidNode[]>()
    for (const n of step.nodes) {
      if (!n.parent) continue
      const arr = kids.get(n.parent) ?? []
      arr.push(n)
      kids.set(n.parent, arr)
    }
    const out: LaidNode[] = []
    const walk = (id: string) => {
      for (const c of (kids.get(id) ?? []).sort((a, b) => a.y - b.y)) {
        out.push(c)
        walk(c.id)
      }
    }
    walk('root')
    return out
  }, [step.nodes])
  return (
    <div className="grid gap-1.5">
      <span className="justify-self-start rounded-full border border-line-strong px-2.5 py-0.5 text-[11px] text-ink-2">корень</span>
      {rows.length === 0 && <p className="text-sm text-ink-3">Дерево пустое.</p>}
      <ul className="grid gap-1.5" aria-label="Дерево префиксов списком">
        <AnimatePresence initial={false}>
          {rows.map((n) => (
            <motion.li
              key={n.id}
              layout
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              className="border-l border-line-strong"
              style={{ marginLeft: (n.depth - 1) * 14 + 6 }}
            >
              <div
                className={cn(
                  'ml-2 grid gap-0.5 rounded-control border px-2.5 py-1.5',
                  n.ghost ? 'border-dashed border-line-strong' : n.hit || n.isNew ? 'border-transparent' : 'border-line',
                )}
                style={{
                  background: n.ghost ? 'transparent' : n.isNew ? tint('prefill', 14) : n.hit ? tint('good', 12) : 'var(--surface-2)',
                  boxShadow: n.ghost ? undefined : n.hit ? `inset 0 0 0 2px ${cssVar('good')}` : n.isNew ? `inset 0 0 0 2px ${cssVar('prefill')}` : undefined,
                }}
              >
                <span className={cn('text-[13px] leading-snug font-semibold', n.ghost ? 'text-ink-2' : 'text-ink')}>{n.label}</span>
                <span className="flex items-center gap-1.5 text-[12px] text-ink-2">
                  {(n.hit || n.isNew) && !n.ghost && (
                    <span aria-hidden className="inline-block h-2 w-2 rounded-full" style={{ background: n.hit ? cssVar('good') : cssVar('prefill') }} />
                  )}
                  {nodeMeta(n)}
                </span>
              </div>
            </motion.li>
          ))}
        </AnimatePresence>
      </ul>
    </div>
  )
}

function HitList({ step }: { step: ReturnType<typeof radixStep> }) {
  const total = step.history.reduce((a, h) => a + h.promptTokens, 0)
  const hits = step.history.reduce((a, h) => a + h.matched, 0)
  return (
    <div className="grid min-w-0 gap-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-sm font-medium text-ink">Сколько промпта взято из кэша</div>
        {step.history.length > 0 && (
          <div className="text-sm text-ink-2">
            всего: <b className="font-semibold text-ink">{fmt.pct(total ? hits / total : 0)}</b> токенов промптов
          </div>
        )}
      </div>
      {step.history.length === 0 ? (
        <p className="text-sm text-ink-3">Запросов ещё не было.</p>
      ) : (
        <ul className="grid gap-1.5">
          {step.history.map((h, i) => {
            const share = h.promptTokens ? h.matched / h.promptTokens : 0
            return (
              <li key={i} className="grid grid-cols-[minmax(0,9.5rem)_minmax(0,1fr)_3.2rem] items-center gap-3 text-[13px]">
                <span className="truncate text-ink-2">{h.label}</span>
                <span className="flex h-2 gap-[2px] overflow-hidden rounded-full" aria-hidden>
                  {h.matched > 0 && <span style={{ flexGrow: h.matched, flexBasis: 0, background: cssVar('good') }} />}
                  {h.promptTokens - h.matched > 0 && <span style={{ flexGrow: h.promptTokens - h.matched, flexBasis: 0, background: cssVar('prefill') }} />}
                </span>
                <span className="text-right text-ink tabular-nums">{fmt.pct(share)}</span>
              </li>
            )
          })}
        </ul>
      )}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-2 w-3 rounded-full" style={{ background: cssVar('good') }} />
          из кэша (попадание)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-2 w-3 rounded-full" style={{ background: cssVar('prefill') }} />
          посчитано в prefill
        </span>
      </div>
    </div>
  )
}

export function RadixFigure() {
  const [presetId, setPresetId] = useState<Preset['id']>('chat')
  const preset = getPreset(presetId)
  const maxCap = totalUniqueTokens(preset)
  // Ёмкость в токенах; не меньше самого длинного запроса с ответом, иначе дерево не удержит даже текущий запрос
  const minCap = longestRequestTokens(preset)
  const [capTokens, setCapTokens] = useState<number | null>(null)
  const capacity = capTokens === null ? maxCap : Math.min(maxCap, Math.max(minCap, capTokens))
  const limited = capacity < maxCap
  const p = usePlayer(preset.requests.length + 1, { intervalMs: 1800, initialStep: 3 })
  const step = useMemo(() => radixStep(preset, p.step, capacity), [preset, p.step, capacity])
  const req = p.step > 0 ? preset.requests[p.step - 1] : null
  const last = step.history[step.history.length - 1]
  const labels = ['Дерево пустое', ...preset.requests.map((r) => r.label)]
  const [treeBox, treeSize] = useElementSize<HTMLDivElement>()

  return (
    <Figure
      id="prefix-caching-figure-radix"
      title="Radix tree: как SGLang хранит общие префиксы"
      width="wide"
      actions={
        <Segmented
          size="sm"
          aria-label="Поток запросов"
          value={presetId}
          onChange={(v) => {
            setPresetId(v)
            p.setPlaying(false)
            p.setStep(Math.min(3, getPreset(v).requests.length))
            // ёмкость остаётся в токенах, но в пределах нового пресета
            const np = getPreset(v)
            if (capTokens !== null) {
              const c = Math.max(longestRequestTokens(np), capTokens)
              setCapTokens(c >= totalUniqueTokens(np) ? null : c)
            }
          }}
          options={PRESETS.map((x) => ({ value: x.id, label: x.short, title: x.name }))}
        />
      }
      controls={
        <>
          <Player {...p.bind} labels={labels} aria-label="Запросы по очереди" />
          <Slider
            label="Ёмкость кэша"
            value={capacity}
            min={minCap}
            max={maxCap}
            step={5}
            onChange={(v) => setCapTokens(v >= maxCap ? null : v)}
            format={(v) => (v >= maxCap ? 'без вытеснения' : fmt.tokens(Math.round(v)))}
            hint="сколько токенов KV помещается в дерево; не меньше самого длинного запроса"
          />
        </>
      }
      caption="Попробуйте: в режиме «Чат» поставьте ёмкость около 1 100 токенов и пройдите все запросы. Старые ветки вытесняются по LRU с листьев, а общий системный промпт живёт, пока на нём держатся ветки."
    >
      <div className="grid min-w-0 gap-4">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm">
          {req && last ? (
            <>
              <span className="font-medium text-ink">
                Запрос {p.step} из {preset.requests.length}: {req.label}
              </span>
              <span className="text-ink-2">
                промпт {fmt.tokens(last.promptTokens)}, из кэша {fmt.int(last.matched)} ({fmt.pct(last.promptTokens ? last.matched / last.promptTokens : 0)}),
                посчитать {fmt.int(last.promptTokens - last.matched)}
              </span>
            </>
          ) : (
            <span className="text-ink-2">Нажмите пуск или шаг вперёд: запросы будут приходить по одному, каждый вместе с ответом модели попадает в дерево.</span>
          )}
        </div>
        <div ref={treeBox} className="min-w-0">
          {treeSize.width > 0 && treeSize.width < 520 ? <TreeList step={step} /> : <Tree step={step} />}
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-[13px] text-ink-2">
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: cssVar('good') }} />
            узел взят из кэша
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: cssVar('prefill') }} />
            узел посчитан сейчас
          </span>
          {step.nodes.some((n) => n.ghost) && (
            <span className="inline-flex items-center gap-1.5">
              <span aria-hidden className="inline-block h-2.5 w-4 rounded-[3px] border border-dashed border-line-strong" />
              вытеснен на этом шаге
            </span>
          )}
          <span>
            в дереве {fmt.tokens(step.size)}
            {limited ? ` из ${fmt.int(capacity)}` : ''}
          </span>
          {step.evicted.length > 0 && (
            <span className="inline-flex items-center gap-1.5 text-ink">
              <BroomIcon size={16} aria-hidden className="text-ink-2" />
              вытеснено по LRU: {step.evicted.join(', ')}
            </span>
          )}
        </div>
        <div className="border-t border-line pt-4">
          <HitList step={step} />
        </div>
      </div>
    </Figure>
  )
}
