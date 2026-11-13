import { useId } from 'react'
import { cn } from '@/ui'
import type { Channel } from '@/engine/types'

interface Props {
  /** Реальная дорожка после шага. */
  current: Channel
  /** Дорожка до шага (если сменилась, подсвечиваем переход). */
  previous: Channel
  /** Придержанный буфер (частичный тег). */
  buffer: string
  /** Старт сразу в рассуждении (forced reasoning). */
  forced?: boolean
  /** Есть ли в сценарии инструменты (иначе узел tool_call бледный). */
  tools?: boolean
  className?: string
}

const NODE_H = 54
/** Ширина моноширинного знака в долях кегля (JetBrains Mono: 600/1000). */
const MONO = 0.6

const LABEL: Record<Channel, { label: string; field: string; fill: string; dashed?: boolean }> = {
  reasoning: { label: 'reasoning', field: 'reasoning_content', fill: 'color-mix(in oklab, var(--c-reasoning) 16%, var(--surface))', dashed: true },
  // content: цвет decode, как у канала content в симуляторе
  content: { label: 'content', field: 'content', fill: 'color-mix(in oklab, var(--c-decode) 14%, var(--surface))' },
  tool_call: { label: 'tool_call', field: 'tool_calls', fill: 'color-mix(in oklab, var(--c-tool) 16%, var(--surface))' },
}

interface EdgeDef {
  from: Channel
  to: Channel
  label: string
}
const EDGES: EdgeDef[] = [
  { from: 'content', to: 'reasoning', label: '<think>' },
  { from: 'reasoning', to: 'content', label: '</think>' },
  { from: 'content', to: 'tool_call', label: '<tool_call>' },
  { from: 'tool_call', to: 'content', label: '</tool_call>' },
]

interface Pt {
  x: number
  y: number
}

/** Кегли подписей в единицах viewBox (вертикальная схема ужимается сильнее, поэтому кегли крупнее). */
interface Fonts {
  node: number
  field: number
  edge: number
  start: number
  buffer: number
}

interface Geo {
  w: number
  h: number
  horizontal: boolean
  nodeW: number
  fs: Fonts
  nodes: Record<Channel, Pt>
  edge: (e: EdgeDef) => { d: string; lx: number; ly: number; anchor: 'start' | 'middle' | 'end' }
  start: (target: Pt) => { dot: Pt; x1: number; y1: number; x2: number; y2: number; tx: number; ty: number }
  /** Верхний левый угол буфера (по центру под узлом). */
  buffer: (n: Pt, width: number) => Pt
}

const H_NODE_W = 150
const H_NODES: Record<Channel, Pt> = { reasoning: { x: 8, y: 70 }, content: { x: 215, y: 70 }, tool_call: { x: 422, y: 70 } }

/** Горизонтально: reasoning ⇄ content ⇄ tool_call. */
const HORIZONTAL: Geo = {
  w: 580,
  h: 216,
  horizontal: true,
  nodeW: H_NODE_W,
  fs: { node: 16, field: 12, edge: 14, start: 13, buffer: 13 },
  nodes: H_NODES,
  edge(e) {
    const a = H_NODES[e.from]
    const b = H_NODES[e.to]
    const forward = e.from === 'content'
    const ltr = a.x < b.x
    const x1 = ltr ? a.x + H_NODE_W - 18 : a.x + 18
    const x2 = ltr ? b.x + 18 : b.x + H_NODE_W - 18
    const y = forward ? a.y : a.y + NODE_H
    const cy = forward ? a.y - 40 : a.y + NODE_H + 40
    const mx = (x1 + x2) / 2
    return { d: `M ${x1} ${y} Q ${mx} ${cy} ${x2} ${y}`, lx: mx, ly: forward ? a.y - 26 : a.y + NODE_H + 34, anchor: 'middle' }
  },
  start: (t) => {
    const x = t.x + H_NODE_W / 2
    return { dot: { x, y: 14 }, x1: x, y1: 19, x2: x, y2: t.y - 4, tx: x + 10, ty: 18 }
  },
  buffer: (n, width) => ({ x: n.x + H_NODE_W / 2 - width / 2, y: n.y + NODE_H + 46 }),
}

/*
 * Вертикально (узкие экраны): reasoning над content, tool_call под ним. На телефоне схема ужимается
 * примерно до 0,8 от viewBox, поэтому кегли от 14: на экране выходит не меньше 11 px.
 */
const V_NODE_W = 166
const V_NODES: Record<Channel, Pt> = { reasoning: { x: 104, y: 28 }, content: { x: 104, y: 158 }, tool_call: { x: 104, y: 288 } }

const VERTICAL: Geo = {
  w: 384,
  h: 398,
  horizontal: false,
  nodeW: V_NODE_W,
  fs: { node: 16, field: 14, edge: 14, start: 14, buffer: 14 },
  nodes: V_NODES,
  edge(e) {
    const a = V_NODES[e.from]
    const b = V_NODES[e.to]
    const forward = e.from === 'content'
    const down = a.y < b.y
    const x = forward ? a.x + 26 : a.x + V_NODE_W - 26
    const y1 = down ? a.y + NODE_H : a.y
    const y2 = down ? b.y : b.y + NODE_H
    const cx = forward ? a.x - 34 : a.x + V_NODE_W + 34
    const my = (y1 + y2) / 2
    return { d: `M ${x} ${y1} Q ${cx} ${my} ${x} ${y2}`, lx: forward ? a.x - 10 : a.x + V_NODE_W + 10, ly: my + 5, anchor: forward ? 'end' : 'start' }
  },
  start: (t) => {
    const y = t.y + NODE_H / 2
    return { dot: { x: t.x - 66, y }, x1: t.x - 61, y1: y, x2: t.x - 4, y2: y, tx: t.x - 66, ty: y - 10 }
  },
  buffer: (n, width) => ({ x: n.x + V_NODE_W / 2 - width / 2, y: n.y + NODE_H + 8 }),
}

/** Диаграмма состояний парсера; текущее состояние обведено цветом текста, последний переход выделен. */
export function StateDiagram({ current, previous, buffer, forced = false, tools = true, className }: Props) {
  const uid = useId().replace(/:/g, '')
  const label = `Машина состояний парсера. Сейчас: ${LABEL[current].label}${buffer ? `, промежуточное состояние maybe_tag, в буфере «${buffer}»` : ''}`
  const common = { current, previous, buffer, forced, tools, label }
  return (
    <div className={className}>
      <Svg geo={HORIZONTAL} id={`${uid}h`} className="mx-auto hidden w-full max-w-[640px] @lg:block" {...common} />
      <Svg geo={VERTICAL} id={`${uid}v`} className="mx-auto block w-full max-w-[400px] @lg:hidden" {...common} />
    </div>
  )
}

interface SvgProps {
  geo: Geo
  id: string
  className: string
  current: Channel
  previous: Channel
  buffer: string
  forced: boolean
  tools: boolean
  label: string
}

function Svg({ geo, id, className, current, previous, buffer, forced, tools, label }: SvgProps) {
  const arrow = `parsers-arrow-${id}`
  const arrowOn = `parsers-arrow-on-${id}`
  const changed = current !== previous
  const st = geo.start(forced ? geo.nodes.reasoning : geo.nodes.content)
  const { fs, nodeW } = geo
  return (
    <svg viewBox={`0 0 ${geo.w} ${geo.h}`} className={cn('h-auto', className)} role="img" aria-label={label}>
      <defs>
        <marker id={arrow} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--line-strong)" />
        </marker>
        <marker id={arrowOn} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--ink)" />
        </marker>
      </defs>

      <g>
        <circle cx={st.dot.x} cy={st.dot.y} r={4} fill="var(--ink-3)" />
        <line x1={st.x1} y1={st.y1} x2={st.x2} y2={st.y2} stroke="var(--line-strong)" strokeWidth={1.5} markerEnd={`url(#${arrow})`} />
        <text x={st.tx} y={st.ty} textAnchor={geo.horizontal ? 'start' : 'middle'} fontSize={fs.start} className="fill-[var(--ink-2)]">
          старт
        </text>
      </g>

      {EDGES.map((e) => {
        const { d, lx, ly, anchor } = geo.edge(e)
        const on = changed && e.from === previous && e.to === current
        const faded = !tools && (e.from === 'tool_call' || e.to === 'tool_call')
        return (
          <g key={`${e.from}-${e.to}`} opacity={faded ? 0.35 : 1}>
            <path
              d={d}
              fill="none"
              stroke={on ? 'var(--ink)' : 'var(--line-strong)'}
              strokeWidth={on ? 2.25 : 1.5}
              markerEnd={on ? `url(#${arrowOn})` : `url(#${arrow})`}
              style={{ transition: 'stroke 200ms, stroke-width 200ms' }}
            />
            <text x={lx} y={ly} textAnchor={anchor} fontSize={fs.edge} className={cn('font-mono', on ? 'fill-[var(--ink)] font-semibold' : 'fill-[var(--ink-2)]')}>
              {e.label}
            </text>
          </g>
        )
      })}

      {(Object.keys(LABEL) as Channel[]).map((ch) => {
        const n = geo.nodes[ch]
        const meta = LABEL[ch]
        const active = ch === current
        const faded = !tools && ch === 'tool_call'
        return (
          <g key={ch} opacity={faded ? 0.4 : 1}>
            {/* текущее состояние обведено цветом текста: зелёный акцент спорил бы с цветом канала tool_calls */}
            <rect
              x={n.x}
              y={n.y}
              width={nodeW}
              height={NODE_H}
              rx={10}
              fill={meta.fill}
              stroke={active ? 'var(--ink)' : 'var(--line-strong)'}
              strokeWidth={active ? 2.25 : 1}
              strokeDasharray={meta.dashed && !active ? '4 3' : undefined}
              style={{ transition: 'stroke 200ms, stroke-width 200ms' }}
            />
            <text x={n.x + nodeW / 2} y={n.y + 23} textAnchor="middle" fontSize={fs.node} className="fill-[var(--ink)] font-mono font-semibold">
              {meta.label}
            </text>
            <text x={n.x + nodeW / 2} y={n.y + 43} textAnchor="middle" fontSize={fs.field} className="fill-[var(--ink-2)] font-mono">
              → {meta.field}
            </text>
          </g>
        )
      })}

      {buffer &&
        (() => {
          const shown = `«${buffer.replace(/\n/g, '↵')}»`
          const title = 'maybe_tag, буфер:'
          // ширина по самой длинной строке (моноширинный шрифт), но не уже узла без отступов
          const width = Math.max(nodeW - 16, Math.max(title.length, shown.length) * fs.buffer * MONO + 18)
          const p = geo.buffer(geo.nodes[current], width)
          const x = Math.max(2, Math.min(geo.w - width - 2, p.x))
          return (
            <g>
              <rect x={x} y={p.y} width={width} height={42} rx={6} fill="var(--surface)" stroke="var(--ink-2)" strokeDasharray="4 3" />
              <text x={x + width / 2} y={p.y + 17} textAnchor="middle" fontSize={fs.buffer} className="fill-[var(--ink-2)] font-mono">
                {title}
              </text>
              <text x={x + width / 2} y={p.y + 34} textAnchor="middle" fontSize={fs.buffer} className="fill-[var(--ink)] font-mono font-semibold">
                {shown}
              </text>
            </g>
          )
        })()}
    </svg>
  )
}
