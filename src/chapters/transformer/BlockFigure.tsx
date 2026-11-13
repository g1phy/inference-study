import { useState, type KeyboardEvent, type ReactNode } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { Badge, ChapterLink, Figure, Formula, Player, Stat, cssVar, fmt, useElementSize, usePlayer } from '@/ui'
import { cn } from '@/ui/cn'
import { type Arch, type Part, partFlops } from './arch'
import { ArchControls } from './ArchControls'

type PartKey = 'in' | 'norm1' | 'attn' | 'add1' | 'norm2' | 'mlp' | 'add2' | 'out'
const ORDER: PartKey[] = ['in', 'norm1', 'attn', 'add1', 'norm2', 'mlp', 'add2', 'out']

/* Геометрия схемы в координатах viewBox 360 × 292. */
const RX = 30
const BOX = {
  norm1: { x: 56, y: 29, w: 96, h: 38 },
  attn: { x: 168, y: 20, w: 186, h: 56 },
  norm2: { x: 56, y: 139, w: 96, h: 38 },
  mlp: { x: 168, y: 130, w: 186, h: 56 },
}
const ADD = { add1: { x: RX, y: 116 }, add2: { x: RX, y: 226 } }
const DOT: Record<PartKey, { x: number; y: number }> = {
  in: { x: RX, y: 22 },
  norm1: { x: 65, y: 48 },
  attn: { x: 180, y: 48 },
  add1: ADD.add1,
  norm2: { x: 65, y: 158 },
  mlp: { x: 180, y: 158 },
  add2: ADD.add2,
  out: { x: RX, y: 262 },
}

function stepLabels(a: Arch): string[] {
  return [
    'Вход блока',
    'RMSNorm перед attention',
    'Attention: обмен между токенами',
    'Сложение с потоком (residual)',
    'RMSNorm перед MLP',
    a.moe ? 'MoE: роутер выбирает экспертов' : 'MLP: каждый токен отдельно',
    'Сложение с потоком (residual)',
    'Выход блока',
  ]
}

export function BlockFigure({
  arch,
  ctx,
  onCtx,
  modelId,
  onModel,
}: {
  arch: Arch
  ctx: number
  onCtx: (n: number) => void
  modelId: string
  onModel: (id: string) => void
}) {
  const labels = stepLabels(arch)
  const p = usePlayer(ORDER.length, { intervalMs: 1800 })
  const key = ORDER[p.step]
  const m = arch.model

  const pick = (k: PartKey) => {
    p.setPlaying(false)
    p.setStep(ORDER.indexOf(k))
  }

  const attnFlops = partFlops(arch.attn, ctx)
  const mlpFlops = partFlops(arch.mlp, ctx)

  return (
    <Figure
      id="transformer-figure-block"
      title={`Один блок ${m.name} изнутри: что делает каждая часть и сколько стоит`}
      controls={<Player {...p.bind} labels={labels} aria-label="Проход вектора через блок" />}
      controlsPosition="top"
      caption="Попробуйте: нажмите на часть схемы или запустите проигрывание. Потяните контекст: FLOPs у attention растут вместе с ним, у MLP нет. Параметры и FLOPs посчитаны по полям config выбранной модели."
    >
      <div className="grid min-w-0 gap-5">
        <div className="grid gap-x-6 gap-y-4 @xl:grid-cols-2">
          <ArchControls modelId={modelId} onModel={onModel} ctx={ctx} onCtx={onCtx} maxCtx={m.maxContext} />
        </div>
        <div className="grid min-w-0 items-start gap-6 @3xl:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
          <BlockDiagram arch={arch} active={key} onPick={pick} />
          {/* на широкой раскладке все панели лежат в одной ячейке: её высота равна самой высокой панели, и ничего под ней не прыгает */}
          <div className="grid min-w-0 self-stretch">
            {ORDER.map((k) => (
              <div key={k} aria-hidden={k !== key} className={cn('min-w-0 [grid-area:1/1]', k === key ? 'block' : 'hidden @3xl:invisible @3xl:block')}>
                <PartDetails arch={arch} part={k} ctx={ctx} />
              </div>
            ))}
          </div>
          <span className="sr-only" aria-live="polite">
            {labels[p.step]}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 border-t border-line pt-4 @2xl:grid-cols-4">
          <Stat label="Параметров в блоке" value={fmt.compact(arch.layerParams)} hint={arch.moe ? `на токен работает ${fmt.compact(arch.layerActiveParams)}` : `${fmt.int(arch.layerParams)}`} />
          <Stat label="FLOPs attention" value={fmt.flops(attnFlops)} hint={`на токен при контексте ${fmt.int(ctx)}: растёт вместе с ним`} />
          <Stat label={arch.moe ? 'FLOPs MoE' : 'FLOPs MLP'} value={fmt.flops(mlpFlops)} hint="на токен, от контекста не зависит" />
          <Stat
            label="Все блоки вместе"
            value={fmt.compact(arch.layerParams * m.layers)}
            hint={`${fmt.int(m.layers)} ${fmt.plural(m.layers, ['блок', 'блока', 'блоков'])} (num_hidden_layers), ${fmt.pct((arch.layerParams * m.layers) / arch.totalParams)} параметров модели`}
          />
        </div>
      </div>
    </Figure>
  )
}

/* ---------------------------------------------------------------------------
 * Схема блока (SVG): поток residual слева, две ветки справа.
 * ------------------------------------------------------------------------- */

function BlockDiagram({ arch, active, onPick }: { arch: Arch; active: PartKey; onPick: (k: PartKey) => void }) {
  const reduce = useReducedMotion()
  const [focus, setFocus] = useState<PartKey | null>(null)
  // viewBox 362 по ширине: на узком экране схема ужимается, и мелкие подписи не должны стать меньше 12px
  const [sizeRef, { width }] = useElementSize<HTMLDivElement>()
  const scale = width > 0 ? width / 362 : 1
  const fs = (base: number) => Math.max(base, 12 / scale)
  const m = arch.model
  const dot = DOT[active]
  const t = reduce ? { duration: 0 } : { type: 'spring' as const, stiffness: 120, damping: 20 }
  const ghost = active === 'norm1' || active === 'attn' ? ADD.add1 : active === 'norm2' || active === 'mlp' ? ADD.add2 : null

  const keyProps = (k: PartKey, label: string) => ({
    role: 'button',
    tabIndex: 0,
    'aria-label': label,
    'aria-pressed': active === k,
    onClick: () => onPick(k),
    onKeyDown: (e: KeyboardEvent<SVGGElement>) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        e.stopPropagation()
        onPick(k)
      }
    },
    onFocus: () => setFocus(k),
    onBlur: () => setFocus(null),
    style: { cursor: 'pointer', outline: 'none' },
  })

  const box = (k: 'norm1' | 'attn' | 'norm2' | 'mlp', title: string, sub?: string) => {
    const b = BOX[k]
    const on = active === k
    return (
      <g {...keyProps(k, title)}>
        <rect
          x={b.x}
          y={b.y}
          width={b.w}
          height={b.h}
          rx={8}
          style={{
            fill: on ? 'var(--accent-soft)' : 'var(--surface-2)',
            stroke: on ? 'var(--accent)' : 'var(--line-strong)',
            strokeWidth: on ? 2 : 1,
            transition: 'fill 150ms, stroke 150ms',
          }}
        />
        {focus === k && <rect x={b.x - 4} y={b.y - 4} width={b.w + 8} height={b.h + 8} rx={10} style={{ fill: 'none', stroke: 'var(--accent)', strokeWidth: 2 }} />}
        <text
          x={b.x + b.w / 2}
          y={b.y + (sub ? b.h / 2 - 8 : b.h / 2)}
          textAnchor="middle"
          dominantBaseline="central"
          style={{ font: `600 ${fs(k.startsWith('norm') ? 13.5 : 15)}px "Onest Variable", sans-serif`, fill: 'var(--ink)' }}
        >
          {title}
        </text>
        {sub && (
          <text x={b.x + b.w / 2} y={b.y + b.h / 2 + 11} textAnchor="middle" dominantBaseline="central" style={{ font: `${fs(12.5)}px "Onest Variable", sans-serif`, fill: 'var(--ink-2)' }}>
            {sub}
          </text>
        )}
      </g>
    )
  }

  const addNode = (k: 'add1' | 'add2') => {
    const c = ADD[k]
    const on = active === k
    return (
      <g {...keyProps(k, 'Сложение с потоком residual')}>
        <circle cx={c.x} cy={c.y} r={16} fill="transparent" />
        <circle
          cx={c.x}
          cy={c.y}
          r={11}
          style={{ fill: on ? 'var(--accent-soft)' : 'var(--surface)', stroke: on ? 'var(--accent)' : 'var(--ink-2)', strokeWidth: on ? 2 : 1.5 }}
        />
        <path d={`M ${c.x - 5} ${c.y} H ${c.x + 5} M ${c.x} ${c.y - 5} V ${c.y + 5}`} style={{ stroke: 'var(--ink)', strokeWidth: 1.75 }} />
        {focus === k && <circle cx={c.x} cy={c.y} r={15} style={{ fill: 'none', stroke: 'var(--accent)', strokeWidth: 2 }} />}
      </g>
    )
  }

  const line = { stroke: 'var(--ink-3)', strokeWidth: 1.5, fill: 'none' } as const
  const rail = { stroke: 'var(--ink-2)', strokeWidth: 2.5, fill: 'none' } as const
  const sub = arch.moe && m.experts ? `${m.experts.active} из ${m.experts.total} экспертов` : `${fmt.int(m.hidden)} → ${fmt.int(m.intermediate)} → ${fmt.int(m.hidden)}`

  return (
    <div ref={sizeRef} className="mx-auto w-full max-w-[400px] min-w-0">
      <svg viewBox="0 0 362 292" className="block h-auto w-full select-none" role="group" aria-label="Схема блока трансформера: нажмите на часть, чтобы увидеть, что она делает">
        <defs>
          <marker id="tb-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" style={{ fill: 'var(--ink-3)' }} />
          </marker>
          <marker id="tb-arrow-rail" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" style={{ fill: 'var(--ink-2)' }} />
          </marker>
        </defs>

        {/* поток residual */}
        <g {...keyProps('in', 'Вход блока')}>
          <text x={RX + 14} y={14} style={{ font: `600 ${fs(13)}px "Onest Variable", sans-serif`, fill: active === 'in' ? 'var(--ink)' : 'var(--ink-2)' }}>
            вход: вектор токена
          </text>
          <rect x={RX - 10} y={2} width={150} height={22} fill="transparent" />
        </g>
        <path d={`M ${RX} 26 V ${ADD.add1.y - 11}`} style={rail} />
        <path d={`M ${RX} ${ADD.add1.y + 11} V ${ADD.add2.y - 11}`} style={rail} />
        <path d={`M ${RX} ${ADD.add2.y + 11} V 272`} style={rail} markerEnd="url(#tb-arrow-rail)" />
        <g {...keyProps('out', 'Выход блока')}>
          <text x={RX + 14} y={270} style={{ font: `600 ${fs(13)}px "Onest Variable", sans-serif`, fill: active === 'out' ? 'var(--ink)' : 'var(--ink-2)' }}>
            выход: к следующему блоку
          </text>
          <rect x={RX - 10} y={256} width={190} height={24} fill="transparent" />
        </g>

        {/* ветка attention */}
        <circle cx={RX} cy={48} r={3} style={{ fill: 'var(--ink-2)' }} />
        <path d={`M ${RX} 48 H ${BOX.norm1.x - 2}`} style={line} markerEnd="url(#tb-arrow)" />
        <path d={`M ${BOX.norm1.x + BOX.norm1.w} 48 H ${BOX.attn.x - 2}`} style={line} markerEnd="url(#tb-arrow)" />
        <path d={`M 261 ${BOX.attn.y + BOX.attn.h} V ${ADD.add1.y} H ${RX + 13}`} style={line} markerEnd="url(#tb-arrow)" />
        <text x={RX + 20} y={ADD.add1.y - 7} style={{ font: `${fs(12)}px "Onest Variable", sans-serif`, fill: 'var(--ink-3)' }}>
          прибавить к потоку
        </text>

        {/* ветка MLP */}
        <circle cx={RX} cy={158} r={3} style={{ fill: 'var(--ink-2)' }} />
        <path d={`M ${RX} 158 H ${BOX.norm2.x - 2}`} style={line} markerEnd="url(#tb-arrow)" />
        <path d={`M ${BOX.norm2.x + BOX.norm2.w} 158 H ${BOX.mlp.x - 2}`} style={line} markerEnd="url(#tb-arrow)" />
        <path d={`M 261 ${BOX.mlp.y + BOX.mlp.h} V ${ADD.add2.y} H ${RX + 13}`} style={line} markerEnd="url(#tb-arrow)" />
        <text x={RX + 20} y={ADD.add2.y - 7} style={{ font: `${fs(12)}px "Onest Variable", sans-serif`, fill: 'var(--ink-3)' }}>
          прибавить к потоку
        </text>

        {box('norm1', 'RMSNorm')}
        {box('attn', 'Attention', `головы: ${m.heads} Q, ${m.kvHeads} KV`)}
        {/* метки Q, K, V в углу attention */}
        {(['q', 'k', 'v'] as const).map((r, i) => (
          <rect key={r} x={BOX.attn.x + BOX.attn.w - 43 + i * 12} y={BOX.attn.y + 7} width={9} height={9} rx={2} style={{ fill: cssVar(r) }} />
        ))}
        {box('norm2', 'RMSNorm')}
        {box('mlp', arch.moe ? 'MoE' : 'MLP · SwiGLU', sub)}
        {addNode('add1')}
        {addNode('add2')}

        {/* исходный вектор ждёт у узла сложения, пока копия идёт через ветку */}
        {ghost && (
          <circle cx={ghost.x} cy={ghost.y - 19} r={5} style={{ fill: 'var(--surface)', stroke: 'var(--accent)', strokeWidth: 1.5, strokeDasharray: '2 2' }} />
        )}
        <motion.circle initial={false} animate={{ cx: dot.x, cy: dot.y }} transition={t} r={6.5} style={{ fill: 'var(--accent)', stroke: 'var(--surface)', strokeWidth: 2 }} />
      </svg>
      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-3">
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-accent" />
          текущий шаг
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="h-2.5 w-2.5 rounded-full border border-dashed border-accent" />
          исходный x ждёт сложения
        </span>
        <span className="flex items-center gap-1.5">
          {(['q', 'k', 'v'] as const).map((r) => (
            <span key={r} className="flex items-center gap-1">
              <span aria-hidden className="h-2.5 w-2.5 rounded-[2px]" style={{ background: cssVar(r) }} />
              {r === 'v' ? 'V:' : r.toUpperCase()}
            </span>
          ))}
          <span>проекции внутри attention</span>
        </span>
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------------------
 * Панель описания выбранной части.
 * ------------------------------------------------------------------------- */

function PartDetails({ arch, part, ctx }: { arch: Arch; part: PartKey; ctx: number }) {
  const m = arch.model
  const d = m.hidden
  let title: string
  let badge: ReactNode = null
  let body: ReactNode
  let tex: string | null = null
  let pinfo: Part | null = null
  let flopsNote: ReactNode = null

  switch (part) {
    case 'in':
    case 'out':
      title = part === 'in' ? 'Вход блока' : 'Выход блока'
      body =
        part === 'in' ? (
          <>
            <p>
              У каждого токена в потоке вектор из {fmt.int(d)} чисел. Для первого блока это эмбеддинг, для следующих выход предыдущего блока.
            </p>
            <p>
              В prefill через блок идут сразу все токены промпта (матрица N × {fmt.int(d)}), в decode один новый токен. Веса блока одни и те же.
            </p>
          </>
        ) : (
          <>
            <p>
              Вектор той же длины {fmt.int(d)} уходит в следующий блок. Размер потока не меняется ни в одном из {m.layers} блоков: каждый блок
              только дописывает свою поправку.
            </p>
            <p>После последнего блока стоят финальный RMSNorm и LM head, они превращают вектор в логиты (схема всей модели ниже).</p>
          </>
        )
      break
    case 'norm1':
    case 'norm2':
      title = 'RMSNorm'
      badge = <Badge>каждый токен отдельно</Badge>
      pinfo = arch.norm
      tex = 'y=\\frac{x}{\\sqrt{\\tfrac{1}{d}\\sum_i x_i^2+\\varepsilon}}\\odot g'
      body = (
        <>
          <p>
            Выравнивает масштаб вектора: делит его на среднеквадратичную величину координат и умножает на выученный вектор весов g. Следующая
            часть получает вход привычной величины, сколько бы поправок ни накопилось в потоке.
          </p>
          <p>
            Нормализуется копия, которая идёт в {part === 'norm1' ? 'attention' : 'MLP'}; сам поток не трогается. Такую схему называют
            pre-norm.
          </p>
        </>
      )
      flopsNote = 'порядка 4·d операций: квадраты, сумма, деление, умножение на g'
      break
    case 'attn':
      title = 'Attention'
      badge = <Badge>между токенами</Badge>
      pinfo = arch.attn
      tex = '\\mathrm{softmax}\\!\\left(\\frac{\\cq{Q}\\ck{K}^{\\top}}{\\sqrt{d_{head}}}+M\\right)\\cv{V}\\,W_O'
      body = (
        <>
          <p>
            Единственное место, где токен получает информацию от других токенов, причём только от предыдущих. Вектор умножается на W_Q, W_K и
            W_V, каждый токен сравнивает свой Q с K всех предыдущих и собирает взвешенную сумму их V. Механику по шагам разбирает глава <ChapterLink to="attention" />.
          </p>
          <p>
            У {m.name} {m.heads} {fmt.plural(m.heads, ['голова', 'головы', 'голов'])} Q делят {m.kvHeads}{' '}
            {fmt.plural(m.kvHeads, ['голову', 'головы', 'голов'])} K и V (GQA)
            {arch.qkNorm ? ', а Q и K перед RoPE ещё проходят свой RMSNorm' : ''}.
            Векторы K и V каждого токена кладутся в KV cache этого слоя.
          </p>
        </>
      )
      flopsNote = (
        <>
          проекции {fmt.flops(arch.attn.flopsFixed)} + Q·Kᵀ и сумма V по {fmt.int(ctx)} токенам {fmt.flops(arch.attn.flopsPerCtx * ctx)}
        </>
      )
      break
    case 'add1':
    case 'add2':
      title = 'Сложение с потоком (residual)'
      badge = <Badge>каждый токен отдельно</Badge>
      pinfo = arch.add
      tex = part === 'add1' ? 'h = x + \\mathrm{Attention}(\\mathrm{RMSNorm}(x))' : '\\mathrm{out} = h + \\mathrm{MLP}(\\mathrm{RMSNorm}(h))'
      body = (
        <>
          <p>
            Результат {part === 'add1' ? 'attention' : 'MLP'} не заменяет вектор, а прибавляется к нему. Блок пишет поправку поверх того, что уже
            есть в потоке.
          </p>
          <p>
            Благодаря этому информация из эмбеддинга и ранних слоёв доходит до последних без потерь, а сеть из десятков блоков вообще удаётся
            обучить. Параметров у сложения нет.
          </p>
        </>
      )
      flopsNote = `${fmt.int(d)} сложений`
      break
    case 'mlp':
      pinfo = arch.mlp
      if (arch.moe && m.experts && m.moeIntermediate) {
        title = 'MoE: роутер и эксперты'
        badge = <Badge>каждый токен отдельно</Badge>
        tex = '\\mathrm{MoE}(x)=\\sum_{i\\in \\mathrm{top}\\text{-}k}w_i\\,\\mathrm{Expert}_i(x)'
        body = (
          <>
            <p>
              Вместо одного большого MLP здесь {m.experts.total} маленьких экспертов (каждый это SwiGLU {fmt.int(d)} → {fmt.int(m.moeIntermediate)} →{' '}
              {fmt.int(d)}). Маленький роутер считает оценку каждого эксперта, и токен идёт только через {m.experts.active} лучших.
            </p>
            <p>
              В памяти лежат все эксперты слоя ({fmt.compact(arch.mlp.params)} параметров), а на один токен работают{' '}
              {fmt.compact(arch.mlp.activeParams)}. Подробнее ниже, в разделе про MoE.
            </p>
          </>
        )
      } else {
        title = 'MLP (SwiGLU)'
        badge = <Badge>каждый токен отдельно</Badge>
        tex = '\\mathrm{MLP}(x)=W_{down}\\big(\\mathrm{SiLU}(xW_{gate})\\odot xW_{up}\\big)'
        body = (
          <>
            <p>
              Обрабатывает каждый токен сам по себе, не глядя на соседей. Вектор расширяется до {fmt.int(m.intermediate)} чисел двумя матрицами,
              одна ветка проходит через «вентиль» SiLU и умножается на другую, затем W_down сжимает результат обратно до {fmt.int(d)}.
            </p>
            <p>
              Здесь {fmt.pct(arch.mlp.params / arch.layerParams)} параметров блока. Условно, attention решает, откуда взять информацию, а MLP
              перерабатывает её и хранит выученные «знания».
            </p>
          </>
        )
      }
      flopsNote = arch.moe ? `только ${m.experts?.active} экспертов и роутер; поэлементные операции не считаем` : 'три умножения на матрицу; SiLU и поэлементное умножение пренебрежимо малы'
      break
  }

  const flops = pinfo ? partFlops(pinfo, ctx) : 0
  return (
    <div className="grid min-w-0 content-start gap-3 rounded-panel bg-surface-2 p-4 sm:p-5">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="text-[15px] leading-snug font-semibold text-ink">{title}</h4>
        {badge}
      </div>
      <div className="grid gap-2 text-[15px] leading-[1.55] text-ink">{body}</div>
      {tex && (
        <div className="min-w-0 overflow-x-auto rounded-control bg-surface px-3 py-2">
          <Formula tex={tex} block />
        </div>
      )}
      {pinfo && pinfo.matrices.length > 0 && (
        <div className="min-w-0 overflow-x-auto">
          <table className="w-full min-w-[300px] text-left text-sm">
            <thead className="text-ink-3">
              <tr>
                <th className="py-1 pr-3 font-medium">Веса</th>
                <th className="py-1 pr-3 font-medium">Размер</th>
                <th className="py-1 text-right font-medium">Параметров</th>
              </tr>
            </thead>
            <tbody className="tabular-nums">
              {pinfo.matrices.map((mx) => (
                <tr key={mx.name} className="border-t border-line">
                  <td className="py-1 pr-3 font-mono text-[12.5px] text-ink">{mx.name}</td>
                  <td className="py-1 pr-3 text-ink-2">
                    {fmt.int(mx.rows)} × {fmt.int(mx.cols)}
                    {mx.count ? <span className="text-ink-3">, {fmt.int(mx.count)} шт.</span> : null}
                  </td>
                  <td className="py-1 text-right text-ink">{fmt.compact(mx.rows * mx.cols * (mx.count ?? 1))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {pinfo && (
        <div className="grid grid-cols-2 gap-x-6 gap-y-3 pt-1">
          <Stat
            label="Параметров"
            value={pinfo.params === 0 ? '0' : fmt.compact(pinfo.params)}
            hint={pinfo.params > 0 ? `${fmt.pct(pinfo.params / arch.layerParams)} блока${pinfo.activeParams < pinfo.params ? `, на токен ${fmt.compact(pinfo.activeParams)}` : ''}` : 'ничего не обучается'}
          />
          <Stat label="FLOPs на токен" value={fmt.flops(flops)} hint={flopsNote} />
        </div>
      )}
    </div>
  )
}
