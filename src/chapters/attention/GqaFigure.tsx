import { useMemo, useState } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { findGpu, getGpu } from '@/engine/data/gpus'
import { FEATURED_MODEL_IDS, getModel } from '@/engine/data/models'
import { useSettings } from '@/state/settings'
import { Figure, fmt, Segmented, Select, Stat, useElementSize } from '@/ui'
import { Swatch } from './bits'
import { COMPARE_CONTEXT, schemeStats, type AttnScheme, type SchemeStats } from './cost'

/** «в 4 раза», «в 8 раз», «в 32 раза». */
const times = (n: number) => `в ${fmt.num(n)} ${fmt.plural(Math.round(n), ['раз', 'раза', 'раз'])}`

/** «144 КиБ» → ['144', 'КиБ'] для Stat. */
function split(s: string): [string, string] {
  const i = s.lastIndexOf(' ')
  return i > 0 ? [s.slice(0, i), s.slice(i + 1)] : [s, '']
}

/** Внутри фигуры все байты в двоичных единицах (КиБ, ГиБ): тогда 144 КиБ × 32 768 сходится с «на запрос». */
const bin = (b: number) => fmt.bytes(b, { mode: 'binary' })
const heads = (n: number) => `${n} ${fmt.plural(n, ['голова', 'головы', 'голов'])}`
/** Подпись схемы для скринридера: «32 головы запросов и 8 пар K/V, по 4 головы запросов на пару». */
const schemeLabel = (s: SchemeStats) =>
  `${heads(s.heads)} запросов и ${fmt.count(s.kvHeads, ['пара', 'пары', 'пар'])} K/V, ${
    s.kvHeads < s.heads ? `по ${s.group} ${fmt.plural(s.group, ['голове', 'головы', 'голов'])} запросов на пару` : 'своя пара у каждой головы'
  }`

const W = 960
const H = 176
const Y_Q = 22
const Y_KV = 128
/** Уже этого схема групп переносится по строкам (иначе на телефоне квадраты голов по 3-7 px). */
const NARROW_W = 600

export function GqaFigure({ id }: { id: string }) {
  const [settings] = useSettings()
  const [modelId, setModelId] = useState(() => (FEATURED_MODEL_IDS.includes(settings.modelId) ? settings.modelId : 'qwen3-8b'))
  const [scheme, setScheme] = useState<AttnScheme>('gqa')
  const [bodyRef, { width }] = useElementSize<HTMLDivElement>()
  const model = getModel(modelId)
  const gpu = findGpu(settings.gpuId) ?? getGpu('h100-sxm')
  const s = useMemo(() => schemeStats(model, gpu, scheme), [model, gpu, scheme])
  const real = useMemo(() => schemeStats(model, gpu, 'gqa'), [model, gpu])
  const narrow = width > 0 && width < NARROW_W

  const [perTok, perTokUnit] = split(bin(s.bytesPerToken))
  const [perSeq, perSeqUnit] = split(bin(s.bytesPerSeq))
  const gpuLabel = `${gpu.name}${s.gpus > 1 ? ` × ${s.gpus}` : ''}`
  const replicated =
    s.replication > 1
      ? s.kvHeads === 1
        ? `; одна пара K/V не делится между ${s.gpus} GPU и копируется на каждый, поэтому запросов ${times(s.replication)} меньше`
        : `; пар K/V меньше, чем GPU, и каждую копируют на ${s.replication} GPU, поэтому запросов ${times(s.replication)} меньше`
      : ''

  const note =
    scheme === 'mha' ? (
      <>
        <b className="font-semibold">MHA</b> (multi-head attention): у каждой из {s.heads} голов свои K и V. Так устроены ранние модели, например OPT-13B из
        статьи про PagedAttention. Для {model.name} это {times(s.bytesPerToken / real.bytesPerToken)} больше кэша, чем на самом деле.
      </>
    ) : scheme === 'gqa' ? (
      <>
        <b className="font-semibold">GQA</b> (grouped-query attention), как в {model.name}: {heads(s.heads)} запросов делят {s.kvHeads} пар K/V, по {s.group} на
        пару. Так устроены Qwen3, Llama 3 и большинство открытых моделей: качество почти как у MHA, а кэш {times(s.vsMha)} меньше.
      </>
    ) : (
      <>
        <b className="font-semibold">MQA</b> (multi-query attention): одна пара K/V на весь слой, кэш {times(s.vsMha)} меньше, чем у MHA. Но качество модели
        обычно хуже, поэтому новые модели чаще выбирают GQA.
      </>
    )

  const groupNote =
    s.kvHeads === s.heads ? 'Рамка обводит голову Q и её собственную пару.' : s.kvHeads === 1 ? 'Рамка одна на весь слой.' : 'Рамка обводит одну группу.'

  return (
    <Figure
      id={id}
      title="Сколько пар K/V хранить: MHA, GQA, MQA"
      width="wide"
      controlsPosition="top"
      controls={
        <>
          <Segmented<AttnScheme>
            label="Схема attention"
            value={scheme}
            onChange={setScheme}
            options={[
              { value: 'mha', label: 'MHA' },
              { value: 'gqa', label: 'GQA' },
              { value: 'mqa', label: 'MQA' },
            ]}
          />
          <Select
            label="Модель"
            value={modelId}
            onChange={setModelId}
            options={FEATURED_MODEL_IDS.map((mid) => {
              const m = getModel(mid)
              return { value: mid, label: `${m.name}: ${m.heads} Q, ${m.kvHeads} KV` }
            })}
          />
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> переключите GQA на MHA и обратно. KV на токен вырастет {times(real.vsMha)}, а запросов на{' '}
          {fmt.int(COMPARE_CONTEXT)} токенов в той же памяти поместится во столько же раз меньше. Числа посчитаны по config модели, байты в BF16.
        </>
      }
    >
      <div ref={bodyRef} className="grid min-w-0 gap-4">
        <div className="grid min-w-0 gap-1.5">
          <div className="flex items-center gap-2 text-sm text-ink-2">
            <Swatch role="q" /> Головы запросов (Q) в одном слое: {s.heads}
          </div>
          {narrow ? <GroupsNarrow s={s} width={width} /> : <GroupsWide s={s} />}
          <div className="grid grid-cols-[auto_minmax(0,1fr)] items-start gap-x-2 text-sm text-ink-2">
            <span className="mt-[5px] inline-flex gap-1">
              <Swatch role="k" />
              <Swatch role="v" />
            </span>
            <span>
              Пары K/V (это они лежат в KV cache): {s.kvHeads}
              {s.kvHeads < s.heads ? `, по ${s.group} ${fmt.plural(s.group, ['голове', 'головы', 'голов'])} запросов на пару` : ', своя у каждой головы'}.{' '}
              {groupNote}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @[720px]:grid-cols-4">
          <Stat label="Пар K/V в слое" value={String(s.kvHeads)} hint={`на ${s.heads} ${fmt.plural(s.heads, ['голову', 'головы', 'голов'])} запросов`} />
          <Stat
            label="KV на токен"
            value={perTok}
            unit={perTokUnit}
            hint={`2 × ${fmt.count(model.layers, ['слой', 'слоя', 'слоёв'])} × ${fmt.count(s.kvHeads, ['пара', 'пары', 'пар'])} × ${model.headDim} чисел × 2 байта`}
          />
          <Stat
            label="KV на запрос"
            value={perSeq}
            unit={perSeqUnit}
            hint={`${fmt.tokens(COMPARE_CONTEXT)}, ${
              scheme === 'gqa'
                ? 'как в модели'
                : s.bytesPerToken > real.bytesPerToken
                  ? `${times(s.bytesPerToken / real.bytesPerToken)} больше, чем в модели`
                  : `${times(real.bytesPerToken / s.bytesPerToken)} меньше, чем в модели`
            }`}
          />
          <Stat
            label="Запросов в памяти"
            value={String(s.maxSeqs)}
            hint={`по ${fmt.int(COMPARE_CONTEXT)} токенов, ${gpuLabel}, под KV ≈ ${bin(s.kvBudget)}${replicated}`}
          />
        </div>
        <p className="max-w-[72ch] text-[15px] leading-relaxed text-ink-2">{note}</p>
      </div>
    </Figure>
  )
}

/** Широкий экран: головы Q в ряд, линии сходятся к парам K/V своей группы. Каждая группа в своей рамке. */
function GroupsWide({ s }: { s: SchemeStats }) {
  const reduce = useReducedMotion() ?? false
  const slotQ = W / s.heads
  const sq = Math.max(3, Math.min(22, slotQ - 4))
  const slotKv = W / s.kvHeads
  const pairW = Math.max(4, Math.min(52, slotKv - 6))
  const skv = (pairW - 2) / 2
  const kvX = (g: number) => g * slotKv + slotKv / 2
  const inset = Math.min(3, slotKv / 10)
  const move = reduce ? { duration: 0 } : { duration: 0.45, ease: [0.2, 0, 0, 1] as const }
  const fade = reduce ? { duration: 0 } : { duration: 0.3 }

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="block h-auto w-full"
      role="img"
      aria-label={schemeLabel(s)}
    >
      <AnimatePresence initial={false}>
        {s.groups.map((g) => (
          <motion.rect
            key={`box-${s.kvHeads}-${g.kvHead}`}
            y={8}
            height={H - 14}
            rx={8}
            initial={{ opacity: 0, x: g.kvHead * slotKv + inset, width: slotKv - 2 * inset }}
            animate={{ opacity: 1, x: g.kvHead * slotKv + inset, width: slotKv - 2 * inset }}
            exit={{ opacity: 0 }}
            transition={fade}
            style={{ fill: 'var(--surface-2)' }}
          />
        ))}
      </AnimatePresence>
      {s.groups.flatMap((g) =>
        g.queryHeads.map((q) => {
          const xq = q * slotQ + slotQ / 2
          return (
            <motion.line
              key={`l-${q}`}
              initial={false}
              animate={{ x1: xq, x2: kvX(g.kvHead) }}
              y1={Y_Q + sq}
              y2={Y_KV}
              transition={move}
              style={{ stroke: 'var(--ink-3)', strokeOpacity: 0.55, strokeWidth: s.heads > 32 ? 0.75 : 1 }}
            />
          )
        }),
      )}
      {Array.from({ length: s.heads }, (_, q) => (
        <rect key={`q-${q}`} x={q * slotQ + (slotQ - sq) / 2} y={Y_Q} width={sq} height={sq} rx={Math.min(4, sq / 4)} style={{ fill: 'var(--c-q)' }} />
      ))}
      <AnimatePresence initial={false}>
        {s.groups.map((g) => (
          <motion.g key={`kv-${g.kvHead}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={fade}>
            <motion.rect
              initial={false}
              animate={{ x: kvX(g.kvHead) - pairW / 2, width: skv, height: Math.max(skv, 18) }}
              y={Y_KV}
              rx={Math.min(4, skv / 4)}
              transition={move}
              style={{ fill: 'var(--c-k)' }}
            />
            <motion.rect
              initial={false}
              animate={{ x: kvX(g.kvHead) - pairW / 2 + skv + 2, width: skv, height: Math.max(skv, 18) }}
              y={Y_KV}
              rx={Math.min(4, skv / 4)}
              transition={move}
              style={{ fill: 'var(--c-v)' }}
            />
          </motion.g>
        ))}
      </AnimatePresence>
    </svg>
  )
}

/**
 * Узкий экран: группы переносятся по строкам, у каждой своя рамка, её головы Q (квадраты от 10 px)
 * и пара K/V под ними. Так схема читается и на телефоне, и для моделей с 64 головами.
 */
function GroupsNarrow({ s, width }: { s: SchemeStats; width: number }) {
  const SQ = 10
  const QGAP = 3
  const PADG = 5
  const GAPG = 4
  const KV_W = 10
  const KV_H = 14
  const LINES = 14
  const pairW = 2 * KV_W + 2
  const one = s.kvHeads === 1
  // сколько квадратов Q в строке внутри группы: у MQA на всю ширину, иначе не больше 4
  const qCols = one ? Math.max(1, Math.floor((width - 2 * PADG + QGAP) / (SQ + QGAP))) : Math.min(s.group, 4)
  const qRows = Math.ceil(s.group / qCols)
  const innerW = Math.max(qCols * SQ + (qCols - 1) * QGAP, pairW)
  const boxW = one ? width : innerW + 2 * PADG
  // сколько групп в строке: сколько влезает, но строки ровные (8 групп: 4 + 4, а не 5 + 3)
  const maxPerRow = one ? 1 : Math.max(1, Math.floor((width + GAPG) / (boxW + GAPG)))
  const perRow = Math.ceil(s.kvHeads / Math.ceil(s.kvHeads / maxPerRow))
  const rows = Math.ceil(s.kvHeads / perRow)
  const boxH = PADG + qRows * SQ + (qRows - 1) * QGAP + LINES + KV_H + PADG
  const height = rows * boxH + (rows - 1) * GAPG

  return (
    <svg
      width={width}
      height={height}
      className="block"
      role="img"
      aria-label={schemeLabel(s)}
    >
      {s.groups.map((g) => {
        const gi = g.kvHead
        const bx = (gi % perRow) * (boxW + GAPG)
        const by = Math.floor(gi / perRow) * (boxH + GAPG)
        const cx = bx + boxW / 2
        const qBlockW = Math.min(s.group, qCols) * SQ + (Math.min(s.group, qCols) - 1) * QGAP
        const qx0 = cx - qBlockW / 2
        const kvTop = by + boxH - PADG - KV_H
        return (
          <g key={`n-${s.kvHeads}-${gi}`}>
            <rect x={bx} y={by} width={boxW} height={boxH} rx={8} style={{ fill: 'var(--surface-2)' }} />
            {g.queryHeads.map((q, k) => {
              const col = k % qCols
              const row = Math.floor(k / qCols)
              const x = qx0 + col * (SQ + QGAP)
              const y = by + PADG + row * (SQ + QGAP)
              return (
                <g key={`q-${q}`}>
                  <line
                    x1={x + SQ / 2}
                    y1={y + SQ}
                    x2={cx}
                    y2={kvTop}
                    style={{ stroke: 'var(--ink-3)', strokeOpacity: 0.55, strokeWidth: s.group > 16 ? 0.6 : 0.9 }}
                  />
                  <rect x={x} y={y} width={SQ} height={SQ} rx={2.5} style={{ fill: 'var(--c-q)' }} />
                </g>
              )
            })}
            <rect x={cx - pairW / 2} y={kvTop} width={KV_W} height={KV_H} rx={2.5} style={{ fill: 'var(--c-k)' }} />
            <rect x={cx - pairW / 2 + KV_W + 2} y={kvTop} width={KV_W} height={KV_H} rx={2.5} style={{ fill: 'var(--c-v)' }} />
          </g>
        )
      })}
    </svg>
  )
}
