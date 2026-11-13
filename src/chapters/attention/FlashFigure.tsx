import { useRef, useState } from 'react'
import { cn, cssVar, Figure, fmt, mix, Segmented, Stat, TipRow, useTip } from '@/ui'
import { Swatch } from './bits'
import { flashTraffic } from './cost'

const LENGTHS = [1024, 4096, 16384, 32768] as const
type Len = (typeof LENGTHS)[number]

interface Seg {
  key: string
  label: string
  bytes: number
  fill: string
  short?: string
  hatch?: boolean
}

/**
 * Трафик HBM одной головы attention (BF16): обычная реализация пишет и читает матрицы S и P размером n × n,
 * FlashAttention считает их тайлами в SRAM и в HBM не пишет (идеализация: только Q, K, V и O).
 */
export function FlashFigure({ id, headDim }: { id: string; headDim: number }) {
  const [n, setN] = useState<Len>(4096)
  const t = flashTraffic(n, headDim)
  const nd = 2 * n * headDim
  const nn = 2 * n * n
  const standard: Seg[] = [
    { key: 'q', label: 'читаем Q', bytes: nd, fill: cssVar('q') },
    { key: 'k', label: 'читаем K', bytes: nd, fill: cssVar('k') },
    { key: 'sw', label: 'пишем S = Q·Kᵀ', bytes: nn, fill: mix('ink-3', 22), short: 'S', hatch: true },
    { key: 'sr', label: 'читаем S', bytes: nn, fill: mix('ink-3', 22), short: 'S' },
    { key: 'pw', label: 'пишем P = softmax(S)', bytes: nn, fill: mix('ink-3', 38), short: 'P', hatch: true },
    { key: 'pr', label: 'читаем P', bytes: nn, fill: mix('ink-3', 38), short: 'P' },
    { key: 'v', label: 'читаем V', bytes: nd, fill: cssVar('v') },
    { key: 'o', label: 'пишем выход O', bytes: nd, fill: mix('ink-3', 60), hatch: true },
  ]
  const flash: Seg[] = [
    { key: 'q', label: 'читаем Q', bytes: nd, fill: cssVar('q') },
    { key: 'k', label: 'читаем K', bytes: nd, fill: cssVar('k') },
    { key: 'v', label: 'читаем V', bytes: nd, fill: cssVar('v') },
    { key: 'o', label: 'пишем выход O', bytes: nd, fill: mix('ink-3', 60), hatch: true },
  ]
  const share = (4 * nn) / t.standardBytes

  return (
    <Figure
      id={id}
      title="Трафик памяти HBM одной головы: обычный attention и FlashAttention"
      width="narrow"
      controls={
        <Segmented<Len>
          label="Длина последовательности n"
          value={n}
          onChange={setN}
          options={LENGTHS.map((v) => ({ value: v, label: fmt.int(v) }))}
        />
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> увеличьте n. Матрицы S и P растут как n², а Q, K, V и O только как n, поэтому
          с длиной доля промежуточных матриц в трафике растёт. FlashAttention показан идеализированно: настоящие ядра перечитывают тайлы K и V, но матрица n
          × n в HBM не попадает. Голова d = {headDim}, BF16.
        </>
      }
    >
      <div className="grid min-w-0 gap-4">
        <Bar title="Обычный attention" segs={standard} total={t.standardBytes} scale={t.standardBytes} />
        <Bar title="FlashAttention" segs={flash} total={t.flashBytes} scale={t.standardBytes} />
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[13px] text-ink-2">
          <span className="inline-flex items-center gap-1.5">
            <Swatch role="q" />
            <Swatch role="k" />
            <Swatch role="v" /> Q, K, V
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-2.5 w-2.5 rounded-[3px]" style={{ background: mix('ink-3', 60) }} /> выход O
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-2.5 w-2.5 rounded-[3px]" style={{ background: mix('ink-3', 22) }} />
            <span aria-hidden className="h-2.5 w-2.5 rounded-[3px]" style={{ background: mix('ink-3', 38) }} /> S и P (n × n)
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-2.5 w-2.5 rounded-[3px] bg-surface-3 bg-hatch" /> штриховка: запись в HBM, без неё чтение
          </span>
        </div>
        <p className="text-sm text-ink-2">
          В обычной реализации {fmt.pct(share)} байт приходится на промежуточные S и P: их записывают в HBM на одном шаге и сразу читают на следующем.
        </p>
        <div className="grid grid-cols-2 gap-x-6 gap-y-3 @[560px]:grid-cols-3">
          <Stat label="Матрица n × n одной головы" value={fmt.bytes(t.scoreMatrixBytes, { mode: 'binary' })} hint="S или P в BF16" />
          <Stat label="Трафик обычного" value={fmt.bytes(t.standardBytes, { mode: 'binary' })} hint="8n² + 8nd байт" />
          <Stat label="Трафик FlashAttention" value={fmt.bytes(t.flashBytes, { mode: 'binary' })} hint={`в ${fmt.num(t.standardBytes / t.flashBytes, 0)} ${fmt.plural(Math.round(t.standardBytes / t.flashBytes), ['раз', 'раза', 'раз'])} меньше`} />
        </div>
      </div>
    </Figure>
  )
}

function Bar({ title, segs, total, scale }: { title: string; segs: Seg[]; total: number; scale: number }) {
  const ref = useRef<HTMLDivElement | null>(null)
  const tip = useTip({ scopeEl: ref.current })
  return (
    <div className="grid min-w-0 gap-1.5">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="font-medium text-ink">{title}</span>
        <span className="text-ink-2 tabular-nums">{fmt.bytes(total, { mode: 'binary' })}</span>
      </div>
      <div
        ref={ref}
        className="flex h-8 w-full overflow-hidden rounded-control"
        role="img"
        aria-label={`${title}: ${segs.map((s) => `${s.label} ${fmt.bytes(s.bytes, { mode: 'binary' })}`).join(', ')}`}
        onPointerLeave={(e) => e.pointerType === 'mouse' && tip.hide()}
      >
        <div className="flex h-full" style={{ width: `${(total / scale) * 100}%`, minWidth: 6 }}>
          {segs.map((s) => (
            <div
              key={s.key}
              className={cn('flex h-full min-w-[1.5px] items-center justify-center border-r border-surface text-[12px] font-medium text-ink last:border-r-0', s.hatch && 'bg-hatch')}
              style={{ width: `${(s.bytes / total) * 100}%`, backgroundColor: s.fill }}
              onPointerEnter={(e) => tip.show(e.currentTarget.getBoundingClientRect(), <TipRow value={fmt.bytes(s.bytes, { mode: 'binary' })} label={s.label} />)}
              onPointerDown={(e) => tip.show(e.currentTarget.getBoundingClientRect(), <TipRow value={fmt.bytes(s.bytes, { mode: 'binary' })} label={s.label} />)}
            >
              {s.short && (s.bytes / scale) * 100 > 6 ? s.short : null}
            </div>
          ))}
        </div>
      </div>
      {tip.node}
    </div>
  )
}
