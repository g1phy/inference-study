import type { ReactNode } from 'react'
import { Figure, cssVar } from '@/ui'

/**
 * Зачем нужен CUDA graph: две временные дорожки одного шага decode (условный масштаб, notes-hardware-software-engines.md,
 * пример 3.9: без графа шаг ≈ max(работа GPU, число ядер × время запуска), с графом ≈ работа GPU + один запуск).
 * Вёрстка на HTML, а не SVG: полосы сжимаются по ширине, а подписи остаются читаемыми на телефоне.
 */
const K = 12 // ядер в шаге (в настоящем шаге их сотни)
const LAUNCH = 40 // CPU тратит на подготовку и запуск одного ядра (условные единицы)
const CPU_BUSY = LAUNCH - 4 // полоса CPU на одно ядро: CPU занят почти без перерыва
const KERNEL = 24 // GPU считает одно ядро
const START = CPU_BUSY + 2 // ядро стартует, когда его запуск дошёл до GPU; CPU в это время уже готовит следующее
const GRAPH_LAUNCH = 12
const END_EAGER = (K - 1) * LAUNCH + START + KERNEL
const END_GRAPH = GRAPH_LAUNCH + K * KERNEL
const SCALE = END_EAGER + 12

const pct = (u: number) => `${(u / SCALE) * 100}%`

export function CudaGraphFigure() {
  const eagerKernels = Array.from({ length: K }, (_, i) => i * LAUNCH + START)
  const graphKernels = Array.from({ length: K }, (_, i) => GRAPH_LAUNCH + i * KERNEL)
  return (
    <Figure
      id="engines-figure-cudagraph"
      title="Один шаг decode без CUDA graph и с ним"
      width="wide"
      caption="Масштаб условный: 12 ядер вместо сотен. Граф записывают заранее под набор размеров batch (1, 2, 4, 8 и дальше), и реальный batch дополняется до ближайшего записанного размера."
    >
      <div
        role="img"
        aria-label="Без графа CPU всё время занят: запуск каждого ядра по отдельности длится дольше, чем GPU его считает, и между ядрами GPU простаивает. С CUDA graph вся цепочка ядер запускается одной командой и идёт подряд, CPU свободен, шаг заметно короче."
        className="grid min-w-0 gap-6"
      >
        <Panel
          title="Без графа: CPU запускает ядра по одному"
          note="CPU всё время занят: подготовка и запуск одного ядра (Python, PyTorch, драйвер) длится дольше, чем GPU считает само ядро, поэтому между ядрами GPU ждёт."
        >
          <Track label="CPU">
            {eagerKernels.map((_, i) => (
              <Bar key={i} at={i * LAUNCH} len={CPU_BUSY} cpu />
            ))}
          </Track>
          <Track label="GPU" end={END_EAGER}>
            {eagerKernels.map((x, i) => (
              <Bar key={i} at={x} len={KERNEL} />
            ))}
            <span
              aria-hidden
              className="absolute top-full mt-0.5 -translate-x-1/2 text-[12px] whitespace-nowrap text-ink-2"
              style={{ left: pct(eagerKernels[2] + KERNEL + (LAUNCH - KERNEL) / 2) }}
            >
              ↑ GPU простаивает
            </span>
          </Track>
        </Panel>
        <Panel
          title="CUDA graph: вся цепочка одной командой"
          note="Ядра идут подряд без пауз: одна команда запускает всю цепочку, а CPU свободен и уже готовит следующий шаг. Разница тем заметнее, чем меньше модель и batch."
        >
          <Track label="CPU">
            <Bar at={0} len={GRAPH_LAUNCH - 2} cpu />
          </Track>
          <Track label="GPU" end={END_GRAPH}>
            {graphKernels.map((x, i) => (
              <Bar key={i} at={x} len={KERNEL - 1.5} />
            ))}
          </Track>
        </Panel>
      </div>
    </Figure>
  )
}

function Panel({ title, note, children }: { title: string; note: string; children: ReactNode }) {
  return (
    <div className="grid min-w-0 gap-2.5">
      <div className="text-[14px] font-semibold text-ink">{title}</div>
      <div className="grid min-w-0 gap-2 pb-5">{children}</div>
      <p className="text-[13px] leading-snug text-ink-3">{note}</p>
    </div>
  )
}

function Track({ label, end, children }: { label: string; end?: number; children: ReactNode }) {
  return (
    <div className="grid min-w-0 grid-cols-[2.75rem_minmax(0,1fr)] items-center gap-2">
      <span className="font-mono text-[12px] text-ink-3">{label}</span>
      <div className="relative h-5 min-w-0 border-b border-line">
        {children}
        {end !== undefined && (
          <span aria-hidden className="absolute -top-1 -bottom-1 border-l border-dashed border-ink-3" style={{ left: pct(end) }}>
            <span className="absolute top-full right-1 mt-0.5 text-[11.5px] whitespace-nowrap text-ink-3">конец шага</span>
          </span>
        )}
      </div>
    </div>
  )
}

function Bar({ at, len, cpu }: { at: number; len: number; cpu?: boolean }) {
  return (
    <span
      aria-hidden
      className="absolute top-0.5 bottom-0.5 rounded-[3px]"
      style={{ left: pct(at), width: pct(len), background: cpu ? 'var(--ink-3)' : cssVar('decode') }}
    />
  )
}
