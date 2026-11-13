import { Fragment, useId, useMemo, useState } from 'react'
import { ArrowRightIcon } from '@phosphor-icons/react'
import { Figure, Segmented, Slider, Stat, fmt, useElementSize } from '@/ui'
import { useSettings } from '@/state/settings'
import { findGpu } from '@/engine/data/gpus'
import { findModel } from '@/engine/data/models'
import { decodeStepMs, prefillMs, weightBytes } from '@/engine/perf'

/** Время маски на один запрос на шаг (notes-tokens §1.15, §2.6). */
const MASK_US = [
  { value: 40, label: '40 мкс', note: 'XGrammar, JSON Schema: меньше 40 мкс на токен (статья XGrammar)' },
  { value: 200, label: '200 мкс', note: 'XGrammar на сложных грамматиках (XML, DSL): меньше 200 мкс' },
  { value: 1500, label: '1,5 мс', note: 'полная маска llguidance для типичной схемы без оптимизации slicer' },
] as const

/** Компиляция схемы в грамматику (notes-tokens §2.6: замер XGrammar 3,8–6 мс; JSONSchemaBench: медианы XGrammar 0,11–0,30 с). */
const COMPILE = [
  { value: 0, label: 'в кэше' },
  { value: 5, label: '≈ 5 мс' },
  { value: 300, label: '≈ 0,3 с' },
] as const

const PROMPT_TOKENS = 200
/** Шагов decode на шкале: на широкой фигуре пять, на узкой три (иначе столбики сливаются). */
const DECODE_STEPS = 5
const DECODE_STEPS_COMPACT = 3
/** Уже этой ширины шкала рисуется в компактной геометрии: короткие подписи дорожек, меньше шагов. */
const COMPACT_BELOW = 560
const CONTEXT = 1024
/** vLLM: больше 128 запросов со схемой → заполнение маски пачками по 16 в min(cpu/2, 8) потоках. */
const PARALLEL_FROM = 128
const THREADS = 8

interface Bar {
  lane: 0 | 1
  start: number
  end: number
  kind: 'compile' | 'mask' | 'prefill' | 'decode' | 'stall'
  label?: string
}

export function TimelineFigure() {
  const [settings] = useSettings()
  const [maskUs, setMaskUs] = useState<number>(40)
  const [batch, setBatch] = useState(32)
  const [compile, setCompile] = useState<number>(5)
  // шкала рисуется в пикселях своей ширины: подписи всегда 12-14 px, на телефоне без прокрутки вбок
  const [sizeRef, size] = useElementSize<HTMLDivElement>()
  const width = size.width >= 1 ? Math.round(size.width) : 1000
  const compact = width < COMPACT_BELOW
  const decodeSteps = compact ? DECODE_STEPS_COMPACT : DECODE_STEPS

  const model = findModel(settings.modelId) ?? findModel('qwen3-8b')!
  const gpu = findGpu(settings.gpuId) ?? findGpu('h100-sxm')!
  // сколько GPU нужно, чтобы веса влезли (TP), иначе время шага бессмысленно
  const gpus = useMemo(() => {
    let n = 1
    while (n < 8 && weightBytes(model) > n * gpu.memGB * 1e9 * 0.8) n *= 2
    return n
  }, [model, gpu])

  const t = useMemo(() => {
    const forward = decodeStepMs(model, gpu, { batch, avgContext: CONTEXT, gpus })
    const prefill = prefillMs(model, gpu, PROMPT_TOKENS, { gpus })
    const perReq = maskUs / 1000
    const cpu = batch > PARALLEL_FROM ? (batch * perReq) / THREADS : batch * perReq
    const stall = Math.max(0, cpu - forward)
    const c = compile
    const bars: Bar[] = []
    let now = 0
    if (c > 0) {
      bars.push({ lane: 0, start: 0, end: c, kind: 'compile', label: `компиляция ${fmt.ms(c)}` })
      now = c
    }
    // prefill: маска для первого токена считается параллельно
    bars.push({ lane: 1, start: now, end: now + prefill, kind: 'prefill', label: `prefill ${fmt.ms(prefill)}` })
    bars.push({ lane: 0, start: now, end: now + perReq, kind: 'mask' })
    now += Math.max(prefill, perReq)
    const sampleAt: number[] = [now]
    for (let i = 0; i < decodeSteps; i++) {
      bars.push({ lane: 1, start: now, end: now + forward, kind: 'decode', label: i === 0 ? `forward ${fmt.ms(forward)}` : undefined })
      bars.push({ lane: 0, start: now, end: now + cpu, kind: 'mask', label: i === 0 ? `маски ${fmt.ms(cpu)}` : undefined })
      if (stall > 0) bars.push({ lane: 1, start: now + forward, end: now + cpu, kind: 'stall' })
      now += Math.max(forward, cpu)
      sampleAt.push(now)
    }
    return { forward, prefill, cpu, stall, compile: c, bars, total: now, sampleAt }
  }, [model, gpu, gpus, batch, maskUs, compile, decodeSteps])
  const parallel = batch > PARALLEL_FROM

  const maskNote = MASK_US.find((m) => m.value === maskUs)?.note

  return (
    <Figure
      id="structured-output-figure-timeline"
      title="Где работает маска: CPU параллельно с forward на GPU"
      width="wide"
      controls={
        <>
          <Segmented<number> label="Маска на один запрос" value={maskUs} onChange={setMaskUs} wrap options={MASK_US.map((m) => ({ value: m.value, label: m.label }))} />
          <Slider label="Запросов со схемой в batch" value={batch} min={1} max={512} scale="log" onChange={(v) => setBatch(Math.round(v))} format={(v) => fmt.int(Math.round(v))} />
          <Segmented<number> label="Компиляция схемы" value={compile} onChange={setCompile} wrap options={COMPILE.map((c) => ({ value: c.value, label: c.label }))} />
        </>
      }
      caption={`Попробуйте: выберите маску 1,5 мс и двигайте batch от 1 до 128. Уже при нескольких запросах CPU перестаёт успевать за GPU, и каждый шаг decode ждёт маску. Правее 128 запросов vLLM раскладывает маски на 8 потоков, и ожидание резко падает: это настоящий порог из кода vLLM. Время forward посчитано для ${model.name} на ${gpus > 1 ? `${gpus} × ` : ''}${gpu.name} (модель и GPU из симулятора). Время маски взято из статьи XGrammar и документации llguidance, время компиляции из бенчмарка JSONSchemaBench и отдельного замера XGrammar на простой схеме; откуда каждое число, написано под шкалой.`}
    >
      <div className="grid gap-5">
        <Pipeline />
        <div ref={sizeRef} className="min-w-0">
          <Timeline bars={t.bars} total={t.total} sampleAt={t.sampleAt} width={width} compact={compact} />
        </div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @xl:grid-cols-4">
          <Stat label="Forward на GPU" value={fmt.num(t.forward, 1)} unit="мс" hint={`batch ${fmt.int(batch)}`} swatch="var(--c-decode)" />
          <Stat
            label="Маски на CPU за шаг"
            value={t.cpu < 1 ? fmt.num(t.cpu, 2) : fmt.num(t.cpu, 1)}
            unit="мс"
            hint={parallel ? `в ${THREADS} потоках: vLLM включает их после ${PARALLEL_FROM} запросов` : `в одном потоке (до ${PARALLEL_FROM} запросов)`}
          />
          <Stat label="GPU ждёт маску" value={t.stall > 0 ? fmt.num(t.stall, 1) : '0'} unit="мс" hint={stallHint(t.stall, t.forward)} />
          <Stat label="Добавка к TTFT" value={fmt.num(t.compile, t.compile < 10 ? 1 : 0)} unit="мс" hint="компиляция новой схемы" />
        </div>
        {maskNote && (
          <p className="text-[13px] text-ink-3">
            Откуда {MASK_US.find((m) => m.value === maskUs)?.label}: {maskNote}. Компиляция ≈ 5 мс: отдельный замер XGrammar на простой схеме (один прогон на CPU Apple Silicon, ориентир); ≈ 0,3 с:
            верх медиан XGrammar в JSONSchemaBench.
          </p>
        )}
      </div>
    </Figure>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Схема пути: от response_format до сэмплера
 * ------------------------------------------------------------------------------------------- */

/** Насколько ожидание маски удлиняет шаг: до удвоения в процентах, дальше во сколько раз. */
function stallHint(stall: number, forward: number): string {
  if (stall <= 0) return 'маска прячется за forward'
  const ratio = stall / forward
  if (ratio <= 1) return `шаг длиннее на ${fmt.pct(ratio, 0)}`
  return `шаг в ${fmt.num((forward + stall) / forward, 1)} раза длиннее`
}

const STAGES = [
  { where: 'Frontend', what: 'response_format → параметры грамматики в запросе движку (structured_outputs в vLLM, guided_decoding в Dynamo)' },
  { where: 'Движок, CPU', what: 'схема → грамматика; результат кэшируется по строке схемы' },
  { where: 'Движок, CPU', what: 'маска на каждом шаге, пока GPU считает forward' },
  { where: 'GPU, сэмплер', what: 'logits[запрещённые] = −∞, softmax, выбор токена' },
  { where: 'Движок, CPU', what: 'accept_token: автомат грамматики сдвигается на токен' },
]

function Pipeline() {
  return (
    <ol className="grid gap-1.5 @3xl:grid-cols-[repeat(9,auto)] @3xl:items-stretch" aria-label="Путь схемы от запроса до сэмплера">
      {STAGES.map((s, i) => (
        <Fragment key={i}>
          <li className="grid content-start gap-0.5 rounded-control bg-surface-2 px-3 py-2 shadow-[inset_0_0_0_1px_var(--line)]">
            <span className="text-[12px] font-medium text-ink-3">{s.where}</span>
            <span className="text-[13px] leading-snug text-ink">{s.what}</span>
          </li>
          {i < STAGES.length - 1 && (
            <li aria-hidden className="flex items-center justify-center text-ink-3">
              <ArrowRightIcon size={16} className="rotate-90 @3xl:rotate-0" />
            </li>
          )}
        </Fragment>
      ))}
    </ol>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Временная шкала
 * ------------------------------------------------------------------------------------------- */

const LANE_H = 34
const LANE_GAP = 20
const TOP = 24

const FILL: Record<Bar['kind'], string> = {
  // компиляция: пунктирный контур на фоне поверхности, чтобы не путать её с маской (сплошная серая)
  compile: 'var(--surface)',
  mask: 'color-mix(in oklab, var(--ink-2) 34%, var(--surface))',
  prefill: 'color-mix(in oklab, var(--c-prefill) 30%, var(--surface))',
  decode: 'color-mix(in oklab, var(--c-decode) 30%, var(--surface))',
  stall: 'transparent',
}
const STROKE: Partial<Record<Bar['kind'], string>> = {
  compile: 'var(--ink-2)',
  mask: 'var(--ink-2)',
  prefill: 'var(--c-prefill)',
  decode: 'var(--c-decode)',
}
const KIND_NAME: Record<Bar['kind'], string> = {
  compile: 'компиляция схемы',
  mask: 'маска на CPU',
  prefill: 'prefill',
  decode: 'forward decode',
  stall: 'GPU ждёт маску',
}

function niceStep(total: number, maxTicks: number): number {
  const raw = total / maxTicks
  const pow = 10 ** Math.floor(Math.log10(raw))
  const m = raw / pow
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * pow
}

/** Шкала в пикселях своей ширины (viewBox = ширина), поэтому кегли подписей на экране ровно такие, как в коде. */
function Timeline({ bars, total, sampleAt, width, compact }: { bars: Bar[]; total: number; sampleAt: number[]; width: number; compact: boolean }) {
  const uid = useId().replace(/:/g, '')
  const hatch = `so-hatch-${uid}`
  const W = Math.max(300, width)
  const LEFT = compact ? 44 : 118
  const x = (ms: number) => LEFT + (ms / total) * (W - LEFT - 8)
  const laneY = (l: number) => TOP + l * (LANE_H + LANE_GAP)
  const step = niceStep(total, compact ? 4 : 6)
  const ticks: number[] = []
  for (let v = 0; v <= total + 1e-9; v += step) ticks.push(v)
  const H = TOP + 2 * LANE_H + LANE_GAP + 34
  // подписи, которые не влезли в столбик, ставим над дорожкой; пересекающиеся пропускаем (числа есть в плитках ниже)
  const outsideLabels = new Map<number, number>()
  const lastEnd = [-Infinity, -Infinity]
  bars.forEach((b, i) => {
    if (!b.label || b.kind === 'stall') return
    const x0 = x(b.start)
    const w = x(b.end) - x0
    const len = b.label.length * 6.8
    if (w > b.label.length * 7.4 + 10) return
    const at = Math.max(x0, LEFT)
    if (at < lastEnd[b.lane] + 8 || at + len > W) return
    outsideLabels.set(i, at)
    lastEnd[b.lane] = at + len
  })
  const lanes = compact ? ['CPU', 'GPU'] : ['CPU движка', 'GPU']
  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        className="block h-auto w-full"
        role="img"
        aria-label="Временная шкала: компиляция грамматики, prefill и шаги decode на GPU, маски на CPU параллельно"
      >
        <defs>
          <pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill="color-mix(in oklab, var(--bad) 10%, transparent)" />
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--bad)" strokeWidth="1.5" opacity="0.6" />
          </pattern>
        </defs>
        {lanes.map((name, l) => (
          <g key={name}>
            <rect x={LEFT} y={laneY(l)} width={W - LEFT - 8} height={LANE_H} rx={6} fill="var(--surface-2)" />
            <text x={LEFT - 10} y={laneY(l) + LANE_H / 2 + 5} textAnchor="end" className="fill-[var(--ink-2)] text-[14px]">
              {name}
            </text>
          </g>
        ))}
        {bars.map((b, i) => {
          const x0 = x(b.start)
          const w = Math.max(b.kind === 'mask' ? 2.5 : 1, x(b.end) - x0)
          const y = laneY(b.lane) + (b.kind === 'mask' ? 7 : 4)
          const h = LANE_H - (b.kind === 'mask' ? 14 : 8)
          const labelFits = b.label && w > b.label.length * 7.4 + 10
          const outside = outsideLabels.get(i)
          return (
            <g key={i}>
              <rect
                x={x0}
                y={y}
                width={w}
                height={h}
                rx={3}
                fill={b.kind === 'stall' ? `url(#${hatch})` : FILL[b.kind]}
                stroke={STROKE[b.kind]}
                strokeWidth={STROKE[b.kind] ? 1.25 : 0}
                strokeDasharray={b.kind === 'compile' ? '4 3' : undefined}
              >
                <title>{`${KIND_NAME[b.kind]}: ${fmt.ms(b.end - b.start)}`}</title>
              </rect>
              {b.label && labelFits && (
                <text x={x0 + 6} y={y + h / 2 + 4.5} className="pointer-events-none fill-[var(--ink)] text-[13px] font-medium">
                  {b.label}
                </text>
              )}
              {outside !== undefined && (
                <text x={outside} y={laneY(b.lane) - 5} className="pointer-events-none fill-[var(--ink-2)] text-[12px]">
                  {b.label}
                </text>
              )}
            </g>
          )
        })}
        {sampleAt.map((s, i) => (
          <line key={i} x1={x(s)} x2={x(s)} y1={laneY(1) - 2} y2={laneY(1) + LANE_H + 2} stroke="var(--ink)" strokeWidth={1.5} />
        ))}
        {ticks.map((v) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1={laneY(1) + LANE_H + 4} y2={laneY(1) + LANE_H + 9} stroke="var(--line-strong)" />
            <text x={x(v)} y={laneY(1) + LANE_H + 24} textAnchor="middle" className="fill-[var(--ink-3)] text-[12px] tabular-nums">
              {fmt.num(v, step < 1 ? 1 : 0)}
            </text>
          </g>
        ))}
        <text x={LEFT - 10} y={laneY(1) + LANE_H + 24} textAnchor="end" className="fill-[var(--ink-3)] text-[12px]">
          мс
        </text>
      </svg>
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1.5 text-[13px] text-ink-2">
        <Legend color="var(--surface)" stroke="var(--ink-2)" dashed label="компиляция схемы" />
        <Legend color="color-mix(in oklab, var(--ink-2) 34%, var(--surface))" stroke="var(--ink-2)" label="маска на CPU" />
        <Legend color="color-mix(in oklab, var(--c-prefill) 30%, var(--surface))" stroke="var(--c-prefill)" label="prefill" />
        <Legend color="color-mix(in oklab, var(--c-decode) 30%, var(--surface))" stroke="var(--c-decode)" label="forward decode" />
        <Legend hatch label="GPU ждёт маску" />
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="h-3.5 w-[2px] bg-ink" />
          маска применена, токен выбран
        </span>
      </div>
    </div>
  )
}

function Legend({ color, stroke, label, hatch, dashed }: { color?: string; stroke?: string; label: string; hatch?: boolean; dashed?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        aria-hidden
        className="h-2.5 w-4 rounded-[2px]"
        style={
          hatch
            ? { background: 'repeating-linear-gradient(45deg, color-mix(in oklab, var(--bad) 65%, transparent) 0 1.5px, color-mix(in oklab, var(--bad) 10%, transparent) 1.5px 4px)' }
            : dashed
              ? { background: color, border: `1.25px dashed ${stroke}` }
              : { background: color, boxShadow: stroke ? `inset 0 0 0 1.25px ${stroke}` : undefined }
        }
      />
      {label}
    </span>
  )
}
