/**
 * Инспектор стадии: заголовок с реальным временем, фраза для новичка, данные стадии и «Подробнее».
 * Следует за плеером; клик по узлу схемы закрепляет стадию (повторный клик или кнопка возвращают слежение).
 */
import type { ReactNode } from 'react'
import { ArrowLeftIcon, PushPinIcon } from '@phosphor-icons/react'
import { Button, ChapterLink, fmt, linkClass, Skeleton, TokenStripSkeleton } from '@/ui'
import { getChapter } from '@/content/chapters'
import { returnFromSimulator, useSimulatorReturn } from '@/state/nav'
import type { FlowRunEx } from '@/engine/flow'
import type { FlowSettings, StageId } from '@/engine/types'
import { STAGES, stageChapters } from './copy'
import { firstStepOf, INSPECTOR_HEADING_ID, neighborSteps, neighborStepsText, nodeStatus, tokenAtStep } from './model'
import { ClientStage, FrontendStage, PreprocessStage, RouteStage } from './stages-in'
import { DecodeStage, PrefillStage, TransferStage } from './stages-gpu'
import { PostprocessStage, StreamStage } from './stages-out'

/** Реальное время стадии одной строкой (с пометкой «условно» у констант). */
function stageTiming(run: FlowRunEx, stage: StageId): string {
  const s = run.stages?.find((x) => x.stage === stage)
  if (!s) return stage === 'transfer' ? 'только в дезагрегированном режиме' : ''
  const approx = s.simulated ? ', условная константа' : ''
  switch (stage) {
    case 'client':
      return `${fmt.ms(s.ms)} по сети до сервера${approx}`
    case 'route':
      return s.count > 1 ? `${s.count} шага роутера, всего ${fmt.ms(s.ms)}${approx}` : `${fmt.ms(s.ms)}${approx}`
    case 'prefill':
      return `${fmt.ms(s.ms)}, посчитано по модели и GPU`
    case 'transfer':
      return `${fmt.ms(s.ms)}, посчитано по объёму KV и каналу`
    case 'decode': {
      const steps = `${s.count} ${fmt.plural(s.count, ['шаг', 'шага', 'шагов'])}, всего ${fmt.ms(s.ms)}`
      // (X-sim-r2-4) шаг с prefill соседа тянет среднее вверх: показываем обычный шаг и долгий отдельно
      const nb = neighborSteps(run)
      return nb ? `${steps}; обычный шаг ${fmt.ms(nb.plainMs)}, ${neighborStepsText(nb)}` : `${steps}; шаг ≈ ${fmt.ms(s.ms / Math.max(1, s.count))}`
    }
    case 'postprocess':
      return `${fmt.ms(s.ms / Math.max(1, s.count))} на токен${approx}`
    case 'stream':
      return `${fmt.ms(s.ms / Math.max(1, s.count))} по сети на чанк${approx}`
    default:
      return `${fmt.ms(s.ms)}${approx}`
  }
}

export interface InspectorProps {
  run: FlowRunEx | null
  loading: boolean
  stage: StageId
  pinned: boolean
  step: number
  playing: boolean
  /** Длительность текущего шага при текущей скорости, мс. */
  stepMs: number
  onUnpin: () => void
  onSelect: (stage: StageId) => void
  patch: (p: Partial<FlowSettings>) => void
}

/**
 * (X-site-r2-2) Видимый путь назад для тех, кто пришёл из главы («Где это в симуляторе», чип «Ваш запрос»,
 * «изменить запрос») и не думает о кнопке «Назад» браузера. returnFromSimulator() делает history.back():
 * читатель возвращается к абзацу и ссылке, с которых ушёл. При любой другой навигации from сбрасывается в null.
 */
function ReturnToChapter() {
  const from = useSimulatorReturn()
  if (!from) return null
  const ch = getChapter(from)
  return (
    <button
      type="button"
      onClick={returnFromSimulator}
      className={`inline-flex items-center gap-1.5 justify-self-start text-left text-[13px] leading-5 ${linkClass}`}
    >
      <ArrowLeftIcon size={16} aria-hidden className="shrink-0" />
      Вернуться к главе {ch.number} «{ch.nav}»
    </button>
  )
}

export function Inspector({ run, loading, stage, pinned, step, playing, stepMs, onUnpin, onSelect, patch }: InspectorProps) {
  const c = STAGES[stage]
  const Icon = c.icon
  const chapters = stageChapters(stage)
  return (
    <section aria-label={`Инспектор стадии ${c.title}`} className="@container grid min-w-0 content-start gap-4">
      <ReturnToChapter />
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
        <div className="flex min-w-0 items-center gap-3">
          <span className="grid grid-cols-[minmax(0,1fr)] h-9 w-9 shrink-0 place-items-center rounded-control bg-surface-2 text-ink-2 shadow-[inset_0_0_0_1px_var(--line)]">
            <Icon size={20} aria-hidden />
          </span>
          <div className="grid grid-cols-[minmax(0,1fr)] min-w-0">
            {/* tabIndex −1: сюда переводит фокус ссылка «Где это в симуляторе» из главы */}
            <h3 id={INSPECTOR_HEADING_ID} tabIndex={-1} className="scroll-mt-20 text-[15px] leading-6 font-semibold text-ink outline-none">
              {c.title}
            </h3>
            <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-[12.5px] leading-5 text-ink-3">
              <span className="rounded-chip bg-surface-2 px-1.5 text-ink-2 shadow-[inset_0_0_0_1px_var(--line)]">
                <span className="sr-only">Где работает: </span>
                {c.where}
              </span>
              <span className="tabular-nums">{run ? stageTiming(run, stage) : ' '}</span>
            </div>
          </div>
        </div>
        {pinned ? (
          <div className="flex items-center gap-1">
            <span className="inline-flex h-8 items-center gap-1.5 rounded-chip bg-accent-soft px-2 text-[12.5px] font-medium text-accent-hover">
              <PushPinIcon size={16} aria-hidden />
              Закреплено
            </span>
            <Button size="sm" variant="ghost" onClick={onUnpin}>
              Следовать за плеером
            </Button>
          </div>
        ) : (
          <span className="pt-1 text-[12.5px] text-ink-3">следует за плеером</span>
        )}
      </div>

      <p className="max-w-[72ch] text-[14.5px] leading-relaxed text-ink-2">{c.help}</p>

      <div className="min-w-0">{run ? <StageBody run={run} stage={stage} step={step} playing={playing} stepMs={stepMs} onSelect={onSelect} patch={patch} /> : <BodySkeleton stage={stage} loading={loading} />}</div>

      {chapters.length > 0 && (
        <p className="border-t border-line pt-3 text-[13px] text-ink-2">
          Подробнее:{' '}
          {chapters.map((id, i) => (
            <span key={id}>
              {i > 0 && ', '}
              <ChapterLink to={id} />
            </span>
          ))}
        </p>
      )}
    </section>
  )
}

function StageBody({
  run,
  stage,
  step,
  playing,
  stepMs,
  onSelect,
  patch,
}: {
  run: FlowRunEx
  stage: StageId
  step: number
  playing: boolean
  stepMs: number
  onSelect: (stage: StageId) => void
  patch: (p: Partial<FlowSettings>) => void
}): ReactNode {
  const pb = run.playback ?? []
  const N = run.decode.length
  const k = tokenAtStep(pb, step, N)
  switch (stage) {
    case 'client':
      return <ClientStage run={run} onSelect={onSelect} />
    case 'frontend':
      return <FrontendStage run={run} />
    case 'preprocess':
      return <PreprocessStage run={run} />
    case 'route':
      return <RouteStage run={run} />
    case 'prefill': {
      const st = nodeStatus(pb, step, 'prefill')
      const phase = st === 'pending' ? 'pending' : st === 'active' && playing ? 'running' : 'done'
      return <PrefillStage run={run} phase={phase} durationMs={stepMs} />
    }
    case 'transfer':
      return <TransferStage run={run} patch={patch} />
    case 'decode':
      return <DecodeStage run={run} k={Math.max(0, k)} reached={k >= 0} />
    case 'postprocess':
      return <PostprocessStage run={run} k={Math.max(0, k)} reached={k >= 0} />
    case 'stream':
      return <StreamStage run={run} sseCount={pb[step]?.sseCount ?? 0} />
  }
}

function BodySkeleton({ stage, loading }: { stage: StageId; loading: boolean }) {
  if (!loading) return <p className="text-[13.5px] text-ink-3">Не удалось посчитать прогон.</p>
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-4">
      {stage === 'preprocess' ? <TokenStripSkeleton count={36} /> : <Skeleton lines={4} height={12} label="Загружаем токенизатор" />}
      <Skeleton height={120} radius={12} label="" />
    </div>
  )
}

/** Шаг плеера, на котором стадия впервые активна (для перехода по ссылке «Где это в симуляторе»). */
export function jumpStepFor(run: FlowRunEx, stage: StageId): number {
  return firstStepOf(run.playback ?? [], stage)
}
