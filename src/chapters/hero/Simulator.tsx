/**
 * Симулятор «Путь запроса»: ввод → схема пайплайна с плеером → инспектор стадии и ответ клиенту →
 * таймлайн с реальными мс → метрики. Прогон buildRun считается на каждое изменение настроек
 * (текст с небольшой задержкой), плеер проигрывает его шаги с замедлением.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useReducedMotion } from 'motion/react'
import { ArrowCounterClockwiseIcon, WarningIcon } from '@phosphor-icons/react'
import { Figure, fmt, Player, Skeleton, usePlayer } from '@/ui'
import { getSettings, selectStage, useStageSelection } from '@/state/settings'
import { resolveSimGpu, resolveSimModel, type FlowRunEx } from '@/engine/flow'
import type { FlowSettings, StageId } from '@/engine/types'
import { ClientPanel } from './ClientPanel'
import { InputPanel } from './InputPanel'
import { Inspector } from './Inspector'
import { Metrics } from './Metrics'
import { PipelineDiagram } from './PipelineDiagram'
import { Timeline } from './Timeline'
import { firstDecodeStep, firstStepOf, INSPECTOR_HEADING_ID, INSPECTOR_ID, samePlaybackShape, stepDurations, visibleWs } from './model'
import type { FlowState } from './useFlowRun'

/** id блока «схема + плеер»: к нему прокручивают «Отправить» и кнопка в hero. */
export const PIPELINE_ID = 'simulator-pipeline'

/** Показать схему с плеером, если она не видна целиком (на телефоне и низком экране «Отправить» стоит выше неё). */
export function revealPipeline(reduce: boolean): void {
  const el = document.getElementById(PIPELINE_ID)
  if (!el) return
  const r = el.getBoundingClientRect()
  // на планшете и телефоне сверху липкая панель навигации
  const top = window.matchMedia('(min-width: 75rem)').matches ? 0 : 56
  if (r.top >= top && r.bottom <= window.innerHeight) return
  el.scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' })
}

/**
 * (X-site-r2-1) Ссылка «Где это в симуляторе» из главы: показать выбранную стадию и перевести на неё фокус.
 * Если схема с плеером и начало инспектора помещаются в окно, прокручиваем к схеме (видно, какой узел горит),
 * иначе (телефон, низкий экран) прямо к инспектору. Фокус на заголовке инспектора: Tab дальше идёт по симулятору,
 * а не возвращает к ссылке в главе.
 */
function revealInspector(reduce: boolean): void {
  requestAnimationFrame(() => {
    const box = document.getElementById(INSPECTOR_ID)
    if (!box) return
    const pipe = document.getElementById(PIPELINE_ID)
    // на планшете и телефоне сверху липкая панель навигации (56 px)
    const bar = window.matchMedia('(min-width: 75rem)').matches ? 0 : 56
    const pageY = (el: HTMLElement) => el.getBoundingClientRect().top + window.scrollY
    const boxTop = pageY(box)
    const pipeTop = pipe ? pageY(pipe) : boxTop
    const room = window.innerHeight - bar
    // от схемы, если под ней остаётся хотя бы 320 px инспектора
    const top = boxTop - pipeTop + 320 <= room - 16 ? pipeTop - bar - 8 : boxTop - bar - 8
    window.scrollTo({ top: Math.max(0, top), behavior: reduce ? 'auto' : 'smooth' })
    document.getElementById(INSPECTOR_HEADING_ID)?.focus({ preventScroll: true })
  })
}

/** Настройки, от которых меняются только числа прогона, а не его шаги: плеер остаётся на месте. */
const KEEP_STEP_KEYS: ReadonlySet<keyof FlowSettings> = new Set<keyof FlowSettings>(['temperature', 'modelId', 'gpuId', 'cacheWarm'])

/** Настройки прогона совпадают с текущими в сторе (снимок useSettings и объект стора бывают разными объектами). */
function sameSettings(a: FlowSettings, b: FlowSettings): boolean {
  return (Object.keys(b) as Array<keyof FlowSettings>).every((k) => Object.is(a[k], b[k]))
}

/** Тот же запрос: текст, сценарий, переключатели, деплой и max_tokens. Модель, GPU, temperature и prefix cache не в счёт. */
function sameRequest(a: FlowRunEx, b: FlowRunEx): boolean {
  const x = a.settings
  const y = b.settings
  return (
    x.userText === y.userText &&
    x.systemText === y.systemText &&
    x.scenarioId === y.scenarioId &&
    x.thinking === y.thinking &&
    x.tools === y.tools &&
    x.json === y.json &&
    x.deployment === y.deployment &&
    a.maxTokens === b.maxTokens
  )
}

export function Simulator({ flow, sendSignal }: { flow: FlowState; sendSignal: number }) {
  const { run, status, error, settings, patch, flush, maxTokens, setMaxTokens } = flow
  const reduce = useReducedMotion() ?? false
  const playback = run?.playback ?? []
  const durations = useMemo(() => (run ? stepDurations(run) : []), [run])
  const durRef = useRef(durations)
  durRef.current = durations
  const player = usePlayer(Math.max(1, playback.length), { intervalMs: (s) => durRef.current[s] ?? 400 })
  const { step, playing, speed, setStep, setPlaying } = player
  /** Прогон пересчитался после правки, а плеер вернулся в начало: подсказываем нажать «Отправить». */
  const [stale, setStale] = useState(false)

  const selection = useStageSelection()
  const pinned = selection.stage
  const cur = playback[step]
  const followStage: StageId = cur?.stage ?? 'client'
  const stage: StageId = pinned ?? followStage

  // правка запроса: пауза (прогон пересчитается, шаги могут сдвинуться). temperature, модель, GPU и prefix cache
  // шагов плеера не двигают: проигрывание идёт дальше (X-sim-r2-5)
  const onEdit = useCallback(
    (p?: Partial<FlowSettings>) => {
      if (p && (Object.keys(p) as Array<keyof FlowSettings>).every((k) => KEEP_STEP_KEYS.has(k))) return
      setPlaying(false)
    },
    [setPlaying],
  )

  // «Отправить»: свежий прогон, инспектор следует за плеером, проигрывание с начала.
  // Если словарь ещё грузится, проигрывание начнётся, как только появится прогон.
  const pendingPlay = useRef(false)
  /** «Отправить» применил отложенный ввод: следующий прогон играем с начала, а не сбрасываем. */
  const sending = useRef(false)
  const hasRun = run !== null
  const send = useCallback(() => {
    const changed = flush()
    if (selection.stage) selectStage(null, { scroll: false })
    setStale(false)
    if (!hasRun) {
      pendingPlay.current = true
      return
    }
    sending.current = changed
    setStep(0)
    setPlaying(true)
    revealPipeline(reduce)
  }, [flush, selection.stage, setStep, setPlaying, hasRun, reduce])
  useEffect(() => {
    if (!run || !pendingPlay.current) return
    pendingPlay.current = false
    setStep(0)
    setPlaying(true)
  }, [run, setStep, setPlaying])

  // кнопка «Отправить запрос» в hero
  const lastSend = useRef(sendSignal)
  useEffect(() => {
    if (sendSignal === lastSend.current) return
    lastSend.current = sendSignal
    send()
  }, [sendSignal, send])

  // ссылка «Где это в симуляторе» из главы: плеер встаёт на первый шаг стадии
  const lastNonce = useRef(selection.nonce)
  const pendingJump = useRef<StageId | null>(null)
  useEffect(() => {
    if (selection.nonce === lastNonce.current) return
    lastNonce.current = selection.nonce
    if (!selection.fromLink || !selection.stage) return
    if (selection.stage === 'transfer' && settings.deployment === 'agg') patch({ deployment: 'disagg' })
    pendingJump.current = selection.stage
  }, [selection, settings.deployment, patch])

  // Прогон сменился (правка запроса, вкладка, деплой): шаг прошлого прогона в новом ничего не значит.
  // Плеер стоял в конце: показываем конец нового прогона. Иначе возвращаемся в начало и ждём «Отправить».
  const prev = useRef<{ run: FlowRunEx | null; atEnd: boolean; step: number }>({ run: null, atEnd: false, step: 0 })
  useEffect(() => {
    const p = prev.current
    if (!run || p.run === run || p.run === null) return
    if (pendingJump.current) return
    if (sending.current) {
      sending.current = false
      setStep(0)
      setPlaying(true)
      setStale(false)
      return
    }
    if (pendingPlay.current) return
    // (X-sim-r2-5) temperature, модель, GPU, prefix cache: запрос и шаги плеера те же, шаг остаётся на месте
    if (sameRequest(p.run, run) && samePlaybackShape(p.run.playback ?? [], run.playback ?? [])) return
    if (p.atEnd) {
      setStep((run.playback?.length ?? 1) - 1)
      setStale(false)
    } else {
      setPlaying(false)
      setStep(0)
      setStale(p.step > 0)
    }
  }, [run, setStep, setPlaying])
  useEffect(() => {
    prev.current = { run, atEnd: step >= (run?.playback?.length ?? 1) - 1 && step > 0, step }
  }, [run, step])

  useEffect(() => {
    const target = pendingJump.current
    if (!target || !run) return
    // (X-sim-r2-7) ждём прогон по настройкам после prepare() ссылки и переключения деплоя для NIXL:
    // иначе шаг посчитался бы по старому прогону, а новый сбросил бы плеер в начало
    if (!sameSettings(run.settings, getSettings())) return
    // (X-sim-r2-6) decode: первый шаг под маской или первый настоящий шаг decode, а не токен из конца prefill
    const i = target === 'decode' ? firstDecodeStep(run) : firstStepOf(run.playback ?? [], target)
    pendingJump.current = null
    if (i < 0) return
    setPlaying(false)
    setStale(false)
    setStep(i)
    revealInspector(reduce)
  }, [run, selection.nonce, setPlaying, setStep, reduce])

  const onSelect = useCallback(
    (s: StageId) => {
      selectStage(pinned === s ? null : s, { scroll: false })
    },
    [pinned],
  )
  const onUnpin = useCallback(() => selectStage(null, { scroll: false }), [])
  const onSeek = useCallback(
    (s: number) => {
      setStale(false)
      setPlaying(false)
      setStep(s)
    },
    [setPlaying, setStep],
  )
  // подсказка «Запрос изменился» уходит, как только читатель сам запустил плеер или сдвинул шаг
  const { onStep: bindStep, onPlayingChange: bindPlaying } = player.bind
  const playerBind = useMemo(
    () => ({
      ...player.bind,
      onStep: (s: number) => {
        setStale(false)
        bindStep(s)
      },
      onPlayingChange: (p: boolean) => {
        if (p) setStale(false)
        bindPlaying(p)
      },
    }),
    [player.bind, bindStep, bindPlaying],
  )

  const stepMs = durations[step] ?? 400
  const labels = useMemo(() => {
    if (!run) return []
    const N = run.decode.length
    return playback.map((p) =>
      p.kind === 'token' && p.token !== undefined
        ? `Токен ${fmt.int(p.token + 1)} из ${fmt.int(N)}: «${run.decode[p.token].token.special ? run.decode[p.token].token.raw : visibleWs(run.decode[p.token].token.text)}»`
        : p.label,
    )
  }, [run, playback])
  const loading = status === 'loading' || status === 'idle'
  const deployment = settings.deployment

  return (
    <Figure
      id="simulator"
      width="full"
      padded={false}
      title="Путь запроса: симулятор"
      actions={<RunBadge run={run} />}
      caption={
        <>
          Попробуйте: нажмите «Отправить», потом включите тёплый prefix cache и отправьте ещё раз. Роутер выберет другой воркер, и prefill посчитает
          меньше токенов, но на коротком вопросе TTFT почти не изменится: такой prefill упирается в чтение весов, а не в число токенов. Теперь откройте
          вкладку «Длинный документ» и повторите: холодный prefill упрётся в вычисления, а тёплый кэш сократит TTFT больше чем вдвое (так на Qwen3-8B и
          H100 в агрегированном режиме, с которых начинается симулятор; у MoE-моделей prefill и здесь упирается в чтение весов).
        </>
      }
    >
      <div className="px-4 pt-3 pb-5 sm:px-5">
        <InputPanel settings={settings} patch={patch} onEdit={onEdit} onSend={send} ready={!!run} maxTokens={maxTokens} onMaxTokens={setMaxTokens} />
      </div>

      <div id={PIPELINE_ID} className="grid grid-cols-[minmax(0,1fr)] gap-3 border-t border-line px-4 py-4 sm:px-5">
        <PipelineDiagram run={run} deployment={deployment} step={step} speed={speed} stepMs={stepMs} pinned={pinned} onSelect={onSelect} />
        {status === 'error' ? (
          <p className="flex items-center gap-2 text-[13.5px] text-ink">
            <WarningIcon size={18} aria-hidden className="text-warn" />
            Не удалось загрузить токенизатор: {error ?? 'ошибка сети'}. Обновите страницу.
          </p>
        ) : (
          <div className="grid grid-cols-[minmax(0,1fr)] gap-1.5">
            {run ? (
              <Player {...playerBind} labels={labels} speeds={[0.5, 1, 2, 4]} aria-label="Проигрывание пути запроса" />
            ) : (
              <Skeleton height={40} radius={8} label="Плеер появится после загрузки токенизатора" />
            )}
            {stale && run && (
              <p className="flex items-center gap-2 text-[13px] leading-snug text-ink">
                <ArrowCounterClockwiseIcon size={16} aria-hidden className="shrink-0 text-ink-2" />
                Запрос изменился: нажмите «Отправить», чтобы проиграть новый прогон.
              </p>
            )}
            <p className="text-[12.5px] leading-snug text-ink-3">
              {run ? (
                <>
                  Момент на реальной шкале: <span className="text-ink-2 tabular-nums">t = {fmt.ms(cur?.tMs ?? 0)}</span> от отправки. Анимация замедлена:
                  реальные стадии длятся от долей миллисекунды (CPU) до миллисекунд (сеть, prefill, шаг decode), весь ответ занимает{' '}
                  {fmt.ms(run.metrics.e2eMs)}. Щелчок по стадии закрепляет её в инспекторе.
                </>
              ) : (
                `Загружаем словарь Qwen3 (${fmt.int(151669)} токенов), чтобы токенизировать ваш текст по-настоящему.`
              )}
            </p>
          </div>
        )}
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)] border-t border-line @3xl:grid-cols-[minmax(0,1fr)_minmax(280px,340px)]">
        <div id={INSPECTOR_ID} className="min-w-0 px-4 py-5 sm:px-5 @3xl:min-h-[460px]">
          <Inspector
            run={run}
            loading={loading}
            stage={stage}
            pinned={pinned !== null}
            step={step}
            playing={playing}
            stepMs={stepMs / Math.max(0.25, speed)}
            onUnpin={onUnpin}
            onSelect={onSelect}
            patch={patch}
          />
        </div>
        <div className="min-w-0 border-t border-line px-4 py-5 sm:px-5 @3xl:border-t-0 @3xl:border-l">
          <div className="@3xl:sticky @3xl:top-[4.5rem]">
            <ClientPanel run={run} step={step} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)] gap-2 border-t border-line px-4 py-5 sm:px-5">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
          <h3 className="text-[15px] leading-6 font-semibold text-ink">Где уходит время</h3>
          <span className="text-[12.5px] text-ink-3">реальные мс, ось разорвана на TTFT; щелчок перематывает плеер</span>
        </div>
        {run ? (
          <Timeline run={run} step={step} stepMs={stepMs} speed={speed} playing={playing} onSeek={onSeek} />
        ) : (
          <Skeleton height={220} radius={8} label="Таймлайн появится после загрузки токенизатора" />
        )}
      </div>

      <div className="border-t border-line px-4 py-5 sm:px-5">
        <Metrics run={run} />
      </div>
    </Figure>
  )
}

function RunBadge({ run }: { run: FlowRunEx | null }) {
  if (!run) return null
  const gpu = resolveSimGpu(run.gpuId ?? '').gpu.name
  const workers = run.cluster?.workers.length ?? 0
  const tp = run.gpus ?? 1
  return (
    <span className="text-[12.5px] leading-5 text-ink-3">
      {resolveSimModel(run.modelId ?? '').model.name}, {tp}× {gpu} на воркер, {workers} {fmt.plural(workers, ['воркер', 'воркера', 'воркеров'])}
    </span>
  )
}
