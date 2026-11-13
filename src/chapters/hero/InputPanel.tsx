/**
 * Панель ввода симулятора: вкладки сценариев, сообщение пользователя, system prompt, переключатели,
 * деплой, prefix cache и «Дополнительно» (модель, GPU, temperature, max_tokens). Всё, кроме max_tokens, пишется
 * в общие настройки сайта. Текст полей живёт в локальном черновике и уходит в настройки с паузой в наборе:
 * печать не перерисовывает весь симулятор на каждую букву.
 */
import { memo, startTransition, useCallback, useEffect, useId, useRef, useState } from 'react'
import { CaretDownIcon, InfoIcon, PaperPlaneRightIcon } from '@phosphor-icons/react'
import { Button, cn, fmt, Kbd, Segmented, Select, Slider, Tabs, TextArea, Toggle, useElementSize } from '@/ui'
import { SETTINGS_LIMITS } from '@/state/settings'
import { DEFAULT_MAX_TOKENS, DEFAULT_SIM_MODEL_ID, NEIGHBOR_MIN_COMPLETION_TOKENS, SIM_GPUS, SIM_MODEL_IDS, SIM_MODELS, resolveSimGpu } from '@/engine/flow'
import { isLongDoc, LONG_DOC_TAB, longDocPatch, resolveScenario, SCENARIOS, scenarioPatch } from '@/engine/scenarios'
import type { FlowSettings, ScenarioId } from '@/engine/types'
import { excerpt } from './model'

export interface InputPanelProps {
  settings: FlowSettings
  patch: (p: Partial<FlowSettings>) => void
  /** Правка запроса: p, что поменялось (без p правка текста или max_tokens). Симулятор решает, ставить ли плеер на паузу. */
  onEdit: (p?: Partial<FlowSettings>) => void
  onSend: () => void
  ready: boolean
  /** max_tokens запроса (не хранится в настройках сайта). */
  maxTokens: number
  onMaxTokens: (n: number) => void
}

type TabId = ScenarioId | typeof LONG_DOC_TAB.id

/** Вкладки: пресеты, «Длинный документ» перед своим текстом. */
const TABS: Array<{ id: TabId; label: string }> = [
  ...SCENARIOS.filter((s) => s.id !== 'custom').map((s) => ({ id: s.id as TabId, label: s.label })),
  { id: LONG_DOC_TAB.id, label: LONG_DOC_TAB.label },
  { id: 'custom', label: SCENARIOS.find((s) => s.id === 'custom')!.label },
]

/** Пауза в наборе, после которой текст уходит в настройки (и в прогон). */
const TEXT_COMMIT_MS = 200
const MAX_TOKENS_OPTIONS = [DEFAULT_MAX_TOKENS, 128, 32]

function Disclosure({
  open,
  onToggle,
  label,
  summary,
  controls,
}: {
  open: boolean
  onToggle: () => void
  label: string
  summary?: string
  controls: string
}) {
  return (
    <button
      type="button"
      aria-expanded={open}
      aria-controls={controls}
      onClick={onToggle}
      className="-ml-1.5 inline-flex min-h-8 max-w-full min-w-0 items-center gap-1.5 rounded-control px-1.5 py-1 text-left text-[13.5px] text-ink-2 transition-colors hover:bg-surface-3 hover:text-ink"
    >
      <CaretDownIcon size={16} aria-hidden className={cn('shrink-0 transition-transform duration-150', !open && '-rotate-90')} />
      <span className="shrink-0 font-medium text-ink">{label}</span>
      {summary && !open && <span className="min-w-0 truncate text-ink-3">{summary}</span>}
    </button>
  )
}

/**
 * Черновик текстового поля: печать меняет только локальное состояние, в настройки текст уходит после паузы
 * (в startTransition) или сразу по flush (отправка, уход из поля). Внешняя правка (вкладка, другая глава) заменяет черновик.
 */
function useDraft(value: string, commit: (v: string, urgent: boolean) => void): [string, (v: string) => void, () => void] {
  const [draft, setDraft] = useState(value)
  const committed = useRef(value)
  const timer = useRef<number | undefined>(undefined)
  const latest = useRef(draft)
  latest.current = draft
  useEffect(() => {
    if (value === committed.current) return
    // настройки поменял кто-то другой: черновик устарел
    window.clearTimeout(timer.current)
    timer.current = undefined
    committed.current = value
    setDraft(value)
  }, [value])
  useEffect(() => () => window.clearTimeout(timer.current), [])
  const push = useCallback(
    (v: string, urgent: boolean) => {
      window.clearTimeout(timer.current)
      timer.current = undefined
      committed.current = v
      commit(v, urgent)
    },
    [commit],
  )
  const change = useCallback(
    (v: string) => {
      setDraft(v)
      window.clearTimeout(timer.current)
      timer.current = window.setTimeout(() => push(v, false), TEXT_COMMIT_MS)
    },
    [push],
  )
  const flush = useCallback(() => {
    if (timer.current !== undefined) push(latest.current, true)
  }, [push])
  return [draft, change, flush]
}

/** Подсказка, если переключатели расходятся с описанием вкладки. */
function toggleMismatch(tab: TabId, s: FlowSettings): string | null {
  if (tab === 'tool' && !s.tools) return 'Инструмент выключен: модель не сможет вызвать get_weather и ответит текстом. Включите «Инструмент get_weather».'
  if (tab === 'json' && !s.json) return 'JSON-схема выключена: без маски модель отвечает болтовнёй и блоком кода. Включите «Ответ по JSON-схеме».'
  if (tab === 'thinking' && !s.thinking) return 'Рассуждение выключено: шаблон допишет пустой блок <think></think>, и модель ответит сразу.'
  return null
}

export const InputPanel = memo(function InputPanel({ settings, patch, onEdit, onSend, ready, maxTokens, onMaxTokens }: InputPanelProps) {
  const [systemOpen, setSystemOpen] = useState(false)
  const [advOpen, setAdvOpen] = useState(false)
  const sysId = useId()
  const advId = useId()
  const [sizeRef, { width }] = useElementSize<HTMLDivElement>()
  const longDoc = isLongDoc(settings)
  const tab: TabId = longDoc ? LONG_DOC_TAB.id : settings.scenarioId
  const description = tab === LONG_DOC_TAB.id ? LONG_DOC_TAB.description : (SCENARIOS.find((s) => s.id === tab) ?? SCENARIOS[1]).description
  const mismatch = toggleMismatch(tab, settings)

  const settingsRef = useRef(settings)
  settingsRef.current = settings

  const edit = (p: Partial<FlowSettings>) => {
    patch(p)
    onEdit(p)
  }

  const commitUser = useCallback(
    (userText: string, urgent: boolean) => {
      const p = { userText, scenarioId: resolveScenario({ scenarioId: settingsRef.current.scenarioId, userText }) }
      if (urgent) patch(p)
      else startTransition(() => patch(p))
    },
    [patch],
  )
  const commitSystem = useCallback(
    (systemText: string, urgent: boolean) => {
      if (urgent) patch({ systemText })
      else startTransition(() => patch({ systemText }))
    },
    [patch],
  )
  const [userDraft, changeUser, flushUser] = useDraft(settings.userText, commitUser)
  const [sysDraft, changeSys, flushSys] = useDraft(settings.systemText, commitSystem)

  const send = () => {
    flushUser()
    flushSys()
    onSend()
  }

  const onTab = (id: TabId) => {
    // черновики сбрасываются эффектом useDraft: вкладка сама пишет текст в настройки.
    // «Свой текст» после «Длинного документа»: регламент остаётся в system prompt, вопрос пишет читатель
    // (иначе текст вкладки совпадал бы с документом и вкладка не переключилась бы).
    if (id === 'custom' && longDoc) edit({ scenarioId: 'custom', userText: '' })
    else edit(id === LONG_DOC_TAB.id ? longDocPatch() : scenarioPatch(id))
    if (id === 'custom') {
      // свой текст: фокус в поле и выделение, можно сразу печатать
      window.setTimeout(() => {
        const el = document.getElementById('prompt-input') as HTMLTextAreaElement | null
        el?.focus({ preventScroll: true })
        el?.select()
      }, 0)
    }
  }

  const modelValue = SIM_MODEL_IDS.includes(settings.modelId) ? settings.modelId : DEFAULT_SIM_MODEL_ID
  const gpuOptions = SIM_GPUS.slice(0, 3).map((g) => ({ value: g.id, label: g.name }))
  const gpu = resolveSimGpu(settings.gpuId).gpu
  if (!gpuOptions.some((o) => o.value === gpu.id)) gpuOptions.push({ value: gpu.id, label: gpu.name })
  const gpuValue = gpu.id
  const modelName = SIM_MODELS.find((m) => m.id === modelValue)?.name ?? modelValue
  const advSummary = `${modelName}, ${gpu.name}, temperature ${fmt.num(settings.temperature, 2)}${maxTokens !== DEFAULT_MAX_TOKENS ? `, max_tokens ${maxTokens}` : ''}`
  const sysSummary = sysDraft.trim() ? `«${excerpt(sysDraft, 40)}»` : 'пустой'
  // вкладки в один ряд, пока помещаются; в узкой колонке сегменты с переносом (иначе «Свой текст» уезжает за край)
  const wideTabs = width === 0 || width >= 820

  return (
    <div ref={sizeRef} className="grid grid-cols-[minmax(0,1fr)] gap-4">
      <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
        {wideTabs ? (
          // описание сценария и есть панель вкладки: у role="tab" должна быть своя role="tabpanel"
          <Tabs tabs={TABS} value={tab} onChange={onTab} aria-label="Сценарий запроса" size="sm" panelClassName="pt-2">
            {() => <p className="text-[13px] leading-snug text-ink-2">{description}</p>}
          </Tabs>
        ) : (
          <>
            <Segmented options={TABS.map((t) => ({ value: t.id, label: t.label }))} value={tab} onChange={onTab} size="sm" wrap aria-label="Сценарий запроса" />
            <p className="text-[13px] leading-snug text-ink-2">{description}</p>
          </>
        )}
        {mismatch && (
          <p className="flex gap-2 text-[13px] leading-snug text-ink">
            <InfoIcon size={16} aria-hidden className="mt-0.5 shrink-0 text-ink-2" />
            {mismatch}
          </p>
        )}
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)] gap-x-8 gap-y-5 @3xl:grid-cols-2 @5xl:grid-cols-[minmax(0,1.4fr)_minmax(0,0.85fr)_minmax(0,1fr)]">
        <div className="grid grid-cols-[minmax(0,1fr)] min-w-0 content-start gap-2 @3xl:col-span-2 @5xl:col-span-1">
          {/* уход из поля сразу отдаёт текст в настройки (onBlur в React ловит и потерю фокуса детьми) */}
          <div className="grid grid-cols-[minmax(0,1fr)] min-w-0" onBlur={flushUser}>
            <TextArea
              id="prompt-input"
              label="Сообщение пользователя"
              value={userDraft}
              onChange={(v) => {
                changeUser(v)
                onEdit()
              }}
              rows={2}
              autoGrow
              maxLength={SETTINGS_LIMITS.maxTextLength}
              placeholder="Напишите свой вопрос модели"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault()
                  send()
                }
              }}
              hint={
                <>
                  Текст общий для всего сайта: главы про шаблон, токены и attention берут его отсюда. <Kbd>Ctrl</Kbd> или <Kbd>⌘</Kbd> + <Kbd>Enter</Kbd>{' '}
                  отправляет.
                </>
              }
            />
          </div>
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0 flex-1">
              <Disclosure open={systemOpen} onToggle={() => setSystemOpen((v) => !v)} label="System prompt" summary={sysSummary} controls={sysId} />
            </div>
            <Button variant="primary" icon={PaperPlaneRightIcon} onClick={send} disabled={!ready} className="shrink-0">
              Отправить
            </Button>
          </div>
          <div id={sysId} hidden={!systemOpen} onBlur={flushSys}>
            {systemOpen && (
              <TextArea
                label="System prompt"
                value={sysDraft}
                onChange={(v) => {
                  changeSys(v)
                  onEdit()
                }}
                rows={2}
                autoGrow
                maxLength={SETTINGS_LIMITS.maxTextLength}
                hint="Пустой system prompt в запрос не попадёт. Своего system prompt по умолчанию у шаблона Qwen3 нет: без system-сообщения системный блок появится, только если включены tools (в нём описание инструментов)."
              />
            )}
          </div>
        </div>

        <div className="grid grid-cols-[minmax(0,1fr)] min-w-0 content-start gap-1" role="group" aria-label="Что умеет запрос">
          <Toggle label="Рассуждение" hint="enable_thinking: сначала <think>" checked={settings.thinking} onChange={(thinking) => edit({ thinking })} />
          <Toggle label="Инструмент get_weather" hint="описание tools уходит в промпт" checked={settings.tools} onChange={(tools) => edit({ tools })} />
          <Toggle label="Ответ по JSON-схеме" hint="response_format: маска на decode" checked={settings.json} onChange={(json) => edit({ json })} />
        </div>

        <div className="grid grid-cols-[minmax(0,1fr)] min-w-0 content-start gap-3">
          <div className="grid grid-cols-[minmax(0,1fr)] gap-1">
            <Segmented
              label="Деплой"
              size="sm"
              wrap
              value={settings.deployment}
              onChange={(deployment) => edit({ deployment })}
              options={[
                { value: 'agg', label: 'Агрегированный' },
                { value: 'disagg', label: 'Дезагрегированный' },
              ]}
            />
            <p className="text-[12.5px] leading-snug text-ink-3">
              {settings.deployment === 'agg'
                ? `Prefill и decode на одном воркере: длинный prefill соседа встаёт в шаг decode вашего запроса, и ITL скачет (в симуляторе в ответах от ${NEIGHBOR_MIN_COMPLETION_TOKENS} токенов). В дезагрегированном у каждой фазы свои GPU, а KV едет по сети через NIXL.`
                : 'У prefill и decode свои GPU, KV едет по сети через NIXL. TTFT выше, зато prefill соседей не тормозит ваш decode: шаги ровные.'}
            </p>
          </div>
          <div className="grid grid-cols-[minmax(0,1fr)] gap-1">
            <Segmented
              label="Prefix cache"
              size="sm"
              wrap
              value={settings.cacheWarm ? 'warm' : 'cold'}
              onChange={(v) => edit({ cacheWarm: v === 'warm' })}
              options={[
                { value: 'cold', label: 'Холодный' },
                { value: 'warm', label: 'Тёплый' },
              ]}
            />
            <p className="text-[12.5px] leading-snug text-ink-3">
              {settings.cacheWarm
                ? 'Тёплый: недавно второй воркер обслужил запрос с тем же началом (system prompt и первые слова сообщения), и его блоки KV ещё лежат в кэше.'
                : 'Холодный: начала вашего промпта нет в кэше ни у одного воркера, prefill посчитает всё.'}
            </p>
          </div>
          <div className="grid grid-cols-[minmax(0,1fr)] min-w-0 gap-3">
            <Disclosure open={advOpen} onToggle={() => setAdvOpen((v) => !v)} label="Дополнительно" summary={advSummary} controls={advId} />
            <div id={advId} hidden={!advOpen} className="grid grid-cols-[minmax(0,1fr)] gap-4">
              {advOpen && (
                <>
                  <Select
                    label="Модель"
                    size="sm"
                    value={modelValue}
                    onChange={(modelId) => edit({ modelId })}
                    options={SIM_MODELS.map((m) => ({ value: m.id, label: m.name }))}
                  />
                  <Select label="GPU" size="sm" value={gpuValue} onChange={(gpuId) => edit({ gpuId })} options={gpuOptions} />
                  <Slider
                    label="Temperature"
                    value={settings.temperature}
                    min={0}
                    max={2}
                    step={0.05}
                    onChange={(temperature) => edit({ temperature })}
                    format={(v) => fmt.num(v, 2)}
                  />
                  <Select
                    label="max_tokens"
                    size="sm"
                    value={String(maxTokens)}
                    onChange={(v) => {
                      onMaxTokens(Number(v))
                      onEdit()
                    }}
                    options={MAX_TOKENS_OPTIONS.map((n) => ({ value: String(n), label: fmt.int(n) }))}
                    hint="Лимит длины ответа: если ответ длиннее, генерация обрывается и finish_reason становится length."
                  />
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
})
