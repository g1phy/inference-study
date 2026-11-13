import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ArrowCounterClockwiseIcon, PlusIcon, ScissorsIcon, WarningIcon, XIcon } from '@phosphor-icons/react'
import { Button, Figure, FigureLive, Segmented, Stat, Term, TextArea, Toggle, cn, fmt, useInView } from '@/ui'
import { renderTemplate } from '@/engine/templates'
import type { ChatMessage, ModelFamily, Tokenizer } from '@/engine/types'
import { SETTINGS_LIMITS } from '@/state/settings'
import { WEATHER_TOOL, requestMessages } from '../request/requestModel'
import { PromptLegend, PromptView } from './PromptView'
import { countPrompt, type PromptCount } from './promptTokens'
import { useO200k } from './useO200k'

interface EditableMessage extends ChatMessage {
  id: number
}

const ASSISTANT_REPLY = {
  reasoning: 'Нужно объяснить без формул и коротко.',
  content: 'KV cache хранит ключи и значения уже прочитанных токенов, чтобы не считать их заново на каждом шаге.',
}
const USER_FOLLOWUP = 'А сколько памяти он занимает?'
const TOOL_RESULT = '{"temp_c": 7, "sky": "облачно"}'
const MAX_MESSAGES = 10

let nextId = 1
const mk = (m: ChatMessage): EditableMessage => ({ ...m, id: nextId++ })

function defaults(systemText: string, userText: string): EditableMessage[] {
  const out: EditableMessage[] = []
  if (systemText.trim()) out.push(mk({ role: 'system', content: systemText }))
  out.push(mk({ role: 'user', content: userText }))
  return out
}

/** Сообщения уже те, что дал бы defaults(): пересоздавать их (и поля ввода с ними) незачем. */
function sameAsDefaults(ms: EditableMessage[], systemText: string, userText: string): boolean {
  const want = requestMessages({ systemText, userText })
  return ms.length === want.length && ms.every((m, i) => m.role === want[i].role && m.content === want[i].content)
}

const FAMILY_OPTIONS: Array<{ value: ModelFamily; label: string }> = [
  { value: 'qwen3', label: 'Qwen3' },
  { value: 'llama3', label: 'Llama 3.1' },
  { value: 'deepseek-r1', label: 'DeepSeek-R1' },
  { value: 'gpt-oss', label: 'gpt-oss' },
]

/** Короткие заметки о формате (notes-tokens §1.3), без тире-разделителей. */
const FAMILY_NOTES: Record<ModelFamily, ReactNode> = {
  qwen3: (
    <>
      ChatML: каждая реплика обёрнута в <code>{'<|im_start|>'}</code>роль … <code>{'<|im_end|>'}</code>, BOS нет. Рассуждение
      модель пишет в <code>{'<think>…</think>'}</code>, вызов инструмента идёт JSON внутри <code>{'<tool_call>'}</code>.
    </>
  ),
  llama3: (
    <>
      Шаблон сам ставит <code>{'<|begin_of_text|>'}</code> и дописывает строки «Cutting Knowledge Date» и «Today Date», которых вы
      не писали. Инструменты уходят в первое сообщение user, режима рассуждения нет.
    </>
  ),
  'deepseek-r1': (
    <>
      Системный текст стоит сразу после BOS без всякого тега. Шаблон сам открывает <code>{'<think>'}</code>, поэтому модель
      начинает ответ прямо с рассуждения. Описание tools этот шаблон не выводит. В спецтокенах стоят полноширинная черта{' '}
      <code>｜</code> и символ <code>▁</code>, а не ASCII.
    </>
  ),
  'gpt-oss': (
    <>
      <Term t="Harmony">Harmony</Term>: сообщения <code>{'<|start|>'}</code>роль<code>{'<|message|>'}</code>…
      <code>{'<|end|>'}</code>, рассуждение в канале analysis, ответ в канале final. Шаблон сам пишет system-блок с сегодняшней
      датой и уровнем рассуждения, а ваш system переносит в developer.
    </>
  ),
}

/** Почему рассуждение прошлого ответа не попало в промпт (notes-tokens §3.3). */
const DROPPED_REASON: Record<ModelFamily, string> = {
  qwen3: 'шаблон Qwen3 вырезает его у ответов до последнего вопроса пользователя.',
  llama3: 'у шаблона Llama 3.1 нет места для рассуждения.',
  'deepseek-r1': 'шаблон DeepSeek-R1 оставляет от прошлых ответов только текст после </think>.',
  'gpt-oss': 'шаблон gpt-oss выбрасывает канал analysis у ходов, которые закончились ответом.',
}

function thinkingHint(f: ModelFamily, on: boolean): string {
  if (f === 'qwen3')
    return on
      ? 'enable_thinking=true (по умолчанию): шаблон ничего не дописывает, модель сама начнёт ответ с <think> и будет рассуждать'
      : 'enable_thinking=false: пустой <think></think> в конце'
  if (f === 'gpt-oss') return on ? 'строка «Reasoning: medium» в system' : 'строка «Reasoning: low» в system'
  if (f === 'deepseek-r1') return 'шаблон R1 всегда открывает <think>'
  return 'у Llama 3.1 режима рассуждения нет'
}

function methodHint(c: PromptCount | null, family: ModelFamily, loading: boolean): string {
  if (family === 'gpt-oss') return loading ? 'загружаем словарь gpt-oss' : 'словарь gpt-oss (o200k)'
  if (!c) return 'загружаем словарь'
  return c.method === 'exact-qwen3' ? 'словарь Qwen3' : 'условно, см. ниже'
}

interface LabInput {
  messages: ChatMessage[]
  tools: boolean
  thinking: boolean
  addGen: boolean
  harmonyDate: string
}

function renderLab(family: ModelFamily, o: LabInput) {
  return renderTemplate({
    family,
    messages: o.messages,
    tools: o.tools ? [WEATHER_TOOL] : undefined,
    enableThinking: o.thinking,
    addGenerationPrompt: o.addGen,
    harmonyDate: o.harmonyDate,
  })
}

/** Сегодняшняя дата для шаблона gpt-oss: настоящий шаблон берёт её из strftime_now. */
function todayIso(): string {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

export interface TemplateLabProps {
  tokenizer: Tokenizer | null
  /** Текст симулятора: лаборатория следует за ним, пока читатель не правил сообщения здесь. */
  systemText: string
  userText: string
  /** Начальные значения из симулятора; при их смене в симуляторе локальные переключатели следуют за ними. */
  initialFamily: ModelFamily
  initialThinking: boolean
  /** Включены ли tools в настройках симулятора: тогда и здесь get_weather (то же число токенов, что в главе 1). */
  initialTools: boolean
}

export function TemplateLab({ tokenizer, systemText, userText, initialFamily, initialThinking, initialTools }: TemplateLabProps) {
  const [msgs, setMsgs] = useState<EditableMessage[]>(() => defaults(systemText, userText))
  const [family, setFamily] = useState<ModelFamily>(initialFamily)
  const [tools, setTools] = useState(initialTools)
  const [thinking, setThinking] = useState(initialThinking)
  const [addGen, setAddGen] = useState(true)
  const [harmonyDate] = useState(todayIso)
  // Глава монтируется в фоне раньше, чем читатель пишет свой запрос в симуляторе, поэтому лаборатория следует за ним:
  // текст, tools, рассуждение и модель приходят сюда и после монтирования. Свой диалог читателя не затираем: правка,
  // удаление или новая реплика отключают следование за текстом, «Сбросить» включает его снова.
  const [edited, setEdited] = useState(false)
  useEffect(() => {
    if (!edited) setMsgs((ms) => (sameAsDefaults(ms, systemText, userText) ? ms : defaults(systemText, userText)))
  }, [systemText, userText, edited])
  useEffect(() => setTools(initialTools), [initialTools])
  useEffect(() => setThinking(initialThinking), [initialThinking])
  useEffect(() => setFamily(initialFamily), [initialFamily])
  // словарь gpt-oss нужен строке сравнения, поэтому грузим его, когда фигура подходит к экрану
  const [viewRef, near] = useInView<HTMLDivElement>({ rootMargin: '600px', once: true })
  const o200k = useO200k(family === 'gpt-oss' || near)

  const input = useMemo<LabInput>(
    () => ({ messages: msgs.map(({ id: _id, ...m }) => m), tools, thinking, addGen, harmonyDate }),
    [msgs, tools, thinking, addGen, harmonyDate],
  )
  const rendered = useMemo(() => renderLab(family, input), [family, input])
  const count = useMemo(
    () => countPrompt(family, rendered.segments, tokenizer, o200k.tokenizer),
    [family, rendered.segments, tokenizer, o200k.tokenizer],
  )
  // тот же диалог во всех четырёх шаблонах: разброс виден сразу, без запоминания чисел
  const perFamily = useMemo(
    () => FAMILY_OPTIONS.map((f) => ({ ...f, count: countPrompt(f.value, renderLab(f.value, input).segments, tokenizer, o200k.tokenizer) })),
    [input, tokenizer, o200k.tokenizer],
  )
  const maxCount = Math.max(1, ...perFamily.map((f) => f.count?.total ?? 0))

  // рассуждение прошлого ответа, которое шаблон вырезал (Qwen3 после следующего вопроса пользователя и т.п.)
  const dropped = useMemo(() => {
    const ids = new Set<number>()
    for (const m of msgs) {
      const r = m.role === 'assistant' ? m.reasoning?.trim() : ''
      if (r && !rendered.text.includes(r)) ids.add(m.id)
    }
    return ids
  }, [msgs, rendered.text])

  // добавили реплику: прокручиваем строку промпта к концу, иначе новая реплика остаётся ниже края окна
  const scrollerRef = useRef<HTMLDivElement>(null)
  const prevLen = useRef(msgs.length)
  useEffect(() => {
    if (msgs.length > prevLen.current) {
      const el = scrollerRef.current
      if (el) el.scrollTop = el.scrollHeight
    }
    prevLen.current = msgs.length
  }, [msgs.length])

  const update = (id: number, patch: Partial<ChatMessage>) => {
    setEdited(true)
    setMsgs((ms) => ms.map((m) => (m.id === id ? { ...m, ...patch } : m)))
  }
  const remove = (id: number) => {
    setEdited(true)
    setMsgs((ms) => (ms.length > 1 ? ms.filter((m) => m.id !== id) : ms))
  }
  const add = (...items: ChatMessage[]) => {
    setEdited(true)
    setMsgs((ms) => (ms.length + items.length > MAX_MESSAGES ? ms : [...ms, ...items.map(mk)]))
  }
  const reset = () => {
    setEdited(false)
    setMsgs(defaults(systemText, userText))
  }
  const full = msgs.length >= MAX_MESSAGES

  const thinkingDisabled = family === 'llama3' || family === 'deepseek-r1'
  const loadingO200k = family === 'gpt-oss' && o200k.status !== 'ready'
  const template = count ? Math.max(0, count.total - count.content - count.tools) : null
  const chars = rendered.text.length

  return (
    <Figure
      id="template-figure-lab"
      title="Шаблон чата: из сообщений в одну строку"
      width="full"
      controls={
        <>
          <Toggle label="Инструменты (tools)" checked={tools} onChange={setTools} hint="функция get_weather в запросе" />
          <Toggle
            label="Рассуждение"
            checked={thinkingDisabled ? family === 'deepseek-r1' : thinking}
            onChange={setThinking}
            disabled={thinkingDisabled}
            hint={thinkingHint(family, thinking)}
          />
          <Toggle
            label="add_generation_prompt"
            checked={addGen}
            onChange={setAddGen}
            hint={addGen ? 'в конце открыта реплика ассистента' : 'промпт кончается на реплике пользователя'}
          />
        </>
      }
      caption="Попробуйте: переключите семейство на gpt-oss и сравните число токенов. Добавьте ответ ассистента: его рассуждение видно в промпте, пока он последний; добавьте следом вопрос пользователя, и шаблон Qwen3 рассуждение выбросит."
    >
      <div ref={viewRef} className="grid gap-5 @min-[820px]:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
        {/* сообщения */}
        <div className="grid min-w-0 content-start gap-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-medium text-ink-2">Сообщения (messages)</span>
            <Button variant="ghost" size="sm" icon={ArrowCounterClockwiseIcon} onClick={reset}>
              Сбросить
            </Button>
          </div>
          {msgs.map((m) => (
            <MessageEditor
              key={m.id}
              m={m}
              dropped={dropped.has(m.id)}
              onChange={(p) => update(m.id, p)}
              onRemove={msgs.length > 1 ? () => remove(m.id) : undefined}
            />
          ))}
          <div className="flex flex-wrap gap-2 pt-1">
            <Button size="sm" icon={PlusIcon} disabled={full} onClick={() => add({ role: 'assistant', content: ASSISTANT_REPLY.content, reasoning: ASSISTANT_REPLY.reasoning })}>
              Ответ ассистента
            </Button>
            <Button size="sm" icon={PlusIcon} disabled={full} onClick={() => add({ role: 'user', content: USER_FOLLOWUP })}>
              Вопрос пользователя
            </Button>
            <Button
              size="sm"
              icon={PlusIcon}
              disabled={msgs.length + 2 > MAX_MESSAGES}
              onClick={() => {
                setTools(true)
                add(
                  { role: 'assistant', content: '', tool_calls: [{ name: 'get_weather', arguments: { city: 'Берлин' } }] },
                  { role: 'tool', content: TOOL_RESULT, name: 'get_weather' },
                )
              }}
            >
              Вызов инструмента
            </Button>
          </div>
        </div>

        {/* промпт */}
        <div className="grid min-w-0 content-start gap-2.5">
          <Segmented<ModelFamily> label="Семейство моделей" size="sm" wrap value={family} onChange={setFamily} options={FAMILY_OPTIONS} />
          <div className="flex flex-wrap items-baseline justify-between gap-x-3">
            <span className="text-sm font-medium text-ink-2">Строка, которую увидит модель</span>
            <span className="text-[13px] text-ink-3 tabular-nums">
              {fmt.int(chars)} {fmt.plural(chars, ['символ', 'символа', 'символов'])}
            </span>
          </div>
          <div
            ref={scrollerRef}
            className="max-h-[460px] overflow-auto rounded-panel bg-surface-2 px-3 py-2.5 shadow-[inset_0_0_0_1px_var(--line)] scrollbar-thin"
            tabIndex={0}
            aria-label="Отрендеренный промпт"
          >
            <PromptView segments={rendered.segments} family={family} />
          </div>
          <PromptLegend segments={rendered.segments} />
          {dropped.size > 0 && (
            <FigureLive as="p" className="flex items-start gap-2 text-[13.5px] leading-snug text-ink-2">
              <ScissorsIcon size={16} aria-hidden className="mt-0.5 shrink-0 text-ink-3" />
              <span>
                <span className="font-medium text-ink">Рассуждение прошлого ответа в промпт не попало:</span> {DROPPED_REASON[family]}
              </span>
            </FigureLive>
          )}
          {rendered.warnings.map((w) => (
            <p key={w} className="flex items-start gap-2 text-[13.5px] leading-snug text-ink-2">
              <WarningIcon size={16} aria-hidden className="mt-0.5 shrink-0 text-warn" />
              <span>
                <span className="font-medium text-ink">Предупреждение.</span> {w.replace(/\s*\(notes[^)]*\)/g, '')}
              </span>
            </p>
          ))}
          <div className="mt-1 grid grid-cols-2 gap-x-5 gap-y-3 @min-[560px]:grid-cols-4">
            <Stat
              label="Токенов"
              value={count && !loadingO200k ? fmt.int(count.total) : '…'}
              hint={methodHint(count, family, loadingO200k)}
              swatch="var(--c-prefill)"
            />
            <Stat label="Ваш текст" value={count && !loadingO200k ? `≈ ${fmt.int(count.content)}` : '…'} hint="то, что вы написали" />
            <Stat label="От шаблона" value={template !== null && !loadingO200k ? `≈ ${fmt.int(template)}` : '…'} hint="роли, маркеры, обвязка" />
            {tools && (
              <Stat label="Описание tools" value={count && !loadingO200k ? `≈ ${fmt.int(count.tools)}` : '…'} hint="в каждом запросе" swatch="var(--c-tool)" />
            )}
          </div>

          <div className="mt-1 grid gap-1.5">
            <div className="text-[13px] font-medium text-ink-2">Этот же диалог в других шаблонах</div>
            <ul className="grid grid-cols-2 gap-1.5 @min-[560px]:grid-cols-4" aria-label="Число токенов этого диалога в шаблонах четырёх семейств">
              {perFamily.map((f) => {
                const approx = f.count?.method === 'approx-qwen3'
                const total = f.count?.total ?? null
                const current = f.value === family
                return (
                  <li
                    key={f.value}
                    aria-current={current ? 'true' : undefined}
                    className={cn(
                      'grid min-w-0 gap-1 rounded-control px-2.5 py-1.5',
                      current ? 'bg-accent-soft shadow-[inset_0_0_0_1px_var(--accent)]' : 'shadow-[inset_0_0_0_1px_var(--line)]',
                    )}
                  >
                    <span className={cn('truncate text-[12.5px]', current ? 'font-medium text-ink' : 'text-ink-2')}>{f.label}</span>
                    <span className="font-mono text-[15px] leading-5 text-ink tabular-nums">
                      {total === null ? '…' : `${approx ? '≈ ' : ''}${fmt.int(total)}`}
                    </span>
                    <span aria-hidden className="h-1 rounded-full bg-surface-3">
                      <span className="block h-1 rounded-full bg-prefill" style={{ width: total === null ? 0 : `${(total / maxCount) * 100}%` }} />
                    </span>
                  </li>
                )
              })}
            </ul>
            <p className="text-[13px] leading-snug text-ink-3">
              Qwen3 и gpt-oss посчитаны их настоящими словарями. Llama 3.1 и DeepSeek-R1 условно (≈): их словарей в браузере нет,
              поэтому каждый спецтокен считаем одним токеном, как в настоящем словаре, а остальной текст режем словарём Qwen3.
            </p>
          </div>
          <p className="text-[13.5px] leading-relaxed text-ink-2">{FAMILY_NOTES[family]}</p>
        </div>
      </div>
    </Figure>
  )
}

function roleLabel(m: ChatMessage): string {
  if (m.role === 'tool') return 'tool: результат get_weather'
  if (m.role === 'assistant' && m.tool_calls?.length) return 'assistant: вызов инструмента'
  return m.role
}

function MessageEditor({
  m,
  dropped,
  onChange,
  onRemove,
}: {
  m: EditableMessage
  /** Рассуждение этого ответа шаблон в промпт не взял. */
  dropped: boolean
  onChange: (p: Partial<ChatMessage>) => void
  onRemove?: () => void
}) {
  const removeBtn = onRemove && (
    <Button
      variant="ghost"
      size="sm"
      iconOnly
      icon={XIcon}
      aria-label={`Удалить сообщение ${roleLabel(m)}`}
      onClick={onRemove}
      className="absolute -top-1.5 right-0 h-7 w-7"
    />
  )
  // крестик стоит справа в строке подписи, там же TextArea показывает счётчик символов: строке подписи отступ
  // под крестик, иначе он закрывает цифры счётчика и длинную подпись на телефоне
  const labelRow = onRemove ? 'pr-9' : undefined
  // лимит как у полей симулятора: текст, пришедший оттуда (например, «Длинный документ»), всегда помещается
  const maxText = SETTINGS_LIMITS.maxTextLength
  if (m.role === 'assistant' && m.tool_calls?.length) {
    const tc = m.tool_calls[0]
    return (
      <div className="relative grid gap-1.5">
        <span className="text-sm font-medium text-ink-2">{roleLabel(m)}</span>
        {removeBtn}
        <div className="rounded-control bg-surface-2 px-3 py-2.5 font-mono text-[13px] text-ink shadow-[inset_0_0_0_1px_var(--line)]">
          {tc.name}({JSON.stringify(tc.arguments)})
        </div>
      </div>
    )
  }
  if (m.role === 'assistant') {
    return (
      <div className="relative grid gap-2 border-l-2 border-line pl-3">
        {removeBtn}
        <TextArea
          label="assistant: рассуждение (reasoning_content)"
          value={m.reasoning ?? ''}
          onChange={(v) => onChange({ reasoning: v })}
          rows={1}
          autoGrow
          maxLength={1200}
          hint={dropped ? 'в промпт не попадёт: шаблон его вырезал' : undefined}
          labelClassName={labelRow}
        />
        <TextArea label="assistant: ответ (content)" value={m.content} onChange={(v) => onChange({ content: v })} rows={2} autoGrow maxLength={maxText} />
      </div>
    )
  }
  return (
    <div className="relative">
      {removeBtn}
      <TextArea
        label={roleLabel(m)}
        value={m.content}
        onChange={(v) => onChange({ content: v })}
        rows={m.role === 'user' ? 2 : 1}
        autoGrow
        maxLength={maxText}
        labelClassName={labelRow}
      />
    </div>
  )
}
