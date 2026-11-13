import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { motion } from 'motion/react'
import { CheckCircleIcon, ProhibitIcon, TreeStructureIcon, XCircleIcon } from '@phosphor-icons/react'
import { Badge, CodeBlock, Figure, Player, Segmented, Skeleton, TextArea, Toggle, TokenChip, cn, fmt, tint, tokenSegments, useInView, usePlayer } from '@/ui'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import type { JsonSchema, Token, Tokenizer } from '@/engine/types'
import { PRESETS, chatterFor, exampleValue, parseSchema, tryParse, validate, type PresetId } from './schemas'
import { buildVocabStore, freeTrajectory, fullMask, maskedTrajectory, type Step, type VocabStore } from './trajectory'
import { LOGITS_WIDTH, VocabCanvas } from './VocabCanvas'

type Choice = PresetId | 'custom'

const CUSTOM_DEFAULT = `{
  "type": "object",
  "properties": {
    "title": { "type": "string" },
    "tags": { "type": "array", "items": { "type": "string" } },
    "score": { "type": "number" },
    "done": { "type": "boolean" }
  },
  "required": ["title", "score"]
}`

/** Компактный словарь: строится один раз на токенизатор. */
let vocabCache: { tk: Tokenizer; store: VocabStore } | null = null
function getVocab(tk: Tokenizer): VocabStore {
  if (!vocabCache || vocabCache.tk !== tk) vocabCache = { tk, store: buildVocabStore(tk) }
  return vocabCache.store
}

interface Setup {
  schema: JsonSchema
  instance: unknown
  chatter: string
  opener: string
  prompt: string
  key: string
  /** Ключевые слова своей схемы, которые маска не проверяет. */
  ignored: string[]
  /** Своя схема разрешает лишние ключи (additionalProperties: true или схема), а маска их не пропускает. */
  extraKeys: boolean
}

export function MaskFigure() {
  const { tokenizer, status } = useTokenizer()
  const [choice, setChoice] = useState<Choice>('person')
  const [masked, setMasked] = useState(true)
  const [customText, setCustomText] = useState(CUSTOM_DEFAULT)
  const parsedCustom = useMemo(() => parseSchema(customText), [customText])
  // последняя корректная своя схема: пока текст с ошибкой, показываем её
  const [customSchema, setCustomSchema] = useState<{ schema: JsonSchema; ignored: string[]; extraKeys: boolean }>(
    () => parseSchema(CUSTOM_DEFAULT) as { schema: JsonSchema; ignored: string[]; extraKeys: boolean },
  )
  const onCustom = (text: string) => {
    setCustomText(text)
    const r = parseSchema(text)
    if (r.ok) setCustomSchema({ schema: r.schema, ignored: r.ignored, extraKeys: r.extraKeys })
  }

  const setup = useMemo<Setup>(() => {
    if (choice !== 'custom') {
      const p = PRESETS.find((x) => x.id === choice)!
      return { schema: p.schema, instance: p.instance, chatter: p.chatter, opener: p.opener, prompt: p.prompt, key: p.id, ignored: [], extraKeys: false }
    }
    const { schema, ignored, extraKeys } = customSchema
    return {
      schema,
      instance: exampleValue(schema),
      chatter: chatterFor(schema),
      opener: 'Конечно',
      prompt: 'Свой запрос: модель отвечает правдоподобными значениями по вашей схеме.',
      key: JSON.stringify(schema),
      ignored,
      extraKeys,
    }
  }, [choice, customSchema])

  const runs = useMemo(() => {
    if (!tokenizer) return null
    const m = maskedTrajectory(tokenizer, setup.schema, setup.instance, setup.opener)
    const f = freeTrajectory(tokenizer, setup.schema, setup.chatter, m.steps[0])
    return { masked: m.steps, free: f, target: m.target }
  }, [tokenizer, setup])

  const steps = runs ? (masked ? runs.masked : runs.free) : []
  const p = usePlayer(Math.max(1, steps.length), { intervalMs: 1300 })
  const { setStep } = p
  useEffect(() => {
    setStep(0)
  }, [setup, masked, setStep])
  const step = Math.min(p.step, Math.max(0, steps.length - 1))
  const cur = steps[step] ?? null

  // маска по всему словарю для текущего шага: считается после отрисовки, с кэшем
  const [full, setFull] = useState<FullMask | null>(null)
  const cache = useRef(new Map<string, Omit<FullMask, 'key'>>())
  const maskKey = cur && masked ? `${setup.key}#${step}` : null
  // словарь на 151 669 токенов и маска по нему стоят около 100 мс: считаем, когда фигура подъехала к экрану,
  // а не при фоновом монтировании главы
  const [viewRef, near] = useInView<HTMLDivElement>({ rootMargin: '100% 0px', once: true })
  useEffect(() => {
    if (!near || !tokenizer || !cur || !maskKey) return
    const hit = cache.current.get(maskKey)
    if (hit) {
      setFull({ key: maskKey, ...hit })
      return
    }
    const id = window.setTimeout(() => {
      const mask = fullMask(setup.schema, cur.prefix, getVocab(tokenizer))
      let allowed = 0
      for (let i = 0; i < mask.length; i++) allowed += mask[i]
      // разрешённых немного: запоминаем их id, чтобы назвать токены под счётчиком
      const ids: number[] = []
      if (allowed <= LIST_ALLOWED) for (let i = 0; i < mask.length && ids.length < allowed; i++) if (mask[i]) ids.push(i)
      // кэш ограничен: маска на весь словарь весит ~150 КБ
      if (cache.current.size >= 40) cache.current.delete(cache.current.keys().next().value as string)
      cache.current.set(maskKey, { mask, allowed, ids })
      setFull({ key: maskKey, mask, allowed, ids })
    }, 30)
    return () => window.clearTimeout(id)
  }, [near, tokenizer, cur, maskKey, setup.schema])
  const fullNow = full && full.key === maskKey ? full : null

  const labels = useMemo(() => steps.map((s) => `токен «${tokenSegments(s.chosen).map((x) => x.value).join('')}»`), [steps])
  // запас высоты на весь прогон: самая длинная таблица кандидатов и итоговый текст ответа.
  // Иначе фигура меняет высоту почти на каждом шаге, и кнопка «Шаг вперёд» уезжает из-под курсора
  const reserve = useMemo(() => {
    const last = steps[steps.length - 1]
    return {
      rows: steps.reduce((m, s) => Math.max(m, s.cands.length), 0),
      // самый длинный выбранный токен прогона: по нему держится высота абзаца о перенормировке
      token: steps.reduce((m, s) => (visible(s.chosen).length > m.length ? visible(s.chosen) : m), ''),
      // спецтокен в конце рисуется чипом с полями: в запасе его литерал и ещё один символ на поля
      answer: last ? last.prefixText + (last.chosen.special ? `${last.chosen.raw}\u00a0` : last.chosen.text) : '',
    }
  }, [steps])

  return (
    <Figure
      id="structured-output-figure-mask"
      title="Генерация JSON под маской грамматики, шаг за шагом"
      width="full"
      controls={<Player {...p.bind} labels={labels} aria-label="Шаги генерации" />}
      caption="Попробуйте: на первом шаге включите «Без ограничений». Модель хочет того же, что и под маской, но без запрета выбирает «Конечно» и дальше пишет текст, который JSON.parse не разберёт. Вероятности модели условные; токены, маски и счётчики разрешённых настоящие (словарь Qwen3, правила XGrammar)."
    >
      <div ref={viewRef} className="grid gap-5">
        <div className="flex flex-wrap items-end gap-x-8 gap-y-4">
          <Segmented<Choice>
            label="Схема ответа"
            value={choice}
            onChange={setChoice}
            wrap
            options={[...PRESETS.map((x) => ({ value: x.id as Choice, label: x.label })), { value: 'custom', label: 'Своя' }]}
          />
          <Toggle
            label="Без ограничений"
            checked={!masked}
            onChange={(v) => setMasked(!v)}
            hint={masked ? 'Сейчас маска включена: запрещённые токены получают −∞' : 'Маска выключена: модель пишет что хочет'}
          />
        </div>

        <div className="grid gap-5 @4xl:grid-cols-[minmax(0,320px)_minmax(0,1fr)]">
          <SchemaPanel
            choice={choice}
            setup={setup}
            customText={customText}
            onCustom={onCustom}
            parsedOk={parsedCustom.ok}
            error={parsedCustom.ok ? null : parsedCustom.error}
          />
          {/* правая колонка сама контейнер: раскладка таблицы и словаря считается по её ширине, а не по ширине всей фигуры */}
          <div className="@container grid min-w-0 content-start gap-4">
            {status === 'error' && <p className="text-sm text-ink-2">Не удалось загрузить словарь Qwen3. Обновите страницу.</p>}
            {(!runs || !cur) && status !== 'error' && <Skeleton lines={6} />}
            {runs && cur && tokenizer && (
              <>
                <AnswerSoFar step={cur} masked={masked} reserve={reserve.answer} />
                <GrammarLine step={cur} masked={masked} />
                <div className="grid gap-4 @3xl:grid-cols-[minmax(0,1fr)_200px]">
                  <CandTable step={cur} masked={masked} rows={reserve.rows} />
                  <VocabPanel masked={masked} full={fullNow} tokenizer={tokenizer} />
                </div>
                <Renorm step={cur} masked={masked} reserve={reserve.token} />
                <VerdictSlot last={step === steps.length - 1} steps={steps} schema={setup.schema} ignored={setup.ignored} />
              </>
            )}
          </div>
        </div>
      </div>
    </Figure>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Схема слева
 * ------------------------------------------------------------------------------------------- */

function SchemaPanel(props: { choice: Choice; setup: Setup; customText: string; onCustom: (s: string) => void; parsedOk: boolean; error: string | null }) {
  const { choice, setup, customText, onCustom, parsedOk, error } = props
  return (
    <div className="grid min-w-0 content-start gap-3">
      <div className="grid gap-1">
        <div className="text-sm font-medium text-ink-2">Запрос</div>
        <p className="text-[14px] leading-snug text-ink">{setup.prompt}</p>
      </div>
      {choice === 'custom' ? (
        <div className="grid gap-2">
          <TextArea label="JSON Schema (можно править)" value={customText} onChange={onCustom} rows={10} mono autoGrow />
          {parsedOk ? (
            <div className="grid gap-1">
              <Badge tone="good">схема принята</Badge>
              {setup.ignored.length > 0 && (
                <p className="text-[13px] text-ink-2">
                  Маска не проверяет: <span className="font-mono text-[12.5px] text-ink">{setup.ignored.join(', ')}</span>. Эти правила ответ может нарушить.
                </p>
              )}
              {setup.extraKeys && (
                <p className="text-[13px] text-ink-2">
                  Маска на этой странице строже схемы: лишних ключей она не пропускает, хотя схема их разрешает. Настоящий XGrammar их бы пропустил.
                </p>
              )}
            </div>
          ) : (
            <div className="grid gap-1">
              <Badge tone="bad">ошибка в схеме</Badge>
              <p className="text-[13px] text-ink-2">{error} Пока показана последняя корректная схема.</p>
            </div>
          )}
          <p className="text-[12.5px] leading-snug text-ink-3">
            Поддерживаются object, string, integer, number, boolean, enum (можно без type) и массивы; ключи идут в порядке объявления. Остальные ключевые
            слова (format, minimum, pattern…) маска пропускает.
          </p>
        </div>
      ) : (
        <CodeBlock lang="json" title="JSON Schema" code={JSON.stringify(setup.schema, null, 2)} maxHeight={360} />
      )}
    </div>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Ответ на данный момент и состояние грамматики
 * ------------------------------------------------------------------------------------------- */

/** reserve: итоговый текст ответа, невидимый, в той же ячейке сетки: поле с первого шага такой высоты, как в конце. */
function AnswerSoFar({ step, masked, reserve }: { step: Step; masked: boolean; reserve: string }) {
  const chosen = step.chosen
  return (
    <div className="grid gap-1.5">
      <div className="text-sm font-medium text-ink-2">Ответ модели к этому шагу</div>
      <div className="grid min-h-[3rem] rounded-control bg-surface-2 px-3 py-2 font-mono text-[14px] leading-relaxed break-words whitespace-pre-wrap text-ink">
        <span aria-hidden className="invisible col-start-1 row-start-1 min-w-0">
          {reserve}
        </span>
        <span className="col-start-1 row-start-1 min-w-0">
          <span>{step.prefixText}</span>
          {chosen.special ? (
            <span className="mx-0.5 rounded-[4px] px-0.5 text-ink-2 outline-1 -outline-offset-1 outline-dashed outline-[var(--c-special)]">{chosen.raw}</span>
          ) : (
            <motion.span
              key={`${step.index}-${chosen.id}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.25 }}
              className="rounded-[3px]"
              style={{ background: masked ? tint('decode', 22) : tint('bad', 18) }}
            >
              {chosen.text}
            </motion.span>
          )}
        </span>
      </div>
    </div>
  )
}

function GrammarLine({ step, masked }: { step: Step; masked: boolean }) {
  // две строки всегда: длинное «ждёт» (enum, «но маска выключена») переносится, и без запаса фигура прыгала бы
  if (step.valid) {
    return (
      <p className="flex min-h-[2lh] items-start gap-2 text-[15px] leading-snug text-ink">
        <TreeStructureIcon size={18} aria-hidden className="mt-0.5 shrink-0 text-ink-2" />
        <span>
          Грамматика ждёт: <b className="font-semibold">{step.expecting}</b>
          {!masked && <span className="text-ink-2"> (но маска выключена)</span>}
        </span>
      </p>
    )
  }
  return (
    <p className="flex min-h-[2lh] items-start gap-2 text-[15px] leading-snug text-ink">
      <XCircleIcon size={18} aria-hidden className="mt-0.5 shrink-0" style={{ color: 'var(--bad)' }} />
      <span>Текст уже нарушает схему: никакое продолжение не сделает его валидным JSON.</span>
    </p>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Кандидаты: до и после маски
 * ------------------------------------------------------------------------------------------- */

/**
 * Столбик от тонкой базовой линии, без серой дорожки (DESIGN §3.3). «Модель хочет» обведён контуром,
 * «После маски» залит: так их видно рядом и в тёмной теме. Ширина без анимации: строки меняются целиком на каждом шаге.
 */
function Bar({ value, color, outline }: { value: number; color: string; outline?: boolean }) {
  const v = Math.max(0, Math.min(1, value))
  return (
    <div className="flex min-w-0 items-center gap-2">
      <div className="relative h-3 flex-1 border-l border-line-strong">
        {v > 0 && (
          <div
            className="absolute inset-y-0 left-0 rounded-r-[3px]"
            style={{
              width: `max(2px, ${v * 100}%)`,
              background: outline ? `color-mix(in oklab, ${color} 16%, var(--surface))` : color,
              boxShadow: outline ? `inset 0 0 0 1.5px ${color}` : undefined,
            }}
          />
        )}
      </div>
      <span className="w-11 shrink-0 text-right text-[13px] text-ink tabular-nums">{fmtP(value)}</span>
    </div>
  )
}

function fmtP(p: number): string {
  if (p <= 0) return '0\u00a0%'
  if (p < 0.01) return '<1\u00a0%'
  return fmt.pct(p, 0)
}

function MaskStatus({ masked, allowed }: { masked: boolean; allowed: boolean }) {
  if (!masked) return <span className="text-ink-3">не применяется</span>
  return allowed ? (
    <span className="inline-flex items-center gap-1 text-ink">
      <CheckCircleIcon size={16} aria-hidden style={{ color: 'var(--good)' }} />
      разрешён
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-ink">
      <ProhibitIcon size={16} aria-hidden style={{ color: 'var(--bad)' }} />
      запрещён, −∞
    </span>
  )
}

/** Подпись для токенов, которые в чипе почти не видно (пробелы, переводы строк). */
function autoHint(t: Token): string | undefined {
  if (t.special) return undefined
  if (t.text === '\n') return 'перевод строки'
  if (t.text === '\n\n') return 'два перевода строки'
  if (t.text === ' ') return 'пробел'
  if (/^\s+$/.test(t.text)) return 'пробелы'
  return undefined
}

/**
 * Четыре колонки, когда сама таблица шире 28rem (она контейнер), иначе две строки на кандидата.
 * Первая колонка шире: подсказка «начало «двойное списание»» помещается в строку, и строки таблицы одной высоты.
 */
const ROW = 'grid grid-cols-2 items-center gap-x-3 gap-y-1.5 @md:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)_7.5rem_minmax(0,1fr)]'

function CandTable({ step, masked, rows }: { step: Step; masked: boolean; rows: number }) {
  return (
    <div className="@container grid min-w-0 content-start gap-2" role="table" aria-label="Кандидаты на этом шаге">
      <div className="text-sm font-medium text-ink-2">Кандидаты на этом шаге</div>
      <div role="row" className={cn(ROW, 'hidden text-[12.5px] text-ink-3 @md:grid')}>
        <span role="columnheader">Токен</span>
        <span role="columnheader">Модель хочет</span>
        <span role="columnheader">Маска</span>
        <span role="columnheader">{masked ? 'После маски' : 'Без маски'}</span>
      </div>
      {step.cands.map((c, i) => (
        <CandRow key={c.token.id} c={c} p={step.probs[i]} fp={step.finalProbs[i]} masked={masked} />
      ))}
      {/* невидимые строки до самого длинного шага прогона: число кандидатов меняется от 3 до 5, а высота таблицы нет */}
      {Array.from({ length: Math.max(0, rows - step.cands.length) }, (_, i) => (
        <CandRow key={`pad-${i}`} c={step.cands[0]} p={0} fp={0} masked={masked} filler />
      ))}
    </div>
  )
}

function CandRow({ c, p, fp, masked, filler }: { c: Step['cands'][number]; p: number; fp: number; masked: boolean; filler?: boolean }) {
  const struck = masked && !c.allowed
  const hint = c.hint ?? autoHint(c.token)
  return (
    <div role={filler ? undefined : 'row'} aria-hidden={filler || undefined} className={cn(ROW, 'border-t border-line pt-2', filler && 'invisible')}>
      {/* в четыре колонки строка не переносится: слишком длинная подсказка (своя схема) обрезается, полный текст в title */}
      <div role={filler ? undefined : 'cell'} className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5 @md:flex-nowrap">
        <span className={cn('inline-flex shrink-0', struck && 'line-through decoration-[var(--bad)] decoration-[1.5px] opacity-80')}>
          <TokenChip token={c.token} variant={c.chosen ? 'generated' : 'default'} highlight={c.chosen} size="sm" />
        </span>
        {c.chosen && <span className="shrink-0 text-[12px] font-medium text-accent-hover">выбран</span>}
        {hint && (
          <span className="min-w-0 text-[12px] text-ink-3 @md:truncate" title={hint}>
            {hint}
          </span>
        )}
      </div>
      <div role={filler ? undefined : 'cell'} className="justify-self-end text-[13px] whitespace-nowrap @md:order-3 @md:justify-self-start">
        <MaskStatus masked={masked} allowed={c.allowed} />
      </div>
      <div role={filler ? undefined : 'cell'} className="min-w-0 @md:order-2">
        <span className="mb-0.5 block text-[11.5px] text-ink-3 @md:hidden">модель хочет</span>
        <Bar value={p} color="var(--c-decode)" outline />
      </div>
      <div role={filler ? undefined : 'cell'} className="min-w-0 @md:order-4">
        <span className="mb-0.5 block text-[11.5px] text-ink-3 @md:hidden">{masked ? 'после маски' : 'без маски'}</span>
        <Bar value={fp} color="var(--c-decode)" />
      </div>
    </div>
  )
}

/** Сколько разрешённых токенов ещё перечисляем словами под счётчиком. */
const LIST_ALLOWED = 12

interface FullMask {
  key: string
  mask: Uint8Array
  allowed: number
  /** id разрешённых токенов, если их не больше LIST_ALLOWED. */
  ids: number[]
}

/** Классы раскладки «словарь под таблицей»: квадрат слева, подписи справа (колонка 28-48rem). */
const SIDE_CANVAS = '@md:@max-3xl:col-start-1 @md:@max-3xl:row-start-1 @md:@max-3xl:row-span-4'
const SIDE_TEXT = '@md:@max-3xl:col-start-2'

function VocabPanel({ masked, full, tokenizer }: { masked: boolean; full: Omit<FullMask, 'key'> | null; tokenizer: Tokenizer }) {
  const vocab = tokenizer.vocabSize
  const describe = (row: number): ReactNode => {
    if (row >= vocab) {
      return (
        <div className="grid gap-0.5">
          <span className="text-xs text-ink-3">строка {fmt.int(row)}</span>
          <span>пустая строка паддинга: запрещена всегда</span>
        </div>
      )
    }
    const t = tokenizer.token(row)
    const ok = !masked || (full ? full.mask[row] === 1 : null)
    return (
      <div className="grid gap-1">
        <span className="text-xs text-ink-3">строка {fmt.int(row)}</span>
        <span className="font-mono text-[13px] break-all">{visible(t)}</span>
        {ok !== null && <span className="text-ink-2">{!masked ? 'маска выключена' : ok ? 'разрешён' : 'замаскирован (−∞)'}</span>}
      </div>
    )
  }
  return (
    <div className="grid content-start gap-2 @md:@max-3xl:grid-cols-[180px_minmax(0,1fr)] @md:@max-3xl:gap-x-4">
      <div className={cn('text-sm font-medium text-ink-2', '@md:@max-3xl:col-start-2 @md:@max-3xl:row-start-1')}>Весь словарь</div>
      <VocabCanvas
        mask={masked ? (full?.mask ?? null) : null}
        allowed={masked ? (full?.allowed ?? null) : vocab}
        free={!masked}
        vocabSize={vocab}
        describe={describe}
        className={cn('h-auto w-full max-w-[200px] cursor-crosshair rounded-[6px] bg-surface-2', SIDE_CANVAS)}
      />
      {masked ? (
        // счётчик и невидимая самая длинная его версия в одной ячейке: пока маска считается, высота та же
        <p className={cn('grid text-[13px] leading-snug text-ink-2', SIDE_TEXT)}>
          <span className="col-start-1 row-start-1 min-w-0">
            {full ? <MaskSummary allowed={full.allowed} /> : <span className="text-ink-3">Считаем маску по {fmt.int(vocab)} токенам…</span>}
          </span>
          <span aria-hidden className="invisible col-start-1 row-start-1 min-w-0">
            <MaskSummary allowed={vocab} share={`${fmt.num(0.0013, 4)}\u00a0%`} />
          </span>
        </p>
      ) : (
        <p className={cn('text-[13px] leading-snug text-ink-2', SIDE_TEXT)}>Маска выключена: разрешены все {fmt.int(vocab)} токенов словаря.</p>
      )}
      {masked && (
        // блок есть всегда, даже когда разрешённых больше 12 или маска ещё считается: иначе фигура прыгала бы на каждом шаге
        <div className={cn('grid gap-1', SIDE_TEXT)}>
          <span className="text-[12px] text-ink-3">{full && full.ids.length === 1 ? 'Разрешён только' : 'Разрешены'}</span>
          {/* 21px: строка чипов sm, у спецтокена с рамкой; столбец 200px вмещает любой список из 12 в две строки */}
          <div className="flex min-h-[21px] flex-wrap items-center gap-1 @3xl:min-h-[46px] @3xl:content-start">
            {full && full.ids.length > 0
              ? full.ids.map((id) => <TokenChip key={id} token={tokenizer.token(id)} size="sm" />)
              : full && <span className="text-[12px] text-ink-3">больше {LIST_ALLOWED} токенов: список не показываем</span>}
          </div>
        </div>
      )}
      <div className={cn('grid gap-1 text-[12px] text-ink-2', SIDE_TEXT)}>
        <span className="inline-flex items-center gap-1.5">
          <span
            aria-hidden
            className="h-2.5 w-2.5 rounded-[2px]"
            style={{ background: `color-mix(in oklab, var(--good) ${masked ? 75 : 45}%, var(--surface-2))` }}
          />
          разрешён
        </span>
        {masked && (
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-2.5 w-2.5 rounded-[2px]" style={{ background: 'color-mix(in oklab, var(--bad) 24%, var(--surface-2))' }} />
            замаскирован (−∞)
          </span>
        )}
        <span className="text-ink-3">Наведите или нажмите на точку, чтобы увидеть её токен.</span>
      </div>
    </div>
  )
}

/** Счётчик разрешённых под квадратом. share: готовая строка доли (для невидимого запаса высоты). */
function MaskSummary({ allowed, share }: { allowed: number; share?: string }) {
  return (
    <>
      Разрешено <b className="font-semibold text-ink tabular-nums">{fmt.int(allowed)}</b> из {fmt.int(LOGITS_WIDTH)} ({share ?? fmtShare(allowed / LOGITS_WIDTH)}).
      Точка = строка логитов; последние 267 строк пустые и запрещены всегда.
    </>
  )
}

function fmtShare(x: number): string {
  if (x >= 0.1) return fmt.pct(x, 1)
  if (x >= 0.0001) return fmt.pct(x, 2)
  return `${fmt.num(x * 100, 4)}\u00a0%`
}

/* ---------------------------------------------------------------------------------------------
 * Перенормировка словами и итог
 * ------------------------------------------------------------------------------------------- */

/**
 * Перенормировка словами. Рядом в той же ячейке невидимый тот же абзац с самым длинным токеном прогона и «100 %»:
 * длина токена меняется от шага к шагу, а высота абзаца нет.
 */
function Renorm({ step, masked, reserve }: { step: Step; masked: boolean; reserve: string }) {
  let text: (token: string, p: (x: number) => string) => string
  if (!masked) {
    text = (token) => `Без маски сэмплер берёт распределение модели как есть, и самый вероятный токен здесь «${token}».`
  } else {
    const allowedSum = step.cands.reduce((a, c, i) => a + (c.allowed ? step.probs[i] : 0), 0)
    const i = step.cands.findIndex((c) => c.chosen)
    text = (token, p) =>
      `Разрешённые кандидаты вместе набирали ${p(allowedSum)}. Сэмплер делит их вероятности на эту сумму, остальные обнуляет: у «${token}» было ${p(step.probs[i])}, стало ${p(step.finalProbs[i])}. Выбирает по-прежнему модель, но только из разрешённого.`
  }
  return (
    <p className="grid text-[14px] leading-relaxed text-ink-2">
      <span className="col-start-1 row-start-1 min-w-0">{text(visible(step.chosen), fmtP)}</span>
      <span aria-hidden className="invisible col-start-1 row-start-1 min-w-0">
        {text(reserve, () => '100\u00a0%')}
      </span>
    </p>
  )
}

function visible(t: Token): string {
  return tokenSegments(t).map((x) => x.value).join('')
}

/**
 * Итог проверки. Итог и подсказка всегда лежат в одной ячейке, видна одна из них: высота слота равна большей из двух,
 * иначе на последнем шаге фигура меняла бы высоту, и плеер уезжал бы из-под курсора.
 */
function VerdictSlot({ last, steps, schema, ignored }: { last: boolean; steps: Step[]; schema: JsonSchema; ignored: string[] }) {
  return (
    <div className="grid">
      <div className={cn('col-start-1 row-start-1', !last && 'invisible')} aria-hidden={last ? undefined : true}>
        <Verdict steps={steps} schema={schema} ignored={ignored} />
      </div>
      <p
        aria-hidden={last ? true : undefined}
        className={cn(
          'col-start-1 row-start-1 grid place-items-center rounded-panel border border-dashed border-line px-4 py-3 text-center text-[14px] text-ink-3',
          last && 'invisible',
        )}
      >
        Итог появится на последнем шаге: ответ проверят JSON.parse и схема.
      </p>
    </div>
  )
}

function Verdict({ steps, schema, ignored }: { steps: Step[]; schema: JsonSchema; ignored: string[] }) {
  const last = steps[steps.length - 1]
  const text = last.prefixText + (last.chosen.special ? '' : last.chosen.text)
  const parsed = tryParse(text)
  const issues = parsed.value !== undefined ? validate(parsed.value, schema) : []
  let body: ReactNode
  if (parsed.whole && issues.length === 0) {
    body = (
      <p className="flex items-start gap-2">
        <CheckCircleIcon size={18} aria-hidden className="mt-0.5 shrink-0" style={{ color: 'var(--good)' }} />
        <span>
          JSON.parse разбирает ответ целиком, и он проходит проверку по схеме.
          {ignored.length > 0 && ` Но ${ignored.join(', ')} маска не проверяла: такие правила проверяйте сами.`}
        </span>
      </p>
    )
  } else {
    body = (
      <div className="grid gap-1.5">
        <p className="flex items-start gap-2">
          <XCircleIcon size={18} aria-hidden className="mt-0.5 shrink-0" style={{ color: 'var(--bad)' }} />
          <span>
            {parsed.whole ? 'JSON разбирается, но нарушает схему.' : 'JSON.parse падает: ответ не начинается с «{».'}
            {!parsed.whole &&
              parsed.value !== undefined &&
              (issues.length > 0
                ? ' Даже если вырезать объект из текста, схема нарушена:'
                : ` Объект внутри текста схему проходит, но клиенту пришлось бы самому искать и вырезать его.${ignored.length > 0 ? ` Правила ${ignored.join(', ')} здесь не проверялись.` : ''}`)}
          </span>
        </p>
        {issues.length > 0 && (
          <ul className="ml-7 grid list-disc gap-0.5 pl-4 text-[14px] text-ink-2">
            {issues.slice(0, 8).map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        )}
      </div>
    )
  }
  return (
    <div className="rounded-panel bg-surface-2/70 px-4 py-3 text-[15px] leading-snug text-ink">{body}</div>
  )
}
