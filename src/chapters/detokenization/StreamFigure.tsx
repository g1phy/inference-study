import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { motion } from 'motion/react'
import { CheckCircleIcon, HourglassMediumIcon, WarningIcon } from '@phosphor-icons/react'
import { Figure, Player, Segmented, TextArea, TokenChip, TokenStripSkeleton, cn, fmt, tint, tokenSegments, usePlayer } from '@/ui'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { incrementalDetokenize, type DetokStepEx } from '@/engine/detok'
import type { Token, Tokenizer } from '@/engine/types'
import { codePointName, countReplacement, hexByte, modifierLabel, onlyModifiers, pendingChar, seqName, shownText, visibleText, type ModifierKind } from './utf8'

const PRESETS = [
  { id: 'ru', label: 'Кириллица и эмодзи', text: 'KV cache щадит GPU и экономит время 🚀' },
  { id: 'skin', label: 'Эмодзи с оттенком кожи', text: 'Объясню на примере 👍🏽' },
] as const

type PresetId = (typeof PRESETS)[number]['id'] | 'custom'

/** Больше токенов в ленте не помещается осмысленно. */
const MAX_TOKENS = 40

interface StepView {
  token: Token
  step: DetokStepEx
  naive: string
  /** Офсеты до обработки токена (по ним считаются prefix_text и new_text). */
  before: { prefix: number; read: number }
  prefixText: string
  newText: string
}

function buildSteps(tokenizer: Tokenizer, tokens: Token[]): StepView[] {
  const steps = incrementalDetokenize(tokens, tokenizer)
  const ids = tokens.map((t) => t.id)
  return steps.map((s, i) => {
    const prev = i === 0 ? { prefix: 0, read: 0 } : { prefix: steps[i - 1].prefixOffset, read: steps[i - 1].readOffset }
    return {
      token: tokens[i],
      step: s,
      naive: tokenizer.decode([tokens[i].id]),
      before: prev,
      prefixText: tokenizer.decode(ids.slice(prev.prefix, prev.read)),
      newText: tokenizer.decode(ids.slice(prev.prefix, i + 1)),
    }
  })
}

/** Подпись токена для плеера: модификаторы (оттенок кожи, ZWJ) кодом, иначе их не видно. */
function tokenLabel(t: Token): string {
  const text = tokenSegments(t)
    .map((x) => x.value)
    .join('')
  const mods = onlyModifiers(t.text)
  if (mods) return `токен ${mods.map((m) => codePointName(m.cp)).join(' ')} (${modifierLabel(mods[0].kind)})`
  return `токен «${text}»`
}

export function StreamFigure() {
  const { tokenizer, status } = useTokenizer()
  const [preset, setPreset] = useState<PresetId>('ru')
  const [text, setText] = useState<string>(PRESETS[0].text)

  const allTokens = useMemo(() => (tokenizer ? tokenizer.encode(text, { allowSpecial: false }) : []), [tokenizer, text])
  const tokens = useMemo(() => allTokens.slice(0, MAX_TOKENS), [allTokens])
  const truncated = allTokens.length > tokens.length
  const views = useMemo(() => (tokenizer && tokens.length > 0 ? buildSteps(tokenizer, tokens) : []), [tokenizer, tokens])
  const p = usePlayer(Math.max(1, views.length), { autoplay: true, intervalMs: 1100 })
  const { setStep } = p

  // новый текст: проигрываем с начала
  useEffect(() => {
    setStep(0)
  }, [text, setStep])

  const choose = (id: PresetId) => {
    setPreset(id)
    const found = PRESETS.find((x) => x.id === id)
    if (found) setText(found.text)
  }

  const labels = useMemo(() => views.map((v) => tokenLabel(v.token)), [views])
  const ready = status === 'ready' && views.length > 0
  // Шагов может стать меньше раньше, чем плеер сбросит шаг (другой пресет, стёртый текст): держим шаг в границах.
  const step = Math.max(0, Math.min(p.step, views.length - 1))
  const cur = ready ? views[step] : null

  return (
    <Figure
      id="detokenization-figure-stream"
      title="Поток токенов: наивно и с удержанием байтов"
      width="wide"
      controls={<Player {...p.bind} step={step} labels={ready ? labels : undefined} aria-label="Шаги детокенизации" />}
      caption="Попробуйте: пройдите шаги стрелками и остановитесь на токене с байтами F0 9F. В наивной панели уже появился «�», а в потоковой текст стоит на месте, пока символ не соберётся. Токены настоящие, словарь Qwen3."
    >
      <div className="grid gap-5">
        <div className="grid gap-3">
          <Segmented<PresetId>
            label="Что «генерирует» модель"
            value={preset}
            onChange={choose}
            wrap
            options={[...PRESETS.map((x) => ({ value: x.id as PresetId, label: x.label })), { value: 'custom', label: 'Свой текст' }]}
          />
          {preset === 'custom' && (
            <TextArea
              label="Текст ответа"
              value={text}
              onChange={setText}
              rows={1}
              autoGrow
              maxLength={160}
              counter
              hint="Добавьте эмодзи или редкую букву после пробела («щ», «ё», «й»): Qwen3 часто режет их между токенами."
            />
          )}
        </div>

        {status === 'error' && <p className="text-sm text-ink-2">Не удалось загрузить словарь Qwen3. Обновите страницу.</p>}
        {status === 'ready' && views.length === 0 && <p className="text-sm text-ink-2">Введите текст, чтобы увидеть его токены.</p>}
        {status !== 'ready' && status !== 'error' && <TokenStripSkeleton count={14} />}

        {ready && cur && (
          <>
            <TokenRow views={views} step={step} total={allTokens.length} truncated={truncated} />
            <div className="grid gap-3 @lg:grid-cols-2">
              <NaivePanel views={views} step={step} />
              <StreamPanel views={views} step={step} source={text} truncated={truncated} />
            </div>
            <StepNote view={cur} prev={step > 0 ? views[step - 1] : null} views={views} step={step} />
          </>
        )}
      </div>
    </Figure>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Лента токенов с офсетами
 * ------------------------------------------------------------------------------------------- */

function TokenRow({ views, step, total, truncated }: { views: StepView[]; step: number; total: number; truncated: boolean }) {
  const cur = views[step].step
  const prefix = cur.prefixOffset
  const read = cur.readOffset
  return (
    <div className="grid gap-2">
      <div className="text-sm font-medium text-ink-2">Токены от модели (байты под каждым)</div>
      <div className="flex flex-wrap items-start gap-x-1.5 gap-y-3" role="list" aria-label="Сгенерированные токены">
        {views.map((v, i) => {
          const future = i > step
          const pending = !future && i >= read
          const inPrefix = !future && i >= prefix && i < read
          return (
            <div key={i} role="listitem" className="grid justify-items-center gap-1">
              <TokenChip token={v.token} variant={future ? 'muted' : 'generated'} highlight={i === step} />
              <span className="px-0.5 font-mono text-[11px] leading-[1.3] whitespace-nowrap text-ink-3">{fmt.hex(v.token.bytes)}</span>
              <span
                aria-hidden
                className={cn(
                  'mx-[3px] h-0 justify-self-stretch border-t-2',
                  inPrefix ? 'border-ink-3' : pending ? 'border-dashed border-ink-2' : 'border-transparent',
                )}
              />
            </div>
          )
        })}
      </div>
      {truncated && (
        <p className="text-[13px] text-ink-3">
          Показаны первые {fmt.int(MAX_TOKENS)} токенов из {fmt.int(total)}: длиннее лента не помещается.
        </p>
      )}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-[13px] text-ink-2">
        <span className="inline-flex items-center gap-2">
          <span aria-hidden className="w-6 border-t-2 border-ink-3" />
          окно prefix: контекст для следующего вызова decode()
        </span>
        <span className="inline-flex items-center gap-2">
          <span aria-hidden className="w-6 border-t-2 border-dashed border-ink-2" />
          прочитано, но ещё не отдано
        </span>
        <span className="font-mono text-xs text-ink-3 tabular-nums">
          prefix_offset = {prefix}, read_offset = {read}
        </span>
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Две панели вывода
 * ------------------------------------------------------------------------------------------- */

function Panel({ title, rule, status, children }: { title: string; rule: string; status: ReactNode; children: ReactNode }) {
  return (
    <section className="grid min-w-0 content-start gap-2 rounded-panel border border-line bg-surface-2/40 p-3.5">
      <div>
        <h4 className="text-[15px] font-semibold text-ink">{title}</h4>
        <p className="text-[13px] leading-snug text-ink-3">{rule}</p>
      </div>
      <div className="min-h-[3.25rem] rounded-control bg-surface px-3 py-2 font-mono text-[14px] leading-relaxed break-words whitespace-pre-wrap text-ink shadow-[inset_0_0_0_1px_var(--line)]">
        {children}
      </div>
      <div className="text-[13px] text-ink-2">{status}</div>
    </section>
  )
}

/** Текст с подсветкой «�» (фон статуса, сам символ цветом текста). */
function WithReplacement({ text }: { text: string }) {
  const parts = text.split('�')
  return (
    <>
      {parts.map((part, i) => (
        <span key={i}>
          {part}
          {i < parts.length - 1 && (
            <span className="rounded-[3px] px-px" style={{ background: tint('bad', 22), boxShadow: 'inset 0 0 0 1px color-mix(in oklab, var(--bad) 55%, transparent)' }}>
              �
            </span>
          )}
        </span>
      ))}
    </>
  )
}

function NaivePanel({ views, step }: { views: StepView[]; step: number }) {
  const pieces = views.slice(0, step + 1).map((v) => v.naive)
  const bad = countReplacement(pieces.join(''))
  return (
    <Panel
      title="Наивно: decode([id]) по одному"
      rule="Каждый токен декодируется отдельно и сразу уходит клиенту."
      status={
        bad > 0 ? (
          <span className="inline-flex items-center gap-1.5">
            <WarningIcon size={16} aria-hidden style={{ color: 'var(--bad)' }} />
            Клиент уже получил «�»: {fmt.int(bad)} шт. Забрать их обратно нельзя.
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5">
            <CheckCircleIcon size={16} aria-hidden style={{ color: 'var(--good)' }} />
            Пока без «�»
          </span>
        )
      }
    >
      {pieces.map((piece, i) => (
        <motion.span
          key={`${i}-${piece}`}
          initial={i === step ? { opacity: 0 } : false}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.25 }}
          className={cn(i === step && 'rounded-[3px]')}
          style={i === step ? { background: tint('decode', 18) } : undefined}
        >
          <WithReplacement text={piece} />
        </motion.span>
      ))}
    </Panel>
  )
}

function StreamPanel({ views, step, source, truncated }: { views: StepView[]; step: number; source: string; truncated: boolean }) {
  const cur = views[step].step
  const chunks = views.slice(0, step + 1).map((v) => v.step.emitted)
  const pending = cur.heldBytes.length > 0 ? pendingChar(cur.heldBytes) : null
  const done = step === views.length - 1
  let status: ReactNode
  if (pending && done && truncated) {
    status = (
      <span className="inline-flex items-center gap-1.5">
        <HourglassMediumIcon size={16} aria-hidden style={{ color: 'var(--warn)' }} />
        Лента обрезана посреди символа: эти байты ждали бы следующий токен.
      </span>
    )
  } else if (pending) {
    status = (
      <span className="inline-flex items-center gap-1.5">
        <HourglassMediumIcon size={16} aria-hidden style={{ color: 'var(--warn)' }} />
        Придержано {fmt.count(cur.heldBytes.length, ['байт', 'байта', 'байт'])}: ждём ещё {fmt.int(pending.need - pending.bytes.length)}
      </span>
    )
  } else {
    const whole = !truncated && chunks.join('') === source
    status = (
      <span className="inline-flex items-center gap-1.5">
        <CheckCircleIcon size={16} aria-hidden style={{ color: 'var(--good)' }} />
        {!done
          ? 'Всё отданное состоит из целых символов'
          : whole
            ? 'Готово: текст совпал с исходным, ни одного «�»'
            : 'Готово: отдан весь текст этих токенов, ни одного «�»'}
      </span>
    )
  }
  return (
    <Panel title="Потоково: prefix / read offsets" rule="Отдаётся только прирост текста, который не кончается на «�»." status={status}>
      {chunks.map((c, i) =>
        c ? (
          <motion.span
            key={`${i}-${c}`}
            initial={i === step ? { opacity: 0 } : false}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.25 }}
            className="rounded-[3px]"
            style={i === step ? { background: tint('decode', 18) } : undefined}
          >
            {c}
          </motion.span>
        ) : null,
      )}
      {pending && <HeldBytes pending={pending} />}
    </Panel>
  )
}

/** Придержанные байты: целые байты перед символом, пришедшие байты символа и пустые ячейки под недостающие. */
function HeldBytes({ pending }: { pending: NonNullable<ReturnType<typeof pendingChar>> }) {
  const slots = Array.from({ length: pending.need }, (_, k) => pending.bytes[k])
  const missing = pending.need - pending.bytes.length
  return (
    <span
      className="ml-1 inline-flex translate-y-[-1px] items-center gap-[3px] rounded-chip border border-dashed border-ink-3 px-1 py-px align-middle"
      aria-label={`придержанные байты: ${[...pending.before, ...pending.bytes].map(hexByte).join(' ')}, не хватает ${missing}`}
    >
      {pending.before.map((b, k) => (
        <span key={`b${k}`} className="text-[12px] text-ink-2">
          {hexByte(b)}
        </span>
      ))}
      {slots.map((b, k) =>
        b === undefined ? (
          <span key={k} aria-hidden className="inline-block h-[18px] w-[22px] rounded-[3px] border border-dashed border-ink-3" />
        ) : (
          <span key={k} className="inline-grid h-[18px] min-w-[22px] place-items-center rounded-[3px] bg-surface-3 px-0.5 text-[11px] leading-none text-ink">
            {hexByte(b)}
          </span>
        ),
      )}
    </span>
  )
}

/* ---------------------------------------------------------------------------------------------
 * Пояснение к шагу
 * ------------------------------------------------------------------------------------------- */

/** Последний видимый символ того, что клиент получил до шага (к нему приклеится модификатор). */
function lastChar(views: StepView[], step: number): string | null {
  const before = views
    .slice(0, step)
    .map((v) => v.step.emitted)
    .join('')
  const chars = [...before.trimEnd()]
  return chars.length > 0 ? chars[chars.length - 1] : null
}

function modifierNote(kind: ModifierKind, code: string, base: string | null, mod: string): ReactNode {
  if (kind === 'skin') {
    return (
      <>
        Это модификатор оттенка кожи <Mono>{code}</Mono>. Шрифт рисует его вместе с предыдущим эмодзи
        {base ? (
          <>
            , поэтому у клиента {base} перекрашивается в {base + mod}
          </>
        ) : null}
        . Разрезан не символ, а пара символов, и для UTF-8 это законно.
      </>
    )
  }
  if (kind === 'zwj') {
    return (
      <>
        Это невидимый соединитель ZWJ <Mono>{code}</Mono>: шрифт склеит соседние эмодзи в одно изображение, когда придёт следующий. Разрезана не буква,
        а последовательность символов, и для UTF-8 это законно.
      </>
    )
  }
  if (kind === 'variation') {
    return (
      <>
        Это селектор варианта <Mono>{code}</Mono>: он просит нарисовать предыдущий символ цветным эмодзи. Сам он не виден.
      </>
    )
  }
  return (
    <>
      Это служебный символ <Mono>{code}</Mono>: он виден только вместе с соседями.
    </>
  )
}

function StepNote({ view, prev, views, step }: { view: StepView; prev: StepView | null; views: StepView[]; step: number }) {
  const { token, step: s } = view
  const hex = fmt.hex(token.bytes)
  const pending = s.heldBytes.length > 0 ? pendingChar(s.heldBytes) : null
  const wasHeld = prev ? prev.step.heldBytes.length > 0 : false
  const mods = onlyModifiers(s.emitted)
  const arrived = token.bytes.length === 1 ? <>Пришёл недостающий байт <Mono>{hex}</Mono></> : <>Пришли недостающие байты <Mono>{hex}</Mono></>
  let text: ReactNode
  if (pending) {
    text = (
      <>
        Байты <Mono>{fmt.hex(pending.bytes)}</Mono>: <Mono>{hexByte(pending.bytes[0])}</Mono> начинает {seqName(pending.need)} символ, пришло{' '}
        {fmt.int(pending.bytes.length)} из {fmt.int(pending.need)}. Декодер дал бы текст с «�» на конце, поэтому детокенизатор ничего не отдаёт и
        ждёт следующий токен.
      </>
    )
  } else if (mods) {
    const code = mods.map((m) => codePointName(m.cp)).join(' ')
    text = (
      <>
        {wasHeld ? <>{arrived}: символ собран.</> : <>Токен <Mono>{hex}</Mono> заканчивается на целом символе.</>}{' '}
        {modifierNote(mods[0].kind, code, lastChar(views, step), s.emitted)}
      </>
    )
  } else if (wasHeld && s.emitted) {
    text = (
      <>
        {arrived}: символ собран, клиент сразу получает «{visibleText(s.emitted)}».
      </>
    )
  } else if (s.emitted) {
    text = (
      <>
        Токен <Mono>{hex}</Mono> заканчивается на целом символе: клиент получает «{visibleText(s.emitted)}».
      </>
    )
  } else {
    text = <>Токен не добавил видимого текста.</>
  }
  return (
    <div className="grid gap-2 rounded-panel bg-surface-2/60 px-3.5 py-3 text-[14px] leading-relaxed text-ink-2">
      <p className="text-ink">{text}</p>
      <p className="font-mono text-[12.5px] break-words text-ink-3">
        prefix_text = «{visibleText(view.prefixText)}» · new_text = «{visibleText(view.newText)}» →{' '}
        {s.emitted ? `отдать «${mods ? shownText(s.emitted) : visibleText(s.emitted)}»` : 'ждать'}
      </p>
    </div>
  )
}

function Mono({ children }: { children: ReactNode }) {
  return <span className="font-mono text-[13px] text-ink">{children}</span>
}
