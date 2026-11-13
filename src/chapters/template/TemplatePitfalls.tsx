import { useMemo, useState, type ReactNode } from 'react'
import { CheckCircleIcon, XCircleIcon } from '@phosphor-icons/react'
import { Figure, Segmented, Stat, Tabs, TokenStrip, TokenStripSkeleton, cn, fmt } from '@/ui'
import { blockHashes, commonPrefixBlocks } from '@/engine/kv'
import { renderTemplate } from '@/engine/templates'
import type { ChatMessage, Tokenizer, ToolDef } from '@/engine/types'
import { WEATHER_TOOL } from '../request/requestModel'
import { DEFAULT_USER_TEXT } from '../request/responseModel'
import { splitByLiterals } from './promptTokens'

type CaseId = 'gen' | 'foreign' | 'space'

export interface TemplatePitfallsProps {
  tokenizer: Tokenizer | null
  systemText: string
  userText: string
}

export function TemplatePitfalls({ tokenizer, systemText, userText }: TemplatePitfallsProps) {
  const [tab, setTab] = useState<CaseId>('gen')
  const messages = useMemo<ChatMessage[]>(() => {
    const m: ChatMessage[] = []
    if (systemText.trim()) m.push({ role: 'system', content: systemText })
    m.push({ role: 'user', content: userText })
    return m
  }, [systemText, userText])

  return (
    <Figure
      id="template-figure-pitfalls"
      title="Что сломается, если ошибиться в шаблоне"
      caption="Попробуйте: откройте вкладку «Лишний символ» и переставьте лишний символ из system в конец вопроса. Чем позже в промпте отличие, тем больше блоков prefix cache выживает."
    >
      {/* на узком экране вкладки переносятся на вторую строку, а не уезжают за край */}
      <Tabs<CaseId>
        aria-label="Ошибки шаблона"
        value={tab}
        onChange={setTab}
        wrap
        tabs={[
          { id: 'gen', label: 'Без приглашения' },
          { id: 'foreign', label: 'Чужие маркеры' },
          { id: 'space', label: 'Лишний символ' },
        ]}
      >
        {(active) =>
          !tokenizer ? (
            <TokenStripSkeleton count={30} />
          ) : active === 'gen' ? (
            <GenPromptCase tk={tokenizer} messages={messages} userText={userText} />
          ) : active === 'foreign' ? (
            <ForeignCase tk={tokenizer} messages={messages} />
          ) : (
            <SpaceCase tk={tokenizer} systemText={systemText} userText={userText} />
          )
        }
      </Tabs>
    </Figure>
  )
}

/* ---------------------------------------------------------------------------
 * 1. add_generation_prompt
 * ------------------------------------------------------------------------- */

function Verdict({ ok, children }: { ok: boolean; children: ReactNode }) {
  const Icon = ok ? CheckCircleIcon : XCircleIcon
  return (
    <p className="flex items-start gap-2 text-[14px] leading-snug text-ink">
      <Icon size={18} aria-hidden className={cn('mt-px shrink-0', ok ? 'text-good' : 'text-bad')} />
      <span>
        <span className="font-medium">{ok ? 'Так правильно:' : 'Ошибка:'}</span> {children}
      </span>
    </p>
  )
}

function Col({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <div className="grid min-w-0 content-start gap-2.5 rounded-panel border border-line px-3.5 py-3">
      <div className="text-[14px] font-semibold text-ink">{title}</div>
      {children}
    </div>
  )
}

function Label({ children }: { children: ReactNode }) {
  return <div className="text-[12.5px] text-ink-3">{children}</div>
}

function GenPromptCase({ tk, messages, userText }: { tk: Tokenizer; messages: ChatMessage[]; userText: string }) {
  const data = useMemo(() => {
    const withGen = tk.encode(renderTemplate({ family: 'qwen3', messages, addGenerationPrompt: true }).text)
    const noGen = tk.encode(renderTemplate({ family: 'qwen3', messages, addGenerationPrompt: false }).text)
    const isDefault = userText.trim() === DEFAULT_USER_TEXT
    const contA = tk.encode(isDefault ? 'KV cache хранит ключи и значения прошлых токенов…' : 'Хороший вопрос. Коротко…')
    const contB = tk.encode('<|im_start|>user\nИ ещё вопрос: а как это устроено на практике?<|im_end|>')
    return { tailA: withGen.slice(-7), tailB: noGen.slice(-7), contA, contB }
  }, [tk, messages, userText])
  return (
    <div className="grid gap-3">
      <p className="max-w-[78ch] text-[15px] leading-relaxed text-ink">
        Флаг <code>add_generation_prompt</code> дописывает в конец промпта начало реплики ассистента:{' '}
        <code>{'<|im_start|>assistant\\n'}</code>. Без него промпт кончается на закрытой реплике пользователя, и модель сама
        решает, кто говорит дальше.
      </p>
      <div className="grid gap-3 @min-[640px]:grid-cols-2">
        <Col title="С приглашением">
          <Label>конец промпта</Label>
          <TokenStrip tokens={data.tailA} variant="prompt" size="sm" />
          <Label>продолжение модели (иллюстрация)</Label>
          <TokenStrip tokens={data.contA} variant="generated" size="sm" />
          <Verdict ok>модель видит открытую реплику ассистента и отвечает от его лица.</Verdict>
        </Col>
        <Col title="Без приглашения">
          <Label>конец промпта</Label>
          <TokenStrip tokens={data.tailB} variant="prompt" size="sm" />
          <Label>продолжение модели (иллюстрация)</Label>
          <TokenStrip tokens={data.contB} variant="generated" size="sm" />
          <Verdict ok={false}>модель может открыть новую реплику пользователя и продолжить диалог за него.</Verdict>
        </Col>
      </div>
      <p className="text-[13px] text-ink-3">
        Токены промпта настоящие (Qwen3), продолжения условные: так ведёт себя модель, которой не показали, что пора отвечать.
      </p>
    </div>
  )
}

/* ---------------------------------------------------------------------------
 * 2. Чужие спецтокены: шаблон Llama 3.1 для модели Qwen3
 * ------------------------------------------------------------------------- */

function ForeignCase({ tk, messages }: { tk: Tokenizer; messages: ChatMessage[] }) {
  const data = useMemo(() => {
    const llama = renderTemplate({ family: 'llama3', messages })
    const toks = tk.encode(llama.text)
    // где в строке стоят маркеры Llama: литералы спецтокенов и их копии в хвосте приглашения
    const spans: Array<[number, number]> = []
    let off = 0
    for (const s of llama.segments) {
      if (s.kind === 'special') spans.push([off, off + s.text.length])
      else if (s.kind === 'gen-prompt') {
        let o = off
        for (const part of splitByLiterals(s.text, 'llama3')) {
          if (part.special) spans.push([o, o + part.text.length])
          o += part.text.length
        }
      }
      off += s.text.length
    }
    const markerTokens = spans.reduce((n, [a, b]) => n + tk.encode(llama.text.slice(a, b)).length, 0)
    // куски словаря Qwen3, на которые развалились маркеры (подсветка в ленте)
    const marked = toks.flatMap((t, i) =>
      t.start !== undefined && t.end !== undefined && spans.some(([a, b]) => t.start! < b && t.end! > a) ? [i] : [],
    )
    // показываем ленту до конца шестого маркера (system и заголовок user), дальше «ещё N токенов». Но не больше 96 чипов:
    // длинный system (вкладка «Длинный документ») иначе выводил больше тысячи чипов, а куски маркеров видны уже в первых
    // ~25 токенах. Диалог по умолчанию (76 токенов до конца шестого маркера) потолок не задевает.
    const upTo = spans[Math.min(5, spans.length - 1)]
    const after = upTo ? toks.findIndex((t) => t.start !== undefined && t.start >= upTo[1]) : -1
    const limit = Math.min(Math.max(22, after < 0 ? toks.length : after), 96)
    const own = tk.encode(renderTemplate({ family: 'qwen3', messages }).text)
    return { toks, markers: spans.length, markerTokens, marked, limit, own }
  }, [tk, messages])
  return (
    <div className="grid gap-3">
      <p className="max-w-[78ch] text-[15px] leading-relaxed text-ink">
        Сервер с моделью Qwen3 по ошибке применил шаблон Llama 3.1. Токенизатор Qwen3 не знает маркера{' '}
        <code>{'<|start_header_id|>'}</code>: для него это просто строка, и она режется на обычные куски.
      </p>
      <div className="grid gap-3 @min-[640px]:grid-cols-2">
        <Col title="Шаблон Llama 3.1 в словаре Qwen3">
          <Label>Подсвечены куски, на которые развалились маркеры Llama.</Label>
          <TokenStrip
            tokens={data.toks}
            size="sm"
            limit={data.limit}
            highlightIndex={data.marked}
            label={`Шаблон Llama 3.1 в словаре Qwen3: ${fmt.tokens(data.toks.length)}, подсвечены куски маркеров`}
          />
          <Verdict ok={false}>
            {fmt.int(data.markers)} маркеров разметки превратились в {fmt.tokens(data.markerTokens)}. Модель видит набор символов
            вместо границ реплик.
          </Verdict>
        </Col>
        <Col title="Свой шаблон Qwen3">
          <TokenStrip tokens={data.own} size="sm" limit={14} />
          <Verdict ok>каждый маркер это один токен, на котором модель училась: {'<|im_start|>'} = 151644.</Verdict>
        </Col>
      </div>
      <p className="text-[13px] text-ink-3">
        Ошибки API не будет: запрос пройдёт, ответ придёт. Просто модель станет заметно хуже понимать, где чья реплика, и может
        начать отвечать за пользователя или не останавливаться.
      </p>
    </div>
  )
}

/* ---------------------------------------------------------------------------
 * 3. Лишний символ ломает prefix cache
 * ------------------------------------------------------------------------- */

type Where = 'system' | 'tools' | 'end'

const WHERE_TEXT: Record<Where, ReactNode> = {
  system: (
    <>
      Клиент прислал system с лишним переводом строки в конце: <code>{'"Ты полезный ассистент.\\n"'}</code>.
    </>
  ),
  tools: (
    <>
      В описании инструмента оказался лишний пробел: <code>{'"Получить текущую погоду в городе "'}</code>.
    </>
  ),
  end: <>Клиент добавил пробел в конец вопроса пользователя.</>,
}

function SpaceCase({ tk, systemText, userText }: { tk: Tokenizer; systemText: string; userText: string }) {
  const [where, setWhere] = useState<Where>('system')
  const sys = systemText.trim() ? systemText : 'Ты полезный ассистент.'
  const data = useMemo(() => {
    const tool2: ToolDef = { ...WEATHER_TOOL, function: { ...WEATHER_TOOL.function, description: WEATHER_TOOL.function.description + ' ' } }
    const build = (s: string, u: string, tool: ToolDef) =>
      tk.encode(
        renderTemplate({
          family: 'qwen3',
          messages: [
            { role: 'system', content: s },
            { role: 'user', content: u },
          ],
          tools: [tool],
        }).text,
      )
    const a = build(sys, userText, WEATHER_TOOL)
    const b =
      where === 'system' ? build(sys + '\n', userText, WEATHER_TOOL) : where === 'tools' ? build(sys, userText, tool2) : build(sys, userText + ' ', WEATHER_TOOL)
    let diff = 0
    while (diff < a.length && diff < b.length && a[diff].id === b[diff].id) diff++
    const ba = blockHashes(a.map((t) => t.id), 16)
    const bb = blockHashes(b.map((t) => t.id), 16)
    const matched = commonPrefixBlocks(ba, bb)
    return { a, b, ba, bb, matched, diff }
  }, [tk, sys, userText, where])

  const { a, b, ba, bb, matched, diff } = data
  const fullB = bb.filter((x) => x.full).length
  const k = Math.floor(diff / 16)
  const inTail = !bb[k]?.full
  const reuse = matched * 16

  return (
    <div className="grid gap-3.5">
      <p className="max-w-[78ch] text-[15px] leading-relaxed text-ink">
        Prefix cache ищет совпадение по токенам, блоками по 16. Запрос здесь с инструментом get_weather, чтобы промпт был длиннее,
        как у агентов: {fmt.tokens(a.length)}. Поменяем в нём один символ.
      </p>
      <Segmented<Where>
        label="Где лишний символ"
        wrap
        value={where}
        onChange={setWhere}
        options={[
          { value: 'system', label: 'в system' },
          { value: 'tools', label: 'в описании tools' },
          { value: 'end', label: 'в конце вопроса' },
        ]}
      />
      <p className="text-[14px] text-ink-2">{WHERE_TEXT[where]}</p>

      <div className="grid gap-2">
        <BlockRow label="Эталон" blocks={ba} />
        <BlockRow label="С ошибкой" blocks={bb} matched={matched} />
        <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] text-ink-2">
          <li className="inline-flex items-center gap-1.5">
            <CheckCircleIcon size={16} aria-hidden className="text-good" /> блок совпал: KV берётся из кэша
          </li>
          <li className="inline-flex items-center gap-1.5">
            <XCircleIcon size={16} aria-hidden className="text-bad" /> промах: блок считается заново
          </li>
          <li className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-3.5 w-5 rounded-[4px] border border-dashed border-line-strong" /> хвост: неполный блок не кэшируется
          </li>
        </ul>
      </div>

      <div className="rounded-panel border border-line px-3.5 py-3">
        {inTail ? (
          <p className="text-[14px] leading-relaxed text-ink">
            Первое отличие на токене {fmt.int(diff + 1)}, в неполном хвосте. Хвост и так не кэшируется, поэтому все{' '}
            {fmt.int(fullB)} полных блоков совпали.
          </p>
        ) : (
          <div className="grid gap-2">
            <div className="text-[14px] text-ink">
              Первое отличие на токене {fmt.int(diff + 1)}, это блок {fmt.int(k)}. Он и все блоки после него получают другие хэши.
            </div>
            <div className="grid gap-1.5 @min-[640px]:grid-cols-[auto_minmax(0,1fr)] @min-[640px]:items-start @min-[640px]:gap-x-3">
              <Label>эталон, блок {k}</Label>
              <TokenStrip tokens={a.slice(k * 16, k * 16 + 16)} size="sm" highlightIndex={diff - k * 16} />
              <Label>с ошибкой, блок {k}</Label>
              <TokenStrip tokens={b.slice(k * 16, k * 16 + 16)} size="sm" highlightIndex={diff - k * 16} />
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-3 gap-x-5">
        <Stat label="Совпало" value={`${fmt.int(matched)} из ${fmt.int(fullB)}`} hint="полных блоков" />
        <Stat label="Из кэша" value={fmt.int(reuse)} unit="ток." />
        <Stat label="Заново" value={fmt.int(b.length - reuse)} unit="ток." hint="считаются в prefill" />
      </div>
    </div>
  )
}

function BlockRow({ label, blocks, matched }: { label: string; blocks: ReturnType<typeof blockHashes>; matched?: number }) {
  return (
    <div className="grid items-center gap-1.5 @min-[640px]:grid-cols-[88px_minmax(0,1fr)]">
      <span className="text-[12.5px] text-ink-3">{label}</span>
      <ol className="flex flex-wrap gap-1" aria-label={label}>
        {blocks.map((bl) => {
          const status = !bl.full ? 'tail' : matched === undefined ? 'ref' : bl.index < matched ? 'hit' : 'miss'
          return (
            <li
              key={bl.index}
              title={bl.full ? `блок ${bl.index}: хэш ${bl.hash}` : `хвост: ${bl.tokenIds.length} токенов, без хэша`}
              aria-label={bl.full ? `блок ${bl.index}, ${status === 'hit' ? 'совпал' : status === 'miss' ? 'промах' : `хэш ${bl.hash}`}` : 'хвост без хэша'}
              className={cn(
                'inline-flex h-8 min-w-[52px] items-center justify-center gap-1 rounded-[6px] px-1.5 font-mono text-[11.5px] text-ink tabular-nums',
                status === 'tail' && 'border border-dashed border-line-strong text-ink-2',
                status === 'ref' && 'bg-surface-2 shadow-[inset_0_0_0_1px_var(--line)]',
                status === 'hit' && 'bg-good/12',
                status === 'miss' && 'bg-bad/12',
              )}
            >
              {status === 'hit' && <CheckCircleIcon size={14} aria-hidden className="text-good" />}
              {status === 'miss' && <XCircleIcon size={14} aria-hidden className="text-bad" />}
              {bl.full ? bl.hash.slice(0, 4) : `+${bl.tokenIds.length}`}
            </li>
          )
        })}
      </ol>
    </div>
  )
}

