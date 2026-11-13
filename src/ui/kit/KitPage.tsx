import { createContext, useContext, useLayoutEffect, useMemo, useState, type ReactNode } from 'react'
import {
  ArrowCounterClockwiseIcon,
  ArrowLeftIcon,
  ArrowRightIcon,
  GearSixIcon,
  PaperPlaneRightIcon,
  SlidersHorizontalIcon,
} from '@phosphor-icons/react'
import {
  BarChart,
  Badge,
  Button,
  ButtonLink,
  Callout,
  Chapter,
  ChapterLink,
  CodeBlock,
  Deeper,
  Figure,
  Formula,
  Heatmap,
  Kbd,
  LineChart,
  MatrixView,
  Panel,
  Player,
  Popover,
  Prose,
  ScrollX,
  Segmented,
  Select,
  SimulatorLink,
  Skeleton,
  Slider,
  Stat,
  StatusIcon,
  Tabs,
  Takeaways,
  Term,
  TextArea,
  Toggle,
  TokenChip,
  TokenStrip,
  TokenStripSkeleton,
  Tooltip,
  fmt,
  legendGradient,
  tint,
  usePlayer,
  useReducedMotion,
  type MatrixCell,
} from '@/ui'
import { ThemeSwitch } from '@/ui/Nav'
import { cn } from '@/ui/cn'
import {
  CANDIDATES,
  DAEMONSET_YAML,
  GENERATED_TOKENS,
  HEAD_TOKENS,
  HEAD_WEIGHTS,
  JINJA_SNIPPET,
  PROMPT_TOKENS,
  PYTHON_SNIPPET,
  RENDERED_TEMPLATE,
  REQUEST_JSON,
  TOY,
  TTFT_SERIES,
  VLLM_BASH,
  WORDS,
  rooflinePoints,
} from './samples'

/* ===========================================================================
 * Режим показа: одна тема (как на сайте) или обе рядом.
 * ========================================================================= */

type Mode = 'single' | 'both'
const ModeCtx = createContext<Mode>('single')

const SECTIONS = [
  { id: 'colors', title: 'Цвета' },
  { id: 'type', title: 'Типографика' },
  { id: 'chapter', title: 'Глава' },
  { id: 'figure', title: 'Figure' },
  { id: 'controls', title: 'Контролы' },
  { id: 'player', title: 'Player' },
  { id: 'tokens', title: 'Токены' },
  { id: 'matrix', title: 'MatrixView' },
  { id: 'heatmap', title: 'Heatmap' },
  { id: 'bars', title: 'BarChart' },
  { id: 'lines', title: 'LineChart' },
  { id: 'formula', title: 'Formula' },
  { id: 'code', title: 'CodeBlock' },
  { id: 'blocks', title: 'Таблица и панели' },
  { id: 'numbers', title: 'Stat и fmt' },
  { id: 'loading', title: 'Skeleton' },
] as const

function KitSection({ id, title, note, children }: { id: string; title: string; note?: ReactNode; children: ReactNode }) {
  const mode = useContext(ModeCtx)
  return (
    <section id={`kit-${id}`} className="scroll-mt-20 border-t border-line py-10 first:border-t-0 first:pt-2">
      <h2 className="text-[1.625rem] leading-tight font-[650] tracking-[-0.015em] text-ink">{title}</h2>
      {note && <p className="mt-2 max-w-[72ch] text-[15px] text-ink-2">{note}</p>}
      <div className={mode === 'both' ? 'mt-6 grid grid-cols-1 gap-4 xl:grid-cols-2' : 'mt-6 min-w-0'}>
        {mode === 'both' ? (
          <>
            <ThemePanel theme="light">{children}</ThemePanel>
            <ThemePanel theme="dark">{children}</ThemePanel>
          </>
        ) : (
          children
        )}
      </div>
    </section>
  )
}

function ThemePanel({ theme, children }: { theme: 'light' | 'dark'; children: ReactNode }) {
  return (
    <div data-theme={theme} className="min-w-0 rounded-panel border border-line bg-bg p-4 sm:p-5">
      <div className="mb-4 text-xs font-medium text-ink-3">{theme === 'light' ? 'Светлая тема' : 'Тёмная тема'}</div>
      {children}
    </div>
  )
}

/** Маленькая подпись к примеру. */
function Demo({ label, children, className }: { label: string; children: ReactNode; className?: string }) {
  return (
    <div className={cn('min-w-0', className)}>
      <div className="mb-2 font-mono text-xs text-ink-3">{label}</div>
      {children}
    </div>
  )
}

/* ===========================================================================
 * Страница
 * ========================================================================= */

export default function KitPage() {
  const [mode, setMode] = useState<Mode>('single')
  const reduce = useReducedMotion()
  const go = (id: string) => document.getElementById(`kit-${id}`)?.scrollIntoView({ behavior: reduce ? 'instant' : 'smooth', block: 'start' })
  // тело в фигурных скобках: в новых Chromium scrollTo возвращает Promise, а эффект может вернуть только функцию очистки
  useLayoutEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' })
  }, [])

  return (
    <ModeCtx.Provider value={mode}>
      <div className="min-h-[100dvh] bg-bg text-ink">
        <header className="sticky top-0 z-40 border-b border-line bg-[color-mix(in_oklab,var(--bg)_92%,transparent)] backdrop-blur-md">
          <div className="mx-auto flex h-14 max-w-[1440px] items-center gap-3 px-4 md:px-6">
            <ButtonLink href="#" variant="ghost" size="sm" icon={ArrowLeftIcon}>
              На сайт
            </ButtonLink>
            <span className="hidden truncate text-[15px] font-semibold sm:block">Витрина ui-kit</span>
            <div className="ml-auto flex items-center gap-2">
              <Segmented<Mode>
                size="sm"
                aria-label="Показ тем"
                value={mode}
                onChange={setMode}
                className="hidden md:grid"
                options={[
                  { value: 'single', label: 'Одна тема' },
                  { value: 'both', label: 'Обе рядом' },
                ]}
              />
              <ThemeSwitch />
            </div>
          </div>
        </header>

        <div className="mx-auto grid max-w-[1440px] gap-10 px-4 py-8 md:px-6 desk:grid-cols-[180px_minmax(0,1fr)]">
          <nav aria-label="Разделы витрины" className="hidden desk:block">
            <ul className="sticky top-20 grid gap-0.5 text-sm">
              {SECTIONS.map((s) => (
                <li key={s.id}>
                  <button type="button" onClick={() => go(s.id)} className="w-full rounded-control px-2.5 py-1 text-left text-ink-2 hover:bg-surface-3/70 hover:text-ink">
                    {s.title}
                  </button>
                </li>
              ))}
            </ul>
          </nav>

          <main className="min-w-0">
            <div className="mb-8 max-w-[72ch]">
              <h1 className="text-[2.25rem] leading-[1.1] font-[650] tracking-tight">Витрина ui-kit</h1>
              <p className="mt-3 text-[17px] text-ink-2">
                Все компоненты сайта на примерах. API описан в <code>src/ui/README.md</code>. Режим «Обе рядом» показывает каждый
                пример в светлой и тёмной теме одновременно. Числа на этой странице условные, кроме отмеченных.
              </p>
            </div>

            <KitSection id="colors" title="Цвета" note="Один акцент для интерфейса, фиксированные цвета сущностей, статусы только с иконкой и словом.">
              <ColorsDemo />
            </KitSection>
            <KitSection id="type" title="Типографика" note="Onest для текста и заголовков, JetBrains Mono для кода, токенов и чисел в фигурах.">
              <TypeDemo />
            </KitSection>
            <KitSection id="chapter" title="Глава" note="Chapter берёт номер, заголовок и лид из реестра. Ниже типичная композиция: проза, врезки, «Глубже», «Запомнить».">
              <ChapterDemo />
            </KitSection>
            <KitSection id="figure" title="Figure" note="Рамка интерактивной фигуры: заголовок, тело, панель контролов, подпись «Попробуйте: …». Ширины narrow, wide, full.">
              <FigureDemo />
            </KitSection>
            <KitSection id="controls" title="Контролы" note="Подпись всегда над полем, фокус акцентом, всё доступно с клавиатуры.">
              <ControlsDemo />
            </KitSection>
            <KitSection id="player" title="Player" note="Все пошаговые демо используют Player и usePlayer. Пробел, ← и → работают, когда фокус внутри фигуры.">
              <PlayerDemo />
            </KitSection>
            <KitSection id="tokens" title="Токены" note="Пробел «·», перевод строки «↵», таб «→», неполный UTF-8 байтами. Спецтокены в пунктирной рамке. Наведите на токен.">
              <TokensDemo />
            </KitSection>
            <KitSection id="matrix" title="MatrixView" note="Упрощённый пример по словам. Нажмите на ячейку весов, чтобы выделить строку.">
              <MatrixDemo />
            </KitSection>
            <KitSection id="heatmap" title="Heatmap" note="Последовательная шкала DESIGN.md, подпись при наведении, стрелки с клавиатуры. Условный паттерн головы.">
              <HeatmapDemo />
            </KitSection>
            <KitSection id="bars" title="BarChart" note="Тонкие столбики, значения у конца, отсечённые приглушены, замаскированные зачёркнуты с иконкой. Условные вероятности.">
              <BarsDemo />
            </KitSection>
            <KitSection id="lines" title="LineChart" note="Одна ось Y, легенда и прямые подписи, crosshair-тултип по всем сериям, аннотации.">
              <LinesDemo />
            </KitSection>
            <KitSection id="formula" title="Formula" note="KaTeX. Макросы \cq{}, \ck{}, \cv{} подкладывают цвет сущности под символ.">
              <FormulaDemo />
            </KitSection>
            <KitSection id="code" title="CodeBlock" note="Подсветка json, yaml, bash, jinja, python, text; спецтокены в пунктирной рамке; кнопка «Копировать».">
              <CodeDemo />
            </KitSection>
            <KitSection
              id="blocks"
              title="Таблица и панели"
              note="Таблица данных .data-table, панель пояснения Panel (inset, outline), StatusIcon, ScrollX для широкого содержимого и слоты Figure: inputs, player, note, source."
            >
              <BlocksDemo />
            </KitSection>
            <KitSection id="numbers" title="Stat и fmt" note="Числа по-русски: неразрывный пробел в тысячах, запятая, настоящий минус.">
              <NumbersDemo />
            </KitSection>
            <KitSection id="loading" title="Skeleton" note="Скелетон в форме будущего результата, например ленты токенов, пока грузится токенизатор.">
              <LoadingDemo />
            </KitSection>
          </main>
        </div>
      </div>
    </ModeCtx.Provider>
  )
}

/* ===========================================================================
 * Демо
 * ========================================================================= */

function Swatch({ name, cssVar, note, dashed }: { name: string; cssVar: string; note?: string; dashed?: boolean }) {
  return (
    <div className="flex min-w-0 items-center gap-3">
      <span
        aria-hidden
        className="h-9 w-9 shrink-0 rounded-[8px] shadow-[inset_0_0_0_1px_var(--line)]"
        style={dashed ? { border: `1.5px dashed var(${cssVar})` } : { background: `var(${cssVar})` }}
      />
      <span className="grid min-w-0 leading-tight">
        <span className="truncate font-mono text-[12.5px] text-ink">{name}</span>
        {note && <span className="truncate text-xs text-ink-3">{note}</span>}
      </span>
    </div>
  )
}

function ColorsDemo() {
  const grid = 'grid grid-cols-[repeat(auto-fill,minmax(190px,1fr))] gap-x-4 gap-y-3'
  return (
    <div className="grid min-w-0 grid-cols-1 gap-8">
      <Demo label="нейтральные">
        <div className={grid}>
          <Swatch name="--bg" cssVar="--bg" note="фон страницы" />
          <Swatch name="--surface" cssVar="--surface" note="фигуры, панели" />
          <Swatch name="--surface-2" cssVar="--surface-2" note="код, поля, ячейки" />
          <Swatch name="--surface-3" cssVar="--surface-3" note="hover, выделение" />
          <Swatch name="--ink" cssVar="--ink" note="основной текст" />
          <Swatch name="--ink-2" cssVar="--ink-2" note="вторичный текст" />
          <Swatch name="--ink-3" cssVar="--ink-3" note="оси, приглушённое" />
          <Swatch name="--line-strong" cssVar="--line-strong" note="оси, базовые линии" />
          <Swatch name="--grid" cssVar="--grid" note="сетка графиков" />
        </div>
      </Demo>
      <Demo label="акцент интерфейса">
        <div className={grid}>
          <Swatch name="--accent" cssVar="--accent" note="кнопки, ссылки, фокус" />
          <Swatch name="--accent-hover" cssVar="--accent-hover" />
          <Swatch name="--accent-soft" cssVar="--accent-soft" note="фон выделения" />
        </div>
      </Demo>
      <Demo label="сущности данных (Tailwind: bg-prefill, text-q …; JS: cssVar('k'), tint('v'))">
        <div className={grid}>
          <Swatch name="--c-prefill" cssVar="--c-prefill" note="prefill, промпт" />
          <Swatch name="--c-decode" cssVar="--c-decode" note="decode, генерация" />
          <Swatch name="--c-q" cssVar="--c-q" note="Query" />
          <Swatch name="--c-k" cssVar="--c-k" note="Key" />
          <Swatch name="--c-v" cssVar="--c-v" note="Value" />
          <Swatch name="--c-tool" cssVar="--c-tool" note="tool call" />
          <Swatch name="--c-reasoning" cssVar="--c-reasoning" note="рассуждение" />
          <Swatch name="--c-special" cssVar="--c-special" note="спецтокены" dashed />
        </div>
      </Demo>
      <Demo label="статусы: всегда с иконкой и подписью">
        <div className="flex flex-wrap gap-2">
          <Badge tone="good">попадание в кэш</Badge>
          <Badge tone="warn">близко к лимиту</Badge>
          <Badge tone="bad">нарушение SLO</Badge>
        </div>
      </Demo>
      <Demo label="шкалы heatmap: seqColor(t), divColor(t)">
        <div className="grid max-w-md gap-3 text-xs text-ink-3">
          <div className="flex items-center gap-3">
            <span className="w-24">последовательная</span>
            <span className="h-3 flex-1 rounded-full" style={{ background: legendGradient('seq') }} />
          </div>
          <div className="flex items-center gap-3">
            <span className="w-24">расходящаяся</span>
            <span className="h-3 flex-1 rounded-full" style={{ background: legendGradient('diverging') }} />
          </div>
        </div>
      </Demo>
    </div>
  )
}

function TypeDemo() {
  return (
    <div className="grid min-w-0 grid-cols-1 gap-6">
      <div className="text-[clamp(2.25rem,4.5vw,3.75rem)] leading-[1.05] font-[650] tracking-tight">Что происходит после Enter</div>
      <div className="text-[2rem] leading-[1.1] font-[650] tracking-[-0.02em] desk:text-[2.5rem]">
        <span className="mr-[0.4em] font-normal text-ink-3">6</span>Attention и Q/K/V
      </div>
      <div className="text-[1.375rem] leading-[1.3] font-semibold">Заголовок h3 внутри главы</div>
      <p className="max-w-[68ch] text-[1.1875rem] leading-[1.6] text-ink-2">
        Лид главы: одно-три предложения простыми словами о том, что это и зачем это знать.
      </p>
      <Prose>
        <p>
          Основной текст: 17px, межстрочный 1,65, ширина до 68 знаков. В тексте бывает <code>inline code</code>, <strong>полужирное</strong>,{' '}
          <em>курсив</em> и <a href="#kit">ссылка</a>. Клавиши: <Kbd>Пробел</Kbd> <Kbd>←</Kbd> <Kbd>→</Kbd>.
        </p>
        <ul>
          <li>маркированный список с приглушёнными маркерами;</li>
          <li>второй пункт.</li>
        </ul>
        <ol>
          <li>нумерованный список;</li>
          <li>с табличными цифрами.</li>
        </ol>
      </Prose>
      <div className="text-caption">Подпись или легенда: 14px, --ink-2.</div>
      <div className="text-axis">Подпись оси: 12px, --ink-3, табличные цифры 0 1 024 2 048</div>
      <div className="font-mono text-[13px]">JetBrains Mono: &lt;|im_start|&gt;user Ċ 151644 0xD0 0x9F</div>
    </div>
  )
}

function ChapterDemo() {
  return (
    <div className="overflow-clip rounded-panel border border-dashed border-line-strong">
      <Chapter
        id="attention"
        lead={
          <>
            Attention позволяет каждому токену собрать информацию из предыдущих. Каждый токен делает три вектора: <b>Query</b>,{' '}
            <b>Key</b> и <b>Value</b>.
          </>
        }
      >
        <p>
          Абзац прозы прямо внутри <code>&lt;Chapter&gt;</code>. Термин из глоссария: <Term t="KV cache" />, а вот <Term t="prefill">prefill</Term> и{' '}
          <Term t="TTFT" />. Ссылка на главу: <ChapterLink to="kv-cache" />.
        </p>
        <Callout kind="intuition">Поиск в библиотеке: Query это запрос, Key это корешки книг, Value это содержимое книги.</Callout>
        <Callout kind="production" title="префиксный кэш">
          <p>
            В vLLM он включён по умолчанию; флаг <code>--enable-prefix-caching</code>.
          </p>
        </Callout>
        <Callout kind="pitfall">Attention не «понимает смысл», он считает взвешенную сумму векторов Value.</Callout>
        <Callout kind="precise">Веса считаются как softmax(QKᵀ/√d_k) по строкам; маска ставит −∞ над диагональю.</Callout>
        <Deeper title="зачем делить на √d_k">
          <p>Без масштаба скалярные произведения растут с размерностью, и softmax становится почти one-hot: градиенты гаснут.</p>
        </Deeper>
        <SimulatorLink stage="prefill" />
        <ChapterLink to="routing" variant="card" />
        <Takeaways
          items={[
            'Каждый токен смотрит только на предыдущие: это causal mask.',
            <>
              K и V прошлых токенов не меняются, поэтому их кэшируют: это <b>KV cache</b>.
            </>,
            'Головы специализируются: одна смотрит на предыдущий токен, другая на начало.',
          ]}
        />
      </Chapter>
    </div>
  )
}

function FigureDemo() {
  const [n, setN] = useState(2048)
  const [on, setOn] = useState(true)
  const [view, setView] = useState<TokenView>('ids')
  // .flow: та же сетка, что в главе, поэтому ширины narrow / wide / full здесь настоящие
  return (
    <div className="flow min-w-0">
      <Figure
        title="Фигура с панелью контролов снизу (width='wide')"
        actions={<Badge tone="prefill">prefill</Badge>}
        controls={
          <>
            <Slider label="Длина промпта" value={n} min={16} max={131072} scale="log" onChange={setN} format={(v) => fmt.tokens(v)} />
            <Toggle label="Prefix cache" checked={on} onChange={setOn} hint="Тёплый кэш пропускает часть prefill" />
          </>
        }
        caption="Попробуйте: потяните длину промпта, шкала логарифмическая."
      >
        <div className="grid gap-4 @md:grid-cols-3">
          <Stat label="Токенов в промпте" value={fmt.int(n)} />
          <Stat label="Из кэша" value={on ? fmt.int(Math.floor(n * 0.6)) : '0'} hint={`условно ${fmt.pct(0.6)}`} />
          <Stat label="KV на запрос" value={fmt.bytes(n * 147_456).split(' ')[0]} unit={fmt.bytes(n * 147_456).split(' ')[1]} hint="Qwen3-8B, BF16" />
        </div>
      </Figure>
      <Figure
        title="Контролы сбоку (controlsPosition='side', width='full')"
        width="full"
        controlsPosition="side"
        controls={<SideControls view={view} onView={setView} />}
      >
        <TokenStrip tokens={PROMPT_TOKENS} variant="prompt" showIds={view === 'ids'} showBytes={view === 'bytes'} />
      </Figure>
      <Figure title="Узкая фигура (width='narrow')" width="narrow" caption="Подпись без интерактива.">
        <p className="text-[15px] text-ink-2">Тело фигуры является container query контейнером: внутри работают @sm: @md: @lg:.</p>
      </Figure>
    </div>
  )
}

type TokenView = 'ids' | 'bytes' | 'text'

function SideControls({ view, onView }: { view: TokenView; onView: (v: TokenView) => void }) {
  return (
    <>
      <Segmented
        label="Показать"
        value={view}
        onChange={onView}
        options={[
          { value: 'text', label: 'Текст' },
          { value: 'ids', label: 'ID' },
          { value: 'bytes', label: 'Байты' },
        ]}
      />
      <Select
        label="Модель"
        value="qwen3-8b"
        onChange={() => {}}
        options={[
          { value: 'qwen3-8b', label: 'Qwen3-8B', group: 'Qwen3' },
          { value: 'qwen3-32b', label: 'Qwen3-32B', group: 'Qwen3' },
          { value: 'llama-3.1-70b', label: 'Llama-3.1-70B', group: 'Llama' },
        ]}
      />
    </>
  )
}

function ControlsDemo() {
  const [temp, setTemp] = useState(0.7)
  const [topK, setTopK] = useState(40)
  const [thinking, setThinking] = useState(true)
  const [tools, setTools] = useState(false)
  const [deploy, setDeploy] = useState<'agg' | 'disagg'>('agg')
  const [gpu, setGpu] = useState('h100-sxm')
  const [tab, setTab] = useState<'qwen3' | 'llama3' | 'deepseek-r1' | 'gpt-oss'>('qwen3')
  const [text, setText] = useState('Объясни простыми словами, что такое KV cache и зачем он нужен.')
  return (
    <div className="grid min-w-0 grid-cols-1 gap-8">
      <Demo label="Button: primary, secondary, ghost; sm, md, lg; icon, iconOnly">
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="primary" icon={PaperPlaneRightIcon}>
            Отправить
          </Button>
          <Button>Сбросить</Button>
          <Button variant="ghost" iconRight={ArrowRightIcon}>
            Подробнее
          </Button>
          <Button size="sm" icon={SlidersHorizontalIcon}>
            Маленькая
          </Button>
          <Button size="lg" variant="primary">
            Большая
          </Button>
          <Button iconOnly icon={ArrowCounterClockwiseIcon} aria-label="Сначала" />
          <Button disabled>Недоступна</Button>
        </div>
      </Demo>
      <div className="grid gap-6 md:grid-cols-2">
        <Slider label="Temperature" value={temp} min={0} max={2} step={0.05} onChange={setTemp} format={(v) => fmt.num(v, 2)} />
        <Slider label="top-k" value={topK} min={1} max={100} onChange={setTopK} hint="0 или меньше выключает отсечение" />
        <Toggle label="Thinking" checked={thinking} onChange={setThinking} hint="Рассуждение в <think>" />
        <Toggle label="Tools" checked={tools} onChange={setTools} />
        <Segmented
          label="Деплой"
          value={deploy}
          onChange={setDeploy}
          options={[
            { value: 'agg', label: 'Агрегированный' },
            { value: 'disagg', label: 'Дезагрегированный' },
          ]}
        />
        <Select
          label="GPU"
          value={gpu}
          onChange={setGpu}
          options={[
            { value: 'h100-sxm', label: 'H100 SXM' },
            { value: 'h200', label: 'H200' },
            { value: 'b200', label: 'B200' },
          ]}
        />
      </div>
      <TextArea label="Сообщение пользователя" value={text} onChange={setText} rows={2} autoGrow counter maxLength={4000} />
      <Tabs
        aria-label="Семейство моделей"
        value={tab}
        onChange={setTab}
        tabs={[
          { id: 'qwen3', label: 'Qwen3' },
          { id: 'llama3', label: 'Llama 3.1' },
          { id: 'deepseek-r1', label: 'DeepSeek-R1' },
          { id: 'gpt-oss', label: 'gpt-oss (Harmony)' },
        ]}
      >
        {(active) => <p className="text-[15px] text-ink-2">Содержимое вкладки «{active}». Стрелки ← → переключают вкладки.</p>}
      </Tabs>
      <Demo label="Badge, Kbd, Tooltip, Popover">
        <div className="flex flex-wrap items-center gap-3">
          <Badge>нейтральный</Badge>
          <Badge tone="accent">текущий шаг</Badge>
          <Badge tone="prefill">prefill</Badge>
          <Badge tone="decode">decode</Badge>
          <Badge tone="q">Q</Badge>
          <Badge tone="k">K</Badge>
          <Badge tone="v">V</Badge>
          <Badge tone="tool">tool call</Badge>
          <Badge tone="special">спецтокен</Badge>
          <Kbd>Пробел</Kbd>
          <Tooltip content="Time to first token: время до первого токена">
            <span className="cursor-help border-b border-dotted border-ink-3 text-[15px]">TTFT</span>
          </Tooltip>
          <Popover
            label="Параметры сэмплинга"
            content={(close) => (
              <div className="grid w-64 gap-3 p-4">
                <Slider label="Temperature" value={temp} min={0} max={2} step={0.05} onChange={setTemp} format={(v) => fmt.num(v, 2)} />
                <Button size="sm" onClick={close}>
                  Готово
                </Button>
              </div>
            )}
          >
            {(p) => (
              <Button {...p} size="sm" icon={GearSixIcon}>
                Popover
              </Button>
            )}
          </Popover>
        </div>
      </Demo>
    </div>
  )
}

const PIPE = ['Клиент', 'Frontend', 'Preprocessor', 'Router', 'Prefill', 'Decode', 'Postprocessor', 'SSE']

function PlayerDemo() {
  const p = usePlayer(PIPE.length, { intervalMs: 900 })
  return (
    <Figure title="Путь запроса по стадиям" controls={<Player {...p.bind} labels={PIPE} />} controlsPosition="bottom" caption="Попробуйте: нажмите пуск или кликните в фигуру и нажмите пробел.">
      <ol className="flex flex-wrap gap-2">
        {PIPE.map((s, i) => (
          <li
            key={s}
            className="rounded-control px-3 py-1.5 text-sm transition-colors"
            style={{
              background: i === p.step ? 'var(--accent-soft)' : i < p.step ? tint('prefill', 10) : 'var(--surface-2)',
              color: i === p.step ? 'var(--ink)' : 'var(--ink-2)',
              boxShadow: i === p.step ? 'inset 0 0 0 1.5px var(--accent)' : undefined,
            }}
          >
            {s}
          </li>
        ))}
      </ol>
    </Figure>
  )
}

function TokensDemo() {
  const [showIds, setShowIds] = useState(true)
  const [showBytes, setShowBytes] = useState(false)
  const [picked, setPicked] = useState<number | null>(null)
  const all = useMemo(() => [...PROMPT_TOKENS, ...GENERATED_TOKENS], [])
  return (
    <div className="grid min-w-0 grid-cols-1 gap-8">
      <Demo label="TokenChip: default, prompt, generated, muted, highlight, special">
        <div className="flex flex-wrap items-start gap-2">
          <TokenChip token={PROMPT_TOKENS[4]} />
          <TokenChip token={PROMPT_TOKENS[5]} variant="prompt" showId />
          <TokenChip token={GENERATED_TOKENS[3]} variant="generated" showId />
          <TokenChip token={PROMPT_TOKENS[9]} variant="muted" />
          <TokenChip token={PROMPT_TOKENS[10]} highlight />
          <TokenChip token={PROMPT_TOKENS[0]} showId />
          <TokenChip token={GENERATED_TOKENS[13]} showBytes />
        </div>
      </Demo>
      <Figure
        title="TokenStrip: промпт и ответ"
        actions={
          <>
            <Toggle label="ID" checked={showIds} onChange={setShowIds} />
            <Toggle label="Байты" checked={showBytes} onChange={setShowBytes} />
          </>
        }
        caption="Попробуйте: нажмите на токен, стрелки двигают выбор. Эмодзи в конце разрезан на два токена по байтам."
      >
        <TokenStrip
          tokens={all}
          showIds={showIds}
          showBytes={showBytes}
          variantOf={(_t, i) => (i < PROMPT_TOKENS.length ? 'prompt' : 'generated')}
          highlightIndex={picked}
          onClickIndex={setPicked}
        />
        <p className="mt-4 text-sm text-ink-2">
          {picked === null ? 'Ничего не выбрано.' : `Выбран токен ${picked + 1}: ID ${all[picked].id}, байты ${fmt.hex(all[picked].bytes)}.`}
        </p>
      </Figure>
      <Demo label="TokenStrip wrap={false}, limit">
        <TokenStrip tokens={all} wrap={false} limit={14} />
      </Demo>
    </div>
  )
}

function MatrixDemo() {
  const [row, setRow] = useState<number | null>(7)
  const [cell, setCell] = useState<MatrixCell | null>(null)
  const scaled = TOY.scores
  return (
    <div className="grid min-w-0 grid-cols-1 gap-8">
      <Figure title="Веса attention после softmax" caption="Попробуйте: нажмите на ячейку, чтобы выделить строку; наведите, чтобы увидеть значение.">
        <div className="grid min-w-0 grid-cols-1 gap-6">
          <MatrixView
            label="Q = X·W_Q (упрощённо, d_k = 2)"
            data={TOY.q}
            rowLabels={WORDS}
            colLabels={['q₁', 'q₂']}
            digits={1}
            colorRole="q"
            heat="diverging"
            cellSize="sm"
          />
          <MatrixView
            label="Оценки с causal mask"
            data={scaled}
            rowLabels={WORDS}
            colLabels={WORDS}
            digits={1}
            heat="diverging"
            cellSize="sm"
            highlight={row !== null ? { rows: [row] } : undefined}
          />
          <MatrixView
            label="Веса: softmax по строкам"
            data={TOY.weights}
            rowLabels={WORDS}
            colLabels={WORDS}
            digits={2}
            heat="seq"
            domain={[0, 1]}
            cellSize="sm"
            highlight={row !== null ? { rows: [row] } : undefined}
            onCellClick={(c) => setRow(c.row)}
            onCellHover={setCell}
            cellTooltip={(c) => (
              <span>
                «{WORDS[c.row]}» → «{WORDS[c.col]}»: <b className="tabular-nums">{fmt.pct(c.value)}</b>
              </span>
            )}
          />
          <p className="text-sm text-ink-2">
            {cell ? `Наведено: строка ${cell.row + 1}, столбец ${cell.col + 1}, значение ${fmt.num(cell.value, 3)}.` : 'Наведите на ячейку весов.'}
          </p>
        </div>
      </Figure>
      <Demo label="colorRole: q, k, v (без заливки по значению)">
        <div className="flex flex-wrap gap-6">
          {(['q', 'k', 'v'] as const).map((r) => (
            <MatrixView key={r} label={r.toUpperCase()} data={TOY.q.slice(0, 3)} colorRole={r} digits={1} cellSize="sm" />
          ))}
        </div>
      </Demo>
    </div>
  )
}

function HeatmapDemo() {
  const [hover, setHover] = useState<string>('Наведите на ячейку')
  return (
    <Figure title="Голова «предыдущий токен + attention sink»" caption="Попробуйте: наведите на ячейку или сфокусируйтесь на карте и двигайтесь стрелками.">
      <Heatmap
        data={HEAD_WEIGHTS}
        rowLabels={HEAD_TOKENS}
        colLabels={HEAD_TOKENS}
        domain={[0, 1]}
        rowAxisLabel="кто смотрит"
        colAxisLabel="на кого смотрит"
        legendLabel="Вес"
        valueFormat={(v) => fmt.pct(v)}
        isMasked={(r, c) => c > r}
        onHover={(c) => setHover(c ? `${HEAD_TOKENS[c.row]} → ${HEAD_TOKENS[c.col]}: ${fmt.pct(c.value)}` : 'Наведите на ячейку')}
        aria-label="Веса attention условной головы"
      />
      <p className="mt-3 font-mono text-sm text-ink-2">{hover}</p>
    </Figure>
  )
}

function BarsDemo() {
  const [temp, setTemp] = useState(1)
  const data = useMemo(() => {
    const logits = CANDIDATES.map((c) => Math.log(c.value))
    const ex = logits.map((l) => Math.exp(l / Math.max(temp, 0.05)))
    const s = ex.reduce((a, b) => a + b, 0)
    return CANDIDATES.map((c, i) => ({ ...c, value: ex[i] / s, id: c.label }))
  }, [temp])
  return (
    <div className="grid min-w-0 grid-cols-1 gap-8">
      <Figure
        title="Следующий токен после «Москва является столицей»"
        controls={<Slider label="Temperature" value={temp} min={0.1} max={2} step={0.05} onChange={setTemp} format={(v) => fmt.num(v, 2)} />}
        caption="Попробуйте: поменяйте температуру и посмотрите, как перераспределяются вероятности."
      >
        <BarChart data={data} max={1} valueFormat={(v) => fmt.pct(v)} labelFont="mono" color="decode" aria-label="Вероятности кандидатов" />
      </Figure>
      <Demo label="horizontal={false}, отрицательные значения (логиты)">
        <BarChart
          horizontal={false}
          height={160}
          data={[
            { label: 'Париж', value: 9.1 },
            { label: 'Лион', value: 3.4 },
            { label: 'город', value: 1.2 },
            { label: 'это', value: -0.8 },
            { label: 'Конечно', value: -2.1, struck: true },
          ]}
          valueFormat={(v) => fmt.num(v, 1)}
        />
      </Demo>
    </div>
  )
}

function LinesDemo() {
  const h100 = useMemo(() => rooflinePoints(989, 3.35), [])
  const h200 = useMemo(() => rooflinePoints(989, 4.8), [])
  return (
    <div className="grid min-w-0 grid-cols-1 gap-8">
      <Figure title="Roofline: H100 SXM и H200, BF16 dense" caption="Попробуйте: ведите курсор по графику; тултип показывает обе серии. Пики из даташитов NVIDIA.">
        <LineChart
          xLabel="Арифметическая интенсивность, FLOP/байт"
          yLabel="Достижимые TFLOPS"
          xScale="log"
          yScale="log"
          series={[
            { id: 'h100', label: 'H100 SXM', color: 'prefill', points: h100 },
            { id: 'h200', label: 'H200', color: 'decode', points: h200 },
          ]}
          annotations={[
            { kind: 'x', x: 295, label: 'ridge point H100 ≈ 295' },
            { kind: 'point', x: 1, y: 3.35, label: 'decode, batch 1' },
          ]}
          xFormat={(v) => fmt.compact(v)}
          yFormat={(v) => fmt.compact(v)}
        />
      </Figure>
      <Figure title="TTFT p50 под нагрузкой (условные данные)">
        <LineChart
          xLabel="Одновременных запросов"
          yLabel="TTFT p50, мс"
          markers
          series={[
            { id: 'agg', label: 'Агрегированный', color: 'prefill', points: TTFT_SERIES.agg },
            { id: 'disagg', label: 'Дезагрегированный', color: 'decode', points: TTFT_SERIES.disagg, dashed: true },
          ]}
          annotations={[{ kind: 'y', y: 500, label: 'SLO 500 мс' }]}
          yFormat={(v) => fmt.int(v)}
        />
      </Figure>
    </div>
  )
}

function FormulaDemo() {
  return (
    <div className="grid min-w-0 grid-cols-1 gap-6">
      <Formula block tex="\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{\cq{Q}\,\ck{K}^{\top}}{\sqrt{d_k}}+M\right)\cv{V}" label="M ставит −∞ над диагональю (causal mask)" />
      <Formula block tex="\text{KV на токен}=2\cdot L\cdot h_{kv}\cdot d_{head}\cdot b=2\cdot 36\cdot 8\cdot 128\cdot 2=147\,456\ \text{байт}" label="Qwen3-8B, BF16: 144 КиБ на токен" />
      <p className="max-w-[68ch] text-[17px]">
        Формула в строке: время шага decode не меньше <Formula tex="\frac{\text{байты весов}}{\text{полоса HBM}}" />, а ridge point H100 около{' '}
        <Formula tex="989 / 3{,}35 \approx 295" /> FLOP/байт.
      </p>
    </div>
  )
}

function CodeDemo() {
  return (
    <div className="grid min-w-0 grid-cols-1 gap-6">
      <CodeBlock lang="json" code={REQUEST_JSON} title="POST /v1/chat/completions" highlightLines={[8]} />
      <CodeBlock lang="bash" code={VLLM_BASH} highlightLines={[3]} />
      <div className="grid gap-6 lg:grid-cols-2">
        <CodeBlock lang="yaml" code={DAEMONSET_YAML} title="kv-cache-daemonset.yaml" maxHeight={320} />
        <CodeBlock lang="jinja" code={JINJA_SNIPPET} title="chat_template (фрагмент)" />
      </div>
      <CodeBlock lang="python" code={PYTHON_SNIPPET} lineNumbers />
      <CodeBlock lang="text" code={RENDERED_TEMPLATE} title="Промпт после шаблона Qwen3" wrap />
    </div>
  )
}

function NumbersDemo() {
  const rows: Array<[string, string]> = [
    ['fmt.int(151936)', fmt.int(151936)],
    ['fmt.num(3.3456)', fmt.num(3.3456)],
    ['fmt.num(-0.52, 2)', fmt.num(-0.52, 2)],
    ['fmt.compact(622_000_000)', fmt.compact(622_000_000)],
    ["fmt.bytes(147456, { mode: 'binary' })", fmt.bytes(147456, { mode: 'binary' })],
    ['fmt.bytes(1.24e9)', fmt.bytes(1.24e9)],
    ['fmt.ms(4.93)', fmt.ms(4.93)],
    ['fmt.ms(1234)', fmt.ms(1234)],
    ['fmt.pct(0.42)', fmt.pct(0.42)],
    ['fmt.rate(205.3)', fmt.rate(205.3)],
    ['fmt.tokens(151669)', fmt.tokens(151669)],
    ['fmt.bandwidth(3350)', fmt.bandwidth(3350)],
    ['fmt.flops(989e12, { perSecond: true })', fmt.flops(989e12, { perSecond: true })],
    ['fmt.hex([0xd0, 0x9f])', fmt.hex([0xd0, 0x9f])],
  ]
  return (
    <div className="grid min-w-0 grid-cols-1 gap-8">
      <div className="grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-4">
        <Stat label="TTFT" value="182" unit="мс" hint="очередь + prefill" swatch="var(--c-prefill)" />
        <Stat label="ITL" value="11,4" unit="мс" hint="между токенами" swatch="var(--c-decode)" />
        <Stat label="Скорость" value="88" unit="ток/с" />
        <Stat label="Из кэша" value="1 024" unit="токена" hint="prefix cache" />
      </div>
      <Stat label="KV cache на 32 запроса по 8K токенов" value="38,7" unit="ГБ" size="lg" hint="Qwen3-8B, BF16 (условный сценарий)" />
      <div className="overflow-x-auto rounded-panel border border-line">
        <table className="w-full min-w-[460px] text-left text-sm">
          <thead className="bg-surface-2 text-ink-2">
            <tr>
              <th className="px-4 py-2 font-medium">Вызов</th>
              <th className="px-4 py-2 font-medium">Результат</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([code, out]) => (
              <tr key={code} className="border-t border-line">
                <td className="px-4 py-2 font-mono text-[12.5px] text-ink-2">{code}</td>
                <td className="px-4 py-2 tabular-nums text-ink">{out}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const FORMATS = [
  { name: 'BF16', bits: 16, kv: '144 КиБ', fits: 'good' as const, note: 'помещается' },
  { name: 'FP8', bits: 8, kv: '72 КиБ', fits: 'good' as const, note: 'помещается' },
  { name: 'FP4', bits: 4, kv: '36 КиБ', fits: 'warn' as const, note: 'нужна проверка качества' },
]

function BlocksDemo() {
  const [scenario, setScenario] = useState<'short' | 'long'>('short')
  const p = usePlayer(6, { intervalMs: 900 })
  return (
    <div className="grid gap-6">
      <div className="grid gap-4 md:grid-cols-2">
        <Panel>
          <p className="text-sm text-ink-2">Panel tone=«inset»: пояснение к шагу или итог расчёта внутри фигуры.</p>
        </Panel>
        <Panel tone="outline">
          <p className="text-sm text-ink-2">Panel tone=«outline»: пример или справка, рамка без заливки.</p>
        </Panel>
      </div>
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">Формат весов</th>
            <th scope="col" className="num">
              Бит
            </th>
            <th scope="col" className="num">
              KV на токен
            </th>
            <th scope="col">Итог</th>
          </tr>
        </thead>
        <tbody>
          {FORMATS.map((f) => (
            <tr key={f.name}>
              <td className="font-mono">{f.name}</td>
              <td className="num">{f.bits}</td>
              <td className="num">{f.kv}</td>
              <td>
                <span className="inline-flex items-center gap-1.5">
                  <StatusIcon status={f.fits} />
                  {f.note}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <ScrollX label="Широкая таблица">
        <table className="data-table min-w-[880px]">
          <thead>
            <tr>
              {['Стадия', 'Где', 'Что настраивают', 'Метрика', 'Флаг vLLM', 'Флаг SGLang'].map((h) => (
                <th key={h} scope="col">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Prefill</td>
              <td>GPU</td>
              <td>chunked prefill, размер куска</td>
              <td>TTFT</td>
              <td className="font-mono">--max-num-batched-tokens</td>
              <td className="font-mono">--chunked-prefill-size</td>
            </tr>
          </tbody>
        </table>
      </ScrollX>
      <Figure
        title="Слоты фигуры"
        note="условные числа"
        inputs={
          <Segmented
            label="Сценарий"
            value={scenario}
            onChange={setScenario}
            options={[
              { value: 'short', label: 'Короткий промпт' },
              { value: 'long', label: 'Длинный промпт' },
            ]}
          />
        }
        player={<Player {...p.bind} labels={['Очередь', 'Prefill', 'Первый токен', 'Decode', 'Decode', 'Стоп']} />}
        caption="Попробуйте: переключите сценарий сверху и запустите проигрывание снизу."
        source="Модель условная: один GPU, без очереди."
      >
        <p className="text-sm text-ink-2">
          Шаг {p.step + 1}: {scenario === 'short' ? 'короткий' : 'длинный'} промпт.
        </p>
      </Figure>
    </div>
  )
}

function LoadingDemo() {
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <Demo label="TokenStripSkeleton">
        <TokenStripSkeleton count={28} />
      </Demo>
      <Demo label="Skeleton lines={3}">
        <Skeleton lines={3} />
      </Demo>
    </div>
  )
}
