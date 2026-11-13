import { useMemo, useState, type ReactNode } from 'react'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import type { Token } from '@/engine/types'
import { CodeBlock, Figure, MEDIA, Segmented, Slider, Stat, Tabs, TokenStripSkeleton, cn, fmt, tint, useElementSize, useMediaQuery } from '@/ui'
import { promptLookupRounds, type LookupRound } from './ngram'
import { Chip, LegendItem } from './TokenBits'

type MethodId = 'draft' | 'eagle' | 'mtp' | 'ngram'

const TABS: Array<{ id: MethodId; label: string; short: string }> = [
  { id: 'draft', label: 'Draft-модель', short: 'Draft' },
  { id: 'eagle', label: 'EAGLE', short: 'EAGLE' },
  { id: 'mtp', label: 'MTP', short: 'MTP' },
  { id: 'ngram', label: 'N-gram', short: 'N-gram' },
]

interface MethodInfo {
  source: ReactNode
  cost: ReactNode
  good: ReactNode
  bad: ReactNode
  flag: string
}

const INFO: Record<Exclude<MethodId, 'ngram'>, MethodInfo> = {
  draft: {
    source: 'Отдельная маленькая модель того же семейства с тем же словарём, например Qwen3-0.6B для Qwen3-32B. Она сама генерирует k токенов обычным decode.',
    cost: 'Черновик должен быть хотя бы в 10 раз меньше target; в исходной статье лучше всего работали модели в ~100 раз меньше, c < 0,05.',
    good: 'Работает без обучения: взяли готовую маленькую модель и включили.',
    bad: 'Свои веса, свой KV cache и свой prefill в той же памяти GPU. Готовые маленькие модели обучены быть самостоятельными, поэтому α у них скромный.',
    flag: `vllm serve Qwen/Qwen3-32B \\
  --speculative-config '{"method": "draft_model", "model": "Qwen/Qwen3-0.6B",
                         "num_speculative_tokens": 4}'`,
  },
  eagle: {
    source: 'Маленькая голова (обычно меньше 1 млрд параметров), которая смотрит на hidden states самой target: в EAGLE-3 на признаки раннего, среднего и позднего слоя. Модель уже «подумала» над контекстом, голова только дорисовывает продолжение.',
    cost: 'Голова живёт в том же forward pass, что и target, без лишних походов на CPU. Предлагает до ~8 токенов с высоким α.',
    good: 'Самый частый выбор в проде, если есть обученная голова под вашу модель. SGLang, Llama-3.1-8B на H100: 158 ток/с без спекуляции, 244 с EAGLE-2, 373 с EAGLE-3.',
    bad: 'Голову нужно обучить под конкретную модель (для популярных моделей есть готовые) и переобучать после fine-tuning.',
    flag: `vllm serve meta-llama/Llama-3.1-8B-Instruct \\
  --speculative-config '{"method": "eagle3", "model": "<путь к EAGLE-3 голове>",
                         "num_speculative_tokens": 3}'`,
  },
  mtp: {
    source: 'Multi-token prediction: модель с самого начала обучена предсказывать не только следующий токен, но и токен после него. Такие головы лежат в чекпоинте (DeepSeek-V3 и R1, MiMo), и движок использует их как готовый черновик.',
    cost: 'Отдельной модели нет, голова маленькая и уже обучена вместе с весами, поэтому α высокий.',
    good: 'Бесплатный черновик для моделей, где MTP есть «из коробки». TensorRT-LLM: DeepSeek-R1 на 8×B200 вырос с 67 до 368 ток/с на пользователя за счёт NVFP4, MTP и слитых ядер.',
    bad: 'Есть только у моделей, обученных с MTP. Обычно предсказывает 1-3 токена вперёд.',
    flag: `vllm serve deepseek-ai/DeepSeek-V3 -tp 8 \\
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`,
  },
}

export function Methods() {
  const [tab, setTab] = useState<MethodId>('eagle')
  // на телефоне короткие подписи и компактные вкладки: все четыре влезают, вкладка N-gram не уходит за край
  const wide = useMediaQuery(MEDIA.tablet)
  const tabs = TABS.map((t) => ({ id: t.id, label: wide ? t.label : t.short }))
  return (
    <Figure
      id="speculative-figure-methods"
      title="Откуда берётся черновик"
      caption="Попробуйте: откройте N-gram и переключите задачу с правки кода на новый текст. На правке вывод копирует промпт, и черновики принимаются длинными сериями; на новом тексте совпадений почти нет. Токены настоящие (словарь Qwen3), ответ модели условный."
    >
      <Tabs tabs={tabs} value={tab} onChange={setTab} size={wide ? 'md' : 'sm'} aria-label="Способ получить черновик">
        {(active) => (active === 'ngram' ? <NgramDemo /> : <MethodCard id={active} />)}
      </Tabs>
    </Figure>
  )
}

function MethodCard({ id }: { id: Exclude<MethodId, 'ngram'> }) {
  const info = INFO[id]
  const [ref, { width }] = useElementSize<HTMLDivElement>()
  return (
    <div className="grid min-w-0 gap-5 pt-4 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div ref={ref} className="min-w-0">
        {width > 0 && width < 400 ? <SchematicNarrow id={id} /> : <Schematic id={id} />}
      </div>
      <dl className="grid min-w-0 content-start gap-3 text-[15px] leading-relaxed">
        <Row term="Откуда черновик">{info.source}</Row>
        <Row term="Цена">{info.cost}</Row>
        <Row term="Плюсы">{info.good}</Row>
        <Row term="Минусы">{info.bad}</Row>
      </dl>
      <div className="min-w-0 @3xl:col-span-2">
        <CodeBlock lang="bash" code={info.flag} title="как включить в vLLM (флаги зависят от версии)" />
      </div>
    </div>
  )
}

function Row({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div className="grid gap-0.5">
      <dt className="text-sm font-medium text-ink-2">{term}</dt>
      <dd className="text-ink">{children}</dd>
    </div>
  )
}

/* ============================================================================
 * Схемы методов (SVG, цвета из токенов)
 * ========================================================================== */

const BOX = { fill: 'var(--surface-2)', stroke: 'var(--line-strong)', strokeWidth: 1 }
const TXT = { fill: 'var(--ink)', fontSize: 13, fontFamily: 'Onest Variable, sans-serif' }
const TXT2 = { fill: 'var(--ink-2)', fontSize: 12, fontFamily: 'Onest Variable, sans-serif' }

function Arrow({ x1, y1, x2, y2, label, dy = -6 }: { x1: number; y1: number; x2: number; y2: number; label?: string; dy?: number }) {
  return (
    <g>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--ink-3)" strokeWidth={1.5} markerEnd="url(#spec-arrow)" />
      {label && (
        <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 + dy} textAnchor="middle" {...TXT2}>
          {label}
        </text>
      )}
    </g>
  )
}

function DraftTokens({ x, y, n }: { x: number; y: number; n: number }) {
  return (
    <g>
      {Array.from({ length: n }, (_, i) => (
        <rect key={i} x={x + i * 22} y={y} width={18} height={14} rx={4} fill="var(--surface)" stroke="var(--ink-3)" strokeWidth={1.2} strokeDasharray="3 2" />
      ))}
    </g>
  )
}

function Schematic({ id }: { id: Exclude<MethodId, 'ngram'> }) {
  const label =
    id === 'draft'
      ? 'Схема: маленькая draft-модель генерирует k токенов, target проверяет их за один проход'
      : id === 'eagle'
        ? 'Схема: EAGLE-голова читает hidden states раннего, среднего и позднего слоя target и предлагает черновик'
        : 'Схема: встроенная MTP-голова модели предсказывает токены после следующего'
  return (
    <svg viewBox="0 0 460 210" className="h-auto w-full" role="img" aria-label={label}>
      <defs>
        <marker id="spec-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
      </defs>
      {id === 'draft' && (
        <g>
          <rect x={10} y={40} width={130} height={70} rx={10} {...BOX} />
          <text x={75} y={68} textAnchor="middle" {...TXT}>
            Draft-модель
          </text>
          <text x={75} y={86} textAnchor="middle" {...TXT2}>
            0,6 млрд параметров
          </text>
          <rect x={30} y={122} width={90} height={26} rx={6} fill="none" stroke="var(--line-strong)" strokeDasharray="4 3" />
          <text x={75} y={139} textAnchor="middle" {...TXT2}>
            свой KV cache
          </text>
          <DraftTokens x={162} y={56} n={4} />
          <Arrow x1={140} y1={75} x2={290} y2={75} label="k черновых токенов" dy={-24} />
          <rect x={292} y={30} width={138} height={95} rx={10} {...BOX} />
          <text x={361} y={64} textAnchor="middle" {...TXT}>
            Target
          </text>
          <text x={361} y={82} textAnchor="middle" {...TXT2}>
            32 млрд параметров
          </text>
          <text x={361} y={100} textAnchor="middle" {...TXT2}>
            проверка за 1 проход
          </text>
          <path d="M 361 125 C 361 185, 75 185, 75 150" fill="none" stroke="var(--ink-3)" strokeWidth={1.5} markerEnd="url(#spec-arrow)" />
          <text x={220} y={196} textAnchor="middle" {...TXT2}>
            принятые токены + 1 от target, черновик продолжает с них
          </text>
        </g>
      )}
      {id === 'eagle' && (
        <g>
          {Array.from({ length: 9 }, (_, i) => (
            <rect key={i} x={20} y={18 + i * 19} width={150} height={14} rx={4} fill={i === 1 || i === 4 || i === 7 ? 'var(--surface-3)' : 'var(--surface-2)'} stroke={i === 1 || i === 4 || i === 7 ? 'var(--ink-3)' : 'var(--line-strong)'} />
          ))}
          <text x={95} y={14} textAnchor="middle" {...TXT2}>
            слои target
          </text>
          <text x={180} y={47} {...TXT2}>
            ранний
          </text>
          <text x={180} y={104} {...TXT2}>
            средний
          </text>
          <text x={180} y={161} {...TXT2}>
            поздний
          </text>
          <Arrow x1={232} y1={43} x2={290} y2={88} />
          <Arrow x1={240} y1={100} x2={290} y2={100} />
          <Arrow x1={235} y1={157} x2={290} y2={112} />
          <rect x={292} y={70} width={138} height={60} rx={10} {...BOX} />
          <text x={361} y={96} textAnchor="middle" {...TXT}>
            EAGLE-голова
          </text>
          <text x={361} y={114} textAnchor="middle" {...TXT2}>
            меньше 1 млрд
          </text>
          <DraftTokens x={316} y={150} n={4} />
          <Arrow x1={361} y1={130} x2={361} y2={148} />
          <text x={361} y={184} textAnchor="middle" {...TXT2}>
            до ~8 черновых токенов,
          </text>
          <text x={361} y={200} textAnchor="middle" {...TXT2}>
            их проверит target
          </text>
        </g>
      )}
      {id === 'mtp' && (
        <g>
          <rect x={20} y={40} width={200} height={110} rx={10} {...BOX} />
          <text x={120} y={70} textAnchor="middle" {...TXT}>
            Модель (DeepSeek-V3)
          </text>
          <text x={120} y={90} textAnchor="middle" {...TXT2}>
            основная голова: токен t + 1
          </text>
          <rect x={40} y={104} width={160} height={30} rx={8} fill="var(--surface-3)" stroke="var(--ink-3)" />
          <text x={120} y={124} textAnchor="middle" {...TXT}>
            MTP-голова: t + 2
          </text>
          <Arrow x1={220} y1={91} x2={296} y2={91} label="токен t + 1" dy={-10} />
          <Arrow x1={200} y1={120} x2={296} y2={141} label="черновик" dy={20} />
          <rect x={300} y={78} width={40} height={26} rx={6} fill={tint('decode', 20)} stroke="var(--line-strong)" />
          <text x={320} y={95} textAnchor="middle" {...TXT2} fill="var(--ink)">
            t+1
          </text>
          <rect x={300} y={128} width={40} height={26} rx={6} fill="var(--surface)" stroke="var(--ink-3)" strokeDasharray="3 2" />
          <text x={320} y={145} textAnchor="middle" {...TXT2} fill="var(--ink)">
            t+2
          </text>
          <text x={348} y={95} {...TXT2}>
            готов
          </text>
          <text x={348} y={139} {...TXT2}>
            проверка на
          </text>
          <text x={348} y={155} {...TXT2}>
            следующем шаге
          </text>
          <text x={220} y={186} textAnchor="middle" {...TXT2}>
            голова обучена вместе с моделью и лежит в чекпоинте
          </text>
        </g>
      )}
    </svg>
  )
}

/** Те же схемы вертикально для узкой фигуры: viewBox 320 в ширину, подписи около 13-14 px на телефоне. */
const NTXT = { fill: 'var(--ink)', fontSize: 15, fontFamily: 'Onest Variable, sans-serif' }
const NTXT2 = { fill: 'var(--ink-2)', fontSize: 14, fontFamily: 'Onest Variable, sans-serif' }

function SchematicNarrow({ id }: { id: Exclude<MethodId, 'ngram'> }) {
  const label =
    id === 'draft'
      ? 'Схема: маленькая draft-модель генерирует k токенов, target проверяет их за один проход'
      : id === 'eagle'
        ? 'Схема: EAGLE-голова читает hidden states раннего, среднего и позднего слоя target и предлагает черновик'
        : 'Схема: встроенная MTP-голова модели предсказывает токены после следующего'
  const h = id === 'draft' ? 300 : id === 'eagle' ? 330 : 310
  return (
    <svg viewBox={`0 0 320 ${h}`} className="h-auto w-full" role="img" aria-label={label}>
      <defs>
        <marker id="spec-arrow-n" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
      </defs>
      {id === 'draft' && (
        <g>
          <rect x={40} y={8} width={170} height={66} rx={10} {...BOX} />
          <text x={125} y={36} textAnchor="middle" {...NTXT}>
            Draft-модель
          </text>
          <text x={125} y={58} textAnchor="middle" {...NTXT2}>
            0,6 млрд параметров
          </text>
          <rect x={218} y={22} width={96} height={38} rx={6} fill="none" stroke="var(--line-strong)" strokeDasharray="4 3" />
          <text x={266} y={46} textAnchor="middle" {...NTXT2} fontSize={13}>
            свой KV cache
          </text>
          <line x1={125} y1={74} x2={125} y2={134} stroke="var(--ink-3)" strokeWidth={1.5} markerEnd="url(#spec-arrow-n)" />
          <text x={137} y={98} {...NTXT2}>
            k черновых токенов
          </text>
          <DraftTokens x={137} y={108} n={4} />
          <rect x={40} y={138} width={240} height={70} rx={10} {...BOX} />
          <text x={160} y={166} textAnchor="middle" {...NTXT}>
            Target, 32 млрд параметров
          </text>
          <text x={160} y={190} textAnchor="middle" {...NTXT2}>
            проверка за 1 проход
          </text>
          <path d="M 40 173 C 14 173, 14 41, 38 41" fill="none" stroke="var(--ink-3)" strokeWidth={1.5} markerEnd="url(#spec-arrow-n)" />
          <text x={160} y={240} textAnchor="middle" {...NTXT2}>
            принятые токены + 1 от target,
          </text>
          <text x={160} y={262} textAnchor="middle" {...NTXT2}>
            черновик продолжает с них
          </text>
        </g>
      )}
      {id === 'eagle' && (
        <g>
          <text x={20} y={16} {...NTXT2}>
            слои target
          </text>
          {Array.from({ length: 9 }, (_, i) => (
            <rect key={i} x={20} y={26 + i * 16} width={150} height={11} rx={3} fill={i === 1 || i === 4 || i === 7 ? 'var(--surface-3)' : 'var(--surface-2)'} stroke={i === 1 || i === 4 || i === 7 ? 'var(--ink-3)' : 'var(--line-strong)'} />
          ))}
          <text x={180} y={47} {...NTXT2}>
            ранний
          </text>
          <text x={180} y={95} {...NTXT2}>
            средний
          </text>
          <text x={180} y={143} {...NTXT2}>
            поздний
          </text>
          {[42, 90, 138].map((y) => (
            <line key={y} x1={248} y1={y} x2={290} y2={y} stroke="var(--ink-3)" strokeWidth={1.5} />
          ))}
          <line x1={290} y1={42} x2={290} y2={186} stroke="var(--ink-3)" strokeWidth={1.5} markerEnd="url(#spec-arrow-n)" />
          <rect x={20} y={190} width={280} height={58} rx={10} {...BOX} />
          <text x={160} y={215} textAnchor="middle" {...NTXT}>
            EAGLE-голова
          </text>
          <text x={160} y={237} textAnchor="middle" {...NTXT2}>
            меньше 1 млрд параметров
          </text>
          <line x1={160} y1={248} x2={160} y2={262} stroke="var(--ink-3)" strokeWidth={1.5} markerEnd="url(#spec-arrow-n)" />
          <DraftTokens x={116} y={266} n={4} />
          <text x={160} y={304} textAnchor="middle" {...NTXT2}>
            до ~8 черновых токенов,
          </text>
          <text x={160} y={324} textAnchor="middle" {...NTXT2}>
            их проверит target
          </text>
        </g>
      )}
      {id === 'mtp' && (
        <g>
          <rect x={20} y={8} width={280} height={124} rx={10} {...BOX} />
          <text x={160} y={36} textAnchor="middle" {...NTXT}>
            Модель (DeepSeek-V3)
          </text>
          <text x={160} y={60} textAnchor="middle" {...NTXT2}>
            основная голова: токен t + 1
          </text>
          <rect x={40} y={76} width={240} height={38} rx={8} fill="var(--surface-3)" stroke="var(--ink-3)" />
          <text x={160} y={101} textAnchor="middle" {...NTXT}>
            MTP-голова: t + 2
          </text>
          <line x1={90} y1={132} x2={90} y2={166} stroke="var(--ink-3)" strokeWidth={1.5} markerEnd="url(#spec-arrow-n)" />
          <line x1={230} y1={114} x2={230} y2={166} stroke="var(--ink-3)" strokeWidth={1.5} markerEnd="url(#spec-arrow-n)" />
          <rect x={62} y={170} width={56} height={30} rx={6} fill={tint('decode', 20)} stroke="var(--line-strong)" />
          <text x={90} y={190} textAnchor="middle" {...NTXT2} fill="var(--ink)">
            t+1
          </text>
          <rect x={202} y={170} width={56} height={30} rx={6} fill="var(--surface)" stroke="var(--ink-3)" strokeDasharray="3 2" />
          <text x={230} y={190} textAnchor="middle" {...NTXT2} fill="var(--ink)">
            t+2
          </text>
          <text x={90} y={222} textAnchor="middle" {...NTXT2}>
            готов
          </text>
          <text x={230} y={222} textAnchor="middle" {...NTXT2}>
            черновик: проверка
          </text>
          <text x={230} y={242} textAnchor="middle" {...NTXT2}>
            на следующем шаге
          </text>
          <text x={160} y={278} textAnchor="middle" {...NTXT2}>
            голова обучена вместе с моделью
          </text>
          <text x={160} y={298} textAnchor="middle" {...NTXT2}>
            и лежит в чекпоинте
          </text>
        </g>
      )}
    </svg>
  )
}

/* ============================================================================
 * N-gram на настоящих токенах
 * ========================================================================== */

const CODE = `tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto")
print(name, model.config.num_hidden_layers)`

const TASKS = {
  edit: {
    label: 'Правка кода',
    prompt: `Замени name на model_id:\n${CODE}\n`,
    output: CODE.replace(/name/g, 'model_id'),
  },
  fresh: {
    label: 'Новый текст',
    prompt: `Объясни, что делает этот код:\n${CODE}\n`,
    output: 'Код загружает токенизатор и саму модель по имени, а затем печатает имя и число слоёв трансформера.',
  },
} as const

type TaskId = keyof typeof TASKS

function NgramDemo() {
  const { tokenizer } = useTokenizer()
  const [task, setTask] = useState<TaskId>('edit')
  const [k, setK] = useState(4)
  const [hover, setHover] = useState<number | null>(null)
  const data = useMemo(() => {
    if (!tokenizer) return null
    const t = TASKS[task]
    const prompt = tokenizer.encode(t.prompt, { allowSpecial: false })
    const output = tokenizer.encode(t.output, { allowSpecial: false })
    const rounds = promptLookupRounds(
      prompt.map((x) => x.id),
      output.map((x) => x.id),
      { k, nMin: 2, nMax: 4 },
    )
    return { prompt, output, rounds }
  }, [tokenizer, task, k])

  if (!data) {
    return (
      <div className="grid gap-3 pt-4" aria-busy="true">
        <TokenStripSkeleton count={30} />
        <TokenStripSkeleton count={24} />
      </div>
    )
  }
  const { prompt, output, rounds } = data
  const active = hover !== null ? rounds[hover] : null
  const matchFrom = active && active.matchAt >= 0 ? active.matchAt : -1
  const matchTo = active && active.matchAt >= 0 ? active.matchAt + active.matchLen : -1
  const srcFrom = matchTo
  const srcTo = active ? matchTo + active.drafts.length : -1
  const lookAt = (i: number): 'prompt' | 'match' | 'source' => (i >= matchFrom && i < matchTo ? 'match' : i >= srcFrom && i < srcTo ? 'source' : 'prompt')
  const accepted = rounds.reduce((s, r) => s + r.accepted, 0)

  return (
    <div className="grid min-w-0 gap-5 pt-4">
      <div className="grid gap-4 @2xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Segmented label="Задача" value={task} onChange={(v) => { setTask(v); setHover(null) }} options={(Object.keys(TASKS) as TaskId[]).map((id) => ({ value: id, label: TASKS[id].label }))} />
        <Slider label="Черновых токенов за раз" value={k} min={1} max={8} onChange={setK} format={(x) => fmt.int(x)} />
      </div>
      <div className="grid grid-cols-2 gap-x-6 gap-y-4 @lg:grid-cols-4">
        <Stat label="Токенов ответа" value={fmt.int(output.length)} />
        <Stat label="Проходов target" value={fmt.int(rounds.length)} swatch="var(--c-decode)" />
        <Stat label="Токенов за проход" value={fmt.num(output.length / rounds.length, 2)} />
        <Stat label="Принято черновиков" value={fmt.int(accepted)} hint="цена черновика почти ноль: это поиск" />
      </div>
      <div className="min-w-0">
        <div className="mb-1.5 text-sm font-medium text-ink-2">Промпт (здесь ищем совпадения)</div>
        <div className="flex flex-wrap gap-x-[3px] gap-y-1.5 rounded-control bg-surface-2/60 p-2.5">
          {prompt.map((t, i) => (
            <Chip key={i} token={t} look={lookAt(i)} />
          ))}
        </div>
      </div>
      <div className="min-w-0">
        <div className="mb-1.5 text-sm font-medium text-ink-2">Ответ по проходам target (наведите или нажмите на группу)</div>
        <div className="flex flex-wrap gap-x-1.5 gap-y-2" onPointerLeave={(e) => e.pointerType === 'mouse' && setHover(null)}>
          {rounds.map((r, ri) => (
            <RoundGroup key={ri} r={r} tokens={output} active={hover === ri} onActivate={() => setHover(ri)} index={ri} />
          ))}
        </div>
      </div>
      <p className="min-h-[2.5rem] text-[14px] leading-snug text-ink-2" aria-live="polite">
        {active
          ? active.drafts.length === 0
            ? `Проход ${hover! + 1}: последние токены нигде раньше не встречались, черновика нет, обычный шаг decode.`
            : `Проход ${hover! + 1}: последние ${fmt.count(active.matchLen, ['токен', 'токена', 'токенов'])} уже встречались ${active.matchAt >= prompt.length ? 'в написанной части ответа' : 'в промпте'}; после них шли ${fmt.count(active.drafts.length, ['токен', 'токена', 'токенов'])}, они стали черновиком. Совпало ${fmt.int(active.accepted)}, плюс 1 токен от target.`
          : 'Наведите или нажмите на группу в ответе: в промпте подсветится найденное совпадение (с рамкой) и подчеркнутся токены, взятые в черновик.'}
      </p>
      <div className="flex flex-wrap gap-x-4 gap-y-1.5">
        <LegendItem swatch={<span className="h-3 w-4 rounded-[3px] shadow-[inset_0_0_0_1.5px_var(--c-decode)]" style={{ background: tint('decode', 13) }} />}>
          в рамке в ответе: токен, который target посчитал сам
        </LegendItem>
        <LegendItem swatch={<span className="h-3 w-4 rounded-[3px] shadow-[inset_0_0_0_1.5px_var(--c-prefill)]" style={{ background: tint('prefill', 34) }} />}>
          в рамке в промпте: найденное совпадение
        </LegendItem>
        <LegendItem swatch={<span className="w-4 border-b-[1.5px] border-dashed border-ink-3 pb-1" />}>подчёркнуто: взято в черновик</LegendItem>
      </div>
    </div>
  )
}

function RoundGroup({ r, tokens, active, onActivate, index }: { r: LookupRound; tokens: Token[]; active: boolean; onActivate: () => void; index: number }) {
  const toks = tokens.slice(r.start, r.start + r.produced)
  return (
    <span
      tabIndex={0}
      role="button"
      aria-label={`Проход ${index + 1}: ${r.produced} ток., принято черновиков ${r.accepted}`}
      onPointerEnter={onActivate}
      onPointerDown={onActivate}
      onFocus={onActivate}
      className={cn(
        'inline-flex flex-wrap items-start gap-[3px] rounded-control p-1 transition-shadow focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent',
        active ? 'shadow-[inset_0_0_0_1.5px_var(--accent)]' : 'shadow-[inset_0_0_0_1px_var(--line)]',
      )}
    >
      {toks.map((t, i) => (
        <Chip key={i} token={t} look="committed" className={i === toks.length - 1 ? 'shadow-[inset_0_0_0_1.5px_var(--c-decode)]' : undefined} />
      ))}
      {r.produced > 1 && <span className="self-center px-0.5 font-mono text-[11px] text-ink-3 tabular-nums">{r.produced}</span>}
    </span>
  )
}
