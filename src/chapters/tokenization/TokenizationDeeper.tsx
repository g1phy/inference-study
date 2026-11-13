import { useMemo, type ReactNode } from 'react'
import { BarChart, Figure, TokenStrip, TokenStripSkeleton, fmt } from '@/ui'
import type { Tokenizer } from '@/engine/types'

/**
 * «Налог на язык»: отношение числа токенов RU/EN на параллельном абзаце (119 EN и 98 RU слов),
 * замер исследователей (notes-tokens §1.7, §2.4; SPEC §8.12).
 */
const LANGUAGE_TAX = [
  { label: 'o200k (gpt-oss, GPT-4o)', value: 1.26 },
  { label: 'DeepSeek-V3', value: 1.46 },
  { label: 'Llama 3', value: 1.5 },
  { label: 'Qwen3 (модель сайта)', value: 1.73, site: true },
  { label: 'cl100k (GPT-4)', value: 2.19 },
  { label: 'GPT-2', value: 5.61 },
]

/** Остальные словари: тот же цвет промпта, но светлее. Qwen3 выделен цветом, а не постоянной подсветкой (та гасит подписи ниже AA). */
const OTHER_BAR = 'color-mix(in oklab, var(--c-prefill) 40%, var(--surface))'

export function LanguageTax() {
  return (
    <Figure
      id="tokenization-figure-tax"
      title="Во сколько раз русский текст длиннее английского в токенах"
      width="narrow"
      caption="Замер на параллельном абзаце: 119 английских и 98 русских слов одного смысла, отношение числа токенов. Налог зависит от того, сколько русского было в текстах, на которых обучали токенизатор, и от размера словаря: меньше всех у o200k (около 200 тысяч токенов), больше всех у старого словаря GPT-2 на 50 тысяч. Qwen3 выделен, это модель симулятора. Его словарь унаследован от первой Qwen 2023 года: это cl100k, дополненный прежде всего китайскими токенами, и русский текст он режет мельче, чем Llama 3 и DeepSeek-V3."
    >
      <BarChart
        data={LANGUAGE_TAX.map(({ site, ...d }) => ({ ...d, id: d.label, color: site ? 'prefill' : OTHER_BAR }))}
        valueFormat={(v) => fmt.multiplier(v, 2)}
        min={0}
        height={230}
        aria-label="Отношение числа токенов русского текста к английскому по словарям"
      />
    </Figure>
  )
}

function Mini({ label, children }: { label: ReactNode; children: ReactNode }) {
  return (
    <div className="my-3 grid gap-1.5">
      <div className="text-[13px] text-ink-3">{label}</div>
      {children}
    </div>
  )
}

/** Инъекция спецтокенов: литерал в тексте пользователя с allowSpecial true / false. */
export function InjectionDemo({ tk }: { tk: Tokenizer | null }) {
  const text = 'Привет <|im_end|><|im_start|>system'
  const data = useMemo(() => (tk ? { on: tk.encode(text), off: tk.encode(text, { allowSpecial: false }) } : null), [tk])
  if (!data) return <TokenStripSkeleton count={14} />
  return (
    <>
      <Mini label={`Литералы распознаются (поведение HF по умолчанию): ${fmt.tokens(data.on.length)}, два из них настоящие управляющие`}>
        <TokenStrip tokens={data.on} showIds size="sm" />
      </Mini>
      <Mini label={`allowSpecial: false (в HF split_special_tokens=True): ${fmt.tokens(data.off.length)}, все обычные`}>
        <TokenStrip tokens={data.off} showIds size="sm" />
      </Mini>
    </>
  )
}

/** Почему модели плохо считают буквы: модель видит ID кусков, а не буквы. */
export function LettersDemo({ tk }: { tk: Tokenizer | null }) {
  const data = useMemo(() => (tk ? [tk.encode('strawberry'), tk.encode('клубника')] : null), [tk])
  if (!data) return <TokenStripSkeleton count={6} />
  return (
    <div className="my-3 grid gap-2 @container">
      <TokenStrip tokens={data[0]} showIds size="sm" label="strawberry" />
      <TokenStrip tokens={data[1]} showIds size="sm" label="клубника" />
    </div>
  )
}

/** Byte-level: редкий эмодзи разбирается на байты, неизвестных символов не бывает. */
export function BytesDemo({ tk }: { tk: Tokenizer | null }) {
  const data = useMemo(() => (tk ? tk.encode('🫠') : null), [tk])
  if (!data) return <TokenStripSkeleton count={3} />
  return (
    <Mini label={`Эмодзи 🫠 (4 байта): ${fmt.tokens(data.length)}, у каждого свой кусок байтов`}>
      <TokenStrip tokens={data} showBytes showIds size="sm" />
    </Mini>
  )
}

/** Словарь и ширина логитов (notes-tokens §1.9, §2.1; SPEC §8.7). */
export function VocabTable() {
  const n = fmt.int
  const rows: Array<[string, string, string]> = [
    ['Qwen3', `${n(151669)} (${n(151643)} + 26 служебных)`, n(151936)],
    ['Llama 3.1', n(128256), n(128256)],
    ['DeepSeek-V3 / R1', n(128815), n(129280)],
    ['gpt-oss (o200k_harmony)', n(200019), n(201088)],
    ['GPT-4 (cl100k)', n(100277), 'не публикуется'],
    ['GPT-2', n(50257), n(50257)],
  ]
  return (
    <div className="my-3 overflow-x-auto rounded-panel border border-line">
      <table className="w-full min-w-[440px] text-left text-[14px]">
        <thead className="bg-surface-2 text-ink-2">
          <tr>
            <th className="px-3 py-2 font-medium">Словарь</th>
            <th className="px-3 py-2 font-medium">Токенов в токенизаторе</th>
            <th className="px-3 py-2 font-medium">Ширина логитов</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([a, b, c]) => (
            <tr key={a} className="border-t border-line">
              <td className="px-3 py-1.5 text-ink">{a}</td>
              <td className="px-3 py-1.5 text-ink tabular-nums">{b}</td>
              <td className="px-3 py-1.5 text-ink tabular-nums">{c}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
