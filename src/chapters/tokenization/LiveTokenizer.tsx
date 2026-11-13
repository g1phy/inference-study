import { useMemo, useState } from 'react'
import { ArrowDownIcon } from '@phosphor-icons/react'
import { BarChart, Button, Figure, Segmented, Stat, TextArea, TokenStrip, TokenStripSkeleton, cn, fmt, useReducedMotion } from '@/ui'
import type { Tokenizer } from '@/engine/types'
import { charCount, utf8Length, wordOfToken } from './textStats'

export type PresetId = 'mine' | 'ru' | 'en' | 'code' | 'num' | 'emoji'

/** Одна мысль на двух языках: для живого сравнения RU и EN. */
export const RU_PHRASE = 'Модель читает текст не по буквам, а токенами: частые слова целиком, редкие по частям.'
export const EN_PHRASE = 'The model reads text not letter by letter but in tokens: common words whole, rare ones in pieces.'

export const PRESETS: Record<Exclude<PresetId, 'mine'>, string> = {
  ru: RU_PHRASE,
  en: EN_PHRASE,
  code: 'def add(a, b):\n    return a + b  # сумма двух чисел',
  num: 'В 2026 году в словаре Qwen3 151669 токенов, а π ≈ 3.14159.',
  emoji: 'Привет 👋🙂 🫠 🧑‍💻 👍🏽',
}

const PRESET_LABEL: Record<PresetId, string> = {
  mine: 'Ваш запрос',
  ru: 'Русский',
  en: 'English',
  code: 'Код',
  num: 'Числа',
  emoji: 'Эмодзи',
}

type View = 'text' | 'ids' | 'bytes'

export interface LiveTokenizerProps {
  tokenizer: Tokenizer | null
  text: string
  userText: string
  onText: (t: string) => void
  /** Клик по токену: слово (пре-токен) уходит в разбор BPE. */
  onPickWord: (word: string) => void
}

export function LiveTokenizer({ tokenizer, text, userText, onText, onPickWord }: LiveTokenizerProps) {
  const [view, setView] = useState<View>('text')
  const [picked, setPicked] = useState<{ word: string; indices: number[] } | null>(null)
  const reduce = useReducedMotion()

  const tokens = useMemo(() => (tokenizer ? tokenizer.encode(text) : null), [tokenizer, text])
  const pieces = useMemo(() => (tokenizer ? tokenizer.preTokenize(text) : []), [tokenizer, text])
  const ruEn = useMemo(() => (tokenizer ? { ru: tokenizer.encode(RU_PHRASE).length, en: tokenizer.encode(EN_PHRASE).length } : null), [tokenizer])

  const chars = charCount(text)
  const bytes = utf8Length(text)
  const n = tokens?.length ?? 0
  const preset = (Object.keys(PRESET_LABEL) as PresetId[]).find((p) => (p === 'mine' ? text === userText : PRESETS[p] === text))

  const setPreset = (p: PresetId) => {
    setPicked(null)
    onText(p === 'mine' ? userText : PRESETS[p])
  }

  const pick = (i: number) => {
    if (!tokens) return
    const w = wordOfToken(text, pieces, tokens, i)
    if (!w) return
    setPicked(w)
    onPickWord(w.word)
  }

  return (
    <Figure
      id="tokenization-figure-live"
      title="Живой токенизатор Qwen3"
      actions={
        <Segmented<View>
          size="sm"
          aria-label="Что показать под токенами"
          value={view}
          onChange={setView}
          options={[
            { value: 'text', label: 'Текст' },
            { value: 'ids', label: 'ID' },
            { value: 'bytes', label: 'Байты' },
          ]}
        />
      }
      caption="Попробуйте: выберите «Числа» и «Эмодзи», включите байты. Потом нажмите на любой токен: ниже его слово разберётся по шагам BPE."
    >
      <div className="grid gap-3">
        <TextArea
          label="Текст"
          value={text}
          onChange={(v) => {
            setPicked(null)
            onText(v)
          }}
          rows={2}
          autoGrow
          maxLength={2000}
        />
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Примеры текста">
          {(Object.keys(PRESET_LABEL) as PresetId[]).map((p) => (
            <button
              key={p}
              type="button"
              aria-pressed={preset === p}
              onClick={() => setPreset(p)}
              className={cn(
                'h-8 rounded-control px-3 text-[13.5px] font-medium transition-colors',
                preset === p ? 'bg-accent-soft text-accent-hover shadow-[inset_0_0_0_1px_var(--accent)]' : 'bg-surface text-ink-2 shadow-[inset_0_0_0_1px_var(--line-strong)] hover:bg-surface-3 hover:text-ink',
              )}
            >
              {PRESET_LABEL[p]}
            </button>
          ))}
        </div>

        <div className="min-h-[60px] rounded-panel bg-surface-2/60 px-3 py-3 shadow-[inset_0_0_0_1px_var(--line)]">
          {!tokens ? (
            <TokenStripSkeleton count={22} />
          ) : (
            <TokenStrip
              tokens={tokens}
              showIds={view === 'ids'}
              showBytes={view === 'bytes'}
              highlightIndex={picked?.indices ?? null}
              onClickIndex={pick}
              empty="Введите текст"
              label={`Токены: ${fmt.tokens(n)}`}
            />
          )}
        </div>

        {picked && (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[14px] text-ink-2" aria-live="polite">
            <span>
              Слово <code className="rounded-[4px] bg-surface-2 px-1 font-mono whitespace-pre text-ink">«{picked.word.replace(/ /g, '·')}»</code>{' '}
              загружено в разбор BPE.
            </span>
            <Button
              variant="ghost"
              size="sm"
              icon={ArrowDownIcon}
              onClick={() =>
                document.getElementById('tokenization-figure-bpe')?.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' })
              }
            >
              К разбору
            </Button>
          </div>
        )}

        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @min-[640px]:grid-cols-4">
          <Stat label="Символов" value={fmt.int(chars)} />
          <Stat label="Токенов" value={tokens ? fmt.int(n) : '…'} swatch="var(--c-prefill)" />
          <Stat
            label="Символов на токен"
            value={tokens && n ? fmt.num(chars / n, 1) : '…'}
            hint={ruEn ? `у английского ≈ ${fmt.num(charCount(EN_PHRASE) / ruEn.en, 1)}` : undefined}
          />
          <Stat label="Байт UTF-8" value={fmt.int(bytes)} hint="кириллица по 2, эмодзи по 4" />
        </div>

        <div className="grid gap-2 rounded-panel border border-line px-4 py-3">
          <div className="text-[14px] font-medium text-ink">Одна мысль на двух языках</div>
          {ruEn ? (
            <>
              <BarChart
                data={[
                  { label: 'Русский', value: ruEn.ru, id: 'ru' },
                  { label: 'English', value: ruEn.en, id: 'en' },
                ]}
                valueFormat={(v) => fmt.tokens(v)}
                height={84}
                aria-label="Число токенов одной фразы по-русски и по-английски"
              />
              <p className="text-[13.5px] leading-snug text-ink-2">
                Пресеты «Русский» и «English» говорят одно и то же. По-русски в словаре Qwen3 выходит в{' '}
                <b className="text-ink">{fmt.num(ruEn.ru / ruEn.en, 2)}</b> раза больше токенов, чем по-английски. На короткой фразе
                отношение скачет, на длинном абзаце (график ниже) у Qwen3 выходит ×1,73.
              </p>
            </>
          ) : (
            <TokenStripSkeleton count={8} />
          )}
        </div>
      </div>
    </Figure>
  )
}
