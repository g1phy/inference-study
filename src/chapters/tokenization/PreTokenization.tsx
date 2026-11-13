import { useMemo, useState } from 'react'
import { Figure, Segmented, TextArea, TokenStrip, TokenStripSkeleton, cn, fmt } from '@/ui'
import { bytesToRaw, utf8Bytes } from '@/engine/tokenizer'
import type { Token, Tokenizer } from '@/engine/types'
import { useO200k } from '../template/useO200k'
import { classifyPiece, type PieceClass } from './textStats'

type Dict = 'qwen3' | 'o200k'
type Preset = 'mix' | 'camel' | 'json' | 'mine'

const PRESETS: Record<Exclude<Preset, 'mine'>, string> = {
  mix: "I'm  learning LLMs in 2025!!\n\nПривет, мир 12345",
  camel: 'HelloWorld camelCase JSONParser',
  json: '{"name": "Иван", "age": 42}',
}
const PRESET_LABEL: Record<Preset, string> = { mix: 'Смесь', camel: 'camelCase', json: 'JSON', mine: 'Ваш запрос' }

const CLASS_LABEL: Record<PieceClass, string> = {
  word: 'слова (с ведущим пробелом)',
  digit: 'цифры',
  punct: 'пунктуация и символы',
  space: 'пробелы и переводы строк',
  contraction: "окончания 's, 'm",
  special: 'спецтокены',
}

interface Alt {
  re: string
  text: string
  cls: PieceClass[]
}

/** Регулярки пре-токенизации по частям (notes-tokens §2.2), с объяснением каждой альтернативы. */
const ALTS: Record<Dict, Alt[]> = {
  qwen3: [
    { re: "(?i:'s|'t|'re|'ve|'m|'ll|'d)", text: 'английские окончания вроде \'s и \'m отдельно', cls: ['contraction'] },
    { re: '[^\\r\\n\\p{L}\\p{N}]?\\p{L}+', text: 'буквы подряд, впереди можно один небуквенный символ, обычно пробел', cls: ['word'] },
    { re: '\\p{N}', text: 'ровно одна цифра', cls: ['digit'] },
    { re: ' ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*', text: 'необязательный пробел, пунктуация и символы, переводы строк после них', cls: ['punct'] },
    { re: '\\s*[\\r\\n]+', text: 'переводы строк', cls: ['space'] },
    { re: '\\s+(?!\\S)|\\s+', text: 'пробелы: все, кроме последнего перед словом', cls: ['space'] },
  ],
  o200k: [
    {
      re: "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|…)?",
      text: 'слово с учётом регистра: заглавные, потом строчные; окончание \'s прилипает',
      cls: ['word', 'contraction'],
    },
    { re: "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|…)?", text: 'слово из заглавных, например аббревиатура', cls: ['word'] },
    { re: '\\p{N}{1,3}', text: 'от одной до трёх цифр', cls: ['digit'] },
    { re: ' ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*', text: 'необязательный пробел, пунктуация и символы', cls: ['punct'] },
    { re: '\\s*[\\r\\n]+', text: 'переводы строк', cls: ['space'] },
    { re: '\\s+(?!\\S)|\\s+', text: 'пробелы: все, кроме последнего перед словом', cls: ['space'] },
  ],
}

function toPseudoTokens(pieces: string[], specials: Set<string>): Token[] {
  return pieces.map((p, i) => {
    const bytes = Array.from(utf8Bytes(p))
    return { id: i, raw: bytesToRaw(bytes), bytes, text: p, special: specials.has(p) }
  })
}

export function PreTokenization({ tokenizer, userText }: { tokenizer: Tokenizer | null; userText: string }) {
  const [text, setText] = useState(PRESETS.mix)
  const [dict, setDict] = useState<Dict>('qwen3')
  const [hl, setHl] = useState<PieceClass | null>(null)
  const o200k = useO200k(dict === 'o200k')
  const tk = dict === 'qwen3' ? tokenizer : o200k.tokenizer

  const data = useMemo(() => {
    if (!tk) return null
    const pieces = tk.preTokenize(text)
    const specials = new Set(tk.specialTokens.map((s) => s.text))
    const toks = toPseudoTokens(pieces, specials)
    const classes = pieces.map((p) => classifyPiece(p, specials.has(p)))
    return { toks, classes, bpe: tk.encode(text).length }
  }, [tk, text])

  const counts = useMemo(() => {
    const m = new Map<PieceClass, number>()
    data?.classes.forEach((c) => m.set(c, (m.get(c) ?? 0) + 1))
    return m
  }, [data])

  const preset = (Object.keys(PRESET_LABEL) as Preset[]).find((p) => (p === 'mine' ? text === userText : PRESETS[p] === text))
  const highlighted = hl && data ? data.classes.flatMap((c, i) => (c === hl ? [i] : [])) : null

  return (
    <Figure
      id="tokenization-figure-pretokenize"
      title="Пре-токенизация: как регулярка режет текст до BPE"
      actions={
        <Segmented<Dict>
          size="sm"
          aria-label="Чья регулярка"
          value={dict}
          onChange={setDict}
          options={[
            { value: 'qwen3', label: 'Qwen3' },
            { value: 'o200k', label: 'o200k (gpt-oss)' },
          ]}
        />
      }
      caption="Попробуйте: на «Смеси» переключите регулярку на o200k и посмотрите на число 2025 и на I'm. Наведите на строку легенды или регулярки (или нажмите на неё), чтобы подсветить её куски."
    >
      <div className="grid gap-3">
        <TextArea label="Текст" value={text} onChange={setText} rows={2} autoGrow maxLength={600} mono />
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Примеры текста">
          {(Object.keys(PRESET_LABEL) as Preset[]).map((p) => (
            <button
              key={p}
              type="button"
              aria-pressed={preset === p}
              onClick={() => setText(p === 'mine' ? userText : PRESETS[p])}
              className={cn(
                'h-8 rounded-control px-3 text-[13.5px] font-medium transition-colors',
                preset === p ? 'bg-accent-soft text-accent-hover shadow-[inset_0_0_0_1px_var(--accent)]' : 'bg-surface text-ink-2 shadow-[inset_0_0_0_1px_var(--line-strong)] hover:bg-surface-3 hover:text-ink',
              )}
            >
              {PRESET_LABEL[p]}
            </button>
          ))}
        </div>

        <div className="min-h-[56px] rounded-panel bg-surface-2/60 px-3 py-3 shadow-[inset_0_0_0_1px_var(--line)]">
          {!data ? (
            <TokenStripSkeleton count={18} label={dict === 'o200k' ? 'Загружаем словарь gpt-oss' : undefined} />
          ) : (
            <TokenStrip tokens={data.toks} tooltip={false} highlightIndex={highlighted} label={`Пре-токены: ${data.toks.length}`} empty="Введите текст" />
          )}
        </div>
        {data && (
          <p className="text-[14px] text-ink-2">
            Пре-токенов: <b className="text-ink tabular-nums">{fmt.int(data.toks.length)}</b>. После BPE внутри каждого из них:{' '}
            <b className="text-ink tabular-nums">{fmt.tokens(data.bpe)}</b> ({dict === 'qwen3' ? 'словарь Qwen3' : 'словарь o200k_harmony'}).
          </p>
        )}

        <div className="grid gap-4 @min-[760px]:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
          <div className="min-w-0">
            <div className="mb-1.5 text-[12.5px] text-ink-3">Что получилось</div>
            <ul className="grid gap-1">
              {(Object.keys(CLASS_LABEL) as PieceClass[])
                .filter((c) => counts.get(c))
                .map((c) => (
                  <li key={c}>
                    <button
                      type="button"
                      onPointerEnter={(e) => e.pointerType === 'mouse' && setHl(c)}
                      onPointerLeave={(e) => e.pointerType === 'mouse' && setHl(null)}
                      onFocus={() => setHl(c)}
                      onBlur={() => setHl(null)}
                      onClick={() => setHl((h) => (h === c ? null : c))}
                      aria-pressed={hl === c}
                      className={cn(
                        'flex w-full items-baseline justify-between gap-3 rounded-control px-2 py-1 text-left text-[14px] transition-colors',
                        hl === c ? 'bg-accent-soft text-ink' : 'text-ink-2 hover:bg-surface-3/60 hover:text-ink',
                      )}
                    >
                      <span>{CLASS_LABEL[c]}</span>
                      <span className="font-mono text-[13px] text-ink tabular-nums">{fmt.int(counts.get(c) ?? 0)}</span>
                    </button>
                  </li>
                ))}
            </ul>
          </div>
          <div className="min-w-0">
            <div className="mb-1.5 text-[12.5px] text-ink-3">Регулярка {dict === 'qwen3' ? 'Qwen3' : 'o200k'} по частям (␣ это пробел)</div>
            <ol className="grid gap-1">
              {ALTS[dict].map((a, i) => {
                const on = hl !== null && a.cls.includes(hl)
                // строка регулярки работает как кнопка: наведение, фокус с клавиатуры и тап подсвечивают её куски
                return (
                  <li key={i}>
                    <button
                      type="button"
                      aria-pressed={on}
                      onPointerEnter={(e) => e.pointerType === 'mouse' && setHl(a.cls[0])}
                      onPointerLeave={(e) => e.pointerType === 'mouse' && setHl(null)}
                      onFocus={() => setHl(a.cls[0])}
                      onBlur={() => setHl(null)}
                      onClick={() => setHl((h) => (h !== null && a.cls.includes(h) ? null : a.cls[0]))}
                      className={cn(
                        'grid w-full gap-0.5 rounded-control px-2 py-1 text-left transition-colors',
                        on ? 'bg-accent-soft' : 'hover:bg-surface-3/60',
                      )}
                    >
                      <code className="font-mono text-[12px] break-all text-ink">{a.re.replace(/^ /, '␣')}</code>
                      <span className="text-[13px] text-ink-2">{a.text}</span>
                    </button>
                  </li>
                )
              })}
            </ol>
          </div>
        </div>
      </div>
    </Figure>
  )
}
