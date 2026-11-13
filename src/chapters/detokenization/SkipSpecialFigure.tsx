import { useMemo, useState } from 'react'
import { CheckCircleIcon, StopCircleIcon, XCircleIcon } from '@phosphor-icons/react'
import { Figure, Segmented, Toggle, TokenStrip, TokenStripSkeleton, cn } from '@/ui'
import { QWEN3_STOP_TOKEN_IDS } from '@/engine/tokenizer'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'

/** Вывод Qwen3 с рассуждением (иллюстрация, токены настоящие). */
const QWEN_OUTPUT = '<think>\nКоротко объясню.\n</think>\n\nKV cache хранит ключи и значения прошлых токенов.<|im_end|>'

/** Вывод gpt-oss в формате Harmony (иллюстрация). Все токены разметки у gpt-oss special=true. */
const HARMONY_OUTPUT =
  '<|channel|>analysis<|message|>Нужна простая аналогия.<|end|><|start|>assistant<|channel|>final<|message|>KV cache хранит ключи и значения.<|return|>'
const HARMONY_SPECIAL = ['<|start|>', '<|end|>', '<|message|>', '<|channel|>', '<|return|>', '<|call|>', '<|constrain|>']
/** Стоп-токены gpt-oss из generation_config (<|return|>, <|call|>): последний из них сервер отрезает до детокенизации. */
const HARMONY_STOP = ['<|return|>', '<|call|>']

type Model = 'qwen3' | 'gpt-oss'

/** Разбивает текст на куски: литералы спецтокенов отдельно. */
function splitSpecial(text: string, literals: readonly string[]): Array<{ text: string; special: boolean }> {
  const out: Array<{ text: string; special: boolean }> = []
  let i = 0
  let buf = ''
  while (i < text.length) {
    const lit = literals.find((l) => text.startsWith(l, i))
    if (lit) {
      if (buf) out.push({ text: buf, special: false })
      buf = ''
      out.push({ text: lit, special: true })
      i += lit.length
    } else {
      buf += text[i]
      i++
    }
  }
  if (buf) out.push({ text: buf, special: false })
  return out
}

export function SkipSpecialFigure() {
  const { tokenizer, status } = useTokenizer()
  const [model, setModel] = useState<Model>('qwen3')
  const [skip, setSkip] = useState(true)

  const qwen = useMemo(() => {
    if (!tokenizer) return null
    const tokens = tokenizer.encode(QWEN_OUTPUT)
    const added = tokens.filter((t) => t.special)
    // стоп-токен в конце (<|im_end|>) сервер отрезает ещё до детокенизации при любом skip_special_tokens:
    // vLLM stop_terminated, SGLang trim_matched_stop, скрытые стоп-id в Dynamo (по умолчанию include_stop_str_in_output=false)
    const last = tokens[tokens.length - 1]
    const body = last && QWEN3_STOP_TOKEN_IDS.includes(last.id) ? tokens.slice(0, -1) : tokens
    const visible = skip ? body.filter((t) => !t.hfSpecial) : body
    return { tokens, added, text: tokenizer.decode(visible.map((t) => t.id)) }
  }, [tokenizer, skip])

  // у gpt-oss завершающий <|return|> тоже стоп-токен: в текст не попадает ни при каком значении флага
  const harmonyParts = useMemo(
    () => splitSpecial(HARMONY_OUTPUT, HARMONY_SPECIAL).map((p, i, all) => ({ ...p, stop: i === all.length - 1 && HARMONY_STOP.includes(p.text) })),
    [],
  )
  const harmonyText = harmonyParts
    .filter((p) => !p.stop && !(skip && p.special))
    .map((p) => p.text)
    .join('')

  return (
    <Figure
      id="detokenization-figure-skip-special"
      title="skip_special_tokens: что исчезает из текста"
      width="wide"
      caption="Попробуйте: выключите skip_special_tokens у Qwen3. Текст ответа не изменится: теги <think> не special и остаются при любом значении флага, а <|im_end|> хоть и special, но это стоп-токен, и по умолчанию сервер отрезает его ещё до детокенизации. Потом переключитесь на gpt-oss: там флаг стирает всю разметку каналов. Тексты условные."
    >
      <div className="grid gap-5">
        <div className="flex flex-wrap items-end gap-x-8 gap-y-4">
          <Segmented<Model>
            label="Модель"
            value={model}
            onChange={setModel}
            wrap
            options={[
              { value: 'qwen3', label: 'Qwen3' },
              { value: 'gpt-oss', label: 'gpt-oss (Harmony)' },
            ]}
          />
          <Toggle label="skip_special_tokens" checked={skip} onChange={setSkip} hint="В vLLM по умолчанию включён" />
        </div>

        {model === 'qwen3' && (
          <>
            {!qwen && status !== 'error' && <TokenStripSkeleton count={20} />}
            {qwen && (
              <div className="grid gap-4">
                <div className="grid gap-2">
                  <div className="text-sm font-medium text-ink-2">Токены ответа</div>
                  <TokenStrip tokens={qwen.tokens} variant="generated" size="sm" />
                </div>
                <ul className="grid gap-1.5 text-[14px] text-ink-2">
                  {dedupe(qwen.added.map((t) => t.id)).map((id) => {
                    const t = qwen.added.find((x) => x.id === id)!
                    const stop = QWEN3_STOP_TOKEN_IDS.includes(t.id)
                    const removed = skip && t.hfSpecial
                    return (
                      <li key={id} className="flex flex-wrap items-center gap-x-2 gap-y-1">
                        <span className="rounded-chip border border-dashed border-special px-1 font-mono text-[13px] text-ink-2">{t.raw}</span>
                        <span className="font-mono text-xs text-ink-3 tabular-nums">id {t.id}</span>
                        <span className="font-mono text-xs text-ink-3">special={t.hfSpecial ? 'true' : 'false'}</span>
                        <span className="inline-flex items-center gap-1">
                          {stop ? (
                            <StopCircleIcon size={16} aria-hidden className="text-ink-2" />
                          ) : removed ? (
                            <XCircleIcon size={16} aria-hidden style={{ color: 'var(--bad)' }} />
                          ) : (
                            <CheckCircleIcon size={16} aria-hidden style={{ color: 'var(--good)' }} />
                          )}
                          {stop ? 'стоп-токен: сервер отрезает его до детокенизации' : removed ? 'выкинут из текста' : 'остаётся в тексте'}
                        </span>
                      </li>
                    )
                  })}
                </ul>
                <Result text={qwen.text} />
              </div>
            )}
          </>
        )}

        {model === 'gpt-oss' && (
          <div className="grid gap-4">
            <div className="grid gap-2">
              <div className="text-sm font-medium text-ink-2">Что печатает модель</div>
              <div className="rounded-control bg-surface-2 px-3 py-2 font-mono text-[13px] leading-relaxed break-words text-ink">
                {harmonyParts.map((part, i) =>
                  part.special ? (
                    <span
                      key={i}
                      className={cn(
                        'mx-px rounded-[4px] px-0.5 text-ink-2 outline-1 -outline-offset-1 outline-dashed outline-[var(--c-special)]',
                        (skip || part.stop) && 'line-through decoration-[var(--bad)] decoration-[1.5px]',
                      )}
                    >
                      {part.text}
                    </span>
                  ) : (
                    <span key={i}>{part.text}</span>
                  ),
                )}
              </div>
            </div>
            <Result text={harmonyText} />
            <p className="text-[14px] text-ink-2">
              {skip
                ? 'Каналы analysis и final склеились в одну строку, и понять, где мысли, а где ответ, уже нельзя. Поэтому vLLM разбирает вывод gpt-oss не по тексту, а по номерам токенов.'
                : 'Со skip_special_tokens=false разметка каналов остаётся в тексте (кроме стоп-токена <|return|> в конце), но vLLM всё равно разбирает Harmony по номерам токенов, а не по тексту.'}
            </p>
          </div>
        )}
      </div>
    </Figure>
  )
}

function Result({ text }: { text: string }) {
  return (
    <div className="grid gap-2">
      <div className="text-sm font-medium text-ink-2">Текст после детокенизации</div>
      <div className="rounded-control bg-surface px-3 py-2 font-mono text-[13px] leading-relaxed break-words whitespace-pre-wrap text-ink shadow-[inset_0_0_0_1px_var(--line)]">
        {text.replace(/\n/g, '↵\n')}
      </div>
    </div>
  )
}

function dedupe(ids: number[]): number[] {
  return [...new Set(ids)]
}
