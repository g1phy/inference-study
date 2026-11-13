import { useEffect, useId, useMemo, useRef } from 'react'
import { motion } from 'motion/react'
import { ArrowRightIcon } from '@phosphor-icons/react'
import { Figure, FieldLabel, Player, TokenStrip, TokenStripSkeleton, cn, fmt, tint, usePlayer } from '@/ui'
import { rawToBytes } from '@/engine/tokenizer'
import type { BpeTrace, Tokenizer } from '@/engine/types'

export const BPE_PRESETS = [' полезный', 'Привет', ' world', ' tokenization', '🤖']
const MAX_LEN = 40

export interface BpeStepsProps {
  tokenizer: Tokenizer | null
  word: string
  onWord: (w: string) => void
}

/** Ключи кусков: смещение первого байта (стабильно между шагами, чтобы склейку можно было анимировать). */
function pieceKeys(raw: string[]): number[] {
  const out: number[] = []
  let off = 0
  for (const r of raw) {
    out.push(off)
    off += rawToBytes(r)?.length ?? r.length
  }
  return out
}

/** Кусок так, как его видит человек: пробел «·», перевод строки «↵», неполный UTF-8 байтами «\xD0» (как в ленте кусков). */
const shown = (piece: string) => piece.replace(/ /g, '·').replace(/\n/g, '↵')

export function BpeSteps({ tokenizer, word, onWord }: BpeStepsProps) {
  const inputId = useId()
  // считаем символы (code points), а не единицы UTF-16: «🤖» это 1 символ, и суррогатная пара не режется пополам
  const clipped = Array.from(word).slice(0, MAX_LEN).join('')
  const clippedLen = Array.from(clipped).length
  const trace: BpeTrace | null = useMemo(() => {
    if (!tokenizer || !clipped) return null
    try {
      return tokenizer.trace(clipped)
    } catch {
      return null
    }
  }, [tokenizer, clipped])
  const steps = trace?.steps.length ?? 1
  const p = usePlayer(steps, { intervalMs: 900 })
  const { setStep, setPlaying } = p

  // новое слово: с начала
  useEffect(() => {
    setPlaying(false)
    setStep(0)
  }, [clipped, setPlaying, setStep])

  const labels = useMemo(() => {
    if (!trace) return []
    return trace.steps.map((s, i) => (i === 0 ? 'байты' : i === trace.steps.length - 1 ? 'итог' : `склейка ${i}, ранг ${fmt.int(trace.steps[i - 1].merge!.rank)}`))
  }, [trace])

  const s = Math.min(p.step, steps - 1)
  const cur = trace?.steps[s]
  const prevMerge = s > 0 ? trace?.steps[s - 1].merge : undefined
  const last = s === steps - 1
  const keys = cur ? pieceKeys(cur.rawPieces) : []
  // применённые склейки: читаемые половинки берём из кусков того шага, на котором склейка случилась
  const applied = trace
    ? trace.steps.slice(0, s).map((x) => {
        const m = x.merge!
        return { ...m, leftText: shown(x.pieces[m.index] ?? m.left), rightText: shown(x.pieces[m.index + 1] ?? m.right) }
      })
    : []
  // журнал склеек прокручивается к последней внутри себя (страницу не двигаем)
  const logRef = useRef<HTMLOListElement>(null)
  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [applied.length])

  return (
    <Figure
      id="tokenization-figure-bpe"
      title="BPE по шагам на настоящих склейках Qwen3"
      controls={<Player {...p.bind} labels={labels} aria-label="Шаги BPE" />}
      caption="Попробуйте: выберите « полезный», нажмите пуск и следите, как половинки русских букв склеиваются раньше самих букв. Потом сравните ранги с « world»."
    >
      <div className="grid gap-4">
        <div className="grid gap-1.5">
          <FieldLabel htmlFor={inputId} aside={`${fmt.int(clippedLen)} / ${MAX_LEN}`}>
            Слово или короткая фраза (пробел в начале важен)
          </FieldLabel>
          <input
            id={inputId}
            value={word}
            maxLength={MAX_LEN}
            onChange={(e) => onWord(e.currentTarget.value)}
            spellCheck={false}
            className="h-10 w-full rounded-control bg-surface-2 px-3 font-mono text-[14px] text-ink shadow-[inset_0_0_0_1px_var(--line)] transition-shadow hover:shadow-[inset_0_0_0_1px_var(--line-strong)] focus-visible:outline-2 focus-visible:outline-offset-0"
          />
          <div className="flex flex-wrap gap-1.5 pt-1" role="group" aria-label="Примеры слов">
            {BPE_PRESETS.map((w) => (
              <button
                key={w}
                type="button"
                aria-pressed={clipped === w}
                onClick={() => onWord(w)}
                className={cn(
                  'h-7 rounded-control px-2.5 font-mono text-[13px] whitespace-pre transition-colors',
                  clipped === w ? 'bg-accent-soft text-accent-hover shadow-[inset_0_0_0_1px_var(--accent)]' : 'bg-surface text-ink-2 shadow-[inset_0_0_0_1px_var(--line-strong)] hover:bg-surface-3 hover:text-ink',
                )}
              >
                {w.replace(/ /g, '·')}
              </button>
            ))}
          </div>
        </div>

        {!tokenizer ? (
          <TokenStripSkeleton count={16} />
        ) : !trace || !cur ? (
          <p className="text-[14px] text-ink-3">Введите слово, чтобы увидеть склейки.</p>
        ) : (
          <div className="grid gap-4 @min-[760px]:grid-cols-[minmax(0,1fr)_220px]">
            <div className="grid min-w-0 content-start gap-3">
              <div>
                <div className="mb-1.5 text-[12.5px] text-ink-3">Байты UTF-8 ({fmt.int(trace.bytes.length)})</div>
                <div className="flex flex-wrap gap-[3px] font-mono text-[11.5px] text-ink-2 tabular-nums">
                  {trace.bytes.map((b, i) => (
                    <span key={i} className="rounded-[4px] bg-surface-2 px-1 py-px">
                      {fmt.hex([b])}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-1.5 text-[12.5px] text-ink-3">
                  {s === 0 ? 'Начало: каждый байт отдельным куском' : `Куски после ${fmt.int(s)}-й склейки`} · {fmt.int(cur.pieces.length)}
                </div>
                <div className="flex flex-wrap items-end gap-[3px]" role="list" aria-label="Куски на текущем шаге">
                  {cur.pieces.map((piece, i) => {
                    const inPair = cur.merge && (i === cur.merge.index || i === cur.merge.index + 1)
                    const fresh = prevMerge && i === prevMerge.index
                    return (
                      <motion.span
                        layout
                        transition={{ duration: 0.25 }}
                        key={keys[i]}
                        role="listitem"
                        className={cn(
                          'inline-flex min-w-[26px] flex-col items-center rounded-[6px] px-1.5 py-1 font-mono',
                          inPair ? 'shadow-[0_0_0_2px_var(--accent)]' : 'shadow-[inset_0_0_0_1px_var(--line)]',
                        )}
                        style={{ background: fresh ? tint('prefill', 18) : 'var(--surface)' }}
                      >
                        <span className={cn('text-[14px] leading-5 whitespace-pre text-ink', /\\x/.test(piece) && 'text-[12px] text-ink-2 italic')}>
                          {piece.replace(/ /g, '·').replace(/\n/g, '↵')}
                        </span>
                        <span className={cn('text-[10.5px] leading-4 whitespace-pre', fresh ? 'text-ink-2' : 'text-ink-3')}>{cur.rawPieces[i]}</span>
                      </motion.span>
                    )
                  })}
                </div>
              </div>

              <div className="rounded-panel border border-line px-3.5 py-2.5 text-[14.5px] leading-relaxed text-ink" aria-live="polite">
                {cur.merge ? (
                  <span className="grid gap-1">
                    <span className="inline-flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span>Следующая склейка:</span>
                      <code className="rounded-[4px] bg-surface-2 px-1 font-mono text-[13px] whitespace-pre">{shown(cur.pieces[cur.merge.index] ?? cur.merge.left)}</code>
                      <span>+</span>
                      <code className="rounded-[4px] bg-surface-2 px-1 font-mono text-[13px] whitespace-pre">{shown(cur.pieces[cur.merge.index + 1] ?? cur.merge.right)}</code>
                      <ArrowRightIcon size={14} aria-hidden className="text-ink-3" />
                      <code className="rounded-[4px] bg-surface-2 px-1 font-mono text-[13px] whitespace-pre">
                        {shown(trace.steps[s + 1]?.pieces[cur.merge.index] ?? cur.merge.left + cur.merge.right)}
                      </code>
                      <span className="text-ink-2">
                        ранг <b className="text-ink tabular-nums">{fmt.int(cur.merge.rank)}</b>: самая ранняя из возможных пар
                      </span>
                    </span>
                    <span className="font-mono text-[12px] text-ink-3">
                      в словаре: {cur.merge.left} + {cur.merge.right} → {cur.merge.left + cur.merge.right}
                    </span>
                  </span>
                ) : (
                  <span>
                    Склеивать больше нечего: ни одной пары соседних кусков нет в списке склеек. Итог:{' '}
                    <b>{fmt.tokens(trace.tokens.length)}</b> за {fmt.int(steps - 1)} {fmt.plural(steps - 1, ['склейку', 'склейки', 'склеек'])}.
                  </span>
                )}
              </div>

              {last && (
                <div>
                  <div className="mb-1.5 text-[12.5px] text-ink-3">Токены и их ID в словаре</div>
                  <TokenStrip tokens={trace.tokens} showIds variant="prompt" />
                </div>
              )}
            </div>

            <div className="min-w-0">
              <div className="mb-1.5 text-[12.5px] text-ink-3">Применённые склейки</div>
              <ol
                ref={logRef}
                tabIndex={0}
                className="grid max-h-[260px] content-start gap-0.5 overflow-y-auto rounded-control pr-1 font-mono text-[12px] scrollbar-thin focus-visible:-outline-offset-2"
                aria-label="Применённые склейки"
              >
                {applied.length === 0 && <li className="font-sans text-[13px] text-ink-3">пока ни одной</li>}
                {applied.map((m, i) => (
                  <li
                    key={i}
                    className={cn('flex items-baseline justify-between gap-2 rounded-[4px] px-1.5 py-0.5', i === applied.length - 1 ? 'bg-accent-soft' : '')}
                  >
                    <span className="min-w-0 truncate whitespace-pre text-ink" title={`в словаре: ${m.left} + ${m.right}`}>
                      {m.leftText} + {m.rightText}
                    </span>
                    <span className="shrink-0 text-ink-2 tabular-nums">{fmt.int(m.rank)}</span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        )}
      </div>
    </Figure>
  )
}
