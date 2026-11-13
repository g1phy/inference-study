/**
 * Jump-forward для главы «Structured output»: какие куски JSON грамматика допускает единственным образом
 * (forcedContinuation из structured.ts) и сколько проходов модели это экономит (оценка).
 */
import { forcedContinuation } from '@/engine/structured'
import type { JsonSchema, Token } from '@/engine/types'
import { WHITESPACE } from './trajectory'

export interface Piece {
  start: number
  end: number
  forced: boolean
}

/** Режет целевой JSON на куски: принудительные (грамматика знает их заранее) и свободные (выбирает модель). */
export function forcedPieces(schema: JsonSchema, text: string): Piece[] {
  const out: Piece[] = []
  let p = 0
  const push = (start: number, end: number, forced: boolean) => {
    const last = out[out.length - 1]
    if (last && last.forced === forced) last.end = end
    else out.push({ start, end, forced })
  }
  while (p < text.length) {
    const fc = forcedContinuation(schema, text.slice(0, p), { whitespace: WHITESPACE })
    if (fc && text.startsWith(fc, p)) {
      push(p, p + fc.length, true)
      p += fc.length
    } else {
      push(p, p + 1, false)
      p += 1
    }
  }
  return out
}

/** Сколько forward-проходов нужно с jump-forward и без (оценка). */
export function jumpForwardCost(tokens: Token[], pieces: Piece[]): { plain: number; jf: number; chunks: number; forcedTokens: number } {
  const forcedAt = (i: number) => pieces.some((pc) => pc.forced && i >= pc.start && i < pc.end)
  let pos = 0
  let free = 0
  let forcedTokens = 0
  for (const t of tokens) {
    const s = t.start ?? pos
    const e = t.end ?? pos + t.text.length
    pos = e
    let all = true
    for (let i = s; i < e; i++) if (!forcedAt(i)) all = false
    if (all) forcedTokens++
    else free++
  }
  const chunks = pieces.filter((p) => p.forced).length
  // +1: шаг, на котором выбирается EOS
  return { plain: tokens.length + 1, jf: free + chunks + 1, chunks, forcedTokens }
}

