/**
 * Инкрементальная (потоковая) детокенизация: id по одному → текст для стрима.
 *
 * Алгоритм prefix_offset / read_offset (HF TGI v0.9.4 `decode_token`, vLLM `detokenize_incrementally`,
 * research/notes-tokens-templates-structured.md §1.10, §3.8; notes-engines §1.8):
 *   prefix_text = decode(ids[prefix_offset : read_offset])
 *   new_text    = decode(ids[prefix_offset : ])
 *   если len(new_text) > len(prefix_text) и new_text не кончается на U+FFFD:
 *       отдать new_text[len(prefix_text):]; prefix_offset = read_offset; read_offset = len(ids)
 *   иначе ничего не отдавать (неполный UTF-8 придержан).
 * Окно «заряжается» последними 5 + 2 = 7 токенами промпта: read_offset = 7, prefix_offset = 7 − 5
 * (INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET = 5, notes-tokens §2.5).
 *
 * Stop-строки (текст, который может пересечь границу токенов, notes-tokens §1.11, §3.10):
 *   'jail' (по умолчанию, как Dynamo backend.rs): держим только самый длинный суффикс текста,
 *          который совпадает с началом какой-нибудь stop-строки;
 *   'vllm': держим фиксированный хвост max(len(stop)) − 1 символов (code point, как len() в Python).
 * При include_stop_str_in_output (includeStopStr) ничего не придерживается ни в vLLM (stop_buffer_length = 0),
 * ни в Dynamo (такие stop-строки «visible», jail их не ждёт).
 * При совпадении текст обрезается до stop-строки (или включает её при includeStopStr), stopped = true.
 *
 * skip_special_tokens: токены с hfSpecial=true не дают текста (у Qwen3 это <|im_end|>, <|endoftext|>;
 * <think> и <tool_call> остаются в тексте, их видят парсеры). По умолчанию true, как в vLLM.
 */
import type { DetokStep, Token, Tokenizer } from './types'

/** Опции детокенизатора. */
export interface DetokOptions {
  /** Stop-строки. */
  stop?: string[]
  /** Выкидывать токены с hfSpecial=true (skip_special_tokens). По умолчанию true. */
  skipSpecial?: boolean
  /** Как придерживать текст под stop-строки: 'jail' (Dynamo, по умолчанию) или 'vllm' (max(len) − 1 code point). */
  stopMode?: 'jail' | 'vllm'
  /** Оставить stop-строку в выводе (include_stop_str_in_output). По умолчанию false. */
  includeStopStr?: boolean
  /** Токены промпта: последние 7 «заряжают» окно (как в vLLM). Можно id или Token. */
  promptTokens?: Array<Token | number>
  /**
   * Стоп-токены (EOS): такой токен заканчивает поток и сам текста не даёт (vLLM stop_terminated);
   * при includeStopStr его текст выводится (если токен не выкинут skipSpecial), как в vLLM.
   */
  stopTokenIds?: number[]
  /** Генерация закончилась на последнем токене: придержанный под stop-строку хвост выдаётся. По умолчанию true. */
  finished?: boolean
}

/** Шаг детокенизации с подробностями для визуализации. */
export interface DetokStepEx extends DetokStep {
  /** Байты токенов после read_offset, которые ещё не сложились в символ. */
  heldBytes: number[]
  /** Текст, придержанный под возможную stop-строку. */
  stopHeld: string
  /** Весь текст, отданный клиенту к концу шага. */
  output: string
  /** Сработавшая stop-строка или id стоп-токена. */
  stopReason?: string | number
}

/** Начальный сдвиг окна в токенах (vLLM INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET). */
export const INITIAL_DETOK_OFFSET = 5

/** Число code point в строке (как len() в Python: vLLM считает буфер stop-строк в символах, а не в UTF-16). */
function cpLength(s: string): number {
  let n = 0
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i)
    // пара суррогатов — один code point
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < s.length) {
      const d = s.charCodeAt(i + 1)
      if (d >= 0xdc00 && d <= 0xdfff) i++
    }
    n++
  }
  return n
}

/** UTF-16 индекс, с которого начинаются последние k code point строки (не режет суррогатную пару). */
function startOfLastCodePoints(s: string, k: number): number {
  let i = s.length
  while (k > 0 && i > 0) {
    i--
    const c = s.charCodeAt(i)
    if (c >= 0xdc00 && c <= 0xdfff && i > 0) {
      const h = s.charCodeAt(i - 1)
      if (h >= 0xd800 && h <= 0xdbff) i--
    }
    k--
  }
  return i
}

/** Длина самого длинного суффикса text, который является собственным префиксом одной из stop-строк. */
export function stopPrefixHoldback(text: string, stops: string[]): number {
  let best = 0
  for (const s of stops) {
    for (let len = Math.min(s.length - 1, text.length); len > best; len--) {
      if (text.endsWith(s.slice(0, len))) {
        best = len
        break
      }
    }
  }
  return best
}

/**
 * Пошаговая детокенизация потока токенов. Возвращает по шагу на каждый обработанный токен;
 * после срабатывания stop массив обрывается. prefixOffset/readOffset считаются в индексах
 * сгенерированных токенов (отрицательные значения указывают в окно из промпта: −1 — последний токен промпта).
 */
export function incrementalDetokenize(tokens: Token[], tokenizer: Tokenizer, opts: DetokOptions = {}): DetokStepEx[] {
  const skip = opts.skipSpecial ?? true
  const stops = (opts.stop ?? []).filter((s) => s.length > 0)
  const maxStop = stops.reduce((m, s) => Math.max(m, s.length), 0)
  // vLLM: stop_buffer_length = max(len(s)) − 1 в символах Python (code point), и 0 при
  // include_stop_str_in_output (vllm/v1/engine/detokenizer.py). Dynamo при include_stop_str_in_output
  // кладёт stop-строки в visible_stop_sequences, а jail держит только префиксы hidden (backend.rs).
  const vllmBuffer = opts.includeStopStr ? 0 : Math.max(0, stops.reduce((m, s) => Math.max(m, cpLength(s)), 0) - 1)
  const mode = opts.stopMode ?? 'jail'
  const finished = opts.finished ?? true
  const stopIds = new Set(opts.stopTokenIds ?? [])

  const promptIds = (opts.promptTokens ?? []).map((t) => (typeof t === 'number' ? t : t.id))
  const ctx = promptIds.slice(-(INITIAL_DETOK_OFFSET + 2))
  const ids: number[] = [...ctx]
  const base = ctx.length
  let prefixOffset = Math.max(ctx.length - INITIAL_DETOK_OFFSET, 0)
  let readOffset = ctx.length

  const visible = (list: number[]) => (skip ? list.filter((id) => !tokenizer.token(id).hfSpecial) : list)
  const decodeRange = (a: number, b: number) => tokenizer.decode(visible(ids.slice(a, b)))

  let output = ''
  let sent = 0
  const steps: DetokStepEx[] = []

  for (let k = 0; k < tokens.length; k++) {
    const tok = tokens[k]
    const isLast = k === tokens.length - 1
    const naiveText = skip && tok.hfSpecial ? '' : tokenizer.decode([tok.id])

    const isStopToken = stopIds.has(tok.id)
    if (isStopToken && !opts.includeStopStr) {
      // Стоп-токен: в текст не идёт, придержанный хвост выдаётся (генерация закончена).
      // При include_stop_str_in_output vLLM детокенизирует и его (с учётом skip_special_tokens) — ниже.
      const emitted = output.slice(sent)
      sent = output.length
      steps.push({
        index: k,
        token: tok,
        naiveText,
        emitted,
        held: '',
        prefixOffset: prefixOffset - base,
        readOffset: readOffset - base,
        stopped: true,
        heldBytes: [],
        stopHeld: '',
        output,
        stopReason: tok.id,
      })
      break
    }

    ids.push(tok.id)
    const prefixText = decodeRange(prefixOffset, readOffset)
    const newText = decodeRange(prefixOffset, ids.length)
    let delta = ''
    let pending = ''
    if (newText.length > prefixText.length && !newText.endsWith('�')) {
      delta = newText.slice(prefixText.length)
      prefixOffset = readOffset
      readOffset = ids.length
    } else {
      pending = newText.slice(prefixText.length)
    }
    const heldBytes: number[] = []
    for (const id of visible(ids.slice(readOffset))) heldBytes.push(...tokenizer.token(id).bytes)

    const before = output.length
    output += delta

    // Поиск stop-строки только там, где мог появиться новый матч.
    if (stops.length > 0 && delta) {
      const from = Math.max(0, before - maxStop + 1)
      let hitAt = -1
      let hitEnd = Infinity
      let hit = ''
      for (const s of stops) {
        const at = output.indexOf(s, from)
        if (at >= 0 && at + s.length < hitEnd) {
          hitAt = at
          hitEnd = at + s.length
          hit = s
        }
      }
      if (hitAt >= 0) {
        const cut = Math.max(sent, opts.includeStopStr ? hitEnd : hitAt)
        const emitted = output.slice(sent, cut)
        output = output.slice(0, cut)
        sent = cut
        steps.push({
          index: k,
          token: tok,
          naiveText,
          emitted,
          held: '',
          prefixOffset: prefixOffset - base,
          readOffset: readOffset - base,
          stopped: true,
          heldBytes: [],
          stopHeld: '',
          output,
          stopReason: hit,
        })
        break
      }
    }

    // Сколько хвоста придержать под возможную stop-строку (UTF-16 позиция конца выдачи).
    let releaseEnd = output.length
    if (stops.length > 0 && !opts.includeStopStr) {
      releaseEnd =
        mode === 'jail' ? output.length - stopPrefixHoldback(output, stops) : startOfLastCodePoints(output, vllmBuffer)
    }
    let end = Math.max(sent, releaseEnd)
    if ((isLast && finished) || isStopToken) end = output.length
    const emitted = output.slice(sent, end)
    sent = end
    const stopHeld = output.slice(sent)
    steps.push({
      index: k,
      token: tok,
      naiveText,
      emitted,
      held: stopHeld + pending,
      prefixOffset: prefixOffset - base,
      readOffset: readOffset - base,
      heldBytes,
      stopHeld,
      output: output.slice(0, sent),
    })
    if (isStopToken) {
      // includeStopStr: текст стоп-токена уже в выводе, генерация закончена
      const last = steps[steps.length - 1]
      last.stopped = true
      last.stopReason = tok.id
      break
    }
  }
  return steps
}

/** Наивная детокенизация «каждый токен отдельно»: так появляются «�» на разрезанных символах. */
export function naiveDetokenize(tokens: Token[], tokenizer: Tokenizer, opts: { skipSpecial?: boolean } = {}): string {
  const skip = opts.skipSpecial ?? true
  return tokens.map((t) => (skip && t.hfSpecial ? '' : tokenizer.decode([t.id]))).join('')
}
