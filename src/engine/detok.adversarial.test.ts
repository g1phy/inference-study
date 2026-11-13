/**
 * Независимая проверка incrementalDetokenize:
 *  - пошагово = HF tokenizers DecodeStream (эталон в fixtures/qwen3-adversarial.json);
 *  - на каждом шаге отданный текст — префикс HF decode(ids[:k+1]), без утечек U+FFFD и потерь байтов;
 *  - stop-строки: итог = текст до первого вхождения, отданное по шагам — префиксы итога, хвост не теряется;
 *  - vLLM-буфер считается в code point (len() в Python) и не режет суррогатные пары;
 *  - include_stop_str_in_output: ни vLLM, ни Dynamo ничего не придерживают.
 */
import { describe, expect, it } from 'vitest'
import { incrementalDetokenize, type DetokOptions } from './detok'
import { mulberry32 } from './rng'
import { loadQwen3ForTests, readProjectFile } from './tokenizer/testing'

const tok = loadQwen3ForTests()
const fx = JSON.parse(readProjectFile('src/engine/tokenizer/fixtures/qwen3-adversarial.json')) as {
  encode: Array<{ text: string; ids: number[] }>
  stream: Array<{ ids: number[]; skip: boolean; steps: Array<string | null> }>
}
const toks = (ids: number[]) => ids.map((id) => tok.token(id))
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/

describe('incrementalDetokenize = HF DecodeStream пошагово', () => {
  it(`${fx.stream.length} последовательностей (skip_special_tokens и без, неполный UTF-8)`, () => {
    expect(fx.stream.length).toBeGreaterThanOrEqual(30)
    for (const c of fx.stream) {
      const steps = incrementalDetokenize(toks(c.ids), tok, { skipSpecial: c.skip })
      expect(steps.map((s) => s.emitted), JSON.stringify(c.ids)).toEqual(c.steps.map((s) => s ?? ''))
    }
  })

  it('с промптом в окне то же самое (окно из 7 токенов промпта не меняет текст byte-level BPE)', () => {
    const prompt = tok.encode('<|im_start|>user\nПривет 🫠🫠<|im_end|>\n<|im_start|>assistant\n')
    for (const c of fx.stream.slice(0, 20)) {
      const steps = incrementalDetokenize(toks(c.ids), tok, { skipSpecial: c.skip, promptTokens: prompt })
      expect(steps.map((s) => s.emitted)).toEqual(c.steps.map((s) => s ?? ''))
    }
  })
})

describe('на каждом шаге: отданное = префикс HF decode(ids[:k+1]), без U+FFFD и потерь', () => {
  const prompts = [undefined, tok.encode('Вопрос:'), tok.encode('эмодзи в конце окна 🫠🫠🫠')]
  for (const [pi, prompt] of prompts.entries()) {
    it(`все ${fx.encode.length} строк фикстуры как выход модели, промпт #${pi}`, () => {
      for (const c of fx.encode) {
        for (const skipSpecial of [true, false]) {
          const steps = incrementalDetokenize(toks(c.ids), tok, { skipSpecial, promptTokens: prompt })
          let out = ''
          steps.forEach((s, k) => {
            out += s.emitted
            const full = tok.decode(c.ids.slice(0, k + 1), { skipSpecial })
            if (!full.startsWith(out)) throw new Error(`не префикс на шаге ${k}: ${JSON.stringify(c.text)}`)
            // всё, что сложилось в символы, уже отдано
            if (!full.endsWith('�') && out !== full) throw new Error(`недодано на шаге ${k}: ${JSON.stringify(c.text)}`)
            expect(s.output).toBe(out)
          })
          const full = tok.decode(c.ids, { skipSpecial })
          if (!full.endsWith('�')) expect(out).toBe(full)
          expect(LONE_SURROGATE.test(out)).toBe(false)
        }
      }
    })
  }
})

/** Эталон stop-строк: текст до самого раннего конца вхождения (как check_stop_strings в vLLM). */
function referenceStop(full: string, stops: string[], include: boolean): { text: string; hit?: string } {
  let best: { end: number; at: number; s: string } | null = null
  for (const s of stops) {
    const at = full.indexOf(s)
    if (at >= 0 && (!best || at + s.length < best.end)) best = { end: at + s.length, at, s }
  }
  if (!best) return { text: full }
  return { text: full.slice(0, include ? best.end : best.at), hit: best.s }
}

describe('stop-строки: свойства на случайных текстах и стопах', () => {
  const rng = mulberry32(20261112)
  const texts = fx.encode.map((c) => c.text).filter((t) => t.length >= 6 && !t.includes('<|'))
  const pick = <T,>(a: T[]) => a[Math.floor(rng() * a.length)]

  for (const mode of ['jail', 'vllm'] as const) {
    for (const includeStopStr of [false, true]) {
      it(`${mode}, includeStopStr=${includeStopStr}: итог = эталон, по шагам — префиксы, буфер не теряется`, () => {
        let hits = 0
        for (let n = 0; n < 400; n++) {
          const text = pick(texts)
          const cps = [...text]
          const a = Math.floor(rng() * cps.length)
          const len = 1 + Math.floor(rng() * Math.min(6, cps.length - a))
          const stops = [cps.slice(a, a + len).join(''), pick(['</answer>', '\nUser:', 'zzz', '🙂x', '"}'])]
          const ids = tok.encode(text).map((t) => t.id)
          const opts: DetokOptions = { stop: stops, stopMode: mode, includeStopStr }
          const full = tok.decode(ids, { skipSpecial: true })
          const ref = referenceStop(full, stops, includeStopStr)
          const steps = incrementalDetokenize(toks(ids), tok, opts)
          const emitted = steps.map((s) => s.emitted).join('')
          const ctx = JSON.stringify({ text, stops, mode, includeStopStr })
          // стоп внутри неполного UTF-8-хвоста в конце не виден ни нам, ни эталону
          if (full.endsWith('�')) continue
          expect(emitted, ctx).toBe(ref.text)
          expect(steps.at(-1)!.output, ctx).toBe(ref.text)
          expect(!!steps.at(-1)!.stopped, ctx).toBe(ref.hit !== undefined)
          if (ref.hit !== undefined) {
            hits++
            expect(steps.at(-1)!.stopReason, ctx).toBe(ref.hit)
          }
          let acc = ''
          for (const s of steps) {
            acc += s.emitted
            expect(ref.text.startsWith(acc), ctx).toBe(true)
            expect(LONE_SURROGATE.test(s.emitted), ctx).toBe(false)
          }
        }
        // свойство не пустое: большинство случайных стопов действительно срабатывают
        expect(hits).toBeGreaterThan(250)
      })
    }
  }

  it('finished=false: придержанный хвост остаётся в held, ничего не отдано сверх возможного', () => {
    for (let n = 0; n < 200; n++) {
      const text = pick(texts)
      const ids = tok.encode(text).map((t) => t.id)
      const steps = incrementalDetokenize(toks(ids), tok, { stop: ['</answer>', '\n\n\n'], finished: false })
      const full = tok.decode(ids, { skipSpecial: true })
      const emitted = steps.map((s) => s.emitted).join('')
      expect(full.startsWith(emitted)).toBe(true)
      const last = steps.at(-1)!
      if (!last.stopped && !full.endsWith('�')) expect(emitted + last.stopHeld).toBe(full)
    }
  })
})

describe('vLLM: буфер stop-строк в code point, include_stop_str_in_output без удержания (баги)', () => {
  it("stop '🙂🙂' (2 code point, 4 единицы UTF-16) держит 1 символ, а не 3, и не режет суррогатную пару", () => {
    const t = toks(tok.encode('ab🫠cd').map((x) => x.id))
    const steps = incrementalDetokenize(t, tok, { stop: ['🙂🙂'], stopMode: 'vllm', finished: false })
    const emitted = steps.map((s) => s.emitted).join('')
    expect(emitted).toBe('ab🫠c') // держим ровно 1 code point: 'd'
    for (const s of steps) expect(LONE_SURROGATE.test(s.emitted)).toBe(false)
    const steps2 = incrementalDetokenize(toks(tok.encode('x🫠').map((q) => q.id)), tok, { stop: ['ab'], stopMode: 'vllm', finished: false })
    // буфер 1 code point: эмодзи придержан целиком, а не половиной пары
    expect(steps2.map((s) => s.emitted).join('')).toBe('x')
    expect(steps2.at(-1)!.stopHeld).toBe('🫠')
  })

  it('include_stop_str_in_output: vLLM (stop_buffer_length = 0) и Dynamo (visible) отдают текст сразу', () => {
    const t = toks(tok.encode('Ответ: 4</answer> и ещё').map((x) => x.id))
    for (const stopMode of ['vllm', 'jail'] as const) {
      const steps = incrementalDetokenize(t, tok, { stop: ['</answer>'], stopMode, includeStopStr: true, finished: false })
      expect(steps.slice(0, 5).map((s) => s.emitted)).toEqual(['От', 'вет', ':', ' ', '4'])
      expect(steps.map((s) => s.emitted).join('')).toBe('Ответ: 4</answer>')
      expect(steps.at(-1)!.stopped).toBe(true)
    }
  })
})
