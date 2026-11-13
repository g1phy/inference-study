/**
 * Независимая проверка templates.ts: дополнительные кейсы, отрендеренные настоящим transformers 5
 * (Qwen3 — apply_chat_template с официальным Jinja; остальные — render_jinja_template),
 * см. tokenizer/fixtures/gen_templates_adversarial.py.
 */
import { describe, expect, it } from 'vitest'
import { renderTemplate, type TemplateInputEx } from './templates'
import { readProjectFile } from './tokenizer/testing'

const fx = JSON.parse(readProjectFile('src/engine/templates.adversarial.fixture.json')) as {
  source: string
  cases: Array<{ name: string; input: TemplateInputEx; expected: string }>
}

describe('renderTemplate побайтно = transformers (доп. кейсы)', () => {
  it('эталон снят transformers, кейсов Qwen3 не меньше 20', () => {
    expect(fx.source).toContain('transformers')
    expect(fx.cases.filter((c) => c.input.family === 'qwen3').length).toBeGreaterThanOrEqual(20)
  })
  for (const c of fx.cases) {
    it(c.name, () => {
      const r = renderTemplate(c.input)
      expect(r.text).toBe(c.expected)
      expect(r.segments.map((s) => s.text).join('')).toBe(r.text)
    })
  }
})

describe('устойчивость входа', () => {
  it('content: null у ответа ассистента с tool_calls (как в OpenAI API) рендерится как пустая строка, без исключения', () => {
    const msgs = [
      { role: 'user', content: 'Погода?' },
      { role: 'assistant', content: null, tool_calls: [{ name: 'get_weather', arguments: { city: 'Москва' } }] },
      { role: 'tool', content: '-3' },
    ] as unknown as TemplateInputEx['messages']
    for (const family of ['qwen3', 'llama3', 'deepseek-r1', 'gpt-oss'] as const) {
      expect(() => renderTemplate({ family, messages: msgs })).not.toThrow()
    }
    const asEmpty = renderTemplate({ family: 'qwen3', messages: msgs.map((m) => ({ ...m, content: m.content ?? '' })) })
    expect(renderTemplate({ family: 'qwen3', messages: msgs }).text).toBe(asEmpty.text)
  })
})
