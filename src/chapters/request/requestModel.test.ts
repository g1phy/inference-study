import { describe, expect, it } from 'vitest'
import { loadQwen3ForTests } from '@/engine/tokenizer/testing'
import { buildRequestBody, requestBlocks, requestText, type RequestOptions } from './requestModel'
import { DEFAULT_USER_TEXT, completionJson, gpusForModel, simulateConversation, simulatedAnswer, warmCachedTokens } from './responseModel'
import { getGpu } from '@/engine/data/gpus'
import { getModel } from '@/engine/data/models'

const base: RequestOptions = {
  model: 'Qwen/Qwen3-8B',
  systemText: 'Ты полезный ассистент.',
  userText: DEFAULT_USER_TEXT,
  temperature: 0.7,
  topP: 0.95,
  maxTokens: 1024,
  tools: false,
  json: false,
  stream: true,
  thinking: true,
}

describe('requestBlocks', () => {
  it('склейка блоков = валидный JSON, равный телу запроса, при любых переключателях', () => {
    for (const tools of [false, true])
      for (const json of [false, true])
        for (const stream of [false, true])
          for (const thinking of [false, true])
            for (const systemText of ['', 'Ты полезный ассистент.', 'Кавычки " и \\ слэш\nи перевод строки']) {
              const o = { ...base, tools, json, stream, thinking, systemText }
              const text = requestText(requestBlocks(o))
              expect(JSON.parse(text)).toEqual(buildRequestBody(o))
            }
  })

  it('поля и порядок: stream последним, chat_template_kwargs только без рассуждения', () => {
    const ids = requestBlocks({ ...base, tools: true, json: true, thinking: false }).map((b) => b.id)
    expect(ids).toEqual(['model', 'messages', 'tools', 'chat_template_kwargs', 'temperature', 'top_p', 'max_tokens', 'stop', 'response_format', 'stream'])
    expect(requestBlocks(base).map((b) => b.id)).not.toContain('chat_template_kwargs')
  })
})

describe('responseModel', () => {
  it('prefix cache: целые блоки по 16, последний токен считается всегда', () => {
    expect(warmCachedTokens(43)).toBe(32)
    expect(warmCachedTokens(32)).toBe(16)
    expect(warmCachedTokens(33)).toBe(32)
    expect(warmCachedTokens(1)).toBe(0)
  })

  it('Qwen3-8B на H100 помещается в одну GPU, Llama-3.1-70B BF16 нет', () => {
    expect(gpusForModel(getModel('qwen3-8b'), getGpu('h100-sxm'))).toBe(1)
    expect(gpusForModel(getModel('llama-3.1-70b'), getGpu('h100-sxm'))).toBeGreaterThan(1)
  })

  it('чанки: все SSE-строки разложены по кадрам, текст собирается, usage сходится', () => {
    const tk = loadQwen3ForTests()
    const sim = simulateConversation({
      tokenizer: tk,
      promptTokens: 43,
      warm: true,
      answer: simulatedAnswer(DEFAULT_USER_TEXT, true, false),
      modelId: 'qwen3-8b',
      gpuId: 'h100-sxm',
      modelLabel: 'Qwen/Qwen3-8B',
    })
    const n = sim.answerTokens.length
    expect(sim.frameTimes).toHaveLength(n + 1)
    expect(sim.textAt).toHaveLength(n + 1)
    // кадры не убывают, последний чанк — [DONE] на последнем кадре
    for (let i = 1; i < sim.chunks.length; i++) expect(sim.chunks[i].frame).toBeGreaterThanOrEqual(sim.chunks[i - 1].frame)
    expect(sim.chunks.at(-1)!.chunk.line).toBe('data: [DONE]')
    expect(sim.chunks.at(-1)!.frame).toBe(n)
    // итоговый текст = рассуждение и ответ без тегов
    const final = sim.textAt[n]
    expect(final.reasoning.trim()).toBe('Нужно объяснить без формул и коротко.')
    expect(final.content).toContain('KV cache хранит ключи')
    expect(final.content).not.toContain('<|im_end|>')
    const usage = (sim.nonStream.usage as Record<string, unknown>)
    expect(usage.prompt_tokens).toBe(43)
    expect(usage.completion_tokens).toBe(n)
    expect(usage.prompt_tokens_details).toEqual({ cached_tokens: 32 })
    expect(sim.ttftMs).toBeGreaterThan(0)
    expect(sim.e2eMs).toBeGreaterThan(sim.ttftMs)
    // плотная запись ответа для фигуры: тот же JSON
    expect(JSON.parse(completionJson(sim))).toEqual(sim.nonStream)
  })

  it('плотная запись ответа совпадает с объектом и без рассуждения, с JSON-ответом', () => {
    const tk = loadQwen3ForTests()
    const sim = simulateConversation({
      tokenizer: tk,
      promptTokens: 190,
      warm: false,
      answer: simulatedAnswer('Другой вопрос', false, true),
      modelId: 'qwen3-8b',
      gpuId: 'h100-sxm',
      modelLabel: 'Qwen/Qwen3-8B',
    })
    expect(JSON.parse(completionJson(sim))).toEqual(sim.nonStream)
  })

  it('response_format: ответ валидный JSON', () => {
    const a = simulatedAnswer('Другой вопрос', false, true)
    expect(a.reasoning).toBeUndefined()
    expect(JSON.parse(a.content)).toHaveProperty('answer')
  })
})
