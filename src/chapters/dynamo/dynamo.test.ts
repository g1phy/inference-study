import { describe, expect, it } from 'vitest'
import { loadQwen3ForTests } from '@/engine/tokenizer/testing'
import { DEFAULT_SETTINGS } from '@/state/settings'
import { blockStored, computeRun, chosenNode } from './lifecycle'
import { piecesText, scriptFor } from '@/engine/scenarios'
import { buildRun } from '@/engine/flow'
import { GPUS_PER_NODE, buildYaml, layout } from './k8sModel'

const tok = loadQwen3ForTests()

describe('жизненный цикл запроса в Dynamo на настоящих токенах', () => {
  const run = computeRun({ ...DEFAULT_SETTINGS }, tok)

  it('промпт читателя: токены и блоки по 16, хвост не хэшируется', () => {
    expect(run.prompt.ids.length).toBe(run.prompt.tokens.length)
    expect(run.prompt.fullBlocks).toBe(Math.floor(run.prompt.ids.length / 16))
    expect(run.prompt.blocks.filter((b) => !b.full).every((b) => b.hash === '')).toBe(true)
    expect(run.kvBlocks).toBe(Math.ceil(run.prompt.ids.length / 16))
  })

  it('prefill-роутер берёт воркер без очереди, decode-роутер смотрит только на нагрузку', () => {
    expect(chosenNode('p', run.prefillRoute)).toBe('p1')
    expect(run.decodeRoute.scores.every((s) => (s.credit ?? 0) === 0)).toBe(true)
    expect(chosenNode('d', run.decodeRoute)).toBe('d2')
  })

  it('время и объём считаются по perf.ts и положительны', () => {
    expect(run.prefillMs).toBeGreaterThan(0)
    expect(run.kvBytes).toBe(run.prompt.ids.length * 147_456)
    expect(run.transfer.every((t) => t.ms > 0)).toBe(true)
  })

  it('ответ тот же, что в симуляторе: парсер разделяет reasoning и content, SSE кончается [DONE]', () => {
    const script = scriptFor({ ...DEFAULT_SETTINGS })
    let reasoning = ''
    let content = ''
    for (const e of run.answer.parser) for (const o of e.out) {
      if (o.kind === 'reasoning') reasoning += o.text
      else if (o.kind === 'content') content += o.text
    }
    expect(reasoning.trim()).toBe(piecesText(script.reasoning).trim())
    expect(content.trim()).toBe(piecesText(script.body).trim())
    expect(content).not.toContain('<think>')
    expect(run.answer.sse[run.answer.sse.length - 1].line).toBe('data: [DONE]')
  })

  it('usage в SSE считает тот воркер, который посчитал prefill в своём режиме', () => {
    const usageOf = (sse: typeof run.answer.sse) => sse.find((c) => c.json && 'usage' in c.json)!.json!.usage as { prompt_tokens_details: { cached_tokens: number } }
    expect(usageOf(run.answer.sse).prompt_tokens_details.cached_tokens).toBe(run.cachedTokens)
    expect(usageOf(run.answer.sseAgg).prompt_tokens_details.cached_tokens).toBe(run.aggCachedTokens)
    expect(run.aggCachedTokens).toBe(0)
  })

  it('с tools: в запросе tools и tool_choice, вызов уходит одним куском tool_calls, finish_reason tool_calls', () => {
    const r = computeRun({ ...DEFAULT_SETTINGS, scenarioId: 'tool', userText: 'Какая погода сейчас в Берлине?', thinking: false, tools: true }, tok)
    expect(r.request.tools).toBeDefined()
    expect(r.request.tool_choice).toBe('auto')
    expect(r.prompt.template.text).toContain('get_weather')
    const toolChunks = r.answer.sse.filter((c) => c.json && JSON.stringify(c.json).includes('"tool_calls"') && (c.json.choices as unknown[]).length > 0)
    const deltas = toolChunks.filter((c) => ((c.json!.choices as Array<{ delta: Record<string, unknown> }>)[0].delta.tool_calls as unknown[] | undefined)?.length)
    expect(deltas).toHaveLength(1)
    const tc = ((deltas[0].json!.choices as Array<{ delta: { tool_calls: Array<{ function: { name: string; arguments: string } }> } }>)[0].delta.tool_calls)[0]
    expect(tc.function.name).toBe('get_weather')
    expect(JSON.parse(tc.function.arguments)).toEqual({ city: 'Берлин' })
    const finish = r.answer.sse.find((c) => c.json && (c.json.choices as Array<{ finish_reason: string | null }>)?.[0]?.finish_reason)
    expect((finish!.json!.choices as Array<{ finish_reason: string }>)[0].finish_reason).toBe('tool_calls')
  })

  it('промпт с tools тот же, что у симулятора (тот же инструмент get_weather)', () => {
    const settings = { ...DEFAULT_SETTINGS, scenarioId: 'tool' as const, userText: 'Какая погода сейчас в Берлине?', thinking: false, tools: true }
    const r = computeRun(settings, tok)
    expect(r.prompt.ids).toEqual(buildRun(settings, tok).promptTokens.map((t) => t.id))
  })

  it('с json: в запросе response_format со схемой', () => {
    const r = computeRun({ ...DEFAULT_SETTINGS, scenarioId: 'json', userText: 'Извлеки данные: Анна, 29 лет, живёт в Казани', thinking: false, json: true }, tok)
    expect((r.request.response_format as { type: string }).type).toBe('json_schema')
    expect(r.jsonSchemaName).toBeTruthy()
    expect(r.answer.bodyKind).toBe('json')
  })

  it('BlockStored сообщает только новые полные блоки, родитель от последнего закэшированного', () => {
    const full = run.prompt.blocks.filter((b) => b.full)
    const none = blockStored(run.prompt, 0)!
    expect(none.event.parent_hash).toBeNull()
    expect(none.newBlocks).toBe(full.length)
    if (full.length >= 2) {
      const one = blockStored(run.prompt, 1)!
      expect(one.newBlocks).toBe(full.length - 1)
      expect(one.event.parent_hash).toBe(full[0].hash)
      expect(one.event.block_hashes[0]).toBe(full[1].hash)
    }
    expect(blockStored(run.prompt, full.length)).toBeNull()
  })

  it('без thinking рассуждения нет', () => {
    const r = computeRun({ ...DEFAULT_SETTINGS, thinking: false }, tok)
    const reasoning = r.answer.parser.flatMap((e) => e.out).filter((o) => o.kind === 'reasoning')
    expect(reasoning).toHaveLength(0)
  })
})

describe('Kubernetes: YAML и раскладка подов', () => {
  it('воркеры занимают GPU нод по 8, лишние уходят на следующую ноду', () => {
    const nodes = layout(2, 3, 2, true, false)
    const gpuNodes = nodes.filter((n) => n.gpu)
    expect(gpuNodes).toHaveLength(2)
    expect(gpuNodes[0].slots.every((s) => s !== null)).toBe(true)
    const used = gpuNodes.flatMap((n) => n.slots).filter((s) => s !== null).length
    expect(used).toBe((2 + 3) * 2)
    expect(gpuNodes[0].slots).toHaveLength(GPUS_PER_NODE)
    // DaemonSet есть на каждой GPU-ноде, Planner на CPU-ноде
    expect(gpuNodes.every((n) => n.pods.some((p) => p.comp === 'daemon'))).toBe(true)
    expect(nodes[0].pods.some((p) => p.comp === 'planner')).toBe(true)
  })

  it('YAML отражает число реплик и GPU, строки компонентов можно подсветить', () => {
    const { code, lines } = buildYaml(3, 4, 1, false)
    expect(code).toContain('kind: DynamoGraphDeployment')
    expect(code).toContain('replicas: 3')
    expect(code).toContain('replicas: 4')
    expect(code).toContain('nvidia.com/gpu: "1"')
    expect(code).not.toContain('type: planner')
    expect(lines.decode.length).toBeGreaterThan(3)
    expect(code.split('\n')[lines.frontend[0] - 1]).toContain('name: frontend')
  })
})
