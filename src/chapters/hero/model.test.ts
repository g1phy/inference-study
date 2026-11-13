import { describe, expect, it } from 'vitest'
import { loadQwen3ForTests } from '@/engine/tokenizer/testing'
import { buildRun } from '@/engine/flow'
import { scenarioPatch } from '@/engine/scenarios'
import type { FlowSettings, StageId } from '@/engine/types'
import {
  brokenAxis,
  clientView,
  excerpt,
  firstDecodeStep,
  firstStepOf,
  lanesUpTo,
  neighborSteps,
  neighborStepsText,
  nodeStatus,
  packetMove,
  prettyJson,
  samePlaybackShape,
  sseSummary,
  stepAtTime,
  stepDurations,
  ticks,
  tokenAtStep,
  tokenStepMs,
  tokensGen,
} from './model'

const tk = loadQwen3ForTests()
const DEFAULTS: FlowSettings = {
  scenarioId: 'thinking',
  systemText: 'Ты полезный ассистент.',
  userText: 'Объясни простыми словами, что такое KV cache и зачем он нужен.',
  thinking: true,
  tools: false,
  json: false,
  deployment: 'agg',
  cacheWarm: false,
  modelId: 'qwen3-8b',
  gpuId: 'h100-sxm',
  temperature: 0.7,
}
const LABELS = { json: 'JSON', tokens: (n: number) => `${n} ток.`, kv: () => 'KV', meta: 'мета' }

describe('шаги плеера', () => {
  const run = buildRun(DEFAULTS, tk)
  const pb = run.playback!

  it('длительности: стадии медленно, первые токены медленно, остальные быстро', () => {
    const d = stepDurations(run)
    expect(d).toHaveLength(pb.length)
    const firstToken = pb.findIndex((p) => p.kind === 'token')
    expect(d[0]).toBeGreaterThan(d[firstToken + 5])
    expect(d[firstToken]).toBeGreaterThan(d[firstToken + 5])
    expect(tokenStepMs(10)).toBe(220)
    expect(tokenStepMs(10_000)).toBe(50)
  })

  it('первый шаг стадии и статусы узлов', () => {
    expect(firstStepOf(pb, 'client')).toBe(0)
    expect(firstStepOf(pb, 'transfer')).toBe(-1)
    const tok = firstStepOf(pb, 'decode')
    expect(pb[tok].kind).toBe('token')
    expect(firstStepOf(pb, 'stream')).toBe(tok)
    expect(nodeStatus(pb, 0, 'client')).toBe('active')
    expect(nodeStatus(pb, 0, 'route')).toBe('pending')
    expect(nodeStatus(pb, 3, 'frontend')).toBe('done')
    expect(nodeStatus(pb, tok + 1, 'decode')).toBe('active')
    expect(nodeStatus(pb, tok + 1, 'stream')).toBe('active')
    expect(nodeStatus(pb, tok + 1, 'prefill')).toBe('done')
    const last = pb.length - 1
    expect(nodeStatus(pb, last, 'stream')).toBe('active')
    expect(nodeStatus(pb, last, 'decode')).toBe('done')
  })

  it('токен на шаге', () => {
    expect(tokenAtStep(pb, 0, run.decode.length)).toBe(-1)
    const tok = firstStepOf(pb, 'decode')
    expect(tokenAtStep(pb, tok, run.decode.length)).toBe(0)
    expect(tokenAtStep(pb, pb.length - 1, run.decode.length)).toBe(run.decode.length - 1)
  })

  it('пакет: в disagg второй route везёт метаданные назад, transfer везёт KV', () => {
    const dis = buildRun({ ...DEFAULTS, deployment: 'disagg' }, tk)
    const p = dis.playback!
    const routes = p.map((s, i) => (s.kind === 'stage' && s.stage === 'route' ? i : -1)).filter((i) => i >= 0)
    expect(routes).toHaveLength(2)
    expect(packetMove(dis, routes[0], LABELS)?.kind).toBe('request')
    expect(packetMove(dis, routes[1], LABELS)).toMatchObject({ from: 'prefill', to: 'route', kind: 'meta' })
    const tr = firstStepOf(p, 'transfer')
    expect(packetMove(dis, tr, LABELS)).toMatchObject({ from: 'prefill', to: 'decode', kind: 'kv' })
    expect(packetMove(dis, firstStepOf(p, 'decode'), LABELS)).toBeNull()
  })

  it('шаг по времени', () => {
    expect(stepAtTime(pb, 0)).toBe(0)
    expect(stepAtTime(pb, run.metrics.ttftMs)).toBe(firstStepOf(pb, 'decode'))
    expect(stepAtTime(pb, 1e9)).toBe(pb.length - 1)
  })
})

describe('что видит клиент', () => {
  const cases: Array<[string, Partial<FlowSettings>]> = [
    ['thinking', {}],
    ['simple', scenarioPatch('simple')],
    ['tool', scenarioPatch('tool')],
    ['json', scenarioPatch('json')],
    ['custom', { scenarioId: 'custom', userText: 'Как работает continuous batching?', thinking: false }],
  ]
  for (const [name, patch] of cases) {
    it(`${name}: склейка всех чанков равна итогу прогона`, () => {
      const run = buildRun({ ...DEFAULTS, ...patch }, tk)
      const v = clientView(run.sse, run.sse.length)
      expect(v.done).toBe(true)
      expect(v.reasoning).toBe(run.output!.reasoning)
      expect(v.content).toBe(run.output!.content)
      expect(v.finishReason).toBe(run.metrics.finishReason)
      expect(v.usage?.completion_tokens).toBe(run.metrics.completionTokens)
      expect(v.tools.map((t) => t.name)).toEqual(run.output!.toolCalls.map((t) => t.name))
      for (const [i, t] of v.tools.entries()) expect(JSON.parse(t.args)).toEqual(run.output!.toolCalls[i].arguments)
      // дорожки парсера к последнему токену: reasoning и content совпадают с клиентом
      const lanes = lanesUpTo(run.parser, run.decode.length)
      expect(lanes.reasoning).toBe(run.output!.reasoning)
      expect(lanes.content).toBe(run.output!.content)
      // короткие подписи чанков не пустые
      for (const ch of run.sse) expect(sseSummary(ch).length).toBeGreaterThan(3)
    })
  }

  it('частичная склейка растёт монотонно', () => {
    const run = buildRun(DEFAULTS, tk)
    let prev = 0
    for (let n = 0; n <= run.sse.length; n += 17) {
      const v = clientView(run.sse, n)
      const len = v.reasoning.length + v.content.length
      expect(len).toBeGreaterThanOrEqual(prev)
      prev = len
    }
    expect(clientView(run.sse, 0).chunks).toBe(0)
  })
})

describe('разрывная ось', () => {
  it('монотонна, TTFT на стыке, обратное преобразование', () => {
    const ax = brokenAxis(18, 1708, 100, 300, 12, 500)
    expect(ax.x(0)).toBe(100)
    expect(ax.x(18)).toBe(400)
    expect(ax.x(18.0001)).toBeGreaterThanOrEqual(412)
    expect(ax.x(1708)).toBe(912)
    let prev = -Infinity
    for (const t of [0, 1, 5, 12, 17.9, 18, 25, 400, 1000, 1708]) {
      const x = ax.x(t)
      expect(x).toBeGreaterThanOrEqual(prev)
      prev = x
      expect(ax.t(x)).toBeCloseTo(t, 3)
    }
  })

  it('круглые деления', () => {
    expect(ticks(0, 18, 4)).toEqual([0, 5, 10, 15])
    expect(ticks(18, 1708, 4)).toEqual([500, 1000, 1500])
    expect(ticks(5, 5)).toEqual([5])
  })
})

describe('prettyJson', () => {
  it('массивы примитивов в строку, объекты с отступами, разбирается обратно', () => {
    const v = { a: [1, 2, 3], b: { c: 'x', d: [{ e: 1 }] }, f: [] as number[], g: null }
    const s = prettyJson(v)
    expect(s).toContain('"a": [1, 2, 3]')
    expect(s).toContain('"f": []')
    expect(JSON.parse(s)).toEqual(v)
  })
})

describe('подписи', () => {
  it('отрывок по границе слова', () => {
    expect(excerpt('Объясни простыми словами, что такое KV cache и зачем он нужен.', 30)).toBe('Объясни простыми словами, что…')
    expect(excerpt('коротко', 30)).toBe('коротко')
  })

  it('все стадии disagg-прогона достижимы плеером', () => {
    const run = buildRun({ ...DEFAULTS, deployment: 'disagg', cacheWarm: true }, tk)
    const stages: StageId[] = ['client', 'frontend', 'preprocess', 'route', 'prefill', 'transfer', 'decode', 'postprocess', 'stream']
    for (const s of stages) expect(firstStepOf(run.playback!, s), s).toBeGreaterThanOrEqual(0)
  })
})

describe('второй круг ревью', () => {
  it('X-sim-r2-6: ссылка на decode ведёт на настоящий шаг decode или на первый шаг под маской', () => {
    const agg = buildRun(DEFAULTS, tk)
    const ia = firstDecodeStep(agg)
    // agg: первый токен выбран в конце prefill, поэтому второй
    expect(agg.playback![ia]).toMatchObject({ kind: 'token', token: 1 })
    expect(agg.decode[1].phase).toBe('decode')
    const dis = buildRun({ ...DEFAULTS, deployment: 'disagg' }, tk)
    expect(dis.playback![firstDecodeStep(dis)]).toMatchObject({ kind: 'token', token: 0 })
    // JSON: первый шаг грамматики, там маска режет болтовню
    const json = buildRun({ ...DEFAULTS, ...scenarioPatch('json') }, tk)
    const ij = firstDecodeStep(json)
    const k = json.playback![ij].token!
    expect(json.decode[k].masked).toBe(true)
    expect(json.decode.slice(0, k).some((d) => d.masked)).toBe(false)
    const jsonThink = buildRun({ ...DEFAULTS, ...scenarioPatch('json'), thinking: true }, tk)
    const kt = jsonThink.playback![firstDecodeStep(jsonThink)].token!
    expect(kt).toBeGreaterThan(0)
    expect(jsonThink.decode[kt].masked).toBe(true)
  })

  it('X-sim-r2-5: temperature, модель и GPU не меняют шаги плеера, деплой и текст меняют', () => {
    const a = buildRun(DEFAULTS, tk)
    const pb = a.playback!
    expect(samePlaybackShape(pb, buildRun({ ...DEFAULTS, temperature: 1.5 }, tk).playback!)).toBe(true)
    expect(samePlaybackShape(pb, buildRun({ ...DEFAULTS, gpuId: 'h200' }, tk).playback!)).toBe(true)
    expect(samePlaybackShape(pb, buildRun({ ...DEFAULTS, modelId: 'qwen3-32b' }, tk).playback!)).toBe(true)
    expect(samePlaybackShape(pb, buildRun({ ...DEFAULTS, cacheWarm: true }, tk).playback!)).toBe(true)
    expect(samePlaybackShape(pb, buildRun({ ...DEFAULTS, deployment: 'disagg' }, tk).playback!)).toBe(false)
    expect(samePlaybackShape(pb, buildRun({ ...DEFAULTS, ...scenarioPatch('simple') }, tk).playback!)).toBe(false)
  })

  it('X-sim-r2-4: обычный шаг рядом со средним, только если сосед попал в шаги decode', () => {
    const agg = buildRun(DEFAULTS, tk)
    const nb = neighborSteps(agg)!
    expect(nb.count).toBe(1)
    expect(nb.plainMs).toBeLessThan(agg.metrics.itlMs)
    expect(nb.stepMs).toBeGreaterThan(10 * nb.plainMs)
    expect(neighborStepsText(nb)).toMatch(/^шаг с prefill соседа \d+/)
    expect(neighborStepsText({ stepMs: 207, count: 3 })).toMatch(/^3\s+шага с prefill соседа, до 207/)
    expect(neighborSteps(buildRun({ ...DEFAULTS, deployment: 'disagg' }, tk))).toBeNull()
    expect(neighborSteps(buildRun({ ...DEFAULTS, ...scenarioPatch('simple') }, tk))).toBeNull()
  })

  it('токены в родительном падеже', () => {
    expect(tokensGen(43)).toMatch(/^43\s+токенов$/)
    expect(tokensGen(21)).toMatch(/^21\s+токена$/)
    expect(tokensGen(1167)).toMatch(/^1\s167\s+токенов$/)
    expect(tokensGen(407904)).toMatch(/токенов$/)
  })
})
