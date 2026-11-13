import { describe, expect, it } from 'vitest'
import { getGpu } from './data/gpus'
import { getModel } from './data/models'
import { blockHashes } from './kv'
import {
  applyRoute,
  cacheBlocks,
  makeTenantWorkload,
  makeWorkers,
  overlapByTier,
  releaseRoute,
  routeKV,
  routeRandom,
  routeRoundRobin,
  selectionProbs,
  simulateRouting,
  type RouterWorker,
} from './router'

const range = (n: number, from = 0) => Array.from({ length: n }, (_, i) => from + i)

/** Воркер с заданными закэшированными хэшами и нагрузкой. */
function worker(id: string, cached: string[], activeBlocks: number, extra: Partial<RouterWorker> = {}): RouterWorker {
  return { id, name: id, cachedBlocks: new Set(cached), activeBlocks, ...extra }
}

describe('пример из документации Dynamo: 3 воркера, стоимости 18 / 10 / 11 (notes-dynamo.md §3.3)', () => {
  // 10 полных блоков по 16 токенов: raw prefill = 10 у всех.
  const blocks = blockHashes(range(160, 100), 16)
  const h = blocks.map((b) => b.hash)
  // potential decode = active + новые блоки: 10, 5, 9. В примере доков «новые» меньше 10, значит,
  // закэшированный префикс держат активные запросы воркера (Dynamo дедуплицирует только с активными).
  const held = (n: number) => ({ activeBlockHashes: new Set(h.slice(0, n)) })
  const workers = [worker('W1', h.slice(0, 2), 2, held(2)), worker('W2', h.slice(0, 5), 0, held(5)), worker('W3', h.slice(0, 8), 7, held(8))]

  it('стоимости и выбор W2', () => {
    const d = routeKV(blocks, workers, { overlapCredit: 1, temperature: 0, activeRequestsWeight: 0 })
    expect(d.scores.map((s) => s.overlapBlocks)).toEqual([2, 5, 8])
    expect(d.scores.map((s) => s.adjustedPrefillBlocks)).toEqual([8, 5, 2])
    expect(d.scores.map((s) => s.decodeBlocks)).toEqual([10, 5, 9])
    expect(d.scores.map((s) => s.cost)).toEqual([18, 10, 11])
    expect(d.chosen).toBe('W2')
    expect(d.scores.map((s) => s.prob)).toEqual([0, 1, 0])
    expect(d.requestBlocks).toBe(10)
    expect(d.mode).toBe('kv')
    expect(d.explanation).toMatch(/Выбран W2/)
    expect(d.explanation).toMatch(/W3 держит больше всего кэша/)
  })
  it('softmax по нормированным стоимостям при T = 0.5 и T = 1', () => {
    const p05 = selectionProbs([18, 10, 11], 0.5)
    expect(p05[0]).toBeCloseTo(0.0707, 4)
    expect(p05[1]).toBeCloseTo(0.5224, 4)
    expect(p05[2]).toBeCloseTo(0.4069, 4)
    const p1 = routeKV(blocks, workers, { overlapCredit: 1, temperature: 1, seed: 1 }).scores.map((s) => s.prob)
    expect(p1[0]).toBeCloseTo(0.1635, 4)
    expect(p1[1]).toBeCloseTo(0.4444, 4)
    expect(p1[2]).toBeCloseTo(0.3922, 4)
    expect(selectionProbs([5, 5, 5], 1)).toEqual([1 / 3, 1 / 3, 1 / 3])
  })
  it('при T > 0 выбор детерминирован по seed и в объяснении есть температура', () => {
    const a = routeKV(blocks, workers, { overlapCredit: 1, temperature: 1, seed: 42 })
    const b = routeKV(blocks, workers, { overlapCredit: 1, temperature: 1, seed: 42 })
    expect(a.chosen).toBe(b.chosen)
    expect(a.explanation).toMatch(/Температура 1/)
    // по многим seed частоты близки к вероятностям
    const counts: Record<string, number> = { W1: 0, W2: 0, W3: 0 }
    for (let s = 0; s < 3000; s++) counts[routeKV(blocks, workers, { overlapCredit: 1, temperature: 1, seed: s }).chosen]++
    expect(counts.W2 / 3000).toBeCloseTo(0.444, 1)
    expect(counts.W1 / 3000).toBeCloseTo(0.164, 1)
  })
  it('при T > 0 выпал не самый дешёвый: объяснение называет самого дешёвого и не сравнивает с ним как с более дорогим', () => {
    // seed 7 (как в главе «Роутинг» у первого запроса): первое случайное число меньше 0,1635, выпадает W1 со стоимостью 18,
    // хотя W2 стоит 10, а W3 с самым большим кэшем 11.
    const d = routeKV(blocks, workers, { overlapCredit: 1, temperature: 1, seed: 7 })
    expect(d.chosen).toBe('W1')
    expect(d.explanation).not.toMatch(/не меньше/)
    expect(d.explanation).not.toMatch(/держит больше всего кэша/)
    expect(d.explanation).toContain('W1 не самый дешёвый: W2 стоит меньше всех (10, шанс 44\u00a0%).')
    expect(d.explanation).toContain('Температура 1: воркер выбирается случайно по softmax от стоимостей, шанс выбрать W1 был 16\u00a0%.')
    // T = 0,05 и нагрузка W2 = 2: стоимости 18 / 12 / 11, выпадает W2 (шанс 5,4 %), а дешевле всех W3 с самым большим кэшем.
    const ws2 = [workers[0], worker('W2', h.slice(0, 5), 2, held(5)), workers[2]]
    const e = routeKV(blocks, ws2, { overlapCredit: 1, temperature: 0.05, seed: 7 })
    expect(e.scores.map((s) => s.cost)).toEqual([18, 12, 11])
    expect(e.chosen).toBe('W2')
    expect(e.explanation).not.toMatch(/нагрузка decode выше/)
    expect(e.explanation).toContain('W2 не самый дешёвый: W3 стоит меньше всех (11, шанс 95\u00a0%).')
    expect(e.explanation).toContain('шанс выбрать W2 был 5,4\u00a0%')
  })
  it('при T > 0 выпал самый дешёвый: сравнение с воркером, у которого больше кэша, остаётся', () => {
    // первое число 0,3 попадает в долю W2 (0,1635…0,6079)
    const d = routeKV(blocks, workers, { overlapCredit: 1, temperature: 1, rng: () => 0.3 })
    expect(d.chosen).toBe('W2')
    expect(d.explanation).toContain('W3 держит больше всего кэша (8 бл.), но его стоимость 11: нагрузка decode выше')
    expect(d.explanation).not.toMatch(/не самый дешёвый/)
    expect(d.explanation).toContain('шанс выбрать W2 был 44\u00a0%')
  })
  it('шанс в объяснении округлён как подписи столбиков: крошечный шанс не превращается в 0 %', () => {
    // T = 0,1: шанс W1 около 0,0035 %, число 0,00001 выбирает именно его
    const d = routeKV(blocks, workers, { overlapCredit: 1, temperature: 0.1, rng: () => 0.00001 })
    expect(d.chosen).toBe('W1')
    expect(d.explanation).toContain('шанс выбрать W1 был <0,1\u00a0%')
    expect(d.explanation).not.toMatch(/был 0\s%/)
  })
})

describe('объяснение при ничьей и при нескольких самых дешёвых', () => {
  const blocks = blockHashes(range(160, 5000), 16) // 10 полных блоков
  const h = blocks.map((b) => b.hash)
  it('T > 0: несколько самых дешёвых перечисляются в именительном падеже', () => {
    const ws = [worker('Воркер 1', [], 0), worker('Воркер 2', [], 0), worker('Воркер 3', [], 10)]
    const d = routeKV(blocks, ws, { temperature: 1, rng: () => 0.99 })
    expect(d.scores.map((s) => s.cost)).toEqual([20, 20, 30])
    expect(d.chosen).toBe('Воркер 3')
    expect(d.explanation).toContain('Воркер 3 не самый дешёвый: Воркер 1 и Воркер 2 стоят меньше всех (20, шанс у каждого 42\u00a0%).')
    expect(d.explanation).not.toMatch(/у Воркер \d/)
  })
  it('T = 0, ничья с воркером, у которого больше кэша: выбор объясняет фраза про одинаковую стоимость', () => {
    // W1: prefill 10 + decode 10; W2: кэш 5 блоков, prefill 5 + decode 15
    const ws = [worker('W1', [], 0), worker('W2', h.slice(0, 5), 5)]
    const d = routeKV(blocks, ws, { temperature: 0 })
    expect(d.scores.map((s) => s.cost)).toEqual([20, 20])
    expect(d.chosen).toBe('W1')
    expect(d.explanation).not.toMatch(/держит больше всего кэша/)
    expect(d.explanation).toMatch(/одинаковой стоимостью/)
  })
})

describe('стоимость на реальных токенах: горячий, но занятый воркер проигрывает (notes-dynamo.md §3.4)', () => {
  const blocks = blockHashes(range(2048, 7), 16) // 128 блоков
  const h = blocks.map((b) => b.hash)
  const prefix = h.slice(0, 96)
  const ws = [
    worker('W1', prefix, 400, { queuedPrefillTokens: 4096, activeBlockHashes: new Set(prefix) }),
    worker('W2', prefix, 500, { activeBlockHashes: new Set(prefix) }),
    worker('W3', [], 420, { activeBlockHashes: new Set() }),
  ]
  it('720 / 564 / 676 → W2', () => {
    const d = routeKV(blocks, ws, { overlapCredit: 1, temperature: 0 })
    expect(d.scores.map((s) => s.newBlocks)).toEqual([32, 32, 128])
    expect(d.scores.map((s) => s.rawPrefillBlocks)).toEqual([384, 128, 128])
    expect(d.scores.map((s) => s.cost)).toEqual([720, 564, 676])
    expect(d.chosen).toBe('W2')
  })
  it('вариации: credit = 0 и prefill_load_scale = 2', () => {
    expect(routeKV(blocks, ws, { overlapCredit: 0, temperature: 0 }).scores.map((s) => s.cost)).toEqual([816, 660, 676])
    expect(routeKV(blocks, ws, { overlapCredit: 1, temperature: 0, prefillLoadScale: 2 }).scores.map((s) => s.cost)).toEqual([1008, 596, 804])
  })
  it('softmax T = 0.5 и T = 1', () => {
    const p05 = selectionProbs([720, 564, 676], 0.5)
    expect(p05.map((x) => +x.toFixed(4))).toEqual([0.0986, 0.7282, 0.1732])
    const p1 = selectionProbs([720, 564, 676], 1)
    expect(p1.map((x) => +x.toFixed(4))).toEqual([0.1983, 0.5389, 0.2628])
  })
})

describe('кредиты нижних уровней (notes-dynamo.md §3.5а)', () => {
  it('весь префикс в RAM хоста выгоднее половины в GPU: 396 против 380', () => {
    const blocks = blockHashes(range(1024, 3), 16) // 64 блока
    const h = blocks.map((b) => b.hash)
    const wa = worker('Wa', h.slice(0, 32), 300, { activeBlockHashes: new Set() })
    const wb = worker('Wb', [], 300, { hostBlocks: new Set(h), activeBlockHashes: new Set() })
    const d = routeKV(blocks, [wa, wb], { overlapCredit: 1, temperature: 0 })
    expect(d.scores[1].hostOverlapBlocks).toBe(64)
    expect(d.scores[1].credit).toBe(48)
    expect(d.scores.map((s) => s.cost)).toEqual([396, 380])
    expect(d.chosen).toBe('Wb')
  })
})

describe('legacy-формула ≤ v1.1 (notes-numbers.md §3.21)', () => {
  const blocks = blockHashes(range(1024), 16)
  const h = blocks.map((b) => b.hash)
  const ws = [worker('A', h.slice(0, 48), 40), worker('B', [], 10)]
  it('вес 1.0: A = 56, B = 74 → A', () => {
    const d = routeKV(blocks, ws, { overlapCredit: 1, temperature: 0, formula: 'legacy' })
    expect(d.scores.map((s) => s.cost)).toEqual([56, 74])
    expect(d.chosen).toBe('A')
  })
  it('вес 0.5: A = 48, B = 42 → B', () => {
    const d = routeKV(blocks, ws, { overlapCredit: 1, temperature: 0, formula: 'legacy', legacyOverlapWeight: 0.5 })
    expect(d.scores.map((s) => s.cost)).toEqual([48, 42])
    expect(d.chosen).toBe('B')
  })
})

describe('перекрытие = самый длинный префикс цепочки (notes-dynamo.md §3.2)', () => {
  it('W1: 6, W2: 4, W3: 0', () => {
    const S = range(64, 1)
    const A = range(40, 500)
    const Bq = range(32, 900)
    const T = range(64, 2000)
    const full = (ids: number[]) => blockHashes(ids, 16).filter((b) => b.full).map((b) => b.hash)
    const w1 = worker('W1', full([...S, ...A]), 0)
    const w2 = worker('W2', full([...S, ...Bq]), 0)
    const w3 = worker('W3', full(T), 0)
    const req = blockHashes([...S, ...A, ...range(56, 3000)], 16)
    expect(req.filter((b) => b.full)).toHaveLength(10)
    expect([w1, w2, w3].map((w) => overlapByTier(req, w).device)).toEqual([6, 4, 0])
  })
})

describe('round-robin, random, applyRoute, releaseRoute', () => {
  const blocks = blockHashes(range(100), 16) // 6 полных + хвост 4
  it('ничья: первый по порядку, после назначения следующий запрос уходит другому', () => {
    let ws = makeWorkers(2)
    const d1 = routeKV(blocks, ws)
    expect(d1.chosen).toBe('worker-0')
    expect(d1.explanation).toMatch(/одинаковой стоимостью/)
    ws = applyRoute(ws, d1, blocks)
    expect(ws[0].cachedBlocks.size).toBe(6)
    expect(ws[0].activeBlocks).toBe(6) // floor(100 / 16): Dynamo считает только хэшируемые (полные) блоки
    expect(ws[0].queuedPrefillTokens).toBe(100)
    expect(ws[0].activeRequests).toBe(1)
    const other = blockHashes(range(100, 5000), 16)
    expect(routeKV(other, ws).chosen).toBe('worker-1')
    // тот же префикс: идёт туда, где он лежит
    const again = routeKV(blocks, releaseRoute(ws, d1, 'prefill'))
    expect(again.chosen).toBe('worker-0')
    expect(again.scores[0].cachedTokens).toBe(96)
  })
  it('cacheBlocks: тёплый кэш без нагрузки; роутер идёт к тёплому воркеру', () => {
    const ws = makeWorkers(3)
    const warm = [ws[0], cacheBlocks(ws[1], blocks), ws[2]]
    expect(ws[1].cachedBlocks.size).toBe(0)
    expect(warm[1].cachedBlocks.size).toBe(6)
    expect(warm[1].activeBlocks).toBe(0)
    const d = routeKV(blocks, warm)
    expect(d.chosen).toBe('worker-1')
    expect(d.explanation).toMatch(/6 из 7\sблоков/) // между числом и словом неразрывный пробел
  })
  it('applyRoute не мутирует исходные воркеры', () => {
    const ws = makeWorkers(2)
    const d = routeKV(blocks, ws)
    const next = applyRoute(ws, d, blocks)
    expect(ws[0].cachedBlocks.size).toBe(0)
    expect(next[1]).toBe(ws[1])
  })
  it('releaseRoute снимает prefill-токены, затем активные блоки', () => {
    let ws = makeWorkers(1)
    const d = routeKV(blocks, ws)
    ws = applyRoute(ws, d, blocks)
    ws = releaseRoute(ws, d, 'prefill')
    expect(ws[0].queuedPrefillTokens).toBe(0)
    expect(ws[0].activeBlocks).toBe(6)
    ws = releaseRoute(ws, d, 'finish')
    expect(ws[0].activeBlocks).toBe(0)
    expect(ws[0].activeRequests).toBe(0)
    expect(ws[0].cachedBlocks.size).toBe(6) // кэш остаётся
  })
  it('ёмкость кэша: старые блоки вытесняются в host', () => {
    let ws = makeWorkers(1, { capacityBlocks: 8, withHostTier: true })
    const a = blockHashes(range(96), 16)
    const b = blockHashes(range(96, 1000), 16)
    ws = applyRoute(ws, routeKV(a, ws), a)
    ws = applyRoute(ws, routeKV(b, ws), b)
    expect(ws[0].cachedBlocks.size).toBe(8)
    expect(ws[0].hostBlocks!.size).toBe(4)
    // часть префикса a теперь в host: кредит 0.75 за блок
    const d = routeKV(a, ws)
    expect(d.scores[0].hostOverlapBlocks).toBeGreaterThan(0)
  })
  it('round-robin идёт по кругу; random детерминирован по seed', () => {
    const ws = makeWorkers(3)
    expect([0, 1, 2, 3, 4].map((k) => routeRoundRobin(blocks, ws, k).chosen)).toEqual(['worker-0', 'worker-1', 'worker-2', 'worker-0', 'worker-1'])
    const r = routeRoundRobin(blocks, ws, 0)
    expect(r.mode).toBe('round-robin')
    expect(r.scores.map((s) => s.prob)).toEqual([1, 0, 0])
    expect(r.explanation).toMatch(/Round-robin/)
    const x = routeRandom(blocks, ws, 123)
    expect(routeRandom(blocks, ws, 123).chosen).toBe(x.chosen)
    expect(x.scores.every((s) => Math.abs(s.prob - 1 / 3) < 1e-12)).toBe(true)
    const seen = new Set(Array.from({ length: 50 }, (_, s) => routeRandom(blocks, ws, s).chosen))
    expect(seen.size).toBe(3)
  })
})

describe('simulateRouting: KV-aware против round-robin на общих системных промптах', () => {
  const model = getModel('qwen3-8b')
  const gpu = getGpu('h100-sxm')
  const reqs = makeTenantWorkload({ tenants: 8, systemTokens: 1024, userTokens: [32, 256], requests: 160, ratePerSec: 8, outputTokens: 120, seed: 3 })
  it('генератор детерминирован и даёт общие префиксы', () => {
    const again = makeTenantWorkload({ tenants: 8, systemTokens: 1024, userTokens: [32, 256], requests: 160, ratePerSec: 8, outputTokens: 120, seed: 3 })
    expect(again.map((r) => r.blocks[0].hash)).toEqual(reqs.map((r) => r.blocks[0].hash))
    expect(new Set(reqs.map((r) => r.blocks[0].hash)).size).toBe(8)
    expect(reqs[reqs.length - 1].arrivalMs).toBeGreaterThan(reqs[0].arrivalMs)
  })
  it('KV-routing даёт больший hit rate и меньший TTFT; баланс разумный', () => {
    const res = simulateRouting(reqs, makeWorkers(4), { model, gpu, capacityBlocks: 200, seed: 1 })
    const kv = res.byMode.kv!
    const rr = res.byMode['round-robin']!
    const rnd = res.byMode.random!
    expect(kv.hitRate).toBeGreaterThan(rr.hitRate + 0.2)
    expect(kv.hitRate).toBeGreaterThan(rnd.hitRate + 0.2)
    expect(kv.meanTtftMs).toBeLessThan(rr.meanTtftMs)
    expect(kv.prefillGpuMs).toBeLessThan(rr.prefillGpuMs)
    expect(kv.loadImbalance).toBeLessThan(2)
    expect(rr.loadImbalance).toBeCloseTo(1, 6)
    expect(kv.assignments).toHaveLength(160)
    expect(kv.perWorker.reduce((a, w) => a + w.requests, 0)).toBe(160)
    expect(res.strategies.map((s) => s.mode)).toEqual(['kv', 'round-robin', 'random'])
  })
})

describe('объяснение роутера читается по-русски', () => {
  const blocks = blockHashes(range(2048, 7), 16) // 128 блоков
  const h = blocks.map((b) => b.hash)
  const ws = [
    worker('Воркер 1', h.slice(0, 96), 400, { queuedPrefillTokens: 4096, activeBlockHashes: new Set(h.slice(0, 96)) }),
    worker('Воркер 2', h.slice(0, 40), 380, { activeBlockHashes: new Set(h.slice(0, 40)) }),
  ]
  it('имя воркера в именительном падеже, разряды неразрывным пробелом, слова согласованы с числом', () => {
    const d = routeKV(blocks, ws, { temperature: 0 })
    expect(d.chosen).toBe('Воркер 2')
    expect(d.explanation).toContain('Воркер 1 держит больше всего кэша (96 бл.)')
    expect(d.explanation).toContain('в очереди prefill уже 4\u00a0096\u00a0токенов')
    expect(d.explanation).toContain('40 из 128\u00a0блоков запроса')
    expect(d.explanation).not.toMatch(/у Воркер \d/)
    const rr = routeRoundRobin(blocks, ws, 0)
    expect(rr.explanation).toContain('по кругу выбран Воркер 1')
    expect(rr.explanation).toContain('в его кэше 96\u00a0блоков запроса')
  })
})
