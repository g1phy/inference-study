/**
 * Независимая проверка router.ts (верификатор A2): все разобранные примеры формулы Dynamo
 * (notes-dynamo.md §1.8, §3.2–3.5; notes-kvcache-tiers.md §3.9; notes-numbers.md §3.21), точная
 * семантика additional_active_blocks из кода Dynamo (lib/router/slot-tracker/src/prompt_registry.rs:
 * `token_sequence.len() − overlap_depth` по trie АКТИВНЫХ запросов воркера; хэшируются только полные
 * блоки, lib/kv-hashing/src/routing.rs) и свойства перекрытия по цепочке хэшей.
 */
import { describe, expect, it } from 'vitest'
import { getGpu } from './data/gpus'
import { getModel } from './data/models'
import { blockHashes, cachedPrefixBlocks, type BlockInfoExt } from './kv'
import { mulberry32, randInt } from './rng'
import {
  applyRoute,
  cacheBlocks,
  makeTenantWorkload,
  makeWorkers,
  overlapByTier,
  releaseRoute,
  routeKV,
  scoreWorker,
  selectionProbs,
  simulateRouting,
  type RouterWorker,
} from './router'

const range = (n: number, from = 0) => Array.from({ length: n }, (_, i) => from + i)
const hashes = (bs: readonly BlockInfoExt[]) => bs.filter((b) => b.full).map((b) => b.hash)
function worker(id: string, cached: string[], activeBlocks: number, extra: Partial<RouterWorker> = {}): RouterWorker {
  return { id, name: id, cachedBlocks: new Set(cached), activeBlocks, ...extra }
}

describe('§3.3 пример документации: softmax по min-max нормированным стоимостям, вся таблица', () => {
  it.each([
    [0.1, [0.0, 0.7773, 0.2227]],
    [0.25, [0.0113, 0.6154, 0.3733]],
    [0.5, [0.0707, 0.5224, 0.4069]],
    [1.0, [0.1635, 0.4444, 0.3922]],
    [2.0, [0.2382, 0.3928, 0.369]],
  ])('T = %f', (T, want) => {
    const p = selectionProbs([18, 10, 11], T)
    p.forEach((x, i) => expect(x).toBeCloseTo(want[i], 4))
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })
  it('T = 0: one-hot у минимума; худший при T = 1 весит e^−1 относительно лучшего', () => {
    expect(selectionProbs([18, 10, 11], 0)).toEqual([0, 1, 0])
    const p = selectionProbs([18, 10, 11], 1)
    expect(p[0] / p[1]).toBeCloseTo(Math.exp(-1), 12)
    // сдвиг и растяжение стоимостей не меняют вероятности (нормировка min-max)
    expect(selectionProbs([118, 110, 111], 1).map((x) => +x.toFixed(10))).toEqual(p.map((x) => +x.toFixed(10)))
    expect(selectionProbs([36, 20, 22], 1).map((x) => +x.toFixed(10))).toEqual(p.map((x) => +x.toFixed(10)))
  })
  it('стоимости 18 / 10 / 11 на настоящих блоках: «potential decode» доков = активные + новые для активных', () => {
    // В примере доков potential decode (10 / 5 / 9) меньше блоков запроса (10): значит, закэшированные
    // блоки считаются уже занятыми активными запросами воркера. Передаём это явно через activeBlockHashes.
    const blocks = blockHashes(range(160, 100), 16)
    const h = hashes(blocks)
    const ws = [
      worker('W1', h.slice(0, 2), 2, { activeBlockHashes: new Set(h.slice(0, 2)) }),
      worker('W2', h.slice(0, 5), 0, { activeBlockHashes: new Set(h.slice(0, 5)) }),
      worker('W3', h.slice(0, 8), 7, { activeBlockHashes: new Set(h.slice(0, 8)) }),
    ]
    const d = routeKV(blocks, ws, { overlapCredit: 1, temperature: 0 })
    expect(d.scores.map((s) => s.rawPrefillBlocks)).toEqual([10, 10, 10])
    expect(d.scores.map((s) => s.adjustedPrefillBlocks)).toEqual([8, 5, 2])
    expect(d.scores.map((s) => s.decodeBlocks)).toEqual([10, 5, 9])
    expect(d.scores.map((s) => s.cost)).toEqual([18, 10, 11])
    expect(d.chosen).toBe('W2')
  })
})

describe('§3.4 горячий, но занятый воркер проигрывает: все вариации', () => {
  const blocks = blockHashes(range(2048, 7), 16) // 128 блоков
  const h = hashes(blocks)
  const prefix = h.slice(0, 96)
  const ws = [
    worker('W1', prefix, 400, { queuedPrefillTokens: 4096, activeBlockHashes: new Set(prefix) }),
    worker('W2', prefix, 500, { activeBlockHashes: new Set(prefix) }),
    worker('W3', [], 420, { activeBlockHashes: new Set() }),
  ]
  it('таблица: raw 384/128/128, adjusted 288/32/128, potential decode 432/532/548, cost 720/564/676', () => {
    const d = routeKV(blocks, ws, { overlapCredit: 1, temperature: 0 })
    expect(d.scores.map((s) => s.overlapBlocks)).toEqual([96, 96, 0])
    expect(d.scores.map((s) => s.rawPrefillBlocks)).toEqual([384, 128, 128])
    expect(d.scores.map((s) => s.adjustedPrefillBlocks)).toEqual([288, 32, 128])
    expect(d.scores.map((s) => s.newBlocks)).toEqual([32, 32, 128])
    expect(d.scores.map((s) => s.decodeBlocks)).toEqual([432, 532, 548])
    expect(d.scores.map((s) => s.cost)).toEqual([720, 564, 676])
    expect(d.chosen).toBe('W2')
  })
  it('softmax через routeKV (не только selectionProbs): T = 0.5 и T = 1', () => {
    const p05 = routeKV(blocks, ws, { overlapCredit: 1, temperature: 0.5, seed: 3 }).scores.map((s) => +s.prob.toFixed(4))
    expect(p05).toEqual([0.0986, 0.7282, 0.1732])
    const p1 = routeKV(blocks, ws, { overlapCredit: 1, temperature: 1, seed: 3 }).scores.map((s) => +s.prob.toFixed(4))
    expect(p1).toEqual([0.1983, 0.5389, 0.2628])
  })
  it('вес active_requests: w · R добавляется к стоимости', () => {
    const wr = ws.map((w, i) => ({ ...w, activeRequests: [10, 0, 2][i] }))
    const d = routeKV(blocks, wr, { overlapCredit: 1, temperature: 0, activeRequestsWeight: 5 })
    expect(d.scores.map((s) => s.cost)).toEqual([770, 564, 686])
  })
  it('без assume_kv_reuse дедупликации нет: новые = все 128 блоков у всех', () => {
    const d = routeKV(blocks, ws, { overlapCredit: 1, temperature: 0, assumeKvReuse: false })
    expect(d.scores.map((s) => s.newBlocks)).toEqual([128, 128, 128])
    expect(d.scores.map((s) => s.cost)).toEqual([288 + 528, 32 + 628, 128 + 548])
  })
})

describe('кредиты нижних уровней', () => {
  it('§3.5а: весь префикс в RAM (0.75 · 64 = 48) выгоднее половины в GPU (32): 396 против 380', () => {
    const blocks = blockHashes(range(1024, 3), 16)
    const h = hashes(blocks)
    // активных запросов с этим префиксом нет: новые блоки = все 64 у обоих (как в примере)
    const d = routeKV(blocks, [worker('Wa', h.slice(0, 32), 300), worker('Wb', [], 300, { hostBlocks: new Set(h) })], { overlapCredit: 1, temperature: 0 })
    expect(d.scores.map((s) => s.newBlocks)).toEqual([64, 64])
    expect(d.scores.map((s) => s.credit)).toEqual([32, 48])
    expect(d.scores.map((s) => s.cost)).toEqual([396, 380])
    expect(d.chosen).toBe('Wb')
  })
  it('notes-kvcache-tiers.md §3.9: B = 64, 528 блоков, префикс 512 в GPU / RAM / диске / нигде → 844 / 922 / 1078 / 1156', () => {
    const B = 64
    const blocks = blockHashes(range(33792, 11), B)
    expect(blocks.length).toBe(528)
    const pre = hashes(blocks).slice(0, 512)
    const ws = [
      worker('A', pre, 300),
      worker('B', [], 250, { hostBlocks: new Set(pre) }),
      worker('C', [], 150, { diskBlocks: new Set(pre) }),
      worker('D', [], 100),
    ]
    const d = routeKV(blocks, ws, { overlapCredit: 1, temperature: 0 })
    expect(d.scores.map((s) => s.credit)).toEqual([512, 384, 128, 0])
    expect(d.scores.map((s) => s.adjustedPrefillBlocks)).toEqual([16, 144, 400, 528])
    expect(d.scores.map((s) => s.cost)).toEqual([844, 922, 1078, 1156])
    expect(d.chosen).toBe('A')
    // A занят чужим prefill на 400 блоков (25 600 токенов): 928 − 512 = 416 → 1244, выигрывает B
    const busy = [{ ...ws[0], queuedPrefillTokens: 400 * B }, ...ws.slice(1)]
    const d2 = routeKV(blocks, busy, { overlapCredit: 1, temperature: 0 })
    expect(d2.scores[0].cost).toBe(1244)
    expect(d2.chosen).toBe('B')
  })
  it('§3.5б общий пул Mooncake: B = 1, 4 блока в пуле, multiplier 0.5 → adjusted 1.0 у W0 и 2.0 у W1', () => {
    const blocks = blockHashes([11, 12, 13, 14], 1) // блоки A, B, C, D
    const h = hashes(blocks)
    const pool = new Set(h)
    const ws = [worker('W0', h.slice(0, 2), 0), worker('W1', [], 0)]
    const d = routeKV(blocks, ws, { overlapCredit: 1, temperature: 0, sharedPoolBlocks: pool })
    expect(d.scores.map((s) => s.sharedBeyondBlocks)).toEqual([2, 4])
    expect(d.scores.map((s) => s.credit)).toEqual([3, 2]) // 2.0 + 0.5·2 и 0 + 0.5·4
    expect(d.scores.map((s) => s.adjustedPrefillBlocks)).toEqual([1, 2])
    expect(d.chosen).toBe('W0')
    expect(d.explanation).toMatch(/общего пула/)
    // без пула кредита нет; множитель настраивается
    expect(routeKV(blocks, ws, { overlapCredit: 1, temperature: 0 }).scores.map((s) => s.credit)).toEqual([2, 0])
    expect(routeKV(blocks, ws, { overlapCredit: 1, temperature: 0, sharedPoolBlocks: pool, sharedCacheMultiplier: 0 }).scores.map((s) => s.credit)).toEqual([2, 0])
  })
  it('уровни засчитываются подряд от блока 0: пропуск блока обрывает перекрытие на всех уровнях', () => {
    const blocks = blockHashes(range(160), 16)
    const h = hashes(blocks)
    const w = worker('X', h.slice(0, 3), 0, { hostBlocks: new Set(h.slice(4, 8)), diskBlocks: new Set(h.slice(8)) })
    expect(overlapByTier(blocks, w)).toEqual({ device: 3, host: 0, disk: 0 }) // блока 3 нет нигде
    const w2 = worker('Y', h.slice(0, 3), 0, { hostBlocks: new Set(h.slice(3, 8)), diskBlocks: new Set(h.slice(8)) })
    expect(overlapByTier(blocks, w2)).toEqual({ device: 3, host: 5, disk: 2 })
  })
})

describe('§3.21 legacy (≤ v1.1) и равенство формул при дефолтах', () => {
  const blocks = blockHashes(range(1024), 16)
  const h = hashes(blocks)
  it('A = 1.0·16 + 40 = 56, B = 64 + 10 = 74; при весе 0.5: 48 / 42', () => {
    const ws = [worker('A', h.slice(0, 48), 40), worker('B', [], 10)]
    expect(routeKV(blocks, ws, { overlapCredit: 1, temperature: 0, formula: 'legacy' }).scores.map((s) => s.cost)).toEqual([56, 74])
    expect(routeKV(blocks, ws, { overlapCredit: 1, temperature: 0, formula: 'legacy', legacyOverlapWeight: 0.5 }).scores.map((s) => s.cost)).toEqual([48, 42])
  })
  it('без очередей prefill v1.2 = legacy + одинаковая для всех добавка (новые блоки) → тот же выбор', () => {
    const rng = mulberry32(17)
    for (let trial = 0; trial < 50; trial++) {
      const ws = Array.from({ length: 4 }, (_, i) => worker(`w${i}`, h.slice(0, randInt(rng, 0, 64)), randInt(rng, 0, 500)))
      const v12 = routeKV(blocks, ws, { overlapCredit: 1, temperature: 0 })
      const leg = routeKV(blocks, ws, { overlapCredit: 1, temperature: 0, formula: 'legacy' })
      expect(v12.chosen).toBe(leg.chosen)
      v12.scores.forEach((s, i) => expect(s.cost - leg.scores[i].cost).toBe(64))
    }
  })
})

describe('additional_active_blocks: точная семантика Dynamo', () => {
  it('кэш на GPU без активных запросов НЕ уменьшает decode-слагаемое (это «не промахи кэша»)', () => {
    const blocks = blockHashes(range(512), 16) // 32 полных блока
    const warm = cacheBlocks(makeWorkers(1)[0], blocks)
    const s = scoreWorker(blocks, warm)
    expect(s.overlapBlocks).toBe(32)
    expect(s.newBlocks).toBe(32)
    expect(s.decodeBlocks).toBe(32)
    expect(s.adjustedPrefillBlocks).toBe(0) // скидка только на prefill
  })
  it('считаются только полные (хэшируемые) блоки: 100 токенов при B = 16 → 6, хвост из 4 токенов не в счёт', () => {
    const blocks = blockHashes(range(100), 16)
    const s = scoreWorker(blocks, makeWorkers(1)[0])
    expect(s.newBlocks).toBe(6)
    expect(s.rawPrefillBlocks).toBe(100 / 16) // prefill-часть дробная, по токенам
  })
  it('activeBlockHashes ведёт applyRoute/releaseRoute: второй запрос с тем же префиксом дедуплицируется', () => {
    const sys = range(256, 9000) // 16 блоков общего системного промпта
    const r1 = blockHashes([...sys, ...range(64, 1)], 16) // 20 блоков
    const r2 = blockHashes([...sys, ...range(64, 500)], 16) // 20 блоков, первые 16 общие
    let ws = makeWorkers(1)
    const d1 = routeKV(r1, ws)
    expect(d1.scores[0].newBlocks).toBe(20)
    ws = applyRoute(ws, d1, r1)
    expect(ws[0].activeBlocks).toBe(20)
    const d2 = routeKV(r2, ws)
    expect(d2.scores[0].newBlocks).toBe(4) // 16 блоков уже держит активный r1
    ws = applyRoute(ws, d2, r2)
    expect(ws[0].activeBlocks).toBe(24)
    // r1 закончился первым: общие 16 блоков ещё держит r2, освобождаются только 4 собственных
    ws = releaseRoute(ws, d1, 'finish')
    expect(ws[0].activeBlocks).toBe(20)
    // новый r3 с тем же префиксом всё ещё дедуплицируется с r2
    const r3 = blockHashes([...sys, ...range(64, 7000)], 16)
    expect(routeKV(r3, ws).scores[0].newBlocks).toBe(4)
    ws = releaseRoute(ws, d2, 'finish')
    expect(ws[0].activeBlocks).toBe(0)
    expect(ws[0].activeRequests).toBe(0)
    expect(routeKV(r3, ws).scores[0].newBlocks).toBe(20) // активных нет: дедуплицировать не с чем
    expect(ws[0].cachedBlocks.size).toBe(24) // а кэш остался
  })
  it('нагрузка не уходит в минус при любом порядке завершения', () => {
    const rng = mulberry32(5)
    const sys = range(128, 4000)
    let ws = makeWorkers(2)
    const live: Array<{ d: ReturnType<typeof routeKV> }> = []
    for (let step = 0; step < 300; step++) {
      if (live.length > 0 && rng() < 0.45) {
        const k = Math.floor(rng() * live.length)
        const [x] = live.splice(k, 1)
        ws = releaseRoute(ws, x.d, 'prefill')
        ws = releaseRoute(ws, x.d, 'finish')
      } else {
        const blocks = blockHashes([...(rng() < 0.7 ? sys : range(128, 20000 + step * 200)), ...range(randInt(rng, 1, 90), 100000 + step * 100)], 16)
        const d = routeKV(blocks, ws)
        ws = applyRoute(ws, d, blocks)
        live.push({ d })
      }
      for (const w of ws) {
        expect(w.activeBlocks).toBeGreaterThanOrEqual(0)
        expect(w.queuedPrefillTokens ?? 0).toBeGreaterThanOrEqual(0)
        expect(w.activeRequests ?? 0).toBeGreaterThanOrEqual(0)
      }
    }
    for (const x of live) {
      ws = releaseRoute(ws, x.d, 'prefill')
      ws = releaseRoute(ws, x.d, 'finish')
    }
    expect(ws.map((w) => [w.activeBlocks, w.queuedPrefillTokens, w.activeRequests])).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ])
  })
})

describe('перекрытие по цепочке хэшей: свойства на случайных последовательностях', () => {
  /** Брутфорс: сколько первых полных блоков запроса совпадает (по токенам) хотя бы с одной закэшированной последовательностью. */
  function bruteOverlap(req: number[], cachedSeqs: number[][], B: number): number {
    let best = 0
    for (const s of cachedSeqs) {
      let k = 0
      while ((k + 1) * B <= req.length && (k + 1) * B <= s.length && req.slice(k * B, (k + 1) * B).every((x, i) => x === s[k * B + i])) k++
      best = Math.max(best, k)
    }
    return best
  }
  it('cachedPrefixBlocks и overlapByTier = брутфорс; правка в блоке k обнуляет все блоки ≥ k', () => {
    const rng = mulberry32(2024)
    for (let trial = 0; trial < 200; trial++) {
      const B = [1, 2, 4, 16][trial % 4]
      const base = Array.from({ length: randInt(rng, 0, 12 * B) }, () => randInt(rng, 0, 3)) // маленький алфавит: много совпадений
      const cachedSeqs = Array.from({ length: randInt(rng, 0, 4) }, () => {
        const s = base.slice(0, randInt(rng, 0, base.length))
        if (s.length > 0 && rng() < 0.5) s[randInt(rng, 0, s.length - 1)] = randInt(rng, 0, 3)
        return s.concat(Array.from({ length: randInt(rng, 0, 3 * B) }, () => randInt(rng, 0, 3)))
      })
      const set = new Set(cachedSeqs.flatMap((s) => hashes(blockHashes(s, B))))
      const req = base.concat(Array.from({ length: randInt(rng, 0, 2 * B) }, () => randInt(rng, 0, 3)))
      const reqBlocks = blockHashes(req, B)
      const want = bruteOverlap(req, cachedSeqs, B)
      expect(cachedPrefixBlocks(reqBlocks, set)).toBe(want)
      expect(overlapByTier(reqBlocks, worker('w', [...set], 0)).device).toBe(want)
      // правка одного токена в блоке k: блоки < k те же, все блоки ≥ k другие
      const full = reqBlocks.filter((b) => b.full)
      if (full.length > 0) {
        const pos = randInt(rng, 0, full.length * B - 1)
        const edited = req.slice()
        edited[pos] = 1000 + trial
        const eb = blockHashes(edited, B)
        const k = Math.floor(pos / B)
        for (let j = 0; j < full.length; j++) {
          if (j < k) expect(eb[j].fullHash).toBe(full[j].fullHash)
          else expect(eb[j].fullHash).not.toBe(full[j].fullHash)
        }
      }
    }
  })
})

describe('simulateRouting: детерминизм и учёт', () => {
  const model = getModel('qwen3-8b')
  const gpu = getGpu('h100-sxm')
  const reqs = makeTenantWorkload({ tenants: 6, systemTokens: 800, userTokens: [16, 200], requests: 120, ratePerSec: 10, outputTokens: 100, seed: 9 })
  it('одинаковый вход → одинаковый результат; temperature > 0 тоже детерминирована по seed', () => {
    const a = simulateRouting(reqs, makeWorkers(3), { model, gpu, capacityBlocks: 150, seed: 4, router: { overlapCredit: 1, temperature: 0.5 } })
    const b = simulateRouting(reqs, makeWorkers(3), { model, gpu, capacityBlocks: 150, seed: 4, router: { overlapCredit: 1, temperature: 0.5 } })
    expect(b).toEqual(a)
  })
  it('KV-роутинг лучше round-robin по hit rate и TTFT; все TTFT > 0 (полный хит всё равно считает 1 токен)', () => {
    const res = simulateRouting(reqs, makeWorkers(3), { model, gpu, capacityBlocks: 150, seed: 1 })
    const kv = res.byMode.kv!
    const rr = res.byMode['round-robin']!
    expect(kv.hitRate).toBeGreaterThan(rr.hitRate)
    expect(kv.meanTtftMs).toBeLessThan(rr.meanTtftMs)
    for (const s of res.strategies) {
      expect(s.ttftMs.every((x) => x > 0)).toBe(true)
      expect(s.cachedTokens).toBeLessThanOrEqual(s.promptTokens)
      expect(s.perWorker.reduce((a, w) => a + w.promptTokens, 0)).toBe(s.promptTokens)
    }
  })
})
