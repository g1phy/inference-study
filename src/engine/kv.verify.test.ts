/**
 * Независимая проверка kv.ts (верификатор A2): property-тесты на случайных последовательностях
 * операций с брутфорс-эталоном. RadixTree: длина совпадения = самый длинный общий префикс среди
 * сохранённых последовательностей, refs не отрицательны и равны числу блокировок в поддереве,
 * вытеснение не трогает заблокированные узлы. PagedAllocator: нет утечек блоков, ref count = число
 * block tables, где стоит блок, copy-on-write не портит чужие данные, отказ атомарен.
 */
import { describe, expect, it } from 'vitest'
import { ContiguousAllocator, PagedAllocator, RadixTree, blockHashes, type PhysicalBlockView } from './kv'
import { mulberry32, randInt, type Rng } from './rng'
import type { PrefixMatch, RadixNodeView } from './types'

/* ------------------------------------------------------------------ RadixTree ------------------------------------------------------------------ */

/** Все пути от корня до каждого узла (последовательности токенов, которые дерево хранит). */
function storedPaths(v: RadixNodeView, prefix: number[] = [], out: number[][] = []): number[][] {
  for (const c of v.children) {
    const p = prefix.concat(c.tokens)
    out.push(p)
    storedPaths(c, p, out)
  }
  return out
}
function lcp(a: readonly number[], b: readonly number[]): number {
  let k = 0
  while (k < a.length && k < b.length && a[k] === b[k]) k++
  return k
}
function allNodes(v: RadixNodeView): RadixNodeView[] {
  return [v, ...v.children.flatMap(allNodes)]
}
/** id узлов в поддереве (включая сам узел). */
function subtreeIds(v: RadixNodeView): Set<string> {
  return new Set(allNodes(v).map((n) => n.id))
}
function randSeq(rng: Rng, maxLen: number, alphabet: number): number[] {
  return Array.from({ length: randInt(rng, 0, maxLen) }, () => randInt(rng, 0, alphabet - 1))
}

/** Структурные инварианты сжатого дерева. */
function checkTreeInvariants(t: RadixTree, locks: string[]) {
  const root = t.view()
  const nodes = allNodes(root)
  let size = 0
  for (const n of nodes) {
    if (n.id !== 'root') {
      expect(n.tokens.length).toBeGreaterThan(0) // пустых рёбер нет
      size += n.tokens.length
    }
    const first = n.children.map((c) => c.tokens[0])
    expect(new Set(first).size).toBe(first.length) // дети различаются первым токеном
    expect(n.refs).toBeGreaterThanOrEqual(0)
    for (const c of n.children) expect(c.lastAccess).toBeLessThanOrEqual(t.now)
  }
  expect(size).toBe(t.size)
  expect(nodes.length - 1).toBe(t.nodeCount())
  // refs узла = число активных блокировок, чей узел лежит в его поддереве
  for (const n of nodes) {
    if (n.id === 'root') continue
    const sub = subtreeIds(n)
    const expected = locks.filter((id) => sub.has(id)).length
    expect(n.refs, `refs узла ${n.id}`).toBe(expected)
  }
  // заблокированные узлы существуют
  const ids = new Set(nodes.map((n) => n.id))
  for (const id of locks) expect(ids.has(id), `заблокированный ${id} пропал`).toBe(true)
}

describe('RadixTree: property-тесты против брутфорса', () => {
  it('без вытеснения: match = max LCP среди вставленных, insert считает новое верно', () => {
    for (let seed = 1; seed <= 40; seed++) {
      const rng = mulberry32(seed)
      const t = new RadixTree()
      const inserted: number[][] = []
      for (let step = 0; step < 60; step++) {
        const base = inserted.length && rng() < 0.6 ? inserted[randInt(rng, 0, inserted.length - 1)] : []
        const q = base.slice(0, randInt(rng, 0, base.length)).concat(randSeq(rng, 6, 3))
        const want = inserted.reduce((m, s) => Math.max(m, lcp(q, s)), 0)
        if (rng() < 0.5) {
          const sizeBefore = t.size
          const r = t.insert(q)
          expect(r.matchedTokens).toBe(want)
          expect(r.insertedTokens).toBe(q.length - want)
          expect(t.size).toBe(sizeBefore + q.length - want)
          inserted.push(q)
        } else {
          const m = t.match(q)
          expect(m.matchedTokens).toBe(want)
          expect(m.totalTokens).toBe(q.length)
          // путь совпадения ведёт ровно на совпавший префикс
          const hits = allNodes(t.view()).filter((n) => n.hit)
          expect(hits.map((n) => n.id).sort()).toEqual([...m.path].sort())
          expect(hits.reduce((a, n) => a + n.tokens.length, 0)).toBe(want)
        }
        checkTreeInvariants(t, [])
      }
      // хранимые пути = все префиксы вставленного (дерево ничего не теряет и не выдумывает)
      const paths = storedPaths(t.view())
      for (const p of paths) expect(inserted.some((s) => lcp(p, s) === p.length)).toBe(true)
      for (const s of inserted) if (s.length) expect(paths.some((p) => p.length === s.length && lcp(p, s) === s.length)).toBe(true)
    }
  })

  it('со случайными lock / unlock / evict: match = max LCP среди хранимых путей, refs ≥ 0, locked не вытесняются', () => {
    for (let seed = 100; seed < 160; seed++) {
      const rng = mulberry32(seed)
      const t = new RadixTree()
      const locks: string[] = [] // id узлов, на которые взяты блокировки (с повторами)
      const lockTargets: PrefixMatch[] = []
      for (let step = 0; step < 120; step++) {
        const op = rng()
        const paths = storedPaths(t.view())
        const base = paths.length && rng() < 0.7 ? paths[randInt(rng, 0, paths.length - 1)] : []
        const q = base.slice(0, randInt(rng, 0, base.length)).concat(randSeq(rng, 5, 3))
        if (op < 0.35) {
          const want = paths.reduce((m, p) => Math.max(m, lcp(q, p)), 0)
          expect(t.insert(q).matchedTokens).toBe(want)
        } else if (op < 0.55) {
          const want = paths.reduce((m, p) => Math.max(m, lcp(q, p)), 0)
          const m = t.match(q)
          expect(m.matchedTokens).toBe(want)
          if (m.path.length && rng() < 0.6) {
            t.lock(m)
            locks.push(m.path[m.path.length - 1])
            lockTargets.push(m)
          }
        } else if (op < 0.7) {
          if (lockTargets.length) {
            const k = randInt(rng, 0, lockTargets.length - 1)
            t.unlock(lockTargets[k])
            lockTargets.splice(k, 1)
            locks.splice(k, 1)
          }
        } else {
          const lockedBefore = new Set(locks)
          const protectedTokens = t.protectedTokens()
          const cap = randInt(rng, 0, t.size)
          const before = t.size
          const ev = t.evictLRU(cap)
          expect(before - ev.evictedTokens).toBe(t.size)
          for (const id of ev.evictedNodes) expect(lockedBefore.has(id)).toBe(false)
          // вытеснение останавливается либо на ёмкости, либо когда остались только защищённые узлы
          expect(t.size <= cap || t.size === protectedTokens || allNodes(t.view()).every((n) => n.id === 'root' || n.children.length > 0 || n.refs > 0)).toBe(true)
        }
        checkTreeInvariants(t, locks)
      }
    }
  })

  it('LRU: вытесняется лист с самым старым доступом; совпадение «освежает» путь', () => {
    const t = new RadixTree()
    t.insert([1, 2, 3]) // A
    t.insert([4, 5]) // B
    t.insert([6]) // C
    t.match([1, 2, 3]) // A свежее всех
    const ev = t.evictLRU(t.size - 2)
    expect(ev.evictedTokens).toBe(2) // ушёл B (самый старый лист)
    expect(t.match([4, 5]).matchedTokens).toBe(0)
    expect(t.match([1, 2, 3]).matchedTokens).toBe(3)
    expect(t.match([6]).matchedTokens).toBe(1)
  })

  it('двойной unlock не делает refs отрицательными и не ломает чужие блокировки предков', () => {
    const t = new RadixTree()
    t.insert([1, 2, 3, 4])
    t.insert([1, 2, 9])
    const a = t.match([1, 2, 3, 4])
    const b = t.match([1, 2, 9])
    t.lock(a)
    t.lock(b)
    t.unlock(a)
    t.unlock(a) // лишний: refs не уходят в минус
    for (const n of allNodes(t.view())) expect(n.refs).toBeGreaterThanOrEqual(0)
    // b по-прежнему защищён: его лист не вытесняется
    t.evictLRU(0)
    expect(t.match([1, 2, 9]).matchedTokens).toBe(3)
  })
})

/* ------------------------------------------------------------------ PagedAllocator ------------------------------------------------------------------ */

interface Model {
  tokens: Map<string, number>
}

/** Инварианты аллокатора против модели «сколько токенов у каждой последовательности». */
function checkPaged(a: PagedAllocator, model: Model) {
  const B = a.blockSize
  const phys: PhysicalBlockView[] = a.physicalBlocks()
  const seqs = a.sequences()
  expect(seqs.map((s) => [s.id, s.tokens]).sort()).toEqual([...model.tokens.entries()].sort())
  // ref count = в скольких block tables стоит блок; свободные = с нулём
  const count = new Array(a.numBlocks).fill(0)
  for (const s of seqs) {
    expect(s.blockTable.length).toBe(Math.ceil(s.tokens / B))
    expect(new Set(s.blockTable).size).toBe(s.blockTable.length) // блок не повторяется в одной таблице
    for (const b of s.blockTable) count[b]++
    // каждый логический блок заполнен ровно настолько, сколько токенов на нём (CoW не испортил чужое)
    s.blockTable.forEach((b, i) => expect(phys[b].filled, `seq ${s.id} блок ${i}`).toBe(Math.min(B, s.tokens - i * B)))
  }
  for (const p of phys) {
    expect(p.refCount).toBe(count[p.id])
    expect(p.free).toBe(p.refCount === 0)
  }
  const used = phys.filter((p) => !p.free)
  expect(a.freeBlocks).toBe(a.numBlocks - used.length)
  const st = a.stats()
  expect(st.allocatedSlots).toBe(used.length * B)
  expect(st.usedSlots).toBe(used.reduce((s, p) => s + p.filled, 0))
  expect(st.logicalTokens).toBe([...model.tokens.values()].reduce((x, y) => x + y, 0))
  expect(st.internalWasteSlots).toBe(st.allocatedSlots - st.usedSlots)
  expect(st.sharedSavedSlots).toBeGreaterThanOrEqual(0)
  // слоты одной последовательности различны и лежат в её блоках
  for (const s of seqs) {
    const slots = new Set<number>()
    for (let pos = 0; pos < s.tokens; pos++) {
      const slot = a.slotOf(s.id, pos)
      expect(Math.floor(slot / B)).toBe(s.blockTable[Math.floor(pos / B)])
      slots.add(slot)
    }
    expect(slots.size).toBe(s.tokens)
    expect(a.slotOf(s.id, s.tokens)).toBe(-1)
  }
}

describe('PagedAllocator: property-тесты (allocate / append / fork / free)', () => {
  it.each([
    [1, 4, 'ascending'],
    [2, 1, 'ascending'],
    [3, 16, 'scattered'],
    [4, 3, 'scattered'],
    [5, 8, 'ascending'],
  ] as const)('seed %i, B = %i, %s', (seed, B, order) => {
    const rng = mulberry32(seed)
    const a = new PagedAllocator({ numBlocks: 40, blockSize: B, freeOrder: order })
    const model: Model = { tokens: new Map() }
    let next = 0
    for (let step = 0; step < 400; step++) {
      const ids = [...model.tokens.keys()]
      const op = rng()
      const snapshot = JSON.stringify([a.sequences(), a.physicalBlocks(), a.freeBlocks])
      if (op < 0.25 || ids.length === 0) {
        const id = `s${next++}`
        const t = randInt(rng, 0, 6 * B)
        if (a.allocate(id, t)) model.tokens.set(id, t)
        else expect(JSON.stringify([a.sequences(), a.physicalBlocks(), a.freeBlocks])).toBe(snapshot) // атомарно
      } else if (op < 0.65) {
        const id = ids[randInt(rng, 0, ids.length - 1)]
        const n = randInt(rng, 0, 2 * B + 1)
        if (a.append(id, n)) model.tokens.set(id, model.tokens.get(id)! + n)
        else expect(JSON.stringify([a.sequences(), a.physicalBlocks(), a.freeBlocks])).toBe(snapshot)
      } else if (op < 0.8) {
        const id = ids[randInt(rng, 0, ids.length - 1)]
        const child = `s${next++}`
        a.fork(id, child)
        model.tokens.set(child, model.tokens.get(id)!)
      } else {
        const id = ids[randInt(rng, 0, ids.length - 1)]
        a.free(id)
        model.tokens.delete(id)
      }
      checkPaged(a, model)
    }
    // нет утечек: после освобождения всего пул снова полон
    for (const id of [...model.tokens.keys()]) a.free(id)
    model.tokens.clear()
    checkPaged(a, model)
    expect(a.freeBlocks).toBe(40)
    expect(a.physicalBlocks().every((p) => p.refCount === 0 && p.filled === 0)).toBe(true)
  })

  it('CoW: после fork и записи в общий неполный блок данные родителя не меняются', () => {
    const a = new PagedAllocator({ numBlocks: 10, blockSize: 4 })
    a.allocate('P', 6) // блоки [4, 2]
    a.fork('P', 'C1')
    a.fork('P', 'C2')
    const shared = a.blockTable('P')
    expect(a.physicalBlocks()[shared[1]].refCount).toBe(3)
    a.append('C1', 3) // CoW: копия 2 токенов, +2 до конца блока, +1 в новый блок
    expect(a.physicalBlocks()[shared[1]].refCount).toBe(2)
    expect(a.physicalBlocks()[shared[1]].filled).toBe(2) // общий блок P и C2 не тронут
    expect(a.blockTable('C1')[0]).toBe(shared[0]) // полный блок по-прежнему общий
    expect(a.blockTable('C1')).toHaveLength(3)
    a.append('C2', 1) // ещё один CoW: блок всё ещё общий с P
    expect(a.lastEvents[0].kind).toBe('cow')
    a.append('P', 1) // теперь P — единственный владелец: пишет на месте
    expect(a.lastEvents.some((e) => e.kind === 'cow')).toBe(false)
    expect(a.blockTable('P')[1]).toBe(shared[1])
  })

  it('CoW при нехватке блоков: append отказывает целиком, ничего не копируя', () => {
    const a = new PagedAllocator({ numBlocks: 2, blockSize: 4 })
    a.allocate('P', 6)
    a.fork('P', 'C')
    expect(a.append('C', 1)).toBe(false) // нужен 1 блок под копию, свободных 0
    expect(a.blockTable('C')).toEqual(a.blockTable('P'))
    expect(a.physicalBlocks()[a.blockTable('P')[1]].refCount).toBe(2)
  })
})

/* ------------------------------------------------------------------ ContiguousAllocator ------------------------------------------------------------------ */

describe('ContiguousAllocator: регионы не пересекаются, учёт сходится', () => {
  it('случайные операции', () => {
    const rng = mulberry32(77)
    const c = new ContiguousAllocator({ capacity: 500, maxLen: 64 })
    const live = new Map<string, { used: number; size: number }>()
    for (let step = 0; step < 500; step++) {
      const ids = [...live.keys()]
      const op = rng()
      if (op < 0.4 || ids.length === 0) {
        const id = `r${step}`
        const t = randInt(rng, 0, 40)
        const reserve = randInt(rng, 0, 80) // 0: бронь нулевой длины (запрос без токенов)
        if (c.allocate(id, t, reserve)) live.set(id, { used: t, size: Math.max(t, reserve) })
      } else if (op < 0.7) {
        const id = ids[randInt(rng, 0, ids.length - 1)]
        const n = randInt(rng, 0, 10)
        const r = live.get(id)!
        expect(c.append(id, n)).toBe(r.used + n <= r.size)
        if (r.used + n <= r.size) r.used += n
      } else {
        const id = ids[randInt(rng, 0, ids.length - 1)]
        c.free(id)
        live.delete(id)
      }
      const regions = c.regions()
      let cur = 0
      for (const r of regions) {
        expect(r.start).toBe(cur) // регионы и дыры покрывают память подряд, без пересечений
        cur += r.length
      }
      expect(cur).toBe(500)
      const st = c.stats()
      expect(st.allocatedSlots).toBe([...live.values()].reduce((a, r) => a + r.size, 0))
      expect(st.usedSlots).toBe([...live.values()].reduce((a, r) => a + r.used, 0))
      expect(st.freeSlots + st.allocatedSlots).toBe(500)
      expect(st.externalWasteSlots).toBeLessThanOrEqual(st.freeSlots)
    }
  })
  it('нулевая бронь не дробит свободную память', () => {
    const c = new ContiguousAllocator({ capacity: 100, maxLen: 40 })
    c.allocate('A', 10, 50)
    c.allocate('Z', 0, 0)
    c.free('A')
    expect(c.regions()).toEqual([{ start: 0, length: 100, seqId: null, used: 0 }])
    expect(c.allocate('B', 10, 80)).toBe(true) // 100 свободных слотов подряд
    expect(c.allocate('Z2', 0, 0)).toBe(true) // нулевая бронь не требует места даже в заполненной памяти
  })
})

/* ------------------------------------------------------------------ blockHashes ------------------------------------------------------------------ */

describe('blockHashes: хэш блока зависит ровно от всех токенов до конца блока', () => {
  it('случайные пары последовательностей: равенство хэша ⇔ равенство всего префикса', () => {
    const rng = mulberry32(31337)
    for (let trial = 0; trial < 300; trial++) {
      const B = randInt(rng, 1, 5)
      const a = randSeq(rng, 6 * B, 2)
      const b = rng() < 0.5 ? a.slice(0, randInt(rng, 0, a.length)).concat(randSeq(rng, 3 * B, 2)) : randSeq(rng, 6 * B, 2)
      const ha = blockHashes(a, B)
      const hb = blockHashes(b, B)
      for (let k = 0; k < Math.min(ha.length, hb.length); k++) {
        if (!ha[k].full || !hb[k].full) continue
        const samePrefix = lcp(a, b) >= (k + 1) * B
        expect(ha[k].fullHash === hb[k].fullHash, `trial ${trial} блок ${k}`).toBe(samePrefix)
      }
    }
  })
})
