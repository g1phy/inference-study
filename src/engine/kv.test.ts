import { describe, expect, it } from 'vitest'
import {
  ContiguousAllocator,
  DEFAULT_BLOCK_SIZE,
  PagedAllocator,
  RadixTree,
  blockHashes,
  cachedPrefixBlocks,
  commonPrefixBlocks,
} from './kv'
import type { RadixNodeView } from './types'

const range = (n: number, from = 0) => Array.from({ length: n }, (_, i) => from + i)

describe('blockHashes', () => {
  it('режет на блоки, хвост не хэшируется', () => {
    const b = blockHashes(range(40), 16)
    expect(b).toHaveLength(3)
    expect(b.map((x) => x.full)).toEqual([true, true, false])
    expect(b[2].hash).toBe('')
    expect(b[2].tokenIds).toEqual(range(8, 32))
    expect(b[0].hash).toMatch(/^[0-9a-f]{8}$/)
    expect(b[0].fullHash).toMatch(/^[0-9a-f]{16}$/)
    expect(b[0].fullHash.startsWith(b[0].hash)).toBe(true)
    expect(b[1].parentHash).toBe(b[0].fullHash)
    expect(b.map((x) => x.index)).toEqual([0, 1, 2])
    expect(b.map((x) => x.start)).toEqual([0, 16, 32])
  })
  it('ровно кратная длина: без хвоста; пустой вход: пусто', () => {
    expect(blockHashes(range(32), 16).every((x) => x.full)).toBe(true)
    expect(blockHashes([], 16)).toEqual([])
    expect(DEFAULT_BLOCK_SIZE).toBe(16)
  })
  it('детерминированно и цепочкой: правка раннего токена меняет все следующие хэши', () => {
    const a = blockHashes(range(64), 16)
    const again = blockHashes(range(64), 16)
    expect(again.map((x) => x.hash)).toEqual(a.map((x) => x.hash))
    const edited = range(64)
    edited[20] = 99999 // блок 1
    const b = blockHashes(edited, 16)
    expect(b[0].hash).toBe(a[0].hash)
    expect(b[1].hash).not.toBe(a[1].hash)
    expect(b[2].hash).not.toBe(a[2].hash) // тот же контент блока 2, но другой родитель
    expect(b[3].hash).not.toBe(a[3].hash)
    expect(commonPrefixBlocks(a, b)).toBe(1)
  })
  it('одинаковые блоки на разных позициях дают разные хэши (позиция через цепочку)', () => {
    const rep = [...range(16), ...range(16)]
    const b = blockHashes(rep, 16)
    expect(b[0].hash).not.toBe(b[1].hash)
  })
  it('соль (тенант / cache_salt) меняет всю цепочку', () => {
    const a = blockHashes(range(48), 16)
    const s = blockHashes(range(48), 16, 'tenant-B')
    expect(s.every((x, i) => x.hash !== a[i].hash)).toBe(true)
    expect(blockHashes(range(48), 16, 'tenant-B').map((x) => x.hash)).toEqual(s.map((x) => x.hash))
  })
  it('пример vLLM: общий system prompt 40 токенов даёт попадание в 2 блока (notes-engines §3.6)', () => {
    const sys = range(40, 1000)
    const r1 = blockHashes([...sys, ...range(10, 5000)], 16)
    const r2 = blockHashes([...sys, ...range(12, 7000)], 16)
    expect(commonPrefixBlocks(r1, r2)).toBe(2)
    const cached = new Set(r1.filter((x) => x.full).map((x) => x.hash))
    expect(cachedPrefixBlocks(r2, cached)).toBe(2)
    const sys3 = sys.slice()
    sys3[5] = 1 // одна «буква» в начале
    expect(commonPrefixBlocks(r1, blockHashes([...sys3, ...range(10, 5000)], 16))).toBe(0)
  })
})

describe('PagedAllocator', () => {
  it('пример статьи «Four score and seven years ago our», B = 4 (notes-engines §3.1)', () => {
    const a = new PagedAllocator({ numBlocks: 8, blockSize: 4 })
    expect(a.allocate('A', 7)).toBe(true)
    expect(a.blockTable('A')).toHaveLength(2)
    expect(a.sequences()[0].filledInLast).toBe(3)
    expect(a.append('A')).toBe(true) // «fathers»: влезает в блок 1
    expect(a.blockTable('A')).toHaveLength(2)
    expect(a.append('A')).toBe(true) // «brought»: новый блок
    expect(a.blockTable('A')).toHaveLength(3)
    expect(a.lastEvents.some((e) => e.kind === 'new-block')).toBe(true)
    const s = a.stats()
    expect(s.usedSlots).toBe(9)
    expect(s.allocatedSlots).toBe(12)
    expect(s.internalWasteSlots).toBe(3) // ≤ B − 1
    expect(s.externalWasteSlots).toBe(0)
    expect(s.utilization).toBeCloseTo(0.75, 6)
    // slot mapping: позиция 8 лежит в третьем блоке со смещением 0
    const t = a.blockTable('A')
    expect(a.slotOf('A', 8)).toBe(t[2] * 4)
    expect(a.slotOf('A', 5)).toBe(t[1] * 4 + 1)
  })
  it('copy-on-write при parallel sampling (notes-engines §3.2)', () => {
    const a = new PagedAllocator({ numBlocks: 8, blockSize: 4 })
    a.allocate('A1', 7)
    a.fork('A1', 'A2')
    const shared = a.blockTable('A1')
    expect(a.blockTable('A2')).toEqual(shared)
    expect(a.physicalBlocks()[shared[1]].refCount).toBe(2)
    expect(a.stats().sharedSavedSlots).toBe(7)
    a.append('A1') // пишет в общий неполный блок → CoW
    expect(a.lastEvents[0].kind).toBe('cow')
    const t1 = a.blockTable('A1')
    expect(t1[0]).toBe(shared[0])
    expect(t1[1]).not.toBe(shared[1])
    expect(a.physicalBlocks()[shared[1]].refCount).toBe(1)
    expect(a.physicalBlocks()[t1[1]].filled).toBe(4) // скопировано 3 + записан 1
    a.append('A2') // ref = 1: пишет напрямую
    expect(a.lastEvents.some((e) => e.kind === 'cow')).toBe(false)
    expect(a.blockTable('A2')[1]).toBe(shared[1])
    expect(a.stats().allocatedSlots).toBe(3 * 4) // 3 блока вместо 4 без шаринга
  })
  it('free возвращает блоки; общие блоки живут, пока нужны', () => {
    const a = new PagedAllocator({ numBlocks: 6, blockSize: 4 })
    a.allocate('X', 8)
    a.fork('X', 'Y')
    a.free('X')
    expect(a.freeBlocks).toBe(4)
    a.free('Y')
    expect(a.freeBlocks).toBe(6)
    expect(a.stats().sequences).toBe(0)
  })
  it('нехватка блоков: allocate и append возвращают false и ничего не меняют', () => {
    const a = new PagedAllocator({ numBlocks: 2, blockSize: 4 })
    expect(a.allocate('A', 8)).toBe(true)
    expect(a.allocate('B', 1)).toBe(false)
    expect(a.lastEvents[0].kind).toBe('fail')
    expect(a.append('A')).toBe(false)
    expect(a.sequences()[0].tokens).toBe(8)
  })
  it('scattered: блоки последовательности не подряд, но все разные', () => {
    const a = new PagedAllocator({ numBlocks: 10, blockSize: 4, freeOrder: 'scattered' })
    a.allocate('A', 12)
    const t = a.blockTable('A')
    expect(new Set(t).size).toBe(3)
    expect(t[1] - t[0] === 1 && t[2] - t[1] === 1).toBe(false)
  })
})

describe('ContiguousAllocator', () => {
  it('бронь под max_len: утилизация низкая (Orca Max)', () => {
    const c = new ContiguousAllocator({ capacity: 4096, maxLen: 2048 })
    expect(c.allocate('A', 7)).toBe(true)
    c.append('A', 3)
    const s = c.stats()
    expect(s.usedSlots).toBe(10)
    expect(s.allocatedSlots).toBe(2048)
    expect(s.utilization).toBeCloseTo(10 / 2048, 6) // 99.5 % брони пропадает
  })
  it('внешняя фрагментация: свободно много, но кусками', () => {
    const c = new ContiguousAllocator({ capacity: 100, maxLen: 40 })
    c.allocate('A', 10, 30)
    c.allocate('B', 10, 30)
    c.allocate('C', 10, 30)
    c.free('B')
    // свободно 30 (дыра) + 10 (хвост) = 40, но непрерывных 40 нет
    expect(c.allocate('D', 5, 40)).toBe(false)
    expect(c.lastEvents[0].note).toMatch(/внешняя фрагментация/)
    const s = c.stats()
    expect(s.freeSlots).toBe(40)
    expect(s.externalWasteSlots).toBe(40)
    expect(c.regions().filter((r) => r.seqId === null).map((r) => r.length)).toEqual([30, 10])
  })
  it('бронь кончилась: append возвращает false', () => {
    const c = new ContiguousAllocator({ capacity: 64, maxLen: 8 })
    c.allocate('A', 7)
    expect(c.append('A')).toBe(true)
    expect(c.append('A')).toBe(false)
  })
  it('PagedAttention утилизирует память лучше на тех же запросах', () => {
    const paged = new PagedAllocator({ numBlocks: 256, blockSize: 16 })
    const cont = new ContiguousAllocator({ capacity: 4096, maxLen: 1024 })
    const lens = [120, 300, 45, 800, 260]
    lens.forEach((l, i) => {
      paged.allocate(`r${i}`, l)
      cont.allocate(`r${i}`, l)
    })
    expect(paged.stats().utilization).toBeGreaterThan(0.95)
    expect(cont.stats().utilization).toBeLessThan(0.35)
  })
})

/** Все узлы дерева списком. */
function flatten(v: RadixNodeView): RadixNodeView[] {
  return [v, ...v.children.flatMap(flatten)]
}

describe('RadixTree', () => {
  it('вставка, совпадение, расщепление узла посреди ребра', () => {
    const t = new RadixTree()
    const sys = [1, 2, 3, 4, 5, 6]
    const r1 = t.insert([...sys, 10, 11])
    expect(r1.matchedTokens).toBe(0)
    expect(r1.insertedTokens).toBe(8)
    expect(t.size).toBe(8)
    expect(t.nodeCount()).toBe(1)
    // второй чат с тем же system prompt: узел первого хода расщепляется
    const r2 = t.insert([...sys, 20, 21, 22])
    expect(r2.matchedTokens).toBe(6)
    expect(r2.insertedTokens).toBe(3)
    expect(t.size).toBe(11)
    expect(t.nodeCount()).toBe(3)
    const root = t.view()
    expect(root.children).toHaveLength(1)
    expect(root.children[0].tokens).toEqual(sys)
    expect(root.children[0].children.map((c) => c.tokens)).toEqual([
      [10, 11],
      [20, 21, 22],
    ])
    const m = t.match([...sys, 10, 99])
    expect(m.matchedTokens).toBe(7)
    expect(m.totalTokens).toBe(8)
    // совпадение кончилось посреди ребра [10, 11] → оно расщепилось на [10] и [11]
    expect(t.nodeCount()).toBe(4)
    expect(m.path).toHaveLength(2)
    const hits = flatten(t.view()).filter((n) => n.hit).map((n) => n.id)
    expect(hits).toEqual(m.path)
  })
  it('вставка префикса уже существующей последовательности ничего не добавляет', () => {
    const t = new RadixTree()
    t.insert([1, 2, 3, 4, 5])
    const r = t.insert([1, 2, 3])
    expect(r.insertedTokens).toBe(0)
    expect(r.matchedTokens).toBe(3)
    expect(t.size).toBe(5)
  })
  it('LRU-вытеснение листьев; заблокированные узлы не трогаются; предки живут дольше', () => {
    const t = new RadixTree()
    t.insert([1, 2, 3, 10, 11]) // чат A
    t.insert([1, 2, 3, 20, 21]) // чат B (общий префикс 1,2,3)
    t.insert([1, 2, 3, 10, 11, 12]) // A продолжается: A свежее B
    const before = t.size
    expect(before).toBe(3 + 2 + 2 + 1)
    const ev = t.evictLRU(before - 2)
    expect(ev.evictedTokens).toBe(2) // первым ушёл лист чата B [20, 21]
    expect(t.match([1, 2, 3, 20, 21]).matchedTokens).toBe(3)
    // блокируем чат A и требуем ёмкость 0: заблокированный путь выживает
    const mA = t.match([1, 2, 3, 10, 11, 12])
    expect(mA.matchedTokens).toBe(6)
    t.lock(mA)
    const ev2 = t.evictLRU(0)
    expect(ev2.evictedTokens).toBe(0)
    expect(t.size).toBe(6)
    expect(t.protectedTokens()).toBe(6)
    t.unlock(mA)
    expect(t.evictLRU(0).evictedTokens).toBe(6)
    expect(t.size).toBe(0)
    expect(t.view().children).toHaveLength(0)
  })
  it('логические часы растут, lastAccess обновляется при совпадении', () => {
    const t = new RadixTree()
    t.insert([5, 6, 7])
    const id = t.view().children[0].id
    const before = t.view().children[0].lastAccess
    t.match([5, 6, 7])
    const after = t.view().children[0]
    expect(after.id).toBe(id)
    expect(after.lastAccess).toBeGreaterThan(before)
    expect(t.now).toBe(2)
  })
  it('пустое дерево: совпадение 0', () => {
    const t = new RadixTree()
    expect(t.match([1, 2]).matchedTokens).toBe(0)
    expect(t.match([1, 2]).path).toEqual([])
  })
})
