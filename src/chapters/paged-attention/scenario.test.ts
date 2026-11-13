import { describe, expect, it } from 'vitest'
import { buildScenario, meanUtilization, CAPACITY } from './scenario'

describe('сценарий PagedAttention против непрерывной брони', () => {
  const snaps = buildScenario()

  it('шагов 13, память 512 слотов', () => {
    expect(snaps.length).toBe(13)
    expect(snaps[0].contiguous.cells.length).toBe(CAPACITY)
  })

  it('D ждёт в непрерывной схеме из-за внешней фрагментации, а блоки принимают его сразу', () => {
    const s7 = snaps[7]
    expect(s7.contiguous.waiting.map((w) => w.id)).toEqual(['D'])
    expect(s7.contiguous.waiting[0].reason).toMatch(/нет непрерывного куска/)
    expect(s7.paged.waiting).toEqual([])
    expect(s7.paged.running).toContain('D')
  })

  it('E стоит в очереди за D (FIFO) в непрерывной схеме', () => {
    expect(snaps[8].contiguous.waiting.map((w) => w.id)).toEqual(['D', 'E'])
    expect(snaps[8].paged.running).toEqual(expect.arrayContaining(['B', 'C', 'D', 'E']))
  })

  it('после ухода B непрерывная схема наконец пускает D и E', () => {
    const s10 = snaps[10]
    expect(s10.contiguous.waiting).toEqual([])
    expect(s10.contiguous.admittedAt.D).toBe(10)
    expect(s10.paged.admittedAt.D).toBe(7)
  })

  it('утилизация выделенного у блоков заметно выше', () => {
    const c = meanUtilization(snaps, 'contiguous')
    const p = meanUtilization(snaps, 'paged')
    expect(p).toBeGreaterThan(0.8)
    expect(c).toBeLessThan(0.55)
    // потери у блоков не больше B − 1 = 15 слотов на последовательность
    for (const s of snaps) {
      const st = s.paged.stats
      expect(st.allocatedSlots - st.usedSlots).toBeLessThanOrEqual(15 * Math.max(1, st.sequences))
    }
  })
})
