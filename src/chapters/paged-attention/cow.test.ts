import { describe, expect, it } from 'vitest'
import { buildCow, slotOf, COW_BLOCK } from './cow'

const P = ['Об', 'ъ', 'яс', 'ни', ' прост', 'ыми', ' слов', 'ами', ',', ' что']
describe('copy-on-write при двух ответах', () => {
  const steps = buildCow(P, [' Это', ' память', ' модели'], [' Пред', 'ставь', 'те'])
  it('8 шагов, промпт 10 токенов в 3 блоках', () => {
    expect(steps.length).toBe(8)
    expect(steps[1].seqs[0].table.length).toBe(3)
  })
  it('после fork все три блока с ref = 2', () => {
    const t = steps[2].seqs[0].table
    for (const b of t) expect(steps[2].blocks[b].ref).toBe(2)
  })
  it('первая запись ответа 1 копирует общий неполный блок', () => {
    const s3 = steps[3]
    const [o1, o2] = s3.seqs
    expect(o1.table.slice(0, 2)).toEqual(o2.table.slice(0, 2))
    expect(o1.table[2]).not.toBe(o2.table[2])
    expect(s3.notes.join(' ')).toMatch(/copy-on-write/)
    const copy = s3.blocks[o1.table[2]]
    expect(copy.copyOf).toBe(o2.table[2])
    expect(copy.slots).toEqual([',', ' что', ' Это', null])
  })
  it('ответ 2 пишет без копии, у последнего блока ref = 1', () => {
    const s4 = steps[4]
    const o2 = s4.seqs.find((s) => s.label === 'Ответ 2')!
    expect(s4.blocks[o2.table[2]].ref).toBe(1)
  })
  it('после конца ответа 1 общие блоки остались с ref = 1', () => {
    const last = steps[7]
    expect(last.seqs.length).toBe(1)
    const t = last.seqs[0].table
    expect(last.blocks[t[0]].ref).toBe(1)
    expect(last.blocks[t[1]].ref).toBe(1)
  })
  it('slot mapping', () => {
    expect(slotOf([7, 1, 3], 9)).toBe(3 * COW_BLOCK + 1)
  })
})
