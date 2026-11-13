import { describe, expect, it } from 'vitest'
import { PRESETS, getPreset, radixStep, totalUniqueTokens, describeTokens, segmentTokens } from './radix'

describe('radix tree: пресеты', () => {
  it('многоходовый чат: второй ход берёт из кэша весь первый ход', () => {
    const p = getPreset('chat')
    const s = radixStep(p, 2, totalUniqueTokens(p))
    const h = s.history[1]
    expect(h.matched).toBe(400 + 40 + 160)
    expect(h.matched / h.promptTokens).toBeGreaterThan(0.9)
  })
  it('чат Б расщепляет узел: системный промпт становится общим', () => {
    const p = getPreset('chat')
    const s = radixStep(p, 3, totalUniqueTokens(p))
    const sys = s.nodes.find((n) => n.label === 'Системный промпт')!
    expect(sys).toBeTruthy()
    expect(sys.shared).toBe(3)
    expect(s.history[2].matched).toBe(400)
  })
  it('без ограничения ёмкости размер дерева = сумма уникальных токенов', () => {
    for (const p of PRESETS) {
      const s = radixStep(p, p.requests.length, totalUniqueTokens(p))
      expect(s.size).toBe(totalUniqueTokens(p))
      expect(s.evicted).toEqual([])
    }
  })
  it('малая ёмкость вытесняет листья, общий системный промпт живёт', () => {
    const p = getPreset('system')
    const s = radixStep(p, 5, 1200)
    expect(s.size).toBeLessThanOrEqual(1200)
    expect(s.nodes.some((n) => n.label === 'Системный промпт')).toBe(true)
    // каждый следующий пользователь всё равно попадает в системный промпт
    for (const h of s.history.slice(1)) expect(h.matched).toBe(800)
  })
  it('подписи узлов', () => {
    const p = getPreset('fewshot')
    const toks = ['I', 'e1', 'e2', 'e3'].flatMap((k) => segmentTokens(p, k))
    expect(describeTokens(p, toks).label).toBe('Инструкция … Пример 3')
    expect(describeTokens(p, toks.slice(0, 100)).label).toBe('часть: Инструкция')
  })
  it('у каждого узла кроме корня есть родитель в списке', () => {
    const p = getPreset('chat')
    const s = radixStep(p, 5, totalUniqueTokens(p))
    const ids = new Set(s.nodes.map((n) => n.id))
    for (const n of s.nodes) if (n.id !== 'root') expect(ids.has(n.parent!)).toBe(true)
  })
})
