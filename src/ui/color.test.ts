import { describe, expect, it } from 'vitest'
import { divColor, divInk, seqColor, seqInk } from './color'

describe('шкалы heatmap', () => {
  it('там, где светлый текст сменяется тёмным, заливка встаёт на ступень под свой текст', () => {
    // seq 3|4: смесь двух ступеней дала бы текст не той ступени
    const t = 3.3 / 7
    expect(seqColor(t)).toBe('var(--seq-3)')
    expect(seqInk(t)).toBe('var(--seq-ink-3)')
    const t2 = 3.7 / 7
    expect(seqColor(t2)).toBe('var(--seq-4)')
    expect(seqInk(t2)).toBe('var(--seq-ink-4)')
    // div 6|7 (положительные около 0,6)
    const d = (6.31 / 8) * 2 - 1
    expect(divColor(d)).toBe('var(--div-6)')
    expect(divInk(d)).toBe('var(--div-ink-6)')
  })
  it('в остальных парах заливка плавная, текст общий у обеих ступеней', () => {
    expect(seqColor(1.5 / 7)).toBe('color-mix(in oklab, var(--seq-1), var(--seq-2) 50%)')
    expect(seqInk(1.4 / 7)).toBe('var(--seq-ink-1)')
    expect(seqColor(0)).toBe('var(--seq-0)')
    expect(seqColor(1)).toBe('var(--seq-7)')
  })
})
