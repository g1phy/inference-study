import { describe, expect, it } from 'vitest'
import { quantizeBlocks, sampleWeights } from './blocks'

// Блок из notes-techniques §3.4 (amax = 0,6)
const BLOCK = [0.12, -0.03, 0.45, 0.08, -0.27, 0.6, 0.01, -0.15, 0.33, -0.52, 0.07, 0.2, -0.09, 0.41, 0.05, -0.36]

describe('блочные масштабы FP4', () => {
  it('MXFP4: масштаб 2^−3 = 0,125, MSE ≈ 0,001241 (как в заметках)', () => {
    const r = quantizeBlocks(BLOCK, 'mxfp4')
    expect(r.blocks[0].scale).toBe(0.125)
    expect(r.q.map((v) => v / 0.125)).toEqual([1, 0, 4, 0.5, -2, 4, 0, -1, 3, -4, 0.5, 1.5, -0.5, 3, 0.5, -3])
    expect(r.mse).toBeCloseTo(0.001241, 6)
    expect(r.bitsPerValue).toBe(4.25)
  })
  it('NVFP4: дробный масштаб 0,1 (глобальный FP32 делает его точным), сетка как в заметках', () => {
    const r = quantizeBlocks(BLOCK, 'nvfp4')
    expect(r.blocks[0].scale).toBeCloseTo(0.1, 12)
    expect(r.q.map((v) => Math.round((v / 0.1) * 1e9) / 1e9)).toEqual([1, -0.5, 4, 1, -3, 6, 0, -1.5, 3, -6, 0.5, 2, -1, 4, 0.5, -4])
    expect(r.mse).toBeCloseTo(0.0008875, 7)
    expect(r.bitsPerValue).toBe(4.5)
  })
  it('NVFP4 точнее MXFP4 на этом блоке', () => {
    expect(quantizeBlocks(BLOCK, 'nvfp4').mse).toBeLessThan(quantizeBlocks(BLOCK, 'mxfp4').mse)
  })
  it('выброс: при одном масштабе мелкие веса обнуляются, блоки спасают', () => {
    const w = sampleWeights({ outlier: true })
    const t = quantizeBlocks(w, 'tensor')
    const mx = quantizeBlocks(w, 'mxfp4')
    const nv = quantizeBlocks(w, 'nvfp4')
    expect(t.blocks).toHaveLength(1)
    expect(mx.blocks).toHaveLength(2)
    expect(nv.blocks).toHaveLength(4)
    expect(t.zeroed).toBeGreaterThan(nv.zeroed)
    expect(t.mse).toBeGreaterThan(nv.mse)
  })
  it('без выброса веса детерминированы и малы', () => {
    const a = sampleWeights({ outlier: false })
    const b = sampleWeights({ outlier: false })
    expect(a).toEqual(b)
    expect(a).toHaveLength(64)
    expect(Math.max(...a.map(Math.abs))).toBeLessThan(0.2)
  })
})
