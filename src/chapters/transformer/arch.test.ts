import { describe, expect, test } from 'vitest'
import { getModel } from '@/engine/data/models'
import { prefillFlopsBreakdown } from '@/engine/perf'
import { ARCH_MODEL_IDS, attentionCrossover, computeArch, modelFlops } from './arch'

describe('разбивка параметров по частям совпадает с данными моделей', () => {
  for (const id of ARCH_MODEL_IDS) {
    test(id, () => {
      const m = getModel(id)
      const a = computeArch(m)
      // всего параметров: как paramsB в данных (там 3-4 значащие цифры после запятой)
      expect(Math.abs(a.totalParams / 1e9 - m.paramsB) / m.paramsB).toBeLessThan(5e-4)
      expect(Math.abs(a.totalActiveParams / 1e9 - m.activeParamsB) / m.activeParamsB).toBeLessThan(5e-4)
      // LM head как в данных
      expect(Math.abs(a.head.params / 1e9 - m.lmHeadParamsB) / m.lmHeadParamsB).toBeLessThan(5e-4)
      // умножения на токен = 2 × activeMatmulParamsB (без таблицы эмбеддингов)
      const matmul = m.layers * (a.attn.flopsFixed + a.mlp.flopsFixed) + a.head.flopsFixed
      expect(Math.abs(matmul / 2e9 - m.activeMatmulParamsB) / m.activeMatmulParamsB).toBeLessThan(5e-4)
    })
  }
})

test('точные числа Qwen3-8B и Llama-3.1-8B (notes-models §3.10)', () => {
  const q = computeArch(getModel('qwen3-8b'))
  expect(q.attn.params).toBe(41_943_040 + 256)
  expect(q.mlp.params).toBe(150_994_944)
  expect(q.layerParams).toBe(192_946_432)
  expect(q.totalParams).toBe(8_190_735_360)
  const l = computeArch(getModel('llama-3.1-8b'))
  expect(l.layerParams).toBe(218_112_000)
  expect(l.totalParams).toBe(8_030_261_248)
})

test('FLOPs согласованы с perf.prefillFlopsBreakdown', () => {
  const m = getModel('qwen3-8b')
  const a = computeArch(m)
  // decode-токен на позиции P смотрит на P + 1 ключей (себя тоже); prefill одного токена после P закэшированных
  const P = 1000
  const viaPerf = prefillFlopsBreakdown(m, P + 1, { cachedTokens: P }).total
  const mine = modelFlops(a, P + 1) - m.layers * (2 * a.norm.flopsFixed + 2 * a.add.flopsFixed) - a.finalNorm.flopsFixed
  // в данных activeMatmulParamsB округлено до 3 знаков, отсюда расхождение ~1e-5
  expect(Math.abs(mine - viaPerf) / viaPerf).toBeLessThan(1e-4)
})

test('порог контекста, после которого attention дороже матриц (Llama-3.1-8B ≈ 28,6 тыс., notes-models §5.7)', () => {
  const l = computeArch(getModel('llama-3.1-8b'))
  expect(attentionCrossover(l)).toBeGreaterThan(28_000)
  expect(attentionCrossover(l)).toBeLessThan(29_000)
})
