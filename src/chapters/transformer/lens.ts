/**
 * Заскриптованный «logit lens» для контекста «Столица Франции —»: что предсказала бы модель,
 * если бы на слое i к потоку приложили финальный RMSNorm и LM head. Числа условные, но форма
 * типичная: ранние слои предлагают частые служебные слова, ответ проявляется в последней трети.
 * На последнем слое логиты совпадают с распределением главы «Логиты и сэмплинг».
 */
import { softmax } from '@/engine/sampling'

interface LensCandidate {
  text: string
  /** Логит в начале и в конце стопки, центр и ширина перехода (доля глубины). */
  from: number
  to: number
  center: number
  width: number
  /** «Горб» в середине: высота и положение. */
  bump?: { h: number; at: number; w: number }
}

export const LENS_CONTEXT = 'Столица Франции —'

const CANDIDATES: LensCandidate[] = [
  { text: ' и', from: 2.2, to: -0.5, center: 0.45, width: 0.08 },
  { text: ' в', from: 2.0, to: -0.8, center: 0.4, width: 0.08 },
  { text: ',', from: 1.8, to: 0.0, center: 0.5, width: 0.1 },
  { text: ' это', from: 0.6, to: 3.6, center: 0.55, width: 0.12 },
  { text: ' город', from: 0.2, to: 4.1, center: 0.45, width: 0.1 },
  { text: ' столица', from: -0.5, to: 0.4, center: 0.5, width: 0.1, bump: { h: 2.6, at: 0.47, w: 0.12 } },
  { text: ' Париж', from: -1.5, to: 6.2, center: 0.64, width: 0.07 },
  { text: ' Лион', from: -1.0, to: 1.6, center: 0.62, width: 0.1 },
]

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))

/** Распределение среди кандидатов на доле глубины f ∈ [0, 1]. */
export function lensAt(f: number): Array<{ text: string; p: number; logit: number }> {
  const logits = CANDIDATES.map((c) => {
    // сигмоида, растянутая так, что на входе ровно from, а на последнем слое ровно to
    const s0 = sigmoid(-c.center / c.width)
    const s1 = sigmoid((1 - c.center) / c.width)
    const ramp = (sigmoid((f - c.center) / c.width) - s0) / (s1 - s0)
    let z = c.from + (c.to - c.from) * ramp
    if (c.bump) z += c.bump.h * Math.exp(-(((f - c.bump.at) / c.bump.w) ** 2))
    return z
  })
  const p = softmax(logits, 1)
  return CANDIDATES.map((c, i) => ({ text: c.text, p: p[i], logit: logits[i] }))
}

/** Топ-k на слое layer из layers (0 = эмбеддинг, layers = выход последнего блока). */
export function lensTop(layer: number, layers: number, k = 3) {
  return lensAt(layer / Math.max(1, layers))
    .slice()
    .sort((a, b) => b.p - a.p)
    .slice(0, k)
}
