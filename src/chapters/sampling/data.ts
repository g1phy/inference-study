/**
 * Условные логиты для контекста «Столица Франции —». Кандидаты показаны целыми словами (упрощение:
 * в словаре Qwen3 « Париж» это три токена). Хвост: остальные строки логитов Qwen3 с одинаковым низким
 * логитом, чтобы вместе с кандидатами получилась вся ширина 151 936.
 */
import { QWEN3_LOGITS_WIDTH } from '@/engine/tokenizer'
import type { HeadCandidate, TailGroup } from './dist'

export const CONTEXT = 'Столица Франции —'

export const HEAD: HeadCandidate[] = [
  { text: ' Париж', logit: 6.2 },
  { text: ' город', logit: 4.1 },
  { text: ' это', logit: 3.6 },
  { text: ' Марсель', logit: 2.3 },
  { text: ' конечно', logit: 1.9 },
  { text: ' Лион', logit: 1.6 },
  { text: ' прекрасный', logit: 1.2 },
  { text: ' один', logit: 0.8 },
  { text: ' самый', logit: 0.5 },
  { text: ' Берлин', logit: 0.3 },
  { text: ' большой', logit: 0.1 },
  { text: ' Москва', logit: -0.2 },
]

export const TAIL: TailGroup = { count: QWEN3_LOGITS_WIDTH - HEAD.length, logit: -10 }

/** Что показывать, если выпал токен из хвоста (иллюстрация: случайные строки словаря). */
export const TAIL_EXAMPLES = [' Kubernetes', 'ž', '_REF', ' ясно', 'ться', ' });', '数', ' Bundesliga', ' квадрат', ' Ebenso', '.npy', ' ты']
