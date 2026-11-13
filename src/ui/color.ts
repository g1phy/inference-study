/**
 * Цвета по ролям (DESIGN §3). Всё возвращает CSS-строки на переменных,
 * поэтому тема переключается без ререндера и работает на поддереве.
 *
 *   cssVar('prefill')       → 'var(--c-prefill)'
 *   tint('decode')          → 'color-mix(in oklab, var(--c-decode) 14%, transparent)'
 *   mix('q', 40)            → непрозрачная смесь с --surface (для ячеек с текстом)
 *   seqColor(0.62)          → цвет последовательной шкалы heatmap
 *   divColor(-0.3)          → цвет расходящейся шкалы (−1..1)
 */

/** Сущности данных: один цвет на всём сайте. */
export type DataRole = 'prefill' | 'decode' | 'q' | 'k' | 'v' | 'tool' | 'reasoning' | 'special'
/** Статусы: всегда вместе с иконкой и подписью. */
export type StatusRole = 'good' | 'warn' | 'bad'
/** Нейтральные токены и акцент (акцент только для интерфейса). */
export type NeutralRole =
  | 'accent'
  | 'ink'
  | 'ink-2'
  | 'ink-3'
  | 'line'
  | 'line-strong'
  | 'grid'
  | 'bg'
  | 'surface'
  | 'surface-2'
  | 'surface-3'
export type ColorRole = DataRole | StatusRole | NeutralRole

export const DATA_ROLES: readonly DataRole[] = ['prefill', 'decode', 'q', 'k', 'v', 'tool', 'reasoning', 'special']
const STATUS_ROLES: readonly string[] = ['good', 'warn', 'bad']
const NEUTRAL_ROLES: readonly string[] = [
  'accent',
  'ink',
  'ink-2',
  'ink-3',
  'line',
  'line-strong',
  'grid',
  'bg',
  'surface',
  'surface-2',
  'surface-3',
]

/** Русские подписи сущностей (для легенд). */
export const ROLE_LABELS: Record<DataRole, string> = {
  prefill: 'Prefill',
  decode: 'Decode',
  q: 'Query',
  k: 'Key',
  v: 'Value',
  tool: 'Tool call',
  reasoning: 'Рассуждение',
  special: 'Спецтокен',
}

export function isColorRole(value: string): value is ColorRole {
  return (DATA_ROLES as readonly string[]).includes(value) || STATUS_ROLES.includes(value) || NEUTRAL_ROLES.includes(value)
}

/** CSS-переменная роли: cssVar('k') → 'var(--c-k)', cssVar('good') → 'var(--good)'. */
export function cssVar(role: ColorRole): string {
  return (DATA_ROLES as readonly string[]).includes(role) ? `var(--c-${role})` : `var(--${role})`
}

/** Роль или произвольный CSS-цвет → CSS-цвет. resolveColor('prefill') → 'var(--c-prefill)'. */
export function resolveColor(color: string | undefined, fallback: ColorRole = 'prefill'): string {
  if (!color) return cssVar(fallback)
  return isColorRole(color) ? cssVar(color) : color
}

/** Мягкий полупрозрачный фон роли (DESIGN §3.3): tint('prefill', 14). */
export function tint(role: ColorRole | string, pct = 14): string {
  return `color-mix(in oklab, ${resolveColor(role)} ${clampPct(pct)}%, transparent)`
}

/** Непрозрачная смесь роли с поверхностью: для ячеек, на которых лежит текст цвета --ink. */
export function mix(role: ColorRole | string, pct: number, base = 'var(--surface)'): string {
  return `color-mix(in oklab, ${resolveColor(role)} ${clampPct(pct)}%, ${base})`
}

function clampPct(p: number): number {
  return Math.round(Math.min(100, Math.max(0, p)) * 10) / 10
}

const clamp01 = (t: number) => (Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0)

/*
 * Пары соседних ступеней, где светлый текст сменяется тёмным (в любой из тем): seq 3|4 (светлая тема)
 * и 4|5 (тёмная), div 1|2 и 6|7 (обе). Внутри такой пары заливка не смешивается, а встаёт на ближнюю
 * ступень: текст подобран под ступень, а на смеси двух ступеней он давал 4,0:1 вместо AA.
 * В остальных парах у обеих ступеней один цвет текста, и плавная смесь безопасна.
 */
const INK_FLIP: Record<string, readonly number[]> = { seq: [3, 4], div: [1, 6] }

function rampIndex(prefix: string, steps: number, t: number): { i: number; f: number; snap: number | null } {
  const p = clamp01(t) * (steps - 1)
  const i = Math.min(Math.floor(p), steps - 1)
  const f = p - i
  if (f < 0.02 || i >= steps - 1) return { i, f: 0, snap: i }
  if (f > 0.98) return { i, f: 1, snap: i + 1 }
  if (INK_FLIP[prefix]?.includes(i)) return { i, f, snap: f < 0.5 ? i : i + 1 }
  return { i, f, snap: null }
}

function rampColor(prefix: string, steps: number, t: number): string {
  const { i, f, snap } = rampIndex(prefix, steps, t)
  if (snap !== null) return `var(--${prefix}-${snap})`
  return `color-mix(in oklab, var(--${prefix}-${i}), var(--${prefix}-${i + 1}) ${Math.round(f * 100)}%)`
}

function rampInk(prefix: string, steps: number, t: number): string {
  const { i, f, snap } = rampIndex(prefix, steps, t)
  return `var(--${prefix}-ink-${snap ?? (f < 0.5 ? i : i + 1)})`
}

export const SEQ_STEPS = 8
export const DIV_STEPS = 9

/** Последовательная шкала (одна синяя гамма): t ∈ [0, 1]. */
export function seqColor(t: number): string {
  return rampColor('seq', SEQ_STEPS, t)
}

/** Цвет текста поверх seqColor(t) с достаточным контрастом в обеих темах. */
export function seqInk(t: number): string {
  return rampInk('seq', SEQ_STEPS, t)
}

/** Расходящаяся шкала: t ∈ [−1, 1]; отрицательные красные, ноль серый, положительные синие. */
export function divColor(t: number): string {
  return rampColor('div', DIV_STEPS, (Math.min(1, Math.max(-1, t || 0)) + 1) / 2)
}

export function divInk(t: number): string {
  return rampInk('div', DIV_STEPS, (Math.min(1, Math.max(-1, t || 0)) + 1) / 2)
}

/** CSS-градиент шкалы для легенд: legendGradient('seq'). */
export function legendGradient(kind: 'seq' | 'diverging', direction = '90deg'): string {
  const n = kind === 'seq' ? SEQ_STEPS : DIV_STEPS
  const prefix = kind === 'seq' ? 'seq' : 'div'
  const stops = Array.from({ length: n }, (_, i) => `var(--${prefix}-${i})`)
  return `linear-gradient(${direction}, ${stops.join(', ')})`
}
