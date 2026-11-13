/**
 * Данные иллюстративной 2D-карты смыслов. Координаты подобраны вручную, отдельно для широкой
 * (горизонтальной) и узкой (вертикальной) раскладки: так группы остаются плотными и разделёнными
 * на любой ширине. Соседей считаем в экранных координатах, поэтому пунктир на картинке совпадает
 * с тем, что написано под ней. meaningMap.test.ts проверяет, что у каждого слова, кроме «ключа»,
 * три ближайших соседа из его же группы.
 */

export type MapGroup = 'geo' | 'num' | 'hw' | 'animal' | 'home' | 'water' | 'key'
type XY = readonly [number, number]

export interface MapWord {
  w: string
  group: MapGroup
  /** Координаты 0..1 в широкой и в узкой раскладке. */
  wide: XY
  narrow: XY
  /** С какой стороны от точки подпись (по умолчанию справа). */
  label?: 'left' | 'top'
}

export const MAP_WORDS: readonly MapWord[] = [
  { w: 'Франция', group: 'geo', wide: [0.21, 0.05], narrow: [0.2, 0.03] },
  { w: 'Париж', group: 'geo', wide: [0.08, 0.09], narrow: [0.03, 0.07] },
  { w: 'Германия', group: 'geo', wide: [0.21, 0.15], narrow: [0.2, 0.11] },
  { w: 'Берлин', group: 'geo', wide: [0.07, 0.2], narrow: [0.03, 0.15] },
  { w: 'Рим', group: 'geo', wide: [0.15, 0.27], narrow: [0.17, 0.19] },
  { w: 'сто', group: 'num', wide: [0.86, 0.05], narrow: [0.82, 0.03] },
  { w: 'один', group: 'num', wide: [0.74, 0.09], narrow: [0.64, 0.07] },
  { w: 'два', group: 'num', wide: [0.87, 0.15], narrow: [0.82, 0.11] },
  { w: 'три', group: 'num', wide: [0.75, 0.2], narrow: [0.64, 0.15] },
  { w: '42', group: 'num', wide: [0.86, 0.26], narrow: [0.79, 0.19] },
  { w: 'GPU', group: 'hw', wide: [0.88, 0.74], narrow: [0.64, 0.78] },
  { w: 'диск', group: 'hw', wide: [0.76, 0.79], narrow: [0.82, 0.82] },
  { w: 'память', group: 'hw', wide: [0.84, 0.85], narrow: [0.64, 0.86] },
  { w: 'байт', group: 'hw', wide: [0.92, 0.91], narrow: [0.82, 0.9] },
  { w: 'кэш', group: 'hw', wide: [0.78, 0.94], narrow: [0.64, 0.94] },
  { w: 'кошка', group: 'animal', wide: [0.07, 0.75], narrow: [0.03, 0.78] },
  { w: 'лошадь', group: 'animal', wide: [0.19, 0.81], narrow: [0.2, 0.82] },
  { w: 'собака', group: 'animal', wide: [0.07, 0.87], narrow: [0.03, 0.86] },
  { w: 'мышь', group: 'animal', wide: [0.19, 0.93], narrow: [0.2, 0.9] },
  { w: 'дверь', group: 'home', wide: [0.31, 0.4], narrow: [0.26, 0.4], label: 'left' },
  { w: 'замок', group: 'home', wide: [0.25, 0.47], narrow: [0.18, 0.44], label: 'left' },
  { w: 'окно', group: 'home', wide: [0.32, 0.53], narrow: [0.26, 0.48], label: 'left' },
  { w: 'стена', group: 'home', wide: [0.26, 0.59], narrow: [0.18, 0.52], label: 'left' },
  { w: 'родник', group: 'water', wide: [0.64, 0.39], narrow: [0.7, 0.4] },
  { w: 'ручей', group: 'water', wide: [0.71, 0.45], narrow: [0.78, 0.44] },
  { w: 'река', group: 'water', wide: [0.7, 0.53], narrow: [0.7, 0.48] },
  { w: 'вода', group: 'water', wide: [0.63, 0.58], narrow: [0.78, 0.52] },
  { w: 'ключ', group: 'key', wide: [0.48, 0.49], narrow: [0.48, 0.47], label: 'top' },
]

export const KEY_INDEX = MAP_WORDS.findIndex((w) => w.w === 'ключ')

export type MapContextId = 'door' | 'spring'

/** «Ключ» в предложении: куда сдвигается его вектор после слоёв (внутрь нужной группы). */
export const MAP_CONTEXTS: Record<MapContextId, { group: MapGroup; phrase: string; label: string; wide: XY; narrow: XY }> = {
  // контекст стоит перед «ключом»: в decoder-only модели токен видит только предыдущие токены
  door: { group: 'home', phrase: '«в замке торчал ключ»', label: 'в замке торчал ключ', wide: [0.31, 0.47], narrow: [0.32, 0.46] },
  spring: { group: 'water', phrase: '«вода из ключа»', label: 'вода из ключа', wide: [0.655, 0.485], narrow: [0.64, 0.46] },
}

/** Подписи групп: четыре в углах холста и две рядом со смыслами «ключа». */
export const MAP_GROUP_LABELS: ReadonlyArray<
  | { label: string; corner: 'tl' | 'tr' | 'bl' | 'br' }
  | { label: string; wide: XY; narrow: XY; anchor: 'start' | 'end' }
> = [
  { label: 'страны и города', corner: 'tl' },
  { label: 'числа', corner: 'tr' },
  { label: 'железо', corner: 'br' },
  { label: 'животные', corner: 'bl' },
  { label: 'дом', wide: [0.33, 0.33], narrow: [0.26, 0.355], anchor: 'end' },
  { label: 'реки и родники', wide: [0.6, 0.33], narrow: [0.62, 0.355], anchor: 'start' },
]

export interface MapLayout {
  kind: 'wide' | 'narrow'
  W: number
  H: number
  padX: number
  padY: number
}

/** Раскладка по ширине контейнера: от 560px широкая (холст не выше 0,62 ширины), уже вертикальная. */
export function mapLayout(width: number): MapLayout {
  const W = Math.max(260, Math.round(width))
  if (W >= 560) {
    const w = Math.min(W, 860)
    return { kind: 'wide', W: w, H: Math.round(w * 0.62), padX: 34, padY: 26 }
  }
  return { kind: 'narrow', W, H: Math.min(Math.round(W * 1.5), 660), padX: 14, padY: 22 }
}

export function project(l: MapLayout, p: { wide: XY; narrow: XY }): { x: number; y: number } {
  const [x, y] = p[l.kind]
  return { x: l.padX + x * (l.W - 2 * l.padX), y: l.padY + y * (l.H - 2 * l.padY) }
}

/** k ближайших точек (индексы) к from в экранных координатах. */
export function nearestTo(points: ReadonlyArray<{ x: number; y: number }>, from: { x: number; y: number }, exclude: readonly number[], k = 3): number[] {
  return points
    .map((q, j) => ({ j, d: Math.hypot(q.x - from.x, q.y - from.y) }))
    .filter((e) => !exclude.includes(e.j))
    .sort((a, b) => a.d - b.d)
    .slice(0, k)
    .map((e) => e.j)
}
