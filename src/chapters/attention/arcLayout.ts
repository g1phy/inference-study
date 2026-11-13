/**
 * Геометрия дуг внимания для ArcDiagram: путь каждой дуги, место её подписи и выбор подписей без наложений.
 * Чистые функции, чтобы раскладку можно было проверить тестом без DOM.
 */

/** Подпись процента: JetBrains Mono 11,5 px, около 7 px на знак, плюс обводка и зазор по 4 px с боков. */
const CHAR_W = 7
const LABEL_PAD = 8
/** Подпись петли на себя стоит на столько px правее центра чипа (у правого края строки левее). */
export const LOOP_DX = 34
/** Базовая линия самой высокой подписи: выше неё прокрутка срезала бы верх цифр. */
export const LABEL_MIN_Y = 14
/** Подпись стоит на столько px выше вершины своей дуги. */
const LABEL_LIFT = 6

export interface Box {
  l: number
  r: number
  t: number
  b: number
}

export const labelWidth = (text: string): number => text.length * CHAR_W + LABEL_PAD

export function labelBox(text: string, x: number, y: number): Box {
  const w = labelWidth(text)
  return { l: x - w / 2, r: x + w / 2, t: y - 11, b: y + 4 }
}

/** Подпись петли справа от чипа, если справа ещё хватает строки. */
export const loopLabelOnRight = (xf: number, width: number): boolean => xf + 52 <= width

export interface ArcGeom {
  j: number
  w: number
  /** Путь SVG. */
  d: string
  /** Точка подписи: центр текста по x, базовая линия по y. */
  lx: number
  ly: number
}

/** Путь дуги от focus к j и точка её подписи; у петли на себя подпись сбоку. */
export function arcGeom(xs: readonly number[], focus: number, j: number, w: number, height: number, width: number): ArcGeom {
  const H = height
  const xf = xs[focus] ?? 0
  if (j === focus) {
    return {
      j,
      w,
      d: `M ${xf - 7} ${H - 2} C ${xf - 18} ${H - 36}, ${xf + 18} ${H - 36}, ${xf + 7} ${H - 2}`,
      // подпись петли справа, а у правого края строки слева (иначе её срежет прокрутка)
      lx: loopLabelOnRight(xf, width) ? xf + LOOP_DX : xf - LOOP_DX,
      ly: H - 22,
    }
  }
  const xj = xs[j] ?? 0
  const dist = Math.abs(xf - xj)
  // вершина кривой Безье на 0,75·h выше концов; у самой высокой дуги над вершиной остаётся место для подписи
  const h = Math.min((H - 2 - LABEL_MIN_Y - LABEL_LIFT) / 0.75, 26 + dist * 0.5)
  return {
    j,
    w,
    d: `M ${xf} ${H - 2} C ${xf} ${H - 2 - h}, ${xj} ${H - 2 - h}, ${xj} ${H - 2}`,
    lx: (xf + xj) / 2,
    ly: H - 2 - 0.75 * h - LABEL_LIFT,
  }
}

/**
 * Какие дуги подписать: из limit самых толстых, по убыванию веса. Подпись, которая налезла бы на уже
 * поставленную, пропускается (длинные дуги упираются в одну высоту, короткие соседние стоят тесно).
 * Если все веса равны, хватает одной подписи: одинаковые проценты иначе слились бы в одну строку.
 */
export function pickLabels(arcs: readonly ArcGeom[], limit: number, text: (w: number) => string): Set<number> {
  const out = new Set<number>()
  if (arcs.length === 0) return out
  const ws = arcs.map((a) => a.w)
  const uniform = arcs.length > 1 && Math.max(...ws) - Math.min(...ws) < 1e-6
  const ranked = [...arcs].sort((a, b) => b.w - a.w).slice(0, uniform ? 1 : limit)
  const placed: Box[] = []
  for (const a of ranked) {
    const b = labelBox(text(a.w), a.lx, a.ly)
    if (placed.some((o) => o.l < b.r && b.l < o.r && o.t < b.b && b.t < o.b)) continue
    placed.push(b)
    out.add(a.j)
  }
  return out
}
