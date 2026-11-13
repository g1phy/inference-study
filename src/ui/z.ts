/**
 * Z-index шкала (DESIGN §5). Других значений z-index на сайте нет.
 * В Tailwind: z-40 / z-50 / z-60. В inline-стилях: zIndex: Z.popover.
 */
export const Z = {
  nav: 40,
  popover: 50,
  tooltip: 50,
  modal: 60,
} as const

export type ZLayer = keyof typeof Z
