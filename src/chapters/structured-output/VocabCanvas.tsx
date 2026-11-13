import { useEffect, useRef, type PointerEvent, type ReactNode } from 'react'
import { TipCard, pointRect, useTheme, useTip } from '@/ui'

/** Ширина логитов Qwen3: 151 936 строк, из них 151 669 токенов и 267 пустых строк паддинга (SPEC §8.7). */
export const LOGITS_WIDTH = 151936
const SIDE = 390 // 390 × 390 = 152 100 ≥ 151 936

interface Props {
  /** Маска по словарю (длина = числу токенов, 1 = разрешён); null: ещё считается. */
  mask: ArrayLike<boolean | number> | null
  /** Сколько разрешено (для подписи и колец вокруг редких разрешённых). */
  allowed: number | null
  /** Маска выключена: разрешены все токены словаря (строки паддинга остаются серыми). */
  free?: boolean
  /** Число токенов словаря (для режима free). */
  vocabSize?: number
  /** Содержимое подсказки для строки логитов под указателем. */
  describe?: (row: number) => ReactNode
  className?: string
}

function cssColor(el: Element, name: string): [number, number, number] {
  const probe = document.createElement('span')
  probe.style.color = `var(${name})`
  probe.style.display = 'none'
  el.appendChild(probe)
  const rgb = getComputedStyle(probe).color
  probe.remove()
  const m = rgb.match(/[\d.]+/g)
  if (!m) return [128, 128, 128]
  return [Number(m[0]), Number(m[1]), Number(m[2])]
}

/**
 * Весь словарь как квадрат 390 × 390: точка = строка логитов (по порядку id).
 * Разрешённые маской светятся статусом «good», запрещённые бледные; если разрешённых мало, они обведены кругами.
 * Последние 267 строк (паддинг) в масштабе фигуры занимают меньше пикселя: они раскрашены как запрещённые, а сказано о них в подписи.
 */
export function VocabCanvas({ mask, allowed, free = false, vocabSize = 151669, describe, className }: Props) {
  const ref = useRef<HTMLCanvasElement | null>(null)
  const { resolved } = useTheme()
  const tip = useTip({ scopeEl: ref.current })

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const good = cssColor(canvas, '--good')
    const bad = cssColor(canvas, '--bad')
    const surface = cssColor(canvas, '--surface-2')
    const img = ctx.createImageData(SIDE, SIDE)
    const d = img.data
    const blend = (c: [number, number, number], a: number): [number, number, number] => [
      Math.round(surface[0] * (1 - a) + c[0] * a),
      Math.round(surface[1] * (1 - a) + c[1] * a),
      Math.round(surface[2] * (1 - a) + c[2] * a),
    ]
    const off = blend(bad, resolved === 'dark' ? 0.24 : 0.16)
    const on = blend(good, resolved === 'dark' ? 0.8 : 0.72)
    // без маски разрешено всё: заливка спокойнее, чтобы сплошной квадрат не кричал
    const onFree = blend(good, resolved === 'dark' ? 0.5 : 0.42)
    const idle = blend([128, 128, 128], 0.18)
    const n = free ? vocabSize : mask ? mask.length : 0
    for (let i = 0; i < SIDE * SIDE; i++) {
      const o = i * 4
      let c: [number, number, number]
      let alpha = 255
      if (i >= LOGITS_WIDTH) {
        alpha = 0
        c = surface
      } else if (free) c = i < n ? onFree : idle
      else if (!mask) c = idle
      else if (i >= n) c = off // паддинг: строки логитов без токена, маска запрещает их всегда
      else c = mask[i] ? on : off
      d[o] = c[0]
      d[o + 1] = c[1]
      d[o + 2] = c[2]
      d[o + 3] = alpha
    }
    ctx.putImageData(img, 0, 0)
    // мало разрешённых: обводим кругами, иначе их не видно
    if (!free && mask && allowed !== null && allowed > 0 && allowed <= 60) {
      ctx.strokeStyle = `rgb(${good.join(',')})`
      ctx.lineWidth = 2
      for (let i = 0; i < n; i++) {
        if (!mask[i]) continue
        ctx.beginPath()
        ctx.arc((i % SIDE) + 0.5, Math.floor(i / SIDE) + 0.5, 6, 0, Math.PI * 2)
        ctx.stroke()
      }
    }
  }, [mask, allowed, free, vocabSize, resolved])

  const show = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!describe) return
    const r = e.currentTarget.getBoundingClientRect()
    const x = Math.floor(((e.clientX - r.left) / r.width) * SIDE)
    const y = Math.floor(((e.clientY - r.top) / r.height) * SIDE)
    const row = y * SIDE + x
    if (x < 0 || y < 0 || x >= SIDE || y >= SIDE || row >= LOGITS_WIDTH) {
      tip.hide()
      return
    }
    tip.show(pointRect(e.clientX, e.clientY, 4), <TipCard>{describe(row)}</TipCard>)
  }

  return (
    <>
      <canvas
        ref={ref}
        width={SIDE}
        height={SIDE}
        role="img"
        aria-label={
          free
            ? `Маска по всему словарю выключена: разрешены все ${vocabSize} токенов`
            : mask && allowed !== null
              ? `Маска по всему словарю: разрешено ${allowed} из ${LOGITS_WIDTH} строк логитов`
              : 'Маска по всему словарю: считается'
        }
        className={className}
        style={{ imageRendering: 'pixelated' }}
        onPointerMove={describe ? show : undefined}
        onPointerDown={describe ? show : undefined}
        onPointerLeave={describe ? (e) => e.pointerType === 'mouse' && tip.hide() : undefined}
      />
      {tip.node}
    </>
  )
}
