import { useCallback, type CSSProperties } from 'react'
import { cn } from './cn'

export interface SkeletonProps {
  width?: number | string
  height?: number | string
  radius?: number | string
  /** Несколько строк текста (последняя короче). */
  lines?: number
  className?: string
  /** Подпись для скринридеров. */
  label?: string
  /** Без перелива: блок ждёт, пока фигура подъедет к экрану, а не загружается прямо сейчас. */
  still?: boolean
}

/*
 * Перелив идёт только на экране. Заглушки, которые ждут подъезда фигуры к экрану, иначе
 * переливались бы вечно за экраном и гоняли пересчёт стилей (и батарею телефона).
 * Один общий IntersectionObserver на все заглушки ставит data-offscreen, CSS ставит анимацию на паузу.
 */
let io: IntersectionObserver | null = null
function observer(): IntersectionObserver | null {
  if (io || typeof IntersectionObserver === 'undefined') return io
  io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) e.target.removeAttribute('data-offscreen')
      else e.target.setAttribute('data-offscreen', '')
    }
  })
  return io
}

function useOffscreenPause() {
  return useCallback((el: HTMLElement | null) => {
    const obs = observer()
    if (!obs || !el) return
    el.setAttribute('data-offscreen', '')
    obs.observe(el)
    return () => obs.unobserve(el)
  }, [])
}

/** Блок-заглушка в форме будущего содержимого. Мягкий перелив выключается при reduced motion и за экраном. */
export function Skeleton({ width = '100%', height = 14, radius = 6, lines, className, label = 'Загрузка', still = false }: SkeletonProps) {
  const ref = useOffscreenPause()
  if (lines && lines > 1) {
    return (
      <div ref={ref} role="status" aria-label={label} className={cn('grid gap-2', className)}>
        {Array.from({ length: lines }, (_, i) => (
          <SkeletonBar key={i} still={still} style={{ width: i === lines - 1 ? '62%' : '100%', height, borderRadius: radius }} />
        ))}
      </div>
    )
  }
  return (
    <div ref={ref} role="status" aria-label={label} className={className} style={{ width }}>
      <SkeletonBar still={still} style={{ width: '100%', height, borderRadius: radius }} />
    </div>
  )
}

function SkeletonBar({ style, className, still }: { style?: CSSProperties; className?: string; still?: boolean }) {
  return (
    <div className={cn('relative overflow-hidden bg-surface-3', className)} style={style} aria-hidden>
      {!still && (
        <div
          className="absolute inset-0 animate-shimmer motion-reduce:hidden"
          style={{ background: 'linear-gradient(90deg, transparent, color-mix(in oklab, var(--surface) 55%, transparent), transparent)' }}
        />
      )}
    </div>
  )
}

/** Скелетон ленты токенов (пока грузится токенизатор): чипы разной ширины. */
export function TokenStripSkeleton({
  count = 24,
  className,
  label = 'Загружаем токенизатор',
  still = false,
}: {
  count?: number
  className?: string
  label?: string
  still?: boolean
}) {
  const ref = useOffscreenPause()
  const widths = [38, 22, 56, 30, 44, 18, 64, 26, 34, 48, 20, 40]
  return (
    <div ref={ref} role="status" aria-label={label} className={cn('flex flex-wrap gap-x-[3px] gap-y-1.5', className)}>
      {Array.from({ length: count }, (_, i) => (
        <SkeletonBar key={i} still={still} style={{ width: widths[i % widths.length], height: 24, borderRadius: 6 }} />
      ))}
    </div>
  )
}
