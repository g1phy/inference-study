import { useEffect, useState } from 'react'

/**
 * Фокус внутри элемента с этим id (обычно фигуры). Живые регионы фигур озвучиваются, только пока читатель
 * работает с фигурой: иначе скринридер читал бы тексты далёких фигур, пока читатель печатает запрос в симуляторе.
 * Фигура получает фокус и по клику (tabIndex = -1), так что мышь тоже включает озвучку.
 */
export function useFocusWithin(id: string): boolean {
  const [inside, setInside] = useState(false)
  useEffect(() => {
    const el = document.getElementById(id)
    if (!el) return
    const onIn = () => setInside(true)
    const onOut = (e: FocusEvent) => {
      const next = e.relatedTarget as Node | null
      if (!next || !el.contains(next)) setInside(false)
    }
    el.addEventListener('focusin', onIn)
    el.addEventListener('focusout', onOut)
    return () => {
      el.removeEventListener('focusin', onIn)
      el.removeEventListener('focusout', onOut)
    }
  }, [id])
  return inside
}
