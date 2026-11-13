import { useState } from 'react'

/**
 * Значение селектора фигуры (модель или GPU), которое следует за симулятором: пока читатель не выбрал в фигуре
 * своё, фигура показывает то же, что симулятор, и меняется вместе с ним. Свой выбор держится, только пока он
 * отличается от симулятора: совпали (читатель выбрал то же или симулятор пришёл к тому же), и фигура снова следует.
 * Поэтому ссылка «вернуть» видна ровно тогда, когда фигура и симулятор расходятся.
 * Фигуры монтируются в фоне вскоре после загрузки, задолго до того, как читатель что-то поменяет в симуляторе:
 * одного начального значения из настроек мало.
 */
export function useFollowSim(simId: string): { id: string; set: (id: string) => void; reset: () => void; overridden: boolean } {
  const [override, setOverride] = useState<string | null>(null)
  // setState во время рендера: React сразу перерисует компонент уже без устаревшего выбора
  if (override !== null && override === simId) setOverride(null)
  const overridden = override !== null && override !== simId
  return {
    id: overridden ? override : simId,
    set: (id) => setOverride(id === simId ? null : id),
    reset: () => setOverride(null),
    overridden,
  }
}

/** Ссылка-кнопка под селектором: вернуть значение из симулятора (как «вернуть длину вашего запроса» в CacheVsNoCache). */
export function BackToSim({ what, name, onClick }: { what: string; name: string; onClick: () => void }) {
  return (
    <button type="button" className="text-left text-accent underline-offset-2 hover:underline" onClick={onClick}>
      вернуть {what} из симулятора ({name})
    </button>
  )
}
