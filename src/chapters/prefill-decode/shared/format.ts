import { fmt } from '@/ui/fmt'

/** «16,4 ГБ» → ['16,4', 'ГБ'] (между числом и единицей у fmt неразрывный пробел). */
export function valueUnit(s: string): [string, string] {
  const i = s.search(/\s\S+$/)
  return i < 0 ? [s, ''] : [s.slice(0, i), s.slice(i + 1)]
}

/** «в 43 раза», «в 295 раз»: число и слово «раз» в нужной форме. */
export function times(n: number): string {
  const r = Math.round(n)
  return `${fmt.int(r)} ${fmt.plural(r, ['раз', 'раза', 'раз'])}`
}

/**
 * Длительность для Stat: число и единица. От минуты fmt.ms пишет «2 мин 1 с», и valueUnit резал бы это
 * на «2 мин 1» и «с»; здесь от минуты показываем целые секунды.
 */
export function durationValueUnit(ms: number): [string, string] {
  if (Math.abs(ms) >= 60_000) return [fmt.num(ms / 1000, 0), 'с']
  return valueUnit(fmt.ms(ms))
}
