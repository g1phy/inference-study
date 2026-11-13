/**
 * Поиск по глоссарию для главы 28: по term, aliases, ru и why.
 *
 * Нормализация: регистр, ё/е, кэш/кеш и хэш/хеш, ведущие дефисы флагов и «-», «_» как пробел
 * («--kv-cache-dtype» находит kv_cache_dtype). Ранжирование: точное совпадение с названием или синонимом,
 * начало, вхождение; затем совпадение всех слов запроса с началами слов названия с учётом окончаний
 * («квантизации», «холодного старта», «реплики», «кэш префикс»); затем текст определения.
 */
import type { GlossaryEntry } from '@/content/glossary'

/** Посимвольная замена (длина строки не меняется: на этом держится подсветка). */
function canon(s: string): string {
  return s
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/кеш/g, 'кэш')
    .replace(/хеш/g, 'хэш')
    .replace(/квантовани/g, 'квантизаци')
    .replace(/[-_]/g, ' ')
}

export function norm(s: string): string {
  return canon(s.trim().replace(/^[-–]+/, '')).replace(/\s+/g, ' ').trim()
}

const WORD_SPLIT = /[^a-zа-я0-9]+/
const RU_ENDING = /(ями|ами|ого|его|ому|ему|ыми|ими|ией|иях|иям|ах|ях|ов|ев|ей|ой|ий|ый|ая|яя|ое|ее|ые|ие|ую|юю|ам|ям|ом|ем|ия|а|я|ы|и|у|ю|е|о|ь)$/

/** Лёгкий стеммер: срезает одно окончание (русское или английское -s/-es/-ing/-e), основа не короче трёх букв. */
export function stem(w: string): string {
  if (/[а-я]/.test(w)) {
    if (w.length < 4) return w
    const m = RU_ENDING.exec(w)
    if (!m) return w
    const s = w.slice(0, w.length - m[1].length)
    return s.length >= 3 ? s : w
  }
  if (/^[a-z]/.test(w) && w.length > 4) return w.replace(/(ing|es|s|e)$/, '')
  return w
}

const words = (s: string) => s.split(WORD_SPLIT).filter(Boolean)
/** Служебные слова флагов: «--enable-prefix-caching», «--speculative-config» ищутся по сути (prefix caching, speculative). */
const FLAG_NOISE = new Set(['enable', 'disable', 'config', 'use'])

export interface Hit {
  entry: GlossaryEntry
  index: number
  score: number
}

interface Prepared {
  term: string
  aliases: string[]
  parts: string[]
  head: string[]
  names: string[][]
  ru: string
  why: string
  text: string[]
}

const cache = new WeakMap<GlossaryEntry, Prepared>()

function prepare(e: GlossaryEntry): Prepared {
  let p = cache.get(e)
  if (!p) {
    const term = norm(e.term)
    const aliases = e.aliases.map(norm)
    // «KV cache» внутри «Tiered KV cache (…)», части в скобках и через «/»
    const parts = e.term
      .split(/[()/]/)
      .map(norm)
      .filter(Boolean)
    const names = [term, ...aliases, ...parts].map(words)
    const ru = norm(e.ru)
    const why = norm(e.why)
    p = { term, aliases, parts, head: names.flat(), names, ru, why, text: words(`${ru} ${why}`) }
    cache.set(e, p)
  }
  return p
}

export function scoreEntry(e: GlossaryEntry, q: string): number {
  if (!q) return 1
  const p = prepare(e)
  if (p.term === q || p.aliases.includes(q)) return 100
  if (p.parts.includes(q)) return 90
  if (p.term.startsWith(q)) return 70
  if (p.aliases.some((a) => a.startsWith(q))) return 60
  if (p.term.includes(q)) return 50
  if (p.aliases.some((a) => a.includes(q))) return 40
  const all = words(q)
  const meaningful = all.filter((w) => !FLAG_NOISE.has(w))
  const stems = (meaningful.length > 0 ? meaningful : all).map(stem)
  if (stems.length === 0) return 0
  const inHead = (st: string) => p.head.some((w) => w.startsWith(st))
  if (stems.every(inHead)) {
    // запрос покрывает одно название целиком (столько же слов): выше, чем часть длинного синонима
    const whole = p.names.some((ws) => ws.length === stems.length && stems.every((st) => ws.some((w) => w.startsWith(st))))
    return whole ? 37 : 35
  }
  if (p.ru.includes(q)) return 20
  if (p.why.includes(q)) return 12
  if (stems.every((st) => inHead(st) || p.text.some((w) => w.startsWith(st)))) return 8
  return 0
}

export function searchGlossary(entries: readonly GlossaryEntry[], query: string, category: string | null): Hit[] {
  const q = norm(query)
  const hits: Hit[] = []
  entries.forEach((entry, index) => {
    if (category && entry.category !== category) return
    const score = scoreEntry(entry, q)
    if (score > 0) hits.push({ entry, index, score })
  })
  if (q) hits.sort((a, b) => b.score - a.score || a.index - b.index)
  return hits
}

/**
 * Отрезки текста с отметкой совпадения (для подсветки). Сравнение идёт по canon(text): та же длина,
 * что у исходной строки. Если запроса целиком в тексте нет, подсвечиваются начала слов по основам.
 */
export function highlightParts(text: string, query: string): Array<{ text: string; hit: boolean }> {
  const q = norm(query)
  if (!q) return [{ text, hit: false }]
  const low = canon(text)
  if (low.length !== text.length) return [{ text, hit: false }]
  const ranges: Array<[number, number]> = []
  for (let j = low.indexOf(q); j >= 0; j = low.indexOf(q, j + q.length)) ranges.push([j, j + q.length])
  if (ranges.length === 0) {
    const stems = words(q).map(stem).filter((s) => s.length >= 3)
    const re = /[a-zа-я0-9]+/g
    for (let m = re.exec(low); m; m = re.exec(low)) {
      const st = stems.find((s) => m![0].startsWith(s))
      if (st) ranges.push([m.index, m.index + st.length])
    }
  }
  if (ranges.length === 0) return [{ text, hit: false }]
  const out: Array<{ text: string; hit: boolean }> = []
  let i = 0
  for (const [a, b] of ranges) {
    if (a > i) out.push({ text: text.slice(i, a), hit: false })
    out.push({ text: text.slice(a, b), hit: true })
    i = b
  }
  if (i < text.length) out.push({ text: text.slice(i), hit: false })
  return out
}

/** Синонимы, которые на карточке повторяют само название (другой регистр, часть «A / B»): для поиска нужны, на карточке шум. */
export function visibleAliases(e: GlossaryEntry): string[] {
  const p = prepare(e)
  const own = new Set([p.term, ...p.parts, ...e.term.split(/\s+vs\.?\s+/i).map(norm)])
  return e.aliases.filter((a) => !own.has(norm(a)))
}

export function slug(term: string): string {
  return term
    .toLowerCase()
    .replace(/ё/g, 'е')
    .replace(/[^a-z0-9а-я]+/g, '-')
    .replace(/^-+|-+$/g, '')
}
