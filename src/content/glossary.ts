import raw from './glossary.json'

export { CATEGORY_LABEL, categoryLabel } from './glossary-categories'

/**
 * Глоссарий (копия research/glossary.json). Используется компонентом <Term> и главой glossary.
 * Поиск по term и aliases без учёта регистра (ё = е, текст в скобках у term тоже ищется).
 */
export interface GlossaryEntry {
  /** Каноническое имя: «KV cache», «LLM (Large Language Model)». */
  term: string
  aliases: string[]
  /** Определение по-русски, 1-3 предложения. */
  ru: string
  /** Почему это важно для инференса. */
  why: string
  /** Категория: «основы», «модель», «кэш», … */
  category: string
  /** id главы, где термин раскрыт (если задан). */
  chapter?: string
  /**
   * true, если ни одна глава термин не разбирает и `chapter` указывает только на ближайшую тему:
   * тогда подсказка пишет «Ближе всего к главе N», а не «Подробно в главе N».
   */
  mentionOnly?: boolean
}

export const GLOSSARY: readonly GlossaryEntry[] = raw as GlossaryEntry[]

export const GLOSSARY_CATEGORIES: readonly string[] = [...new Set(GLOSSARY.map((e) => e.category))]

export function normalizeTerm(s: string): string {
  return s.toLowerCase().replace(/ё/g, 'е').replace(/[\s ]+/g, ' ').trim()
}

let index: Map<string, GlossaryEntry> | null = null

function buildIndex(): Map<string, GlossaryEntry> {
  const map = new Map<string, GlossaryEntry>()
  const add = (key: string, e: GlossaryEntry) => {
    const k = normalizeTerm(key)
    if (k && !map.has(k)) map.set(k, e)
  }
  // сначала канонические имена, потом алиасы: при конфликте выигрывает term.
  // Части имени тоже канонические: «ISL / OSL» находится по «ISL» и «OSL»,
  // «Router vs load balancer» по «router», «Worker (воркер)» по «воркер».
  for (const e of GLOSSARY) {
    add(e.term, e)
    const m = /^(.*?)\s*\((.+)\)\s*$/.exec(e.term)
    if (m) {
      add(m[1], e)
      add(m[2], e)
    }
    for (const part of (m ? m[1] : e.term).split(/\s*\/\s*|\s+vs\s+/)) add(part, e)
  }
  for (const e of GLOSSARY) for (const a of e.aliases ?? []) add(a, e)
  return map
}

/** Найти термин: findTerm('kv cache'), findTerm('TTFT'), findTerm('веса'). */
export function findTerm(t: string): GlossaryEntry | undefined {
  if (!index) index = buildIndex()
  return index.get(normalizeTerm(t))
}
