import { describe, expect, it } from 'vitest'
import { CHAPTERS, CHAPTER_IDS, PARTS, chaptersOfPart, isFirstInPart, nextChapter, prevChapter } from './chapters'

describe('реестр глав', () => {
  it('29 глав (hero + 28), номера по порядку, id уникальны', () => {
    expect(CHAPTERS).toHaveLength(29)
    expect(CHAPTERS.map((c) => c.id)).toEqual([...CHAPTER_IDS])
    CHAPTERS.forEach((c, i) => expect(c.number).toBe(i))
    expect(new Set(CHAPTER_IDS).size).toBe(CHAPTER_IDS.length)
  })
  it('у каждой главы есть часть (кроме hero), заголовок, короткое имя и лид', () => {
    for (const c of CHAPTERS) {
      expect(c.part === null).toBe(c.id === 'hero')
      expect(c.title.length).toBeGreaterThan(2)
      expect(c.nav.length).toBeGreaterThan(1)
      expect(c.lead.length).toBeGreaterThan(20)
      expect(c.lead).not.toMatch(/—/)
    }
  })
  it('части покрывают главы 1-28 подряд', () => {
    expect(PARTS).toHaveLength(8)
    const flat = PARTS.flatMap((p) => chaptersOfPart(p.id).map((c) => c.number))
    expect(flat).toEqual(Array.from({ length: 28 }, (_, i) => i + 1))
    expect(isFirstInPart('request')).toBe(true)
    expect(isFirstInPart('template')).toBe(false)
  })
  it('соседи', () => {
    expect(nextChapter('hero')?.id).toBe('request')
    expect(prevChapter('request')?.id).toBe('hero')
    expect(nextChapter('glossary')).toBeUndefined()
  })
})
