import { describe, expect, test } from 'vitest'
import { KEY_INDEX, MAP_CONTEXTS, MAP_WORDS, mapLayout, nearestTo, project } from './meaningMap'

const WIDTHS = [260, 280, 309, 343, 400, 440, 479, 519, 559, 560, 600, 700, 860, 1000]

describe('карта смыслов: соседи в экранных координатах совпадают с группами', () => {
  for (const width of WIDTHS) {
    test(`ширина ${width}`, () => {
      const l = mapLayout(width)
      const pts = MAP_WORDS.map((w) => project(l, w))
      MAP_WORDS.forEach((w, i) => {
        if (w.group === 'key') return
        for (const j of nearestTo(pts, pts[i], [i])) expect(MAP_WORDS[j].group, `${w.w} → ${MAP_WORDS[j].w}`).toBe(w.group)
      })
      // «ключ» без контекста лежит между двумя своими смыслами
      const keyGroups = nearestTo(pts, pts[KEY_INDEX], [KEY_INDEX]).map((j) => MAP_WORDS[j].group)
      expect(keyGroups).toContain('home')
      expect(keyGroups).toContain('water')
      // после слоёв вектор «ключа» оказывается внутри нужной группы
      for (const c of Object.values(MAP_CONTEXTS)) {
        for (const j of nearestTo(pts, project(l, c), [KEY_INDEX])) expect(MAP_WORDS[j].group, c.phrase).toBe(c.group)
      }
    })
  }
})
