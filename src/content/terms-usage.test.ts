import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { findTerm } from './glossary'

/** Все .tsx под src (кроме тестов). */
function tsxFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) tsxFiles(p, out)
    else if (p.endsWith('.tsx')) out.push(p)
  }
  return out
}

describe('термины в тексте сайта', () => {
  it('каждый <Term t="…"> находится в глоссарии (иначе у термина нет карточки)', () => {
    const srcDir = decodeURIComponent(new URL('..', import.meta.url).pathname)
    const used = new Map<string, string>()
    for (const file of tsxFiles(srcDir)) {
      const text = readFileSync(file, 'utf8')
      for (const m of text.matchAll(/<Term\s+t=\{?(?:"([^"]+)"|'([^']+)')/g)) {
        const t = m[1] ?? m[2]
        if (!used.has(t)) used.set(t, file.slice(srcDir.length))
      }
    }
    expect(used.size).toBeGreaterThan(200)
    const missing = [...used].filter(([t]) => !findTerm(t)).map(([t, f]) => `${t} (${f})`)
    expect(missing).toEqual([])
  })
})
