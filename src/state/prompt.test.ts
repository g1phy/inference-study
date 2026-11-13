import { describe, expect, test } from 'vitest'
import { buildRun } from '@/engine/flow'
import { SCENARIOS, longDocPatch, scenarioPatch } from '@/engine/scenarios'
import { loadQwen3ForTests } from '@/engine/tokenizer/testing'
import type { FlowSettings } from '@/engine/types'
import { readerPromptTokens } from './prompt'
import { DEFAULT_SETTINGS } from './settings'

describe('общий промпт читателя (@/state/prompt)', () => {
  test('столько же токенов, сколько в симуляторе, во всех вкладках (с tools тоже)', () => {
    const tk = loadQwen3ForTests()
    const tabs: Array<[string, FlowSettings]> = [
      ...SCENARIOS.map((sc): [string, FlowSettings] => [sc.id, { ...DEFAULT_SETTINGS, ...scenarioPatch(sc.id) }]),
      ['long', { ...DEFAULT_SETTINGS, ...longDocPatch() }],
    ]
    for (const [id, s] of tabs) {
      const n = readerPromptTokens(tk, s.systemText, s.userText, s.thinking, s.tools).length
      expect(n, id).toBe(buildRun(s, tk).promptTokens.length)
    }
  })

  test('tools меняют ключ кэша: описание инструмента добавляет больше сотни токенов', () => {
    const tk = loadQwen3ForTests()
    const s: FlowSettings = { ...DEFAULT_SETTINGS, ...scenarioPatch('tool') }
    const withTools = readerPromptTokens(tk, s.systemText, s.userText, s.thinking, true).length
    const without = readerPromptTokens(tk, s.systemText, s.userText, s.thinking, false).length
    expect(withTools - without).toBeGreaterThan(100)
  })
})
