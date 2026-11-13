import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  DEFAULT_SETTINGS,
  SETTINGS_COMMIT_DELAY,
  flushCommittedSettings,
  getCommittedSettings,
  getSettings,
  patchSettings,
  resetSettings,
  sanitizeSettings,
} from './settings'

describe('sanitizeSettings', () => {
  it('мусор превращается в значения по умолчанию', () => {
    expect(sanitizeSettings(null)).toEqual(DEFAULT_SETTINGS)
    expect(sanitizeSettings('oops')).toEqual(DEFAULT_SETTINGS)
    expect(sanitizeSettings({ scenarioId: 'nope', deployment: 'mesh', thinking: 'yes', temperature: 'hot' })).toEqual(DEFAULT_SETTINGS)
  })
  it('корректные поля сохраняются, неверные откатываются по одному', () => {
    const s = sanitizeSettings({ scenarioId: 'json', json: true, modelId: 'qwen3-32b', gpuId: 'B200!', temperature: 1.234 })
    expect(s.scenarioId).toBe('json')
    expect(s.json).toBe(true)
    expect(s.modelId).toBe('qwen3-32b')
    expect(s.gpuId).toBe(DEFAULT_SETTINGS.gpuId)
    expect(s.temperature).toBe(1.23)
  })
  it('температура зажата в [0, 2], текст обрезан', () => {
    expect(sanitizeSettings({ temperature: 9 }).temperature).toBe(2)
    expect(sanitizeSettings({ temperature: -1 }).temperature).toBe(0)
    expect(sanitizeSettings({ userText: 'x'.repeat(5000) }).userText.length).toBe(4000)
    expect(sanitizeSettings({ systemText: '' }).systemText).toBe('')
  })
})

describe('принятые настройки (текст с задержкой)', () => {
  afterEach(() => {
    resetSettings()
    flushCommittedSettings()
    vi.useRealTimers()
  })

  it('текст доходит после паузы в наборе', () => {
    vi.useFakeTimers()
    patchSettings({ userText: 'П' })
    patchSettings({ userText: 'Пр' })
    expect(getSettings().userText).toBe('Пр')
    expect(getCommittedSettings().userText).toBe(DEFAULT_SETTINGS.userText)
    vi.advanceTimersByTime(SETTINGS_COMMIT_DELAY - 50)
    patchSettings({ userText: 'При' })
    vi.advanceTimersByTime(SETTINGS_COMMIT_DELAY - 50)
    expect(getCommittedSettings().userText).toBe(DEFAULT_SETTINGS.userText)
    vi.advanceTimersByTime(60)
    expect(getCommittedSettings().userText).toBe('При')
  })

  it('смена не текстового поля применяет всё сразу', () => {
    vi.useFakeTimers()
    patchSettings({ userText: 'Новый вопрос' })
    patchSettings({ thinking: !DEFAULT_SETTINGS.thinking })
    expect(getCommittedSettings().thinking).toBe(!DEFAULT_SETTINGS.thinking)
    expect(getCommittedSettings().userText).toBe('Новый вопрос')
  })
})
