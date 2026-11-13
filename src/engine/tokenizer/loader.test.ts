import { afterEach, describe, expect, it, vi } from 'vitest'
import { getLoadedTokenizer, getTokenizerState, loadTokenizer, resetTokenizerCacheForTests, subscribeTokenizer, tokenizerUrl } from './loader'
import { readProjectFile } from './testing'

describe('loadTokenizer: ленивый fetch и кэш на модуль', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    resetTokenizerCacheForTests()
  })

  it('качает словарь один раз и уведомляет подписчиков', async () => {
    const body = readProjectFile('public/tokenizer/qwen3.json')
    const fetchMock = vi.fn(async (url: string) => {
      expect(url).toBe(tokenizerUrl())
      return new Response(body, { status: 200, headers: { 'content-type': 'application/json' } })
    })
    vi.stubGlobal('fetch', fetchMock)
    const seen: string[] = []
    const unsub = subscribeTokenizer(() => seen.push(getTokenizerState().status))
    expect(getTokenizerState().status).toBe('idle')
    const [a, b] = await Promise.all([loadTokenizer(), loadTokenizer()])
    expect(a).toBe(b)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(seen).toEqual(['loading', 'ready'])
    expect(getLoadedTokenizer()).toBe(a)
    // notes-tokens §3.6: Qwen3 'Привет, мир!' = Пр · ив · ет · , · ␣мир · !
    expect(a.encode('Привет, мир!').map((t) => t.id)).toEqual([53645, 26991, 8178, 11, 137144, 0])
    unsub()
  })

  it('URL строится от BASE_URL', () => {
    expect(tokenizerUrl()).toMatch(/tokenizer\/qwen3\.json$/)
  })

  it('ошибка HTTP → status error, следующий вызов пробует снова', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('nope', { status: 404 })))
    await expect(loadTokenizer()).rejects.toThrow(/HTTP 404/)
    expect(getTokenizerState().status).toBe('error')
    expect(getTokenizerState().error).toMatch(/404/)
    const body = readProjectFile('public/tokenizer/qwen3.json')
    vi.stubGlobal('fetch', vi.fn(async () => new Response(body, { status: 200 })))
    const t = await loadTokenizer()
    expect(t.vocabSize).toBe(151669)
    expect(getTokenizerState().status).toBe('ready')
  })
})
