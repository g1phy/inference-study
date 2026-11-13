import { createContext, createElement, useContext, type ReactNode } from 'react'
import type { FlowSettings, ScenarioId, StageId } from '@/engine/types'
import { scrollToSimulator } from './nav'
import { createStore, shallowEqual, useStore, useStoreSelector, useTrackedStore, type Store } from './store'

/**
 * Глобальные настройки запроса (FlowSettings) и выбранная стадия симулятора.
 * Внешний стор на useSyncExternalStore, сохраняется в localStorage с валидацией.
 *
 *   const modelId = useSetting('modelId')      // по умолчанию: подписка на одно поле
 *   const [settings, patch] = useSettings()    // объект целиком (ререндер только по прочитанным полям)
 *   patch({ userText: 'Привет' })
 *   selectStage('route')                        // открыть стадию и проскроллить к симулятору
 *
 * Два вида настроек:
 * - «живые»: меняются на каждую букву (симулятор, глава hero);
 * - «принятые» (committed): текстовые поля (userText, systemText) приходят через
 *   SETTINGS_COMMIT_DELAY мс после последнего нажатия, остальные поля сразу. Главы вне hero
 *   читают принятые настройки (App оборачивает их в <SettingsScope mode="committed">), поэтому
 *   печать в симуляторе не перерисовывает и не перетокенизирует весь сайт на каждую букву.
 */

export const SETTINGS_STORAGE_KEY = 'inference-study:settings:v1'

export const SCENARIO_IDS: readonly ScenarioId[] = ['simple', 'thinking', 'tool', 'json', 'custom']

export const STAGE_IDS: readonly StageId[] = [
  'client',
  'frontend',
  'preprocess',
  'route',
  'prefill',
  'transfer',
  'decode',
  'postprocess',
  'stream',
]

/** Значения по умолчанию (SPEC §3). */
export const DEFAULT_SETTINGS: Readonly<FlowSettings> = Object.freeze({
  scenarioId: 'thinking',
  systemText: 'Ты полезный ассистент.',
  userText: 'Объясни простыми словами, что такое KV cache и зачем он нужен.',
  thinking: true,
  tools: false,
  json: false,
  deployment: 'agg',
  cacheWarm: false,
  modelId: 'qwen3-8b',
  gpuId: 'h100-sxm',
  temperature: 0.7,
})

/** Ограничения на поля (ввод пользователя и данные из localStorage). */
export const SETTINGS_LIMITS = {
  maxTextLength: 4000,
  temperature: { min: 0, max: 2 },
} as const

const ID_RE = /^[a-z0-9][a-z0-9._-]{0,63}$/

/**
 * Привести что угодно к корректным FlowSettings: каждое поле проверяется
 * отдельно, неподходящее значение заменяется значением по умолчанию.
 */
export function sanitizeSettings(raw: unknown, base: FlowSettings = DEFAULT_SETTINGS): FlowSettings {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const text = (v: unknown, d: string) =>
    typeof v === 'string' ? v.slice(0, SETTINGS_LIMITS.maxTextLength) : d
  const bool = (v: unknown, d: boolean) => (typeof v === 'boolean' ? v : d)
  const id = (v: unknown, d: string) => (typeof v === 'string' && ID_RE.test(v) ? v : d)
  const temp = (v: unknown, d: number) => {
    if (typeof v !== 'number' || !Number.isFinite(v)) return d
    const { min, max } = SETTINGS_LIMITS.temperature
    return Math.round(Math.min(max, Math.max(min, v)) * 100) / 100
  }
  return {
    scenarioId: SCENARIO_IDS.includes(r.scenarioId as ScenarioId) ? (r.scenarioId as ScenarioId) : base.scenarioId,
    systemText: text(r.systemText, base.systemText),
    userText: text(r.userText, base.userText),
    thinking: bool(r.thinking, base.thinking),
    tools: bool(r.tools, base.tools),
    json: bool(r.json, base.json),
    deployment: r.deployment === 'agg' || r.deployment === 'disagg' ? r.deployment : base.deployment,
    cacheWarm: bool(r.cacheWarm, base.cacheWarm),
    modelId: id(r.modelId, base.modelId),
    gpuId: id(r.gpuId, base.gpuId),
    temperature: temp(r.temperature, base.temperature),
  }
}

function load(): FlowSettings {
  if (typeof window === 'undefined') return { ...DEFAULT_SETTINGS }
  try {
    const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY)
    return raw ? sanitizeSettings(JSON.parse(raw)) : { ...DEFAULT_SETTINGS }
  } catch {
    return { ...DEFAULT_SETTINGS }
  }
}

const settingsStore = createStore<FlowSettings>(load(), shallowEqual)

/* Запись в localStorage с небольшой задержкой (ввод текста не должен писать на каждый символ). */
let saveTimer: number | undefined
function persist() {
  if (typeof window === 'undefined') return
  window.clearTimeout(saveTimer)
  saveTimer = window.setTimeout(flush, 250)
}
function flush() {
  window.clearTimeout(saveTimer)
  saveTimer = undefined
  try {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settingsStore.get()))
  } catch {
    /* переполнение или приватный режим: настройки живут до перезагрузки */
  }
}
if (typeof window !== 'undefined') {
  window.addEventListener('pagehide', () => saveTimer !== undefined && flush())
}

export function getSettings(): FlowSettings {
  return settingsStore.get()
}

/** Изменить часть настроек. Результат проходит валидацию. */
export function patchSettings(patch: Partial<FlowSettings>): void {
  const prev = settingsStore.get()
  const next = sanitizeSettings({ ...prev, ...patch }, prev)
  if (shallowEqual(prev, next)) return
  settingsStore.set(next)
  persist()
}

export function resetSettings(): void {
  settingsStore.set({ ...DEFAULT_SETTINGS })
  persist()
}

export function subscribeSettings(listener: () => void): () => void {
  return settingsStore.subscribe(listener)
}

/* ---------- принятые настройки: текст с задержкой, остальное сразу ---------- */

/** Через сколько мс после последнего нажатия текст доходит до глав вне симулятора. */
export const SETTINGS_COMMIT_DELAY = 250

const TEXT_KEYS: ReadonlySet<keyof FlowSettings> = new Set<keyof FlowSettings>(['userText', 'systemText'])

const committedStore = createStore<FlowSettings>(settingsStore.get(), shallowEqual)
let commitTimer: ReturnType<typeof setTimeout> | undefined

/** Сразу отдать главам текущие настройки (не ждать паузы в наборе). */
export function flushCommittedSettings(): void {
  if (commitTimer !== undefined) clearTimeout(commitTimer)
  commitTimer = undefined
  committedStore.set(settingsStore.get())
}

settingsStore.subscribe(() => {
  const live = settingsStore.get()
  const committed = committedStore.get()
  const keys = Object.keys(live) as Array<keyof FlowSettings>
  // поменялось что-то кроме текста (сценарий, флаги, модель): всё сразу и одним обновлением
  if (keys.some((k) => !TEXT_KEYS.has(k) && !Object.is(live[k], committed[k]))) {
    flushCommittedSettings()
    return
  }
  if (commitTimer !== undefined) clearTimeout(commitTimer)
  commitTimer = setTimeout(flushCommittedSettings, SETTINGS_COMMIT_DELAY)
})

export function getCommittedSettings(): FlowSettings {
  return committedStore.get()
}

export function subscribeCommittedSettings(listener: () => void): () => void {
  return committedStore.subscribe(listener)
}

/* ---------- область: живые или принятые настройки ---------- */

export type SettingsMode = 'live' | 'committed'

const SettingsModeContext = createContext<SettingsMode>('live')

/**
 * Какие настройки читают хуки внутри: 'live' (по умолчанию, симулятор) или 'committed'
 * (текст с задержкой). App оборачивает в committed каждую главу, кроме hero.
 */
export function SettingsScope({ mode, children }: { mode: SettingsMode; children?: ReactNode }) {
  return createElement(SettingsModeContext.Provider, { value: mode }, children)
}

export function useSettingsMode(): SettingsMode {
  return useContext(SettingsModeContext)
}

function useScopedStore(): Store<FlowSettings> {
  return useContext(SettingsModeContext) === 'committed' ? committedStore : settingsStore
}

/**
 * [настройки, patch]. Для симулятора и мест, где нужен объект целиком.
 * Ререндер только при изменении тех полей, которые компонент прочитал (снимок отслеживает чтение),
 * так что глава, читающая settings.modelId, не перерисовывается от набора текста.
 * В главах предпочитайте useSetting('modelId'): это явно и дешевле.
 */
export function useSettings(): [FlowSettings, (patch: Partial<FlowSettings>) => void] {
  return [useTrackedStore(useScopedStore()), patchSettings]
}

/** Подписка на одно поле (ререндер только при его изменении). Внутри глав текст приходит с задержкой. */
export function useSetting<K extends keyof FlowSettings>(key: K): FlowSettings[K] {
  return useStoreSelector(useScopedStore(), (s) => s[key])
}

/** Поле из принятых настроек (текст с задержкой) в любом месте, например в навигации. */
export function useCommittedSetting<K extends keyof FlowSettings>(key: K): FlowSettings[K] {
  return useStoreSelector(committedStore, (s) => s[key])
}

/** Поле из живых настроек в любом месте (каждая буква). Нужно редко: только самому полю ввода. */
export function useLiveSetting<K extends keyof FlowSettings>(key: K): FlowSettings[K] {
  return useStoreSelector(settingsStore, (s) => s[key])
}

/* ---------- выбранная стадия симулятора ---------- */

export interface StageSelection {
  /** Какая стадия открыта в инспекторе (null: инспектор следует за плеером). */
  stage: StageId | null
  /** Растёт при каждом вызове selectStage, даже с той же стадией (чтобы реагировать на повторный клик). */
  nonce: number
  /** true, если выбор пришёл извне симулятора (ссылка «Где это в симуляторе» из главы). */
  fromLink: boolean
}

const stageStore = createStore<StageSelection>({ stage: null, nonce: 0, fromLink: false })

/**
 * Выбрать стадию симулятора. По умолчанию ещё и скроллит к симулятору:
 * так работают ссылки «Где это в симуляторе» из глав. Прыжок издалека добавляет запись в историю
 * (scrollToSimulator), и «Назад» возвращает к абзацу, с которого ушли.
 * Внутри самого симулятора вызывайте с { scroll: false }.
 */
export function selectStage(stage: StageId | null, opts: { scroll?: boolean } = {}): void {
  const scroll = opts.scroll ?? true
  stageStore.set((prev) => ({ stage, nonce: prev.nonce + 1, fromLink: scroll }))
  if (scroll) scrollToSimulator()
}

export function getStageSelection(): StageSelection {
  return stageStore.get()
}

export function useSelectedStage(): StageId | null {
  return useStoreSelector(stageStore, (s) => s.stage)
}

export function useStageSelection(): StageSelection {
  return useStore(stageStore)
}
