import { useRef, useSyncExternalStore } from 'react'

/**
 * Минимальный внешний стор для useSyncExternalStore (без сторонних библиотек).
 * Используется настройками, темой, навигацией. Можно использовать и в главах.
 */
export interface Store<T> {
  get(): T
  set(next: T | ((prev: T) => T)): void
  subscribe(listener: () => void): () => void
}

export function createStore<T>(initial: T, equals: (a: T, b: T) => boolean = Object.is): Store<T> {
  let state = initial
  const listeners = new Set<() => void>()
  return {
    get: () => state,
    set(next) {
      const value = typeof next === 'function' ? (next as (prev: T) => T)(state) : next
      if (equals(state, value)) return
      state = value
      listeners.forEach((l) => l())
    },
    subscribe(listener) {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
  }
}

/** Подписка на весь стор. */
export function useStore<T>(store: Store<T>): T {
  return useSyncExternalStore(store.subscribe, store.get, store.get)
}

/**
 * Подписка на часть стора. Селектор должен возвращать примитив или стабильную ссылку
 * (не собирайте новый объект в селекторе, иначе будет бесконечный ререндер).
 */
export function useStoreSelector<T, S>(store: Store<T>, selector: (state: T) => S): S {
  const get = () => selector(store.get())
  return useSyncExternalStore(store.subscribe, get, get)
}

/** Поверхностное сравнение объектов (для сторов с объектным состоянием). */
export function shallowEqual<T extends object>(a: T, b: T): boolean {
  if (Object.is(a, b)) return true
  const ka = Object.keys(a) as Array<keyof T>
  const kb = Object.keys(b) as Array<keyof T>
  if (ka.length !== kb.length) return false
  return ka.every((k) => Object.is(a[k], b[k]))
}

/* ===========================================================================
 * Подписка «по факту чтения» (tracked snapshot).
 *
 * Компонент получает снимок-прокси объекта из стора и перерисовывается только тогда, когда
 * меняется поле, которое он прочитал (в рендере, в эффекте или в обработчике). Так
 * `const [settings] = useSettings()` в главе, которая читает только settings.modelId,
 * не перерисовывается на каждую букву, набранную в симуляторе.
 *
 * Правила снимков:
 * - пока меняются только непрочитанные поля, возвращается тот же снимок (тот же объект),
 *   и он показывает свежие значения этих полей: первое чтение нового поля всегда актуально;
 * - как только меняется прочитанное поле, появляется новый снимок, а старый замирает
 *   на последнем состоянии стора перед изменением (сравнение «было/стало» работает);
 * - перебор ключей (spread, Object.keys, JSON.stringify) подписывает на все поля.
 * ========================================================================= */

export interface Tracker<T extends object> {
  readonly store: Store<T>
  /** Снимок для useSyncExternalStore (стабильная ссылка, пока прочитанные поля не менялись). */
  getSnapshot(): T
  /** Прочитанные поля (для тестов и отладки). */
  readonly seen: ReadonlySet<PropertyKey>
  /** Подписан на все поля (после перебора ключей). */
  readonly all: boolean
}

export function createTracker<T extends object>(store: Store<T>): Tracker<T> {
  const seen = new Set<PropertyKey>()
  let all = false
  let holder = { base: store.get() }

  const track = (base: T, k: PropertyKey) => {
    if (!all && typeof k === 'string' && Object.prototype.hasOwnProperty.call(base, k)) seen.add(k)
  }

  const make = (h: { base: T }): T =>
    new Proxy({} as T, {
      get: (_t, k) => {
        track(h.base, k)
        return Reflect.get(h.base, k)
      },
      has: (_t, k) => {
        track(h.base, k)
        return Reflect.has(h.base, k)
      },
      ownKeys: () => {
        all = true
        return Reflect.ownKeys(h.base)
      },
      getOwnPropertyDescriptor: (_t, k) => {
        track(h.base, k)
        const d = Reflect.getOwnPropertyDescriptor(h.base, k)
        // свойства нет на пустой цели прокси: дескриптор обязан быть configurable
        return d ? { value: d.value, writable: false, enumerable: d.enumerable, configurable: true } : undefined
      },
      // снимок только для чтения: состояние меняют через стор (patchSettings и т.п.)
      set: () => false,
      defineProperty: () => false,
      deleteProperty: () => false,
    })

  let snap = make(holder)

  return {
    store,
    get seen() {
      return seen
    },
    get all() {
      return all
    },
    getSnapshot() {
      const s = store.get()
      if (s === holder.base) return snap
      let changed = all
      if (!changed) {
        for (const k of seen) {
          if (!Object.is((holder.base as Record<PropertyKey, unknown>)[k], (s as Record<PropertyKey, unknown>)[k])) {
            changed = true
            break
          }
        }
      }
      if (changed) {
        // старый снимок замирает на прежнем состоянии, новый смотрит на текущее
        holder = { base: s }
        snap = make(holder)
      } else {
        // поменялись только непрочитанные поля: тот же снимок, но со свежими значениями
        holder.base = s
      }
      return snap
    },
  }
}

/**
 * Подписка на объектный стор «по факту чтения» (см. createTracker): ререндер только при
 * изменении полей, которые компонент читал.
 */
export function useTrackedStore<T extends object>(store: Store<T>): T {
  const ref = useRef<Tracker<T> | null>(null)
  if (ref.current === null || ref.current.store !== store) ref.current = createTracker(store)
  const t = ref.current
  return useSyncExternalStore(store.subscribe, t.getSnapshot, t.getSnapshot)
}
