import { describe, expect, it } from 'vitest'
import { createStore, createTracker, shallowEqual } from './store'

interface S {
  a: number
  b: string
  c: boolean
}

const make = () => createStore<S>({ a: 1, b: 'x', c: false }, shallowEqual)

describe('createTracker: подписка по факту чтения', () => {
  it('непрочитанные поля не меняют снимок, а прочитанные меняют', () => {
    const store = make()
    const t = createTracker(store)
    const s1 = t.getSnapshot()
    expect(s1.a).toBe(1)
    store.set((p) => ({ ...p, b: 'y' }))
    // b никто не читал: тот же объект, React не перерисует компонент
    expect(t.getSnapshot()).toBe(s1)
    store.set((p) => ({ ...p, a: 2 }))
    const s2 = t.getSnapshot()
    expect(s2).not.toBe(s1)
    expect(s2.a).toBe(2)
  })

  it('первое чтение нового поля даёт свежее значение', () => {
    const store = make()
    const t = createTracker(store)
    const s1 = t.getSnapshot()
    void s1.a
    store.set((p) => ({ ...p, b: 'y' }))
    const same = t.getSnapshot()
    expect(same).toBe(s1)
    expect(same.b).toBe('y')
    // теперь b прочитано: его изменение даёт новый снимок
    store.set((p) => ({ ...p, b: 'z' }))
    expect(t.getSnapshot()).not.toBe(s1)
  })

  it('старый снимок замирает: сравнение «было/стало» работает', () => {
    const store = make()
    const t = createTracker(store)
    const before = t.getSnapshot()
    void before.b
    store.set((p) => ({ ...p, b: 'new' }))
    const after = t.getSnapshot()
    expect(before.b).toBe('x')
    expect(after.b).toBe('new')
  })

  it('перебор ключей подписывает на все поля и работает как обычный объект', () => {
    const store = make()
    const t = createTracker(store)
    const s1 = t.getSnapshot()
    expect({ ...s1 }).toEqual({ a: 1, b: 'x', c: false })
    expect(JSON.parse(JSON.stringify(s1))).toEqual({ a: 1, b: 'x', c: false })
    expect(Object.keys(s1)).toEqual(['a', 'b', 'c'])
    expect('a' in s1).toBe(true)
    expect(t.all).toBe(true)
    store.set((p) => ({ ...p, c: true }))
    expect(t.getSnapshot()).not.toBe(s1)
  })

  it('снимок только для чтения', () => {
    const t = createTracker(make())
    const s = t.getSnapshot() as { a: number }
    expect(() => {
      s.a = 5
    }).toThrow(TypeError)
    expect(s.a).toBe(1)
  })

  it('повторный вызов без изменений возвращает тот же снимок', () => {
    const store = make()
    const t = createTracker(store)
    const s1 = t.getSnapshot()
    void s1.a
    store.set((p) => ({ ...p, a: 3 }))
    const s2 = t.getSnapshot()
    expect(t.getSnapshot()).toBe(s2)
  })
})
