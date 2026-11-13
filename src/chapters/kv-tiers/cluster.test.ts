import { describe, expect, it } from 'vitest'
import { initialState, reduce, type ClusterState, type Action, type Timing } from './cluster'

const T: Timing = { l1: 30, local: 40, 'node-dram': 40, 'node-nvme': 70, peer: 45, cluster: 300, miss: 200 }
const run = (s: ClusterState, ...acts: Action[]) => acts.reduce((st, a) => reduce(st, a, T), s)
const ids = (s: ClusterState, node = 0) => s.nodes[node].workers

describe('кластер: DaemonSet', () => {
  it('KV, посчитанный одним воркером, переиспользует второй на той же ноде', () => {
    let s = initialState('daemonset')
    const [w1, w2] = ids(s)
    s = run(s, { type: 'send', prefix: 'A', workerId: w1 })
    expect(s.flow?.source).toBe('miss')
    s = run(s, { type: 'send', prefix: 'A', workerId: w2 })
    expect(s.flow?.source).toBe('node-dram')
  })
  it('удаление воркера не теряет кэш ноды', () => {
    let s = initialState('daemonset')
    const [w1] = ids(s)
    s = run(s, { type: 'send', prefix: 'A', workerId: w1 }, { type: 'kill', workerId: w1 })
    const fresh = ids(s)[0]
    s = run(s, { type: 'ready', workerId: fresh }, { type: 'send', prefix: 'A', workerId: fresh })
    expect(s.flow?.source).toBe('node-dram')
  })
  it('новая нода сама получает под кэша (сначала Pending)', () => {
    let s = initialState('daemonset')
    s = run(s, { type: 'addNode' })
    expect(s.nodes.length).toBe(3)
    expect(s.nodes[2].cache?.phase).toBe('Pending')
    s = run(s, { type: 'cacheReady', nodeId: s.nodes[2].id })
    expect(s.nodes[2].cache?.phase).toBe('Running')
  })
  it('без общего пула соседняя нода промахивается, с пулом попадает', () => {
    let s = initialState('daemonset')
    s = run(s, { type: 'send', prefix: 'B', workerId: ids(s, 0)[0] }, { type: 'send', prefix: 'B', workerId: ids(s, 1)[0] })
    expect(s.flow?.source).toBe('miss')
    let p = initialState('daemonset', true)
    p = run(p, { type: 'send', prefix: 'B', workerId: ids(p, 0)[0] }, { type: 'send', prefix: 'B', workerId: ids(p, 1)[0] })
    expect(p.flow?.source).toBe('peer')
  })
  it('рестарт пода кэша теряет RAM, NVMe остаётся', () => {
    let s = initialState('daemonset')
    const [w1, w2] = ids(s)
    s = run(s, { type: 'send', prefix: 'A', workerId: w1 }, { type: 'restartCache', nodeId: s.nodes[0].id }, { type: 'send', prefix: 'A', workerId: w2 })
    expect(s.flow?.source).toBe('node-nvme')
  })
})

describe('кластер: embedded и кэш-кластер', () => {
  it('embedded: второй воркер ноды кэш первого не видит, рестарт теряет всё', () => {
    let s = initialState('embedded')
    const [w1, w2] = ids(s)
    s = run(s, { type: 'send', prefix: 'A', workerId: w1 }, { type: 'send', prefix: 'A', workerId: w2 })
    expect(s.flow?.source).toBe('miss')
    s = run(s, { type: 'kill', workerId: w1 })
    expect(s.log[0].tone).toBe('bad')
  })
  it('кэш-кластер: попадание с любой ноды, а медленная сеть превращается в пересчёт', () => {
    let s = initialState('cluster')
    s = run(s, { type: 'send', prefix: 'C', workerId: ids(s, 0)[0] }, { type: 'send', prefix: 'C', workerId: ids(s, 1)[0] })
    expect(s.flow?.source).toBe('cluster')
    expect(s.flow?.recomputed).toBe(true) // 300 мс по сети > 200 мс prefill
    expect(s.flow?.ttftMs).toBe(200)
  })
  it('отказ ноды: в кэш-кластере данные целы, последняя нода не падает', () => {
    let s = initialState('cluster')
    s = run(s, { type: 'send', prefix: 'A' }, { type: 'failNode', nodeId: s.nodes[0].id })
    expect(s.nodes.length).toBe(1)
    expect(s.store).toContain('A')
    s = run(s, { type: 'failNode', nodeId: s.nodes[0].id })
    expect(s.nodes.length).toBe(1)
  })
  it('round-robin ходит по готовым воркерам', () => {
    let s = initialState('daemonset')
    const seen = new Set<string>()
    for (let i = 0; i < 4; i++) {
      s = run(s, { type: 'send', prefix: 'A' })
      seen.add(s.flow!.worker)
    }
    expect(seen.size).toBe(4)
  })
})
