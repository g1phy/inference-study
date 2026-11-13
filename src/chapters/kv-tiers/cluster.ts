/**
 * Игрушечная модель кластера Kubernetes для фигуры «L3 KV cache как DaemonSet на каждой ноде».
 * Три способа разместить кэш: внутри воркера (embedded), по поду кэша на ноду (DaemonSet) и
 * отдельный кэш-кластер. Чистый редьюсер: состояние → действие → новое состояние, всё детерминированно.
 *
 * Опора на реальные системы (research/notes-kvcache-tiers.md §1.12–1.14):
 *  - DaemonSet: LMCache Operator (CR LMCacheEngine → DaemonSet, один сервер на GPU-ноду, Service
 *    internalTrafficPolicy: Local, LMCacheMPConnector, порт 5555, cross-Pod CUDA IPC); в режиме «общий пул»
 *    так работает llm-d + Mooncake (DaemonSet mooncake-client на каждой ноде, метаданные в mooncake-master).
 *  - embedded: vLLM OffloadingConnector (CPU-уровень в памяти пода), SGLang HiCache L2.
 *  - кэш-кластер: Mooncake Store как Deployment, production-stack cache server.
 */
import { shortHash } from '@/engine/rng'

export type CacheMode = 'embedded' | 'daemonset' | 'cluster'
export type PrefixId = 'A' | 'B' | 'C'
export const PREFIXES: PrefixId[] = ['A', 'B', 'C']

export type Source = 'l1' | 'local' | 'node-dram' | 'node-nvme' | 'peer' | 'cluster' | 'miss'

/** Ёмкости уровней в «префиксах» (условно): HBM воркера, RAM пода, RAM и NVMe пода кэша, кэш-кластер. */
export const CAP = { l1: 2, local: 3, dram: 2, nvme: 8, store: 12 } as const

export interface Worker {
  id: string
  name: string
  node: string
  phase: 'Running' | 'ContainerCreating'
  /** Префиксы в HBM воркера (L1), последний самый свежий. */
  l1: PrefixId[]
  /** embedded: префиксы в RAM самого пода (L2 внутри процесса движка). */
  local: PrefixId[]
}

export interface CachePod {
  name: string
  phase: 'Running' | 'Pending'
  dram: PrefixId[]
  nvme: PrefixId[]
}

export interface NodeState {
  id: string
  name: string
  workers: string[]
  cache: CachePod | null
}

export interface LogEntry {
  id: number
  tone: 'good' | 'warn' | 'bad' | 'info'
  text: string
}

export interface Flow {
  worker: string
  node: string
  source: Source
  peerNode?: string
  prefix: PrefixId
  ttftMs: number
  /** Загрузка дольше пересчёта: адаптивный движок пересчитал бы. */
  recomputed?: boolean
}

export interface ClusterState {
  mode: CacheMode
  pooled: boolean
  nodes: NodeState[]
  workers: Record<string, Worker>
  store: PrefixId[]
  log: LogEntry[]
  rr: number
  seq: number
  nodeSeq: number
  stats: { requests: number; ttftSum: number; by: Record<Source, number> }
  flow: Flow | null
  /** Что подсветить после последнего действия, кроме запроса (удалённый под, новая нода и т. п.). */
  flash: { kind: 'kill' | 'node' | 'restart' | 'fail'; id: string } | null
}

/** TTFT по источнику KV, мс (считается в компоненте через perf.ts). */
export type Timing = Record<Source, number>

export type Action =
  | { type: 'send'; prefix: PrefixId; workerId?: string }
  | { type: 'kill'; workerId: string }
  | { type: 'ready'; workerId: string }
  | { type: 'addNode' }
  | { type: 'cacheReady'; nodeId: string }
  | { type: 'restartCache'; nodeId: string }
  | { type: 'failNode'; nodeId: string }
  | { type: 'setMode'; mode: CacheMode }
  | { type: 'setPooled'; pooled: boolean }
  | { type: 'reset' }

export const MAX_NODES = 3

const podName = (prefix: string, n: number) => `${prefix}-${shortHash(`${prefix}:${n}`).slice(0, 5)}`

function touch(list: PrefixId[], p: PrefixId, cap: number): { list: PrefixId[]; evicted: PrefixId[] } {
  const next = list.filter((x) => x !== p).concat(p)
  const evicted: PrefixId[] = []
  while (next.length > cap) evicted.push(next.shift()!)
  return { list: next, evicted }
}

function emptyStats(): ClusterState['stats'] {
  return { requests: 0, ttftSum: 0, by: { l1: 0, local: 0, 'node-dram': 0, 'node-nvme': 0, peer: 0, cluster: 0, miss: 0 } }
}

function makeNode(s: { seq: number; nodeSeq: number }, mode: CacheMode, workersPerNode: number, phase: Worker['phase']) {
  s.nodeSeq += 1
  const id = `node-${s.nodeSeq}`
  const workers: Worker[] = []
  for (let i = 0; i < workersPerNode; i++) {
    s.seq += 1
    workers.push({ id: `w${s.seq}`, name: podName('vllm', s.seq), node: id, phase, l1: [], local: [] })
  }
  s.seq += 1
  const node: NodeState = {
    id,
    name: `gpu-node-${s.nodeSeq}`,
    workers: workers.map((w) => w.id),
    cache: mode === 'daemonset' ? { name: podName('kv-cache', s.seq), phase: phase === 'Running' ? 'Running' : 'Pending', dram: [], nvme: [] } : null,
  }
  return { node, workers }
}

export function initialState(mode: CacheMode, pooled = false): ClusterState {
  const counters = { seq: 0, nodeSeq: 0 }
  const nodes: NodeState[] = []
  const workers: Record<string, Worker> = {}
  for (let i = 0; i < 2; i++) {
    const { node, workers: ws } = makeNode(counters, mode, 2, 'Running')
    nodes.push(node)
    for (const w of ws) workers[w.id] = w
  }
  const intro: Record<CacheMode, string> = {
    embedded: 'Кэш живёт внутри каждого пода vLLM: HBM плюс RAM самого пода. Отправьте запрос.',
    daemonset: 'На каждой GPU-ноде DaemonSet держит один под кэша, он владеет RAM и NVMe ноды. Отправьте запрос.',
    cluster: 'Кэш вынесен в отдельный кэш-кластер (StatefulSet), воркеры ходят к нему по сети. Отправьте запрос.',
  }
  return {
    mode,
    pooled,
    nodes,
    workers,
    store: [],
    log: [{ id: 0, tone: 'info', text: intro[mode] }],
    rr: 0,
    seq: counters.seq,
    nodeSeq: counters.nodeSeq,
    stats: emptyStats(),
    flow: null,
    flash: null,
  }
}

const SOURCE_TEXT: Record<Source, string> = {
  l1: 'уже в HBM этого воркера (L1)',
  local: 'нашёлся в RAM этого пода (L2)',
  'node-dram': 'нашёлся в кэше ноды: RAM пода kv-cache, данные по CUDA IPC',
  'node-nvme': 'нашёлся в кэше ноды на NVMe',
  peer: 'нашёлся у кэша соседней ноды: общий пул, данные по RDMA',
  cluster: 'нашёлся в кэш-кластере, данные по сети',
  miss: 'нигде нет: prefill всего промпта',
}

function fmtMs(ms: number): string {
  if (ms < 100) return `${ms.toFixed(ms < 10 ? 1 : 0).replace('.', ',')} мс`
  if (ms < 1000) return `${Math.round(ms)} мс`
  return `${(ms / 1000).toFixed(1).replace('.', ',')} с`
}

function pushLog(s: ClusterState, tone: LogEntry['tone'], text: string): LogEntry[] {
  const id = (s.log[0]?.id ?? 0) + 1
  return [{ id, tone, text }, ...s.log].slice(0, 30)
}

export function reduce(s: ClusterState, a: Action, timing: Timing): ClusterState {
  switch (a.type) {
    case 'reset':
      return initialState(s.mode, s.pooled)
    case 'setMode':
      return initialState(a.mode, s.pooled)
    case 'setPooled': {
      const next = { ...s, pooled: a.pooled, flow: null, flash: null }
      next.log = pushLog(
        s,
        'info',
        a.pooled
          ? 'Поды кэша объединены в общий пул: метаданные в mooncake-master, данные между нодами идут напрямую по RDMA.'
          : 'Поды кэша снова независимы: каждый знает только свою ноду.',
      )
      return next
    }
    case 'send':
      return send(s, a.prefix, a.workerId, timing)
    case 'kill': {
      const w = s.workers[a.workerId]
      if (!w) return s
      const seq = s.seq + 1
      const nw: Worker = { id: `w${seq}`, name: podName('vllm', seq), node: w.node, phase: 'ContainerCreating', l1: [], local: [] }
      const workers = { ...s.workers }
      delete workers[w.id]
      workers[nw.id] = nw
      const nodes = s.nodes.map((n) => (n.id === w.node ? { ...n, workers: n.workers.map((id) => (id === w.id ? nw.id : id)) } : n))
      const node = s.nodes.find((n) => n.id === w.node)!
      const lost = [...new Set([...w.l1, ...(s.mode === 'embedded' ? w.local : [])])]
      let tone: LogEntry['tone'] = 'warn'
      let text = `Под ${w.name} удалён. `
      if (s.mode === 'embedded') {
        tone = lost.length ? 'bad' : 'warn'
        text += lost.length ? `Вместе с ним пропали HBM и RAM пода: ${lost.length === 1 ? 'промпт' : 'промпты'} ${lost.join(', ')}. ` : 'Кэша в нём ещё не было. '
        text += `ReplicaSet создаёт ${nw.name}, он начнёт с холодным кэшем.`
      } else if (s.mode === 'daemonset') {
        const kept = node.cache ? [...new Set([...node.cache.dram, ...node.cache.nvme])] : []
        tone = 'good'
        text += `Потерян только его HBM. Под кэша ${node.cache?.name ?? ''} цел${kept.length ? `, в нём ${kept.join(', ')}` : ''}. Новый под ${nw.name} подхватит кэш ноды.`
      } else {
        tone = 'good'
        text += `Потерян только его HBM, кэш-кластер цел${s.store.length ? ` (${s.store.join(', ')})` : ''}. Новый под ${nw.name} достанет KV по сети.`
      }
      return { ...s, seq, workers, nodes, log: pushLog(s, tone, text), flow: null, flash: { kind: 'kill', id: nw.id } }
    }
    case 'ready': {
      const w = s.workers[a.workerId]
      if (!w || w.phase === 'Running') return s
      return { ...s, workers: { ...s.workers, [w.id]: { ...w, phase: 'Running' } } }
    }
    case 'addNode': {
      if (s.nodes.length >= MAX_NODES) return s
      const counters = { seq: s.seq, nodeSeq: s.nodeSeq }
      const { node, workers: ws } = makeNode(counters, s.mode, 1, 'ContainerCreating')
      const workers = { ...s.workers }
      for (const w of ws) workers[w.id] = w
      const text =
        s.mode === 'daemonset'
          ? `Нода ${node.name} вступила в кластер. DaemonSet сам создал на ней под ${node.cache?.name}: кэш пока пустой. Deployment поставил туда новый воркер.`
          : s.mode === 'embedded'
            ? `Нода ${node.name} вступила в кластер, Deployment поставил на неё воркер. Его кэш внутри пода и пока пустой.`
            : `Нода ${node.name} вступила в кластер, Deployment поставил на неё воркер. Он сразу видит весь кэш-кластер.`
      return {
        ...s,
        seq: counters.seq,
        nodeSeq: counters.nodeSeq,
        nodes: [...s.nodes, node],
        workers,
        log: pushLog(s, 'info', text),
        flow: null,
        flash: { kind: 'node', id: node.id },
      }
    }
    case 'cacheReady':
      return {
        ...s,
        nodes: s.nodes.map((n) => (n.id === a.nodeId && n.cache ? { ...n, cache: { ...n.cache, phase: 'Running' } } : n)),
      }
    case 'restartCache': {
      const node = s.nodes.find((n) => n.id === a.nodeId)
      if (!node || !node.cache) return s
      const seq = s.seq + 1
      const lost = node.cache.dram.filter((p) => !node.cache!.nvme.includes(p))
      const cache: CachePod = { name: podName('kv-cache', seq), phase: 'Running', dram: [], nvme: node.cache.nvme }
      const text =
        `Обновление DaemonSet перезапустило под кэша на ${node.name}: RAM-часть потеряна` +
        (lost.length ? ` (только там лежали ${lost.join(', ')})` : '') +
        `, NVMe-часть восстановлена${node.cache.nvme.length ? `: ${node.cache.nvme.join(', ')}` : ''}. Восстановление с диска умеют не все бэкенды.`
      return {
        ...s,
        seq,
        nodes: s.nodes.map((n) => (n.id === node.id ? { ...n, cache } : n)),
        log: pushLog(s, 'warn', text),
        flow: null,
        flash: { kind: 'restart', id: node.id },
      }
    }
    case 'failNode': {
      if (s.nodes.length <= 1) return s
      const node = s.nodes.find((n) => n.id === a.nodeId)
      if (!node) return s
      const workers = { ...s.workers }
      const lostPrefixes = new Set<PrefixId>()
      for (const id of node.workers) {
        const w = workers[id]
        if (w) [...w.l1, ...w.local].forEach((p) => lostPrefixes.add(p))
        delete workers[id]
      }
      if (node.cache) [...node.cache.dram, ...node.cache.nvme].forEach((p) => lostPrefixes.add(p))
      const nodes = s.nodes.filter((n) => n.id !== node.id)
      let text = `Нода ${node.name} упала. `
      if (s.mode === 'cluster') text += `Пропали HBM её воркеров, кэш-кластер цел${s.store.length ? ` (${s.store.join(', ')})` : ''}.`
      else if (s.mode === 'daemonset')
        text += `Вместе с ней пропали HBM воркеров и под кэша этой ноды${lostPrefixes.size ? ` (${[...lostPrefixes].join(', ')})` : ''}${s.pooled ? '; общий пул потерял её часть' : ''}.`
      else text += `Пропал кэш её подов${lostPrefixes.size ? ` (${[...lostPrefixes].join(', ')})` : ''}.`
      return { ...s, nodes, workers, log: pushLog(s, 'bad', text), flow: null, flash: { kind: 'fail', id: node.id }, rr: 0 }
    }
  }
}

function send(s: ClusterState, prefix: PrefixId, workerId: string | undefined, timing: Timing): ClusterState {
  const running = s.nodes.flatMap((n) => n.workers).filter((id) => s.workers[id]?.phase === 'Running')
  if (!running.length) return { ...s, log: pushLog(s, 'bad', 'Нет готовых воркеров: запрос ждёт.'), flow: null, flash: null }
  let id = workerId && s.workers[workerId]?.phase === 'Running' ? workerId : undefined
  let rr = s.rr
  if (!id) {
    id = running[rr % running.length]
    rr += 1
  }
  const w = s.workers[id]
  const nodeIdx = s.nodes.findIndex((n) => n.id === w.node)
  const node = s.nodes[nodeIdx]
  const cache = node.cache && node.cache.phase === 'Running' ? node.cache : null

  let source: Source = 'miss'
  let peerNode: string | undefined
  if (w.l1.includes(prefix)) source = 'l1'
  else if (s.mode === 'embedded' && w.local.includes(prefix)) source = 'local'
  else if (s.mode === 'daemonset' && cache && cache.dram.includes(prefix)) source = 'node-dram'
  else if (s.mode === 'daemonset' && cache && cache.nvme.includes(prefix)) source = 'node-nvme'
  else if (s.mode === 'daemonset' && s.pooled) {
    const peer = s.nodes.find((n) => n.id !== node.id && n.cache?.phase === 'Running' && (n.cache.dram.includes(prefix) || n.cache.nvme.includes(prefix)))
    if (peer) {
      source = 'peer'
      peerNode = peer.id
    }
  } else if (s.mode === 'cluster' && s.store.includes(prefix)) source = 'cluster'

  // Если загрузка дольше полного prefill, адаптивный движок просто пересчитал бы.
  let ttft = timing[source]
  const recomputed = source !== 'miss' && source !== 'l1' && ttft > timing.miss
  if (recomputed) ttft = timing.miss

  // обновить кэши: HBM воркера и запись вниз (write-through)
  const workers = { ...s.workers }
  const nw: Worker = { ...w, l1: [...w.l1], local: [...w.local] }
  const l1 = touch(nw.l1, prefix, CAP.l1)
  nw.l1 = l1.list
  let nodes = s.nodes
  let store = s.store
  if (s.mode === 'embedded') nw.local = touch(nw.local, prefix, CAP.local).list
  if (s.mode === 'daemonset' && cache && source !== 'peer') {
    const dram = touch(cache.dram, prefix, CAP.dram)
    const nvme = touch(cache.nvme, prefix, CAP.nvme)
    nodes = s.nodes.map((n) => (n.id === node.id ? { ...n, cache: { ...cache, dram: dram.list, nvme: nvme.list } } : n))
  }
  if (s.mode === 'cluster') store = touch(s.store, prefix, CAP.store).list
  workers[nw.id] = nw

  const where = node.name
  const target =
    s.mode === 'embedded' ? 'в RAM этого пода' : s.mode === 'daemonset' ? (cache ? `в под кэша ноды ${where}` : 'некуда: под кэша ещё не готов') : 'в кэш-кластер'
  let text = `${w.name} на ${where}, промпт ${prefix}: ${SOURCE_TEXT[source]}${source === 'peer' && peerNode ? ` (${s.nodes.find((n) => n.id === peerNode)?.name})` : ''}. TTFT ≈ ${fmtMs(ttft)}.`
  if (source === 'miss') text += ` KV сохранён ${target}.`
  if (recomputed) {
    const how = source === 'cluster' || source === 'peer' ? ' по сети' : source === 'node-nvme' ? ' с NVMe' : ''
    text += ` Загрузка${how} шла бы дольше пересчёта, поэтому движку выгоднее пересчитать prefill: TTFT как при промахе.`
  }
  // нашлось, но пересчитали: не промах и не попадание, а предупреждение (канал слишком медленный)
  const tone: LogEntry['tone'] = source === 'miss' ? 'bad' : recomputed ? 'warn' : 'good'
  const stats = {
    requests: s.stats.requests + 1,
    ttftSum: s.stats.ttftSum + ttft,
    by: { ...s.stats.by, [recomputed ? 'miss' : source]: s.stats.by[recomputed ? 'miss' : source] + 1 },
  }
  return {
    ...s,
    rr,
    workers,
    nodes,
    store,
    stats,
    log: pushLog(s, tone, text),
    flow: { worker: w.id, node: node.id, source, peerNode, prefix, ttftMs: ttft, recomputed },
    flash: null,
  }
}
