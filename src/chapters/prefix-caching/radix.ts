/**
 * Фигура «Radix tree (SGLang)»: пресеты потоков запросов, прогон через настоящий RadixTree движка
 * (insert / match / evictLRU / view) и раскладка дерева для SVG. Токены синтетические: у каждой части
 * промпта (системный промпт, вопрос, ответ, пример) свой диапазон id, длины условные, но правдоподобные.
 */
import { RadixTree } from '@/engine/kv'
import type { RadixNodeView } from '@/engine/types'

export interface Segment {
  key: string
  label: string
  tokens: number
}

export interface PresetRequest {
  label: string
  prompt: string[]
  answer: string
}

export interface Preset {
  id: 'chat' | 'system' | 'fewshot'
  name: string
  /** Короткое имя для переключателя. */
  short: string
  segments: Segment[]
  requests: PresetRequest[]
}

const seg = (key: string, label: string, tokens: number): Segment => ({ key, label, tokens })

export const PRESETS: Preset[] = [
  {
    id: 'chat',
    name: 'Многоходовый чат',
    short: 'Чат',
    segments: [
      seg('S', 'Системный промпт', 400),
      seg('aq1', 'Вопрос А1', 40),
      seg('aa1', 'Ответ А1', 160),
      seg('aq2', 'Вопрос А2', 30),
      seg('aa2', 'Ответ А2', 180),
      seg('aq3', 'Вопрос А3', 35),
      seg('aa3', 'Ответ А3', 140),
      seg('bq1', 'Вопрос Б1', 50),
      seg('ba1', 'Ответ Б1', 150),
      seg('bq2', 'Вопрос Б2', 40),
      seg('ba2', 'Ответ Б2', 120),
    ],
    requests: [
      { label: 'Чат А, ход 1', prompt: ['S', 'aq1'], answer: 'aa1' },
      { label: 'Чат А, ход 2', prompt: ['S', 'aq1', 'aa1', 'aq2'], answer: 'aa2' },
      { label: 'Чат Б, ход 1', prompt: ['S', 'bq1'], answer: 'ba1' },
      { label: 'Чат А, ход 3', prompt: ['S', 'aq1', 'aa1', 'aq2', 'aa2', 'aq3'], answer: 'aa3' },
      { label: 'Чат Б, ход 2', prompt: ['S', 'bq1', 'ba1', 'bq2'], answer: 'ba2' },
    ],
  },
  {
    id: 'system',
    name: 'Общий системный промпт',
    short: 'Общий промпт',
    segments: [
      seg('S', 'Системный промпт', 800),
      seg('q1', 'Вопрос 1', 45),
      seg('a1', 'Ответ 1', 210),
      seg('q2', 'Вопрос 2', 30),
      seg('a2', 'Ответ 2', 160),
      seg('q3', 'Вопрос 3', 60),
      seg('a3', 'Ответ 3', 240),
      seg('q4', 'Вопрос 4', 35),
      seg('a4', 'Ответ 4', 180),
      seg('q5', 'Вопрос 5', 50),
      seg('a5', 'Ответ 5', 200),
    ],
    requests: [1, 2, 3, 4, 5].map((i) => ({ label: `Пользователь ${i}`, prompt: ['S', `q${i}`], answer: `a${i}` })),
  },
  {
    id: 'fewshot',
    name: 'Few-shot',
    short: 'Few-shot',
    segments: [
      seg('I', 'Инструкция', 150),
      seg('e1', 'Пример 1', 200),
      seg('e2', 'Пример 2', 180),
      seg('e3', 'Пример 3', 220),
      seg('q1', 'Вопрос 1', 40),
      seg('r1', 'Ответ 1', 60),
      seg('q2', 'Вопрос 2', 35),
      seg('r2', 'Ответ 2', 50),
      seg('q3', 'Вопрос 3', 45),
      seg('r3', 'Ответ 3', 70),
      seg('q4', 'Вопрос 4', 30),
      seg('r4', 'Ответ 4', 55),
    ],
    requests: [1, 2, 3, 4].map((i) => ({ label: `Запрос ${i}`, prompt: ['I', 'e1', 'e2', 'e3', `q${i}`], answer: `r${i}` })),
  },
]

export const getPreset = (id: Preset['id']) => PRESETS.find((p) => p.id === id) ?? PRESETS[0]

/** Синтетические id токенов: у сегмента i диапазон 10 000 + i·5 000 … */
function segBase(i: number): number {
  return 10_000 + i * 5_000
}

export function segmentTokens(p: Preset, key: string): number[] {
  const i = p.segments.findIndex((s) => s.key === key)
  if (i < 0) throw new Error(`нет сегмента ${key}`)
  const base = segBase(i)
  return Array.from({ length: p.segments[i].tokens }, (_, k) => base + k)
}

export function requestTokens(p: Preset, r: PresetRequest): { prompt: number[]; full: number[] } {
  const prompt = r.prompt.flatMap((k) => segmentTokens(p, k))
  return { prompt, full: prompt.concat(segmentTokens(p, r.answer)) }
}

/** Самый длинный запрос пресета вместе с ответом: меньше этой ёмкости дерево не удержит даже текущий запрос. */
export function longestRequestTokens(p: Preset): number {
  return p.requests.reduce((a, r) => Math.max(a, requestTokens(p, r).full.length), 0)
}

/** Максимальная ёмкость, при которой ничего не вытесняется (сумма уникальных токенов всех запросов). */
export function totalUniqueTokens(p: Preset): number {
  const keys = new Set<string>()
  for (const r of p.requests) {
    r.prompt.forEach((k) => keys.add(k))
    keys.add(r.answer)
  }
  return [...keys].reduce((a, k) => a + (p.segments.find((s) => s.key === k)?.tokens ?? 0), 0)
}

/** Какие сегменты (и части сегментов) покрывает последовательность токенов узла. */
export function describeTokens(p: Preset, tokens: number[]): { label: string; parts: string[] } {
  const parts: string[] = []
  let i = 0
  while (i < tokens.length) {
    const idx = Math.floor((tokens[i] - 10_000) / 5_000)
    const s = p.segments[idx]
    let j = i
    while (j < tokens.length && Math.floor((tokens[j] - 10_000) / 5_000) === idx) j++
    const whole = s && j - i === s.tokens
    parts.push(s ? (whole ? s.label : `часть: ${s.label}`) : '?')
    i = j
  }
  const label = parts.length <= 2 ? parts.join(' + ') : `${parts[0]} … ${parts[parts.length - 1]}`
  return { label, parts }
}

/* ============================================================================
 * Прогон и раскладка
 * ========================================================================== */

export interface LaidNode {
  id: string
  x: number
  y: number
  depth: number
  tokens: number
  label: string
  parts: string[]
  /** Участвовал в совпадении последнего запроса. */
  hit: boolean
  /** Создан последним запросом (новая часть промпта посчитана в prefill, ответ в decode). */
  isNew: boolean
  /** Сколько запросов (из уже обработанных) проходят через узел. */
  shared: number
  /** Узел вытеснен по LRU на этом шаге: рисуется пунктирным «призраком» один шаг. */
  ghost: boolean
  parent: string | null
  lastAccess: number
}

export interface RequestStat {
  label: string
  promptTokens: number
  matched: number
}

export interface RadixStep {
  /** Индекс последнего обработанного запроса (−1: дерево пустое). */
  index: number
  nodes: LaidNode[]
  size: number
  evicted: string[]
  history: RequestStat[]
  width: number
  height: number
}

export const NODE_W = 172
export const NODE_H = 46
const H_GAP = 34
const V_GAP = 14
export const ROOT_W = 52

/** Прогоняет первые k запросов пресета при ёмкости capacity (токенов) и раскладывает дерево. */
export function radixStep(p: Preset, k: number, capacity: number): RadixStep {
  const tree = new RadixTree()
  const history: RequestStat[] = []
  let evicted: string[] = []
  let newLeaf: string | null = null
  const fulls: number[][] = []
  // дерево до вытеснения на последнем шаге: вытесненные узлы показываем на своих местах пунктиром
  let preView: RadixNodeView | null = null
  let evictedIds: string[] = []
  for (let i = 0; i < k; i++) {
    const r = p.requests[i]
    const { prompt, full } = requestTokens(p, r)
    const m = tree.match(prompt)
    const ins = tree.insert(full)
    newLeaf = ins.insertedTokens > 0 ? ins.path[ins.path.length - 1] : null
    fulls.push(full)
    history.push({ label: r.label, promptTokens: prompt.length, matched: m.matchedTokens })
    // запомнить подписи до вытеснения, чтобы назвать вытесненное
    const before = new Map<string, number[]>()
    const walk = (n: RadixNodeView) => {
      before.set(n.id, n.tokens)
      n.children.forEach(walk)
    }
    walk(tree.view())
    const pre = i === k - 1 ? tree.view() : null
    const ev = tree.evictLRU(capacity)
    evicted = i === k - 1 ? ev.evictedNodes.map((id) => describeTokens(p, before.get(id) ?? []).label) : []
    if (i === k - 1 && ev.evictedNodes.length) {
      preView = pre
      evictedIds = ev.evictedNodes
    }
    if (newLeaf && ev.evictedNodes.includes(newLeaf)) newLeaf = null
  }
  return layout(p, preView ?? tree.view(), tree.size, k - 1, evicted, history, newLeaf, fulls, new Set(evictedIds))
}

function layout(
  p: Preset,
  root: RadixNodeView,
  size: number,
  index: number,
  evicted: string[],
  history: RequestStat[],
  newLeaf: string | null,
  fulls: number[][],
  ghosts: ReadonlySet<string> = new Set(),
): RadixStep {
  const nodes: LaidNode[] = []
  let leafRow = 0
  const place = (n: RadixNodeView, depth: number, prefix: number[], parent: string | null): number => {
    const abs = prefix.concat(n.tokens)
    let y: number
    if (!n.children.length) {
      y = leafRow * (NODE_H + V_GAP)
      leafRow++
    } else {
      const ys = n.children.map((c) => place(c, depth + 1, abs, n.id))
      y = (ys[0] + ys[ys.length - 1]) / 2
    }
    if (n.id !== 'root') {
      const d = describeTokens(p, n.tokens)
      const shared = fulls.filter((f) => abs.length <= f.length && abs.every((t, i) => f[i] === t)).length
      nodes.push({
        id: n.id,
        x: ROOT_W + H_GAP + (depth - 1) * (NODE_W + H_GAP),
        y,
        depth,
        tokens: n.tokens.length,
        label: d.label,
        parts: d.parts,
        hit: Boolean(n.hit),
        isNew: n.id === newLeaf,
        shared,
        parent,
        lastAccess: n.lastAccess,
        ghost: ghosts.has(n.id),
      })
    }
    return y
  }
  const rootY = place(root, 0, [], null)
  const maxDepth = nodes.reduce((a, n) => Math.max(a, n.depth), 0)
  const width = ROOT_W + (maxDepth > 0 ? H_GAP + maxDepth * (NODE_W + H_GAP) - H_GAP : 0)
  const height = Math.max(NODE_H, leafRow * (NODE_H + V_GAP) - V_GAP)
  // корень: запишем его y как отдельный узел с depth 0 (для рёбер)
  nodes.push({ id: 'root', x: 0, y: rootY, depth: 0, tokens: 0, label: 'корень', parts: [], hit: false, isNew: false, shared: fulls.length, parent: null, lastAccess: 0, ghost: false })
  return { index, nodes, size, evicted, history, width, height }
}
