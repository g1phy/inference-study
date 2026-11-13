/**
 * Игрушечная «модель» для фигуры генерации по шагам: распределения заскриптованы вручную, по словам,
 * а не по токенам. Состояние = что уже сказано (упрощённо), у каждого состояния 3-7 кандидатов
 * с условными логитами и переходом в следующее состояние. Выбор делает настоящий сэмплер движка.
 *
 * Greedy-путь: «KV cache хранит ключи и значения всех прошлых токенов, чтобы не пересчитывать их заново.»
 */
import { sample, type SamplingParamsEx } from '@/engine/sampling'
import type { Candidate, Token } from '@/engine/types'

export interface Edge {
  w: string
  z: number
  next: string
}

export const TOY_PREFIX = 'KV cache хранит'
export const END = 'END'
export const MAX_STEPS = 16

/** Слова, которые делают текст фактически неверным или абсурдным (для подписи). */
export const ODD_WORDS = new Set([' запросы', ' запросов', ' Q', ' пароли', ' котиков', ' квартиры', ' собачек', ' воспоминания', ' навсегда'])

export const GRAPH: Record<string, Edge[]> = {
  s0: [
    { w: ' ключи', z: 3.6, next: 'keys' },
    { w: ' векторы', z: 2.7, next: 'vec' },
    { w: ' результаты', z: 2.1, next: 'res' },
    { w: ' промежуточные', z: 1.5, next: 'inter' },
    { w: ' всё', z: 0.8, next: 'all' },
    { w: ' воспоминания', z: -0.6, next: 'mem' },
    { w: ' котиков', z: -1.8, next: 'weird' },
  ],
  keys: [
    { w: ' и', z: 4.2, next: 'keysAnd' },
    { w: ',', z: 1.6, next: 'keysComma' },
    { w: ' внимания', z: 1.0, next: 'keysAttn' },
    { w: ' от', z: -0.4, next: 'from' },
  ],
  keysAnd: [
    { w: ' значения', z: 4.6, next: 'np' },
    { w: ' векторы', z: 1.0, next: 'np' },
    { w: ' запросы', z: 0.4, next: 'np' },
    { w: ' пароли', z: -1.2, next: 'weird' },
  ],
  keysComma: [
    { w: ' значения', z: 3.4, next: 'np' },
    { w: ' векторы', z: 1.2, next: 'np' },
  ],
  keysAttn: [
    { w: ' и', z: 3.6, next: 'keysAnd' },
    { w: ',', z: 1.5, next: 'keysComma' },
  ],
  from: [
    { w: ' квартиры', z: 2.2, next: 'weirdEnd' },
    { w: ' всех', z: 1.8, next: 'allPrev' },
    { w: ' токенов', z: 1.5, next: 'tokens' },
  ],
  vec: [
    { w: ' K', z: 3.0, next: 'vecK' },
    { w: ' ключей', z: 2.6, next: 'vecKeys' },
    { w: ' внимания', z: 1.2, next: 'np' },
  ],
  vecK: [{ w: ' и', z: 4.4, next: 'vecKAnd' }],
  vecKAnd: [
    { w: ' V', z: 4.8, next: 'np' },
    { w: ' Q', z: 0.6, next: 'np' },
  ],
  vecKeys: [{ w: ' и', z: 4.4, next: 'vecKeysAnd' }],
  vecKeysAnd: [
    { w: ' значений', z: 4.8, next: 'np' },
    { w: ' запросов', z: 0.5, next: 'np' },
  ],
  res: [
    { w: ' проекций', z: 2.4, next: 'np' },
    { w: ' вычислений', z: 2.6, next: 'np' },
    { w: ' прошлых', z: 2.0, next: 'prev' },
  ],
  inter: [
    { w: ' результаты', z: 3.4, next: 'np' },
    { w: ' вычисления', z: 2.2, next: 'np' },
    { w: ' данные', z: 2.0, next: 'np' },
  ],
  all: [
    { w: ' нужное', z: 2.6, next: 'np' },
    { w: ',', z: 1.8, next: 'comma' },
    { w: ' подряд', z: 1.0, next: 'np' },
  ],
  mem: [
    { w: ' модели', z: 2.5, next: 'np' },
    { w: ',', z: 1.5, next: 'comma' },
  ],
  weird: [
    { w: '.', z: 2.0, next: END },
    { w: ' и', z: 1.6, next: 'weirdAnd' },
    { w: ',', z: 1.4, next: 'comma' },
    { w: ' в', z: 1.2, next: 'in' },
  ],
  weirdAnd: [
    { w: ' собачек', z: 1.8, next: 'weirdEnd' },
    { w: ' значения', z: 1.6, next: 'np' },
    { w: ' всё', z: 1.0, next: 'np' },
  ],
  weirdEnd: [
    { w: '.', z: 2.5, next: END },
    { w: ',', z: 1.8, next: 'comma' },
  ],
  np: [
    { w: ' всех', z: 3.0, next: 'allPrev' },
    { w: ' прошлых', z: 2.8, next: 'prev' },
    { w: ' для', z: 1.6, next: 'for' },
    { w: ' в', z: 1.4, next: 'in' },
    { w: ',', z: 1.6, next: 'comma' },
    { w: '.', z: 1.0, next: END },
    { w: ' навсегда', z: -0.8, next: 'weirdEnd' },
  ],
  allPrev: [
    { w: ' прошлых', z: 3.4, next: 'prev' },
    { w: ' предыдущих', z: 3.1, next: 'prev' },
    { w: ' токенов', z: 2.0, next: 'tokens' },
    { w: ' слоёв', z: 1.0, next: 'layers' },
  ],
  prev: [
    { w: ' токенов', z: 4.4, next: 'tokens' },
    { w: ' шагов', z: 2.0, next: 'tokens' },
    { w: ' слов', z: 1.4, next: 'tokens' },
    { w: ' запросов', z: 0.6, next: 'tokens' },
  ],
  for: [
    { w: ' всех', z: 2.6, next: 'allPrev' },
    { w: ' каждого', z: 2.5, next: 'each' },
    { w: ' прошлых', z: 2.0, next: 'prev' },
  ],
  each: [
    { w: ' токена', z: 3.6, next: 'tokens' },
    { w: ' слоя', z: 2.6, next: 'layers' },
  ],
  layers: [
    { w: ',', z: 2.4, next: 'comma' },
    { w: '.', z: 2.0, next: END },
    { w: ' и', z: 1.4, next: 'layersAnd' },
  ],
  layersAnd: [
    { w: ' каждого', z: 2.0, next: 'each' },
    { w: ' всех', z: 1.6, next: 'allPrev' },
  ],
  tokens: [
    { w: ',', z: 3.4, next: 'comma' },
    { w: '.', z: 2.4, next: END },
    { w: ' в', z: 1.6, next: 'in' },
  ],
  in: [
    { w: ' памяти', z: 3.6, next: 'memLoc' },
    { w: ' видеопамяти', z: 2.0, next: 'memLoc' },
    { w: ' HBM', z: 1.5, next: 'memLoc' },
  ],
  memLoc: [
    { w: ' GPU', z: 3.4, next: 'gpu' },
    { w: ',', z: 1.6, next: 'comma' },
    { w: '.', z: 1.4, next: END },
  ],
  gpu: [
    { w: ',', z: 3.0, next: 'comma' },
    { w: '.', z: 2.6, next: END },
  ],
  comma: [
    { w: ' чтобы', z: 4.0, next: 'so' },
    { w: ' и', z: 1.4, next: 'and2' },
    { w: ' поэтому', z: 1.2, next: 'hence' },
  ],
  so: [
    { w: ' не', z: 4.2, next: 'not' },
    { w: ' модель', z: 1.6, next: 'model' },
    { w: ' быстро', z: 0.8, next: 'fast' },
  ],
  not: [
    { w: ' пересчитывать', z: 3.6, next: 'recompute' },
    { w: ' считать', z: 3.0, next: 'recompute' },
    { w: ' повторять', z: 1.6, next: 'recompute' },
  ],
  recompute: [
    { w: ' их', z: 3.8, next: 'them' },
    { w: ' всё', z: 1.4, next: 'them' },
    { w: ' заново', z: 1.2, next: 'again' },
  ],
  them: [
    { w: ' заново', z: 3.6, next: 'again' },
    { w: ' на', z: 2.2, next: 'on' },
    { w: '.', z: 1.6, next: END },
  ],
  on: [{ w: ' каждом', z: 4.0, next: 'eachStep' }],
  eachStep: [{ w: ' шаге', z: 4.4, next: 'again' }],
  again: [
    { w: '.', z: 4.6, next: END },
    { w: ' и', z: 1.0, next: 'andFast' },
  ],
  andFast: [
    { w: ' ответ', z: 2.0, next: 'answer' },
    { w: ' генерация', z: 1.6, next: 'answer' },
  ],
  answer: [
    { w: ' идёт', z: 2.6, next: 'goes' },
    { w: ' ускоряется', z: 2.2, next: 'final' },
  ],
  goes: [
    { w: ' быстрее', z: 3.6, next: 'final' },
    { w: ' быстро', z: 2.0, next: 'final' },
  ],
  final: [{ w: '.', z: 4.0, next: END }],
  model: [
    { w: ' не', z: 3.0, next: 'notModel' },
    { w: ' отвечала', z: 1.8, next: 'fastAns' },
  ],
  notModel: [
    { w: ' пересчитывала', z: 3.4, next: 'recompute' },
    { w: ' считала', z: 2.6, next: 'recompute' },
  ],
  // «чтобы быстро отвечать.»: без второго «быстрее»
  fast: [
    { w: ' отвечать', z: 2.6, next: 'final' },
    { w: ' писать', z: 2.2, next: 'final' },
  ],
  fastAns: [
    { w: ' быстрее', z: 2.6, next: 'final' },
    { w: '.', z: 2.2, next: END },
  ],
  and2: [
    { w: ' это', z: 2.6, next: 'this' },
    { w: ' поэтому', z: 1.4, next: 'hence' },
  ],
  this: [
    { w: ' ускоряет', z: 3.0, next: 'speeds' },
    { w: ' экономит', z: 2.4, next: 'saves' },
  ],
  speeds: [
    { w: ' генерацию', z: 3.0, next: 'final' },
    { w: ' decode', z: 2.6, next: 'final' },
    { w: ' ответ', z: 1.8, next: 'final' },
  ],
  saves: [
    { w: ' время', z: 3.2, next: 'final' },
    { w: ' вычисления', z: 2.6, next: 'final' },
  ],
  hence: [
    { w: ' decode', z: 2.4, next: 'dec' },
    { w: ' генерация', z: 2.2, next: 'dec' },
    { w: ' модель', z: 1.4, next: 'modelWorks' },
  ],
  dec: [
    { w: ' идёт', z: 2.8, next: 'goes' },
    { w: ' быстрее', z: 2.4, next: 'final' },
  ],
  modelWorks: [
    { w: ' работает', z: 2.6, next: 'works' },
    { w: ' отвечает', z: 2.4, next: 'works' },
  ],
  works: [{ w: ' быстрее', z: 3.4, next: 'final' }],
}

export interface ToyStep {
  state: string
  edges: Edge[]
  probs: number[]
  finalProbs: number[]
  kept: boolean[]
  chosen: number
  u?: number
}

export interface ToyRun {
  steps: ToyStep[]
  words: string[]
  finished: boolean
}

function asToken(w: string, i: number): Token {
  return { id: i, raw: w, bytes: [], text: w, special: false }
}

/**
 * Прогон с общими числами u (по одному на шаг): у дорожек с разной температурой случайность одна и та же,
 * отличается только распределение.
 */
export function runToy(us: readonly number[], params: SamplingParamsEx): ToyRun {
  const steps: ToyStep[] = []
  const words: string[] = []
  let state = 's0'
  for (let t = 0; t < MAX_STEPS && state !== END; t++) {
    const edges = GRAPH[state]
    const cands: Candidate[] = edges.map((e, i) => ({ token: asToken(e.w, i), logit: e.z }))
    const u = us[t % us.length]
    const r = sample(cands, params, () => u)
    steps.push({ state, edges, probs: r.probs, finalProbs: r.finalProbs, kept: r.kept, chosen: r.chosen, u: r.u })
    words.push(edges[r.chosen].w)
    state = edges[r.chosen].next
  }
  return { steps, words, finished: state === END }
}

/** Склеить слова в фразу (пунктуация без пробела). */
export function joinWords(words: readonly string[]): string {
  return words.join('')
}
