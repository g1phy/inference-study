/**
 * Условная маршрутизация MoE для иллюстрации: оценки роутера псевдослучайные, но детерминированные
 * для пары (токен, слой). Дальше схема выбора как у настоящих роутеров:
 * - Qwen3-MoE: softmax по всем экспертам, top-k, веса выбранных делятся на свою сумму (norm_topk_prob = true);
 * - gpt-oss: top-k по сырым оценкам, softmax только по выбранным (те же веса, что у схемы Qwen3);
 * - DeepSeek-V3: sigmoid оценок, эксперты разбиты на группы, у группы оценка = сумма двух лучших;
 *   остаются лучшие группы, top-k выбирается внутри них, веса делятся на свою сумму
 *   (в модели их ещё умножают на routed_scaling_factor 2,5; поправку балансировки здесь не моделируем).
 */
import { fmix32, mulberry32, normal } from '@/engine/rng'
import { softmax } from '@/engine/sampling'

export type RouterScheme = 'softmax-topk' | 'topk-softmax' | 'sigmoid-group'

export interface RouterSpec {
  scheme: RouterScheme
  /** Первый слой с экспертами (1-based): у DeepSeek-V3 первые 3 слоя плотные (first_k_dense_replace = 3). */
  firstMoeLayer: number
  /** sigmoid-group: число групп и сколько лучших групп остаётся (n_group, topk_group). */
  groups?: number
  topGroups?: number
}

export const ROUTERS: Record<string, RouterSpec> = {
  'qwen3-30b-a3b': { scheme: 'softmax-topk', firstMoeLayer: 1 },
  'qwen3-235b-a22b': { scheme: 'softmax-topk', firstMoeLayer: 1 },
  'gpt-oss-120b': { scheme: 'topk-softmax', firstMoeLayer: 1 },
  'deepseek-v3': { scheme: 'sigmoid-group', firstMoeLayer: 4, groups: 8, topGroups: 4 },
}

const DEFAULT_ROUTER: RouterSpec = { scheme: 'softmax-topk', firstMoeLayer: 1 }

export function routerOf(modelId: string): RouterSpec {
  return ROUTERS[modelId] ?? DEFAULT_ROUTER
}

export interface RouteResult {
  /** Выбранные эксперты по убыванию веса. */
  chosen: Array<{ expert: number; weight: number; prob: number }>
  /** sigmoid-group: номера групп, которые оставил роутер. */
  groups?: number[]
}

function rawScores(tokenId: number, layer: number, experts: number): number[] {
  const rng = mulberry32(fmix32((Math.imul(tokenId + 1, 0x9e3779b1) ^ Math.imul(layer + 7, 0x85ebca6b)) >>> 0))
  return Array.from({ length: experts }, () => normal(rng, 0, 1.3))
}

export function routeToken(tokenId: number, layer: number, experts: number, k: number, router: RouterSpec = DEFAULT_ROUTER): RouteResult {
  const logits = rawScores(tokenId, layer, experts)

  if (router.scheme === 'sigmoid-group' && router.groups && router.topGroups) {
    const scores = logits.map((z) => 1 / (1 + Math.exp(-z)))
    const size = experts / router.groups
    const groupScore = Array.from({ length: router.groups }, (_, g) => {
      const top2 = scores.slice(g * size, (g + 1) * size).sort((a, b) => b - a)
      return top2[0] + (top2[1] ?? 0)
    })
    const groups = groupScore
      .map((sc, g) => ({ g, sc }))
      .sort((a, b) => b.sc - a.sc)
      .slice(0, router.topGroups)
      .map((x) => x.g)
    const allowed = new Set(groups)
    const order = scores
      .map((p, e) => ({ e, p }))
      .filter((o) => allowed.has(Math.floor(o.e / size)))
      .sort((a, b) => b.p - a.p)
      .slice(0, k)
    const sum = order.reduce((s, o) => s + o.p, 0)
    return { chosen: order.map((o) => ({ expert: o.e, prob: o.p, weight: o.p / sum })), groups: groups.sort((a, b) => a - b) }
  }

  // softmax → top-k → нормировка и top-k → softmax по выбранным дают одни и те же веса
  const probs = softmax(logits, 1)
  const order = probs
    .map((p, e) => ({ e, p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, k)
  const sum = order.reduce((s, o) => s + o.p, 0)
  return { chosen: order.map((o) => ({ expert: o.e, prob: o.p, weight: o.p / sum })) }
}

/** Множество экспертов, задетых токенами 0..upto (включительно) в слое. */
export function touchedExperts(
  tokenIds: readonly number[],
  upto: number,
  layer: number,
  experts: number,
  k: number,
  router: RouterSpec = DEFAULT_ROUTER,
): Set<number> {
  const s = new Set<number>()
  for (let i = 0; i <= upto && i < tokenIds.length; i++) for (const c of routeToken(tokenIds[i], layer, experts, k, router).chosen) s.add(c.expert)
  return s
}
