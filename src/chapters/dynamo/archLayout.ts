/**
 * Геометрия схемы Dynamo (viewBox 1000 × 470) в двух режимах: дезагрегированном и агрегированном.
 * Узлы с одинаковым смыслом имеют одинаковые id, детали по ним лежат в archData.tsx.
 */
import type { DEdge, DNode } from './diagram'

export const ARCH_W = 1000
export const ARCH_H = 470

const FRONT: DNode[] = [
  { id: 'client', x: 16, y: 196, w: 112, h: 58, title: 'Клиент', sub: 'OpenAI SDK, curl' },
  { id: 'fe', x: 150, y: 14, w: 330, h: 442, title: 'Frontend', sub: 'Rust, порт 8000', variant: 'group' },
  { id: 'http', x: 168, y: 46, w: 294, h: 58, title: 'HTTP API', sub: '/v1/chat/completions, SSE' },
  { id: 'pre', x: 168, y: 128, w: 294, h: 58, title: 'Preprocessor', sub: 'шаблон и токенизация → token IDs' },
  { id: 'router', x: 168, y: 210, w: 294, h: 58, title: 'KV router', sub: 'хэши блоков, overlap, стоимость' },
  { id: 'post', x: 168, y: 380, w: 294, h: 58, title: 'Postprocessor', sub: 'детокенизация, stop, парсеры' },
  { id: 'events', x: 486, y: 222, w: 72, h: 28, title: 'events', variant: 'pill' },
  { id: 'planner', x: 884, y: 196, w: 104, h: 64, title: 'Planner', sub: 'сколько реплик' },
]

const FRONT_EDGES: DEdge[] = [
  { id: 'client-http', points: [[72, 196], [72, 75], [168, 75]], label: 'запрос', labelAt: [118, 67], touches: ['client', 'http'] },
  { id: 'http-pre', points: [[315, 104], [315, 128]], touches: ['http', 'pre'] },
  { id: 'pre-router', points: [[315, 186], [315, 210]], touches: ['pre', 'router'] },
  { id: 'post-client', points: [[168, 409], [72, 409], [72, 254]], label: 'SSE', labelAt: [118, 401], touches: ['post', 'client'] },
  { id: 'events-router', points: [[486, 236], [462, 236]], dashed: true, touches: ['events', 'router'] },
]

export const DISAGG_NODES: DNode[] = [
  ...FRONT,
  { id: 'pg', x: 566, y: 14, w: 290, h: 154, title: 'Prefill-воркеры', variant: 'group' },
  { id: 'p1', x: 580, y: 44, w: 126, h: 108, title: 'Prefill 1', sub: 'свой KV cache', role: 'prefill' },
  { id: 'p2', x: 716, y: 44, w: 126, h: 108, title: 'Prefill 2', sub: 'свой KV cache', role: 'prefill' },
  { id: 'nixl', x: 646, y: 194, w: 130, h: 66, title: 'NIXL', sub: 'KV из GPU в GPU' },
  { id: 'dg', x: 566, y: 286, w: 290, h: 170, title: 'Decode-воркеры', variant: 'group' },
  { id: 'd1', x: 580, y: 316, w: 126, h: 124, title: 'Decode 1', sub: 'свой KV cache', role: 'decode' },
  { id: 'd2', x: 716, y: 316, w: 126, h: 124, title: 'Decode 2', sub: 'свой KV cache', role: 'decode' },
]

export const DISAGG_EDGES: DEdge[] = [
  ...FRONT_EDGES,
  { id: 'router-prefill', points: [[462, 218], [476, 218], [476, 98], [566, 98]], label: 'prefill', labelAt: [522, 90], touches: ['router', 'p1', 'p2'] },
  { id: 'router-decode', points: [[462, 260], [476, 260], [476, 372], [566, 372]], label: 'decode', labelAt: [522, 364], touches: ['router', 'd1', 'd2'] },
  { id: 'prefill-nixl', points: [[711, 168], [711, 194]], touches: ['nixl', 'p1', 'p2'] },
  { id: 'nixl-decode', points: [[711, 260], [711, 286]], touches: ['nixl', 'd1', 'd2'] },
  { id: 'decode-post', points: [[566, 424], [462, 424]], label: 'token IDs', labelAt: [514, 416], touches: ['post', 'd1', 'd2'] },
  { id: 'prefill-events', points: [[566, 136], [522, 136], [522, 222]], dashed: true, touches: ['events', 'p1', 'p2'] },
  { id: 'decode-events', points: [[566, 318], [522, 318], [522, 250]], dashed: true, touches: ['events', 'd1', 'd2'] },
  { id: 'planner-prefill', points: [[884, 228], [870, 228], [870, 91], [856, 91]], dashed: true, label: 'реплики', labelAt: [876, 150], labelAnchor: 'start', touches: ['planner'] },
  { id: 'planner-decode', points: [[870, 228], [870, 371], [856, 371]], dashed: true, touches: ['planner'] },
]

export const AGG_NODES: DNode[] = [
  ...FRONT,
  { id: 'wg', x: 566, y: 104, w: 290, h: 262, title: 'Воркеры', variant: 'group' },
  { id: 'w1', x: 580, y: 136, w: 126, h: 214, title: 'Воркер 1', sub: 'prefill и decode на одних GPU' },
  { id: 'w2', x: 716, y: 136, w: 126, h: 214, title: 'Воркер 2', sub: 'prefill и decode на одних GPU' },
]

export const AGG_EDGES: DEdge[] = [
  ...FRONT_EDGES,
  { id: 'router-workers', points: [[462, 218], [476, 218], [476, 176], [566, 176]], label: 'запрос', labelAt: [522, 168], touches: ['router', 'w1', 'w2'] },
  { id: 'workers-post', points: [[566, 336], [500, 336], [500, 409], [462, 409]], label: 'token IDs', labelAt: [532, 328], touches: ['post', 'w1', 'w2'] },
  { id: 'workers-events', points: [[566, 280], [522, 280], [522, 250]], dashed: true, touches: ['events', 'w1', 'w2'] },
  { id: 'planner-workers', points: [[884, 228], [870, 228], [870, 176], [856, 176]], dashed: true, label: 'реплики', labelAt: [876, 160], labelAnchor: 'start', touches: ['planner'] },
]

/** Узел схемы → карточка деталей (оба prefill-воркера описаны одной карточкой и т. д.). */
export function detailKey(nodeId: string): string {
  if (nodeId === 'p1' || nodeId === 'p2' || nodeId === 'pg') return 'prefill'
  if (nodeId === 'd1' || nodeId === 'd2' || nodeId === 'dg') return 'decode'
  if (nodeId === 'w1' || nodeId === 'w2' || nodeId === 'wg') return 'workers'
  return nodeId
}

/* ---------------------------------------------------------------------------------------------
 * Вертикальная раскладка для узких экранов (viewBox 320 × 470): слева frontend, справа пулы воркеров.
 * Id узлов и рёбер те же, что у широкой схемы, поэтому шаги фигуры жизненного цикла работают без изменений.
 * Подписи короче: в узкой колонке длинные не помещаются, детали остаются в карточках.
 * ------------------------------------------------------------------------------------------- */

export const ARCH_V_W = 320
export const ARCH_V_H = 470
/** Уже этой ширины (px) схема переходит на вертикальную раскладку. */
export const ARCH_V_BREAK = 640

const FRONT_V: DNode[] = [
  { id: 'client', x: 4, y: 4, w: 132, h: 46, title: 'Клиент', sub: 'OpenAI SDK, curl' },
  { id: 'fe', x: 14, y: 62, w: 140, h: 352, title: 'Frontend', sub: 'Rust', variant: 'group' },
  { id: 'http', x: 22, y: 92, w: 124, h: 46, title: 'HTTP API', sub: 'OpenAI API, SSE' },
  { id: 'pre', x: 22, y: 150, w: 124, h: 46, title: 'Preprocessor', sub: 'шаблон, токены' },
  { id: 'router', x: 22, y: 208, w: 124, h: 46, title: 'KV router', sub: 'хэши, overlap' },
  { id: 'post', x: 22, y: 346, w: 124, h: 58, title: 'Postprocessor', sub: 'детокенизация, парсеры' },
  { id: 'planner', x: 170, y: 418, w: 132, h: 46, title: 'Planner', sub: 'сколько реплик' },
]

const FRONT_V_EDGES: DEdge[] = [
  { id: 'client-http', points: [[100, 50], [100, 92]], touches: ['client', 'http'] },
  { id: 'http-pre', points: [[84, 138], [84, 150]], touches: ['http', 'pre'] },
  { id: 'pre-router', points: [[84, 196], [84, 208]], touches: ['pre', 'router'] },
  { id: 'post-client', points: [[22, 375], [8, 375], [8, 50]], touches: ['post', 'client'] },
]

export const DISAGG_V_NODES: DNode[] = [
  ...FRONT_V,
  { id: 'events', x: 170, y: 218, w: 58, h: 26, title: 'events', variant: 'pill' },
  { id: 'pg', x: 170, y: 62, w: 132, h: 136, title: 'Prefill-воркеры', variant: 'group' },
  { id: 'p1', x: 178, y: 92, w: 116, h: 46, title: 'Prefill 1', sub: 'свой KV cache', role: 'prefill' },
  { id: 'p2', x: 178, y: 144, w: 116, h: 46, title: 'Prefill 2', sub: 'свой KV cache', role: 'prefill' },
  { id: 'nixl', x: 238, y: 210, w: 64, h: 42, title: 'NIXL' },
  { id: 'dg', x: 170, y: 264, w: 132, h: 136, title: 'Decode-воркеры', variant: 'group' },
  { id: 'd1', x: 178, y: 294, w: 116, h: 46, title: 'Decode 1', sub: 'свой KV cache', role: 'decode' },
  { id: 'd2', x: 178, y: 346, w: 116, h: 46, title: 'Decode 2', sub: 'свой KV cache', role: 'decode' },
]

export const DISAGG_V_EDGES: DEdge[] = [
  ...FRONT_V_EDGES,
  { id: 'events-router', points: [[170, 231], [146, 231]], dashed: true, touches: ['events', 'router'] },
  { id: 'router-prefill', points: [[146, 216], [158, 216], [158, 115], [170, 115]], touches: ['router', 'p1', 'p2'] },
  { id: 'router-decode', points: [[146, 246], [158, 246], [158, 317], [170, 317]], touches: ['router', 'd1', 'd2'] },
  { id: 'prefill-nixl', points: [[270, 198], [270, 210]], touches: ['nixl', 'p1', 'p2'] },
  { id: 'nixl-decode', points: [[270, 252], [270, 264]], touches: ['nixl', 'd1', 'd2'] },
  { id: 'decode-post', points: [[170, 375], [146, 375]], touches: ['post', 'd1', 'd2'] },
  { id: 'prefill-events', points: [[192, 198], [192, 218]], dashed: true, touches: ['events', 'p1', 'p2'] },
  { id: 'decode-events', points: [[192, 264], [192, 244]], dashed: true, touches: ['events', 'd1', 'd2'] },
  { id: 'planner-prefill', points: [[302, 441], [312, 441], [312, 130], [302, 130]], dashed: true, touches: ['planner'] },
  { id: 'planner-decode', points: [[312, 441], [312, 332], [302, 332]], dashed: true, touches: ['planner'] },
]

export const AGG_V_NODES: DNode[] = [
  ...FRONT_V,
  { id: 'events', x: 170, y: 232, w: 58, h: 26, title: 'events', variant: 'pill' },
  { id: 'wg', x: 170, y: 62, w: 132, h: 158, title: 'Воркеры', variant: 'group' },
  { id: 'w1', x: 178, y: 92, w: 116, h: 56, title: 'Воркер 1', sub: 'prefill и decode' },
  { id: 'w2', x: 178, y: 154, w: 116, h: 56, title: 'Воркер 2', sub: 'prefill и decode' },
]

export const AGG_V_EDGES: DEdge[] = [
  ...FRONT_V_EDGES,
  { id: 'events-router', points: [[170, 245], [146, 245]], dashed: true, touches: ['events', 'router'] },
  { id: 'router-workers', points: [[146, 216], [158, 216], [158, 120], [170, 120]], touches: ['router', 'w1', 'w2'] },
  { id: 'workers-events', points: [[192, 220], [192, 232]], dashed: true, touches: ['events', 'w1', 'w2'] },
  { id: 'workers-post', points: [[280, 220], [280, 375], [146, 375]], touches: ['post', 'w1', 'w2'] },
  { id: 'planner-workers', points: [[302, 441], [312, 441], [312, 141], [302, 141]], dashed: true, touches: ['planner'] },
]

/** Раскладка схемы под режим и ширину контейнера. */
export function archLayout(mode: 'disagg' | 'agg', narrow: boolean): { width: number; height: number; nodes: DNode[]; edges: DEdge[]; minWidth: number } {
  if (narrow) {
    return mode === 'disagg'
      ? { width: ARCH_V_W, height: ARCH_V_H, nodes: DISAGG_V_NODES, edges: DISAGG_V_EDGES, minWidth: 280 }
      : { width: ARCH_V_W, height: ARCH_V_H, nodes: AGG_V_NODES, edges: AGG_V_EDGES, minWidth: 280 }
  }
  return mode === 'disagg'
    ? { width: ARCH_W, height: ARCH_H, nodes: DISAGG_NODES, edges: DISAGG_EDGES, minWidth: 860 }
    : { width: ARCH_W, height: ARCH_H, nodes: AGG_NODES, edges: AGG_EDGES, minWidth: 860 }
}
