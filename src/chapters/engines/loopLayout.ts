/**
 * Геометрия схемы «внутри движка» (vLLM V1) для главы 18, viewBox 1000 × 470.
 * Три процесса: API server, EngineCore, GPU-воркеры (notes-engines.md §1.3-1.4).
 * Между EngineCore и GPU-воркерами зазор 50: подпись стрелки «что считать» помещается в него, не задевая рамок.
 */
import type { DEdge, DNode } from '../dynamo/diagram'

export const LOOP_W = 1000
export const LOOP_H = 470

export const LOOP_NODES: DNode[] = [
  { id: 'client', x: 12, y: 200, w: 100, h: 58, title: 'Клиент', sub: 'HTTP' },
  { id: 'api', x: 136, y: 14, w: 250, h: 442, title: 'API server', sub: 'процесс, CPU', variant: 'group' },
  { id: 'http', x: 152, y: 46, w: 218, h: 62, title: 'HTTP, OpenAI API', sub: 'FastAPI и Uvicorn' },
  { id: 'input', x: 152, y: 140, w: 218, h: 70, title: 'Шаблон и токенизация', sub: 'текст → EngineCoreRequest' },
  { id: 'output', x: 152, y: 358, w: 218, h: 82, title: 'Детокенизация и парсеры', sub: 'токены → текст, reasoning, SSE' },
  { id: 'core', x: 420, y: 14, w: 278, h: 442, title: 'EngineCore', sub: 'процесс, busy loop', variant: 'group' },
  { id: 'sched', x: 436, y: 46, w: 246, h: 154, title: 'Scheduler', sub: 'очереди и бюджет токенов', align: 'top' },
  { id: 'kvm', x: 436, y: 222, w: 246, h: 132, title: 'KV cache manager', sub: 'блоки по 16, prefix cache', align: 'top' },
  { id: 'som', x: 436, y: 376, w: 246, h: 64, title: 'Structured output', sub: 'маски грамматики на CPU' },
  { id: 'gpu', x: 748, y: 14, w: 248, h: 442, title: 'GPU-воркеры', sub: 'процесс на GPU', variant: 'group' },
  { id: 'runner', x: 764, y: 46, w: 216, h: 92, title: 'Model runner', sub: 'batch без padding, CUDA graphs' },
  { id: 'kernels', x: 764, y: 170, w: 216, h: 96, title: 'Модель и ядра', sub: 'слои трансформера, FlashAttention, FlashInfer' },
  { id: 'sampler', x: 764, y: 298, w: 216, h: 82, title: 'Sampler', sub: 'маска, temperature, выбор токена' },
]

export const LOOP_EDGES: DEdge[] = [
  { id: 'client-http', points: [[62, 200], [62, 77], [152, 77]], label: 'POST', labelAt: [106, 69], touches: ['client', 'http'] },
  { id: 'http-input', points: [[261, 108], [261, 140]], touches: ['http', 'input'] },
  { id: 'input-sched', points: [[370, 175], [436, 175]], label: 'ZMQ', labelAt: [403, 167], touches: ['input', 'sched'] },
  { id: 'sched-kvm', points: [[559, 200], [559, 222]], touches: ['sched', 'kvm'] },
  { id: 'sched-runner', points: [[682, 92], [764, 92]], label: 'что\nсчитать', labelAt: [723, 70], labelSize: 11, touches: ['sched', 'runner'] },
  { id: 'runner-kernels', points: [[872, 138], [872, 170]], touches: ['runner', 'kernels'] },
  { id: 'kernels-sampler', points: [[872, 266], [872, 298]], label: 'логиты', labelAt: [901, 286], touches: ['kernels', 'sampler'] },
  { id: 'sampler-sched', points: [[764, 340], [723, 340], [723, 160], [682, 160]], touches: ['sampler', 'sched'] },
  { id: 'som-sampler', points: [[682, 410], [731, 410], [731, 366], [764, 366]], dashed: true, touches: ['som', 'sampler'] },
  // токены уходят из EngineCore (scheduler дописал их в запрос) обратно в API server
  { id: 'core-output', points: [[436, 190], [404, 190], [404, 400], [370, 400]], label: 'ZMQ', labelAt: [404, 300], touches: ['sched', 'output'] },
  { id: 'output-client', points: [[152, 412], [62, 412], [62, 258]], label: 'SSE', labelAt: [106, 404], touches: ['output', 'client'] },
]
