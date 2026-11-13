/**
 * Модель фигуры Kubernetes (глава 19): YAML DynamoGraphDeployment и упрощённая раскладка подов по нодам.
 * Чистые функции, React не нужен.
 */
export type Comp = 'operator' | 'frontend' | 'prefill' | 'decode' | 'planner' | 'daemon'

export interface Pod {
  name: string
  comp: Comp
  gpus: number
}

export interface NodeView {
  name: string
  gpu: boolean
  pods: Pod[]
  /** Кому принадлежит каждый из 8 GPU (индекс пода в pods) или null. */
  slots: Array<number | null>
}

export const GPUS_PER_NODE = 8

export const COMP_LABEL: Record<Comp, string> = {
  operator: 'Оператор',
  frontend: 'Frontend',
  prefill: 'Prefill',
  decode: 'Decode',
  planner: 'Planner',
  daemon: 'DaemonSet',
}

export const COMP_HINT: Record<Comp, string> = {
  operator: 'Deployment, один на кластер: следит за DynamoGraphDeployment и создаёт поды и сервисы компонентов',
  frontend: 'под на CPU: HTTP, preprocessor, роутер и postprocessor; снаружи доступен через Service на порту 8000',
  prefill: 'поды-воркеры: движок на своих GPU, роль prefill',
  decode: 'поды-воркеры: движок на своих GPU, роль decode',
  planner: 'под Planner: по метрикам меняет replicas у prefill и decode прямо в DynamoGraphDeployment',
  daemon: 'по одному поду на каждой GPU-ноде: драйвер и device plugin GPU Operator, RDMA, при желании общий кэш KV ноды',
}

export function buildYaml(prefill: number, decode: number, gpus: number, planner: boolean): { code: string; lines: Record<Comp, number[]> } {
  const out: string[] = []
  const lines: Record<Comp, number[]> = { operator: [], frontend: [], prefill: [], decode: [], planner: [], daemon: [] }
  const push = (comp: Comp | null, s: string) => {
    out.push(s)
    if (comp) lines[comp].push(out.length)
  }
  push(null, 'apiVersion: nvidia.com/v1beta1')
  push('operator', 'kind: DynamoGraphDeployment')
  push(null, 'metadata:')
  push(null, '  name: qwen3-8b-disagg')
  push(null, 'spec:')
  push(null, '  components:')
  push('frontend', '    - name: frontend')
  push('frontend', '      type: frontend')
  push('frontend', '      replicas: 1')
  const worker = (comp: 'prefill' | 'decode', n: number) => {
    push(comp, `    - name: ${comp}`)
    push(comp, `      type: ${comp}`)
    push(comp, `      replicas: ${n}`)
    push(comp, '      sharedMemorySize: 16Gi')
    push(comp, '      podTemplate:')
    push(comp, '        spec:')
    push(comp, '          containers:')
    push(comp, '            - name: main')
    push(comp, '              image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0')
    push(comp, '              command: [python3, -m, dynamo.vllm]')
    push(comp, `              args: [--model, Qwen/Qwen3-8B, --disaggregation-mode, ${comp}]`)
    push(comp, '              resources:')
    push(comp, `                limits: { nvidia.com/gpu: "${gpus}" }`)
  }
  worker('prefill', prefill)
  worker('decode', decode)
  if (planner) {
    push('planner', '    - name: planner')
    push('planner', '      type: planner')
    push('planner', '      replicas: 1')
  }
  return { code: out.join('\n'), lines }
}

/** Раскладка подов по нодам: CPU-нода для frontend и Planner, GPU-ноды по 8 GPU, воркеры подряд. */
export function layout(prefill: number, decode: number, gpus: number, planner: boolean, cacheDs: boolean): NodeView[] {
  const cpu: NodeView = {
    name: 'CPU-нода',
    gpu: false,
    pods: [{ name: 'dynamo-operator', comp: 'operator', gpus: 0 }, { name: 'frontend-0', comp: 'frontend', gpus: 0 }, ...(planner ? [{ name: 'planner-0', comp: 'planner' as Comp, gpus: 0 }] : [])],
    slots: [],
  }
  const workers: Pod[] = [
    ...Array.from({ length: prefill }, (_, i) => ({ name: `prefill-${i}`, comp: 'prefill' as Comp, gpus })),
    ...Array.from({ length: decode }, (_, i) => ({ name: `decode-${i}`, comp: 'decode' as Comp, gpus })),
  ]
  const nodes: NodeView[] = []
  const daemon = (): Pod[] => [
    { name: 'gpu-operator', comp: 'daemon', gpus: 0 },
    { name: 'rdma', comp: 'daemon', gpus: 0 },
    ...(cacheDs ? [{ name: 'kv-cache (LMCache MP)', comp: 'daemon' as Comp, gpus: 0 }] : []),
  ]
  let cur: NodeView | null = null
  let free = 0
  for (const w of workers) {
    if (!cur || free < w.gpus) {
      cur = { name: `GPU-нода ${nodes.length + 1}`, gpu: true, pods: daemon(), slots: Array(GPUS_PER_NODE).fill(null) }
      nodes.push(cur)
      free = GPUS_PER_NODE
    }
    const idx = cur.pods.push(w) - 1
    let left = w.gpus
    for (let s = 0; s < GPUS_PER_NODE && left > 0; s++) {
      if (cur.slots[s] === null) {
        cur.slots[s] = idx
        left--
      }
    }
    free -= w.gpus
  }
  return [cpu, ...nodes]
}

