import type { Token } from '@/engine/types'

/**
 * Данные для витрины ui-kit. Это иллюстрация компонентов, не факты для глав:
 * ID обычных токенов условные (настоящие только у спецтокенов Qwen3).
 */

/* ---------- byte-level BPE: байт → символ словаря (как в GPT-2 / Qwen) ---------- */
const BYTE_TO_CHAR: string[] = (() => {
  const bs: number[] = []
  for (let b = 33; b <= 126; b++) bs.push(b)
  for (let b = 161; b <= 172; b++) bs.push(b)
  for (let b = 174; b <= 255; b++) bs.push(b)
  const cs = [...bs]
  let n = 0
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b)
      cs.push(256 + n)
      n++
    }
  }
  const map: string[] = []
  bs.forEach((b, i) => (map[b] = String.fromCodePoint(cs[i])))
  return map
})()

const enc = new TextEncoder()
const dec = new TextDecoder('utf-8', { fatal: false })

/** Токен из текста (или из сырых байтов) с условным ID. */
export function tok(text: string, id: number, opts: { special?: boolean; bytes?: number[] } = {}): Token {
  const bytes = opts.bytes ?? [...enc.encode(text)]
  const special = Boolean(opts.special)
  return {
    id,
    raw: special ? text : bytes.map((b) => BYTE_TO_CHAR[b]).join(''),
    bytes,
    text: opts.bytes ? dec.decode(new Uint8Array(bytes)) : text,
    special,
  }
}

/** Промпт после шаблона Qwen3 (спецтокены с настоящими ID, остальные условные). */
export const PROMPT_TOKENS: Token[] = [
  tok('<|im_start|>', 151644, { special: true }),
  tok('user', 872),
  tok('\n', 198),
  tok('Объ', 129870),
  tok('ясни', 132044),
  tok(' простыми', 135563),
  tok(' словами', 133429),
  tok(',', 11),
  tok(' что', 47389),
  tok(' такое', 132678),
  tok(' KV', 84648),
  tok(' cache', 6500),
  tok('?', 30),
  tok('<|im_end|>', 151645, { special: true }),
  tok('\n', 198),
  tok('<|im_start|>', 151644, { special: true }),
  tok('assistant', 77091),
  tok('\n', 198),
]

/** Сгенерированные токены, включая эмодзи, разрезанный на байты. */
export const GENERATED_TOKENS: Token[] = [
  tok('<think>', 151667, { special: true }),
  tok('\n', 198),
  tok('Нужно', 128463),
  tok(' объяснить', 141296),
  tok(' кэш', 147019),
  tok('.', 13),
  tok('\n', 198),
  tok('</think>', 151668, { special: true }),
  tok('\n\n', 271),
  tok('KV', 82707),
  tok(' cache', 6500),
  tok(' хранит', 140998),
  tok('\t', 197),
  tok(' ', 220, { bytes: [0xf0, 0x9f] }),
  tok(' ', 247, { bytes: [0x99, 0x82] }),
]

/* ---------- детерминированный ГПСЧ для примеров ---------- */
export function rng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export const WORDS = ['Кошка', 'не', 'перепрыгнула', 'забор', ',', 'потому', 'что', 'она', 'устала']

/** Упрощённый пример: «сырые» оценки Q·Kᵀ/√d_k с маской и softmax по строкам. */
export const TOY = (() => {
  const r = rng(7)
  const n = WORDS.length
  const scores = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (j > i ? -Infinity : Math.round((r() * 2.4 - 0.8) * 10) / 10)),
  )
  scores[7][0] = 2.6 // «она» → «Кошка»
  scores[7][3] = 0.4 // «она» → «забор»
  scores[8][0] = 1.8
  const weights = scores.map((row) => {
    const finite = row.filter(Number.isFinite)
    const mx = Math.max(...finite)
    const ex = row.map((v) => (Number.isFinite(v) ? Math.exp(v - mx) : 0))
    const sum = ex.reduce((a, b) => a + b, 0)
    return ex.map((e) => e / sum)
  })
  const q = [
    [0.9, 0.1],
    [0.0, 0.4],
    [0.3, -0.8],
    [0.2, -0.5],
    [-0.1, 0.0],
    [0.1, 0.6],
    [0.0, 0.5],
    [1.2, 0.3],
    [0.7, -0.2],
  ]
  return { scores, weights, q }
})()

/** Паттерн головы для heatmap: attention sink на первый токен + предыдущий токен + себя. */
export const HEAD_TOKENS = ['<|im_start|>', 'user', '↵', 'Объ', 'ясни', '·что', '·такое', '·KV', '·cache', '?', '<|im_end|>', '↵']
export const HEAD_WEIGHTS: number[][] = (() => {
  const n = HEAD_TOKENS.length
  const r = rng(42)
  return Array.from({ length: n }, (_, i) => {
    const row = Array.from({ length: n }, (_, j) => {
      if (j > i) return 0
      let w = 0.04 + r() * 0.06
      if (j === 0) w += 0.9
      if (j === i - 1) w += 0.7
      if (j === i) w += 0.3
      return w
    })
    const s = row.reduce((a, b) => a + b, 0)
    return row.map((v) => v / s)
  })
})()

/** Кандидаты следующего токена (условные вероятности) для BarChart. */
export const CANDIDATES = [
  { label: ' России', value: 0.62 },
  { label: ' Российской', value: 0.14 },
  { label: ' РФ', value: 0.09 },
  { label: ' нашей', value: 0.06, muted: true },
  { label: ' страны', value: 0.04, muted: true },
  { label: 'Конечно', value: 0.03, struck: true },
]

/** Roofline: достижимые TFLOPS = min(пик, полоса × интенсивность). H100 SXM и H200, BF16 dense. */
export function rooflinePoints(peakTflops: number, bwTBs: number): Array<[number, number]> {
  const xs = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 295, 500, 1000, 2000, 5000, 10000]
  const ridge = peakTflops / bwTBs
  const all = [...new Set([...xs, Math.round(ridge)])].sort((a, b) => a - b)
  return all.map((x) => [x, Math.min(peakTflops, bwTBs * x)])
}

/** Условные данные: TTFT p50 от числа одновременных запросов. */
export const TTFT_SERIES = (() => {
  const xs = [1, 4, 8, 16, 32, 48, 64, 96, 128]
  return {
    agg: xs.map((c) => [c, Math.round(60 + c * 4.2 + (c > 64 ? (c - 64) ** 1.6 : 0))] as [number, number]),
    disagg: xs.map((c) => [c, Math.round(75 + c * 2.1 + (c > 96 ? (c - 96) ** 1.5 : 0))] as [number, number]),
  }
})()

export const REQUEST_JSON = `{
  "model": "Qwen/Qwen3-8B",
  "messages": [
    { "role": "system", "content": "Ты полезный ассистент." },
    { "role": "user", "content": "Объясни простыми словами, что такое KV cache?" }
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": true
}`

export const VLLM_BASH = `# OpenAI-совместимый сервер vLLM
$ vllm serve Qwen/Qwen3-8B \\
    --enable-prefix-caching \\
    --max-num-batched-tokens 8192 \\
    --gpu-memory-utilization 0.90 \\
    --reasoning-parser qwen3`

export const DAEMONSET_YAML = `apiVersion: apps/v1
kind: DaemonSet           # по одному поду кэша на каждой ноде
metadata:
  name: kv-cache-l3
spec:
  template:
    spec:
      containers:
        - name: cache
          image: "lmcache/server:latest"
          resources:
            limits:
              memory: 256Gi
          volumeMounts:
            - name: nvme
              mountPath: /mnt/nvme
      volumes:
        - name: nvme
          hostPath: { path: /mnt/nvme }`

export const JINJA_SNIPPET = `{%- for message in messages %}
  {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
  {{- '<|im_start|>assistant\\n' }}
{%- endif %}`

export const PYTHON_SNIPPET = `from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
stream = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Привет!"}],
    stream=True,  # первый токен придёт через TTFT
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")`

export const RENDERED_TEMPLATE = `<|im_start|>system
Ты полезный ассистент.<|im_end|>
<|im_start|>user
Объясни простыми словами, что такое KV cache?<|im_end|>
<|im_start|>assistant
<think>`
