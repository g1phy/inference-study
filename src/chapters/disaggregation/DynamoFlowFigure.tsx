import { useMemo, useState } from 'react'
import { CodeBlock, Figure, Player, Segmented, fmt, useElementSize, usePlayer } from '@/ui'
import { useSettings } from '@/state/settings'
import { kvBytesForSeq } from '@/engine/perf'
import { simConfig } from '../batching/config'
import { Diagram, type DEdge, type DNode, type Pt } from '../dynamo/diagram'
import { useReaderPromptTokens } from './promptTokens'

type Backend = 'vllm' | 'sglang'
type BoxId = 'client' | 'pre' | 'router' | 'post' | 'p1' | 'p2' | 'd1' | 'd2'
type EdgeId = 'client-pre' | 'router-p1' | 'p1-router' | 'router-d1' | 'nixl-data' | 'nixl-req' | 'd1-post' | 'post-client'

interface StepEdge {
  id: EdgeId
  label?: string
  dashed?: boolean
}

interface FlowStep {
  title: string
  text: string
  /** Активные узлы. */
  nodes: BoxId[]
  /** Рёбра шага (геометрия зависит от раскладки); пакет едет по первому. */
  edges: StepEdge[]
  /** Кто начинает передачу KV (подпись у узла). */
  initiator?: { node: 'p1' | 'd1'; text: string }
  code?: { title: string; lang: 'json' | 'text'; code: string }
}

interface Layout {
  vb: [number, number]
  box: Record<BoxId, [number, number, number, number]>
  groups: Array<{ id: string; title: string; rect: [number, number, number, number] }>
  edges: Record<EdgeId, { points: Pt[]; labelAt: Pt; anchor: 'start' | 'middle' | 'end' }>
  /** Где подписать инициатора передачи KV. */
  initiatorAt: Pt
  initiatorAnchor: 'start' | 'end'
}

const BOX_TITLE: Record<BoxId, string> = {
  client: 'Клиент',
  pre: 'Preprocessor',
  router: 'PrefillRouter',
  post: 'Postprocessor',
  p1: 'Prefill 1',
  p2: 'Prefill 2',
  d1: 'Decode 1',
  d2: 'Decode 2',
}
const BOX_ROLE: Partial<Record<BoxId, string>> = { p1: 'prefill', p2: 'prefill', d1: 'decode', d2: 'decode' }

/** Широкая раскладка: frontend слева, пулы справа. */
const WIDE: Layout = {
  vb: [900, 300],
  box: {
    client: [10, 126, 100, 46],
    pre: [196, 52, 180, 44],
    router: [196, 128, 180, 44],
    post: [196, 204, 180, 44],
    p1: [626, 54, 124, 46],
    p2: [762, 54, 114, 46],
    d1: [626, 222, 124, 46],
    d2: [762, 222, 114, 46],
  },
  groups: [
    { id: 'g-fe', title: 'Frontend', rect: [176, 14, 220, 272] },
    { id: 'g-p', title: 'Prefill-пул', rect: [610, 14, 280, 104] },
    { id: 'g-d', title: 'Decode-пул', rect: [610, 182, 280, 104] },
  ],
  edges: {
    'client-pre': { points: [[110, 149], [150, 149], [150, 74], [196, 74]], labelAt: [12, 112], anchor: 'start' },
    'router-p1': { points: [[376, 141], [480, 141], [480, 77], [626, 77]], labelAt: [488, 126], anchor: 'start' },
    'p1-router': { points: [[626, 92], [500, 92], [500, 158], [376, 158]], labelAt: [404, 178], anchor: 'start' },
    'router-d1': { points: [[376, 164], [460, 164], [460, 245], [626, 245]], labelAt: [468, 236], anchor: 'start' },
    'nixl-data': { points: [[668, 100], [668, 222]], labelAt: [660, 154], anchor: 'end' },
    'nixl-req': { points: [[712, 222], [712, 100]], labelAt: [720, 154], anchor: 'start' },
    'd1-post': { points: [[626, 240], [376, 240]], labelAt: [500, 230], anchor: 'middle' },
    'post-client': { points: [[196, 226], [60, 226], [60, 172]], labelAt: [66, 248], anchor: 'start' },
  },
  initiatorAt: [720, 154],
  initiatorAnchor: 'start',
}

/** Узкая раскладка для телефона: всё друг под другом. Выбранные воркеры справа, к ним ведёт правый коридор. */
const TALL: Layout = {
  vb: [360, 528],
  box: {
    client: [120, 8, 120, 40],
    pre: [24, 100, 190, 40],
    router: [24, 154, 190, 40],
    post: [24, 208, 190, 40],
    p2: [24, 330, 140, 44],
    p1: [196, 330, 140, 44],
    d2: [24, 462, 140, 44],
    d1: [196, 462, 140, 44],
  },
  groups: [
    { id: 'g-fe', title: 'Frontend', rect: [8, 66, 344, 196] },
    { id: 'g-p', title: 'Prefill-пул', rect: [8, 296, 344, 92] },
    { id: 'g-d', title: 'Decode-пул', rect: [8, 428, 344, 92] },
  ],
  edges: {
    'client-pre': { points: [[160, 48], [160, 100]], labelAt: [348, 61], anchor: 'end' },
    'router-p1': { points: [[214, 182], [300, 182], [300, 330]], labelAt: [292, 284], anchor: 'end' },
    'p1-router': { points: [[250, 330], [250, 166], [214, 166]], labelAt: [242, 284], anchor: 'end' },
    'router-d1': { points: [[214, 190], [344, 190], [344, 484], [336, 484]], labelAt: [336, 284], anchor: 'end' },
    'nixl-data': { points: [[240, 374], [240, 462]], labelAt: [232, 404], anchor: 'end' },
    'nixl-req': { points: [[300, 462], [300, 374]], labelAt: [292, 426], anchor: 'end' },
    'd1-post': { points: [[336, 476], [344, 476], [344, 228], [214, 228]], labelAt: [336, 284], anchor: 'end' },
    'post-client': { points: [[214, 236], [256, 236], [256, 28], [240, 28]], labelAt: [264, 124], anchor: 'start' },
  },
  initiatorAt: [232, 420],
  initiatorAnchor: 'end',
}

/** Номера блоков KV для примера: 17, 18, 19, …, последний. */
function blockList(blocks: number): string {
  const first = 17
  if (blocks <= 4) return Array.from({ length: blocks }, (_, i) => first + i).join(', ')
  return `${first}, ${first + 1}, ${first + 2}, "…", ${first + blocks - 1}`
}

function vllmSteps(kv: string, tokens: number, blocks: number): FlowStep[] {
  const ids = `"…${fmt.int(tokens)} ID…"`
  const lastBlock = 16 + blocks
  return [
    {
      title: 'Запрос приходит во frontend',
      text: `Preprocessor во frontend рендерит шаблон чата и токенизирует промпт: ${fmt.tokens(tokens)}. Токены нужны уже здесь: в режиме kv роутер ищет по ним, у кого из воркеров лежит префикс в KV cache.`,
      nodes: ['client', 'pre'],
      edges: [{ id: 'client-pre', label: 'POST /v1/chat/completions' }],
      code: {
        title: 'клиент → frontend',
        lang: 'json',
        code: '{"model": "Qwen/Qwen3-8B",\n "messages": [{"role": "user", "content": "…"}],\n "max_tokens": 512, "stream": true}',
      },
    },
    {
      title: 'PrefillRouter первым выбирает prefill-воркер',
      text: 'Prefill-запрос делают клоном исходного с max_tokens = 1: воркер должен посчитать KV всего промпта и остановиться. Prefill-воркер выбирает роутер: в режиме --router-mode kv по перекрытию префикса и нагрузке, по умолчанию просто по очереди (round-robin).',
      nodes: ['router', 'p1'],
      edges: [{ id: 'router-p1', label: 'клон, max_tokens = 1' }],
      code: {
        title: 'PrefillRouter → prefill-воркер (упрощённо)',
        lang: 'json',
        code: `{"token_ids": [${ids}],\n "stop_conditions": {"max_tokens": 1},\n "kv_transfer_params": {"do_remote_decode": true,\n   "do_remote_prefill": false}}`,
      },
    },
    {
      title: 'Prefill-воркер считает KV',
      text: `vLLM считает KV для ${fmt.count(blocks, ['блока', 'блоков', 'блоков'])} по 16 токенов (${kv}) и сэмплирует один токен. Блоки не освобождаются сразу: они ждут, пока их заберёт decode-воркер.`,
      nodes: ['p1'],
      edges: [],
      code: {
        title: 'что происходит на prefill-воркере',
        lang: 'text',
        code: `prefill: ${fmt.tokens(tokens)} → KV в блоках 17…${lastBlock}\nсэмплирован 1 токен (клиенту не уходит)\nблоки удерживаются до чтения (lease)`,
      },
    },
    {
      title: 'В ответ приходит квитанция, а не данные',
      text: 'Prefill-воркер возвращает disaggregated_params: номера блоков, адрес своего NIXL-агента и размер TP. Сам KV остаётся в его видеопамяти. Токен prefill-воркера клиенту не идёт: первый видимый токен посчитает decode.',
      nodes: ['p1', 'router'],
      edges: [{ id: 'p1-router', label: 'disaggregated_params' }],
      code: {
        title: 'prefill-воркер → PrefillRouter (значения условные)',
        lang: 'json',
        code: `{"token_ids": [1234],\n "disaggregated_params": {"kv_transfer_params": {\n   "do_remote_prefill": true,\n   "remote_block_ids": [[${blockList(blocks)}]],\n   "remote_engine_id": "e3f1…",\n   "remote_host": "10.0.3.17", "remote_port": 5600,\n   "tp_size": 1}}}`,
      },
    },
    {
      title: 'Decode-воркер выбирается только по нагрузке',
      text: 'PrefillRouter возвращает исходный max_tokens = 512, кладёт квитанцию в поле prefill_result и выбирает наименее загруженный decode-воркер. Кэш префикса здесь не учитывается: KV всё равно приедет с prefill-воркера.',
      nodes: ['router', 'd1'],
      edges: [{ id: 'router-d1', label: 'запрос + prefill_result' }],
      code: {
        title: 'PrefillRouter → decode-воркер (упрощённо)',
        lang: 'json',
        code: `{"token_ids": [${ids}],\n "stop_conditions": {"max_tokens": 512},\n "prefill_result": {\n   "disaggregated_params": {"kv_transfer_params": {"…": "…"}},\n   "prompt_tokens_details": {"cached_tokens": 0}}}`,
      },
    },
    {
      title: 'NIXL переносит KV напрямую между GPU',
      text: `Decode-воркер один раз знакомится с NIXL-агентом prefill-воркера и читает ${fmt.count(blocks, ['блок', 'блока', 'блоков'])} (${kv}) прямо из его видеопамяти: по RDMA (в Kubernetes так даже внутри одной ноды) или по NVLink, если процессы видят GPU друг друга, почти без участия CPU. GPU prefill-воркера тем временем считает следующие промпты.`,
      nodes: ['p1', 'd1'],
      edges: [
        { id: 'nixl-data', label: `NIXL READ: ${kv}` },
        { id: 'nixl-req', label: 'запрос блоков', dashed: true },
      ],
      initiator: { node: 'd1', text: 'забирает decode (pull)' },
      code: {
        title: 'decode-воркер (vLLM, NixlConnector)',
        lang: 'text',
        code: `запрос в состоянии WAITING_FOR_REMOTE_KVS\nNIXL READ ${fmt.count(blocks, ['блок', 'блока', 'блоков'])} с 10.0.3.17:5600\nKV на месте → RUNNING, сразу decode`,
      },
    },
    {
      title: 'Decode генерирует, Postprocessor собирает ответ',
      text: 'Decode-воркер генерирует до 512 токенов и отдаёт во frontend только их ID. Postprocessor делает из них текст (детокенизация), выделяет reasoning и tool calls и собирает SSE-чанки для клиента.',
      nodes: ['d1', 'post', 'client'],
      edges: [
        { id: 'd1-post', label: 'ID токенов' },
        { id: 'post-client', label: 'текст, SSE-чанки' },
      ],
      code: {
        title: 'frontend → клиент',
        lang: 'text',
        code: 'data: {"choices": [{"delta": {"content": "KV"}}]}\n\ndata: {"choices": [{"delta": {"content": " cache"}}]}\n\n…\n\ndata: [DONE]',
      },
    },
  ]
}

function sglangSteps(kv: string, tokens: number, blocks: number): FlowStep[] {
  const s = vllmSteps(kv, tokens, blocks)
  return [
    s[0],
    {
      title: 'Prefill стартует в фоне',
      text: 'В SGLang точка встречи известна заранее: роутер вычисляет bootstrap_room и отправляет prefill-запрос, не дожидаясь ответа.',
      nodes: ['router', 'p1'],
      edges: [{ id: 'router-p1', label: 'prefill, bootstrap_room' }],
      code: {
        title: 'PrefillRouter → prefill-воркер (SGLang, упрощённо)',
        lang: 'json',
        code: `{"token_ids": ["…${fmt.int(tokens)} ID…"],\n "bootstrap_info": {\n   "bootstrap_host": "10.0.3.17",\n   "bootstrap_port": 8998,\n   "bootstrap_room": 7321}}`,
      },
    },
    {
      title: 'Decode-запрос уходит сразу',
      text: 'Почти одновременно decode-воркер получает запрос с тем же bootstrap_info и заранее выделяет страницы под KV этого промпта.',
      nodes: ['router', 'd1'],
      edges: [{ id: 'router-d1', label: 'запрос + bootstrap_info' }],
      code: { title: 'PrefillRouter → decode-воркер (SGLang)', lang: 'json', code: '{"bootstrap_info": {\n   "bootstrap_host": "10.0.3.17",\n   "bootstrap_port": 8998,\n   "bootstrap_room": 7321}}' },
    },
    {
      title: 'Prefill сам записывает KV в память decode',
      text: `Модель передачи push: prefill-воркер считает KV и пишет его в заранее выделенные страницы decode-воркера (RDMA WRITE, ${kv}). В самом SGLang транспорт по умолчанию Mooncake, можно NIXL.`,
      nodes: ['p1', 'd1'],
      edges: [{ id: 'nixl-data', label: `RDMA WRITE: ${kv}` }],
      initiator: { node: 'p1', text: 'пишет prefill (push)' },
      code: {
        title: 'что происходит при передаче (SGLang)',
        lang: 'text',
        code: `prefill: ${fmt.tokens(tokens)} → KV\nRDMA WRITE в страницы decode-воркера (bootstrap_room 7321)\ndecode: KV на месте → сразу decode`,
      },
    },
    s[6],
  ]
}

export function DynamoFlowFigure() {
  const [settings] = useSettings()
  const [backend, setBackend] = useState<Backend>('vllm')
  const cfg = simConfig(settings.modelId, settings.gpuId)
  const reader = useReaderPromptTokens()
  const tokens = reader ?? FALLBACK_TOKENS
  const blocks = Math.max(1, Math.ceil(tokens / 16))
  // десятичные единицы, как в hero (инспектор NIXL) и в остальных фигурах главы
  const kv = fmt.bytes(kvBytesForSeq(cfg.model, tokens))
  const steps = useMemo(() => (backend === 'vllm' ? vllmSteps(kv, tokens, blocks) : sglangSteps(kv, tokens, blocks)), [backend, kv, tokens, blocks])
  const player = usePlayer(steps.length, { intervalMs: 4200 })
  const cur = steps[Math.min(player.step, steps.length - 1)]
  const [wrapRef, { width }] = useElementSize<HTMLDivElement>()
  const tall = width > 0 && width < 720
  const lay = tall ? TALL : WIDE

  const nodes: DNode[] = useMemo(
    () => [
      ...lay.groups.map((g): DNode => ({ id: g.id, variant: 'group', title: g.title, x: g.rect[0], y: g.rect[1], w: g.rect[2], h: g.rect[3] })),
      ...(Object.keys(lay.box) as BoxId[]).map((id): DNode => {
        const [x, y, w, h] = lay.box[id]
        return { id, x, y, w, h, title: BOX_TITLE[id], role: BOX_ROLE[id] }
      }),
    ],
    [lay],
  )
  const edges: DEdge[] = cur.edges.map((e) => {
    const g = lay.edges[e.id]
    return { id: e.id, points: g.points, label: e.label, labelAt: g.labelAt, labelAnchor: g.anchor, dashed: e.dashed }
  })
  // на узком экране у пунктирного запроса блоков подписи нет: её место занимает подпись инициатора
  if (tall) for (const e of edges) if (e.id === 'nixl-req') e.label = undefined
  const first = cur.edges[0]
  const packetColor = first && (first.id === 'd1-post' || first.id === 'post-client') ? 'decode' : 'prefill'

  return (
    <Figure
      id="disaggregation-figure-dynamo"
      title="Путь дезагрегированного запроса в Dynamo"
      player={<Player {...player.bind} aria-label="Шаги запроса" />}
      controls={
        <>
          <Segmented<Backend>
            label="Движок воркеров"
            value={backend}
            onChange={(b) => {
              setBackend(b)
              player.setStep(0)
            }}
            options={[
              { value: 'vllm', label: 'vLLM' },
              { value: 'sglang', label: 'SGLang' },
            ]}
          />
        </>
      }
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> пройдите шаги для vLLM, потом переключитесь на SGLang: там prefill и decode
          стартуют почти одновременно, а KV не забирают, а присылают. Имена полей взяты из кода Dynamo и vLLM, значения условные; KV посчитан для{' '}
          {cfg.model.name} и{' '}
          {reader !== null ? (
            <>
              вашего запроса из симулятора ({fmt.tokens(tokens)} вместе с шаблоном чата{settings.tools ? ' и описанием инструмента' : ''})
            </>
          ) : (
            <>промпта в {fmt.tokens(tokens)}</>
          )}
          .
        </>
      }
    >
      <div className="grid gap-5">
        <div ref={wrapRef} className="min-w-0">
          <div className={tall ? 'mx-auto max-w-[440px]' : 'mx-auto max-w-[920px]'}>
            <Diagram
              width={lay.vb[0]}
              height={lay.vb[1]}
              nodes={nodes}
              edges={edges}
              activeNodes={new Set(cur.nodes)}
              activeEdges={new Set(cur.edges.map((e) => e.id))}
              packet={first ? { edgeId: first.id, key: `${backend}-${player.step}-${tall}`, color: packetColor } : null}
              minWidth={tall ? 300 : 680}
              aria-label={`Схема Dynamo, шаг ${player.step + 1} из ${steps.length}: ${cur.title}`}
            >
              {cur.initiator && (
                <text
                  x={lay.initiatorAt[0]}
                  y={lay.initiatorAt[1] + (tall ? 0 : cur.edges.some((e) => e.id === 'nixl-req') ? 15 : 0)}
                  textAnchor={lay.initiatorAnchor}
                  fontSize={11.5}
                  fill="var(--ink-2)"
                  stroke="var(--surface)"
                  strokeWidth={4}
                  strokeLinejoin="round"
                  paintOrder="stroke"
                >
                  {cur.initiator.text}
                </text>
              )}
            </Diagram>
          </div>
        </div>
        <div className={cur.code ? 'grid min-w-0 gap-4 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]' : 'grid min-w-0 gap-4'} aria-live="polite">
          <div className="grid content-start gap-2">
            <div className="text-[15px] leading-snug font-semibold text-ink">{cur.title}</div>
            <p className="max-w-[68ch] text-[15px] leading-relaxed text-ink">{cur.text}</p>
          </div>
          {cur.code && <CodeBlock lang={cur.code.lang} title={cur.code.title} code={cur.code.code} wrap copy={false} />}
        </div>
      </div>
    </Figure>
  )
}

/** Пока токенизатор грузится: условный промпт. */
const FALLBACK_TOKENS = 1024
