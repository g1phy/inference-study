import { useMemo, useState } from 'react'
import { ArrowDownIcon, ArrowUpIcon, CheckCircleIcon, LightningIcon } from '@phosphor-icons/react'
import { mulberry32 } from '@/engine/rng'
import { Button, Figure, MEDIA, cn, useMediaQuery } from '@/ui'
import { ChipGroup } from './ChipGroup'

type MetricId = 'ttft' | 'itl' | 'queue' | 'running' | 'kv' | 'hit' | 'preempt' | 'gpu' | 'errors' | 'rps' | 'isl' | 'replicas'
type Move = 'flat' | 'up' | 'down' | 'spikes'
type ScenarioId = 'normal' | 'traffic' | 'long' | 'kvfull' | 'prefix' | 'nodefail'

interface MetricInfo {
  id: MetricId
  title: string
  name: string
  what: string
  /** Рост этой метрики плохой знак (для подписи). */
  upIsBad: boolean
}

const METRICS: MetricInfo[] = [
  { id: 'ttft', title: 'TTFT p50 и p99', name: 'vllm:time_to_first_token_seconds', what: 'очередь + prefill; алерт на p99', upIsBad: true },
  { id: 'itl', title: 'ITL p50 и p99', name: 'vllm:inter_token_latency_seconds', what: 'скорость печати; рывки видны в p99', upIsBad: true },
  { id: 'queue', title: 'Очередь', name: 'vllm:num_requests_waiting', what: 'главный сигнал для автоскейлинга', upIsBad: true },
  { id: 'running', title: 'Запросов в работе', name: 'vllm:num_requests_running', what: 'текущий batch реплики', upIsBad: false },
  { id: 'kv', title: 'Заполнение KV cache', name: 'vllm:kv_cache_usage_perc', what: 'у 1,0 начинаются ожидание и preemption', upIsBad: true },
  { id: 'hit', title: 'Prefix cache hit rate', name: 'prefix_cache_hits / prefix_cache_queries', what: 'доля токенов промпта из кэша', upIsBad: false },
  { id: 'preempt', title: 'Preemptions', name: 'vllm:num_preemptions', what: 'запросы, вытесненные из KV', upIsBad: true },
  {
    id: 'gpu',
    title: 'Занятость полосы памяти и тензорных ядер',
    name: 'DCGM_FI_PROF_DRAM_UTIL_RATIO, DCGM_FI_PROF_TENSOR_UTIL_RATIO',
    what: 'decode упирается в полосу памяти, prefill в тензорные ядра; не путать с utilization.gpu',
    upIsBad: false,
  },
  { id: 'errors', title: 'Ошибки 5xx и таймауты', name: 'метрики gateway', what: 'плюс 4xx от валидации запросов', upIsBad: true },
  { id: 'rps', title: 'Запросов в секунду', name: 'sum(rate(vllm:request_success_total[5m]))', what: 'объём трафика', upIsBad: false },
  { id: 'isl', title: 'Длина промптов', name: 'vllm:request_prompt_tokens', what: 'ISL; рядом стоит смотреть и OSL', upIsBad: false },
  { id: 'replicas', title: 'Реплики', name: 'Kubernetes: ready / starting', what: 'сколько готово и сколько стартует', upIsBad: false },
]

const SCENARIOS: Record<ScenarioId, { label: string; moves: Partial<Record<MetricId, Move>>; what: string; fix: string }> = {
  normal: {
    label: 'Норма',
    moves: {},
    what: 'Все линии ровные. Полезно знать, как выглядит ваша норма: алерты ставят на отклонения p99 и очереди, а не на среднее.',
    fix: 'Ничего. Проверьте, что алерты срабатывают на тестовом всплеске (load test).',
  },
  traffic: {
    label: 'Всплеск трафика',
    moves: { rps: 'up', running: 'up', queue: 'up', ttft: 'up', kv: 'up', gpu: 'up', replicas: 'up', itl: 'up' },
    what: 'Запросов стало больше, чем держат реплики: batch упёрся в потолок, остальное ждёт в очереди, растёт TTFT. Автоскейлер уже заказал реплики, но они ещё стартуют.',
    fix: 'Быстрее холодный старт, запас минимальных реплик, приоритетная очередь для платных запросов; масштабировать по очереди и concurrency.',
  },
  long: {
    label: 'Длинные промпты',
    moves: { isl: 'up', ttft: 'up', itl: 'up', gpu: 'up', kv: 'up', queue: 'up' },
    what: 'Число запросов то же, но промпты стали длиннее (RAG, документы, агенты). Нагрузка в токенах выросла: TTFT растёт из-за prefill, ITL из-за длинных prefill-кусков в общих шагах.',
    fix: 'Балансировать по токенам, а не по числу запросов; проверить chunked prefill и бюджет токенов; при большом масштабе дезагрегация.',
  },
  kvfull: {
    label: 'KV cache полон',
    moves: { kv: 'up', preempt: 'spikes', itl: 'spikes', queue: 'up', ttft: 'up', running: 'down' },
    what: 'KV cache заполнен: новым запросам негде хранить K и V, они ждут, а часть запущенных вытесняется и потом пересчитывается заново. Отсюда рывки ITL и рост p99.',
    fix: 'FP8 KV, меньше max_num_seqs или max_model_len, больше реплик; проверить, не раздулись ли длины ответов.',
  },
  prefix: {
    label: 'Деплой сломал prefix cache',
    moves: { hit: 'down', ttft: 'up', gpu: 'up' },
    what: 'Ошибок нет, трафик прежний, но hit rate упал, а TTFT и загрузка GPU выросли. Обычно в начало системного промпта попала дата или поменялся шаблон, и общий префикс у запросов пропал.',
    fix: 'Найти изменение в начале промпта (статичное в начало, переменное в конец), откатить канарейку; следить за hit rate после каждого деплоя.',
  },
  nodefail: {
    label: 'Отказал узел',
    moves: { replicas: 'down', errors: 'spikes', queue: 'up', ttft: 'up', running: 'up' },
    what: 'Узел с GPU упал: его реплики пропали, их запросы оборвались ошибками 5xx, остальные реплики забрали трафик, и очередь растёт, пока поднимается замена. Сбои, обрывающие работу, случаются примерно раз на 50\u00a0000 GPU-часов (обучение Llama 3: 419 неожиданных остановок за 54 дня на 16 тысячах GPU, больше половины из-за самих GPU и их памяти): на флоте из 1 000 GPU это раз в двое суток.',
    fix: 'Ретраи в gateway и на клиенте, запас реплик, cordon сломанного узла (запрет ставить на него новые поды) и перезапуск подов на других узлах, несколько зон или кластеров.',
  },
}

function series(id: MetricId, move: Move, seed: number): number[] {
  const rng = mulberry32(seed)
  const n = 40
  const out: number[] = []
  let noise = 0
  for (let i = 0; i < n; i++) {
    noise = 0.6 * noise + 0.4 * (rng() - 0.5)
    const t = i / (n - 1)
    const after = Math.max(0, Math.min(1, (t - 0.5) / 0.12))
    let v = 0.42 + noise * 0.12
    if (move === 'up') v += 0.42 * after
    if (move === 'down') v -= 0.3 * after
    if (move === 'spikes' && t > 0.5 && rng() > 0.55) v += 0.25 + rng() * 0.3
    if (id === 'hit' && move === 'flat') v += 0.25
    out.push(Math.max(0.03, Math.min(0.97, v)))
  }
  return out
}

function Spark({ values, highlight }: { values: number[]; highlight: boolean }) {
  const w = 120
  const h = 34
  const d = values.map((v, i) => `${i === 0 ? 'M' : 'L'} ${(i / (values.length - 1)) * w} ${h - v * h}`).join(' ')
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-[34px] w-full" aria-hidden preserveAspectRatio="none">
      <line x1={w / 2} x2={w / 2} y1={0} y2={h} stroke="var(--line-strong)" strokeDasharray="3 3" />
      <path d={d} fill="none" stroke={highlight ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={1.6} vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

function MoveBadge({ move, upIsBad }: { move: Move; upIsBad: boolean }) {
  if (move === 'flat')
    return (
      <span className="inline-flex items-center gap-1 text-[12.5px] text-ink-2">
        <CheckCircleIcon size={14} weight="fill" className="text-good" aria-hidden />
        ровно
      </span>
    )
  const bad = move === 'spikes' || (move === 'up') === upIsBad
  const Icon = move === 'up' ? ArrowUpIcon : move === 'down' ? ArrowDownIcon : LightningIcon
  return (
    <span className="inline-flex items-center gap-1 text-[12.5px] text-ink">
      <Icon size={14} weight="bold" className={bad ? 'text-bad' : 'text-ink-2'} aria-hidden />
      {move === 'up' ? 'растёт' : move === 'down' ? 'падает' : 'скачки'}
    </span>
  )
}

export function MonitoringBoard() {
  const [scenario, setScenario] = useState<ScenarioId>('traffic')
  const [showAll, setShowAll] = useState(false)
  const wide = useMediaQuery(MEDIA.tablet)
  const sc = SCENARIOS[scenario]
  const data = useMemo(
    () => METRICS.map((m, i) => ({ m, move: (sc.moves[m.id] ?? 'flat') as Move, values: series(m.id, sc.moves[m.id] ?? 'flat', 11 + i * 7) })),
    [sc],
  )
  // на телефоне карточки в одну колонку: изменившиеся первыми, неизменные свёрнуты в одну строку
  const changed = data.filter((d) => d.move !== 'flat')
  const flat = data.filter((d) => d.move === 'flat')
  const collapse = !wide && !showAll && changed.length > 0
  const cards = wide || showAll || changed.length === 0 ? data : changed
  return (
    <Figure
      id="production-figure-monitoring"
      title="Что мониторить и как метрики выдают причину"
      controls={
        <ChipGroup
          label="Что случилось"
          value={scenario}
          onChange={(v) => {
            setScenario(v)
            setShowAll(false)
          }}
          options={(Object.keys(SCENARIOS) as ScenarioId[]).map((id) => ({ value: id, label: SCENARIOS[id].label }))}
          className="col-span-full"
        />
      }
      controlsPosition="top"
      caption="Попробуйте: сравните «Всплеск трафика» и «Длинные промпты». TTFT растёт в обоих случаях, а различить их помогают объём запросов и длина промптов."
      source="Графики условные; пунктир посередине отмечает момент изменения. Имена метрик vLLM и DCGM exporter зависят от версии."
    >
      <div className="grid min-w-0 gap-4">
        <div className="grid gap-2 rounded-panel bg-surface-2/60 p-4 text-[14.5px] leading-relaxed" aria-live="polite">
          <p className="text-ink">
            <b className="font-semibold">Что видно. </b>
            {sc.what}
          </p>
          <p className="text-ink-2">
            <b className="font-semibold text-ink">Что делать. </b>
            {sc.fix}
          </p>
        </div>
        <div className="grid min-w-0 grid-cols-1 gap-2.5 @md:grid-cols-2 @3xl:grid-cols-3">
          {cards.map(({ m, move, values }) => (
            <div
              key={m.id}
              className={cn('grid min-w-0 gap-1.5 rounded-control p-3 transition-shadow', move !== 'flat' ? 'bg-surface shadow-[inset_0_0_0_1.5px_var(--ink-3)]' : 'bg-surface-2/60 shadow-[inset_0_0_0_1px_var(--line)]')}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-[14px] font-medium text-ink">{m.title}</span>
                <MoveBadge move={move} upIsBad={m.upIsBad} />
              </div>
              <Spark values={values} highlight={move !== 'flat'} />
              <code className="font-mono text-[11.5px] break-words text-ink-2 [overflow-wrap:anywhere]">
                {m.name}
              </code>
              <span className="text-[12.5px] leading-snug text-ink-3">{m.what}</span>
            </div>
          ))}
        </div>
        {collapse && (
          <div className="grid justify-items-start gap-2 rounded-control p-3 shadow-[inset_0_0_0_1px_var(--line)]">
            <p className="flex gap-1.5 text-[13.5px] leading-snug text-ink-2">
              <CheckCircleIcon size={16} weight="fill" className="mt-0.5 shrink-0 text-good" aria-hidden />
              <span>
                <span className="font-medium text-ink">Без изменений: </span>
                {flat.map((d) => d.m.title).join(', ')}.
              </span>
            </p>
            <Button size="sm" variant="secondary" onClick={() => setShowAll(true)}>
              Показать все {data.length} карточек
            </Button>
          </div>
        )}
      </div>
    </Figure>
  )
}
