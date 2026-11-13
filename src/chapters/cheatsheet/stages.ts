import type { ChapterId } from '@/content/chapters'
import type { StageId } from '@/engine/types'

/**
 * Стадия пути запроса для шпаргалки (SPEC §6.27). Тексты сверены с главами сайта и SPEC §8;
 * кто что делает в vLLM, SGLang и Dynamo: notes-engines.md §1.0, notes-dynamo.md, SPEC §8.4 и §8.11.
 */
export interface CheatStage {
  id: string
  n: number
  name: string
  /** Где выполняется: устройство и компонент. */
  where: string
  /** Совсем коротко, для узла схемы. */
  short: string
  what: string
  /** Какой процесс делает стадию в vLLM, SGLang и Dynamo. */
  stack?: string
  tune: string
  metric: string
  chapters: ChapterId[]
  sim?: StageId
  role?: 'prefill' | 'decode'
}

export const STAGES: CheatStage[] = [
  {
    id: 'client',
    n: 1,
    name: 'Клиент',
    where: 'приложение',
    short: 'POST /v1/chat/completions',
    what: 'Шлёт JSON с messages и параметрами сэмплинга по OpenAI-совместимому API, обычно со stream: true.',
    tune: 'Таймауты и ретраи, переиспользование соединений, стриминг, max_tokens.',
    metric: 'TTFT и E2E глазами клиента',
    chapters: ['request', 'metrics'],
    sim: 'client',
  },
  {
    id: 'frontend',
    n: 2,
    name: 'Frontend',
    where: 'CPU: frontend Dynamo или API-сервер движка',
    short: 'HTTP, проверка, лимиты',
    what: 'Принимает HTTP, проверяет запрос, собирает параметры сэмплинга и условия остановки.',
    stack: 'vLLM: API-сервер. SGLang: HTTP-сервер в главном процессе. Dynamo: frontend на Rust.',
    tune: 'Лимиты на длину и concurrency, приём нагрузки (admission), rate limiting.',
    metric: 'Запросов в секунду, коды ответов, очередь',
    chapters: ['request', 'engines', 'dynamo', 'production'],
    sim: 'frontend',
  },
  {
    id: 'preprocess',
    n: 3,
    name: 'Preprocessor: шаблон и токенизация',
    where: 'CPU: frontend Dynamo или API-сервер движка',
    short: 'chat template → token id',
    what: 'Chat template склеивает сообщения в одну строку со спецтокенами, токенизатор режет её на ID словаря.',
    stack: 'vLLM: API-сервер. SGLang: TokenizerManager в главном процессе. Dynamo: preprocessor во frontend.',
    tune: 'Правильный шаблон модели, стабильное начало промпта (для prefix cache), версия токенизатора.',
    metric: 'Токенов в промпте (ISL)',
    chapters: ['template', 'tokenization', 'engines'],
    sim: 'preprocess',
  },
  {
    id: 'route',
    n: 4,
    name: 'Router',
    where: 'CPU: роутер Dynamo (у одиночного движка роутера нет)',
    short: 'куда отправить запрос',
    what: 'Выбирает воркер: где уже лежит KV префикса и кто меньше загружен (в Dynamo режим kv, по умолчанию round-robin).',
    stack: 'vLLM: внешний роутер (vLLM Router, llm-d). SGLang: Model Gateway с cache-aware политикой. Dynamo: KV router во frontend.',
    tune: 'Режим роутера, вес перекрытия и температура, размер блока как у движка.',
    metric: 'Prefix cache hit rate, баланс нагрузки',
    chapters: ['routing', 'prefix-caching', 'kv-tiers', 'dynamo'],
    sim: 'route',
  },
  {
    id: 'prefill',
    n: 5,
    name: 'Prefill',
    where: 'GPU: движок (vLLM, SGLang, TensorRT-LLM)',
    short: 'весь промпт за проход',
    what: 'Один проход по всему промпту: эмбеддинги, слои трансформера с attention, K и V каждого токена в KV cache, логиты последней позиции дают первый токен. Промпт от нескольких сотен токенов упирается в вычисления, а короткий, как и шаг decode, в чтение весов. Если начало промпта уже считали, его KV не пересчитывают, а берут из кэша: из HBM воркера (L1), RAM хоста (L2) или общего L3 (LMCache, Mooncake; в Kubernetes кэш-сервер ставят DaemonSet на каждую ноду).',
    stack: 'vLLM: EngineCore и GPU-воркеры. SGLang: Scheduler. Dynamo: воркер с vLLM, SGLang или TensorRT-LLM внутри.',
    tune: 'Prefix caching и оффлоад KV, бюджет chunked prefill (max_num_batched_tokens), FP8, tensor parallel.',
    metric: 'TTFT (вместе с очередью)',
    chapters: ['prefill-decode', 'embeddings', 'transformer', 'attention', 'roofline', 'prefix-caching', 'kv-tiers', 'batching', 'parallelism'],
    sim: 'prefill',
    role: 'prefill',
  },
  {
    id: 'transfer',
    n: 6,
    name: 'Передача KV',
    where: 'сеть между GPU',
    short: 'только при дезагрегации',
    what: 'В дезагрегированном режиме готовый KV едет с prefill-воркера на decode-воркер через NIXL (NVLink, InfiniBand, RoCE).',
    stack: 'vLLM: NixlConnector. SGLang: Mooncake (по умолчанию) или NIXL. Dynamo: NIXL.',
    tune: 'Канал и RDMA, соотношение prefill- и decode-воркеров (xPyD), формат KV.',
    metric: 'Время передачи, вклад в TTFT',
    chapters: ['disaggregation', 'dynamo'],
    sim: 'transfer',
  },
  {
    id: 'decode',
    n: 7,
    name: 'Decode',
    where: 'GPU: движок (vLLM, SGLang, TensorRT-LLM)',
    short: 'по токену за шаг',
    what: 'Шаг за шагом по одному токену: читает все веса и весь KV, сэмплер выбирает токен (маска structured output до выбора). Упор в память.',
    stack: 'vLLM: EngineCore и GPU-воркеры. SGLang: Scheduler. Dynamo: воркер с движком; маску structured output считает движок decode-воркера.',
    tune: 'Размер batch (max_num_seqs), память под KV, квантизация, speculative decoding, tensor parallel.',
    metric: 'ITL (TPOT), throughput, заполненность KV',
    chapters: ['prefill-decode', 'roofline', 'kv-cache', 'structured-output', 'speculative', 'quantization', 'sampling', 'paged-attention', 'batching', 'parallelism'],
    sim: 'decode',
    role: 'decode',
  },
  {
    id: 'postprocess',
    n: 8,
    name: 'Postprocessor: детокенизация и парсеры',
    where: 'CPU: frontend Dynamo или API-сервер движка',
    short: 'ID → текст, каналы',
    what: 'Инкрементально превращает ID в текст, придерживает неполные символы и stop-строки, разбирает reasoning и tool calls.',
    stack: 'vLLM: API-сервер. SGLang: DetokenizerManager, отдельный процесс. Dynamo: postprocessor во frontend на Rust.',
    tune: 'Парсеры reasoning и tool calls под модель, stop-строки, skip_special_tokens.',
    metric: 'Правильные reasoning_content, content, tool_calls',
    chapters: ['detokenization', 'parsers'],
    sim: 'postprocess',
  },
  {
    id: 'stream',
    n: 9,
    name: 'SSE-стрим',
    where: 'сеть',
    short: 'data: {…} … [DONE]',
    what: 'Отдаёт клиенту чанки chat.completion.chunk, в конце finish_reason, usage и data: [DONE].',
    stack: 'vLLM и SGLang: HTTP-сервер движка. Dynamo: frontend.',
    tune: 'include_usage, сколько токенов в чанке, keep-alive.',
    metric: 'ITL по чанкам у клиента',
    chapters: ['parsers', 'metrics'],
    sim: 'stream',
  },
]
