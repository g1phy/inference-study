/**
 * Карточки компонентов Dynamo для интерактивной схемы (глава 19). Факты и флаги из
 * research/notes-dynamo.md (§1.2-1.18, состояние на ноябрь 2026, Dynamo 1.6.x).
 */
import type { ReactNode } from 'react'
import type { ChapterId } from '@/content/chapters'
import type { CodeLang } from '@/ui'

export interface ArchItem {
  title: string
  /** Где живёт: процесс, под, язык. */
  where: string
  what: ReactNode
  facts: ReactNode[]
  config: { lang: CodeLang; title: string; code: string }
  links?: ChapterId[]
}

export const ARCH: Record<string, ArchItem> = {
  client: {
    title: 'Клиент',
    where: 'вне кластера',
    what: (
      <>
        Любой клиент OpenAI API: SDK, curl, ваш бэкенд. Он отправляет <code>POST /v1/chat/completions</code> с JSON и при <code>stream: true</code>{' '}
        получает ответ кусками, server-sent events.
      </>
    ),
    facts: [
      <>Кроме chat completions frontend отвечает на <code>/v1/completions</code>, <code>/v1/embeddings</code>, <code>/v1/responses</code>, <code>/v1/models</code>.</>,
      <>Можно указать воркер напрямую через расширение <code>nvext</code> или заголовок <code>x-dynamo-worker-instance-id</code>: удобно для отладки.</>,
    ],
    config: {
      lang: 'bash',
      title: 'запрос',
      code: `curl http://frontend:8000/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{"model": "Qwen/Qwen3-8B", "stream": true,
       "messages": [{"role": "user", "content": "Что такое KV cache?"}]}'`,
    },
    links: ['request'],
  },
  http: {
    title: 'HTTP API (frontend)',
    where: 'под frontend, Rust, CPU',
    what: (
      <>
        OpenAI-совместимый сервер на Rust. Он знает, какие модели сейчас обслуживаются: воркер при старте регистрирует модель и публикует её «паспорт»,
        Model Deployment Card. В нём токенизатор, chat template, размер KV-блока, длина контекста и имена парсеров. Весов frontend не нужно.
      </>
    ),
    facts: [
      <>Скачивает только конфиг и файлы токенизатора (<code>config.json</code>, <code>tokenizer.json</code>, <code>tokenizer_config.json</code> и подобные), без весов.</>,
      <>При перегрузке отвечает HTTP 529 (не 503). Тело запроса до 45 МБ.</>,
      <>Метрики на том же порту: <code>/metrics</code>, проверки живости <code>/health</code> и <code>/live</code>.</>,
      'Реплики воркеров одной модели обязаны иметь одинаковую контрольную сумму паспорта, иначе реплика не получит трафик.',
    ],
    config: { lang: 'bash', title: 'запуск', code: 'python -m dynamo.frontend --http-port 8000 --router-mode kv' },
    links: ['request'],
  },
  pre: {
    title: 'Preprocessor',
    where: 'внутри frontend, Rust',
    what: (
      <>
        Превращает JSON в token IDs и правила генерации. Шаблон чата из <code>tokenizer_config.json</code> рендерит minijinja, реализация Jinja2 на
        Rust, а токенизирует библиотека HuggingFace tokenizers, тоже на Rust. Затем проверяет, что промпт плюс <code>max_tokens</code> помещаются в контекст,
        и собирает <code>PreprocessedRequest</code>.
      </>
    ),
    facts: [
      <>Конвейер: перевод формата → дефолты модели → шаблон → токенизация.</>,
      <>Если <code>max_tokens</code> не задан, берётся весь остаток контекста. EOS модели добавляются как скрытые стоп-токены.</>,
      <>Свой шаблон задают на воркере: <code>--custom-jinja-template</code>. Режим thinking по умолчанию тоже задают на воркере: <code>--dyn-default-thinking-mode</code>; явный <code>enable_thinking</code> в запросе важнее.</>,
      <>Время стадий видно в метриках <code>dynamo_frontend_template_seconds</code> и <code>dynamo_frontend_tokenize_seconds</code>.</>,
    ],
    config: {
      lang: 'json',
      title: 'PreprocessedRequest (упрощённо)',
      code: `{
  "model": "Qwen/Qwen3-8B",
  "token_ids": [151644, 8948, 198, 33995, 4552, 28519, "…"],
  "stop_conditions": {
    "max_tokens": 512,
    "stop_token_ids_hidden": [151645, 151643]
  },
  "sampling_options": { "temperature": 0.7 },
  "output_options": { "skip_special_tokens": true }
}`,
    },
    links: ['template', 'tokenization'],
  },
  router: {
    title: 'KV router',
    where: 'внутри frontend, Rust',
    what: (
      <>
        Выбирает воркер. По token IDs считает хэши блоков по 16 токенов, по индексу из KV events узнаёт, сколько первых блоков уже лежит у каждого воркера,
        и минимизирует стоимость: prefill-работу за вычетом кэша плюс decode-нагрузку. В дезагрегированном режиме роутер сначала выбирает prefill-воркер,
        потом decode-воркер по нагрузке.
      </>
    ),
    facts: [
      <>Режим по умолчанию <code>round-robin</code>: KV-aware включают флагом <code>--router-mode kv</code>.</>,
      'Без KV events от воркеров режим kv считает только нагрузку.',
      'Размер блока роутера должен совпадать с размером блока движка.',
      <>С версии 1.6 параметры стоимости переезжают в YAML: <code>--router-policy-config</code>.</>,
    ],
    config: {
      lang: 'bash',
      title: 'frontend',
      code: `python -m dynamo.frontend --router-mode kv
# в disagg prefill-роутер включается сам, когда в namespace
# зарегистрированы и prefill-, и decode-воркеры одной модели`,
    },
    links: ['routing'],
  },
  events: {
    title: 'Event plane и KV events',
    where: 'между процессами',
    what: (
      <>
        Асинхронные сообщения между компонентами. Воркеры сообщают роутеру <code>BlockStored</code>, <code>BlockRemoved</code>,{' '}
        <code>AllBlocksCleared</code>, движки отдают пошаговые метрики для Planner. Сами запросы и стримы ответов идут отдельно, по request plane.
      </>
    ),
    facts: [
      'Плоскостей три, и они независимы: discovery (кто жив и где), request plane (запросы), event plane (события).',
      'По умолчанию request plane работает на прямых TCP-соединениях, event plane на ZMQ, а discovery в Kubernetes идёт через ресурсы DynamoWorkerMetadata и EndpointSlice, локально через etcd.',
      'NATS и etcd больше не обязательны: чарт платформы 1.6.2 по умолчанию их не ставит. В старых версиях через NATS шли и запросы.',
    ],
    config: {
      lang: 'bash',
      title: 'переменные окружения',
      code: `DYN_REQUEST_PLANE=tcp              # запросы и стримы ответов
DYN_EVENT_PLANE=zmq                # KV events и метрики
DYN_DISCOVERY_BACKEND=kubernetes   # локально: etcd`,
    },
    links: ['routing'],
  },
  prefill: {
    title: 'Prefill-воркер',
    where: 'под с GPU, Python-обёртка над движком',
    what: (
      <>
        Воркер служит тонкой обёрткой над движком: <code>python -m dynamo.vllm</code>, <code>dynamo.sglang</code> или <code>dynamo.trtllm</code>. Он получает уже
        токенизированный запрос и гоняет движок в режиме «токены на входе, токены на выходе», мимо его собственного HTTP-сервера. Prefill-воркеру роутер
        отправляет копию запроса с <code>max_tokens = 1</code>: посчитать KV всего промпта и вернуть, где этот KV лежит.
      </>
    ),
    facts: [
      <>У vLLM prefill возвращает <code>disaggregated_params</code> с <code>kv_transfer_params</code> (номера блоков и адрес). У SGLang адрес встречи <code>bootstrap_info</code> роутер знает заранее, поэтому prefill идёт фоном, а decode стартует сразу.</>,
      'Prefix caching в vLLM-воркере Dynamo включён по умолчанию, поэтому общий префикс не считается дважды.',
      <>KV events у vLLM включаются явно: <code>--kv-events-config</code>.</>,
    ],
    config: {
      lang: 'bash',
      title: 'vLLM-воркер',
      code: `python -m dynamo.vllm --model Qwen/Qwen3-8B \\
  --disaggregation-mode prefill \\
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'`,
    },
    links: ['disaggregation', 'engines'],
  },
  nixl: {
    title: 'NIXL',
    where: 'библиотека внутри воркеров',
    what: (
      <>
        NVIDIA Inference Xfer Library переносит данные точка-точка между видеопамятью, RAM и хранилищами, почти без участия CPU. При знакомстве воркеры один
        раз обмениваются описанием своей памяти, а на каждый запрос передаются только номера блоков. Сами байты KV идут напрямую из GPU в GPU по RDMA или
        NVLink.
      </>
    ),
    facts: [
      'vLLM: decode-воркер сам читает блоки из памяти prefill-воркера (NIXL READ), prefill держит их до чтения.',
      'SGLang: decode заранее выделяет страницы, prefill записывает в них KV (WRITE), и decode стартует, не дожидаясь конца prefill.',
      'Без RDMA перенос откатывается на TCP и деградирует в 200–500 раз: в примере документации TTFT около 98 с вместо 200–500 мс.',
    ],
    config: {
      lang: 'json',
      title: 'kv_transfer_params в ответе prefill (иллюстрация)',
      code: `{
  "do_remote_prefill": true,
  "remote_block_ids": [[17, 18, 19, "…", 80]],
  "remote_engine_id": "e3f1…",
  "remote_host": "10.0.3.17",
  "remote_port": 5600,
  "tp_size": 1
}`,
    },
    links: ['disaggregation'],
  },
  decode: {
    title: 'Decode-воркер',
    where: 'под с GPU, Python-обёртка над движком',
    what: (
      <>
        Получает исходный запрос вместе с метаданными prefill, забирает KV через NIXL и генерирует ответ. Наружу стримит только token IDs: текстом их сделает
        frontend. Маску structured output и сэмплинг применяет движок здесь же, на decode-стороне.
      </>
    ),
    facts: [
      'Decode-воркер роутер выбирает только по нагрузке: кредит за кэш там равен нулю.',
      <>Если prefill-воркеров нет, запрос идёт прямо на decode-воркер; флаг <code>--decode-fallback</code> разрешает ему делать и prefill.</>,
      'В пути vLLM первый видимый токен сэмплирует decode: токен prefill-воркера отбрасывается.',
    ],
    config: {
      lang: 'bash',
      title: 'vLLM-воркер',
      code: `python -m dynamo.vllm --model Qwen/Qwen3-8B \\
  --disaggregation-mode decode \\
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'`,
    },
    links: ['disaggregation', 'structured-output'],
  },
  workers: {
    title: 'Агрегированный воркер',
    where: 'под с GPU, Python-обёртка над движком',
    what: (
      <>
        В агрегированном режиме один воркер делает и prefill, и decode на своих GPU, как обычный движок. Это режим по умолчанию: роль{' '}
        <code>--disaggregation-mode</code> не задаётся. Переноса KV нет, зато prefill длинного промпта делит шаги с чужими decode.
      </>
    ),
    facts: [
      'При низкой конкурентности агрегированный режим часто выгоднее: нет переноса и раздробленных пулов.',
      'KV-aware роутинг работает и здесь: кэш у каждого воркера свой.',
    ],
    config: {
      lang: 'bash',
      title: 'vLLM-воркер',
      code: `python -m dynamo.vllm --model Qwen/Qwen3-8B \\
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'`,
    },
    links: ['disaggregation', 'engines'],
  },
  post: {
    title: 'Postprocessor',
    where: 'внутри frontend, Rust',
    what: (
      <>
        Обратный путь. Детокенизатор (DecodeStream из HF tokenizers) превращает поток token IDs в текст и придерживает неполные символы. Хвост, похожий на
        начало стоп-строки, сидит в «jail», пока не станет ясно, стоп это или нет. Затем парсеры раскладывают текст на <code>reasoning_content</code>,{' '}
        <code>content</code> и <code>tool_calls</code>, собираются SSE-чанки, считаются usage, TTFT и ITL.
      </>
    ),
    facts: [
      <>Парсеры написаны на Rust и одинаковы для vLLM, SGLang и TRT-LLM (<code>--dyn-chat-processor dynamo</code>, по умолчанию).</>,
      <>Флаги парсеров ставят на воркер, он публикует их в паспорте модели: <code>--dyn-reasoning-parser</code>, <code>--dyn-tool-call-parser</code>.</>,
      <>Метрики <code>dynamo_frontend_time_to_first_token_seconds</code> и <code>dynamo_frontend_inter_token_latency_seconds</code>.</>,
    ],
    config: {
      lang: 'bash',
      title: 'воркер с парсерами Qwen3',
      code: `python -m dynamo.vllm --model Qwen/Qwen3-8B \\
  --dyn-reasoning-parser qwen3 --dyn-tool-call-parser hermes`,
    },
    links: ['detokenization', 'parsers'],
  },
  planner: {
    title: 'Planner',
    where: 'отдельный под',
    what: (
      <>
        Автоскейлер prefill- и decode-пулов, каждый пул отдельно. По умолчанию (<code>optimization_target: throughput</code>) быстрый контур раз в 5 с
        смотрит на очереди и заполненность KV и добавляет или убирает по одной реплике. В режиме <code>sla</code> медленный контур раз в 180 с
        прогнозирует трафик (число запросов, длины входа и выхода) и задаёт нижнюю границу реплик под цели по TTFT и ITL; быстрый контур там включают
        отдельно, и документация советует включать оба. В Kubernetes Planner меняет число реплик прямо в DynamoGraphDeployment.
      </>
    ),
    facts: [
      'Цели режима sla по умолчанию: TTFT 500 мс и ITL 50 мс (50 мс между токенами дают около 20 ток/с на пользователя). В режиме по умолчанию целей по задержкам нет.',
      'Пока новая реплика поднимается (под, загрузка весов), новые решения не выдаются.',
      'Бюджет по умолчанию: не больше 8 GPU на развёртывание.',
    ],
    config: {
      lang: 'yaml',
      title: 'параметры (упрощённо)',
      code: `optimization_target: sla
ttft_ms: 500
itl_ms: 50
throughput_adjustment_interval_seconds: 180
load_adjustment_interval_seconds: 5
max_gpu_budget: 8`,
    },
    links: ['production', 'metrics'],
  },
}
