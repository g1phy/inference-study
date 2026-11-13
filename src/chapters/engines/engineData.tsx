/**
 * Тексты главы 18: карточки компонентов vLLM V1 для схемы «внутри движка» и карточки сравнения движков.
 * Факты: research/notes-engines.md (§1.0-1.31, §7), notes-hardware-software-engines.md (§1.20-1.34),
 * состояние на ноябрь 2026 (vLLM v0.32, SGLang v0.5.23, TensorRT-LLM 1.2.1 / 1.4.0rc).
 */
import type { ReactNode } from 'react'
import type { ChapterId } from '@/content/chapters'
import type { CodeLang } from '@/ui'

export interface PartItem {
  title: string
  where: string
  what: ReactNode
  facts: ReactNode[]
  config?: { lang: CodeLang; title: string; code: string }
  links?: ChapterId[]
}

export const PARTS: Record<string, PartItem> = {
  client: {
    title: 'Клиент',
    where: 'вне движка',
    what: <>Шлёт OpenAI-совместимый запрос и читает ответ потоком. Движок сам по себе уже HTTP-сервер: для одной реплики больше ничего не нужно.</>,
    facts: [<>Тот же API понимают SGLang, TensorRT-LLM (<code>trtllm-serve</code>) и Dynamo.</>],
    links: ['request'],
  },
  http: {
    title: 'HTTP-сервер',
    where: 'процесс API server',
    what: (
      <>
        FastAPI и Uvicorn принимают запросы OpenAI API и стримят ответы. Это отдельный процесс, чтобы медленный Python-код (разбор JSON, токенизация,
        стриминг) не задерживал цикл, который кормит GPU.
      </>
    ),
    facts: [
      <>По умолчанию API server один, при data parallel их столько же, сколько DP-рангов (<code>--api-server-count</code>).</>,
      'Каждый API server связан с каждым EngineCore через ZMQ.',
    ],
    config: { lang: 'bash', title: 'запуск', code: 'vllm serve Qwen/Qwen3-8B --port 8000' },
    links: ['request'],
  },
  input: {
    title: 'Шаблон и токенизация',
    where: 'процесс API server',
    what: (
      <>
        Рендерит chat template модели, токенизирует результат и упаковывает в <code>EngineCoreRequest</code>: token IDs плюс параметры сэмплинга. Дальше
        движок работает только с числами.
      </>
    ),
    facts: [
      <>Свой шаблон: <code>--chat-template</code>. Несовпадение шаблона частая причина ответов «хуже, чем в демо».</>,
      <>Для внешнего frontend вроде Dynamo есть режим «токены на входе и на выходе»: <code>/inference/v1/generate</code>.</>,
    ],
    links: ['template', 'tokenization'],
  },
  output: {
    title: 'Детокенизация и парсеры',
    where: 'процесс API server',
    what: (
      <>
        Превращает новые token IDs в текст инкрементально (DecodeStream), проверяет стоп-строки и раскладывает текст на рассуждение, ответ и вызовы
        инструментов. Стоп-строку ловят здесь, после чего запрос прерывается в EngineCore, а стоп-токены и <code>max_tokens</code> проверяет сам EngineCore.
      </>
    ),
    facts: [
      <>Парсеры: <code>--reasoning-parser qwen3</code>, <code>--enable-auto-tool-choice --tool-call-parser hermes</code>.</>,
      <>Поле с рассуждением vLLM сейчас называет <code>reasoning</code> (раньше <code>reasoning_content</code>, как у SGLang и Dynamo).</>,
    ],
    config: {
      lang: 'bash',
      title: 'Qwen3 с парсерами',
      code: `vllm serve Qwen/Qwen3-8B \\
  --reasoning-parser qwen3 \\
  --enable-auto-tool-choice --tool-call-parser hermes`,
    },
    links: ['detokenization', 'parsers'],
  },
  sched: {
    title: 'Scheduler',
    where: 'процесс EngineCore',
    what: (
      <>
        На каждом шаге решает, сколько токенов какого запроса посчитать. Фаз «prefill» и «decode» у него нет: у запроса есть число уже посчитанных токенов, и
        шаг даёт ему столько, сколько влезает в бюджет. Сначала по токену получают идущие запросы (running), остаток бюджета достаётся новым (waiting).
        Длинный промпт, который не влез, режется на куски: chunked prefill.
      </>
    ),
    facts: [
      <>Бюджет шага <code>max_num_batched_tokens</code>: у <code>vllm serve</code> на H100 по умолчанию 8 192, на B200 16 384.</>,
      <>Не больше <code>max_num_seqs</code> запросов одновременно (1 024 на H100).</>,
      'Нет места под KV: вытесняется последний допущенный запрос, его KV стирается и потом пересчитывается (только recompute).',
      'С v0.14 CPU планирует шаг N+1, пока GPU считает шаг N (async scheduling).',
    ],
    config: { lang: 'bash', title: 'бюджет и лимит', code: 'vllm serve Qwen/Qwen3-8B \\\n  --max-num-batched-tokens 8192 \\\n  --max-num-seqs 256' },
    links: ['batching'],
  },
  kvm: {
    title: 'KV cache manager',
    where: 'процесс EngineCore',
    what: (
      <>
        Раздаёт запросам блоки KV по 16 токенов из общего пула и ведёт таблицы блоков (PagedAttention). Он же ищет полные блоки промпта в prefix cache по
        цепочке хэшей: совпавшие не считаются заново. Освобождённые блоки не стираются сразу и остаются переиспользуемыми, пока их не займут.
      </>
    ),
    facts: [
      'Размер блока 16 во всех релизах vLLM (в основной ветке с ноября 2026 уже 64).',
      <>Бюджет памяти <code>gpu_memory_utilization</code>: доля всей памяти GPU на инстанс (0,92 с v0.20), KV получает то, что осталось после весов и активаций.</>,
      <>При старте в логе видно ёмкость: <code>GPU KV cache size: N tokens</code>.</>,
    ],
    config: { lang: 'bash', title: 'память', code: 'vllm serve Qwen/Qwen3-8B \\\n  --gpu-memory-utilization 0.92 \\\n  --block-size 16' },
    links: ['paged-attention', 'prefix-caching', 'kv-cache'],
  },
  som: {
    title: 'Structured output',
    where: 'процесс EngineCore',
    what: (
      <>
        Если запрос требует JSON по схеме, грамматика компилируется в пуле потоков, а запрос ждёт (<code>WAITING_FOR_FSM</code>). Потом на каждом шаге,
        параллельно с forward на GPU, здесь заполняется битовая маска разрешённых токенов, которую sampler накладывает на логиты.
      </>
    ),
    facts: [<>Backend по умолчанию <code>auto</code>: xgrammar, при неподдерживаемой схеме guidance.</>, 'Маска нужна только при выборе токена: prefill она не замедляет.'],
    links: ['structured-output'],
  },
  runner: {
    title: 'Model runner',
    where: 'процесс GPU-воркера (по одному на GPU)',
    what: (
      <>
        Готовит входы шага: токены всех запросов склеены в одну последовательность без padding, рядом позиции и адреса слотов KV. Однородные decode-шаги
        запускает через CUDA graphs: записанную последовательность сотен ядер, которую GPU выполняет по одной команде.
      </>
    ),
    facts: [
      <>Режим по умолчанию <code>FULL_AND_PIECEWISE</code>: полный граф для шагов из одних decode-токенов, кусочный для остальных. <code>--enforce-eager</code> выключает графы.</>,
      'Графы стоят памяти GPU и времени на старте: их захватывают под набор размеров batch, реальный batch дополняется до ближайшего.',
      'Воркеров у одного EngineCore TP × PP: каждый держит свою часть весов и свою часть KV.',
    ],
    links: ['parallelism'],
  },
  kernels: {
    title: 'Модель и ядра',
    where: 'процесс GPU-воркера',
    what: (
      <>
        Обычная модель на PyTorch, но горячие места заменены быстрыми ядрами. Attention считают ядра FlashAttention или FlashInfer: они читают KV прямо из
        разбросанных блоков по таблице блоков и не пишут в память промежуточную матрицу весов. Линейные слои считаются как GEMM из cuBLAS или CUTLASS.
      </>
    ),
    facts: [
      'PagedAttention прежде всего про управление памятью: само ядро в статье было на 20–26 % медленнее ядра FasterTransformer, выигрыш дал больший batch.',
      <>Ядро attention выбирается автоматически под GPU и модель (для MLA свои: FlashMLA и другие).</>,
    ],
    links: ['attention', 'roofline'],
  },
  sampler: {
    title: 'Sampler',
    where: 'процесс GPU-воркера',
    what: (
      <>
        Берёт логиты последней позиции каждого запроса, накладывает маску грамматики (запрещённым −∞), применяет temperature, top-k, top-p и выбирает токен.
        Токен уходит в EngineCore, который дописывает его в запрос и проверяет стоп-токены и <code>max_tokens</code>.
      </>
    ),
    facts: ['Пока длинный промпт идёт кусками, выход промежуточных шагов отбрасывается: первый токен появляется после последнего куска.'],
    links: ['sampling'],
  },
}

/* ---------------------------------------------------------------------------------------------
 * Сравнение движков
 * ------------------------------------------------------------------------------------------- */

export interface EngineCard {
  id: 'vllm' | 'sglang' | 'trtllm' | 'local'
  name: string
  origin: string
  idea: ReactNode
  procs: string[]
  strengths: ReactNode[]
  choose: ReactNode[]
  defaults: Array<[string, ReactNode]>
  /** Оценка из таблицы книги (с. 106): скорость, простота, поддержка моделей, железо. */
  book?: Array<[string, string]>
  run: { lang: CodeLang; code: string }
}

export const ENGINES: EngineCard[] = [
  {
    id: 'vllm',
    name: 'vLLM',
    origin: 'UC Berkeley, июнь 2023; сейчас проект PyTorch Foundation',
    idea: (
      <>
        <strong>PagedAttention</strong>: KV cache режется на блоки-страницы и выдаётся по мере роста запроса, как ОС выдаёт процессу страницы памяти. Поверх
        этого scheduler с бюджетом токенов, который в каждом шаге смешивает prefill-куски и decode, и prefix cache на цепочке хэшей блоков.
      </>
    ),
    procs: ['API server', 'EngineCore', 'GPU-воркеры'],
    strengths: [
      'Самая широкая поддержка: почти любая открытая модель работает с первого дня, на GPU NVIDIA, AMD и Intel и на TPU.',
      'Проще всего начать: одна команда, хорошие значения по умолчанию.',
      'Богатая экосистема коннекторов KV: LMCache, NIXL, Mooncake, свой оффлоад в RAM.',
    ],
    choose: [
      'Быстро поднять сервер с хорошей производительностью почти для любой открытой модели.',
      'Нужно не только железо NVIDIA или маленькие и старые GPU, где TensorRT-LLM почти ничего не даёт.',
    ],
    defaults: [
      ['блок KV', '16 токенов'],
      ['chunked prefill', 'включён'],
      ['prefix caching', 'включён'],
      ['если кончился KV', 'вытеснение и пересчёт'],
    ],
    book: [
      ['скорость', 'хорошая'],
      ['простота', 'просто'],
      ['модели', 'больше всего'],
      ['железо', 'GPU разных вендоров, TPU'],
    ],
    run: { lang: 'bash', code: 'vllm serve Qwen/Qwen3-8B --tensor-parallel-size 1' },
  },
  {
    id: 'sglang',
    name: 'SGLang',
    origin: 'команда LMSYS, декабрь 2023',
    idea: (
      <>
        <strong>RadixAttention</strong>: весь посчитанный KV, и промпты, и ответы, хранится как дерево префиксов с точностью до токена. Новый запрос спускается
        по дереву, вытесняются сначала давно не нужные листья. Плюс overlap scheduler: CPU собирает следующий batch, пока GPU считает текущий.
      </>
    ),
    procs: ['HTTP и TokenizerManager', 'Scheduler с моделью', 'DetokenizerManager'],
    strengths: [
      'Нагрузки с большим переиспользованием префиксов: агенты, многоходовые чаты, few-shot.',
      'Большие MoE-модели на многих нодах (DeepSeek, Kimi): DP attention, expert parallelism и PD-дезагрегация.',
      'Cache-aware роутер, иерархический кэш HiCache, быстрый structured output (xgrammar).',
    ],
    choose: [
      'Отличный throughput при приличной latency на больших MoE.',
      'Много общих префиксов, нужен контроль и кастомизация, готовность участвовать в сообществе.',
    ],
    defaults: [
      ['страница KV', '1 токен'],
      ['chunked prefill', 'включён, размер от памяти GPU'],
      [
        'batch',
        <>
          если есть новые промпты, шаг целиком уходит на prefill; смешивать с decode можно флагом <code>--enable-mixed-chunk</code>
        </>,
      ],
      ['если кончился KV', 'отзыв decode-запросов в очередь'],
    ],
    book: [
      ['скорость', 'хорошая'],
      ['простота', 'просто'],
      ['модели', 'больше всего'],
      ['железо', 'NVIDIA, AMD'],
    ],
    run: { lang: 'bash', code: 'python -m sglang.launch_server --model-path Qwen/Qwen3-8B --tp 1' },
  },
  {
    id: 'trtllm',
    name: 'TensorRT-LLM',
    origin: 'NVIDIA, октябрь 2023; версия 1.0 в 2025',
    idea: (
      <>
        <strong>Ядра NVIDIA</strong>: рукописные и вручную слитые kernels под Hopper и Blackwell, часть закрытые, поддержка FP8 и NVFP4. С версии 1.0 это
        PyTorch-рантайм по умолчанию, без отдельной сборки движка; старый TensorRT-бэкенд удалён в ветке 1.4.
      </>
    ),
    procs: ['LLM API', 'PyExecutor (C++ scheduler и KV)', 'ядра на GPU'],
    strengths: [
      'Обычно самый быстрый на хорошо поддержанных моделях на Hopper и новее.',
      'Квантизация FP8 и NVFP4, спекулятивный декодинг, in-flight batching на C++.',
      'Самая глубокая интеграция с Dynamo.',
    ],
    choose: [
      'Популярная архитектура на H100, H200, B200, и вы готовы к лишней инженерной работе ради максимума.',
      'Весь парк на NVIDIA: другие GPU TensorRT-LLM не поддерживает.',
    ],
    defaults: [
      ['блок KV', '32 токена'],
      ['chunked prefill', 'выключен в LLM API'],
      ['допуск запросов', 'только если KV хватит до конца'],
      ['overlap scheduler', 'включён (+1 шаг decode)'],
    ],
    book: [
      ['скорость', 'лучшая'],
      ['простота', 'сложно'],
      ['модели', 'часть'],
      ['железо', 'только NVIDIA'],
    ],
    run: { lang: 'bash', code: 'trtllm-serve Qwen/Qwen3-8B --config config.yaml' },
  },
  {
    id: 'local',
    name: 'llama.cpp и Ollama',
    origin: 'локальный запуск',
    idea: (
      <>
        Движки для своего компьютера, а не для кластера. Модели обычно в формате <strong>GGUF</strong>: один файл, где веса, токенизатор и даже chat template,
        чаще всего сильно квантованные. Ollama оборачивает llama.cpp: скачивает модель одной командой и поднимает локальный HTTP API.
      </>
    ),
    procs: ['один процесс', 'CPU, Apple Silicon или игровая GPU'],
    strengths: [
      'Без сети и облака: приватность, нет сетевой задержки, бесплатно для разработчика.',
      'Сильная квантизация позволяет запускать модели на 100 млрд параметров и больше на мощном личном железе, MoE помогает.',
    ],
    choose: [
      'Разработка, эксперименты, приватные данные на ноутбуке.',
      'Не для прода с сотнями одновременных пользователей. Continuous batching и параллельные слоты у llama-server есть, но серверные движки рассчитаны на большую нагрузку: KV блоками с prefix cache, общим для всех одновременных запросов, отлаженный tensor и expert parallel на нескольких GPU, дезагрегация prefill и decode и интеграция с Dynamo.',
    ],
    defaults: [
      ['формат', 'GGUF'],
      ['типичная точность', '4–8 бит на вес'],
    ],
    run: { lang: 'bash', code: 'ollama run qwen3:8b' },
  },
]
