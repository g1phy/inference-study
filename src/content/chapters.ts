import type { ComponentType } from 'react'

/**
 * РЕЕСТР ГЛАВ (SPEC.md §2). Источник правды для навигации, заголовков,
 * нумерации и ссылок «Дальше: …». Тексты лидов здесь черновые: глава может
 * передать свой лид пропсом `lead` в <Chapter>.
 *
 * Каждая глава живёт в src/chapters/<id>/index.tsx (default export) и
 * подгружается лениво через `load`.
 */

export type PartId = 'input' | 'model' | 'phases' | 'memory' | 'batch' | 'engines' | 'output' | 'prod'

export interface PartInfo {
  id: PartId
  number: number
  /** Римская цифра для подписи «Часть II». */
  roman: string
  title: string
}

export const PARTS: readonly PartInfo[] = [
  { id: 'input', number: 1, roman: 'I', title: 'Вход: от текста к числам' },
  { id: 'model', number: 2, roman: 'II', title: 'Внутри модели' },
  { id: 'phases', number: 3, roman: 'III', title: 'Две фазы генерации' },
  { id: 'memory', number: 4, roman: 'IV', title: 'Память: KV cache' },
  { id: 'batch', number: 5, roman: 'V', title: 'Много запросов и много GPU' },
  { id: 'engines', number: 6, roman: 'VI', title: 'Движки и оркестрация' },
  { id: 'output', number: 7, roman: 'VII', title: 'Выход и формат ответа' },
  { id: 'prod', number: 8, roman: 'VIII', title: 'Ускорения, прод и итог' },
]

export const CHAPTER_IDS = [
  'hero',
  'request',
  'template',
  'tokenization',
  'embeddings',
  'transformer',
  'attention',
  'sampling',
  'prefill-decode',
  'roofline',
  'metrics',
  'kv-cache',
  'paged-attention',
  'prefix-caching',
  'kv-tiers',
  'batching',
  'disaggregation',
  'parallelism',
  'engines',
  'dynamo',
  'routing',
  'detokenization',
  'parsers',
  'structured-output',
  'speculative',
  'quantization',
  'production',
  'cheatsheet',
  'glossary',
] as const

export type ChapterId = (typeof CHAPTER_IDS)[number]

export type ChapterModule = { default: ComponentType }

export interface ChapterInfo {
  id: ChapterId
  /** null только у hero. */
  part: PartId | null
  /** 0 у hero, дальше 1..28. */
  number: number
  title: string
  /** Короткое имя для навигации. */
  nav: string
  /** Черновой лид: 1-3 предложения простыми словами. */
  lead: string
  /** Примерная высота главы в px до монтирования (резерв места под якоря). */
  estimatedHeight?: number
  load: () => Promise<ChapterModule>
}

export const CHAPTERS: readonly ChapterInfo[] = [
  {
    id: 'hero',
    part: null,
    number: 0,
    title: 'Что происходит после Enter',
    nav: 'Путь запроса',
    lead: 'Весь путь запроса к LLM, от JSON до последнего токена: шаблон, токены, роутер, GPU, парсеры. На вашем собственном тексте.',
    estimatedHeight: 7300,
    load: () => import('../chapters/hero/index'),
  },
  {
    id: 'request',
    part: 'input',
    number: 1,
    title: 'Запрос: что приходит на сервер',
    nav: 'Запрос',
    lead: 'Всё начинается с одного HTTP-запроса: клиент отправляет POST на /v1/chat/completions с JSON внутри. Формат придумала OpenAI, и на нём же говорят vLLM, SGLang и NVIDIA Dynamo. По полям этого JSON можно прочитать карту всей системы: каждое поле включает свой механизм на сервере.',
    estimatedHeight: 6600,
    load: () => import('../chapters/request/index'),
  },
  {
    id: 'template',
    part: 'input',
    number: 2,
    title: 'Шаблон чата',
    nav: 'Шаблон чата',
    lead: 'Модель не видит JSON. Она видит одну длинную строку, где реплики разделены служебными маркерами вроде <|im_start|>. Эту строку собирает chat template, маленькая программа на языке шаблонов Jinja из tokenizer_config.json модели. Ошибка в шаблоне не роняет сервер, а тихо портит ответы.',
    estimatedHeight: 6400,
    load: () => import('../chapters/template/index'),
  },
  {
    id: 'tokenization',
    part: 'input',
    number: 3,
    title: 'Токенизация',
    nav: 'Токенизация',
    lead: 'Модель работает только с числами. Токенизатор режет текст на токены, куски слов, и заменяет каждый номером в словаре: у Qwen3 их 151 669. Это обычная программа на CPU, без нейросети, но от неё зависят цена запроса, место в контексте и размер KV cache.',
    estimatedHeight: 7000,
    load: () => import('../chapters/tokenization/index'),
  },
  {
    id: 'embeddings',
    part: 'model',
    number: 4,
    title: 'Эмбеддинги: токен становится вектором',
    nav: 'Эмбеддинги',
    lead: 'Номер токена сам по себе ничего не значит. Каждый ID выбирает строку в большой таблице эмбеддингов: вектор из d_model чисел (у Qwen3-8B их 4 096). С этого момента модель работает только с векторами.',
    estimatedHeight: 4700,
    load: () => import('../chapters/embeddings/index'),
  },
  {
    id: 'transformer',
    part: 'model',
    number: 5,
    title: 'Слои трансформера',
    nav: 'Слои трансформера',
    lead: 'Вектор каждого токена проходит N одинаковых по устройству блоков: у Qwen3-8B их 36. В каждом блоке attention даёт токену забрать информацию у предыдущих токенов, а MLP обрабатывает каждый токен сам по себе. После последнего блока LM head превращает вектор в оценки для всех токенов словаря.',
    estimatedHeight: 8000,
    load: () => import('../chapters/transformer/index'),
  },
  {
    id: 'attention',
    part: 'model',
    number: 6,
    title: 'Attention и Q/K/V',
    nav: 'Attention и Q/K/V',
    lead: 'Attention позволяет каждому токену собрать информацию из предыдущих токенов. Для этого токен строит три вектора: Query (что я ищу), Key (по чему меня найти) и Value (что я передам тем, кто на меня посмотрит). В трансформере это единственное место, где токены обмениваются информацией, и отсюда же появляется KV cache.',
    estimatedHeight: 9900,
    load: () => import('../chapters/attention/index'),
  },
  {
    id: 'sampling',
    part: 'model',
    number: 7,
    title: 'Логиты и сэмплинг',
    nav: 'Логиты и сэмплинг',
    lead: 'На выходе модель даёт не слово, а 151 936 чисел: по оценке (логиту) на каждую строку словаря. Softmax превращает их в вероятности, а сэмплер выбирает один токен. Temperature, top-k и top-p решают, насколько смело он рискует.',
    estimatedHeight: 5900,
    load: () => import('../chapters/sampling/index'),
  },
  {
    id: 'prefill-decode',
    part: 'phases',
    number: 8,
    title: 'Prefill и decode',
    nav: 'Prefill и decode',
    lead: 'Модель пишет ответ в две фазы. Prefill прогоняет через неё весь промпт за один проход и выдаёт первый токен. Decode дописывает остальные по одному токену за проход, пока не встретит стоп. У фаз разная физика, поэтому у них разные метрики и разные проблемы в проде.',
    estimatedHeight: 5700,
    load: () => import('../chapters/prefill-decode/index'),
  },
  {
    id: 'roofline',
    part: 'phases',
    number: 9,
    title: 'Compute-bound и memory-bound',
    nav: 'Compute- и memory-bound',
    lead: 'У GPU два предела: сколько операций в секунду он успевает сделать и сколько байт в секунду успевает подвезти из памяти. Какой упрётся первым, зависит от того, сколько вычислений приходится на каждый прочитанный байт. Prefill использует каждый вес для сотен и тысяч токенов, decode при batch 1 только для одного. Поэтому prefill упирается в вычисления, а decode в память.',
    estimatedHeight: 7500,
    load: () => import('../chapters/roofline/index'),
  },
  {
    id: 'metrics',
    part: 'phases',
    number: 10,
    title: 'Метрики: TTFT, ITL, throughput',
    nav: 'Метрики',
    lead: 'Как понять, быстрый ли сервис? Пользователь чувствует две вещи: сколько ждать первого слова и как быстро печатается ответ. Бизнес считает, сколько токенов в секунду выдаёт весь парк GPU и сколько запросов уложились в обещанные задержки. Это TTFT, ITL, throughput и goodput, и смотрят на них перцентилями, а не средним.',
    estimatedHeight: 7200,
    load: () => import('../chapters/metrics/index'),
  },
  {
    id: 'kv-cache',
    part: 'memory',
    number: 11,
    title: 'KV cache: что хранится и сколько весит',
    nav: 'KV cache',
    lead: 'Чтобы на каждом шаге decode не пересчитывать прошлое, модель хранит векторы K и V каждого прочитанного токена в каждом слое. Это и есть KV cache: он экономит огромный объём вычислений, но занимает память GPU и растёт с длиной контекста и числом одновременных запросов.',
    estimatedHeight: 7900,
    load: () => import('../chapters/kv-cache/index'),
  },
  {
    id: 'paged-attention',
    part: 'memory',
    number: 12,
    title: 'PagedAttention: память страницами',
    nav: 'PagedAttention',
    lead: 'Если резервировать память под максимальную длину каждого запроса, большая часть KV cache стоит пустой. PagedAttention режет кэш на блоки фиксированного размера, выдаёт их по мере роста запроса и ведёт для каждого запроса таблицу блоков, как операционная система ведёт таблицу страниц памяти.',
    estimatedHeight: 6500,
    load: () => import('../chapters/paged-attention/index'),
  },
  {
    id: 'prefix-caching',
    part: 'memory',
    number: 13,
    title: 'Prefix caching и RadixAttention',
    nav: 'Prefix caching',
    lead: 'Многие запросы начинаются одинаково: системный промпт, few-shot примеры, документ, история диалога. Если KV этого общего начала уже лежит в кэше, prefill для него можно пропустить: запрос посчитает только свой новый хвост, и первый токен придёт быстрее.',
    estimatedHeight: 6700,
    load: () => import('../chapters/prefix-caching/index'),
  },
  {
    id: 'kv-tiers',
    part: 'memory',
    number: 14,
    title: 'Иерархия KV cache: L1, L2, L3',
    nav: 'Иерархия KV cache',
    lead: 'Память GPU быстрая, но маленькая, поэтому готовый KV вытесняют, и следующий похожий запрос снова платит за prefill. Если увозить KV на уровни побольше и помедленнее (RAM хоста, локальный NVMe, сетевое хранилище), его можно привезти обратно, и это часто быстрее, чем пересчитать. В Kubernetes такой нижний уровень обычно ставят DaemonSet: по кэш-серверу на каждую GPU-ноду, общему для всех её воркеров и переживающему их рестарты.',
    estimatedHeight: 9200,
    load: () => import('../chapters/kv-tiers/index'),
  },
  {
    id: 'batching',
    part: 'batch',
    number: 15,
    title: 'Continuous batching и chunked prefill',
    nav: 'Continuous batching',
    lead: 'Один GPU обслуживает сразу десятки запросов: за один шаг он делает по токену для каждого, а веса читает один раз на всех. Continuous batching пересобирает эту группу на каждом шаге, а chunked prefill режет длинный промпт на куски, чтобы он не останавливал остальных.',
    estimatedHeight: 8100,
    load: () => import('../chapters/batching/index'),
  },
  {
    id: 'disaggregation',
    part: 'batch',
    number: 16,
    title: 'Агрегированный и дезагрегированный serving',
    nav: 'Дезагрегация',
    lead: 'В агрегированном режиме одни и те же GPU делают и prefill, и decode, и фазы мешают друг другу. Дезагрегация разводит их по разным пулам: prefill-воркеры считают KV cache промпта и передают его decode-воркерам по быстрому каналу между GPU.',
    estimatedHeight: 10100,
    load: () => import('../chapters/disaggregation/index'),
  },
  {
    id: 'parallelism',
    part: 'batch',
    number: 17,
    title: 'Параллелизм: TP, PP, EP, DP',
    nav: 'Параллелизм',
    lead: 'Большие модели не помещаются в один GPU, а если и помещаются, одному GPU не хватает памяти под KV cache или скорости. Модель раскладывают на несколько GPU четырьмя способами: TP режет матрицы каждого слоя, PP делит слои на стадии, EP раздаёт экспертов MoE, DP делает независимые реплики.',
    estimatedHeight: 7500,
    load: () => import('../chapters/parallelism/index'),
  },
  {
    id: 'engines',
    part: 'engines',
    number: 18,
    title: 'Движки: vLLM, SGLang, TensorRT-LLM',
    nav: 'Движки',
    lead: 'Движок крутит одну реплику модели. На каждом шаге он решает, какие токены каких запросов прогнать через GPU, где держать их KV cache, какими ядрами считать и как выбрать следующий токен. vLLM, SGLang и TensorRT-LLM делают одно и то же, но с разными акцентами.',
    estimatedHeight: 9400,
    load: () => import('../chapters/engines/index'),
  },
  {
    id: 'dynamo',
    part: 'engines',
    number: 19,
    title: 'NVIDIA Dynamo',
    nav: 'NVIDIA Dynamo',
    lead: 'Dynamo не движок, а распределённый слой над движками. Он сам принимает HTTP, превращает текст в токены, выбирает воркер по KV cache и нагрузке, разводит prefill и decode по разным GPU и собирает ответ обратно в текст. Матрицы при этом считает обычный vLLM, SGLang или TensorRT-LLM.',
    estimatedHeight: 9100,
    load: () => import('../chapters/dynamo/index'),
  },
  {
    id: 'routing',
    part: 'engines',
    number: 20,
    title: 'KV-aware routing',
    nav: 'KV-aware routing',
    lead: 'У каждого воркера свой KV cache. Если отправить запрос туда, где уже лежит его префикс, prefill станет короче. Если же слать всё на самый «тёплый» воркер, он захлебнётся. KV-aware роутер взвешивает обе вещи одной функцией стоимости.',
    estimatedHeight: 8200,
    load: () => import('../chapters/routing/index'),
  },
  {
    id: 'detokenization',
    part: 'output',
    number: 21,
    title: 'Детокенизация',
    nav: 'Детокенизация',
    lead: 'Модель выдаёт номера токенов, а пользователь ждёт текст, причём сразу и по кусочку. Детокенизатор переводит номера обратно в текст на лету и решает, какой кусок уже можно отправить: символ бывает разрезан между двумя токенами, а стоп-строка может прийти по частям в нескольких токенах.',
    estimatedHeight: 6600,
    load: () => import('../chapters/detokenization/index'),
  },
  {
    id: 'parsers',
    part: 'output',
    number: 22,
    title: 'Парсеры reasoning и tool calls',
    nav: 'Парсеры',
    lead: 'Модель пишет одну сплошную ленту: мысли в <think>, ответ, вызовы инструментов в <tool_call>. Клиенту нужны раздельные поля. Парсер на сервере читает ленту по мере генерации и раскладывает её по полям reasoning_content, content и tool_calls, прежде чем отправить клиенту.',
    estimatedHeight: 8400,
    load: () => import('../chapters/parsers/index'),
  },
  {
    id: 'structured-output',
    part: 'output',
    number: 23,
    title: 'Structured output',
    nav: 'Structured output',
    lead: 'Модель можно заставить ответить валидным JSON по вашей схеме. На каждом шаге decode сервер вычисляет, какие токены не сломают JSON, и ставит остальным логит −∞ ещё до сэмплинга. Синтаксис гарантирован (если ответ не оборвался по max_tokens), а содержание по-прежнему выбирает модель.',
    estimatedHeight: 7800,
    load: () => import('../chapters/structured-output/index'),
  },
  {
    id: 'speculative',
    part: 'prod',
    number: 24,
    title: 'Speculative decoding',
    nav: 'Speculative decoding',
    lead: 'Decode упирается в память: на каждом шаге GPU читает все веса модели ради одного токена, а вычислители почти простаивают. Speculative decoding тратит этот запас: дешёвый черновик угадывает несколько токенов вперёд, большая модель проверяет их все за один проход. Ответ остаётся тем же, а токенов за проход становится больше.',
    estimatedHeight: 7600,
    load: () => import('../chapters/speculative/index'),
  },
  {
    id: 'quantization',
    part: 'prod',
    number: 25,
    title: 'Квантизация',
    nav: 'Квантизация',
    lead: 'Квантизация хранит веса, а часто и KV cache, в меньшем числе бит: BF16, FP8, FP4. Меньше байт читать из памяти, значит быстрее decode, больше места под KV и дешевле железо. Цена: небольшая потеря точности, которую обязательно проверяют.',
    estimatedHeight: 8000,
    load: () => import('../chapters/quantization/index'),
  },
  {
    id: 'production',
    part: 'prod',
    number: 26,
    title: 'Прод: масштабирование и эксплуатация',
    nav: 'Прод',
    lead: 'Модель, которая быстро отвечает на одной GPU, ещё не сервис. Трафик меняется в течение суток, новая реплика поднимается минутами, а пользователь замечает каждую секунду в очереди. Здесь про то, как масштабироваться по правильному сигналу, сокращать холодный старт, выкатывать без простоя и что смотреть в мониторинге.',
    estimatedHeight: 11300,
    load: () => import('../chapters/production/index'),
  },
  {
    id: 'cheatsheet',
    part: 'prod',
    number: 27,
    title: 'Итог: весь путь на одной странице',
    nav: 'Весь путь',
    lead: 'Весь путь запроса на одной странице: что происходит на каждой стадии, что в ней настраивают и по какой метрике её судят. Формулы ниже посчитаны для вашего запроса, вашей модели и вашего GPU.',
    estimatedHeight: 8100,
    load: () => import('../chapters/cheatsheet/index'),
  },
  {
    id: 'glossary',
    part: 'prod',
    number: 28,
    title: 'Глоссарий',
    nav: 'Глоссарий',
    lead: 'Все термины сайта в одном месте: короткое объяснение по-русски, почему это важно для инференса, и глава, где термин разобран на примерах.',
    estimatedHeight: 12000,
    load: () => import('../chapters/glossary/index'),
  },
]

const BY_ID = new Map<string, ChapterInfo>(CHAPTERS.map((c) => [c.id, c]))
const INDEX = new Map<string, number>(CHAPTERS.map((c, i) => [c.id, i]))

export function isChapterId(value: string): value is ChapterId {
  return BY_ID.has(value)
}

export function getChapter(id: ChapterId): ChapterInfo {
  return BY_ID.get(id)!
}

/** То же, что getChapter, но для произвольной строки (например, из URL). */
export function findChapter(id: string): ChapterInfo | undefined {
  return BY_ID.get(id)
}

export function chapterIndex(id: ChapterId): number {
  return INDEX.get(id) ?? -1
}

export function nextChapter(id: ChapterId): ChapterInfo | undefined {
  const i = chapterIndex(id)
  return i >= 0 ? CHAPTERS[i + 1] : undefined
}

export function prevChapter(id: ChapterId): ChapterInfo | undefined {
  const i = chapterIndex(id)
  return i > 0 ? CHAPTERS[i - 1] : undefined
}

export function getPart(id: PartId): PartInfo {
  return PARTS.find((p) => p.id === id)!
}

export function chaptersOfPart(id: PartId): ChapterInfo[] {
  return CHAPTERS.filter((c) => c.part === id)
}

/** Глава, с которой начинается часть (для разделителя «Часть II» в ленте). */
export function isFirstInPart(id: ChapterId): boolean {
  const c = getChapter(id)
  if (!c.part) return false
  const prev = prevChapter(id)
  return !prev || prev.part !== c.part
}
