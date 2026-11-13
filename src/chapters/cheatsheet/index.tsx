import { useMemo, type ReactNode } from 'react'
import { ArrowSquareOutIcon, CaretRightIcon } from '@phosphor-icons/react'
import { DEFAULT_MBU, DEFAULT_MFU, STEP_OVERHEAD_MS, decodeStepBreakdown, prefillBreakdown } from '@/engine/perf'
import { Block, Callout, Chapter, ChapterLink, ScrollX, Takeaways, Term, cssVar, fmt } from '@/ui'
import type { ChapterId } from '@/content/chapters'
import { times, useHardwareState } from '@/chapters/prefill-decode/shared/hardware'
import { usePromptTokens } from '@/chapters/prefill-decode/shared/prompt'
import { FormulasFigure } from './Formulas'
import { NumbersFigure } from './NumbersTabs'
import { PathMapFigure } from './PathMap'
import { STAGES, type CheatStage } from './stages'

interface Doc {
  title: string
  href: string
  note: string
}

/** Официальная документация (ссылки из research/notes-*.md). */
const DOCS: Array<{ group: string; items: Doc[] }> = [
  {
    group: 'vLLM',
    items: [
      {
        title: 'Обзор архитектуры',
        href: 'https://docs.vllm.ai/en/latest/design/arch_overview.html',
        note: 'процессы, движок, планировщик',
      },
      {
        title: 'Prefix caching',
        href: 'https://docs.vllm.ai/en/latest/design/prefix_caching.html',
        note: 'хэши блоков и вытеснение',
      },
      {
        title: 'Оптимизация и тюнинг',
        href: 'https://docs.vllm.ai/en/latest/configuration/optimization.html',
        note: 'chunked prefill, preemption, параллелизм',
      },
      {
        title: 'Метрики',
        href: 'https://docs.vllm.ai/en/latest/usage/metrics.html',
        note: 'что отдаёт /metrics',
      },
    ],
  },
  {
    group: 'SGLang',
    items: [
      {
        title: 'Параметры сервера',
        href: 'https://docs.sglang.io/advanced_features/server_arguments.html',
        note: 'все флаги запуска',
      },
      {
        title: 'HiCache',
        href: 'https://docs.sglang.io/advanced_features/hicache_design.html',
        note: 'KV cache на уровнях L1, L2, L3',
      },
      {
        title: 'PD-дезагрегация',
        href: 'https://docs.sglang.io/advanced_features/pd_disaggregation.html',
        note: 'prefill и decode на разных GPU',
      },
    ],
  },
  {
    group: 'NVIDIA Dynamo',
    items: [
      {
        title: 'Архитектура',
        href: 'https://docs.nvidia.com/dynamo/knowledge-base/concepts/architecture',
        note: 'frontend, router, воркеры, NIXL',
      },
      {
        title: 'Устройство роутера',
        href: 'https://docs.nvidia.com/dynamo/knowledge-base/modular-components/router/router-design',
        note: 'KV-aware routing и стоимость',
      },
      {
        title: 'Дезагрегированный serving',
        href: 'https://docs.nvidia.com/dynamo/knowledge-base/concepts/system-architecture/disaggregated-serving',
        note: 'как едет KV между воркерами',
      },
    ],
  },
]

const BOOK_CHAPTERS: Array<{ ch: string; title: string; site: string }> = [
  {
    ch: 'Гл. 0-1',
    title: 'Inference, Prerequisites',
    site: 'зачем инференс, метрики TTFT и TPS, перцентили',
  },
  {
    ch: 'Гл. 2',
    title: 'Models',
    site: 'трансформер, attention, KV cache, prefill и decode, roofline',
  },
  {
    ch: 'Гл. 3',
    title: 'Hardware',
    site: 'GPU, память, интерконнекты, выбор железа',
  },
  {
    ch: 'Гл. 4',
    title: 'Software',
    site: 'CUDA, движки vLLM, SGLang, TensorRT-LLM, Dynamo',
  },
  {
    ch: 'Гл. 5',
    title: 'Techniques',
    site: 'квантизация, speculative decoding, кэширование, параллелизм, дезагрегация',
  },
  {
    ch: 'Гл. 7',
    title: 'Production',
    site: 'автоскейлинг, cold start, наблюдаемость, деплой',
  },
]

export default function CheatsheetChapter() {
  const state = useHardwareState()
  const prompt = usePromptTokens()
  const userPrompt = prompt.tokens?.length ?? null
  const hw = state.hw
  // для вопроса «почему prefill 2 000 токенов не в 2 000 раз дольше шага decode»
  const q = useMemo(() => {
    const pre = prefillBreakdown(hw.model, hw.gpu, 2000, hw.perf)
    const dec = decodeStepBreakdown(hw.model, hw.gpu, {
      ...hw.perf,
      batch: 1,
      avgContext: 2000,
    })
    return { pre: pre.ms, step: dec.ms, preBound: pre.bound, preWeights: pre.weightBytes, decWeights: dec.weightBytes }
  }, [hw])
  const onGpus = `${hw.tp > 1 ? `${hw.tp} × ` : ''}${hw.gpu.name}`

  return (
    <Chapter
      id="cheatsheet"
      lead={
        <>
          Весь путь запроса на одной странице: что происходит на каждой стадии, что в ней настраивают и по какой метрике её судят. Формулы ниже
          посчитаны для вашего запроса, вашей модели и вашего GPU.
        </>
      }
    >
      <p>
        Запрос проходит девять стадий. На CPU он превращается из JSON в токены, на GPU становится <Term t="KV cache" /> и новыми токенами, потом снова
        на CPU превращается в текст и уходит клиенту потоком. Две стадии на GPU, <Term t="prefill" /> и <Term t="decode" />, тратят почти всё время и
        почти все деньги; остальные решают, сколько этого времени придётся ждать и сколько работы можно не делать.
      </p>

      <PathMapFigure />

      <h3>Шпаргалка по стадиям</h3>
      <p>То же самое одной таблицей: удобно держать открытой рядом с дашбордом.</p>

      <Block width="wide">
        <div className="hidden md:block">
          <ScrollX label="Шпаргалка по стадиям" scrollerClassName="rounded-panel border border-line bg-surface">
            <table className="w-full min-w-[920px] text-left text-[14px] leading-snug">
              <caption className="sr-only">Стадии пути запроса: что происходит и кто это делает, что тюнят, главная метрика, главы</caption>
              <thead className="bg-surface-2 text-[13px] text-ink-2">
                <tr>
                  <th scope="col" className="w-[16%] px-3 py-2.5 font-medium">
                    Стадия
                  </th>
                  <th scope="col" className="w-[36%] px-3 py-2.5 font-medium">
                    Что происходит и кто это делает
                  </th>
                  <th scope="col" className="w-[19%] px-3 py-2.5 font-medium">
                    Что тюнят
                  </th>
                  <th scope="col" className="w-[13%] px-3 py-2.5 font-medium">
                    Главная метрика
                  </th>
                  <th scope="col" className="w-[16%] px-3 py-2.5 font-medium">
                    Главы
                  </th>
                </tr>
              </thead>
              <tbody>
                {STAGES.map((s) => (
                  <tr key={s.id} className="border-t border-line align-top">
                    <th scope="row" className="px-3 py-2.5 font-medium text-ink">
                      <StageName s={s} />
                      <div className="mt-1 text-[12px] leading-snug font-normal text-ink-3">{s.where}</div>
                    </th>
                    <td className="px-3 py-2.5 text-ink-2">
                      {s.what}
                      {s.stack && <div className="mt-1.5 text-[13px] text-ink-2">{s.stack}</div>}
                    </td>
                    <td className="px-3 py-2.5 text-ink-2">{s.tune}</td>
                    <td className="px-3 py-2.5 text-ink">{s.metric}</td>
                    <td className="px-3 py-2.5">
                      <ChapterList ids={s.chapters} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollX>
        </div>
        {/* телефон: та же шпаргалка карточками, без горизонтальной прокрутки */}
        <ol className="grid gap-3 md:hidden">
          {STAGES.map((s) => (
            <li key={s.id} className="grid gap-2.5 rounded-panel border border-line bg-surface p-3.5">
              <div className="grid gap-0.5">
                <div className="text-[15px] font-semibold text-ink">
                  <StageName s={s} />
                </div>
                <div className="text-[12px] leading-snug text-ink-3">{s.where}</div>
              </div>
              <dl className="grid gap-2 text-[14px] leading-snug">
                <CardRow label="Что происходит">{s.what}</CardRow>
                {s.stack && <CardRow label="Кто делает">{s.stack}</CardRow>}
                <CardRow label="Что тюнят">{s.tune}</CardRow>
                <CardRow label="Главная метрика">{s.metric}</CardRow>
                <CardRow label="Главы">
                  <ChapterList ids={s.chapters} />
                </CardRow>
              </dl>
            </li>
          ))}
        </ol>
      </Block>

      <h3>Ключевые формулы</h3>
      <p>
        Почти любую прикидку в инференсе можно сделать восемью формулами. Ridge point, нижняя граница шага decode и передача KV здесь идеальные:
        100&nbsp;% полосы памяти, без накладных. Prefill, время ответа и закон Литтла посчитаны так же, как в главах 8-10 ({fmt.pct(DEFAULT_MBU)}{' '}
        полосы, {fmt.pct(DEFAULT_MFU)} пика, {fmt.ms(STEP_OVERHEAD_MS)} накладных на шаг), поэтому они совпадают с фигурами этих глав. Подробные
        выводы в главах <ChapterLink to="roofline" />, <ChapterLink to="kv-cache" /> и <ChapterLink to="disaggregation" />.
      </p>

      <FormulasFigure state={state} userPrompt={userPrompt} />

      <h3>Ключевые числа</h3>
      <p>Справочник, к которому возвращаются чаще всего: железо, модели, каналы и дефолты движков.</p>

      <NumbersFigure />

      <Callout kind="pitfall" title="что в источниках успело устареть">
        <ul>
          <li>
            Дезагрегация в Dynamo 1.x: prefill-воркер выбирается первым, а KV едет через NIXL. Схема «сначала decode, а prefill через очередь» из
            старых докладов и книги это прежний дизайн.
          </li>
          <li>
            Формула стоимости KV-роутера Dynamo изменилась в 1.2; вариант «вес × prefill-блоки + decode-блоки» относится к версиям 1.1 и раньше.
          </li>
          <li>KVBM в Dynamo объявлен устаревшим в 1.5, а к версии 1.7 его убирают (из основной ветки уже удалён); на его месте нативный оффлоад движков, LMCache, SGLang HiCache.</li>
          <li>vLLM переименовал поле рассуждений в стриме: reasoning_content стало reasoning; SGLang и Dynamo используют reasoning_content.</li>
          <li>Сеть: есть InfiniBand XDR на 800&nbsp;Гбит/с и Ethernet на 400&nbsp;Гбит/с на порт, «Ethernet до 100&nbsp;Гбит/с» больше не верно.</li>
        </ul>
      </Callout>

      <h3>Проверьте себя</h3>
      <p>Восемь вопросов, которые собирают весь путь вместе. Попробуйте ответить сами, потом раскройте ответ.</p>

      <Block>
        <div className="grid gap-2">
          <Quiz n={1} q="Почему prefill промпта на 2 000 токенов идёт не в 2 000 раз дольше шага decode?" to={['roofline', 'prefill-decode']}>
            Шаг decode почти целиком уходит на чтение весов: одного токена мало, чтобы загрузить вычислители.{' '}
            {hw.model.experts ? (
              <>
                У {hw.model.name} (<Term t="MoE (Mixture of Experts)">MoE</Term>) шаг decode читает только активные веса, {fmt.bytes(q.decWeights)}, а
                prefill 2&nbsp;000 токенов задевает почти всех экспертов и читает почти всю модель, {fmt.bytes(q.preWeights)}, зато один раз на все
                2&nbsp;000 токенов.{' '}
                {q.preBound === 'memory'
                  ? 'Каждый токен считает только своими экспертами, поэтому prefill ещё упирается в чтение весов, а не в вычисления.'
                  : 'Этого хватает, чтобы prefill упёрся в вычисления.'}{' '}
                На {onGpus} это около {fmt.ms(q.pre)} против {fmt.ms(q.step)} на шаг decode, то есть в {times(q.pre / q.step)}, а не в 2&nbsp;000.
              </>
            ) : (
              <>
                Prefill читает те же веса один раз на все 2&nbsp;000 токенов и{' '}
                {q.preBound === 'compute' ? 'упирается уже в вычисления' : 'при такой длине всё ещё упирается в их чтение'}. Для {hw.model.name} на{' '}
                {onGpus} это около {fmt.ms(q.pre)} против {fmt.ms(q.step)} на шаг decode, то есть в {times(q.pre / q.step)}, а не в 2&nbsp;000.
              </>
            )}
          </Quiz>
          <Quiz n={2} q="Зачем токенизация стоит до роутера?" to={['routing', 'template']}>
            KV-aware роутер сравнивает хэши блоков токенов запроса с тем, что уже лежит в KV cache воркеров, и ищет самый длинный совпадающий префикс.
            Для этого ему нужны token id, а не текст. Поэтому в Dynamo шаблон и токенизация работают во frontend до роутера, а воркер получает уже
            готовые токены.
          </Quiz>
          <Quiz n={3} q="Что будет с prefix cache, если в начало system prompt поставить текущее время?" to={['prefix-caching']}>
            Хэш каждого блока зависит от всех токенов до него, поэтому изменение в самом начале меняет хэши всех блоков промпта. Совпадений не будет
            ни у одного запроса: каждый раз считается полный prefill, и TTFT растёт. Меняющиеся части (время, имя пользователя) ставят в конец
            промпта.
          </Quiz>
          <Quiz n={4} q="Где применяется маска structured output и почему не в postprocessor?" to={['structured-output', 'sampling']}>
            Маску накладывают на логиты в сэмплере decode-воркера, до выбора токена: запрещённым токенам ставят −∞, и модель не может выбрать
            неверный. Postprocessor видит уже выбранные токены: отбросить там неверный значило бы перегенерировать. Поэтому structured output это
            работа на стороне decode.
          </Quiz>
          <Quiz n={5} q="Когда дезагрегация не окупается?" to={['disaggregation']}>
            Когда нет хотя бы одного из условий: большого трафика, большой модели и prefill-тяжёлой нагрузки (длинные входы, короткие ответы). При
            низкой конкурентности агрегированный воркер часто лучше: нет передачи KV и дробления памяти на пулы. А если узкое место в ёмкости KV на
            decode, лишние prefill-реплики эту ёмкость только уменьшают.
          </Quiz>
          <Quiz n={6} q="Почему L3 KV cache удобно держать DaemonSet-ом?" to={['kv-tiers']}>
            <Term t="DaemonSet" /> ставит на каждую GPU-ноду ровно один под кэша: он владеет RAM и NVMe ноды, а воркеры ходят к нему локально. Кэш
            живёт отдельно от подов движка и переживает их рестарт, новая нода получает свой под сама, а соседние поды на одной ноде берут KV друг у
            друга.
          </Quiz>
          <Quiz n={7} q="Чем vLLM отличается от Dynamo?" to={['engines', 'dynamo']}>
            vLLM, как SGLang и TensorRT-LLM, это движок: он ведёт batch и KV cache одной реплики и гоняет модель на её GPU. Dynamo не движок, а
            распределённый слой над движками: frontend принимает HTTP, рендерит шаблон, токенизирует, выбирает воркер по KV cache и нагрузке,
            переносит KV при дезагрегации, детокенизирует и парсит. Воркеры с vLLM, SGLang или TensorRT-LLM получают и отдают только token id.
          </Quiz>
          <Quiz n={8} q="Почему автоскейлер нельзя настраивать по загрузке GPU из nvidia-smi?" to={['production', 'metrics']}>
            Эта метрика показывает лишь долю времени, когда на GPU исполнялось хоть одно ядро (kernel). Decode при batch 1 даёт почти 100&nbsp;%,
            используя меньше процента пиковых FLOPS, поэтому она не отличает недогруженный GPU от перегруженного. Масштабируются по concurrency и
            очереди запросов, а число реплик прикидывают законом Литтла.
          </Quiz>
        </div>
      </Block>

      <h3>Что читать дальше</h3>
      <p>
        Главный источник фактов этого сайта: книга Philip Kiely «Inference Engineering» (Baseten Books, 2026). Сайт пересказывает её своими словами и
        сверяет с документацией на ноябрь 2026, поэтому в нескольких местах расходится с книгой (список выше). Как главы книги ложатся на сайт:
      </p>

      <Block width="wide">
        <div className="grid gap-x-6 gap-y-2 rounded-panel border border-line bg-surface p-4 sm:grid-cols-2">
          {BOOK_CHAPTERS.map((b) => (
            <div key={b.ch} className="grid grid-cols-[4.5rem_minmax(0,1fr)] gap-2 text-[14px] leading-snug">
              <span className="text-ink-3">{b.ch}</span>
              <span>
                <span className="font-medium text-ink">{b.title}</span>
                <span className="text-ink-2">: {b.site}</span>
              </span>
            </div>
          ))}
        </div>
      </Block>

      <p>Официальная документация, где появляются флаги и новые версии:</p>

      <Block width="wide">
        <div className="grid gap-4 @container sm:grid-cols-3">
          {DOCS.map((d) => (
            <section key={d.group} className="grid content-start gap-2 rounded-panel border border-line bg-surface p-4">
              <h4 className="text-[15px] font-semibold text-ink">{d.group}</h4>
              <ul className="grid gap-2">
                {d.items.map((it) => (
                  <li key={it.href} className="text-[14px] leading-snug">
                    <a href={it.href} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 font-medium">
                      {it.title}
                      <ArrowSquareOutIcon size={14} aria-hidden />
                      <span className="sr-only">(открывается в новой вкладке)</span>
                    </a>
                    <div className="text-[13px] text-ink-3">{it.note}</div>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      </Block>

      <Takeaways
        items={[
          <>
            Запрос: JSON → шаблон → токены → роутер → prefill (задаёт TTFT) → decode (задаёт ITL) → детокенизация и парсеры → SSE. Prefill и decode
            делает движок (vLLM, SGLang, TensorRT-LLM) на GPU; Dynamo над движками готовит токены, выбирает воркер по кэшу, переносит KV и собирает
            ответ.
          </>,
          <>
            Prefill длинного промпта упирается в вычисления, decode в полосу памяти (короткий prefill тоже ждёт память). KV cache растёт с длиной контекста и числом запросов и ограничивает batch;
            вытесненный из HBM KV увозят в RAM, на NVMe или в кэш ноды (DaemonSet) и привозят обратно, когда это быстрее, чем пересчитать prefill.
          </>,
          <>
            Почти каждая оптимизация либо убирает работу (prefix caching, квантизация, speculative decoding), либо лучше раскладывает её по GPU
            (continuous batching, KV-aware routing, дезагрегация, параллелизм).
          </>,
          <>Мерить перцентилями, рабочую точку выбирать по goodput при заданных SLO.</>,
        ]}
      />
    </Chapter>
  )
}

function StageName({ s }: { s: CheatStage }) {
  return (
    <>
      <span className="mr-1.5 text-[12px] font-normal text-ink-3">{s.n}</span>
      {s.role && <span aria-hidden className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: cssVar(s.role) }} />}
      {s.name}
    </>
  )
}

function ChapterList({ ids }: { ids: ChapterId[] }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1">
      {ids.map((c) => (
        <ChapterLink key={c} to={c} />
      ))}
    </div>
  )
}

function CardRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid gap-0.5">
      <dt className="text-[13px] text-ink-3">{label}</dt>
      <dd className="text-ink-2">{children}</dd>
    </div>
  )
}

/** Вопрос с ответом под катом: нативный details, раскрывается мышью, клавиатурой и поиском по странице. */
function Quiz({ n, q, to, children }: { n: number; q: string; to: ChapterId[]; children: ReactNode }) {
  return (
    <details className="group rounded-panel border border-line bg-surface">
      <summary className="flex cursor-pointer list-none items-start gap-2.5 rounded-panel px-4 py-3 text-left transition-colors hover:bg-surface-3/60 sm:px-5 [&::-webkit-details-marker]:hidden">
        <CaretRightIcon size={16} aria-hidden className="mt-[5px] shrink-0 text-ink-3 transition-transform duration-200 group-open:rotate-90" />
        <span className="text-[15px] leading-6 font-semibold text-ink">
          <span className="mr-1.5 font-normal text-ink-3">{n}.</span>
          {q}
        </span>
      </summary>
      <div className="px-4 pb-4 sm:px-5">
        <div className="grid gap-2 border-t border-line pt-3 text-[15px] leading-relaxed text-ink-2">
          <p>{children}</p>
          <p className="flex flex-wrap gap-x-3 gap-y-1 text-[14px]">
            <span className="text-ink-3">Подробно:</span>
            {to.map((c) => (
              <ChapterLink key={c} to={c} />
            ))}
          </p>
        </div>
      </div>
    </details>
  )
}
