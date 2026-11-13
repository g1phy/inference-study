import { memo, useCallback, useState } from 'react'
import { useReducedMotion } from 'motion/react'
import { ArrowDownIcon, PaperPlaneRightIcon } from '@phosphor-icons/react'
import { Block, Button, Callout, Chapter, ChapterLink, Deeper, fmt, Skeleton, Takeaways, Term } from '@/ui'
import { scrollToSimulator, navigateTo } from '@/state/nav'
import { AGG_NEIGHBOR_PREFILL_TOKENS, FLOW_TIMING, NEIGHBOR_MIN_COMPLETION_TOKENS, resolveSimGpu, resolveSimModel, type FlowRunEx } from '@/engine/flow'
import { isLongDoc, LONG_DOC_PROMPT_TOKENS } from '@/engine/scenarios'
import { HeroPipeline } from './HeroPipeline'
import { neighborSteps, tokensGen } from './model'
import { PIPELINE_ID, Simulator } from './Simulator'
import { useFlowRun } from './useFlowRun'

/**
 * Глава 0: hero «Что происходит после Enter» и симулятор «Путь запроса» (SPEC §6.0).
 * Договорённости с каркасом: блок симулятора id="simulator", поле сообщения id="prompt-input".
 */
export default function HeroChapter() {
  const flow = useFlowRun()
  const [sendSignal, setSendSignal] = useState(0)
  const reduce = useReducedMotion() ?? false

  const sendFromHero = useCallback(() => {
    // к схеме с плеером, а не к началу фигуры: на телефоне иначе на экране оказался бы только ввод
    const pipeline = document.getElementById(PIPELINE_ID)
    if (pipeline) pipeline.scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' })
    else scrollToSimulator()
    document.getElementById('simulator')?.focus({ preventScroll: true })
    setSendSignal((n) => n + 1)
  }, [reduce])

  return (
    <Chapter id="hero" header="none">
      <Block width="full" className="-mt-4 desk:-mt-8">
        <div className="grid grid-cols-[minmax(0,1fr)] items-center gap-x-12 gap-y-10 lg:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)]">
          <div className="grid grid-cols-[minmax(0,1fr)] content-center justify-items-start gap-6">
            <h1
              id="hero-title"
              data-chapter-heading
              tabIndex={-1}
              className="max-w-[15ch] text-[clamp(2.25rem,3.7vw,3.5rem)] leading-[1.05] font-[650] tracking-tight text-ink outline-none"
            >
              Что происходит после Enter
            </h1>
            <p className="max-w-[40ch] text-[1.1875rem] leading-[1.55] text-ink-2 desk:text-[1.25rem]">
              Весь путь запроса к LLM, от JSON до последнего токена: шаблон, токены, роутер, GPU, парсеры. На вашем собственном тексте.
            </p>
            <div className="flex flex-wrap gap-3">
              <Button variant="primary" size="lg" icon={PaperPlaneRightIcon} onClick={sendFromHero}>
                Отправить запрос
              </Button>
              <Button variant="secondary" size="lg" iconRight={ArrowDownIcon} onClick={() => navigateTo('request')}>
                Читать по порядку
              </Button>
            </div>
          </div>
          <HeroPipeline run={flow.run} status={flow.status} />
        </div>
      </Block>

      <Intro />

      <Simulator flow={flow} sendSignal={sendSignal} />

      <HowToRead />
      <RunSummary run={flow.run} />

      <AfterSimulator />
    </Chapter>
  )
}

/** Перед симулятором один абзац: что делать, на какие вопросы искать ответ, что посчитано. Статично, при наборе не перерисовывается. */
const Intro = memo(function Intro() {
  return (
    <Block width="full">
      {/* ширина full: абзац стоит на одной линии с hero и симулятором, а не прыгает в центральную колонку прозы */}
      <p className="max-w-[60ch]">
        Ниже тот же путь в подробностях. Выберите сценарий или напишите свой вопрос, нажмите «Отправить» и смотрите, какие стадии работают на CPU, какие
        на GPU и куда уходит время до первого токена; щелчок по стадии открывает её данные. Схема устроена как <Term t="NVIDIA Dynamo" /> с <Term t="воркер">воркерами</Term>{' '}
        <Term t="vLLM" />. Модель в браузере не запускается: ответ заготовлен; шаблон, токены и роутинг посчитаны по-настоящему, время prefill и decode
        оценено по параметрам модели и GPU, а сеть и CPU-стадии взяты условными константами.
      </p>
    </Block>
  )
})

/** «Как читать этот путь»: где работают стадии, туда и обратно, две фазы, деплой. Статично. */
const HowToRead = memo(function HowToRead() {
  return (
    <>
      <h3>Как читать этот путь</h3>
      <p>
        В этой схеме frontend, preprocessor, роутер и postprocessor работают в одном процессе на CPU (в Dynamo это dynamo.frontend, его рабочие части
        написаны на Rust), а prefill и decode выполняет движок на GPU-воркерах. Воркер получает номера токенов и возвращает номера токенов, текста он не видит. В
        одиночном vLLM или <Term t="SGLang" /> те же стадии живут внутри одного сервера: шаблон, токенизация, детокенизация и парсеры работают в его
        CPU-процессах, а роутер ставят отдельно, когда копий модели несколько (<ChapterLink to="engines" />, <ChapterLink to="dynamo" />).
      </p>
      <p>
        Запрос идёт туда и обратно. Туда: JSON с сообщениями склеивается по <Term t="chat template">шаблону чата</Term> в один текст,{' '}
        <Term t="tokenizer">токенизатор</Term> режет текст на токены, <Term t="KV-aware routing">роутер</Term> выбирает воркер, и номера токенов уходят на
        его GPU. Обратно: номера снова становятся текстом, текст раскладывается по полям ответа и уходит клиенту по кусочкам через <Term t="SSE" />.
      </p>
      <p>
        На GPU работа делится на две фазы. Сначала <Term t="prefill" /> прогоняет весь промпт за один проход и заполняет <Term t="KV cache" />: ключи и
        значения каждого токена, которые понадобятся дальше. Потом <Term t="decode" /> выдаёт ответ по одному токену за шаг и на каждом шаге читает этот
        кэш. Отсюда две главные метрики задержки: <Term t="TTFT" /> (сколько ждать первый токен) и <Term t="ITL" /> (как часто приходят следующие).
      </p>
      <p>
        Переключатель «Деплой» показывает два способа разместить эти фазы. В агрегированном режиме prefill и decode идут на одном воркере. В
        дезагрегированном у каждой фазы свой пул GPU: длинный prefill чужого запроса больше не тормозит ваш decode, и каждый пул масштабируется под свою
        метрику (prefill под TTFT, decode под ITL). В симуляторе видно и то и другое. Цена: KV cache промпта едет по сети через <Term t="NIXL" />, а
        decode-воркер пересчитывает последний токен промпта, поэтому TTFT в дезагрегированном режиме больше. Выигрыш виден под нагрузкой: пока идёт ответ
        (в симуляторе в ответах от {fmt.int(NEIGHBOR_MIN_COMPLETION_TOKENS)} токенов), на воркер приходит длинный промпт соседа (условно{' '}
        {fmt.int(AGG_NEIGHBOR_PREFILL_TOKENS)} токенов), и в агрегированном режиме <Term t="Chunked prefill">chunked prefill</Term> ставит его в шаг decode
        вашего запроса; этот шаг длится в десятки раз дольше остальных. В дезагрегированном режиме его считает prefill-воркер, и шаги decode ровные (
        <ChapterLink to="disaggregation" />).
      </p>
    </>
  )
})

/** После симулятора: интуиция, карта сайта, подводные камни, что посчитано, «Запомнить». Статично. */
const AfterSimulator = memo(function AfterSimulator() {
  return (
    <>
      <Callout kind="intuition" title="чтение и письмо">
        Prefill похож на чтение письма: все слова промпта уже есть, и GPU обрабатывает их одной большой порцией. Decode похож на ответ от руки: следующее
        слово не напишешь, пока не написано предыдущее. Но за каждый проход, хоть на сотни токенов, хоть на один, GPU читает из памяти все веса модели (у
        Qwen3-8B в BF16 это около 16 ГБ). Prefill делит это чтение на все токены промпта, и на длинном промпте время задают вычисления: prefill{' '}
        <Term t="Compute-bound">compute-bound</Term>. Decode тратит то же чтение на один токен каждого запроса, вычислители простаивают, и шаг упирается в
        полосу памяти: decode <Term t="Memory-bound">memory-bound</Term> (<ChapterLink to="roofline" />).
      </Callout>

      <h3>Карта сайта по стадиям</h3>
      <p>
        Главы сгруппированы по стадиям этого пути. Порядок почти совпадает с порядком запроса, кроме роутера: его глава стоит после KV cache, потому что
        роутер выбирает воркер по содержимому кэша.
      </p>
      <ol>
        <li>
          Вход, то есть frontend и preprocessor: формат запроса, шаблон чата, токенизация (<ChapterLink to="request" />, <ChapterLink to="template" />,{' '}
          <ChapterLink to="tokenization" />).
        </li>
        <li>
          Внутри модели: эмбеддинги, слои, attention с векторами Q, K и V, логиты и сэмплинг (<ChapterLink to="embeddings" />,{' '}
          <ChapterLink to="transformer" />, <ChapterLink to="attention" />, <ChapterLink to="sampling" />).
        </li>
        <li>
          Две фазы: почему prefill упирается в вычисления, а decode в память, и как это видно в TTFT и ITL (<ChapterLink to="prefill-decode" />,{' '}
          <ChapterLink to="roofline" />, <ChapterLink to="metrics" />).
        </li>
        <li>
          Память: что лежит в KV cache, блоки, prefix cache и уровни L1, L2, L3, включая кэш ноды как <Term t="DaemonSet" /> в Kubernetes (
          <ChapterLink to="kv-cache" />, <ChapterLink to="paged-attention" />, <ChapterLink to="prefix-caching" />, <ChapterLink to="kv-tiers" />).
        </li>
        <li>
          Много запросов и много GPU: continuous batching, агрегированный и дезагрегированный режим с передачей KV через NIXL, параллелизм (
          <ChapterLink to="batching" />, <ChapterLink to="disaggregation" />, <ChapterLink to="parallelism" />).
        </li>
        <li>
          Кто это выполняет: движки vLLM, SGLang и TensorRT-LLM считают модель на GPU, а NVIDIA Dynamo стоит над ними и выбирает воркер KV-aware роутером (
          <ChapterLink to="engines" />, <ChapterLink to="dynamo" />, <ChapterLink to="routing" />).
        </li>
        <li>
          Обратный путь, то есть postprocessor и стрим: детокенизация и парсеры рассуждений и вызовов инструментов (<ChapterLink to="detokenization" />,{' '}
          <ChapterLink to="parsers" />). <Term t="Structured output" /> стоит рядом, но работает раньше: маской в сэмплере на каждом шаге decode (
          <ChapterLink to="structured-output" />).
        </li>
        <li>
          Ускорения и прод: speculative decoding, квантизация, автоскейлинг и мониторинг; в конце весь путь на одной странице и глоссарий (
          <ChapterLink to="speculative" />, <ChapterLink to="quantization" />, <ChapterLink to="production" />, <ChapterLink to="cheatsheet" />,{' '}
          <ChapterLink to="glossary" />).
        </li>
      </ol>
      <p>
        Как читать: идите по главам по порядку или открывайте их из симулятора ссылками «Подробнее». Термины с пунктиром показывают короткое определение.
        Блоки «Глубже» для тех, кому нужны флаги и формулы: без них глава понятна. Подпись «Попробуйте» под фигурой говорит, что покрутить. Если времени
        мало, короткий маршрут: <ChapterLink to="tokenization" />, <ChapterLink to="attention" />, <ChapterLink to="prefill-decode" />,{' '}
        <ChapterLink to="roofline" />, <ChapterLink to="kv-cache" />, <ChapterLink to="disaggregation" />, <ChapterLink to="dynamo" />,{' '}
        <ChapterLink to="routing" />, <ChapterLink to="structured-output" />, <ChapterLink to="cheatsheet" />.
      </p>

      <Callout kind="production" title="где на этом пути болит">
        <ul>
          <li>
            Очередь перед prefill. В симуляторе воркер сразу берёт запрос. Под нагрузкой запрос ждёт места в <Term t="батч">batch</Term>, и TTFT растёт,
            хотя модель и железо те же (<ChapterLink to="batching" />, <ChapterLink to="production" />).
          </li>
          <li>
            Память под KV cache. Каждый активный запрос держит свой кэш, поэтому длинные контексты и много одновременных пользователей упираются в память
            GPU (<ChapterLink to="kv-cache" />).
          </li>
          <li>
            Промахи <Term t="prefix caching">prefix cache</Term>. Без <Term t="KV-aware routing">KV-aware роутинга</Term> одинаковые системные промпты
            считаются заново на разных воркерах (<ChapterLink to="routing" />).
          </li>
          <li>
            Парсеры. Если выбран не тот парсер или маркеры вырезаны до него (skip_special_tokens), поля tool_calls и reasoning_content приходят пустыми,
            хотя модель ответила правильно (<ChapterLink to="parsers" />).
          </li>
        </ul>
      </Callout>

      <Callout kind="pitfall" title="чего здесь нет">
        Симулятор показывает один запрос в кластере с условной фоновой нагрузкой. Очередь prefill у воркеров учитывает только роутер в стоимости, на TTFT
        вашего запроса она не влияет; вытеснения нет, соседи по batch не меняются. Из чужих запросов смоделирован один: длинный промпт соседа, который в
        агрегированном режиме попадает в шаг decode вашего запроса, если ответ не короче {fmt.int(NEIGHBOR_MIN_COMPLETION_TOKENS)} токенов. Сеть и CPU-стадии
        взяты условными константами (сеть {fmt.ms(FLOW_TIMING.networkMs)} в одну сторону), и на коротком промпте одна сеть занимает заметную долю TTFT (у
        Qwen3-8B на H100 в агрегированном режиме больше половины). В реальном сервисе к этому добавятся TLS, балансировщик и очередь.
      </Callout>

      <Deeper title="что посчитано по-настоящему, а что заготовлено">
        <p>По-настоящему, теми же модулями, что и в главах:</p>
        <ul>
          <li>шаблон Qwen3 и токены словаря Qwen3 ({fmt.int(151669)} токенов) для вашего текста, со смещениями и ID;</li>
          <li>
            блоки KV по 16 токенов и их хэши цепочкой; неполный хвост не хэшируется, как в vLLM и Dynamo (значения хэшей свои, у Dynamo XXH3);
          </li>
          <li>
            решение KV-aware роутера по формуле Dynamo 1.2 и новее: стоимость prefill с кредитом за совпавшие блоки плюс нагрузка decode. По умолчанию
            роутер Dynamo раздаёт запросы по кругу (round-robin), KV-режим включают флагом <code>--router-mode kv</code>;
          </li>
          <li>
            в дезагрегированном режиме PrefillRouter сначала выбирает prefill-воркер (запрос-клон с max_tokens = 1), потом decode-воркер по нагрузке, а
            KV едет через NIXL; время передачи посчитано по объёму KV и полосе InfiniBand;
          </li>
          <li>
            время prefill и каждого шага decode по параметрам модели и GPU, с условными КПД (50 % пиковых FLOPS, 70 % полосы памяти) и 0,3 мс накладных на
            шаг; маски structured output, детокенизация, парсеры и SSE-чанки.
          </li>
        </ul>
        <p>
          Заготовлено: текст ответа (пресеты сценариев и заготовки по темам своего текста), логиты кандидатов (настоящие токены словаря, условные числа),
          фоновая нагрузка воркеров и константы сети и CPU. Выбранный токен всегда заскриптованный: temperature меняет вероятности, но не ответ. Ответ
          одинаков для всех моделей и GPU: от них зависят только время, размер KV cache и число GPU на воркер.
        </p>
      </Deeper>

      <Takeaways
        items={[
          <>
            Запрос проходит путь туда (JSON → текст по шаблону → токены → воркер) и обратно (токены → текст → поля ответа → SSE). Шаблон, токены, роутер и
            парсеры работают на CPU, на GPU только prefill и decode.
          </>,
          <>
            Prefill считает весь промпт за один проход и на длинных промптах упирается в вычисления (compute-bound); decode выдаёт по токену за шаг и
            каждый раз читает из памяти веса (у MoE только задетых экспертов) и KV cache, поэтому упирается в память (memory-bound).
          </>,
          <>
            TTFT складывается из сети, подготовки, очереди и prefill (в дезагрегированном режиме ещё передача KV и лишний шаг decode); остальное время
            ответа занимает decode, по ITL на токен.
          </>,
          <>
            KV-aware роутер выбирает воркер по совпадению начала промпта в KV cache и по нагрузке: тёплый кэш экономит вычисления prefill, а на длинных
            промптах и TTFT. В Dynamo этот режим включают флагом --router-mode kv, по умолчанию запросы раздаются по кругу (round-robin).
          </>,
        ]}
      />
    </>
  )
})

/** Абзацы с числами из прогона читателя (обновляются вместе с симулятором). */
function RunSummary({ run }: { run: FlowRunEx | null }) {
  if (!run) return <Skeleton lines={2} height={14} label="Считаем ваш прогон" />
  const m = run.metrics
  const p = run.prefill
  const modelSpec = resolveSimModel(run.modelId ?? '').model
  const model = modelSpec.name
  const moe = !!modelSpec.experts
  const gpu = resolveSimGpu(run.gpuId ?? '').gpu.name
  const N = m.completionTokens
  const genMs = m.e2eMs - m.ttftMs
  const share = m.e2eMs > 0 ? genMs / m.e2eMs : 0
  // (X-sim-r2-4) шаг с prefill соседа тянет среднее вверх: обычный шаг рядом, а «короткий или длинный ответ» решаем без него
  const nb = neighborSteps(run)
  const extra = run.decode.reduce((s, d) => s + (d.neighborPrefill ? d.durationMs - d.neighborPrefill.plainMs : 0), 0)
  const plainShare = m.e2eMs - extra > 0 ? (genMs - extra) / (m.e2eMs - extra) : 0
  // обычный шаг decode (без prefill соседа)
  const step = (run.decode.find((d) => d.breakdown && !d.neighborPrefill) ?? run.decode.find((d) => d.breakdown))?.durationMs ?? m.itlMs
  const r = step > 0 ? p.ms / step : 0
  // порог для всего промпта без кэша; вкладка «Длинный документ» сменит упор, только если её промпт длиннее порога
  const threshold = run.prefillThreshold ?? 0
  const onLongDoc = isLongDoc(run.settings)
  const docFlips = threshold > 0 && threshold < LONG_DOC_PROMPT_TOKENS
  const routes = run.timeline.filter((t) => t.stage === 'route' && t.step === undefined)
  const r2 = routes.length > 1 ? routes[1].endMs - routes[1].startMs : 0
  return (
    <>
      <p>
        В вашем прогоне ({model} на {gpu}) prefill промпта в {fmt.tokens(p.totalTokens)} занял {fmt.ms(p.ms)}, первый чанк пришёл через {fmt.ms(m.ttftMs)}
        , а остальные {fmt.tokens(Math.max(0, N - 1))} шли ещё {fmt.ms(genMs)},{' '}
        {nb ? (
          <>
            в среднем по {fmt.ms(m.itlMs)} на токен (обычный шаг {fmt.ms(nb.plainMs)}).
          </>
        ) : (
          <>по {fmt.ms(m.itlMs)} на токен.</>
        )}{' '}
        {plainShare > 0.85 ? (
          <>Это {fmt.pct(share)} всего времени ответа: на длинных ответах почти всё время уходит на decode.</>
        ) : (
          <>Ответ короткий, поэтому путь до первого токена занимает заметную долю времени: {fmt.pct(1 - share)}.</>
        )}
        {run.transfer && (
          <>
            {' '}
            К TTFT добавились второй шаг роутера ({fmt.ms(r2)}), передача KV ({fmt.ms(run.transfer.ms)}) и шаг decode-воркера, который пересчитывает
            последний токен промпта ({fmt.ms(run.decode[0].durationMs)}).
          </>
        )}
        {run.neighbor && (
          <>
            {' '}
            {run.neighbor.applied ? (
              <>
                Шаг decode токена {fmt.int(run.neighbor.atStep + 1)} занял {fmt.ms(run.neighbor.stepMs)} вместо {fmt.ms(run.neighbor.plainMs)}: в него попал
                prefill соседа, и пауза между токенами там подскочила. Переключите «Деплой» на дезагрегированный: шаги decode станут ровными.
              </>
            ) : (
              <>
                Зато шаги decode ровные: длинный prefill соседа посчитал prefill-воркер, а в агрегированном режиме он растянул бы один шаг до{' '}
                {fmt.ms(run.neighbor.stepMs)}.
              </>
            )}
          </>
        )}
      </p>
      <p>
        {p.bound === 'compute' ? (
          <>
            Prefill {tokensGen(p.computedTokens)} занял {fmt.ms(p.ms)}, как {fmt.count(Math.max(1, Math.round(r)), ['шаг', 'шага', 'шагов'])} decode: на
            таком промпте время задают вычисления, а шаг decode по-прежнему упирается в чтение весов и KV cache.
          </>
        ) : p.cachedTokens > 0 ? (
          moe ? (
            <>
              Prefill посчитал только {fmt.int(p.computedTokens)} из {tokensGen(p.totalTokens)}, остальные взяты из кэша, но всё равно занял {fmt.ms(p.ms)}, в{' '}
              {fmt.num(r, 1)} раза дольше шага decode ({fmt.ms(step)}): у MoE даже такая порция задевает большинство экспертов, и prefill читает почти все
              веса.
            </>
          ) : (
            <>
              Prefill посчитал только {fmt.int(p.computedTokens)} из {tokensGen(p.totalTokens)}, остальные взяты из кэша, и время задало чтение весов:
              prefill занял {fmt.ms(p.ms)}, почти как шаг decode ({fmt.ms(step)}).
            </>
          )
        ) : moe ? (
          <>
            Заметьте: prefill промпта ({fmt.tokens(p.totalTokens)}) занял {fmt.ms(p.ms)}, в {fmt.num(r, 1)} раза дольше шага decode ({fmt.ms(step)}). У
            MoE токены промпта вместе задевают почти всех экспертов, и prefill читает почти все веса, а шаг decode читает веса только тех экспертов, которых
            задел его batch. Считает GPU в обоих случаях мало, так что время задаёт чтение памяти. В вычисления prefill этой модели упрётся только на
            промптах длиннее примерно {fmt.tokens(threshold)}.
          </>
        ) : (
          <>
            Заметьте: prefill промпта ({fmt.tokens(p.totalTokens)}) занял почти столько же, сколько один шаг decode ({fmt.ms(step)}). Оба раза GPU в
            основном читал веса, а считал мало. В вычисления prefill этой модели упирается на промптах длиннее примерно {fmt.tokens(threshold)}
            {!onLongDoc && docFlips ? <>: откройте вкладку «Длинный документ» и сравните.</> : '.'}
          </>
        )}{' '}
        В этом прогоне сеть туда и обратно заняла {fmt.ms(2 * FLOW_TIMING.networkMs)} из {fmt.ms(m.ttftMs)} TTFT (условно), а prefill {fmt.ms(p.ms)}.
      </p>
    </>
  )
}
