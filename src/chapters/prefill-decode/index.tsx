import { useMemo } from 'react'
import { decodeStepBreakdown, decodeStepMs, kvBytesPerToken, moeExpertsTouched, prefillBreakdown, prefillMs, weightBytes } from '@/engine/perf'
import { Callout, Chapter, ChapterLink, Deeper, SimulatorLink, Takeaways, Term, fmt } from '@/ui'
import { DOC_TOKENS, SideBySideFigure } from './SideBySide'
import { RequestTimelineFigure } from './Timeline'
import { requestTiming } from './shared/timing'
import { DTYPE_LABEL, times, useHardwareState } from './shared/hardware'
import { usePromptTokens } from './shared/prompt'
import { WideFormula } from './shared/WideFormula'

/** Сколько запросов делят шаг decode в примере «под нагрузкой» и какой у них контекст. */
const LOAD_BATCH = 64
const LOAD_CONTEXT = 1500

export default function PrefillDecodeChapter() {
  const prompt = usePromptTokens()
  const hwState = useHardwareState()
  const hw = hwState.hw
  const N = prompt.tokens?.length ?? null

  const nums = useMemo(() => {
    const { model, gpu, perf } = hw
    const n = N ?? 43
    const long = requestTiming(hw, 8000, 400, 0)
    const reasoning = requestTiming(hw, n, 4000, 0)
    const pre = prefillBreakdown(model, gpu, n, perf)
    const preDoc = prefillBreakdown(model, gpu, n + DOC_TOKENS, perf)
    // те же проходы, что в фигуре: у MoE prefill и шаг decode читают разный объём весов
    const dec = decodeStepBreakdown(model, gpu, { ...perf, batch: 1, avgContext: n + 1 })
    // цена токена: один запрос на свободном GPU против шага decode, который делят LOAD_BATCH запросов
    const prefillPerTok2k = prefillMs(model, gpu, 2000, perf) / 2000
    const itl2k = requestTiming(hw, 2000, 400, 0).itlMs
    const loadedPerTok = decodeStepMs(model, gpu, { ...perf, batch: LOAD_BATCH, avgContext: LOAD_CONTEXT }) / LOAD_BATCH
    return {
      n,
      weights: weightBytes(model, hw.dtype),
      kvTok: kvBytesPerToken(model, 'bf16'),
      prefill: pre.ms,
      prefillBound: pre.bound,
      preWeights: pre.weightBytes,
      decode: dec.ms,
      decWeights: dec.weightBytes,
      // MoE: сколько разных экспертов на слой задели токены промпта
      touched: moeExpertsTouched(model, n),
      prefillDoc: preDoc.ms,
      prefillDocBound: preDoc.bound,
      decodeDoc: decodeStepMs(model, gpu, { ...perf, batch: 1, avgContext: n + DOC_TOKENS + 1 }),
      prefill8k: long.prefillMs,
      e2eLongPrompt: long.e2eMs,
      e2eReasoning: reasoning.e2eMs,
      kv32k: kvBytesPerToken(model, 'bf16') * 32768,
      costSingle: itl2k / prefillPerTok2k,
      costLoaded: loadedPerTok / prefillPerTok2k,
    }
  }, [hw, N])

  const nText = N === null ? '…' : fmt.tokens(N)
  const similar = Math.abs(nums.prefill / nums.decode - 1) < 0.15
  // MoE: сколько экспертов на слой и сколько из них выбирает каждый токен
  const moe = hw.model.experts ?? null

  return (
    <Chapter
      id="prefill-decode"
      lead={
        <>
          Модель пишет ответ в две фазы. Prefill прогоняет через неё весь промпт за один проход и выдаёт первый токен. Decode дописывает остальные по
          одному токену за проход, пока не встретит стоп. У фаз разная физика, поэтому у них разные метрики и разные проблемы в проде.
        </>
      }
    >
      <p>
        В главе про <ChapterLink to="sampling">логиты и сэмплинг</ChapterLink> модель выдавала один токен: промпт прошёл через все слои, на выходе
        получились логиты, из них выбрали токен. Но ответ состоит из сотен токенов. Как из одного шага получается целый ответ?
      </p>
      <p>
        Модель умеет предсказывать только следующий токен. Поэтому генерация идёт циклом: выбрали токен, дописали его в конец последовательности, снова
        запустили модель. Это <Term t="авторегрессия">авторегрессия</Term>, а каждый запуск модели называют проходом (<Term t="forward pass" />). Весь
        вопрос в том, сколько работы делает каждый проход. Ниже обе фазы рядом, на вашем запросе: сколько токенов проходит через модель за один проход и
        сколько данных GPU для этого читает из памяти?
      </p>

      <SideBySideFigure prompt={prompt} hw={hw} />

      <h3>Prefill: весь промпт за один проход</h3>
      <p>
        Ваш запрос после <ChapterLink to="template">шаблона чата</ChapterLink> занимает {nText}. Все они известны заранее, и модель обрабатывает их
        одним проходом, все позиции параллельно. Это <Term t="prefill" />. Параллелить можно благодаря causal mask из главы про{' '}
        <ChapterLink to="attention">attention</ChapterLink>: каждый токен видит только предыдущие, поэтому, если посчитать все позиции разом,
        получится то же, что и по одной, только намного быстрее. Для GPU это большие умножения матрицы на матрицу, а именно для них он и создан.
      </p>
      <p>
        По дороге в каждом слое для каждого токена считаются векторы K и V. Их сохраняют в <Term t="KV cache" />, чтобы больше никогда не
        пересчитывать. Логиты последней позиции промпта дают первый токен ответа. <SimulatorLink stage="prefill" />
      </p>

      <h3>Decode: по одному токену за проход</h3>
      <p>
        Дальше параллелить нечего: следующий токен зависит от только что выбранного. Каждый следующий проход берёт на вход один новый токен, считает
        для него Q, K и V, дописывает K и V в кэш, а attention нового токена смотрит на K и V всех предыдущих, которые уже лежат в кэше. Это{' '}
        <Term t="decode" />. Он повторяется, пока модель не выдаст стоп-токен (у Qwen3 это <code>{'<|im_end|>'}</code>) или не кончится лимит{' '}
        <code>max_tokens</code>. <SimulatorLink stage="decode" />
      </p>

      <p>
        {moe ? (
          <>
            Что заметить в фигуре. Prefill прочитал {fmt.bytes(nums.preWeights)} весов в {DTYPE_LABEL[hw.dtype]} один раз и обработал сразу {nText}:
            разные токены промпта ушли к разным экспертам и задели{' '}
            {nums.touched >= 0.95 * moe.total ? `почти всех ${moe.total}` : `около ${fmt.int(nums.touched)} из ${moe.total}`} экспертов каждого слоя
            (это <Term t="MoE (Mixture of Experts)">MoE</Term>). Каждый шаг decode читает {fmt.bytes(nums.decWeights)} ради одного токена: токен
            проходит через все слои, но в каждом слое только через {moe.active}{' '}
            {fmt.plural(moe.active, ['выбранного эксперта', 'выбранных эксперта', 'выбранных экспертов'])} из {moe.total}, а вся модель весит{' '}
            {fmt.bytes(nums.weights)}. Кроме того, шаг читает весь KV cache, который растёт на {fmt.bytes(nums.kvTok, { mode: 'binary' })} с каждым
            токеном.
          </>
        ) : (
          <>
            Что заметить в фигуре. Prefill прочитал веса модели ({fmt.bytes(nums.weights)} в {DTYPE_LABEL[hw.dtype]}) один раз и обработал сразу{' '}
            {nText}. Каждый шаг decode читает те же {fmt.bytes(nums.weights)} ради одного токена: токен обязан пройти через все слои, и каждый слой
            умножает его на свои матрицы целиком. Кроме того, шаг читает весь KV cache, который растёт на{' '}
            {fmt.bytes(nums.kvTok, { mode: 'binary' })} с каждым токеном.
          </>
        )}
      </p>
      <p>
        {nums.prefillBound === 'memory' ? (
          similar ? (
            <>
              На коротком промпте prefill длится примерно столько же, сколько один шаг decode: около {fmt.ms(nums.prefill)}. В обоих случаях время уходит
              на чтение весов, а на счёт {nText} вместо одного у GPU почти не уходит времени.
            </>
          ) : (
            <>
              На коротком промпте prefill, как и decode, упирается в чтение весов: {fmt.ms(nums.prefill)} против {fmt.ms(nums.decode)} на шаг decode.
              {hw.model.experts ? ' Prefill дольше, потому что разные токены промпта задевают разных экспертов MoE и весов читается больше.' : ''}
            </>
          )
        ) : (
          <>
            Ваш промпт уже достаточно длинный, чтобы prefill упирался в вычисления: {fmt.ms(nums.prefill)} против {fmt.ms(nums.decode)} на шаг decode.
            Но в пересчёте на токен prefill всё равно в {times((nums.decode * nums.n) / nums.prefill)} дешевле.
          </>
        )}
      </p>
      <p>
        Теперь переключите промпт на вариант с документом. Prefill{' '}
        {nums.prefillDoc / nums.prefill >= 1.5 ? <>стал в {times(nums.prefillDoc / nums.prefill)} дольше</> : 'почти не стал дольше'}, а токенов
        обработал в {times((nums.n + DOC_TOKENS) / nums.n)} больше.{' '}
        {nums.prefillDocBound === 'compute' ? (
          <>
            Теперь его держат вычисления (<Term t="compute-bound" />
            ), а шаг decode по-прежнему ждёт память (<Term t="memory-bound" />
            ): ради одного токена он читает {moe ? `только активные веса, ${fmt.bytes(nums.decWeights)}` : `все ${fmt.bytes(nums.weights)} весов`}, а
            длинный документ в кэше удлинил шаг лишь с {fmt.ms(nums.decode)} до {fmt.ms(nums.decodeDoc)}.
          </>
        ) : (
          <>
            У {hw.model.name} даже такой prefill ещё упирается в чтение весов, а шаг decode и подавно ждёт память (<Term t="memory-bound" />
            ).
          </>
        )}{' '}
        Почему так происходит и где граница между этими режимами, разбираем в главе <ChapterLink to="roofline" />.
      </p>

      <Callout kind="intuition" title="читать задание и писать ответ">
        Prefill похож на чтение задания: один взгляд на страницу, и все слова обработаны разом. Decode похож на письмо: ответ пишется по одному слову, и
        перед каждым словом вы заглядываете в свой конспект прочитанного. Конспект и есть KV cache. Без него перед каждым словом пришлось бы заново
        перечитывать всё задание и всё уже написанное.
      </Callout>

      <h3>Таймлайн одного запроса</h3>
      <p>
        Для пользователя две фазы выглядят как два ожидания. Сначала пауза до первого токена, <Term t="TTFT" /> (time to first token). Она
        складывается из очереди (запрос ждёт места в <Term t="batch" />, группе запросов, которые GPU считает одним проходом) и prefill. Потом ответ
        печатается с шагом <Term t="ITL" /> (inter-token latency), а это время одного шага decode. Полное время ответа из N токенов:
      </p>
      <WideFormula tex="\text{E2E} \approx \cprefill{\text{TTFT}} + \cdecode{\text{ITL}} \cdot (N - 1)" label="TTFT = очередь + prefill, ITL = шаг decode" />
      <p>Что сильнее растягивает запрос: длинный промпт или длинный ответ?</p>

      <RequestTimelineFigure state={hwState} promptCount={N} />

      <p>
        Prefill дешёвый в пересчёте на токен: токены промпта идут через модель пачкой. Decode дорогой: каждый токен ответа это отдельный проход по всем
        весам. Для {hw.model.name} на {hw.gpu.name} промпт на {fmt.int(8000)} токенов (длинный документ в RAG) обходится в {fmt.ms(nums.prefill8k)}{' '}
        prefill, а ответ из 400 токенов после него займёт ещё {fmt.ms(nums.e2eLongPrompt - nums.prefill8k)}. Если же модель рассуждает и пишет{' '}
        {fmt.int(4000)} токенов, ваш короткий запрос растягивается до {fmt.ms(nums.e2eReasoning)}.
      </p>
      <p>
        Это цена одного запроса на свободном GPU: для промпта в {fmt.int(2000)} токенов токен ответа дороже токена промпта в{' '}
        {times(nums.costSingle)}. Когда шаг decode делят {LOAD_BATCH} запроса с контекстом около {fmt.int(LOAD_CONTEXT)} токенов, токен ответа
        дороже уже примерно в {times(nums.costLoaded)}
        {nums.costLoaded < 15 ? ': поэтому в прайсах API выходные токены дороже входных лишь в несколько раз.' : '.'} Как batch делит чтение весов,
        разбираем в главе <ChapterLink to="roofline" />.
      </p>

      <Callout kind="production" title="две фазы, два профиля нагрузки">
        <p>
          TTFT в основном складывается из очереди и prefill, ITL почти целиком из шага decode. Смотреть на них нужно отдельно, подробно в главе{' '}
          <ChapterLink to="metrics" />.
        </p>
        <p>
          Нагрузки бывают разные. RAG и агенты с длинным контекстом нагружают prefill, рассуждающие модели и длинные ответы нагружают decode. На общих
          GPU фазы мешают друг другу: длинный prefill останавливает decode у всех соседей по batch, и их ITL подскакивает. Поэтому prefill режут на куски
          (<ChapterLink to="batching">chunked prefill</ChapterLink>), а на больших масштабах разносят фазы по разным пулам GPU (
          <ChapterLink to="disaggregation" />
          ). Повторяющееся начало промпта можно не считать вовсе: <ChapterLink to="prefix-caching" />.
        </p>
      </Callout>

      <Deeper title="где рождается первый токен и что ещё входит в TTFT">
        <p>
          Первый токен ответа получается из логитов последней позиции промпта, то есть в prefill. В дезагрегированном serving (prefill и decode на разных
          GPU) детали зависят от движка: в vLLM (и в Dynamo поверх vLLM) decode-воркер сам пересчитывает последний токен промпта и выбирает первый
          выходной токен, а токен prefill-воркера выбрасывается; в SGLang и TensorRT-LLM первый токен выбирает prefill-воркер, а decode продолжает с
          него.
        </p>
        <p>
          TTFT, который видит клиент, длиннее серверного: сверху добавляются сеть, TLS и gateway. Шаблон и токенизация в свежих версиях vLLM и во
          frontend Dynamo уже входят в серверный TTFT (таймер стартует до них), но точка старта серверного таймера зависит от движка и версии. Как это
          измерять честно, разбираем в главе <ChapterLink to="metrics" />.
        </p>
        <p>
          Короткий prefill упирается не в вычисления, а в чтение весов. На H100 в BF16 граница проходит около 300 токенов (точное число зависит от
          того, насколько хорошо ядра используют вычислители и память): промпт короче этого обрабатывается примерно за время одного шага decode. Поэтому
          движки добивают шаги decode кусками чужих промптов: вычислители всё равно простаивают.
        </p>
        <p>
          Prefill записывает в KV cache по {fmt.bytes(nums.kvTok, { mode: 'binary' })} на токен для {hw.model.name}: на промпт в {fmt.int(32768)}{' '}
          токенов это {fmt.bytes(nums.kv32k, { mode: 'binary' })} на один запрос, ещё до первого токена ответа. Сколько таких запросов влезает в GPU,
          считаем в главе <ChapterLink to="kv-cache" />.
        </p>
        <p>
          Decode останавливается по одному из условий: модель выдала стоп-токен (<code>{'<|im_end|>'}</code> у Qwen3, отсюда{' '}
          <code>finish_reason: "stop"</code>), достигнут <code>max_tokens</code> (<code>"length"</code>) или в тексте встретилась stop-строка (об этом
          в главе <ChapterLink to="detokenization" />
          ).
        </p>
      </Deeper>

      <Takeaways
        items={[
          <>Prefill: весь промпт за один проход, K и V каждого токена в KV cache и первый токен ответа. Он задаёт TTFT.</>,
          <>
            Decode: один токен за проход. Каждый шаг читает все веса модели (у MoE вместо всех экспертов только выбранные) и весь KV cache. Он задаёт
            ITL, то есть скорость печати.
          </>,
          <>
            Prefill длинного промпта упирается в вычисления, decode в чтение памяти. При одном запросе на GPU токен ответа в десятки и сотни раз дороже
            токена промпта; под нагрузкой batch делит чтение весов, и разница сжимается до нескольких раз.
          </>,
          <>TTFT ≈ очередь + prefill, E2E ≈ TTFT + ITL × (N − 1).</>,
        ]}
      />
    </Chapter>
  )
}
