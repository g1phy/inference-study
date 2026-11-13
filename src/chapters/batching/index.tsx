import { useMemo } from 'react'
import { Callout, Chapter, ChapterLink, CodeBlock, Deeper, SimulatorLink, Takeaways, Term, fmt } from '@/ui'
import { decodeStepMs, moeExpertsTouched, prefillMs, stepWeightBytes } from '@/engine/perf'
import { BudgetFigure } from './BudgetFigure'
import { ChunkedFigure } from './ChunkedFigure'
import { ECON_CONTEXT, EconomicsFigure, useEconomics } from './EconomicsFigure'
import { GanttFigure } from './GanttFigure'
import { SpecTable } from './chart'
import { configLabel } from './config'

const LONG_PROMPT = 16384

export default function BatchingChapter() {
  const { cfg, rows } = useEconomics()
  const nums = useMemo(() => {
    const perf = { gpus: cfg.tp, dtype: cfg.dtype }
    const weights = stepWeightBytes(cfg.model, cfg.dtype, 1)
    const longPrefill = prefillMs(cfg.model, cfg.gpu, LONG_PROMPT, perf)
    const decode16 = decodeStepMs(cfg.model, cfg.gpu, { ...perf, batch: 16, avgContext: 1200 })
    return { weights, longPrefill, decode16 }
  }, [cfg])
  const r1 = rows[0]
  const r32 = rows.find((r) => r.batch === 32) ?? rows[rows.length - 1]
  const rLast = rows[rows.length - 1]
  const growth = Math.round(rLast.totalTokS / r1.totalTokS)
  const weightsLast = stepWeightBytes(cfg.model, cfg.dtype, rLast.batch)
  // MoE: чем больше запросов в шаге, тем больше разных экспертов читается (как stepWeightBytes)
  const experts = cfg.model.experts && cfg.model.expertParamsB ? cfg.model.experts : null
  const weights32 = stepWeightBytes(cfg.model, cfg.dtype, r32.batch)

  return (
    <Chapter
      id="batching"
      lead={
        <>
          Один GPU обслуживает сразу десятки запросов: за один шаг он делает по токену для каждого, а веса читает один раз на всех. Continuous
          batching пересобирает эту группу на каждом шаге, а chunked prefill режет длинный промпт на куски, чтобы он не останавливал остальных.
        </>
      }
    >
      <h3>Почему один запрос на GPU расточителен</h3>
      {experts ? (
        <p>
          В главе <ChapterLink to="roofline" /> мы видели, что шаг <Term t="decode" /> почти целиком уходит на чтение весов из памяти GPU. Для{' '}
          {cfg.model.name} при одном запросе это {fmt.bytes(nums.weights)} на шаг
          {cfg.tp > 1 ? `: модель лежит на ${cfg.tp} GPU, и каждый читает свои ${fmt.bytes(nums.weights / cfg.tp)}` : ''}. Это{' '}
          <Term t="MoE">MoE-модель</Term> (глава <ChapterLink to="transformer" />): в эти байты входят общие веса и только те {experts.active} из{' '}
          {experts.total} экспертов каждого слоя, к которым пошёл токен. Если в шаге один запрос, все эти байты читаются ради одного токена. Если
          запросов {r32.batch}, каждый эксперт по-прежнему читается один раз на всех, кто к нему пришёл, но разные запросы зовут разных экспертов:
          задето в среднем {fmt.int(moeExpertsTouched(cfg.model, r32.batch))} из {experts.total} на слой (если выбор равномерный), и за шаг читается
          уже {fmt.bytes(weights32)} вместо {fmt.bytes(nums.weights)}. Токенов при этом выходит {r32.batch}.
        </p>
      ) : (
        <p>
          В главе <ChapterLink to="roofline" /> мы видели, что шаг <Term t="decode" /> почти целиком уходит на чтение весов из памяти GPU. Для{' '}
          {cfg.model.name} это {fmt.bytes(nums.weights)} на каждом шаге
          {cfg.tp > 1 ? `: модель лежит на ${cfg.tp} GPU, и каждый читает свои ${fmt.bytes(nums.weights / cfg.tp)}` : ''}. Если в шаге один запрос, все
          эти байты читаются ради одного токена. Если запросов {r32.batch}, веса всё так же читаются один раз, а токенов выходит {r32.batch}.
        </p>
      )}
      <p>
        Группу запросов, которые проходят через модель одним проходом, называют <Term t="batching">batch</Term>. Посмотрите, сколько стоит шаг при
        разном размере batch.
      </p>
      <EconomicsFigure rows={rows} label={configLabel(cfg)} moe={experts !== null} />
      {experts ? (
        <p>
          При batch {rLast.batch} шаг длиннее в {fmt.num(rLast.stepMs / r1.stepMs, 1)} раза, а токенов в секунду в {fmt.int(growth)}{' '}
          {fmt.plural(growth, ['раз', 'раза', 'раз'])} больше, и цена токена падает почти во столько же раз. Платит за это каждый пользователь: его
          скорость падает с {fmt.rate(r1.perUserTokS)} до {fmt.rate(rLast.perUserTokS)}. Почему шаг растёт? У {cfg.model.name} главная причина в
          экспертах: при {rLast.batch} запросах задеты почти все, и за шаг читается {fmt.bytes(weightsLast)} весов вместо {fmt.bytes(nums.weights)} при
          одном запросе. <Term t="KV cache" /> у каждого запроса свой и тоже читается на каждом шаге: при {rLast.batch} запросах по{' '}
          {fmt.tokens(ECON_CONTEXT)} это ещё {fmt.bytes(rLast.kvBytes)}. Когда задеты все эксперты, объём весов расти перестаёт, и дальше у роста те
          же пределы, что у плотной модели: чтение KV, которое растёт с batch и длиной контекста; память под KV всех запросов (глава{' '}
          <ChapterLink to="kv-cache" />
          ); и при очень больших batch вычисления.
        </p>
      ) : (
        <p>
          При batch {rLast.batch} шаг длиннее всего в {fmt.num(rLast.stepMs / r1.stepMs, 1)} раза, а токенов в секунду в {fmt.int(growth)}{' '}
          {fmt.plural(growth, ['раз', 'раза', 'раз'])} больше, и цена токена падает почти во столько же раз. Платит за это каждый пользователь: его
          скорость падает с {fmt.rate(r1.perUserTokS)} до {fmt.rate(rLast.perUserTokS)}. Почему шаг вообще растёт? Веса читаются один раз на всех, а{' '}
          <Term t="KV cache" /> у каждого запроса свой, и его тоже приходится читать на каждом шаге: при {rLast.batch} запросах по{' '}
          {fmt.tokens(ECON_CONTEXT)} это {fmt.bytes(rLast.kvBytes)} KV против {fmt.bytes(weightsLast)} весов. Поэтому у роста три предела: чтение KV,
          которое растёт с batch и длиной контекста; память под KV всех запросов (глава <ChapterLink to="kv-cache" />); и при очень больших batch
          вычисления.
        </p>
      )}
      <p>Остаётся вопрос, как собирать такие группы, если запросы приходят в разное время, а ответы у них разной длины.</p>

      <h3>Static batching: ждать, пока соберутся все</h3>
      <p>
        Самый простой способ: дождаться B запросов, сделать для них общий <Term t="prefill" /> и генерировать, пока не закончит самый длинный ответ.
        Так работает <Term t="Static batching">static batching</Term>, и у него три потери.
      </p>
      <ul>
        <li>Первый пришедший ждёт, пока наберётся группа. <Term t="Dynamic batching">Dynamic batching</Term> ограничивает это ожидание таймером, но не убирает.</li>
        <li>
          Кто ответил коротко, держит свой слот до конца batch: GPU крутит шаги, в которых часть мест пустая. Новые запросы тем временем стоят в
          очереди до следующего batch.
        </li>
        <li>Промпты в batch выравниваются по самому длинному (padding): короткие считаются вместе с пустыми токенами, и общий prefill дольше.</li>
      </ul>
      <p>
        Для LLM это особенно больно, потому что длина ответа заранее неизвестна и сильно различается: один запрос получит «Да», другой эссе на
        тысячу токенов.
      </p>

      <h3>Continuous batching: пересобирать batch на каждом шаге</h3>
      <p>
        <Term t="Continuous batching" /> убирает все три потери: состав batch решается заново на каждом шаге модели, а промпты не выравниваются. Закончил запрос, на следующем шаге его
        место отдают новому. Пришёл запрос, он попадает в ближайший шаг, а не в следующий batch. Идея пришла из системы Orca (2022), где её назвали
        iteration-level scheduling; в TensorRT-LLM то же самое называют in-flight batching. Сегодня так работают vLLM, SGLang и TensorRT-LLM.
      </p>
      <p>Что это даёт на практике? Ниже один и тот же поток из восьми запросов обслуживают оба способа. Ответы нарочно разной длины, как в жизни.</p>
      <GanttFigure />
      <p>
        В static видно, как первые запросы ждут четвёртого, prefill растягивается на дополнение до самого длинного промпта, а короткие ответы
        оставляют за собой штрихованные пустые слоты. В continuous каждый
        запрос начинает работу почти сразу после прихода, поэтому <Term t="TTFT" /> падает в разы, а весь поток заканчивается раньше: GPU не тратит
        шаги на пустые места.
      </p>
      <Callout kind="intuition">
        Static batching похож на маршрутку, которая едет, только когда заполнена, и стоит на конечной, пока не выйдет последний пассажир. Continuous
        batching похож на эскалатор: встал, когда пришёл, сошёл, когда доехал, и место сразу занимает следующий.
      </Callout>
      <SimulatorLink stage="decode" />

      <h3>Что происходит внутри одного шага</h3>
      <p>
        Кто попадёт в каждый шаг, решает <Term t="Engine scheduler">планировщик</Term> движка. В vLLM V1 у него нет отдельных режимов prefill и decode. У каждого
        запроса есть число уже посчитанных токенов и число токенов, которые нужно посчитать, и планировщик просто даёт запросу догнать самого себя.
        Для генерирующего запроса это один токен, для нового весь промпт.
      </p>
      <p>
        Ограничений два. <Term t="Token budget">Бюджет токенов</Term> <code>max_num_batched_tokens</code> задаёт, сколько токенов поместится в один
        шаг. Предел <code>max_num_seqs</code> задаёт, сколько запросов может быть в работе одновременно. Сначала места получают запросы, которые уже
        в работе, потом новые из очереди по порядку прихода, пока хватает бюджета. Промпт, который не влез в остаток, входит частично и досчитывается
        на следующих шагах; первый токен такой запрос выдаст, только когда посчитает весь промпт.
      </p>
      <BudgetFigure />
      <p>
        Один forward-проход здесь смешанный: при бюджете 10 на шаге 2 вместе едут decode-токен A, хвост промпта B и весь промпт C. Это нормально и
        даже выгодно: в одном шаге они нагружают разное, decode полосу памяти, prefill вычислители.
      </p>

      <h3>Chunked prefill: длинный промпт не должен останавливать всех</h3>
      <p>
        Теперь представьте, что 16 пользователей читают свои ответы, и приходит запрос с документом на {fmt.tokens(LONG_PROMPT)}. Его prefill на{' '}
        {cfg.model.name} занимает около {fmt.ms(nums.longPrefill)}. Если положить промпт в один шаг целиком, этот шаг будет длиться столько же, и все 16
        пользователей увидят паузу: следующий токен придёт через {fmt.ms(nums.longPrefill)} вместо обычных {fmt.ms(nums.decode16)}. Это всплеск{' '}
        <Term t="ITL" />, в стриминге он выглядит как подвисание текста.
      </p>
      <p>
        <Term t="Chunked prefill" /> режет такой промпт на куски не больше бюджета шага. Каждый шаг тогда несёт decode-токены всех пользователей и один
        кусок промпта. Шаги становятся длиннее обычных, зато вместо одной паузы в {fmt.ms(nums.longPrefill)} у всех получается несколько коротких.
      </p>
      <ChunkedFigure />
      <p>
        Заметьте, что ступеньки оранжевой линии поднимаются: каждый следующий кусок считает attention ко всем уже посчитанным токенам этого
        промпта, поэтому поздние куски дороже ранних.
      </p>
      <p>
        Бюджет шага работает как ручка компромисса. Маленький бюджет держит ITL ровным, но длинному запросу нужно больше шагов, и его первый токен
        приходит позже. Большой бюджет даёт быстрый TTFT новым запросам и больше throughput, но каждый кусок промпта сильнее тормозит тех, кто уже
        генерирует. Документация vLLM советует около {fmt.int(2048)} токенов ради ITL и больше {fmt.int(8192)} ради максимального throughput.
      </p>
      <Callout kind="production" title="флаги и метрики">
        <p>
          В vLLM V1 chunked prefill включён по умолчанию. Бюджет по умолчанию зависит от GPU: у <code>vllm serve</code> на H100 и H200 это{' '}
          {fmt.tokens(8192)}, на B200 {fmt.int(16384)}, на A100 и L40S {fmt.int(2048)}. В SGLang аналог называется <code>--chunked-prefill-size</code> (
          {fmt.int(8192)} для GPU на 80 ГБ).
        </p>
        <p>
          В метриках смотрите <code>vllm:num_requests_running</code> и <code>vllm:num_requests_waiting</code> (сколько запросов в batch и в
          очереди), хвост <code>vllm:inter_token_latency_seconds</code> (всплески ITL видны в p99, а не в среднем) и{' '}
          <code>vllm:num_preemptions</code>.
        </p>
      </Callout>
      <Callout kind="pitfall" title="chunked prefill не равно дезагрегации">
        Обе техники борются с тем, что prefill мешает decode, но разными средствами. Chunked prefill оставляет обе фазы на одних GPU и смешивает их
        аккуратно, по кусочку. Дезагрегация разносит фазы на разные GPU и передаёт KV cache по сети. Об этом следующая глава,{' '}
        <ChapterLink to="disaggregation" />.
      </Callout>

      <Deeper title="бюджет, предел запросов и смешанные шаги в трёх движках">
        <p>Названия и умолчания отличаются, а смысл один: сколько токенов и сколько запросов помещается в шаг.</p>
        <SpecTable
          caption="Ограничения планировщика в vLLM, SGLang и TensorRT-LLM"
          head={['Движок', 'Бюджет токенов шага', 'Запросов в работе', 'Chunked prefill', 'Смешанный шаг']}
          rows={[
            [
              'vLLM V1',
              <>
                <code>--max-num-batched-tokens</code>: {fmt.int(8192)} (H100, H200), {fmt.int(16384)} (B200), {fmt.int(2048)} (A100, L40S)
              </>,
              <>
                <code>--max-num-seqs</code>: {fmt.int(1024)} или 256
              </>,
              'всегда, когда возможно',
              'всегда: сначала decode, потом prefill в остаток',
            ],
            [
              'SGLang',
              <>
                <code>--chunked-prefill-size</code>: {fmt.int(8192)} на 80 ГБ, {fmt.int(16384)} на B200; <code>--max-prefill-tokens</code> {fmt.int(16384)}
              </>,
              <code key="s">--max-running-requests</code>,
              'включён (−1 выключает)',
              <>
                по умолчанию шаг либо prefill, либо decode; смешивание <code>--enable-mixed-chunk</code>
              </>,
            ],
            [
              'TensorRT-LLM',
              <code key="t">max_num_tokens</code>,
              <code key="t2">max_batch_size</code>,
              <>
                <code>enable_chunked_prefill</code>, по умолчанию выключен; куски кратны блоку KV
              </>,
              'in-flight batching: generation-токены первыми, затем context',
            ],
          ]}
        />
        <p>
          Числа vLLM даны для <code>vllm serve</code> v0.32; у Python API <code>LLM(...)</code> бюджет больше ({fmt.int(16384)} на H100). Если выключить chunked
          prefill, бюджет должен быть не меньше <code>max_model_len</code>, иначе длинный промпт не войдёт ни в один шаг.
        </p>
      </Deeper>

      <Deeper title="preemption: что делает vLLM, когда кончился KV cache">
        <p>
          vLLM пускает запросы в работу оптимистично: блоки KV выдаются по мере генерации, а не сразу под максимальную длину (как именно, в главе{' '}
          <ChapterLink to="paged-attention" />
          ). Если на очередной токен места нет, планировщик вытесняет последний допущенный запрос: освобождает все его блоки и ставит в начало очереди.
          Это <Term t="Preemption">preemption</Term>.
        </p>
        <p>
          Вернувшись, запрос пересчитывает промпт и уже сгенерированные токены одним prefill. В V1 это единственный режим (recompute): выгрузку в
          память CPU (swap) убрали, параметр <code>swap_space</code> удалён в v0.18. На шаге, где кого-то вытеснили, новые запросы из очереди не
          допускаются. Частые вытеснения лечат большей долей памяти под KV, меньшими <code>max_num_seqs</code> и{' '}
          <code>max_num_batched_tokens</code>, увеличением TP или PP; счётчик <code>vllm:num_preemptions</code> показывает, что пора.
        </p>
        <p>
          Другие движки осторожнее. SGLang оценивает будущий расход памяти с коэффициентом, который со временем смелеет, и при ошибке отзывает часть
          decode-запросов (retraction). TensorRT-LLM по умолчанию резервирует место до конца генерации (<code>GUARANTEED_NO_EVICT</code>) и никого не
          вытесняет, зато держит пустую бронь.
        </p>
      </Deeper>

      <Deeper title="overlap scheduler: CPU собирает следующий шаг, пока GPU считает текущий">
        <p>
          Подготовка шага (планирование, выделение блоков KV, сборка входов) и разбор его результата (проверка stop-условий, обновление запросов)
          идут на CPU, а сам forward и сэмплинг на GPU. Если делать CPU-часть строго между шагами, GPU каждый раз немного простаивает, и на маленьких
          моделях с короткими шагами это заметно. Поэтому движки планируют шаг N+1, пока GPU считает шаг N.
        </p>
        <p>
          SGLang называет это zero-overhead (overlap) scheduler: токены, которых ещё нет, подставляются как «будущие» ссылки, и GPU сам заменяет их
          значениями. В версии 0.4 это дало 1,1× к прошлой версии и 1,3× к другим движкам; выключается флагом <code>--disable-overlap-schedule</code>.
          В vLLM тот же приём называется async scheduling и включён по умолчанию с v0.14 (<code>--no-async-scheduling</code>). У TensorRT-LLM overlap
          scheduler стоит одного лишнего шага decode на запрос.
        </p>
        <CodeBlock
          lang="bash"
          title="ручки планировщика vLLM"
          code={'vllm serve Qwen/Qwen3-8B \\\n  --max-num-batched-tokens 2048 \\\n  --max-num-seqs 256'}
        />
      </Deeper>

      <Takeaways
        items={[
          <>
            Шаг decode плотной модели при 32 запросах ненамного дороже, чем при одном: веса читаются один раз на всех. У MoE-модели batch задевает
            всё больше экспертов, и шаг растёт быстрее, но токенов в секунду всё равно в разы больше. Поэтому batching становится главным рычагом цены
            токена; платят за него задержкой каждого пользователя, а рост ограничивает KV, который читается для каждого запроса.
          </>,
          <>Continuous batching пересобирает batch на каждом шаге: нет ожидания набора группы и пустых слотов после коротких ответов.</>,
          <>
            Кто войдёт в шаг, решают бюджет токенов (<code>max_num_batched_tokens</code>) и предел запросов (<code>max_num_seqs</code>); длинный промпт
            режется на куски, это chunked prefill.
          </>,
          <>Размер куска задаёт компромисс: меньше бюджет, ровнее ITL у всех, но позже первый токен у длинного запроса.</>,
        ]}
      />
    </Chapter>
  )
}
