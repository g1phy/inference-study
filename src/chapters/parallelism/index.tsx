import { useMemo } from 'react'
import { Callout, Chapter, ChapterLink, CodeBlock, Deeper, Takeaways, Term, fmt } from '@/ui'
import { getModel } from '@/engine/data/models'
import { getGpu } from '@/engine/data/gpus'
import { decodeLowerBoundMs, moeExpertsTouched, weightBytes } from '@/engine/perf'
import { SpecTable } from '../batching/chart'
import { FitsFigure } from './FitsFigure'
import { PipelineFigure } from './PipelineFigure'
import { STRAT_CONTEXT, StrategyFigure } from './StrategyFigure'
import { allReduceMs, IB_PER_GPU_GBS, LAUNCH_MS, plan, usablePerGpu } from './plan'
import { nvlinkFor } from '../disaggregation/logic'

/** Нагрузка для сравнения EP16 и TP8 × PP2 в «Глубже» (максимум в фигуре). */
const MOE_LOAD = 256
/* неразрывные пробелы вокруг «×», чтобы строка не рвалась посередине обозначения */
const TWO_NODES = '16\u00a0×\u00a0H100'
const HYBRID = 'TP8\u00a0×\u00a0PP2'

export default function ParallelismChapter() {
  const n = useMemo(() => {
    const llama = getModel('llama-3.1-70b')
    const ds = getModel('deepseek-v3')
    const qwen = getModel('qwen3-235b-a22b')
    const h100 = getGpu('h100-sxm')
    const h200 = getGpu('h200')
    const b200 = getGpu('b200')
    const nvH100 = nvlinkFor(h100).GBs
    const msg = 32 * llama.hidden * 2
    // EP16 и TP8 × PP2 на двух нодах H100 в FP8, 256 запросов: как в фигуре
    const two = (model: typeof ds, strategy: 'ep' | 'tppp') =>
      plan({ model, gpu: h100, dtype: 'fp8', strategy, gpus: 16, concurrency: MOE_LOAD, context: STRAT_CONTEXT })
    return {
      dsEp: two(ds, 'ep'),
      dsHybrid: two(ds, 'tppp'),
      qwenEp: two(qwen, 'ep'),
      qwenHybrid: two(qwen, 'tppp'),
      llamaBf16: weightBytes(llama, 'bf16'),
      dsFp8: weightBytes(ds, 'fp8'),
      h100Process8: 8 * usablePerGpu(h100),
      h200Process8: 8 * usablePerGpu(h200),
      lb1: decodeLowerBoundMs(llama, h100, 'fp8', 1),
      lb4: decodeLowerBoundMs(llama, h100, 'fp8', 4),
      nvH100,
      arNv: 2 * llama.layers * allReduceMs(msg, 8, nvlinkFor(b200).GBs),
      arIb: 2 * llama.layers * allReduceMs(msg, 8, IB_PER_GPU_GBS),
      launchNv: 2 * llama.layers * LAUNCH_MS.nvlink,
      launchIb: 2 * llama.layers * LAUNCH_MS.ib,
      lb8b200: decodeLowerBoundMs(llama, b200, 'fp8', 8),
      touched64: moeExpertsTouched(qwen, 64),
      layers: llama.layers,
    }
  }, [])

  return (
    <Chapter
      id="parallelism"
      lead={
        <>
          Большие модели не помещаются в один GPU, а если и помещаются, одному GPU не хватает памяти под KV cache или скорости. Модель раскладывают
          на несколько GPU четырьмя способами: TP режет матрицы каждого слоя, PP делит слои на стадии, EP раздаёт экспертов MoE, DP делает
          независимые реплики.
        </>
      }
    >
      <h3>Зачем делить модель</h3>
      <p>
        Первая причина в памяти. Llama-3.1-70B в BF16 весит {fmt.bytes(n.llamaBf16)}, а у H100 80 ГБ. DeepSeek-V3 даже в FP8 занимает{' '}
        {fmt.bytes(n.dsFp8)}: это больше, чем доступно движку на целой ноде из восьми H100 ({fmt.bytes(n.h100Process8)}). В одну ноду модель
        помещается только с более ёмкими GPU, например 8 × H200 ({fmt.bytes(n.h200Process8)}); на H100 нужны две ноды. А кроме весов нужно место
        под <Term t="KV cache" /> всех запросов (глава <ChapterLink to="kv-cache" />).
      </p>
      <p>
        Вторая причина в скорости. Шаг <Term t="decode" /> упирается в чтение весов из памяти (глава <ChapterLink to="roofline" />). Если веса разложены
        на четыре GPU, каждый читает четверть, и шаг короче: у Llama-3.1-70B в FP8 на H100 нижняя граница шага {fmt.ms(n.lb1)} на одном GPU и{' '}
        {fmt.ms(n.lb4)} на четырёх.
      </p>
      <p>
        Платить за это приходится обменом данными между GPU. Внутри ноды GPU соединены <Term t="NVLink / NVSwitch">NVLink</Term>: у H100 это {fmt.bandwidth(n.nvH100)} в каждую
        сторону. Между нодами идёт <Term t="InfiniBand" />, {fmt.bandwidth(IB_PER_GPU_GBS)} на карту, почти в 10 раз медленнее. И то и другое в разы
        медленнее памяти самого GPU (3,35 ТБ/с у H100). Поэтому главный вопрос любой схемы: как часто и сколько данных GPU пересылают друг другу и
        по какому каналу. Книга называет это topology-aware parallelism.
      </p>
      <p>
        Дальше четыре стратегии по очереди, а после них все вместе на одной модели в общей фигуре: что лежит на каждом GPU, где идёт связь и что
        происходит с памятью и скоростью.
      </p>

      <h3>Tensor parallelism: каждый слой на всех GPU</h3>
      <p>
        <Term t="Tensor parallelism (TP)">Tensor parallelism</Term> режет каждую большую матрицу каждого слоя на N частей: GPU хранит свои столбцы или
        строки и умножает только их. Каждый GPU читает 1/N весов, поэтому токен выходит быстрее. Но после attention и после MLP частичные результаты
        нужно сложить со всех GPU, прежде чем идти дальше. Пример: если вторую матрицу MLP разрезать по строкам между двумя GPU, каждый получит своё
        слагаемое выхода, y = x₁·W₁ + x₂·W₂, и сложить их можно, только обменявшись данными. Это all-reduce, по два на каждый слой, у Llama-3.1-70B{' '}
        {fmt.int(2 * n.layers)} обменов на каждый шаг decode.
      </p>
      <p>
        Сообщения маленькие (для batch 32 это 32 вектора по {fmt.int(8192)} числа), но их много и они идут строго по очереди со слоями. По NVLink набегает
        порядка {fmt.ms(n.launchNv)} на шаг, и это в основном задержка запуска каждого обмена, а не полоса; по InfiniBand в разы больше, дольше чтения
        самих весов. Отсюда правило: TP живёт внутри ноды. Это стратегия по умолчанию для низкой задержки и для плотных, и для MoE-моделей.
      </p>
      <Callout kind="precise" title="цена all-reduce">
        <p>
          Сообщение одного all-reduce: batch × hidden × 2 байта. Кольцевой all-reduce прогоняет через каждый GPU 2(N − 1)/N от объёма. Llama-3.1-70B,
          batch 32, TP8: {fmt.int(2 * n.layers)} обменов по 512 КиБ дают {fmt.ms(n.arNv)} на шаг по NVLink 5 и {fmt.ms(n.arIb)} по одной карте
          InfiniBand на GPU, при том что чтение весов с B200 занимает {fmt.ms(n.lb8b200)}. Это только полоса. К ней добавляется задержка запуска
          каждого обмена: малый all-reduce стоит микросекунды при любом размере. Если считать условно {fmt.ms(LAUNCH_MS.nvlink)} по NVLink и{' '}
          {fmt.ms(LAUNCH_MS.ib)} по InfiniBand, на {fmt.int(2 * n.layers)} обменах набегает ещё {fmt.ms(n.launchNv)} и {fmt.ms(n.launchIb)}: больше, чем
          сама передача.
        </p>
      </Callout>
      <p>
        У TP есть подвох с KV cache. KV-головы делятся между GPU, но если GPU больше, чем KV-голов, головы приходится копировать. У Qwen3-235B-A22B
        четыре KV-головы: на TP8 каждая лежит на двух GPU, и KV занимает вдвое больше места. У моделей с MLA (DeepSeek-V3) на токен хранится один
        общий для всех голов вектор, и при TP он целиком лежит на каждом GPU. Это лечит <Term t="DP attention">DP-attention</Term>: attention работает
        как реплики, и каждый GPU хранит KV только своих запросов. Другой способ, <Term t="Context parallelism (CP)">decode context parallel</Term>, описан в «Глубже» ниже.
      </p>

      <h3>Pipeline parallelism: слои по стадиям</h3>
      <p>
        <Term t="Pipeline parallelism (PP)">Pipeline parallelism</Term> режет модель поперёк: первый GPU держит первые слои, второй следующие, и так
        далее. Между стадиями передаются только активации токенов, это мало, поэтому PP спокойно работает поверх медленной сети. Зато токен проходит
        стадии по очереди: один пользователь не ускоряется, а пока стадия ждёт входа от предыдущей, она простаивает. Эти простои называют{' '}
        <Term t="Pipeline bubble">пузырями</Term>. Чтобы стадии не ждали, запросы делят на несколько групп (microbatch) и пускают их по конвейеру
        друг за другом.
      </p>
      <PipelineFigure />
      <p>
        В decode пузыри исчезают, когда групп в конвейере не меньше, чем стадий: vLLM для этого держит в работе не меньше batch, чем стадий.
        Если запросов меньше, чем стадий, часть GPU простаивает. Задержку отдельного запроса это не уменьшает: токен всё равно проходит стадии по
        очереди. Поэтому внутри ноды с NVLink PP не советуют: TP там быстрее. На картах без NVLink (например, L40S) документация vLLM, наоборот,
        советует PP вместо TP ради throughput и меньших затрат на связь. А на NVLink-системах PP нужен, когда модель не помещается в одну ноду: тогда
        внутри нод работает TP, а между нодами PP, например TP8 × PP2 на двух нодах по восемь GPU.
      </p>

      <h3>Expert parallelism: эксперты целиком</h3>
      <p>
        В <Term t="MoE (Mixture of Experts)">MoE</Term>-моделях MLP-блок слоя состоит из многих «экспертов», и каждый токен идёт только в несколько из
        них: у Qwen3-235B-A22B 128 экспертов, токен идёт в 8. <Term t="Expert parallelism (EP)">Expert parallelism</Term> не режет экспертов, а раздаёт
        их целиком: на EP8 каждый GPU держит 16 экспертов. Маленький роутер, который решает, к каким экспертам идёт токен, копируется на все GPU.
      </p>
      <p>
        Связь здесь другая: по сети едут векторы токенов (их скрытые состояния), к GPU с их экспертами и обратно (
        <Term t="All-to-all">all-to-all</Term>, этапы dispatch и combine), а не частичные суммы каждого слоя. Трафика меньше, чем у TP, поэтому EP
        масштабируется на несколько нод. По скорости отдельного токена EP уступает TP: эксперт не делится между GPU, токен ждёт тот GPU, которому
        досталось больше всего его экспертов, и на каждом MoE-слое дважды пересылается между GPU. Зато система держит больше запросов одновременно,
        поэтому EP выбирают ради throughput.
      </p>
      <Callout kind="pitfall" title="22B активных не значит 22B в памяти">
        В памяти должны лежать все эксперты: 235B параметров, а не 22B. И в batch разные запросы идут к разным экспертам: при 64 токенах в шаге у
        Qwen3-235B-A22B задето в среднем {fmt.num(n.touched64, 0)} экспертов из 128 на слой, то есть читаются почти все веса. Разреженность MoE
        экономит вычисления на токен, но не память и не полосу при больших batch.
      </Callout>

      <h3>Data parallelism: реплики</h3>
      <p>
        <Term t="Data parallelism (DP)">Data parallelism</Term> проще всех: N полных копий модели, каждая обслуживает свои запросы, балансировщик
        раздаёт их между копиями. Связи на каждом шаге нет, throughput растёт почти в N раз, задержка одного запроса как у одной копии. Ограничение
        одно: модель должна помещаться в копию. Если не помещается, копией становится группа GPU, например две реплики по TP4 на восьми GPU.
      </p>
      <p>
        Стратегии комбинируют. Типичная схема для больших MoE: attention работает как DP (каждый GPU ведёт свои запросы и хранит их KV), а эксперты
        раздаются по EP. DeepSeek в проде держит prefill на EP32, а decode на EP144:{' '}
        <ChapterLink to="disaggregation">дезагрегация</ChapterLink> позволяет дать фазам разный параллелизм.
      </p>
      <Callout kind="production" title="флаги">
        <p>
          vLLM: <code>--tensor-parallel-size</code>, <code>--pipeline-parallel-size</code>, <code>--data-parallel-size</code>,{' '}
          <code>--enable-expert-parallel</code> (EP вместо TP для MoE-слоёв). SGLang: <code>--tp</code>, <code>--dp</code>,{' '}
          <code>--enable-dp-attention</code>. TensorRT-LLM: <code>--tp_size</code>, <code>--ep_size</code>. Пример из книги: DeepSeek-V3.1 на восьми
          GPU.
        </p>
        <CodeBlock lang="bash" title="vLLM: TP8 для attention, EP для экспертов" code={'vllm serve deepseek-ai/DeepSeek-V3.1 \\\n  --enable-expert-parallel \\\n  --tensor-parallel-size 8'} />
      </Callout>

      <h3>Четыре стратегии на одной модели</h3>
      <p>
        Теперь сравните все стратегии на одной модели: переключайте стратегию и число GPU и смотрите, что лежит на каждом GPU, где идёт связь и что
        происходит с памятью и скоростью. На 16 GPU, то есть на двух нодах, есть и гибрид TP8 × PP2.
      </p>
      <StrategyFigure />

      <h3>Влезет ли модель</h3>
      <p>
        Выбор стратегии начинается с вопроса, сколько GPU вообще требуется. Память каждого GPU делится на веса, резерв под активации и CUDA graphs и
        KV cache запросов. Если KV-памяти мало, в batch помещается мало запросов, и GPU простаивает (глава <ChapterLink to="batching" />).
      </p>
      <FitsFigure />
      <p>
        Поэтому GPU часто берут больше минимума даже для средних моделей: лишняя память уходит под KV cache, а TP ещё и ускоряет каждого
        пользователя. Если модели не нужно больше одной ноды, лишние ноды выгоднее отдать под реплики или дезагрегацию, чем растягивать одну копию.
      </p>

      <Deeper title="несколько нод: TP внутри, PP или EP между">
        <p>
          Больше восьми GPU нужно для огромных моделей в высокой точности, для входов на миллионы токенов или ради максимальной скорости. Стандартная
          нода HGX связывает восемь GPU через NVSwitch, между нодами работает InfiniBand. Стойка GB200 NVL72 держит 72 GPU в одном NVLink-домене и
          снимает часть ограничений.
        </p>
        <SpecTable
          caption="Схемы для двух нод по восемь GPU"
          minWidth={480}
          head={['Модель', 'Схема', 'Что получаем']}
          rows={[
            ['Плотная', 'TP8 внутри ноды + PP2 между нодами', 'all-reduce идут по NVLink, между нодами едут только активации'],
            ['MoE, задержка', 'TP8 × PP2', 'ниже задержка на пользователя, чем у EP16'],
            ['MoE, throughput', 'EP16', 'обычно больше запросов в памяти (KV не копируется) и выше общий throughput; связи меньше, чем у TP16'],
          ]}
        />
        <p>
          Откуда у EP16 выигрыш по throughput? Attention в нём работает как DP, и KV каждого запроса лежит только на своём GPU. TP8 копирует KV,
          если KV-голов у модели меньше восьми или у неё MLA: копия занимает память и читается на каждом шаге. У DeepSeek-V3 (MLA) копия лежит на
          всех восьми GPU ноды, поэтому на {TWO_NODES} (веса в FP8) схема {HYBRID} помещает {fmt.int(n.dsHybrid.maxSeqs)} запросов по{' '}
          {fmt.tokens(STRAT_CONTEXT)}, а EP16 {fmt.int(n.dsEp.maxSeqs)}, и уже при {MOE_LOAD} запросах EP16 выдаёт {fmt.rate(n.dsEp.totalTokS)}{' '}
          против {fmt.rate(n.dsHybrid.totalTokS)}. У Qwen3-235B-A22B KV копируется только вдвое, и в фигуре при тех же {MOE_LOAD} запросах{' '}
          {HYBRID} даже чуть впереди ({fmt.rate(n.qwenHybrid.totalTokS)} против {fmt.rate(n.qwenEp.totalTokS)}). Отчасти это упрощение фигуры:
          она не перекрывает обмены all-to-all вычислениями, а движки это умеют (в SGLang, например, two-batch overlap).
        </p>
        <p>
          Context parallelism делит одну длинную последовательность между GPU. В LLM-инференсе он нужен в основном для длинных контекстов: в vLLM есть
          decode context parallel (флаг <code>-dcp</code>), он режет KV cache по токенам между GPU и, например, у DeepSeek с MLA при TP8 убирает
          восьмикратное дублирование KV, не добавляя GPU. Важен context parallelism и для генерации видео.
        </p>
      </Deeper>

      <Deeper title="как считаются цифры в фигуре">
        <p>
          Время шага считается по той же roofline-модели, что в главе <ChapterLink to="roofline" />: max(байты / (полоса × 0,7), FLOPs / (пик × 0,5))
          + 0,3 мс накладных.
        </p>
        <ul>
          <li>TP: веса и KV делятся на N (с дублированием KV при TP больше числа KV-голов), плюс 2 × L кольцевых all-reduce.</li>
          <li>
            PP: запросы делятся на группы (microbatch); одна стадия читает 1/N весов и KV своих слоёв, токен проходит N стадий по очереди, между стадиями
            передаются активации. Если запросов меньше, чем стадий, конвейер не заполнен.
          </li>
          <li>EP: каждый GPU читает свою копию общих весов, тех своих экспертов, к которым пришли токены, и KV своих запросов, плюс 2 × L обмена all-to-all.</li>
          <li>DP: каждая копия считает свою долю запросов, связи нет.</li>
          <li>TP8 × PP2: две стадии по TP8; all-reduce внутри ноды по NVLink, одна передача активаций между нодами по InfiniBand.</li>
        </ul>
        <p>
          Связь считается как условная задержка запуска каждого обмена ({fmt.ms(LAUNCH_MS.nvlink)} по NVLink, {fmt.ms(LAUNCH_MS.ib)} по InfiniBand) плюс
          время по полосе, без перекрытия связи с вычислениями и неравномерной загрузки экспертов. Больше восьми GPU значит две ноды: связь между нодами
          идёт по одной карте InfiniBand NDR на GPU.
        </p>
      </Deeper>

      <Takeaways
        items={[
          <>TP режет каждый слой: ускоряет одного пользователя, но требует all-reduce на каждом слое, поэтому живёт внутри ноды на NVLink.</>,
          <>PP делит слои на стадии: даёт память и терпит медленную связь, но не ускоряет токен и простаивает, если запросов меньше, чем стадий. Это способ выйти за пределы ноды: TP8 × PP2 на двух нодах.</>,
          <>EP раздаёт экспертов MoE целиком и гоняет токены all-to-all: растёт прежде всего throughput, а по скорости токена EP уступает TP. DP делает реплики, самый простой способ масштабировать throughput.</>,
          <>Влезет ли модель, решают веса, резерв и KV cache; при TP больше числа KV-голов KV дублируется.</>,
        ]}
      />
    </Chapter>
  )
}
