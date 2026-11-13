import { useMemo } from 'react'
import {
  arithmeticIntensity,
  decodeAttentionIntensity,
  decodeBatchSweep,
  decodeLowerBoundMs,
  decodeStepBreakdown,
  linearLayerIntensity,
  moeExpertsTouched,
  peakTflops,
  prefillBreakdown,
  ridgePoint,
  stepWeightBytes,
  usableMemBytes,
  weightBytes,
} from '@/engine/perf'
import { getGpu } from '@/engine/data/gpus'
import { getModel } from '@/engine/data/models'
import { Block, Callout, Chapter, ChapterLink, Deeper, SimulatorLink, Stat, Takeaways, Term, cssVar, fmt } from '@/ui'
import { DTYPE_LABEL, times, useHardwareState, valueUnit } from '@/chapters/prefill-decode/shared/hardware'
import { usePromptTokens } from '@/chapters/prefill-decode/shared/prompt'
import { texNum } from '@/chapters/prefill-decode/shared/tex'
import { DecodeCalculator } from './DecodeCalculator'
import { PrefillCalculator } from './PrefillCalculator'
import { RooflineFigure } from './RooflineFigure'
import { WideFormula } from '@/chapters/prefill-decode/shared/WideFormula'

/** «8 экспертов», «4 эксперта»: у gpt-oss каждый токен идёт к четырём экспертам. */
const EXPERT_FORMS = ['эксперт', 'эксперта', 'экспертов'] as const

export default function RooflineChapter() {
  const state = useHardwareState()
  const hw = state.hw
  const prompt = usePromptTokens()
  const userPrompt = prompt.tokens?.length ?? null

  const n = useMemo(() => {
    const { model, gpu, dtype, tp, perf } = hw
    const peak = peakTflops(gpu, dtype)
    const wBytes = stepWeightBytes(model, dtype, 1)
    const P = (model.activeMatmulParamsB ?? model.activeParamsB) * 1e9
    const readMs = (wBytes / (gpu.bwTBs * 1e12)) * 1000
    const computeMs = ((2 * P) / (peak * 1e12)) * 1000
    const ideal = { ...perf, avgContext: 1024, mbu: 1, mfu: 1, overheadMs: 0 }
    const sweep = decodeBatchSweep(model, gpu, [1, 64], ideal)
    const pre2k = prefillBreakdown(model, gpu, 2000, perf)
    const h100 = getGpu('h100-sxm')
    const llama = getModel('llama-3.1-8b')
    return {
      peak,
      ridge: ridgePoint(gpu, dtype),
      wBytes,
      // вся модель (у MoE больше, чем читает шаг decode: тот читает только активных экспертов)
      wAll: weightBytes(model, dtype),
      P,
      readMs,
      computeMs,
      ratio: readMs / computeMs,
      lb: decodeLowerBoundMs(model, gpu, dtype, tp),
      b1: sweep[0],
      b64: sweep[1],
      // у MoE шаг с batch 64 задевает почти всех экспертов и читает почти всю модель
      b64Weights: decodeStepBreakdown(model, gpu, { ...ideal, batch: 64 }).weightBytes,
      touched64: moeExpertsTouched(model, 64),
      prefill2k: pre2k.ms,
      pre2kBound: pre2k.bound,
      pre2kWeights: pre2k.weightBytes,
      attnAi: decodeAttentionIntensity(model, 'bf16'),
      lin64: linearLayerIntensity(64, 4096, 4096),
      h100Ridge: ridgePoint(h100, 'bf16'),
      h100Fp8Ridge: ridgePoint(h100, 'fp8'),
      h200Ridge: ridgePoint(getGpu('h200'), 'bf16'),
      llamaLb: decodeLowerBoundMs(llama, h100, 'bf16'),
      // интенсивность decode при огромном batch: чтение KV растёт вместе с batch, и она упирается в предел
      aiLim1024: arithmeticIntensity(model, { phase: 'decode', batch: 1e6, avgContext: 1024, dtype }).intensity,
      ai1024at64: arithmeticIntensity(model, { phase: 'decode', batch: 1024, avgContext: 64, dtype }).intensity,
      // пол ITL под нагрузкой: шаг читает почти всю память GPU (веса + KV активных запросов)
      memBytes: usableMemBytes(gpu),
      fullMemMs: (usableMemBytes(gpu) / (gpu.bwTBs * 1e12)) * 1000,
    }
  }, [hw])

  const { model, gpu, dtype } = hw
  // MoE: сколько экспертов на слой и сколько из них выбирает каждый токен
  const moe = model.experts ?? null
  const onGpus = `${hw.tp > 1 ? `${hw.tp} × ` : ''}${gpu.name}`

  return (
    <Chapter
      id="roofline"
      lead={
        <>
          У GPU два предела: сколько операций в секунду он успевает сделать и сколько байт в секунду успевает подвезти из памяти. Какой упрётся первым,
          зависит от того, сколько вычислений приходится на каждый прочитанный байт. Prefill использует каждый вес для сотен и тысяч токенов, decode при
          batch 1 только для одного. Поэтому prefill упирается в вычисления, а decode в память.
        </>
      }
    >
      <p>
        Возьмём {gpu.name} и {model.name} (как в симуляторе; в фигурах ниже их можно поменять). У GPU два главных числа. Пиковая скорость тензорных
        ядер: {fmt.int(n.peak)} TFLOPS в {DTYPE_LABEL[dtype]}, то есть {fmt.int(n.peak)}·10¹² операций с плавающей точкой в секунду (<Term t="FLOPS" />).
        И <Term t="memory bandwidth">полоса памяти</Term> <Term t="high-bandwidth memory">HBM</Term>: {fmt.bandwidth(gpu.bwTBs * 1000)}. С такой скоростью
        данные едут из памяти GPU к вычислителям.
      </p>

      <Block>
        <div className="grid gap-4 rounded-panel border border-line bg-surface px-4 py-4">
          <div className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4">
            <Stat label="Пик вычислений" value={fmt.int(n.peak)} unit="TFLOPS" hint={`${gpu.name}, ${DTYPE_LABEL[dtype]}, dense`} />
            <Stat label="Полоса памяти" value={valueUnit(fmt.bandwidth(gpu.bwTBs * 1000))[0]} unit={valueUnit(fmt.bandwidth(gpu.bwTBs * 1000))[1]} hint="HBM" />
            <Stat label="Прочитать веса" value={valueUnit(fmt.ms(n.readMs))[0]} unit={valueUnit(fmt.ms(n.readMs))[1]} hint={`${model.name}: ${fmt.bytes(n.wBytes)}`} swatch="var(--c-decode)" />
            <Stat label="Посчитать токен" value={valueUnit(fmt.ms(n.computeMs))[0]} unit={valueUnit(fmt.ms(n.computeMs))[1]} hint="2 FLOP на параметр" swatch="var(--c-prefill)" />
          </div>
          {/* одна шкала времени: чтение весов против счёта одного токена */}
          <div className="grid gap-2.5 border-t border-line pt-4" role="img" aria-label={`Шаг decode при batch 1: прочитать веса ${fmt.ms(n.readMs)}, посчитать токен ${fmt.ms(n.computeMs)}`}>
            <TimeBar label={`Прочитать веса, ${fmt.bytes(n.wBytes)}`} value={fmt.ms(n.readMs)} frac={1} role="decode" />
            <TimeBar label={`Посчитать один токен, ${fmt.flops(2 * n.P)}`} value={fmt.ms(n.computeMs)} frac={n.computeMs / n.readMs} role="prefill" />
            <p className="text-[13px] leading-snug text-ink-3">
              Чтение весов длится в {times(n.ratio)} дольше счёта: при batch 1 вычислители почти всё время ждут.
            </p>
          </div>
        </div>
      </Block>

      <p>
        Теперь один шаг decode при <Term t="батч">batch</Term> 1. Batch это группа запросов, которые GPU обрабатывает одним проходом: каждый получает
        свой следующий токен, а веса модели читаются один раз на всех (подробно в главе <ChapterLink to="batching" />). Batch 1 значит, что запрос на
        GPU один.
      </p>
      <p>
        {moe ? (
          <>
            Чтобы выдать один токен, нужно прочитать все веса, через которые он проходит, {fmt.bytes(n.wBytes)}. У{' '}
            <Term t="MoE (Mixture of Experts)">MoE</Term> это только активная часть: всё, кроме экспертов, плюс{' '}
            {fmt.count(moe.active, EXPERT_FORMS)} из {moe.total} в каждом слое, а вся модель весит {fmt.bytes(n.wAll)}. Чтение занимает{' '}
            {fmt.ms(n.readMs)}.
          </>
        ) : (
          <>
            Чтобы выдать один токен, нужно прочитать все веса модели, {fmt.bytes(n.wBytes)}: это {fmt.ms(n.readMs)}.
          </>
        )}{' '}
        Посчитать при этом нужно около 2 FLOP на каждый параметр (умножение и сложение), всего {fmt.flops(2 * n.P)}, и у тензорных ядер это займёт{' '}
        {fmt.ms(n.computeMs)}. Чтение длится в {times(n.ratio)} дольше счёта. Вычислители почти всё время ждут, пока подвезут данные. Это и есть упор
        в память, <Term t="memory-bound" />.
      </p>
      {moe ? (
        <p>
          У MoE prefill устроен иначе. Промпт из 2 000 токенов задевает почти всех экспертов, поэтому prefill читает почти всю модель,{' '}
          {fmt.bytes(n.pre2kWeights)}, а не {fmt.bytes(n.wBytes)}, как шаг decode. Работы в 2 000 раз больше, а чтения в{' '}
          {times(n.pre2kWeights / n.wBytes)} больше: каждый токен считает только {fmt.plural(moe.active, ['своим', 'своими', 'своими'])} {moe.active}{' '}
          {fmt.plural(moe.active, ['экспертом', 'экспертами', 'экспертами'])} из {moe.total}.{' '}
          {n.pre2kBound === 'memory' ? (
            <>
              Поэтому у {model.name} на {onGpus} prefill 2 000 токенов ещё упирается в чтение весов. Упор в вычисления (<Term t="compute-bound" />
              ), когда уже память ждёт вычислителей, наступает только на тысячах токенов.
            </>
          ) : (
            <>
              Этого хватает, чтобы уже память ждала вычислителей, это упор в вычисления, <Term t="compute-bound" />. Но запас у MoE куда меньше, чем у
              плотной модели.
            </>
          )}{' '}
          Prefill 2 000 токенов идёт примерно {fmt.ms(n.prefill2k)}, а не {fmt.ms(n.readMs * 2000)}, как было бы при чтении весов на каждый токен.
        </p>
      ) : (
        <p>
          Prefill промпта из 2 000 токенов читает те же веса один раз, но умножает их на 2 000 векторов: работы в 2 000 раз больше на тот же объём
          чтения. Теперь уже память ждёт вычислителей, это упор в вычисления, <Term t="compute-bound" />. Prefill 2 000 токенов идёт примерно{' '}
          {fmt.ms(n.prefill2k)}, а не {fmt.ms(n.readMs * 2000)}, как было бы при чтении весов на каждый токен.
        </p>
      )}

      <h3>Интенсивность и ridge point</h3>
      <p>
        Обе ситуации описывает одно число: <Term t="arithmetic intensity">арифметическая интенсивность</Term>, сколько FLOP операция делает на каждый
        байт, прочитанный из памяти. У линейного слоя в BF16 каждый вес занимает 2 байта и на каждый токен даёт 2 FLOP. Если за один проход через слой
        идут T токенов, интенсивность растёт примерно как T: у decode при batch 1 около 1 FLOP/байт, у prefill на {fmt.int(2000)} токенов больше
        тысячи.
      </p>
      <p>
        У GPU тоже есть такое число: пиковые FLOPS, делённые на полосу. Это <Term t="ridge point" />: сколько FLOP на байт нужно, чтобы вычислители и
        память были загружены поровну. Для H100 в BF16 это 989 / 3,35 ≈ {fmt.int(n.h100Ridge)} FLOP/байт
        {gpu.id === 'h100-sxm' && dtype === 'bf16' && Math.abs(n.ratio / n.ridge - 1) > 0.03 && (
          <>
            {' '}
            (выше чтение оказалось дольше счёта в {times(n.ratio)}, а не в {times(n.ridge)}: там в весах учтена и таблица эмбеддингов, а decode
            читает из неё лишь одну строку на токен)
          </>
        )}
        . Ниже ridge point операция упирается в память, выше в вычисления. Достижимая скорость:
      </p>
      <WideFormula
        tex="\text{достижимые FLOPS} = \min\left(\text{пик},\ \text{интенсивность} \times \text{полоса}\right)"
        narrowTex="\begin{aligned} &\text{достижимые FLOPS} = \\ &\min\left(\text{пик},\ \text{интенсивность} \times \text{полоса}\right) \end{aligned}"
        label="модель roofline"
      />
      <p>
        На графике в логарифмических осях эта формула рисует «крышу», отсюда название <Term t="roofline" />: наклонный участок, где скорость задаёт
        память, и горизонтальный, где её задают вычисления.
      </p>

      <Callout kind="intuition" title="повар и холодильник">
        <p>
          Повар режет очень быстро, а холодильник стоит в другом конце кухни. Пока помощник приносит один продукт, повар успел бы нарезать около 300.
          Если на каждый принесённый продукт приходится одна порция (decode при batch 1), повар почти всё время стоит без дела. Если из одной поездки
          готовят сотни порций сразу (prefill, большой batch), ждать начинает уже помощник.
        </p>
        <p>
          Есть подвох: у каждого гостя ещё и свой личный набор продуктов, его KV cache. Его приходится везти отдельно для каждого гостя, сколько бы гостей
          ни было за столом. Поэтому большой batch спасает от поездок за общими продуктами (весами), но не за личными.
        </p>
      </Callout>

      <p>Где на этой крыше стоят decode и prefill и что их двигает?</p>

      <RooflineFigure state={state} userPrompt={userPrompt} />

      <p>
        Что заметить. Decode при batch 1 стоит около 1 FLOP/байт, примерно в {times(n.ridge)} левее излома: GPU использует меньше процента своего
        пика. Batch двигает точку вправо: веса читаются один раз на весь batch, а работы в B раз больше
        {moe ? ' (у MoE медленнее: чем больше batch, тем больше задето экспертов и тем больше весов читается)' : ''}.{' '}
        {n.aiLim1024 < n.ridge ? (
          <>
            Но при контексте {fmt.int(1024)} токена точка decode не доходит до излома ни при каком batch: вместе с batch растёт и чтение KV, и
            интенсивность упирается примерно в {fmt.int(n.aiLim1024)} FLOP/байт.{' '}
            {n.ai1024at64 >= n.ridge ? 'Поставьте контекст 64 токена, и излом становится достижим. ' : 'У этой модели излом недостижим даже при контексте 64 токена. '}
          </>
        ) : (
          <>
            При контексте {fmt.int(1024)} токена излом достижим, но только при batch в сотни последовательностей: вместе с batch растёт и чтение KV.{' '}
          </>
        )}
        Полая точка attention не двигается: у каждой
        последовательности свой KV, и сколько ни добавляй последовательностей, на каждый байт KV приходится одинаково мало работы (около{' '}
        {fmt.num(n.attnAi)} FLOP/байт у {model.name}).{' '}
        {moe ? (
          <>
            Prefill MoE переходит излом только на тысячах токенов, а не на сотнях, как плотная модель: каждый токен идёт лишь к {moe.active}{' '}
            {fmt.plural(moe.active, ['эксперту', 'экспертам', 'экспертам'])} из {moe.total}, каждому эксперту достаётся около 1/
            {fmt.num(moe.total / moe.active)} промпта, и на прочитанный байт весов приходится в разы меньше работы.
          </>
        ) : (
          'Prefill переходит излом уже на нескольких сотнях токенов.'
        )}{' '}
        <SimulatorLink stage="decode" />
      </p>

      <h3>Decode: скорость покупается полосой памяти</h3>
      <p>
        Раз decode упирается в память, его скорость задаёт полоса, а не FLOPS. Нижняя граница шага: время один раз прочитать все веса, через которые
        проходит токен. Отсюда потолок скорости для одного пользователя: для {model.name} на {onGpus} не больше {fmt.int(1000 / n.lb)} ток/с, и
        никакая оптимизация ядер это не обойдёт. Поднять потолок можно тремя путями: читать меньше байт на токен (<Term t="квантизация" />), читать
        веса быстрее (GPU с более быстрой памятью или несколько GPU в <Term t="Tensor parallelism (TP)">tensor parallel</Term>: каждый читает только
        свою долю весов, но в каждом слое GPU обмениваются данными, глава <ChapterLink to="parallelism" />) или выдавать несколько токенов за одно
        чтение весов (<Term t="speculative decoding" />). Для сравнения, Llama-3.1-8B в BF16 на H100 не может быть быстрее {fmt.ms(n.llamaLb)} на
        токен.
      </p>
      <p>
        Зато batch почти бесплатен, пока шаг упирается в память: веса читаются один раз на всех, и шаг удлиняется только на чтение KV добавленных
        последовательностей. Сколько пользователей можно обслужить одним шагом и чем за это платит каждый из них?
      </p>

      <DecodeCalculator state={state} />

      <p>
        Что заметить. В идеале при контексте 1 024 токена batch 1 даёт {fmt.int(n.b1.perUserTokS)} ток/с, а batch 64 даёт {fmt.int(n.b64.totalTokS)}{' '}
        ток/с на всех: в {times(n.b64.totalTokS / n.b1.totalTokS)} больше, при этом каждый пользователь получает {fmt.int(n.b64.perUserTokS)} ток/с.
        Это главный компромисс сервинга: throughput против скорости на пользователя (подробно в главе <ChapterLink to="metrics" />). Чем длиннее
        контекст, тем раньше чтение KV съедает выигрыш, а потом KV просто перестаёт помещаться в память (глава <ChapterLink to="kv-cache" />).
      </p>
      <p>
        У ITL под нагрузкой есть и свой пол: если память GPU занята весами и KV активных запросов, каждый шаг читает почти всю HBM. На {gpu.name} это
        около {fmt.bytes(n.memBytes)} / {fmt.bandwidth(gpu.bwTBs * 1000)} ≈ {fmt.ms(n.fullMemMs)}, то есть при полностью занятой памяти не больше{' '}
        {fmt.int(1000 / n.fullMemMs)} ток/с на пользователя. Поэтому жёсткий SLO по ITL ограничивает batch раньше, чем кончается память (глава{' '}
        <ChapterLink to="metrics" />
        ).
      </p>

      <h3>Prefill: время считают по FLOPs</h3>
      <p>
        Для prefill работает обратная логика. Длинный промпт упирается в вычисления, поэтому время prefill считают по количеству операций. На каждый
        токен промпта нужно
        около 2 FLOP на каждый параметр P, который участвует в умножениях, плюс attention, который растёт квадратично с длиной. Время = FLOPs / (пик ×
        MFU). <Term t="model FLOPs utilization">MFU</Term> (model FLOPs utilization) это доля пика, которую реально получают ядра; на практике около 40-60 %.
      </p>

      <PrefillCalculator state={state} userPrompt={userPrompt} />

      <p>
        Что заметить. Линия вычислений растёт с длиной промпта, линия чтения весов почти горизонтальна: веса читаются один раз, сколько бы токенов ни шло
        через модель.{' '}
        {moe ? (
          <>
            Пересечение и есть граница. У MoE она на тысячах токенов, а не на сотнях: каждый токен считает только своими экспертами. Короче неё
            prefill обрабатывается за время чтения весов, но это не один шаг decode: уже несколько десятков токенов задевают большую часть экспертов,
            и читается почти вся модель.
          </>
        ) : (
          <>
            Пересечение и есть граница: промпт короче нескольких сотен токенов обрабатывается за время чтения весов, примерно как один шаг decode.
          </>
        )}{' '}
        На длинных промптах заметна доля attention: квадратичная часть начинает догонять линейную.
      </p>

      <Callout kind="production" title="что ускоряет каждую фазу">
        <ul>
          <li>
            Decode: полоса памяти и меньше байт на шаг. H200 при тех же FLOPS, что у H100, имеет 4,8 ТБ/с вместо 3,35 и даёт decode до 1,4 раза быстрее, а
            prefill такой же. Квантизация весов в FP8 вдвое уменьшает чтение и поднимает потолок decode (глава <ChapterLink to="quantization" />).
            Несколько GPU в tensor parallel ускоряют и decode: каждый читает свою долю весов (глава <ChapterLink to="parallelism" />).
          </li>
          <li>
            Простаивающие вычислители decode можно загрузить: batch поднимает throughput, а speculative decoding проверяет несколько угаданных токенов
            за один проход (глава <ChapterLink to="speculative" />).
          </li>
          <li>
            Prefill: FLOPS (B200 против H100), FP8-вычисления, несколько GPU (<ChapterLink to="parallelism" />) и самое дешёвое: не считать повторяющийся
            префикс вовсе (<ChapterLink to="prefix-caching" />).
          </li>
        </ul>
      </Callout>

      <Callout kind="pitfall" title="цифры из даташита">
        <p>
          В даташите H100 написано «1 979 TFLOPS BF16», но со сноской «with sparsity»: это режим структурной разреженности 2:4, который в инференсе LLM
          почти не применяют. Плотная (dense) цифра вдвое меньше: 989. Все расчёты здесь по dense.
        </p>
        <p>
          И ещё: «GPU utilization 100 %» в <code>nvidia-smi</code> означает лишь, что всё время замера на GPU исполнялось хотя бы одно{' '}
          <Term t="kernel">ядро</Term> (kernel, то есть программа на GPU), а не что заняты все вычислители. Decode при batch 1 может показывать 100 % и
          при этом использовать меньше процента пиковых FLOPS.
        </p>
      </Callout>

      <Deeper title="формулы интенсивности, MFU и MBU">
        <h4>Интенсивность линейного слоя</h4>
        <p>Слой с матрицей весов d_in × d_out в BF16 (2 байта на число), T токенов за проход:</p>
        <WideFormula
          tex="\text{AI} = \frac{2\,T\,d_{in}\,d_{out}}{2\,d_{in}\,d_{out} + 2\,T\,(d_{in} + d_{out})} \approx T \quad (T \ll d)"
          narrowTex="\begin{aligned} \text{AI} &= \frac{2\,T\,d_{in}\,d_{out}}{2\,d_{in}\,d_{out} + 2\,T\,(d_{in} + d_{out})} \\ &\approx T \quad (T \ll d) \end{aligned}"
          label="числитель: умножение и сложение на каждый вес для каждого токена; знаменатель: веса плюс входные и выходные активации"
        />
        <p>
          Например, d = 4096 и T = 64 дают {fmt.num(n.lin64, 1)} FLOP/байт: почти ровно T. Отсюда правило: на H100 в BF16 линейные слои перестают упираться в
          память, когда за проход идут примерно 300-350 токенов. Для decode это batch, для prefill длина промпта (или куска при chunked prefill).
        </p>
        <h4>Ridge point разных GPU</h4>
        <p>
          H100 BF16: 989 / 3,35 ≈ {fmt.int(n.h100Ridge)}. В FP8 пик вдвое выше и ridge ≈ {fmt.int(n.h100Fp8Ridge)}, но и вес занимает 1 байт вместо 2,
          поэтому интенсивность того же batch тоже удваивается: граница по batch почти не сдвигается. H200: те же FLOPS, полоса 4,8 ТБ/с, ridge ≈{' '}
          {fmt.int(n.h200Ridge)}. Чем ниже ridge, тем меньший batch нужен, чтобы загрузить вычислители.
        </p>
        <h4>Почему attention в decode всегда упирается в память</h4>
        <p>
          На каждый токен в кэше на каждом слое каждая Q-голова делает около 4 × head_dim FLOP (скалярное произведение с K и взвешенная сумма V), а
          читается K и V одной KV-головы: 2 × head_dim чисел по 2 байта. При GQA одну KV-голову делят heads / kv_heads голов запроса:
        </p>
        <WideFormula tex={`\\text{AI}_{\\text{attn}} \\approx \\frac{h_q}{h_{kv}} = \\frac{${model.heads}}{${model.kvHeads}} = ${texNum(n.attnAi)}\\ \\text{FLOP/байт}`} label={`${model.name}, BF16 KV, не зависит от batch`} />
        <p>
          Batch тут не помогает: каждая последовательность читает свой KV целиком. Поэтому на длинном контексте decode не становится compute-bound ни при
          каком batch: чтение KV растёт вместе с batch. Исключение среди популярных архитектур: MLA у DeepSeek-V3, где интенсивность attention около 240.
        </p>
        <h4>MFU и MBU</h4>
        <p>
          MFU (model FLOPs utilization): достигнутые FLOP/с, делённые на пик. Для prefill это главный показатель качества ядер.{' '}
          <Term t="MBU" /> (model bandwidth utilization): достигнутая полоса, делённая на пиковую, где достигнутая полоса = (байты весов + байты KV) / время шага.
          Пример: модель на 7 млрд параметров в FP16 (14 ГБ) с шагом 14 мс прокачивает 1 ТБ/с; на GPU с полосой 2 ТБ/с это MBU 50 %. Низкий MBU
          значит, что тормозит софт (запуски ядер, планировщик), а не железо. В калькуляторах этой главы по умолчанию MFU {fmt.pct(0.5)} и MBU{' '}
          {fmt.pct(0.7)}: это допущения, реальные значения зависят от движка и модели.
        </p>
      </Deeper>

      <Takeaways
        items={[
          <>Интенсивность = FLOP на байт из памяти. Ниже ridge point (у H100 в BF16 ≈ 295) операция ждёт память, выше считает на пределе.</>,
          <>Decode при batch 1 делает около 1 FLOP на байт, поэтому его скорость задают полоса памяти и размер весов: шаг ≥ веса / полоса.</>,
          moe ? (
            <>
              У MoE batch помогает меньше: при контексте {fmt.int(1024)} токена batch 64 даёт в {times(n.b64.totalTokS / n.b1.totalTokS)} больше
              токенов в секунду, а не в 64, потому что 64 запроса задевают{' '}
              {n.touched64 >= 0.95 * moe.total ? 'почти всех экспертов' : `около ${fmt.int(n.touched64)} экспертов из ${moe.total}`} и шаг читает уже{' '}
              {fmt.bytes(n.b64Weights)} весов вместо {fmt.bytes(n.wBytes)}. KV каждой последовательности читается целиком: attention в decode всегда
              упирается в память.
            </>
          ) : (
            <>
              Batch делит чтение весов на всех и поднимает throughput почти пропорционально, пока суммарный KV batch мал по сравнению с весами: при
              контексте {fmt.int(1024)} токена batch 64 даёт в {times(n.b64.totalTokS / n.b1.totalTokS)} больше токенов в секунду, а не в 64. KV каждой
              последовательности читается целиком: attention в decode всегда упирается в память.
            </>
          ),
          <>
            Prefill считают по FLOPs: время ≈ 2 · P · N / (пик · MFU). Промпт короче нескольких сотен токенов ещё упирается в память
            {moe ? ', у MoE граница в тысячах токенов' : ''}.
          </>,
        ]}
      />
    </Chapter>
  )
}

/** Полоса времени на общей шкале (без серой дорожки): длина пропорциональна времени, подпись сразу за концом. */
function TimeBar({ label, value, frac, role }: { label: string; value: string; frac: number; role: 'prefill' | 'decode' }) {
  return (
    <div className="grid gap-1">
      <div className="text-[13px] leading-snug text-ink-2">{label}</div>
      <div className="flex min-w-0 items-center gap-2">
        <span
          aria-hidden
          className="h-3.5 shrink-0 rounded-[3px]"
          style={{ width: `calc((100% - 5rem) * ${Math.max(0, Math.min(1, frac))})`, minWidth: 2, background: cssVar(role) }}
        />
        <span className="text-[13px] font-medium whitespace-nowrap text-ink">{value}</span>
      </div>
    </div>
  )
}
