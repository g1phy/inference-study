import { useMemo, useState } from 'react'
import { Callout, Chapter, ChapterLink, Deeper, SimulatorLink, Takeaways, Term, fmt } from '@/ui'
import { useSettings } from '@/state/settings'
import { getModel } from '@/engine/data/models'
import { getGpu } from '@/engine/data/gpus'
import { moeExpertsTouched, ridgePoint, weightBytes } from '@/engine/perf'
import { archById, attentionCrossover, computeArch, isArchModel } from './arch'
import { BlockFigure } from './BlockFigure'
import { ResidualFigure } from './ResidualFigure'
import { ModelFigure } from './ModelFigure'
import { MoeFigure } from './MoeFigure'

export default function TransformerChapter() {
  const [settings] = useSettings()
  const [picked, setPicked] = useState<string | null>(null)
  const modelId = picked ?? (isArchModel(settings.modelId) ? settings.modelId : 'qwen3-8b')
  const arch = useMemo(() => archById(modelId), [modelId])
  const [ctxRaw, setCtx] = useState(4096)
  const ctx = Math.min(ctxRaw, arch.model.maxContext)

  // числа для прозы: всегда Qwen3-8B (опорная модель сайта) и Qwen3-30B-A3B
  const a8 = useMemo(() => computeArch(getModel('qwen3-8b')), [])
  const m8 = a8.model
  const matmul8 = m8.layers * (a8.attn.flopsFixed + a8.mlp.flopsFixed) + a8.head.flopsFixed
  const moe = getModel('qwen3-30b-a3b')
  const big = getModel('qwen3-235b-a22b')
  const touched64 = moeExpertsTouched(moe, 64)
  const h100 = getGpu('h100-sxm')
  const ridge = ridgePoint(h100, 'bf16')

  return (
    <Chapter
      id="transformer"
      lead={
        <>
          Вектор каждого токена проходит N одинаковых по устройству блоков: у {m8.name} их {m8.layers}. В каждом блоке attention даёт токену
          забрать информацию у предыдущих токенов, а MLP обрабатывает каждый токен сам по себе. После последнего блока LM head превращает вектор
          в оценки для всех токенов словаря.
        </>
      }
    >
      <p>
        После <ChapterLink to="embeddings">эмбеддингов</ChapterLink> у нас матрица: по вектору из d_model чисел на каждый токен. Дальше
        начинается основная часть <Term t="трансформер">трансформера</Term>: стопка <Term t="Transformer block (layer)">блоков</Term>, их ещё
        называют слоями (decoder layers). Блоки устроены одинаково, но веса у каждого свои.
      </p>
      <p>
        Внутри блока две части. Attention смешивает информацию между токенами: местоимение может забрать признаки у существительного, к
        которому относится. <Term t="MLP" /> работает с каждым токеном отдельно: берёт вектор и перерабатывает его. Обе части не заменяют
        вектор, а прибавляют к нему поправку. Перед каждой частью стоит <Term t="RMSNorm" />: она только выравнивает масштаб вектора и почти
        ничего не стоит.
      </p>
      <p>
        Что делает каждая часть и почему у неё столько параметров? Стоимость будем считать в <Term t="FLOP" />, то есть в числе умножений и
        сложений на один токен (откуда «2 FLOP на параметр», объяснено ниже, после схемы). Нажмите на часть схемы или запустите проход вектора
        по блоку.
      </p>

      <BlockFigure arch={arch} ctx={ctx} onCtx={setCtx} modelId={modelId} onModel={setPicked} />

      <p>Из схемы стоит вынести две вещи.</p>
      <ul>
        <li>
          Почти все параметры блока сидят в матрицах. У {m8.name} MLP это {fmt.pct(a8.mlp.params / a8.layerParams)} параметров блока, attention{' '}
          {fmt.pct(a8.attn.params / a8.layerParams)}, а обе нормализации вместе всего {fmt.int(2 * a8.norm.params)}{' '}
          {fmt.plural(2 * a8.norm.params, ['число', 'числа', 'чисел'])}.
        </li>
        <li>
          От длины контекста зависит только attention: каждый новый токен сравнивается со всеми предыдущими. Стоимость остальных частей на
          токен постоянная.
        </li>
      </ul>

      <Callout kind="precise" title="откуда «2 FLOP на параметр»">
        <p>
          FLOP это одна операция с плавающей точкой: сложение или умножение. Умножить вектор на матрицу d_in × d_out значит для каждого из
          d_out выходов сложить d_in произведений. Каждый вес участвует в одном умножении и одном сложении, отсюда 2 FLOP на параметр на
          токен.
        </p>
        <p>
          Поэтому FLOPs на токен ≈ 2 × параметры матриц + attention по контексту. У {m8.name} матрицы дают {fmt.flops(matmul8)} на токен, а
          attention добавляет по {fmt.flops(m8.layers * a8.attn.flopsPerCtx)} на каждый токен контекста.
        </p>
      </Callout>

      <h3>Поток, в который пишут все блоки</h3>
      <p>
        Самая полезная картинка трансформера такая. У каждого токена есть поток, <Term t="residual stream" />: вектор длиной d_model, который
        идёт через все блоки по очереди. Каждый блок читает поток через RMSNorm и прибавляет к нему свой результат. Никто не стирает
        написанное до него.
      </p>
      <Callout kind="intuition" title="общий черновик">
        Представьте документ, который передают по цепочке из {m8.layers} редакторов. Каждый читает текущую версию и дописывает свои пометки на
        полях, ничего не удаляя. Последний редактор видит вклад всех, а исходный текст никуда не делся.
      </Callout>
      <p>Как меняется вектор от слоя к слою и когда в нём появляется ответ?</p>

      <ResidualFigure arch={arch} />

      <p>
        Размер вектора не меняется ни в одном блоке, меняется содержимое. Если приложить финальный RMSNorm и LM head к потоку посередине стопки
        (этот приём называют <Term t="Logit lens">logit lens</Term>), видно, что ранние слои ещё предлагают частые служебные слова, а конкретный ответ проявляется ближе к концу. В
        настоящих моделях картина та же, только шумнее.
      </p>

      <h3>От ID до логитов</h3>
      <p>
        Соберём всё вместе: ID токенов, таблица эмбеддингов, {m8.layers} блоков, финальный RMSNorm и <Term t="LM head" />. LM head это матрица{' '}
        {fmt.int(m8.hidden)} × {fmt.int(m8.vocab)}: она превращает последний вектор в {fmt.int(m8.vocab)} чисел, по одному на каждую строку
        словаря. Эти числа называют <Term t="логиты">логитами</Term>, из них выбирается следующий токен (
        <ChapterLink to="sampling" />
        ).
      </p>
      <p>Где в этой цепочке лежат параметры и куда уходят вычисления на каждый токен?</p>

      <ModelFigure arch={arch} ctx={ctx} onCtx={setCtx} modelId={modelId} onModel={setPicked} />

      <Callout kind="precise" title="LM head в prefill">
        <p>
          Следующий токен выбирается только по логитам последней позиции. Поэтому в <Term t="prefill" /> движки умножают на LM head один
          вектор, а не все
          N векторов промпта (если клиент не попросил logprobs для промпта). У {m8.name} на промпте из {fmt.int(2000)} токенов это экономит
          около {fmt.flops(1999 * a8.head.flopsFixed)}.
        </p>
      </Callout>

      <Callout kind="production" title="что это значит для железа">
        <p>
          Один шаг <Term t="decode" /> {m8.name} при одном запросе (batch 1) это ≈ {fmt.flops(matmul8)} вычислений, но для них нужно прочитать
          из памяти все {fmt.bytes(weightBytes(m8, 'bf16'))} весов в BF16. Получается около {fmt.num(matmul8 / weightBytes(m8, 'bf16'), 1)} FLOP
          на байт, а {h100.name} за время чтения одного байта успевает около {fmt.int(ridge)} FLOP. Поэтому decode упирается в скорость памяти,
          а не в вычисления (<ChapterLink to="roofline" />
          ).
        </p>
        <p>
          Блоки идут строго по очереди: блок 2 не начнётся, пока не закончил блок 1. Параллельна работа внутри каждого умножения на матрицу:
          даже для одного токена GPU считает тысячи её выходов одновременно, а векторы всех токенов прохода умножаются на неё за один раз.
          Один <Term t="forward pass" /> это все {m8.layers} блоков, и каждый шаг decode делает такой проход ради одного нового токена на
          последовательность.
        </p>
      </Callout>
      <SimulatorLink stage="prefill" />

      <h3>Mixture of Experts: экспертов много, работают немногие</h3>
      <p>
        В плотной (dense) модели каждый токен проходит через весь MLP каждого блока. В <Term t="MoE">MoE-модели</Term> MLP заменён набором
        маленьких экспертов и крошечным роутером. Роутер для каждого токена в каждом слое выбирает несколько экспертов, а остальные для этого
        токена не считаются.
      </p>
      <p>
        У {moe.name} {moe.experts?.total} экспертов в каждом из {moe.layers} слоёв, и токен идёт через {moe.experts?.active}. Отсюда и имя: около
        30B параметров всего и около 3B <Term t="активные параметры">активных</Term> на токен (A3B). Как роутер распределяет токены вашего
        запроса?
      </p>

      <MoeFigure />

      <p>
        Зачем это нужно: модель хранит знания {fmt.num(moe.paramsB, 1)} млрд параметров, а на каждый токен считает как модель на{' '}
        {fmt.num(moe.activeParamsB, 1)} млрд, поэтому отвечает заметно быстрее плотной модели похожего качества. Цена в памяти: все эксперты
        должны лежать на GPU, потому что следующий токен может выбрать любого. И чем больше токенов в шаге, тем больше разных экспертов
        задето: при 64 токенах в среднем {fmt.num(touched64, 1)}{' '}
        из {moe.experts?.total} ({fmt.pct(touched64 / (moe.experts?.total ?? 128))}), если выбор равномерный. Под нагрузкой decode читает почти
        все веса.
      </p>

      <Callout kind="pitfall" title="A3B не значит «модель на 3 млрд»">
        Активные параметры определяют FLOPs на один токен, но не память. А при большом <Term t="batch" /> разные токены зовут разных
        экспертов, поэтому
        выигрыш по скорости decode заметно меньше, чем отношение полных параметров к активным.
      </Callout>

      <Callout kind="production" title="MoE на нескольких GPU">
        <p>
          {big.name} в BF16 весит {fmt.bytes(weightBytes(big, 'bf16'))}: только под веса нужно минимум 6 GPU по 80 ГБ. Поэтому большие MoE
          раскладывают по GPU целыми экспертами: это <Term t="Expert parallelism (EP)">expert parallelism</Term>, например по 16 экспертов на
          каждую из 8 GPU (<ChapterLink to="parallelism" />
          ). На каждом MoE-слое токены пересылаются к GPU своих экспертов и обратно.
        </p>
      </Callout>

      <Deeper title="чем современный блок отличается от трансформера 2017 года">
        <ul>
          <li>
            В оригинальной статье нормализация стояла после сложения (post-norm), позиции прибавлялись к эмбеддингу, а MLP состоял из двух
            матриц с ReLU. Современные decoder-only модели ставят RMSNorm перед каждой частью (pre-norm), поворачивают Q и K через RoPE внутри
            attention, используют <Term t="SwiGLU" /> из трёх матриц и <Term t="GQA" />.
          </li>
          <li>
            Qwen3 добавляет QK-norm: перед RoPE каждая голова Q и K нормируется RMSNorm по своим {fmt.int(m8.headDim)} числам, а вектор весов у
            Q и у K один на все головы. Это 2 × {fmt.int(m8.headDim)} чисел на слой.
          </li>
          <li>
            gpt-oss чередует слои с полным attention и слои со скользящим окном на 128 токенов, а DeepSeek-V3 сжимает K и V в один латентный
            вектор (<Term t="MLA" />) и держит общего эксперта рядом с маршрутизируемыми.
          </li>
          <li>
            Порог, после которого attention по контексту дороже всех матриц, зависит от формы модели: у {m8.name} это ≈{' '}
            {fmt.compact(attentionCrossover(a8))} токенов, у Llama-3.1-8B ≈{' '}
            {fmt.compact(attentionCrossover(computeArch(getModel('llama-3.1-8b'))))} токенов. На коротких запросах основная работа в MLP.
          </li>
        </ul>
      </Deeper>

      <Takeaways
        items={[
          <>
            Модель это таблица эмбеддингов, N одинаковых по устройству блоков, финальный RMSNorm и LM head. У {m8.name} {m8.layers} блоков, на
            выходе {fmt.int(m8.vocab)} логитов.
          </>,
          <>
            Attention смешивает информацию между токенами, MLP перерабатывает каждый токен отдельно. Оба прибавляют поправку к residual stream,
            размер вектора не меняется.
          </>,
          <>
            FLOPs на токен ≈ 2 × параметры матриц + attention по контексту. MLP держит большую часть параметров, attention дорожает с длиной
            контекста.
          </>,
          <>MoE: в памяти все эксперты, на токен работают немногие, а под нагрузкой читаются почти все веса.</>,
        ]}
      />
    </Chapter>
  )
}
