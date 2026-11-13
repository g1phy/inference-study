import { useMemo } from 'react'
import { useSettings } from '@/state/settings'
import { findGpu, getGpu } from '@/engine/data/gpus'
import { findModel, getModel } from '@/engine/data/models'
import { kvCapacityTokens } from '@/engine/perf'
import { DEFAULT_BLOCK_SIZE } from '@/engine/kv'
import { Callout, Chapter, ChapterLink, CodeBlock, Deeper, SimulatorLink, Takeaways, Term } from '@/ui'
import { EngineLoopFigure } from './EngineLoopFigure'
import { ProcessFigure } from './ProcessFigure'
import { CompareFigure } from './CompareFigure'
import { MemoryFractionFigure } from './MemoryFractionFigure'
import { WhoIsWhoFigure } from './WhoIsWhoFigure'
import { CudaGraphFigure } from './CudaGraphFigure'

/**
 * Строка лога старта vLLM с оценкой ёмкости KV для модели и GPU из настроек (perf.kvCapacityTokens, доля 0,92).
 * Как в vLLM V1 (kv_cache_utils.get_kv_cache_capacity): ёмкость целыми блоками по 16 токенов, конкурентность для
 * max_model_len (по умолчанию полный контекст модели, у Qwen3-8B 40 960); с v0.28 обе величины в одной строке.
 * Те же числа показывают калькулятор главы «KV cache» и глава «PagedAttention» (407 904 токена у Qwen3-8B на H100).
 */
function StartupLog() {
  const [settings] = useSettings()
  const model = findModel(settings.modelId) ?? getModel('qwen3-8b')
  const gpu = findGpu(settings.gpuId) ?? getGpu('h100-sxm')
  const code = useMemo(() => {
    const blocks = Math.floor(Math.max(0, kvCapacityTokens(model, gpu, { memUtil: 0.92 })) / DEFAULT_BLOCK_SIZE)
    const tokens = blocks * DEFAULT_BLOCK_SIZE
    const conc = blocks / Math.ceil(model.maxContext / DEFAULT_BLOCK_SIZE)
    const en = (n: number) => n.toLocaleString('en-US')
    return [
      `# vllm serve ${model.hfId ?? model.name} на ${gpu.name} (оценка perf.ts, не реальный лог)`,
      `GPU KV cache size: ${en(tokens)} tokens, Maximum concurrency for ${en(model.maxContext)} tokens per request: ${conc.toFixed(2)}x`,
      '',
      '# метрики Prometheus, на которые смотрят первыми',
      'vllm:num_requests_running, vllm:num_requests_waiting',
      'vllm:kv_cache_usage_perc',
      'vllm:num_preemptions',
      'vllm:prefix_cache_hits, vllm:prefix_cache_queries',
    ].join('\n')
  }, [model, gpu])
  // строка лога длинная (с v0.28 ёмкость и concurrency в одной строке): перенос вместо прокрутки, число concurrency видно сразу
  return <CodeBlock lang="bash" code={code} wrap />
}

export default function EnginesChapter() {
  return (
    <Chapter
      id="engines"
      lead={
        <>
          Движок крутит одну реплику модели. На каждом шаге он решает, какие токены каких запросов прогнать через GPU, где держать их KV cache, какими
          ядрами считать и как выбрать следующий токен. vLLM, SGLang и TensorRT-LLM делают одно и то же, но с разными акцентами.
        </>
      }
    >
      <h3>Что делает движок</h3>
      <p>
        Предыдущие главы разобрали механизмы по отдельности: prefill и decode, KV cache и страницы, prefix caching, continuous batching, параллелизм.{' '}
        <Term t="Inference engine">Движок инференса</Term> собирает их в одну работающую программу для одной реплики модели на её GPU.
      </p>
      <p>
        Если смотреть на стек снизу вверх, внизу CUDA и библиотеки ядер, над ними PyTorch, над ним движки, а ещё выше распределённый слой вроде{' '}
        <Term t="NVIDIA Dynamo">Dynamo</Term> (глава <ChapterLink to="dynamo" />). Инженеру инференса свои ядра писать почти не приходится: их пишут авторы библиотек и движков, а движок
        выбирает подходящие под модель и GPU и склеивает их в цикл, который не даёт GPU простаивать.
      </p>
      <p>
        В этом цикле четыре главные роли. <Term t="Engine scheduler">Scheduler</Term> решает, сколько токенов какого запроса посчитать на этом шаге. KV cache
        manager раздаёт блоки памяти. GPU-воркеры выполняют forward. Sampler выбирает следующий токен. Вокруг цикла живут HTTP-сервер, шаблон и
        токенизация на входе, детокенизация и парсеры на выходе.
      </p>

      <h3>Внутри vLLM</h3>
      <p>
        Разберём устройство на <Term t="vLLM" />, самом распространённом движке. В версии V1 он разнесён по трём видам процессов: API server на CPU,{' '}
        <Term t="EngineCore" /> с бесконечным циклом планирования и по одному процессу-воркеру на каждый GPU. Процессы общаются через <Term t="ZMQ">ZMQ</Term> (ZeroMQ, библиотека
        быстрых сообщений между процессами). Ниже ваш запрос проходит через них по шагам.
      </p>

      <EngineLoopFigure />

      <p>
        Обратите внимание на три вещи. Вся CPU-работа вокруг модели (HTTP, токенизация, детокенизация, парсеры) вынесена из процесса, который кормит GPU:
        шаг decode маленькой модели длится миллисекунды, и любая задержка на CPU сразу видна в скорости. У scheduler нет режимов «prefill» и «decode»: в
        одном шаге ваш промпт считается вместе с токенами чужих ответов. И состав batch меняется на каждом шаге: это <Term t="Continuous batching">continuous batching</Term>.
      </p>
      <Callout kind="precise" title="CPU и GPU работают внахлёст">
        Пока GPU считает шаг N, scheduler уже планирует шаг N + 1 и подставляет заглушки вместо токенов, которые ещё не готовы. В vLLM это <Term t="Async scheduling">async scheduling</Term>{' '}
        (по умолчанию с v0.14), в SGLang zero-overhead scheduler, в TensorRT-LLM overlap scheduler, который стоит одного лишнего шага decode.
      </Callout>

      <h3>Откуда скорость на GPU: CUDA graphs и ядра</h3>
      <p>
        Шаг decode состоит из сотен маленьких <Term t="CUDA kernel">ядер</Term>. Если CPU запускает их по одному, GPU простаивает между запусками.{' '}
        <Term t="CUDA graphs">CUDA graph</Term> решает это: последовательность ядер один раз записывают, а потом запускают одной командой. Граф записан под
        конкретную форму тензоров, поэтому движок захватывает графы для набора размеров batch и дополняет реальный batch до ближайшего. За это платят памятью
        GPU и временем на старте.
      </p>
      <CudaGraphFigure />
      <p>
        Порядок величин (условный расчёт): шаг из 400 ядер при 5 мкс на запуск тратит на одни запуски 2 мс. У модели на 1 млрд параметров весь шаг decode
        на H100 около 0,6 мс, так что без графа GPU большую часть времени ждёт CPU, а с графом шаг примерно втрое быстрее. У модели на 8 млрд параметров
        шаг около 5 мс, запуски почти прячутся за работой GPU, и выигрыш скромнее.
      </p>
      <p>
        Attention считают специализированные ядра: <Term t="FlashAttention" /> или <Term t="FlashInfer" />. Они читают KV прямо из разбросанных блоков по таблице блоков (
        <ChapterLink to="paged-attention" />) и не пишут в видеопамять промежуточную матрицу весов attention.
      </p>
      <Callout kind="pitfall" title="PagedAttention не делает attention быстрее">
        Это прежде всего способ управлять памятью. В исходной статье само ядро PagedAttention было на 20–26 % медленнее ядра FasterTransformer, а выигрыш в
        2–4 раза по throughput дало то, что в ту же память влезает больше запросов. За скорость самого attention отвечают другие ядра: FlashAttention и его наследники.
      </Callout>

      <h3>Сколько процессов на самом деле</h3>
      <p>
        Одна реплика vLLM на нескольких GPU состоит из нескольких процессов. При <Term t="Tensor parallelism (TP)">tensor parallel</Term> каждый GPU держит долю каждого
        слоя, при <Term t="Pipeline parallelism (PP)">pipeline parallel</Term> свою группу слоёв, при <Term t="Data parallelism (DP)">data parallel</Term>{' '}
        реплик несколько, и у каждой свой EngineCore (глава <ChapterLink to="parallelism" />).
      </p>

      <ProcessFigure />

      <h3>vLLM, SGLang, TensorRT-LLM</h3>
      <p>
        Основа у всех трёх одна: continuous batching, KV cache блоками, prefix caching, квантизация, спекулятивный декодинг, параллелизм и дезагрегация
        есть у каждого. Различаются главная идея, значения по умолчанию и философия: насколько смело пускать запросы, как хранить кэш, сколько оставить
        ручной настройки. Книга оценивает так: у vLLM и <Term t="SGLang" /> хорошая скорость и простой старт, у <Term t="TensorRT-LLM" /> лучшая
        скорость, но больше работы и только железо NVIDIA. Baseten, где работает автор книги, использует все три и выбирает под каждый деплой.
      </p>

      <CompareFigure />

      <p>
        Ещё одна разница, которая часто стоит денег: одно и то же число в конфиге значит в разных движках разное. Доля памяти «0,9» в vLLM считается от всей
        видеопамяти, в TensorRT-LLM от того, что осталось после весов.
      </p>

      <MemoryFractionFigure />

      <Callout kind="production" title="что смотреть после запуска">
        <p>
          Первое, что стоит проверить в логе старта: сколько токенов KV поместилось и сколько запросов полной длины это даёт. Дальше метрики: очередь и
          число бегущих запросов, заполненность KV, вытеснения (каждое означает пересчёт prefill) и попадания в prefix cache.
        </p>
        <StartupLog />
      </Callout>

      <h3>Кто есть кто: движки и слой над ними</h3>
      <p>
        Все три движка обслуживают одну реплику модели. Когда реплик много, над ними появляется ещё один слой: он выбирает реплику для каждого запроса,
        переносит KV между машинами и решает, сколько реплик держать. Так устроен NVIDIA Dynamo, о нём следующая глава.
      </p>

      <WhoIsWhoFigure />

      <p>
        vLLM, SGLang и TensorRT-LLM конкурируют между собой. Dynamo с ними не конкурирует: он запускает любой из них воркером и забирает всё, что вокруг
        GPU. Его соседи по слою: vLLM Router, <Term t="llm-d" /> и <Term t="SGLang Model Gateway" />.
      </p>

      <Deeper title="три философии допуска запросов под KV cache">
        <p>
          Движок не знает заранее, сколько токенов сгенерирует запрос. vLLM пускает смело: блоки выдаются по ходу генерации, а если места на очередной токен
          нет, последний допущенный запрос вытесняется, его KV стирается, и потом он пересчитывается целиком (в V1 только recompute, swap в RAM убран).
          SGLang пускает с прогнозом: под каждый бегущий запрос резервируется часть оставшихся токенов, коэффициент начинается с 0,7 и со временем снижается
          до 0,098, а при нехватке памяти часть decode-запросов отзывается в очередь. TensorRT-LLM по умолчанию (<code>GUARANTEED_NO_EVICT</code>) пускает,
          только если KV хватит до самого <code>max_tokens</code>: вытеснений не бывает, зато память недогружена бронью, которая может не понадобиться.
        </p>
      </Deeper>
      <Deeper title="одни и те же ручки под разными именами">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] border-collapse text-[14px]">
            <thead>
              <tr className="border-b border-line text-left text-[13px] text-ink-3">
                <th className="py-2 pr-3 font-normal">Что</th>
                <th className="px-3 py-2 font-normal">vLLM</th>
                <th className="px-3 py-2 font-normal">SGLang</th>
                <th className="px-3 py-2 font-normal">TensorRT-LLM</th>
              </tr>
            </thead>
            <tbody className="font-mono text-[12.5px] text-ink">
              <tr className="border-b border-line">
                <td className="py-2 pr-3 font-sans text-[14px] text-ink-2">токенов в шаге</td>
                <td className="px-3 py-2">max_num_batched_tokens</td>
                <td className="px-3 py-2">chunked_prefill_size, max_prefill_tokens</td>
                <td className="px-3 py-2">max_num_tokens</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-3 font-sans text-[14px] text-ink-2">запросов сразу</td>
                <td className="px-3 py-2">max_num_seqs</td>
                <td className="px-3 py-2">max_running_requests</td>
                <td className="px-3 py-2">max_batch_size</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-3 font-sans text-[14px] text-ink-2">токенов в блоке KV</td>
                <td className="px-3 py-2">block_size = 16</td>
                <td className="px-3 py-2">page_size = 1</td>
                <td className="px-3 py-2">tokens_per_block = 32</td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-2 pr-3 font-sans text-[14px] text-ink-2">память под KV</td>
                <td className="px-3 py-2">gpu_memory_utilization</td>
                <td className="px-3 py-2">mem_fraction_static</td>
                <td className="px-3 py-2">free_gpu_memory_fraction</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Deeper>
      <Deeper title="где живут детокенизация и парсеры">
        <p>
          В vLLM детокенизация, стоп-строки и парсеры работают в процессе API server, а стоп-токены и <code>max_tokens</code> проверяет EngineCore. В SGLang
          детокенизацией занят отдельный процесс DetokenizerManager. В TensorRT-LLM это делает слой LLM API, а в классической схеме с Triton отдельные модели
          preprocessing и postprocessing вокруг движка. Dynamo выносит то же самое во frontend на Rust: движок получает и отдаёт только token IDs.
        </p>
      </Deeper>
      <Deeper title="TensorRT-LLM и номер версии">
        <p>
          Версии 0.x были плагином к TensorRT: модель заранее компилировали командой <code>trtllm-build</code> в движок под конкретную GPU и лимиты. С версии
          1.0 (2025) по умолчанию работает PyTorch-рантайм, чекпоинты Hugging Face грузятся напрямую, а запуск выглядит как у соседей:{' '}
          <code>trtllm-serve модель --config config.yaml</code>. В ветке 1.4 старый TensorRT-бэкенд удалён совсем. Старые инструкции со сборкой движка
          встречаются до сих пор, поэтому смотрите, к какой версии они относятся.
        </p>
      </Deeper>

      <SimulatorLink stage="prefill" />

      <Takeaways
        items={[
          'Движок крутит цикл: scheduler решает, сколько токенов каких запросов посчитать, KV cache manager раздаёт блоки, GPU-воркеры считают forward на CUDA graphs и быстрых ядрах, sampler выбирает токены.',
          'Вся CPU-работа вокруг модели вынесена в отдельные процессы и идёт внахлёст с GPU: шаг decode длится миллисекунды, и задержки CPU сразу видны.',
          'vLLM берут за ширину и простоту, SGLang за работу с префиксами и большими MoE, TensorRT-LLM за максимум на железе NVIDIA; llama.cpp и Ollama нужны для локального запуска, а не для прода.',
          'Одинаковые по смыслу параметры называются и считаются по-разному: доля памяти, бюджет токенов и размер блока зависят от движка и его версии.',
        ]}
      />
    </Chapter>
  )
}
