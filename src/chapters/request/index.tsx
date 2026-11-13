import { useEffect, useMemo, useState } from 'react'
import { Callout, Chapter, ChapterLink, Deeper, SimulatorLink, Takeaways, Term } from '@/ui'
import { useSettings } from '@/state/settings'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { renderTemplate } from '@/engine/templates'
import { kvBytesPerToken } from '@/engine/perf'
import type { ModelFamily } from '@/engine/types'
import { countPrompt } from '../template/promptTokens'
import { RequestMap, type MapContext } from './RequestMap'
import { ResponseCompare } from './ResponseCompare'
import { DEFAULT_TOP_P, WEATHER_TOOL, requestMessages, type RequestOptions } from './requestModel'
import { resolveHardware, simulateConversation, simulatedAnswer } from './responseModel'

const TOP_P = DEFAULT_TOP_P
/** Как в запросе симулятора (DEFAULT_MAX_TOKENS в engine/flow.ts): глава показывает тот же запрос. */
const MAX_TOKENS = 512

/** Семейство шаблона для модели из настроек (в симуляторе есть Qwen3 и Llama 3.1). */
function templateFamily(family: string | undefined): ModelFamily {
  return family === 'llama3' ? 'llama3' : 'qwen3'
}

export default function RequestChapter() {
  const [settings] = useSettings()
  const { tokenizer } = useTokenizer()
  const [tools, setTools] = useState(settings.tools)
  const [json, setJson] = useState(settings.json)
  const [stream, setStream] = useState(true)
  const [warm, setWarm] = useState(settings.cacheWarm)
  // симулятор меняет настройки и после фонового монтирования главы: подхватываем их (переключатели в фигурах остаются
  // локальными до следующей правки в симуляторе), иначе глава 1 расходится с главами 2 и 3 в числе токенов
  useEffect(() => setTools(settings.tools), [settings.tools])
  useEffect(() => setJson(settings.json), [settings.json])
  useEffect(() => setWarm(settings.cacheWarm), [settings.cacheWarm])

  const hw = useMemo(() => resolveHardware(settings.modelId, settings.gpuId), [settings.modelId, settings.gpuId])
  const family = templateFamily((hw.model as { family?: string }).family)
  const modelLabel = (hw.model as { hfId?: string }).hfId ?? hw.model.name

  const opts: RequestOptions = useMemo(
    () => ({
      model: modelLabel,
      systemText: settings.systemText,
      userText: settings.userText,
      temperature: settings.temperature,
      topP: TOP_P,
      maxTokens: MAX_TOKENS,
      tools,
      json,
      stream,
      thinking: settings.thinking,
    }),
    [modelLabel, settings.systemText, settings.userText, settings.temperature, settings.thinking, tools, json, stream],
  )

  // токены промпта после шаблона: с инструментом и без (разница = цена описания tools)
  const counts = useMemo(() => {
    if (!tokenizer) return null
    const messages = requestMessages(opts)
    const base = renderTemplate({ family, messages, enableThinking: settings.thinking })
    const withTools = renderTemplate({ family, messages, enableThinking: settings.thinking, tools: [WEATHER_TOOL] })
    const a = countPrompt(family, base.segments, tokenizer, null)
    const b = countPrompt(family, withTools.segments, tokenizer, null)
    if (!a || !b) return null
    return { base: a.total, withTools: b.total }
  }, [tokenizer, opts, family, settings.thinking])

  const promptTokens = counts ? (tools ? counts.withTools : counts.base) : null

  const ctx: MapContext = {
    modelName: hw.model.name,
    promptTokens,
    toolTokens: counts ? counts.withTools - counts.base : null,
    toolsOn: tools,
    kvPerToken: kvBytesPerToken(hw.model),
    maxContext: hw.model.maxContext,
    maxTokens: MAX_TOKENS,
  }

  const sim = useMemo(() => {
    if (!tokenizer || promptTokens === null) return null
    return simulateConversation({
      tokenizer,
      promptTokens,
      warm,
      answer: simulatedAnswer(settings.userText, settings.thinking, json),
      modelId: settings.modelId,
      gpuId: settings.gpuId,
      modelLabel,
    })
  }, [tokenizer, promptTokens, warm, settings.userText, settings.thinking, json, settings.modelId, settings.gpuId, modelLabel])

  return (
    <Chapter
      id="request"
      lead={
        <>
          Всё начинается с одного HTTP-запроса: клиент отправляет POST на /v1/chat/completions с JSON внутри. Формат придумала
          OpenAI, и на нём же говорят vLLM, SGLang и NVIDIA Dynamo. По полям этого JSON можно прочитать карту всей системы:
          каждое поле включает свой механизм на сервере.
        </>
      }
    >
      <p>
        Когда вы нажимаете Enter в чате, клиент (браузер, приложение, агент) упаковывает диалог в JSON и отправляет его на сервер
        инференса. Такой формат называют <Term t="OpenAI-compatible API">OpenAI-совместимым API</Term>: его понимают облачные API,
        открытые движки <Term t="vLLM" />, <Term t="SGLang" /> и TensorRT-LLM, а также оркестратор{' '}
        <Term t="NVIDIA Dynamo">NVIDIA Dynamo</Term>. Поэтому стандартным клиентом стал OpenAI SDK: чтобы переехать с одного движка
        на другой, достаточно поменять адрес сервера, код приложения остаётся прежним.
      </p>
      <p>
        Поля запроса делятся на пять групп по назначению. <code>model</code> называет модель. <code>messages</code> несёт диалог:
        список реплик с ролями system, user, assistant и tool. <code>temperature</code>, <code>top_p</code>,{' '}
        <code>max_tokens</code> и <code>stop</code> управляют генерацией. <code>stream</code> и <code>response_format</code>{' '}
        задают форму ответа. <code>tools</code> описывает функции, которые модель может попросить вызвать.
      </p>

      <h3>Путь запроса по стадиям</h3>
      <p>
        Названия стадий здесь и на схеме ниже взяты из Dynamo. Там preprocessor, router и postprocessor работают звеньями одного
        конвейера внутри процесса frontend (CPU, код на Rust), а prefill и decode считают воркеры на GPU. В одиночном vLLM те же
        шаги делают процессы одной машины: frontend, preprocessor и postprocessor живут в процессе API-сервера, планировщик и
        GPU-воркеры в своих процессах. Порядок везде один и тот же:
      </p>
      <ol>
        <li>
          <b>
            <Term t="Dynamo Frontend">Frontend</Term>
          </b>{' '}
          принимает HTTP, проверяет JSON и находит модель.
        </li>
        <li>
          <b>
            <Term t="Preprocessor" />
          </b>{' '}
          склеивает <code>messages</code> <Term t="Chat template">шаблоном чата</Term> в одну строку и режет её на токены. Этому
          посвящены две следующие главы: <ChapterLink to="template" /> и <ChapterLink to="tokenization" />.
        </li>
        <li>
          <b>
            <Term t="router">Router</Term>
          </b>{' '}
          выбирает воркер. По умолчанию Dynamo раздаёт запросы <Term t="Round-robin">по кругу</Term>, а KV-aware роутер (флаг{' '}
          <code className="whitespace-nowrap">--router-mode kv</code>) отправляет запрос туда, где уже лежит{' '}
          <Term t="KV cache" /> такого же начала промпта, с учётом нагрузки воркеров (<ChapterLink to="routing" />).
        </li>
        <li>
          <b>
            <Term t="Prefill" />
          </b>{' '}
          на GPU: весь промпт за один проход, KV cache заполняется, появляется первый токен (<ChapterLink to="prefill-decode" />).
        </li>
        <li>
          <b>
            <Term t="KV transfer">Передача KV</Term>
          </b>{' '}
          (только в дезагрегированном режиме): готовый KV cache едет с prefill-воркера на decode-воркер через{' '}
          <Term t="NIXL" /> (<ChapterLink to="disaggregation" />).
        </li>
        <li>
          <b>
            <Term t="Decode" />
          </b>
          : по одному токену за шаг. На каждом шаге модель выдаёт <Term t="Logits">логиты</Term>, оценки всех токенов словаря.
          Если задан <code>response_format</code>, маска <Term t="Structured output">structured output</Term> сначала запрещает
          токены, которые сломали бы JSON. Затем <Term t="Sampling">сэмплер</Term> выбирает следующий токен (
          <ChapterLink to="sampling" />, <ChapterLink to="structured-output" />).
        </li>
        <li>
          <b>
            <Term t="Postprocessor" />
          </b>{' '}
          превращает токены обратно в текст, ищет стоп-строки и отделяет рассуждение и вызовы инструментов от ответа (
          <ChapterLink to="detokenization" />, <ChapterLink to="parsers" />).
        </li>
        <li>
          <b>Ответ</b> уходит клиенту одним JSON или потоком <Term t="Streaming (SSE)">SSE</Term>.
        </li>
      </ol>

      <p>
        Главное, что стоит понять сразу: поля читают разные части сервера, и далеко не всегда там, где запрос приняли. Ниже ваш
        собственный запрос, system и user взяты из симулятора в начале страницы. Какая стадия читает каждое поле?
      </p>

      <RequestMap
        opts={opts}
        ctx={ctx}
        onToggle={(p) => {
          if (p.tools !== undefined) setTools(p.tools)
          if (p.json !== undefined) setJson(p.json)
          if (p.stream !== undefined) setStream(p.stream)
        }}
      />

      <p>
        Теперь о том, где поля читают. <code>messages</code> и <code>tools</code> превращаются в текст и токены ещё до GPU, на
        CPU. <code>temperature</code>, <code>top_p</code> и <code>response_format</code>, наоборот, доживают до самого конца: их
        применяют на каждом шаге decode, для каждого нового токена. А <code>stop</code> проверяется уже после GPU, на готовом
        тексте. Если какое-то поле «не работает», первым делом выясните, какая стадия за него отвечает.
      </p>
      <SimulatorLink stage="client" />

      <h3>Ответ: целиком или потоком</h3>
      <p>
        Сервер отвечает в одном из двух режимов. С <code>stream: false</code> он копит ответ и присылает один JSON{' '}
        <code>chat.completion</code>: текст в <code>choices[0].message</code>, причина остановки в <code>finish_reason</code>,
        расход токенов в <code>usage</code>. С <code>stream: true</code> ответ идёт потоком Server-Sent Events: строки{' '}
        <code>{'data: {…}'}</code> с маленькими кусочками текста (<code>delta</code>), которые клиент склеивает сам, а в конце
        строка <code>data: [DONE]</code>.
      </p>
      <p>
        На GPU оба режима работают одинаково. Разница в том, что видит человек перед экраном: со стримом первые слова приходят
        через <Term t="TTFT" /> (time to first token, время до первого токена), без стрима только вместе с последним. Переключатель
        «Повторный запрос» отправляет тот же запрос второй раз: начало промпта уже лежит в{' '}
        <Term t="Prefix caching">prefix cache</Term>, и сервер не считает его заново. Что человек видит в каждый момент?
      </p>

      <ResponseCompare sim={sim} warm={warm} onWarm={setWarm} />

      <p>
        Стрим не ускоряет генерацию: последний токен приходит в то же время. Зато первые слова появляются через TTFT, а не через
        полное время ответа. Для чата это разница между «отвечает сразу» и «зависло».
      </p>
      <p>
        Посмотрите и на <Term t="finish_reason / usage">usage</Term>: <code>prompt_tokens</code> и <code>completion_tokens</code>{' '}
        идут в счёт за API, а <code>prompt_tokens_details.cached_tokens</code> показывает, сколько токенов промпта сервер взял из
        prefix cache, вместо того чтобы считать заново. vLLM по умолчанию кэширует только целые блоки по 16 токенов, поэтому
        здесь число кратно 16. Собственный OpenAI-сервер vLLM (<code>vllm serve</code>) отдаёт это поле только с флагом{' '}
        <code className="whitespace-nowrap">--enable-prompt-tokens-details</code>, сервер SGLang только с{' '}
        <code className="whitespace-nowrap">--enable-cache-report</code>: кэш работает и без них, но в usage о попаданиях не
        пишут.
      </p>
      <p>
        Заметьте и пропорции на шкале времени. У короткого промпта prefill занимает миллисекунды: GPU всё равно читает все веса
        модели, и несколько десятков токенов почти ничего к этому не добавляют. Почти всё время уходит на decode, токен за токеном.
        Поэтому повторный запрос с prefix cache здесь экономит вычисления, но не время. На промптах в тысячи токенов картина
        обратная: prefill становится главной частью TTFT, и кэш сокращает его заметно. Почему так, объясняет глава{' '}
        <ChapterLink to="roofline" />.
      </p>
      <p>
        Рассуждение модели здесь приходит отдельным полем <code>reasoning_content</code>: так делают SGLang и Dynamo. В vLLM с
        конца 2025 года это поле называется <code>reasoning</code>. В стандарте OpenAI Chat Completions его нет вовсе, поэтому
        клиенту безопаснее читать оба названия.
      </p>
      <SimulatorLink stage="stream" />

      <Callout kind="pitfall" title="usage и finish_reason в стриме">
        <p>
          В стриме <code>usage</code> приходит, только если в запросе есть <code>stream_options.include_usage</code>, и только в
          последнем чанке с пустым <code>choices</code>. Оборвалось соединение раньше: счётчика токенов у клиента не будет. А{' '}
          <code>finish_reason: "stop"</code> ещё не значит, что модель закончила сама: так же выглядит и сработавшая стоп-строка.
        </p>
      </Callout>

      <Callout kind="production" title="таймауты, ретраи, стриминг">
        <ul>
          <li>
            <b>Таймауты.</b> Синхронный запрос обычно обрывается через несколько минут, а прокси и балансировщики закрывают
            соединения, по которым долго не идут байты. Длинное рассуждение без стрима легко упирается в такой лимит. Со стримом
            байты идут с первого токена, а на клиенте разумно ограничивать паузу между чанками, а не весь ответ.
          </li>
          <li>
            <b>Ретраи.</b> Повторять стоит ответы о перегрузке (HTTP 429 и 503, Dynamo отдаёт 529) с растущей паузой и случайным
            разбросом. Каждый повтор заново проходит prefill и снова тратит GPU, prefix cache сэкономит только совпавшее начало.
            Если воркер упал посреди генерации, Dynamo умеет перенести запрос на другой (migration, флаг{' '}
            <code className="whitespace-nowrap">--migration-limit</code>, по умолчанию выключено).
          </li>
          <li>
            <b>Отмена.</b> Пользователь закрыл вкладку: сервер должен прервать генерацию, иначе GPU продолжит считать ответ, который
            никто не прочтёт, и держать под него KV cache.
          </li>
          <li>
            <b>Соединения.</b> TLS-рукопожатие стоит десятки миллисекунд. Переиспользуйте соединения между запросами, OpenAI SDK
            делает это сам.
          </li>
          <li>
            <b>Стриминг</b> улучшает воспринимаемую задержку: человек видит текст через TTFT. По моментам прихода чанков снимают и
            сами метрики TTFT и ITL (<ChapterLink to="metrics" />).
          </li>
        </ul>
      </Callout>

      <Deeper title="что ещё бывает в запросе">
        <ul>
          <li>
            <code>n</code> (несколько ответов на один промпт), <code>seed</code> (повторяемый сэмплинг), <code>logprobs</code> и{' '}
            <code>top_logprobs</code> (вероятности выбранных токенов), <code>presence_penalty</code> и{' '}
            <code>frequency_penalty</code> (штрафы за повторы).
          </li>
          <li>
            <code>tool_choice</code>: <code>"auto"</code> (модель решает сама), <code>"none"</code>, <code>"required"</code> или
            конкретная функция. При <code>"auto"</code> корректность вызова ничем не гарантирована: аргументы могут оказаться
            битым JSON. С <code>"required"</code> или конкретной функцией движок может ограничить ответ грамматикой вызова.
          </li>
          <li>
            <code>max_completion_tokens</code>: новое имя <code>max_tokens</code> в API OpenAI. vLLM понимает оба.
          </li>
          <li>
            Расширения движков. vLLM принимает в том же JSON <code>top_k</code>, <code>min_p</code>,{' '}
            <code>repetition_penalty</code> и <code>chat_template_kwargs</code>. Dynamo принимает блок <code>nvext</code>: например,{' '}
            <code>cache_salt</code> изолирует prefix cache разных арендаторов, а <code>backend_instance_id</code> отправляет запрос
            на конкретный воркер в обход роутера.
          </li>
        </ul>
      </Deeper>

      <Takeaways
        items={[
          <>
            Запрос к LLM это JSON по OpenAI-совместимому API: <code>POST /v1/chat/completions</code>. vLLM, SGLang и Dynamo
            понимают один и тот же формат.
          </>,
          <>
            Поля читают разные стадии: <code>messages</code> и <code>tools</code> превращаются в токены до GPU, параметры
            сэмплинга и <code>response_format</code> работают на каждом шаге decode, <code>stop</code> проверяется на тексте после
            GPU.
          </>,
          <>
            <code>stream</code> не ускоряет генерацию, но показывает первые слова через TTFT вместо полного времени ответа.
          </>,
          <>
            <code>usage</code> считает токены промпта и ответа: это деньги, место в контексте и память KV cache.
          </>,
        ]}
      />
    </Chapter>
  )
}
