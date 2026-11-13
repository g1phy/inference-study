import { Block, Callout, Chapter, ChapterLink, CodeBlock, Deeper, SimulatorLink, Takeaways, Term } from '@/ui'
import { FamiliesFigure } from './FamiliesFigure'
import { ParserFigure } from './ParserFigure'
import { SCENARIOS } from './scenarios'
import { SseFigure } from './SseFigure'

const WANTED = `{
  "role": "assistant",
  "reasoning_content": "Нужен точный расчёт.",
  "content": null,
  "tool_calls": [{
    "id": "chatcmpl-tool-…",
    "type": "function",
    "function": {
      "name": "kv_cache_size",
      "arguments": "{\\"context\\": 8192}"
    }
  }]
}`

const CLIENT = `reasoning, content, calls, finish, usage = "", "", {}, None, None
for line in response.iter_lines():          # httpx; у requests: iter_lines(decode_unicode=True)
    if not line.startswith("data: "):
        continue                            # пустые строки разделяют события
    payload = line[len("data: "):]
    if payload == "[DONE]":
        break
    chunk = json.loads(payload)
    if not chunk["choices"]:                # чанк с usage
        usage = chunk.get("usage")
        continue
    choice = chunk["choices"][0]
    delta = choice["delta"]
    reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""
    content += delta.get("content") or ""
    for tc in delta.get("tool_calls") or []:
        slot = calls.setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
        slot["id"] = tc.get("id") or slot["id"]
        fn = tc.get("function") or {}
        slot["name"] += fn.get("name") or ""
        slot["arguments"] += fn.get("arguments") or ""
    finish = choice.get("finish_reason") or finish
# json.loads(slot["arguments"]) только здесь и с обработкой ошибок`

export default function ParsersChapter() {
  return (
    <Chapter
      id="parsers"
      lead={
        <>
          Модель пишет одну сплошную ленту: мысли в &lt;think&gt;, ответ, вызовы инструментов в &lt;tool_call&gt;. Клиенту нужны раздельные поля. Парсер
          на сервере читает ленту по мере генерации и раскладывает её по полям reasoning_content, content и tool_calls, прежде чем отправить клиенту.
        </>
      }
    >
      <p>
        После детокенизатора (глава <ChapterLink to="detokenization" />) у сервера есть текст ответа, который растёт по кусочку. У reasoning-модели вроде Qwen3 это
        служебные теги вперемешку с текстом (первый блок ниже). А клиенту, который работает через <Term t="Chat Completions API">OpenAI-совместимый API</Term>,
        нужен структурированный ответ (второй блок): мысли отдельно, текст отдельно, вызов функции объектом с именем и аргументами.
      </p>

      <Block width="wide">
        <div className="grid gap-4 md:grid-cols-2">
          <CodeBlock lang="text" title="что печатает модель" code={SCENARIOS[1].text} wrap />
          <CodeBlock lang="json" title="что нужно клиенту (упрощённо)" code={WANTED} />
        </div>
      </Block>

      <p>
        Эту работу делает парсер в <Term t="postprocessor" />. Модель он не меняет и токены не трогает, он только решает, в какое поле положить очередной
        кусок текста. Парсеров два вида: <Term t="reasoning parser">reasoning-парсер</Term> отделяет мысли, <Term t="tool-call parser">tool-парсер</Term>{' '}
        вынимает <Term t="function calling">вызовы инструментов</Term>. Порядок фиксирован: сначала reasoning-парсер, потом tool-парсер по тому, что осталось в
        content. Внутри рассуждения вызовы не ищутся.
      </p>

      <h3>Машина состояний</h3>
      <p>
        Парсер устроен как маленький конечный автомат. У него три состояния, по одному на поле ответа: reasoning, content и tool_call. Переходы происходят на
        тегах: &lt;think&gt; переводит в reasoning, &lt;/think&gt; обратно в content, &lt;tool_call&gt; в tool_call, &lt;/tool_call&gt; снова в content.
        Сами теги клиенту не уходят.
      </p>
      <p>
        Есть и промежуточное состояние. Если кусок кончается на «&lt;th», это может быть начало &lt;think&gt;, а может быть обычный текст. Отправить его
        нельзя (вдруг это тег), выбросить тоже (вдруг это текст), поэтому парсер придерживает его в буфере до следующего куска. Это та же идея, что jail
        для стоп-строк в детокенизаторе.
      </p>
      <Callout kind="intuition">
        Представьте сортировщика у конвейера. Текст едет сплошной лентой, а метки &lt;think&gt; и &lt;tool_call&gt; говорят, в какой ящик класть следующие
        куски. Если на ленте показался обрывок, похожий на начало метки, сортировщик придерживает его рукой, пока не увидит продолжение.
      </Callout>
      <p>Как парсер решает, куда положить очередной кусок, и что в этот момент уходит клиенту? Пройдите по шагам.</p>

      <ParserFigure />

      <p>
        В сценарии с вызовом инструмента (он открыт по умолчанию) заметьте три вещи. Теги &lt;think&gt; и &lt;/think&gt; меняют состояние, но в поля не
        попадают. Имя функции уходит клиенту одним чанком, как только парсер увидел закрывающую кавычку после <code>"name"</code>, а аргументы дальше идут
        кусками строки. Последний кусок аргументов «{'}'}» короче токена «{'}}'}↵»: парсер отрезал скобку внешнего объекта и перевод строки. В режиме
        кусков по 3 символа в полях получается тот же результат, хотя теги приходят частями. Так и должно быть: ответ парсера не может зависеть от того,
        как поток порезан на куски.
      </p>

      <Callout kind="precise" title="ловить теги по номеру токена или по тексту">
        <p>
          У Qwen3 и DeepSeek теги &lt;think&gt; и &lt;/think&gt; являются одиночными токенами (у Qwen3 это номера 151667 и 151668), поэтому reasoning-парсер
          vLLM сверяет номера токенов, и половины тега у него не бывает. Но парсер должен уметь работать и с текстом: у многих моделей тег состоит из
          нескольких токенов и приходит частями, а на вход парсеру поступают куски текста от детокенизатора, в которых бывает сразу несколько токенов
          (например, при <Term t="speculative decoding" />). SGLang копит буфер, пока он остаётся началом &lt;think&gt; или &lt;/think&gt;, а{' '}
          <Term t="Hermes (формат tool calls)">hermes-парсер</Term> vLLM придерживает хвост, похожий на начало &lt;tool_call&gt;.
        </p>
      </Callout>

      <SimulatorLink stage="postprocess" />

      <h3>Почему нельзя один парсер на всех</h3>
      <p>
        Формат мыслей и вызовов каждое семейство моделей выучило своё, и сервер должен знать, какой ждать. Поэтому парсер выбирают флагом при запуске
        сервера, под конкретную модель. Посмотрите, как один и тот же по смыслу ответ выглядит у четырёх семейств.
      </p>

      <FamiliesFigure />

      <p>
        Главные различия в том, где начинается рассуждение и чем размечены каналы. У DeepSeek-R1 открывающий &lt;think&gt; дописан в промпт (так устроен
        шаблон, глава <ChapterLink to="template" />), и вывод начинается сразу внутри рассуждения: парсер работает в режиме <Term t="Forced reasoning">forced reasoning</Term>. Вызовы
        инструментов официально умеет версия R1-0528, а её шаблон &lt;think&gt; уже не дописывает: модель печатает тег сама. У gpt-oss вся разметка{' '}
        <Term t="Harmony (формат gpt-oss)">Harmony</Term> состоит из спецтокенов с special=true: если детокенизировать её со skip_special_tokens, каналы
        пропадут, поэтому vLLM разбирает Harmony по номерам токенов. Llama 3.1 рассуждений не печатает, а вызов пишет голым JSON с ключом parameters.
      </p>
      <p>
        Откуда в флагах Qwen3 имя hermes? Формат вызовов (JSON внутри &lt;tool_call&gt;) пришёл из моделей Hermes от Nous Research, поэтому и парсер для
        Qwen3 в vLLM и Dynamo называется hermes.
      </p>

      <Callout kind="pitfall" title="забытый или чужой парсер">
        <p>
          Без <code>--reasoning-parser</code> мысли Qwen3 придут клиенту в content вместе с тегами &lt;think&gt;. С вызовами vLLM строг: если в запросе
          есть tools, tool_choice по умолчанию становится <code>"auto"</code>, и без пары <code>--enable-auto-tool-choice --tool-call-parser</code> сервер
          отклонит запрос с ошибкой 400. Коварнее SGLang и <Term t="Dynamo" />: без <code>--tool-call-parser</code> (в Dynamo{' '}
          <code>--dyn-tool-call-parser</code>) вызов молча придёт обычным текстом в content, а <Term t="finish_reason" /> будет <code>"stop"</code> вместо{' '}
          <code>"tool_calls"</code>: агент решит, что модель просто ответила. С чужим парсером бывает ещё хуже: модель вызывает функцию, а клиент видит
          кусок JSON в ответе.
        </p>
        <p>
          В Dynamo парсит frontend, но флаги ставят на воркер с префиксом <code>--dyn-</code>: воркер объявляет выбор парсера в карточке своей
          модели (<Term t="MDC">Model Deployment Card</Term>), и frontend берёт его оттуда. Если парсинг передан
          движку (<code>--dyn-chat-processor vllm</code> или <code>sglang</code>), флаги ставят на сам frontend и без префикса.
        </p>
      </Callout>

      <h3>SSE: как это выглядит на проводе</h3>
      <p>
        Всё, что выдал парсер, сервер упаковывает в <Term t="SSE">Server-Sent Events</Term>. Это обычный HTTP-ответ, в котором каждое событие является
        строкой <code>data:</code> с JSON-объектом <code>chat.completion.chunk</code>, а события разделены пустой строкой. Поле <code>delta</code> внутри
        чанка несёт приращение: кусок рассуждения, кусок content или фрагмент tool_calls. Поток заканчивается строкой <code>data: [DONE]</code>.
      </p>

      <SseFigure />

      <p>
        Обратите внимание на поля, одинаковые во всех чанках (id, created, model), на пустой delta в финальном чанке и на usage: он приходит отдельным чанком
        с пустым choices и только если клиент попросил его через <code>stream_options.include_usage</code>. А <code>[DONE]</code> не является JSON, его
        нужно проверить до разбора. Так клиент собирает поток:
      </p>
      <CodeBlock lang="python" title="клиент: сборка стрима" code={CLIENT} />

      <Callout kind="production" title="reasoning_content или reasoning">
        <p>
          Поле для мыслей нестандартное: в Chat Completions у OpenAI его нет. SGLang, Dynamo и DeepSeek API называют его reasoning_content. vLLM в ноябре
          2025 года (v0.11.1) переименовал его в reasoning, а старое имя удалил в v0.16 (февраль 2026). Симулятор на этом сайте показывает
          reasoning_content, как Dynamo. Клиенту, который должен работать с разными серверами, проще читать оба поля.
        </p>
        <p>
          Мысли стоят денег, даже если их не показывать: в vLLM <code>include_reasoning=false</code> прячет рассуждение из ответа, но токены всё равно
          генерируются и входят в completion_tokens.
        </p>
      </Callout>

      <SimulatorLink stage="stream" />

      <Deeper title="потоковые аргументы и битый JSON">
        <p>
          Hermes-парсер vLLM на каждом шаге заново разбирает весь накопленный текст вызова. Как только регулярное выражение находит закрытое{' '}
          <code>"name": "…"</code>, он один раз отправляет чанк с index, id, type и именем. Дальше он отправляет разность: ту часть строки после{' '}
          <code>"arguments":</code>, которую ещё не слал. Когда JSON становится валидным, у аргументов отрезается закрывающая скобка внешнего объекта. Поэтому
          последний токен «{'}}'}↵» в фигуре дал фрагмент «{'}'}».
        </p>
        <p>
          Dynamo по умолчанию поступает иначе. Встретив начало вызова, он копит текст в своём jail и наружу отдаёт пустые чанки, а по закрывающему тегу
          разбирает вызов и отправляет его одним чанком с готовыми arguments. Для клиента оба варианта законны, поэтому склеивать фрагменты по index нужно
          всегда.
        </p>
        <p>
          Модель может закрыть &lt;/tool_call&gt;, не дописав скобку. Если клиенту к этому моменту ничего не ушло, парсер вернёт исходный текст в content, и
          вызова не будет. Если фрагменты уже ушли, клиент получит строку arguments, которая не разбирается. Без strict или{' '}
          <code>tool_choice="required"</code> гарантий нет: аргументы нужно проверять (json.loads, затем схема инструмента), а при ошибке переспрашивать
          модель.
        </p>
      </Deeper>

      <Deeper title="связь со structured output">
        <p>
          Парсер только читает то, что модель уже написала. Гарантировать, что вызов разберётся, может лишь ограничение на этапе генерации: маска грамматики в
          сэмплере (глава <ChapterLink to="structured-output" />). В vLLM <code>tool_choice="required"</code> и выбор конкретной функции проходят через structured
          outputs: вызов гарантированно разбирается, но не обязательно хорош по смыслу. При <code>tool_choice="auto"</code> ограничения включаются, только если
          у инструмента стоит <code>strict: true</code>. Dynamo для поддерживаемых пар парсера и бэкенда по умолчанию включает structural tags XGrammar
          (флаг воркера <code>--dyn-structural-tag</code>): грамматика ограничивает именно формат вызова этого парсера.
        </p>
        <p>
          Reasoning-парсер ещё и вовремя включает грамматику. Если применять маску с первого токена, модель не сможет подумать: от неё сразу потребуют «{'{'}».
          Поэтому vLLM с настроенным reasoning-парсером включает маску только после &lt;/think&gt;. В Dynamo для этого на воркере ставят оба флага: нативный{' '}
          <code>--reasoning-parser</code> движка решает, когда стартует грамматика, а <code>--dyn-reasoning-parser</code> заполняет reasoning_content.
        </p>
      </Deeper>

      <Takeaways
        items={[
          'Модель печатает одну ленту с тегами, а парсер раскладывает её по полям reasoning_content, content и tool_calls. Сначала работает reasoning-парсер, потом tool-парсер по content.',
          'Парсер устроен как машина состояний с буфером: кусок, похожий на начало тега, придерживается до следующего куска. Результат не должен зависеть от того, как поток порезан.',
          'Формат у каждого семейства свой (Qwen3: <think> и hermes; DeepSeek-R1: рассуждение без открывающего тега; gpt-oss: каналы Harmony по номерам токенов; Llama 3.1: JSON с parameters), поэтому парсер выбирают флагом под модель.',
          <>
            На проводе это строки <code>data:</code> с chat.completion.chunk, финальный чанк с finish_reason (<code>"tool_calls"</code>, если был вызов),
            usage по запросу и <code>data: [DONE]</code>. Аргументы вызова приходят кусками строки и разбираются только в конце.
          </>,
        ]}
      />
    </Chapter>
  )
}
