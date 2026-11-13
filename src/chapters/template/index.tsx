import { useMemo } from 'react'
import { Callout, Chapter, ChapterLink, CodeBlock, Deeper, SimulatorLink, Takeaways, Term, fmt } from '@/ui'
import { useSettings } from '@/state/settings'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import type { ChatMessage, ModelFamily } from '@/engine/types'
import { DEFAULT_TOP_P, type RequestOptions } from '../request/requestModel'
import { resolveHardware } from '../request/responseModel'
import { PreprocessorFlow } from './PreprocessorFlow'
import { TemplateLab } from './TemplateLab'
import { TemplatePitfalls } from './TemplatePitfalls'
import { QWEN3_JINJA_EXCERPT, ReasoningStripDemo, ThinkTail, useToolsOverhead } from './TemplateDeeper'

export default function TemplateChapter() {
  const [settings] = useSettings()
  const { tokenizer } = useTokenizer()
  const hw = useMemo(() => resolveHardware(settings.modelId, settings.gpuId), [settings.modelId, settings.gpuId])
  const modelFamily = (hw.model as { family?: string }).family
  const initialFamily: ModelFamily = modelFamily === 'llama3' ? 'llama3' : modelFamily === 'gpt-oss' ? 'gpt-oss' : 'qwen3'

  const messages = useMemo<ChatMessage[]>(() => {
    const m: ChatMessage[] = []
    if (settings.systemText.trim()) m.push({ role: 'system', content: settings.systemText })
    m.push({ role: 'user', content: settings.userText })
    return m
  }, [settings.systemText, settings.userText])
  const toolsOverhead = useToolsOverhead(tokenizer, messages)
  const modelLabel = (hw.model as { hfId?: string }).hfId ?? hw.model.name

  // запрос читателя как в главе «Запрос» и в симуляторе: те же tools, enable_thinking и response_format
  const requestOpts = useMemo<RequestOptions>(
    () => ({
      model: modelLabel,
      systemText: settings.systemText,
      userText: settings.userText,
      temperature: settings.temperature,
      topP: DEFAULT_TOP_P,
      maxTokens: 0, // в этой фигуре max_tokens нарочно не задан: его подставляет сервер
      tools: settings.tools,
      json: settings.json,
      stream: true,
      thinking: settings.thinking,
    }),
    [modelLabel, settings.systemText, settings.userText, settings.temperature, settings.tools, settings.json, settings.thinking],
  )

  return (
    <Chapter
      id="template"
      lead={
        <>
          Модель не видит JSON. Она видит одну длинную строку, где реплики разделены служебными маркерами вроде{' '}
          <code>{'<|im_start|>'}</code>. Эту строку собирает chat template, маленькая программа на языке шаблонов Jinja из
          tokenizer_config.json модели. Ошибка в шаблоне не роняет сервер, а тихо портит ответы.
        </>
      }
    >
      <p>
        Трансформер умеет ровно одно: продолжать последовательность токенов. Список реплик с ролями ему ни о чём не говорит.
        Поэтому прежде чем запрос попадёт на GPU, сервер делает два шага на CPU: склеивает сообщения в одну строку по{' '}
        <Term t="Chat template">шаблону чата</Term> и режет эту строку на токены. Вместе их называют препроцессингом, а в Dynamo так
        и называется компонент: <Term t="Preprocessor" />.
      </p>
      <p>
        Где именно это происходит, зависит от системы. В <Term t="NVIDIA Dynamo">Dynamo</Term> препроцессор живёт во frontend,
        процессе на Rust: шаблон рендерит <Term t="minijinja" />, токенизирует библиотека HF tokenizers, и всё это до роутера.
        Причина простая: <Term t="KV-aware routing">KV-aware роутер</Term> выбирает воркер по хэшам блоков, а хэши считаются из
        token IDs. В vLLM те же шаги делает процесс API-сервера, в SGLang компонент TokenizerManager в главном процессе сервера.
        Проведите свой запрос по шагам:
      </p>

      <PreprocessorFlow tokenizer={tokenizer} opts={requestOpts} maxContext={hw.model.maxContext} />

      <p>Из этой схемы три следствия для прода:</p>
      <ul>
        <li>
          Шаблон и токенизация стоят на критическом пути к первому токену: пока они не закончились, prefill не начнётся. На
          коротком запросе это незаметно, но под нагрузкой и на длинных контекстах CPU фронтенда может стать узким местом, и его
          масштабируют отдельно от GPU.
        </li>
        <li>Ошибка в шаблоне не порождает ошибку API. Модель получает формат, на котором её не учили, и просто отвечает хуже.</li>
        <li>
          <Term t="Prefix caching">Prefix cache</Term> и KV-aware роутер сравнивают token IDs, а те зависят от каждого байта строки
          после шаблона. Два «одинаковых» запроса, которые отличаются одним пробелом, для кэша совпадают только до этого пробела
          (в vLLM и Dynamo с точностью до блока в 16 токенов): всё дальше считается заново.
        </li>
      </ul>
      <SimulatorLink stage="preprocess" />

      <h3>Как выглядит строка, которую видит модель</h3>
      <p>
        Шаблон лежит рядом с весами, в tokenizer_config.json под ключом <code>chat_template</code> (или отдельным файлом
        chat_template.jinja). На вход он получает <code>messages</code>, <code>tools</code> и флаги вроде{' '}
        <Term t="add_generation_prompt">
          <code>add_generation_prompt</code>
        </Term>{' '}
        и <Term t="enable_thinking"><code>enable_thinking</code></Term>, на выходе отдаёт строку. Каждое семейство моделей обучали на своём формате, поэтому и
        шаблоны у них разные. В них есть <Term t="Special tokens">служебные токены</Term>: цельные записи словаря со своим номером
        (<code>{'<|im_start|>'}</code> = 151644), которые BPE никогда не склеивает из букв.
      </p>
      <p>Во что превратятся ваши сообщения у разных моделей и сколько это стоит в токенах?</p>

      <TemplateLab
        tokenizer={tokenizer}
        systemText={settings.systemText}
        userText={settings.userText}
        initialFamily={initialFamily}
        initialThinking={settings.thinking}
        initialTools={settings.tools}
      />

      <p>
        Посмотрите на разброс в строке «Этот же диалог в других шаблонах»: одни и те же реплики у разных семейств стоят
        по-разному, и почти вся разница приходится на текст, который шаблон дописал сам. Llama 3.1 вставляет даты, которых вы не
        писали, gpt-oss пишет целый system-блок про каналы и уровень рассуждения. Эта невидимая обвязка проходит через prefill и
        занимает KV cache в каждом запросе. В настоящих словарях каждой модели короткий диалог «Ты полезный ассистент.» плюс
        «Сколько будет 2+2?» занимает 23 токена у DeepSeek-R1, 32 у Qwen3, 53 у Llama 3.1 и 89 у gpt-oss (в фигуре Llama и
        DeepSeek считаются условно, словарём Qwen3).
      </p>
      <p>
        Дата у gpt-oss к тому же стоит в самом начале промпта, в первом system-блоке, и шаблон берёт сегодняшнее число. Раз в
        сутки меняется почти весь префикс, и prefix cache вчерашних запросов уже не подходит. У Llama 3.1 дата по умолчанию одна и
        та же, 26 июля 2024 года.
      </p>
      <p>
        Обратите внимание и на хвост. Строка заканчивается не на вопросе пользователя, а на открытой реплике ассистента: модель
        продолжит текст с этого места, и её ответ будет репликой ассистента. За это отвечает флаг{' '}
        <code>add_generation_prompt</code>, в vLLM он включён по умолчанию.
      </p>

      <Callout kind="precise" title="BOS и двойной BOS">
        <p>
          Шаблоны Llama 3.1 и DeepSeek сами ставят в начало токен начала последовательности (<Term t="BOS">BOS</Term>). Если потом
          токенизировать
          отрендеренную строку с <code>add_special_tokens=True</code> (так по умолчанию в transformers), BOS окажется дважды:{' '}
          <code>[128000, 128000, 128006, …]</code>. Поэтому vLLM для chat completions по умолчанию токенизирует с{' '}
          <code>add_special_tokens=False</code>: BOS уже вставил шаблон. У Qwen3 и gpt-oss BOS в шаблоне нет вовсе.
        </p>
      </Callout>

      <h3>Что сломается, если ошибиться</h3>
      <p>Шаблон нужно воспроизвести байт в байт. Вот три классические ошибки и то, что видит модель в каждой из них:</p>

      <TemplatePitfalls tokenizer={tokenizer} systemText={settings.systemText} userText={settings.userText} />

      <p>
        Общее у всех трёх: сервер не выдаёт никакой ошибки. Ответы становятся хуже или дороже, и это видно только по качеству и по
        доле попаданий в prefix cache. Отсюда практическое правило: всё, что меняется от запроса к запросу (дата, имя
        пользователя, свежие документы), ставят как можно ближе к концу промпта, а неизменную часть (system, описания
        инструментов) держат одинаковой до байта, включая порядок инструментов и ключей в их описаниях. Подробнее о кэше в главе{' '}
        <ChapterLink to="prefix-caching" />.
      </p>

      <Callout kind="production" title="где живёт шаблон">
        <p>
          В Dynamo шаблон рендерит препроцессор во frontend: crate dynamo-renderer на minijinja берёт <code>chat_template</code>{' '}
          из tokenizer_config.json, а свой шаблон задаётся воркеру флагом <code className="whitespace-nowrap">--custom-jinja-template</code>. В vLLM есть флаг{' '}
          <code className="whitespace-nowrap">--chat-template</code>, аргументы шаблона (например <code>enable_thinking</code>) приходят в запросе в{' '}
          <code>chat_template_kwargs</code>. В SGLang тоже есть <code className="whitespace-nowrap">--chat-template</code>.
        </p>
        <p>
          Несовпадение шаблона входит в классические причины жалоб «модель стала тупее». Типичные сценарии: обновили образ, а с ним
          и шаблон; переехали на другое зеркало весов или квантизованную копию с другим tokenizer_config.json. Пример из жизни:
          зеркало NousResearch для Llama 3.1 хранит старый короткий шаблон Llama 3 без строк с датами. Проверка простая:
          отрендерить промпт тем же путём, что и сервер, и побайтно сравнить с эталоном <code>apply_chat_template</code> из
          transformers.
        </p>
      </Callout>

      <Deeper title="как устроен шаблон Qwen3 внутри">
        <p>Главное из официального шаблона Qwen3, длинные строки сокращены:</p>
        <CodeBlock lang="jinja" code={QWEN3_JINJA_EXCERPT} title="chat_template (Qwen/Qwen3-8B, фрагмент)" wrap />
        <p>
          <b>enable_thinking=false</b> не выключает рассуждение внутри модели. Шаблон дописывает пустой блок, и модель видит, будто
          уже подумала и пора отвечать. Хвост промпта становится таким:
        </p>
        <ThinkTail tk={tokenizer} />
        <p>
          Четыре лишних токена: 151667 и 151668 это <code>{'<think>'}</code> и <code>{'</think>'}</code>, 271 это два перевода
          строки одним токеном.
        </p>
        <p>
          <b>Инструменты</b> шаблон кладёт в system: каждая функция одной строкой JSON внутри <code>{'<tools>'}</code>, плюс
          инструкция, как оформлять вызов. Для вашего запроса одна маленькая функция get_weather стоит{' '}
          <b>{toolsOverhead === null ? '…' : fmt.tokens(toolsOverhead)}</b>, и они повторяются в каждом запросе. Поэтому длинный
          system с инструментами это первый кандидат на prefix caching.
        </p>
        <p>
          <b>Рассуждение прошлых ходов</b> шаблон вырезает: <code>{'<think>'}</code> остаётся только у ответов после последнего
          вопроса пользователя (в агентном цикле с инструментами это весь текущий ход). Модели это не мешает, а prefix cache
          ломает:
        </p>
        <ReasoningStripDemo tk={tokenizer} messages={messages} />
        <p>
          Другие семейства поступают похоже: DeepSeek-R1 оставляет от прошлого ответа только текст после{' '}
          <code>{'</think>'}</code>, gpt-oss выбрасывает канал analysis у ходов, которые закончились ответом.
        </p>
      </Deeper>

      <Takeaways
        items={[
          <>Модель видит одну строку: шаблон склеивает роли, служебные токены и приглашение ассистента. У каждого семейства свой формат.</>,
          <>
            Обвязка шаблона стоит токенов: один короткий диалог в настоящих словарях занимает от 23 до 89 токенов в зависимости от
            модели, а описание одной функции get_weather в Qwen3 добавляет{' '}
            {toolsOverhead === null ? 'около 150 токенов' : fmt.tokens(toolsOverhead)} к каждому запросу.
          </>,
          <>
            Шаблон воспроизводится байт в байт. Лишний символ меняет токены, и prefix cache выручает только до него, а чужие маркеры
            тихо ухудшают ответы.
          </>,
          <>
            В Dynamo шаблон и токенизация работают в preprocessor до роутера, в vLLM в API-сервере, в SGLang в TokenizerManager. Это
            CPU-работа на пути к первому токену.
          </>,
        ]}
      />
    </Chapter>
  )
}
