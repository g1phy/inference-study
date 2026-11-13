import { useMemo, useState } from 'react'
import { Callout, Chapter, ChapterLink, Deeper, SimulatorLink, Takeaways, Term, fmt } from '@/ui'
import { useSettings } from '@/state/settings'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { renderTemplate } from '@/engine/templates'
import { kvBytesPerToken } from '@/engine/perf'
import type { ChatMessage, ModelFamily } from '@/engine/types'
import { countPrompt } from '../template/promptTokens'
import { WEATHER_TOOL } from '../request/requestModel'
import { resolveHardware } from '../request/responseModel'
import { LiveTokenizer } from './LiveTokenizer'
import { BpeSteps } from './BpeSteps'
import { PreTokenization } from './PreTokenization'
import { VocabCompare } from './VocabCompare'
import { BytesDemo, InjectionDemo, LanguageTax, LettersDemo, VocabTable } from './TokenizationDeeper'
import { charCount } from './textStats'

/** Слово для разбора BPE по умолчанию: первое слово запроса с ведущим пробелом (как « простыми»). */
function firstWord(pieces: string[]): string | null {
  return pieces.find((p) => /^ \p{L}{4,}$/u.test(p)) ?? pieces.find((p) => /\p{L}{3,}/u.test(p)) ?? null
}

export default function TokenizationChapter() {
  const [settings] = useSettings()
  const { tokenizer } = useTokenizer()
  const [ownText, setOwnText] = useState<string | null>(null)
  const [ownWord, setOwnWord] = useState<string | null>(null)
  const text = ownText ?? settings.userText

  const defaultWord = useMemo(() => (tokenizer ? firstWord(tokenizer.preTokenize(settings.userText)) : null) ?? ' полезный', [tokenizer, settings.userText])
  const word = ownWord ?? defaultWord

  // ваш промпт после шаблона и его KV cache на модели из настроек
  const hw = useMemo(() => resolveHardware(settings.modelId, settings.gpuId), [settings.modelId, settings.gpuId])
  const family: ModelFamily = (hw.model as { family?: string }).family === 'llama3' ? 'llama3' : 'qwen3'
  const promptTokens = useMemo(() => {
    if (!tokenizer) return null
    const messages: ChatMessage[] = []
    if (settings.systemText.trim()) messages.push({ role: 'system', content: settings.systemText })
    messages.push({ role: 'user', content: settings.userText })
    // те же tools и enable_thinking, что в симуляторе: число совпадает с главами «Запрос» и «Шаблон чата»
    const r = renderTemplate({ family, messages, enableThinking: settings.thinking, tools: settings.tools ? [WEATHER_TOOL] : undefined })
    return countPrompt(family, r.segments, tokenizer, null)?.total ?? null
  }, [tokenizer, family, settings.systemText, settings.userText, settings.thinking, settings.tools])
  const kvPerToken = kvBytesPerToken(hw.model)

  // во сколько раз символов вашего запроса больше, чем токенов: цена побуквенной токенизации
  const charsPerToken = useMemo(() => {
    if (!tokenizer) return null
    const n = tokenizer.encode(settings.userText).length
    return n > 0 ? charCount(settings.userText) / n : null
  }, [tokenizer, settings.userText])

  return (
    <Chapter
      id="tokenization"
      lead={
        <>
          Модель работает только с числами. Токенизатор режет текст на токены, куски слов, и заменяет каждый номером в словаре: у
          Qwen3 их 151 669. Это обычная программа на CPU, без нейросети, но от неё зависят цена запроса, место в контексте и размер
          KV cache.
        </>
      }
    >
      <p>
        <Term t="Token">Токен</Term> это не слово и не буква, а кусок байтов из словаря модели. Частое слово вместе с пробелом перед
        ним обычно занимает один токен, редкое или русское режется на два-три куска, и у каждого куска есть номер, ID. Модель видит
        только последовательность этих номеров. Сам <Term t="Tokenizer">токенизатор</Term> не нейросеть: это{' '}
        <Term t="Vocabulary">словарь</Term> и список правил склейки, детерминированная функция «текст в числа» и обратно.
      </p>
      <p>
        Почему куски, а не буквы или целые слова? Если кормить модель буквами или байтами, последовательность выйдет в несколько
        раз длиннее: в вашем запросе символов{' '}
        {charsPerToken === null ? 'заметно больше' : `в ${fmt.num(charsPerToken, 1)} раза больше`}, чем токенов. Каждая лишняя
        позиция это ещё один шаг <Term t="Decode">decode</Term> при генерации, ещё одна строка в <Term t="KV cache" /> и больше
        работы для <Term t="Attention">attention</Term>, которая растёт с длиной. Словарь из целых слов, наоборот, пришлось бы
        делать огромным, и он всё равно не знал бы новых слов, опечаток, имён и кода. BPE берёт середину: частые куски получают
        свой номер, а всё редкое собирается из байтов.
      </p>
      <p>Как режется ваш собственный запрос и сколько он стоит в токенах?</p>

      <LiveTokenizer
        tokenizer={tokenizer}
        text={text}
        userText={settings.userText}
        onText={setOwnText}
        onPickWord={setOwnWord}
      />

      <p>
        Посмотрите на границы. Пробел почти всегда приклеен к следующему слову: в « простыми» он вошёл в первый кусок «·прост». Русские
        слова режутся на части чаще английских, а числа Qwen3 разбирает по одной цифре. Эмодзи бывает одним токеном, а бывает
        несколькими кусками байтов, и тогда отдельный кусок вообще не является символом: в ленте он показан байтами.
      </p>
      <SimulatorLink stage="preprocess" />

      <h3>Почему русский дороже</h3>
      <p>
        Английская буква занимает в <Term t="UTF-8" /> один байт, русская два. Токенизаторы обучали в основном на английском
        тексте, поэтому частых русских кусков в словаре меньше, и одна и та же мысль по-русски распадается на большее число
        токенов. Во сколько раз русский текст длиннее английского в токенах, называют налогом на язык. Его размер зависит от
        словаря:
      </p>

      <LanguageTax />

      <p>
        Это не косметика. Время <Term t="Prefill">prefill</Term> растёт с числом токенов промпта, время ответа с числом токенов ответа, а контекстное окно и
        счёт за API считаются в токенах. На Qwen3 тот же смысл по-русски стоит примерно в 1,7 раза больше по всем трём статьям.
      </p>
      <Callout kind="pitfall" title="«токен это четыре символа»">
        <p>
          Правило «один токен примерно четыре символа» пришло из английского текста и словарей OpenAI. Для русского текста на Qwen3
          символов на токен заметно меньше, посмотрите статистику в живом токенизаторе. Лимиты контекста и бюджеты считайте
          настоящим токенизатором той модели, которая будет отвечать.
        </p>
      </Callout>

      <h3>Как BPE собирает токены</h3>
      <p>
        Почти все современные LLM режут текст алгоритмом <Term t="Byte-level BPE">byte-level BPE</Term> (byte pair encoding).
        Слово раскладывается на байты UTF-8. Затем среди соседних кусков ищется пара с наименьшим рангом в списке склеек (
        <Term t="merges" />
        ), и два куска становятся одним. Так повторяется, пока ни одной пары из списка не останется. Ранг это номер правила: при
        обучении токенизатора самые частые пары добавлялись в словарь первыми. У Qwen3 таких правил {fmt.int(151387)}.
      </p>
      <Callout kind="intuition">
        <p>
          BPE похож на архиватор, который заранее выучил самые частые пары символов и заменяет каждую одним кодом, потом пары
          кодов, и так далее. Частое слово сжимается в один код, редкое остаётся набором кусков. Только «символы» здесь байты, а
          словарь выучен один раз при обучении токенизатора и дальше не меняется.
        </p>
      </Callout>
      <p>Посмотрим на это по шагам, на настоящих правилах Qwen3:</p>

      <BpeSteps tokenizer={tokenizer} word={word} onWord={setOwnWord} />

      <p>
        Самое неожиданное здесь это промежуточные куски. BPE не знает, где кончается буква: он склеивает половину русской буквы с
        соседним байтом, если такая пара частая, и буквы собираются только по ходу. Сравните и ранги: « world» начинается со
        склейки ранга 13, а русские слова со склеек около тысячи: частых русских пар при обучении токенизатора встретилось
        меньше.
      </p>

      <h3>Пре-токенизация: где режут до BPE</h3>
      <p>
        Склейки никогда не переходят границы, которые заранее расставила регулярка{' '}
        <Term t="Pre-tokenization">пре-токенизатора</Term>. Она режет текст на куски:
        буквы вместе с одним ведущим символом (обычно пробелом), цифры, пунктуацию, пробелы. Отсюда и пробел в начале слова, и
        цифры по одной. У каждого семейства своя регулярка.
      </p>

      <PreTokenization tokenizer={tokenizer} userText={settings.userText} />

      <p>
        Сравните две регулярки на одной строке. Qwen3 режет число 2025 на четыре цифры, o200k на «202» и «5». o200k учитывает
        регистр и делит camelCase на части, а сокращение I'm оставляет одним куском. JSON-пунктуация вроде <code>{'{"'}</code> и{' '}
        <code>{'":'}</code> склеивается в один пре-токен у обоих, и это важно для structured output: один токен может закрывать сразу несколько
        элементов JSON (<ChapterLink to="structured-output" />).
      </p>

      <h3>Один текст, два словаря</h3>
      <p>
        Словарь у каждого семейства свой, и ID не переносятся между моделями: номер 151644 значит <code>{'<|im_start|>'}</code> только
        у Qwen. Тот же текст в другом словаре даёт другие куски, другие номера и другое число токенов. Сравним текст из живого
        токенизатора в словаре Qwen3 и в o200k_harmony, словаре gpt-oss:
      </p>

      <VocabCompare tokenizer={tokenizer} text={text} />

      <p>
        Многие границы совпадают: обе регулярки пристёгивают пробел к слову, а частые слова в обоих словарях стали целыми
        токенами. Расходятся они там, где регулярки режут по-разному (числа, camelCase) или где один словарь выучил кусок, которого
        нет в другом: отсюда и разное число токенов. А номера разные почти у всех кусков, даже одинаковых: совпадают только
        одиночные байты вроде «,» (11) и «.» (13). Поэтому токены одной модели нельзя скормить другой, счётчик токенов одной модели
        врёт для другой, а prefix cache и роутер обязаны пользоваться ровно тем же токенизатором, что и движок.
      </p>

      <Callout kind="production" title="токенизация на CPU, токены в счёте">
        <ul>
          <li>
            Токенизация работает на CPU до GPU. В Dynamo это Rust во frontend: crate dynamo-tokenizers поверх HF tokenizers
            (флаг <code className="whitespace-nowrap">--tokenizer</code>, по умолчанию HF). В vLLM токенизирует процесс API-сервера, в SGLang TokenizerManager
            (процессов можно добавить флагом <code className="whitespace-nowrap">--tokenizer-worker-num</code>).
          </li>
          <li>
            Frontend и воркеры должны работать с одними и теми же файлами модели. Воркер Dynamo получает готовые token IDs и сам
            текст не токенизирует, поэтому чужой tokenizer.json во frontend даст модели не те номера. Dynamo сверяет контрольную
            сумму <Term t="MDC">карточки модели</Term> (в ней и токенизатор, и шаблон) у всех реплик и не пускает трафик на реплику с другой суммой.
          </li>
          <li>
            Токены это деньги, контекст и память. Цена API считается за миллион токенов, контекстное окно ограничено в токенах, а
            KV cache растёт с каждым токеном. Ваш запрос после шаблона:{' '}
            <b>{promptTokens === null ? '…' : fmt.tokens(promptTokens)}</b>, на {hw.model.name} это{' '}
            {fmt.bytes(kvPerToken, { mode: 'binary' })} на токен, то есть{' '}
            <b>{promptTokens === null ? '…' : fmt.bytes(promptTokens * kvPerToken, { mode: 'binary' })}</b> KV cache ещё до первого
            слова ответа (<ChapterLink to="kv-cache" />).
          </li>
        </ul>
      </Callout>

      <Deeper title="байты, Ġ, буквы, спецтокены и размеры словарей">
        <p>
          <b>Неизвестных символов не бывает.</b> Базовый алфавит byte-level BPE это все 256 значений байта, поэтому любая строка
          (любой язык, эмодзи, мусорные байты) раскладывается хотя бы на отдельные байты. Редкий символ просто займёт несколько
          токенов:
        </p>
        <BytesDemo tk={tokenizer} />
        <p>
          <b>Откуда Ġ и Ċ.</b> Чтобы хранить словарь печатными строками, 188 «печатных» байтов отображаются сами в себя, а остальные
          68 в символы U+0100…U+0143. Пробел 0x20 становится <code>Ġ</code>, перевод строки 0x0A становится <code>Ċ</code>. Поэтому в
          словаре « world» записан как <code>Ġworld</code>, а «Пр» как <code>ÐŁÑĢ</code>: это байты D0 9F D1 80. Наведите на любой токен в лентах выше: форма «в словаре» показана в
          подсказке.
        </p>
        <p>
          <b>Почему модели плохо считают буквы.</b> Модель не видит букв, она видит номера кусков. Сколько «r» в strawberry, из трёх
          ID напрямую не прочитать:
        </p>
        <LettersDemo tk={tokenizer} />
        <p>
          <b>Спецтокены и <Term t="инъекция спецтокенов">инъекции</Term>.</b> Служебный токен вроде <code>{'<|im_start|>'}</code> это отдельная запись словаря. Но
          HF-токенизаторы по умолчанию распознают его литерал прямо в тексте. Сервер сначала склеивает промпт в одну строку, а потом
          токенизирует её целиком, поэтому не отличит маркер из шаблона от того же текста, набранного пользователем. Пользователь
          может «закрыть» свою реплику и подделать системную:
        </p>
        <InjectionDemo tk={tokenizer} />
        <p>
          Защита: до шаблона вычищать или экранировать в тексте пользователя литералы служебных токенов. Флаг{' '}
          <code>split_special_tokens=True</code> в HF (в нашей фигуре <code>allowSpecial: false</code>) защищает лишь отчасти: он
          помогает, только если текст пользователя токенизируют отдельно от шаблона (на всей склеенной строке он разрежет и
          настоящие маркеры), и действует только на токены с пометкой special, а <code>{'<think>'}</code> и{' '}
          <code>{'<tool_call>'}</code> у Qwen3 оставляет цельными. У tiktoken и npm-пакета gpt-tokenizer по умолчанию ещё строже:
          встретив спецтокен в тексте, они бросают ошибку.
        </p>
        <p>
          <b>Цифры.</b> Qwen2 и Qwen3 режут числа по одной цифре, Llama 3, DeepSeek-V3 и o200k группами до трёх: «12345678» это 8
          токенов у Qwen3 и 3 у o200k. Это влияет на арифметику модели и на structured output: в позиции «начало целого числа»
          грамматике у Qwen3 нужно разрешить 11 токенов, у Llama 3 больше тысячи.
        </p>
        <p>
          <b>Размер словаря не равен ширине логитов.</b> Строк в таблице эмбеддингов и в выходном слое часто больше, чем токенов:
          их добивают до удобного для GPU размера, хвостовые строки пустые. Маски structured output считаются по ширине логитов:
        </p>
        <VocabTable />
      </Deeper>

      <Takeaways
        items={[
          <>Токен это кусок байтов, а не слово и не буква. ID это номер в словаре конкретной модели: у каждого семейства словарь свой.</>,
          <>Byte-level BPE: регулярка режет текст на куски, байты внутри куска склеиваются по рангам merges. Неизвестных символов не бывает.</>,
          <>Русский текст на Qwen3 занимает примерно в 1,7 раза больше токенов, чем тот же смысл по-английски: это время prefill и decode, деньги и KV cache.</>,
          <>Токенизация это CPU-работа во frontend до GPU. Литералы спецтокенов из текста пользователя нельзя пропускать как управляющие токены.</>,
        ]}
      />
    </Chapter>
  )
}
