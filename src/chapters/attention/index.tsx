import { getModel } from '@/engine/data/models'
import { kvBytesForSeq, kvBytesPerToken, prefillFlopsBreakdown, weightBytes } from '@/engine/perf'
import { Callout, Chapter, ChapterLink, CodeBlock, Deeper, fmt, Formula, ScrollX, SimulatorLink, Takeaways, Term } from '@/ui'
import { BridgeFigure } from './BridgeFigure'
import { attentionCrossover, attnCost, flashTraffic, linearFlopsPerToken, withKvHeads } from './cost'
import { FlashFigure } from './FlashFigure'
import { GqaFigure } from './GqaFigure'
import { HeadsFigure } from './HeadsFigure'
import { ScaleFigure } from './ScaleFigure'
import { StepsFigure } from './StepsFigure'
import { FEATURES, FOCUS, OGRADA, PROMPT_LEN, TOY, TRACE, trianglePairs } from './toy'

/* Числа текста: из данных модели и из игрушечного расчёта, не руками. */
const M = getModel('qwen3-8b')
const M_MHA = withKvHeads(M, M.heads)
const KV_TOKEN = kvBytesPerToken(M)
const KV_MHA = kvBytesPerToken(M_MHA)
const KV_MLA = kvBytesPerToken(getModel('deepseek-v3'))
const GROUP = M.heads / M.kvHeads
const ATTN_PARAMS_LAYER = 2 * M.hidden * M.heads * M.headDim + 2 * M.hidden * M.kvHeads * M.headDim
const ATTN_PARAMS = ATTN_PARAMS_LAYER * M.layers
const W_ONA = TRACE.weights[FOCUS.query]
const W_ONA_OGRADA = OGRADA.trace.weights[FOCUS.query]
const COST_ROWS = [1024, 8192, 32768].map((n) => {
  const b = prefillFlopsBreakdown(M, n)
  return { ...attnCost(M, n), share: b.attention / b.total }
})
const kib = (b: number) => fmt.bytes(b, { mode: 'binary' })
const FLASH_N = 4096
const pct = (v: number) => fmt.pct(v)
const vec = (v: readonly number[]) => `[${v.map((x) => fmt.num(x)).join('; ')}]`

/* Байты одного шага decode при одном запросе: все веса плюс KV этого запроса (GQA против MHA). */
const stepBytes = (m: typeof M, n: number) => weightBytes(m) + kvBytesForSeq(m, n)
const GQA_SAVE_SHORT = 1 - stepBytes(M, 4096) / stepBytes(M_MHA, 4096)
const GQA_RATIO_LONG = stepBytes(M_MHA, 32768) / stepBytes(M, 32768)

/* Счётчики фигуры «мост»: prefill промпта и первый шаг decode. */
const BRIDGE_PREFILL_PAIRS = trianglePairs(PROMPT_LEN)
const BRIDGE_DECODE_PAIRS = PROMPT_LEN + 1

const TEX_ATTENTION = '\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}\\!\\left(\\frac{\\cq{Q}\\,\\ck{K}^{\\top}}{\\sqrt{d_k}}+M\\right)\\cv{V}'
const TEX_MASK = 'M_{ij}=\\begin{cases}0, & j \\le i\\\\ -\\infty, & j > i\\end{cases}'
const TEX_ROPE = '(R_m\\,\\cq{q})^{\\top}(R_n\\,\\ck{k}) = \\cq{q}^{\\top} R_{\\,n-m}\\,\\ck{k}'

const LAYER_CODE = `# один слой attention Qwen3-8B, n токенов (упрощённо)
xn = rmsnorm(x)        # норма перед attention (pre-norm)
q = xn @ W_q           # [n, 4096] → 32 головы × 128
k = xn @ W_k           # [n, 1024] →  8 KV-голов × 128
v = xn @ W_v           # [n, 1024] →  8 KV-голов × 128
q, k = rope(rmsnorm(q), rmsnorm(k), positions)  # QK-norm по головам, затем RoPE
kv_cache.append(k, v)  # кэшируются только K и V
for h in range(32):    # голова h берёт пару K/V номер h // 4
    s = q[h] @ k[h // 4].T / sqrt(128) + causal_mask
    out[h] = softmax(s) @ v[h // 4]
x = x + concat(out) @ W_o   # [n, 4096], residual`

export default function AttentionChapter() {
  return (
    <Chapter
      id="attention"
      lead={
        <>
          Attention позволяет каждому токену собрать информацию из предыдущих токенов. Для этого токен строит три вектора: Query (что я ищу), Key (по чему
          меня найти) и Value (что я передам тем, кто на меня посмотрит). В трансформере это единственное место, где токены обмениваются информацией, и
          отсюда же появляется KV cache.
        </>
      }
    >
      <h3 id="attention-why">Зачем токенам смотреть друг на друга</h3>
      <p>
        После <ChapterLink to="embeddings">эмбеддингов</ChapterLink> у каждого токена есть свой вектор, но этот вектор знает только о самом токене. Вектор
        слова «она» одинаков в любой фразе, и ответа на вопрос, кто такая «она», в нём нет.
      </p>
      <p>
        Возьмём фразу «Кошка не перепрыгнула забор, потому что она устала». Человек сразу понимает, что устала кошка. В русском это подсказывает род: «она»
        женского рода, «кошка» тоже, а «забор» мужского. Чтобы модель сделала тот же вывод, информация из вектора «Кошки» должна попасть в вектор «она».
      </p>
      <p>
        Этим и занимается <Term t="attention" />. В каждом блоке трансформера (глава <ChapterLink to="transformer" />) две главные части:{' '}
        <Term t="MLP" /> обрабатывает каждый токен отдельно, а attention даёт токенам посмотреть друг на друга. Другого способа обменяться информацией у токенов нет.
      </p>
      <p>
        Механика в двух предложениях. Каждый токен строит три вектора, <Term t="Q / K / V">Q, K и V</Term>: запрос описывает, что токен ищет, ключ
        описывает, по каким признакам его можно найти, а значение несёт то, что он отдаст. Запрос сравнивается с ключами всех предыдущих токенов и с
        собственным ключом (будущие токены закрывает маска, <Term t="causal mask">causal mask</Term>), оценки превращаются в доли через{' '}
        <Term t="softmax" />, и токен забирает смесь значений в этих долях.
      </p>

      <Callout kind="intuition" title="поиск в библиотеке">
        <p>
          У вас в руках листок с запросом (Query): «нужно про кошек». На корешке каждой книги наклеен ярлык (Key), а внутри лежит содержимое (Value). Вы
          сравниваете листок со всеми ярлыками и выписываете из каждой книги тем больше, чем лучше её ярлык подходит к запросу.
        </p>
        <p>
          Отличие от настоящей библиотеки одно, но важное: attention не выбирает одну книгу, а берёт понемногу из всех сразу. Поэтому на выходе получается
          смесь, а не копия одного источника.
        </p>
      </Callout>

      <p>Вся операция записывается одной формулой. Ниже каждая её часть разобрана на числах.</p>
      <Formula
        block
        className="max-md:text-[13px]"
        tex={TEX_ATTENTION}
        label={
          <>
            Q, K, V: матрицы запросов, ключей и значений всех токенов; d<sub>k</sub>: длина одного вектора ключа; M: маска, которая закрывает будущее
          </>
        }
      />

      <h3 id="attention-steps">Attention по шагам на одной фразе</h3>
      <p>
        Возьмём игрушку, где каждое число можно проверить в уме. У каждого слова {FEATURES.length} признака с понятными именами:{' '}
        {FEATURES.join(', ')}. Векторы Q, K и V короткие, по два числа (d<sub>k</sub> = {TOY.dK}). Матрицы весов подобраны вручную, в настоящей модели их
        находит обучение.
      </p>
      <p>
        Вопрос фигуры: на кого посмотрит «она» и почему? Проходите шаги кнопками или стрелками на клавиатуре, нажимайте на слова и на ячейки матриц.
      </p>

      <StepsFigure id="attention-figure-steps" />

      <p>
        Что здесь важно. В строке «она» после softmax {pct(W_ONA[FOCUS.target])} веса достаётся «Кошке» и только {pct(W_ONA[FOCUS.distractor])}{' '}
        «забору». Решили дело два числа запроса: «она» ищет предмет ({TRACE.Q[FOCUS.query][0]}) и ещё сильнее женский род ({TRACE.Q[FOCUS.query][1]}).
        Ключ «Кошки» отвечает на оба вопроса, ключ «забора» только на первый.
      </p>
      <p>
        Посмотрите и на соседние строки. «Перепрыгнула» тоже смотрит на «Кошку» ({pct(TRACE.weights[2][0])}): глагол женского рода ищет, кто действует. А
        у служебных слов запрос нулевой, все оценки равны, и внимание делится поровну: им нечего искать.
      </p>
      <p>
        Заметьте, что запрос и ключ одного и того же слова разные. «Она» ищет предмет женского рода (запрос {vec(TRACE.Q[FOCUS.query])}), но сама предметом
        не является, поэтому её ключ всего {vec(TRACE.K[FOCUS.query])}. Отсюда две отдельные матрицы: то, что слово ищет, и то, по чему его находят, обычно
        не совпадает. Значение V отвечает на третий вопрос: что слово передаст. В игрушке V для простоты копирует два признака, а в настоящей модели по
        ключу находят одно, а через значение передают другое: «Кошку» находят по роду, а передаёт она смысл, живое существо, которое может устать.
      </p>
      <p>
        Теперь переключите в фигуре «забор» на «ограду». Ограда тоже женского рода, и ключи у обоих слов одинаковые, {vec(OGRADA.trace.K[FOCUS.target])}:
        строка «она» делит вес поровну, по {pct(W_ONA_OGRADA[FOCUS.target])} «Кошке» и «ограде». Одной голове, которая различает слова только по
        признакам «предмет» и «женский род», здесь не на что опереться. Настоящая модель решает такие случаи другими головами и следующими слоями, которые
        учитывают смысл: устают кошки, а не ограды.
        Поэтому голов в слое много, а слоёв десятки.
      </p>

      <Callout kind="pitfall" title="Q, K и V берутся из одного вектора">
        <p>
          Это не три разных куска текста, а три проекции одного и того же вектора токена. Каждый токен одновременно спрашивает (Q), отвечает на чужие вопросы
          (K) и отдаёт содержимое (V). И смотрит он в том числе на себя: маска не закрывает диагональ, поэтому слово «она» отдаёт {pct(W_ONA[FOCUS.query])}{' '}
          веса самому себе.
        </p>
      </Callout>

      <Callout kind="precise" title="что в настоящей модели">
        <p>
          В игрушке признаков {FEATURES.length}, и у них есть имена. В {M.name} вектор токена состоит из {fmt.int(M.hidden)} чисел без имён, а запрос каждой головы
          получается своей матрицей {fmt.int(M.hidden)} × {M.headDim}. Веса модель выучивает сама, и у измерений нет понятных имён: «ищу женский род» в
          настоящей голове никто не подписывал, и такой признак обычно размазан по многим числам сразу. Механика при этом та же самая.
        </p>
      </Callout>

      <h3 id="attention-heads">Ваш промпт: у каждой головы свой паттерн</h3>
      <p>
        Настоящий слой делает не одну такую операцию, а несколько параллельно. Каждую называют <Term t="голова внимания">головой</Term>: у неё свои веса,
        поэтому каждая голова ищет своё.
      </p>
      <p>
        Зачем несколько голов, а не одна большая? Одна голова раздаёт ровно 100 % веса и потому смотрит, по сути, в одну сторону. А токену часто нужно
        несколько вещей сразу: глаголу «устала» нужно найти, кто устал, и одновременно соседа слева, чтобы собрать слово из кусочков-токенов. Несколько
        голов дают несколько независимых взглядов за один слой. У {M.name} {M.heads} головы в каждом из {M.layers} слоёв, то есть{' '}
        {fmt.int(M.heads * M.layers)} {fmt.plural(M.heads * M.layers, ['отдельная картина', 'отдельные картины', 'отдельных картин'])} внимания для каждого
        токена.
      </p>
      <p>
        Как такие картины выглядят на настоящих токенах? Ниже начало вашего запроса из симулятора, нарезанное токенизатором Qwen3 (глава{' '}
        <ChapterLink to="tokenization" />
        ), и семь типичных паттернов голов. Строка тепловой карты показывает, куда смотрит один токен, а дуги показывают то же для выбранного токена.
      </p>

      <HeadsFigure id="attention-figure-heads" />

      <p>
        Что заметить. «Предыдущий токен» даёт диагональ сразу под главной: каждый токен смотрит на соседа слева, так модель склеивает слово из кусочков.
        «<Term t="Attention sink">Attention sink</Term>» сваливает почти весь вес на первый токен: softmax обязан раздать все 100 %, и голове, которой нечего искать, удобно «парковаться» на
        начале. <Term t="Induction head">Induction-голова</Term> оживает только там, где в тексте есть повтор.
      </p>
      <p>
        Это иллюстрации типичного поведения, а не веса настоящей модели. Похожие головы находят в обученных моделях, но у каждой модели свой набор, и
        большинство голов не так просто описать словами.
      </p>

      <h3 id="attention-gqa">Много голов и GQA: сколько K и V хранить</h3>
      <p>
        Выходы всех голов склеиваются в один вектор и проходят ещё одну матрицу, W<sub>O</sub>, которая возвращает размер {fmt.int(M.hidden)}. В
        классическом multi-head attention (MHA) у каждой головы свои K и V.
      </p>
      <p>
        Для памяти это дорого. K и V хранятся для каждого токена, в каждом слое и для каждой KV-головы: именно они и составляют{' '}
        <Term t="KV cache" />. Поэтому в
        современных моделях несколько голов запросов делят одну пару K/V. Это <Term t="GQA" /> (grouped-query attention): у {M.name} {M.heads} головы
        запросов и всего {M.kvHeads} пар K/V. Крайний случай, MQA, оставляет одну пару на весь слой. Другой путь выбрали DeepSeek-V3 и Kimi K2:{' '}
        <Term t="MLA" /> хранит вместо K и V один сжатый вектор на слой, это {kib(KV_MLA)} на токен (глава <ChapterLink to="kv-cache" />).
      </p>
      <p>Сколько памяти это экономит и чем приходится платить?</p>

      <GqaFigure id="attention-figure-gqa" />

      <p>
        GQA в {M.name} уменьшает кэш в {GROUP} раза: {kib(KV_TOKEN)} на токен вместо {kib(KV_MHA)} при MHA. На практике это в {GROUP} раза больше
        одновременных длинных запросов в той же памяти GPU и в {GROUP} раза меньше байт KV cache, которые каждый шаг <Term t="decode" /> читает из памяти.
        Веса модели шаг decode читает целиком в любом случае, поэтому на коротком контексте и маленьком batch шаг ускоряется ненамного, а на длинных
        контекстах и большом batch, когда KV cache всех запросов по объёму сравним с весами или больше их, заметно. У {M.name} при одном запросе на{' '}
        {fmt.int(4096)} токенов байт за шаг меньше примерно на {fmt.pct(GQA_SAVE_SHORT, 0)}, а на {fmt.int(32768)} токенах примерно в{' '}
        {fmt.num(GQA_RATIO_LONG, 1)} раза.
      </p>
      <p>
        Платят за это качеством: чем меньше пар K/V, тем меньше у голов разных ключей и значений. По опыту статьи про GQA, общая пара K/V на группу из
        нескольких голов почти не ухудшает качество по сравнению с MHA. С одной парой на весь слой (MQA) модель теряет в качестве больше, а её дообучение
        идёт менее устойчиво. Поэтому новые модели выбирают GQA. Подробный расчёт памяти в главе <ChapterLink to="kv-cache" />.
      </p>

      <h3 id="attention-bridge">Мост к KV cache: prefill и decode</h3>
      <p>
        Осталось связать attention с тем, как модель пишет ответ. Генерация идёт в две фазы (подробно в главе <ChapterLink to="prefill-decode" />).{' '}
        <Term t="prefill">Prefill</Term> пропускает через модель весь промпт за один проход. Decode добавляет по одному токену за шаг.
      </p>
      <p>Какую часть матрицы внимания считает каждая фаза и что из уже посчитанного можно не считать заново?</p>

      <BridgeFigure id="attention-figure-bridge" />

      <p>
        В prefill считается весь нижний треугольник: запрос каждого слова промпта сравнивается с ключом этого же слова и с ключами всех слов перед ним. В
        decode новому токену нужна одна новая строка: его запрос сравнивается с ключами всех предыдущих токенов и с собственным ключом, и токен забирает
        смесь их значений.
      </p>
      <p>
        Главное наблюдение: K и V старых токенов никогда не меняются. Маска не даёт им смотреть вперёд, поэтому новые токены на них не влияют. Значит, их
        достаточно посчитать один раз и сохранить. Это и есть KV cache: на шаге decode модель считает q, k и v только для нового токена,
        дописывает его k и v в кэш и читает оттуда всё остальное. Запросы старых токенов больше никому не нужны, поэтому Q не хранят.
      </p>
      <p>
        Сравните счётчики фигуры. Prefill на {PROMPT_LEN} словах сделал {BRIDGE_PREFILL_PAIRS} произведений q·k и ничего не читал из кэша: только что
        посчитанные ключи тут же пошли в дело, большинство по нескольку раз. Шаг decode ради {BRIDGE_DECODE_PAIRS} произведений читает из кэша{' '}
        {PROMPT_LEN} пар K и V, и каждая пара нужна ему ровно один раз. На каждый прочитанный байт приходится мало вычислений, поэтому decode упирается в
        скорость памяти, а не в вычисления: он <Term t="memory-bound" /> (глава <ChapterLink to="roofline" />).
      </p>
      <p>
        В настоящей модели так происходит в каждом слое для каждой KV-головы. У {M.name} на один токен в кэше лежит 2 (K и V) ×{' '}
        {fmt.count(M.layers, ['слой', 'слоя', 'слоёв'])} × {M.kvHeads} KV-{fmt.plural(M.kvHeads, ['голова', 'головы', 'голов'])} × {M.headDim} чисел × 2
        байта (BF16) = {kib(KV_TOKEN)}. Кэш растёт с каждым токеном, и ему посвящена вся часть IV.
      </p>
      <div className="flex flex-wrap gap-x-6">
        <SimulatorLink stage="prefill">Prefill в симуляторе</SimulatorLink>
        <SimulatorLink stage="decode">Decode в симуляторе</SimulatorLink>
      </div>

      <Callout kind="pitfall" title="«с KV cache attention линейный»">
        <p>
          Это верно только про один шаг decode: его стоимость растёт пропорционально длине контекста, O(n). Кэш экономит пересчёт K и V, но не их чтение:
          каждый шаг читает весь кэш целиком. Поэтому суммарная работа на m сгенерированных токенов всё равно растёт квадратично, а prefill остаётся
          квадратичным по длине промпта.
        </p>
      </Callout>

      <Callout kind="production" title="где attention видно в метриках">
        <p>
          В движках attention считают специальные ядра (<Term t="kernel">kernels</Term>): <Term t="FlashAttention" />, <Term t="FlashInfer" /> и другие. vLLM
          и SGLang выбирают attention backend под GPU и модель, и от этого выбора зависят даже допустимые размеры блоков KV cache.
        </p>
        <p>
          Цена attention заметна в двух метриках. Чем длиннее промпт, тем большую долю prefill занимает квадратичный attention, и растёт{' '}
          <Term t="TTFT" />. Чем длиннее контекст, тем больше KV cache читает каждый шаг decode, и растёт <Term t="ITL" />. GQA,{' '}
          <Term t="KV cache quantization">FP8 KV cache</Term> и <Term t="sliding window" /> уменьшают именно этот объём.
        </p>
      </Callout>

      <Deeper
        title={
          <>
            формула целиком, √d<sub>k</sub> и размеры матриц в {M.name}
          </>
        }
      >
        <Formula block tex={`${TEX_ATTENTION},\\qquad ${TEX_MASK}`} narrowTex={`\\begin{gathered} ${TEX_ATTENTION}, \\\\ ${TEX_MASK} \\end{gathered}`} />
        <p>
          Для одной головы и n токенов: Q, K и V имеют размер n × {M.headDim}, матрица оценок и матрица весов n × n, выход снова n × {M.headDim}. Выходы{' '}
          {M.heads} голов склеиваются в n × {fmt.int(M.heads * M.headDim)} и умножаются на W<sub>O</sub>. В {M.name} матрица W<sub>Q</sub> имеет размер{' '}
          {fmt.int(M.hidden)} × {fmt.int(M.heads * M.headDim)}, а W<sub>K</sub> и W<sub>V</sub> всего {fmt.int(M.hidden)} ×{' '}
          {fmt.int(M.kvHeads * M.headDim)}: это GQA в весах. Вместе с W<sub>O</sub> это {fmt.compact(ATTN_PARAMS_LAYER)} параметров на слой и{' '}
          {fmt.num(ATTN_PARAMS / 1e9, 2)} млрд на всю модель, около {pct(ATTN_PARAMS / (M.paramsB * 1e9))} весов.
        </p>
        <CodeBlock lang="python" title="слой attention, псевдокод" code={LAYER_CODE} copy={false} />
        <h4>
          Зачем делить на √d<sub>k</sub>
        </h4>
        <p>
          Если компоненты q и k независимы, со средним 0 и дисперсией 1, то дисперсия их скалярного произведения равна d<sub>k</sub>: разброс оценок растёт
          как √d<sub>k</sub>. У головы {M.name} d<sub>k</sub> = {M.headDim}, и без деления оценки разлетались бы примерно на ±{fmt.num(Math.sqrt(M.headDim), 0)}.
          Softmax почти всегда выбирал бы один ключ, а градиенты при обучении становились бы крошечными. Деление на √{M.headDim} ≈{' '}
          {fmt.num(Math.sqrt(M.headDim), 1)} возвращает разброс к единице. Qwen3 вдобавок нормирует векторы q и k каждой головы (QK-RMSNorm), что тоже держит
          оценки в разумных пределах.
        </p>
        <ScaleFigure id="attention-figure-scale" headDim={M.headDim} />
      </Deeper>

      <Deeper title="сколько стоит attention: O(n²·d) в prefill и O(n·d) на шаг decode">
        <p>
          В prefill каждый из n токенов сравнивается со всеми предыдущими: это n(n + 1)/2 пар (запрос, ключ) на голову и слой. Каждая пара стоит около 2d
          FLOP на скалярное произведение (d = {M.headDim}, длина вектора одной головы) и ещё 2d на смесь значений, отсюда O(n²·d). На шаге decode пар всего
          n + 1, то есть O(n·d).
        </p>
        <div className="grid min-w-0 gap-2">
          <p id="attention-cost-caption" className="text-sm leading-snug text-ink-2">
            Attention {M.name}: все {M.layers} слоёв и {M.heads} головы, FLOP на QKᵀ и взвешивание V. Доля считается от всех FLOPs prefill вместе с
            линейными слоями.
          </p>
          <ScrollX label="Таблица стоимости attention" hint="таблица шире экрана: прокрутите её вбок">
            <table aria-labelledby="attention-cost-caption" className="w-full min-w-[600px] border-collapse text-left text-[14px] tabular-nums">
              <thead>
                <tr className="border-b border-line-strong text-ink-2">
                  <th className="py-2 pr-4 font-medium">Контекст n</th>
                  <th className="py-2 pr-4 font-medium">Пар на голову в prefill</th>
                  <th className="py-2 pr-4 font-medium">Prefill целиком</th>
                  <th className="py-2 pr-4 font-medium">Доля attention в prefill</th>
                  <th className="py-2 font-medium">Один шаг decode</th>
                </tr>
              </thead>
              <tbody>
                {COST_ROWS.map((r) => (
                  <tr key={r.context} className="border-b border-line">
                    <td className="py-2 pr-4 text-ink">{fmt.int(r.context)}</td>
                    <td className="py-2 pr-4 text-ink">{fmt.compact(r.prefillPairs)}</td>
                    <td className="py-2 pr-4 text-ink">{fmt.flops(r.prefillFlops)}</td>
                    <td className="py-2 pr-4 text-ink">≈ {fmt.pct(r.share, 0)}</td>
                    <td className="py-2 text-ink">{fmt.flops(r.decodeFlops)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollX>
        </div>
        <p>
          Для сравнения: все линейные слои {M.name} тратят около {fmt.flops(linearFlopsPerToken(M))} на токен. Attention одного шага decode обгоняет их только
          после ≈ {fmt.num(attentionCrossover(M) / 1000, 1)} тыс. токенов контекста. До этого по FLOPs дороже умножение на веса, а сам attention на decode
          упирается не в вычисления, а в чтение KV cache из памяти (глава <ChapterLink to="roofline" />). В prefill доля attention растёт вместе с длиной
          промпта: при {fmt.int(COST_ROWS[0].context)} токенах это около {fmt.pct(COST_ROWS[0].share, 0)} всех FLOPs, при{' '}
          {fmt.int(COST_ROWS[2].context)} уже около {fmt.pct(COST_ROWS[2].share, 0)}.
        </p>
      </Deeper>

      <Deeper title="FlashAttention: тайлы вместо матрицы n × n">
        <p>
          Обычная реализация считает матрицу оценок S = QKᵀ целиком, пишет её в медленную память GPU (<Term t="HBM / VRAM">HBM</Term>), читает обратно для
          softmax, пишет веса P и снова читает их для умножения на V. Для одной головы при n = {fmt.int(FLASH_N)} каждая такая матрица занимает{' '}
          {kib(flashTraffic(FLASH_N, M.headDim).scoreMatrixBytes)}, и почти весь трафик памяти уходит на неё.
        </p>
        <p>
          <Term t="FlashAttention" /> режет Q, K и V на тайлы, которые помещаются в быструю память на кристалле (<Term t="SRAM / on-chip cache">SRAM</Term>
          ). Для каждого тайла запросов он проходит по тайлам ключей и значений, считает кусок оценок и тут же обновляет softmax на лету: хранит текущий
          максимум и сумму экспонент и поправляет накопленный выход. Матрица n × n целиком не появляется нигде.
        </p>
        <p>
          Результат тот же, это точный алгоритм, а не приближение. Операций даже немного больше, но память перестаёт быть узким местом. В статье про
          FlashAttention (GPT-2 medium, A100, прямой и обратный проход) трафик HBM упал с 40,3 до 4,4 ГБ, а время с 41,7 до 7,3 мс. FlashAttention-3 на H100
          выжимает до 740 TFLOPS в FP16, около 75 % пика.
        </p>
        <FlashFigure id="attention-figure-flash" headDim={M.headDim} />
        <p>
          FlashAttention экономит память и время, но не отменяет квадратичность: пар (запрос, ключ) по-прежнему около n²/2. Обойти её пытаются приближённые
          схемы: sliding window, linear attention, сжатие старого контекста.
        </p>
      </Deeper>

      <Deeper title="RoPE: позиция живёт внутри Q и K">
        <p>
          В самой формуле attention нет порядка слов: если переставить ключи местами, оценки просто переставятся вместе с ними. Позицию модель узнаёт через{' '}
          <Term t="RoPE" />: перед сравнением векторы q и k поворачиваются. Каждая пара координат поворачивается на угол m·θ<sub>i</sub>, где m означает номер
          позиции, а частоты θ<sub>i</sub> = base<sup>−2i/d</sup> фиксированы и не обучаются (у Qwen3 база {fmt.int(1_000_000)}).
        </p>
        <Formula block tex={TEX_ROPE} label="повёрнутые q на позиции m и k на позиции n: результат зависит только от расстояния n − m" />
        <p>
          V не поворачивают: позиция нужна, чтобы решить, куда смотреть, а не что забрать. Игрушка с поворотом вектора есть в главе{' '}
          <ChapterLink to="embeddings" />.
        </p>
        <p>
          Следствие для инференса: в KV cache лежат уже повёрнутые K, то есть кэш привязан к позициям. Вдобавок K и V токена в верхних слоях зависят от
          всех токенов перед ним. Поэтому без пересчёта и без потери качества готовый KV можно переиспользовать только при точном совпадении с начала
          промпта. Кусок из середины переиспользуют лишь приближённо: ключи поворачивают под новые позиции и часть токенов пересчитывают заново (CacheBlend,
          глава <ChapterLink to="prefix-caching" />).
        </p>
      </Deeper>

      <Takeaways
        items={[
          <>
            Attention: каждый токен сравнивает свой запрос Q с ключами K всех предыдущих токенов и своим, softmax превращает оценки в веса, и токен забирает
            смесь значений V. В трансформере это единственное место, где токены обмениваются информацией.
          </>,
          <>Causal mask закрывает будущее, поэтому K и V уже обработанных токенов не меняются. Их хранят в KV cache, а не пересчитывают.</>,
          <>
            Голов много: у {M.name} {M.heads} головы запросов в каждом из {M.layers} слоёв. GQA отдаёт им всего {M.kvHeads} пар K/V и уменьшает кэш в{' '}
            {GROUP} раза, до {kib(KV_TOKEN)} на токен.
          </>,
          <>
            Prefill считает весь треугольник n × n сразу: работа растёт как квадрат длины промпта, O(n²). Decode добавляет одну строку за шаг: работа шага
            растёт линейно с длиной контекста, O(n), но каждый раз читается весь KV cache.
          </>,
        ]}
      />
    </Chapter>
  )
}
