import { useState } from 'react'
import { getModel } from '@/engine/data/models'
import { getGpu } from '@/engine/data/gpus'
import { KV_TIERS, getInterconnect } from '@/engine/data/interconnects'
import { kvBytesPerToken, kvCapacityTokens, loadVsRecompute } from '@/engine/perf'
import { Callout, Chapter, ChapterLink, CodeBlock, Deeper, SimulatorLink, Tabs, Takeaways, Term, fmt } from '@/ui'
import { ClusterFigure } from './ClusterFigure'
import { LifecycleFigure } from './LifecycleFigure'
import { TermsTable } from './TermsTable'
import { TierFigure } from './TierFigure'
import { findCrossover } from './tiers'

const QWEN = getModel('qwen3-8b')
const H100 = getGpu('h100-sxm')
const PER_TOKEN = kvBytesPerToken(QWEN)
const POOL = kvCapacityTokens(QWEN, H100, { memUtil: 0.92 })
const SESSIONS = 1000
const SESSION_LEN = 8192
const RAM = KV_TIERS.find((t) => t.id === 'g2')!.nodeCapacityBytes
const NVME = KV_TIERS.find((t) => t.id === 'g3')!.nodeCapacityBytes
// Порог полосы на примере: 32 768 токенов Qwen3-8B на H100 (загрузка выгоднее, если канал быстрее KV/токен × скорость prefill)
const BE_TOKENS = 32768
const BE = loadVsRecompute(QWEN, H100, BE_TOKENS, 1)
const PREFILL_TPS = BE_TOKENS / (BE.recomputeMs / 1000)
const TCP_GBS = getInterconnect('tcp-25g').GBs
const TCP_PARITY = findCrossover(QWEN, H100, TCP_GBS, 0, 1).tokens

const DAEMONSET_YAML = `# Упрощённый пример: по одному кэш-серверу KV на каждую GPU-ноду
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: kv-cache
spec:
  selector:
    matchLabels: { app: kv-cache }
  template:
    metadata:
      labels: { app: kv-cache }
    spec:
      nodeSelector:
        nvidia.com/gpu.present: "true"      # только GPU-ноды
      priorityClassName: system-node-critical
      runtimeClassName: nvidia              # видит GPU ноды для CUDA IPC
      containers:
        - name: lmcache-server
          image: REGISTRY/lmcache:TAG        # подставьте свой образ
          command: ["lmcache", "server"]
          env:
            - { name: NVIDIA_VISIBLE_DEVICES, value: all }   # но nvidia.com/gpu не запрашивает
          args:
            - --host=0.0.0.0                 # слушать IP пода, иначе Service не достучится
            - --port=5555
            - --l1-size-gb=60                # RAM ноды под KV
            - --eviction-policy=LRU
            - '--l2-adapter={"type":"fs","base_path":"/data/kv"}'
          resources:
            requests: { memory: 65Gi, cpu: "4" }   # 60 ГБ кэша + накладные
          volumeMounts:
            - { name: nvme, mountPath: /data/kv }
      volumes:
        - name: nvme
          hostPath: { path: /mnt/nvme/kv, type: DirectoryOrCreate }   # NVMe ноды
---
apiVersion: v1
kind: Service
metadata:
  name: kv-cache
spec:
  selector: { app: kv-cache }
  internalTrafficPolicy: Local    # трафик только в под на своей ноде
  ports:
    - { name: zmq, port: 5555 }`

const OPERATOR_YAML = `# То же через LMCache Operator: из этого ресурса оператор сам создаст
# DaemonSet, Service с internalTrafficPolicy: Local и ConfigMap для vLLM
apiVersion: lmcache.lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: kv-cache
spec:
  l1:
    sizeGB: 60`

const VLLM_BASH = `# Под vLLM на той же ноде: клиент кэш-сервера.
# У всех подов, которые делят кэш, одинаковое хэширование ключей: переменную задают до запуска
export PYTHONHASHSEED=0
vllm serve Qwen/Qwen3-8B \\
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"lmcache.mp.host":"tcp://kv-cache.default.svc.cluster.local","lmcache.mp.port":5555}}'`

function YamlTabs() {
  const [tab, setTab] = useState<'ds' | 'op' | 'vllm'>('ds')
  return (
    <Tabs
      aria-label="Примеры манифестов"
      value={tab}
      onChange={setTab}
      wrap
      tabs={[
        { id: 'ds', label: 'DaemonSet и Service' },
        { id: 'op', label: 'Через оператор' },
        { id: 'vllm', label: 'Сторона vLLM' },
      ]}
    >
      {(active) =>
        active === 'ds' ? (
          <CodeBlock lang="yaml" title="kv-cache.yaml, упрощённый пример" code={DAEMONSET_YAML} highlightLines={[3, 14, 22, 24, 35, 43]} maxHeight={520} />
        ) : active === 'op' ? (
          <CodeBlock lang="yaml" title="lmcache-engine.yaml, упрощённый пример" code={OPERATOR_YAML} />
        ) : (
          <CodeBlock lang="bash" title="запуск воркера, упрощённый пример" code={VLLM_BASH} wrap />
        )
      }
    </Tabs>
  )
}

export default function KvTiersChapter() {
  return (
    <Chapter
      id="kv-tiers"
      lead={
        <>
          Память GPU быстрая, но маленькая, поэтому готовый KV вытесняют, и следующий похожий запрос снова платит за prefill. Если увозить
          KV на уровни побольше и помедленнее (RAM хоста, локальный NVMe, сетевое хранилище), его можно привезти обратно, и это часто
          быстрее, чем пересчитать. В Kubernetes такой нижний уровень обычно ставят DaemonSet: по кэш-серверу на каждую GPU-ноду, общему
          для всех её воркеров и переживающему их рестарты.
        </>
      }
    >
      <h3>Зачем кэшу этажи</h3>
      <p>
        <ChapterLink to="prefix-caching">Prefix cache</ChapterLink> из прошлой главы живёт в <Term t="HBM / VRAM">HBM</Term>. У
        Qwen3-8B на одной H100 пул KV вмещает около {fmt.compact(POOL, 0)} токенов. Кажется много, пока не посчитаешь нагрузку агентов:{' '}
        {fmt.int(SESSIONS)} активных сессий с историей по {fmt.int(SESSION_LEN)} {fmt.plural(SESSION_LEN, ['токену', 'токена', 'токенов'])} это {fmt.compact(SESSIONS * SESSION_LEN)}{' '}
        токенов, в {fmt.int(Math.round((SESSIONS * SESSION_LEN) / POOL))} раз больше. Пул заполняется, LRU вытесняет, и следующий ход вытесненной сессии снова платит полный prefill.
      </p>
      <p>
        Выход тот же, что в процессорах: не выбрасывать, а увозить ниже. Под HBM есть RAM хоста: на ноде класса DGX H100 это 2 ТБ, или
        около {fmt.compact(RAM / PER_TOKEN, 0)} токенов KV Qwen3-8B. Ниже локальные <Term t="NVMe SSD">NVMe-диски</Term>: {fmt.bytes(NVME)}, около{' '}
        {fmt.compact(NVME / PER_TOKEN, 0)} токенов. Ещё ниже сетевое хранилище, общее для всего кластера. Чем ниже ступень, тем она больше
        и обычно тем дольше до неё добираться. Это и есть <Term t="Tiered KV cache (G1–G4 / L1–L3)">многоуровневый KV cache</Term>.
      </p>
      <p>
        Но есть ли смысл везти KV с медленной полки, если GPU может просто пересчитать prefill? Время загрузки это объём KV, делённый на
        полосу канала: оно растёт линейно с длиной. Время пересчёта растёт быстрее, потому что attention на prefill квадратичен. Значит, чем
        длиннее префикс, тем выгоднее грузить. Где проходит граница для каждого уровня?
      </p>

      <TierFigure />

      <p>
        Что заметить. У Qwen3-8B на H100 RAM хоста по <Term t="PCIe" /> обгоняет пересчёт в десять и больше раз, и сеть с RDMA на графике
        не уступает: одна карта 400 Гбит/с по полосе почти равна PCIe, а медленнее её в жизни делают задержка, соседи по сети и носитель
        на той стороне. Хороший NVMe выигрывает в разы. Обычная сеть без RDMA и объектное хранилище выигрывают только на длинных
        контекстах или не выигрывают вовсе, поэтому решение «грузить или считать» в
        хороших системах адаптивное. И даже при равенстве загрузка полезна: она не тратит вычисления GPU, и те достаются другим запросам.
      </p>
      <p>
        Удобно думать о пороге полосы: загрузка выгоднее, если канал быстрее, чем KV на токен × скорость prefill. Qwen3-8B на H100
        пересчитывает {fmt.int(BE_TOKENS)} токенов примерно за {fmt.ms(BE.recomputeMs)}, то есть около {fmt.compact(PREFILL_TPS, 0)}{' '}
        токенов в секунду; умножаем на {fmt.bytes(PER_TOKEN, { mode: 'binary' })} и
        получаем около {fmt.bandwidth(BE.breakEvenGBs)}. PCIe, NVMe и RDMA намного быстрее, а 25-гигабитный TCP ({fmt.bandwidth(TCP_GBS)})
        как раз на границе{TCP_PARITY ? <>, отсюда паритет около {fmt.compact(TCP_PARITY, 0)} токенов</> : null}.
      </p>

      <Callout kind="intuition" title="стол, ящик, шкаф и склад">
        <p>
          HBM это рабочий стол: всё под рукой, но места мало. RAM хоста это ящик стола: в несколько раз больше, но доставать из него
          дольше. NVMe это
          шкаф в кабинете, сетевое хранилище это общий склад на другом этаже. Частое держат на столе, редкое уносят ниже, но не выбрасывают:
          сходить в шкаф дешевле, чем переписывать конспект заново.
        </p>
      </Callout>

      <h3>Как блок путешествует между уровнями</h3>
      <p>
        Единица хранения та же, что в prefix cache: блок KV с хэшем всей цепочки до него. Хэш служит адресом на любом уровне: «есть ли у тебя
        блок 9c61…?». Поведение задают три политики:
      </p>
      <ul>
        <li>
          <b>запись вниз</b>: копировать блок ниже сразу (<Term t="Write-through / write-back">write-through</Term>), только при вытеснении сверху (write-back) или только популярные
          блоки (selective);
        </li>
        <li>
          <b>подъём вверх</b> (<Term t="Prefetch">prefetch</Term>): сколько ждать подгрузки, прежде чем начать считать; ждать до конца, не ждать вовсе или ждать не
          дольше таймаута;
        </li>
        <li>
          <b>вытеснение</b>: кого убрать, когда уровень полон; обычно того, кем дольше всех не пользовались (<Term t="LRU" />
          ).
        </li>
      </ul>

      <LifecycleFigure />

      <p>
        Главное здесь: вытеснение из HBM больше не означает потерю. Блок уезжает ниже, по хэшу его потом находят, и он возвращается наверх
        слой за слоем, пока GPU уже считает новый хвост. Поэтому в лучшем случае загрузка прячется за вычислениями целиком.
      </p>

      <h3>L3 как DaemonSet на каждой ноде</h3>
      <p>
        Где физически живёт нижний уровень, если сервис крутится в <Term t="Kubernetes" />? Напомню словарь: нода это машина с GPU, под это
        контейнер (или несколько), Deployment держит N одинаковых подов, а <Term t="DaemonSet" /> гарантирует ровно один под на каждой
        подходящей ноде. Добавили ноду, и под DaemonSet появился на ней сам.
      </p>
      <p>
        Для кэша KV это естественная форма. RAM, NVMe и сетевая карта с <Term t="RDMA / GPUDirect RDMA">RDMA</Term> принадлежат ноде, а не
        поду, и ими удобно владеть одному процессу на ноду. Такой под делят все движки ноды, он переживает рестарты и обновления воркеров и
        не забирает GPU у воркеров: в Kubernetes он не запрашивает <code>nvidia.com/gpu</code>, а к памяти воркеров обращается через{' '}
        <Term t="CUDA IPC" />. Это и называют <Term t="L3 KV cache (node-level)">L3 KV cache на ноде</Term>.
      </p>
      <p>
        Почему L3, если у пода кэша есть RAM? Номер уровня здесь считают от воркера, как в SGLang HiCache: L1 это его HBM, L2 его
        собственная RAM внутри процесса движка (её не видит никто, кроме него), а L3 первый уровень за пределами процесса, общий для
        соседей. Носитель у L3 бывает любой: RAM ноды, её NVMe или распределённый пул. В лестнице выше уровни подписаны по носителю,
        поэтому RAM хоста там L2; здесь та же RAM ноды, отданная общему поду, уже L3.
      </p>
      <p>Так устроены реальные системы:</p>
      <ul>
        <li>
          <Term t="LMCache" /> Operator: ресурс <code>LMCacheEngine</code> превращается в DaemonSet (один кэш-сервер на GPU-ноду), Service с{' '}
          <code>internalTrafficPolicy: Local</code> и ConfigMap для vLLM. Воркер подключается коннектором <code>LMCacheMPConnector</code> на
          порт 5555, а тензоры передаются между подами через CUDA IPC. Этот вариант есть среди официальных рецептов{' '}
          <Term t="NVIDIA Dynamo">Dynamo</Term> для оффлоада KV («Deploy LMCache MP»), наряду со встроенным оффлоадом движков, LMCache
          внутри воркера, FlexKV и SGLang HiCache.
        </li>
        <li>
          <Term t="llm-d" /> с <Term t="Mooncake" />: DaemonSet <code>mooncake-client</code> на каждой ноде владеет RAM и NVMe (через hostPath)
          и сетью RDMA и регистрирует их у <code>mooncake-master</code>. Поды vLLM только клиенты, своей памяти в пул не дают, поэтому пул
          переживает их рестарт. Вместе такие DaemonSet образуют общий пул на весь кластер.
        </li>
      </ul>
      <p>Посмотрим, как это ведёт себя при обычных событиях кластера и чем отличается от двух других вариантов.</p>

      <ClusterFigure />

      <p>
        Итог сравнения: DaemonSet это середина спектра. Горячий путь не выходит за пределы ноды, поэтому он ближе к GPU, чем кэш-кластер, а
        кэш живёт дольше, чем во встроенном варианте. Цена: хостовые права (hostPath, общая память для CUDA IPC, устройства RDMA) и то, что
        нода остаётся доменом отказа. Пережить потерю ноды или автоскейлинг вниз помогает только общий глобальный уровень, книга называет его
        G4.
      </p>
      <p>Как это выглядит в манифестах (упрощённо, без проверок и тонкой настройки):</p>

      <YamlTabs />

      <Callout kind="production" title="размеры, единицы, хэши и подвохи Kubernetes">
        <ul>
          <li>
            Каждый нижний уровень должен быть больше верхнего. Если RAM-кэш меньше кэша в HBM, блоки гоняются туда-обратно без пользы: в
            SGLang хиты из L3 некуда положить, а KVBM в Dynamo в такой настройке «молотил» (он устарел в версии 1.5, а в 1.7 его
            убирают).
          </li>
          <li>
            Не путайте единицы: PCIe Gen5 x16 это около 63 ГБ/с в одну сторону (128 у NVIDIA это сумма двух направлений), а сетевая карта
            на 400 Гбит/с это 50 ГБ/с. Реальная полоса ещё зависит от размера передачи: мелкие куски KV по несколько килобайт не загружают
            канал, поэтому KV укладывают крупными блоками.
          </li>
          <li>
            У всех подов, которые делят хранилище, должны совпадать модель, формат KV, размер блока и <code>PYTHONHASHSEED</code> (обычно
            0), иначе ключи не совпадут и шаринга не будет.
          </li>
          <li>
            <code>internalTrafficPolicy: Local</code> не откатывается на другую ноду: если на ноде нет пода кэша (не совпал nodeSelector,
            taint), для её воркеров Service пуст. Обновление DaemonSet по умолчанию (RollingUpdate) перезапускает поды кэша по очереди,
            одну ноду за другой, и RAM-часть кэша на каждой ноде остывает; чтобы перезапускать по своему графику, есть стратегия
            OnDelete.
          </li>
          <li>
            Память пода кэша (например, 65 ГиБ) вычитается из памяти ноды, а CPU-уровень vLLM в <code>/dev/shm</code> засчитывается в лимит
            пода воркера. Метрики: <code>vllm:kv_offload_*</code>, <code>lmcache_mp_lookup_hit_tokens_total</code>,{' '}
            <code>sglang:hicache_host_used_tokens</code>.
          </li>
        </ul>
      </Callout>

      <Callout kind="pitfall" title="«L3 это всегда SSD»">
        <p>
          Не обязательно. В SGLang HiCache L3 это любой бэкенд хранения, в том числе распределённый пул в RAM (Mooncake). В LMCache RAM
          сервера называется L1, а диск и удалённые хранилища L2. В книге и в KVBM локальный SSD это G3, а сетевой G4. Всегда уточняйте, в
          чьей системе названий вы говорите.
        </p>
      </Callout>

      <SimulatorLink stage="prefill">Сколько токенов взято из кэша на стадии prefill</SimulatorLink>

      <Deeper title="HiCache, LMCache, Mooncake, KVBM, llm-d и карта названий">
        <h4>Карта названий</h4>
        <p>
          Одни и те же полки разные системы называют по-своему. Наведите на столбец или нажмите на его заголовок, чтобы подсветить его во
          всех строках:
        </p>
        <TermsTable />
        <h4>SGLang HiCache</h4>
        <p>
          Добавляет к дереву префиксов (RadixAttention) отметку, где лежит KV каждого узла: L1 это GPU, L2 это RAM хоста, L3 это общий бэкенд
          хранения (<code>file</code>, <code>mooncake</code>, <code>hf3fs</code>, <code>nixl</code> и другие). L1 и L2 приватны для
          инстанса даже на одной ноде, делиться между инстансами можно только через L3. Включается <code>--enable-hierarchical-cache</code>,
          размер L2 задаёт <code>--hicache-ratio</code> (по умолчанию 2 от пула GPU). Политика записи по умолчанию write_through. Подгрузка из
          L3 запускается, если там нашлось хотя бы 256 токенов; режим <code>timeout</code> ждёт не дольше, чем 2 с плюс 0,1 с на каждые
          1&nbsp;024 токена (не больше 30 с): для 32 тысяч токенов это 5,2 с.
        </p>
        <h4>LMCache</h4>
        <p>
          Отдельный слой кэша KV для vLLM и SGLang. В режиме отдельного сервера (MP) L1 это RAM сервера, L2 это адаптеры хранения: локальный
          диск, Redis, Mooncake, S3. Кэш режется на куски по 256 токенов, вытеснение LRU стартует при заполнении на 80 % и сбрасывает 20 %.
          Кроме <code>LMCacheEngine</code> у оператора есть <code>LMCacheCoordinator</code> для глобального поиска, но это один Deployment на
          кластер, а не DaemonSet.
        </p>
        <h4>Mooncake Store</h4>
        <p>
          Пришёл из Kimi (Moonshot AI). Master хранит только метаданные: где лежит блок, сколько реплик, что вытеснять. Данные между нодами
          возит Transfer Engine по RDMA напрямую, минуя master. Клиенты, владеющие RAM и SSD, бывают встроены в движок (пул живёт и умирает
          вместе с ним) или работают отдельным процессом, тогда пул переживает рестарты движков.
        </p>
        <h4>Dynamo KVBM</h4>
        <p>
          <Term t="KVBM" /> (KV Block Manager) двигал блоки по уровням G1 (GPU), G2 (pinned RAM), G3 (локальный NVMe) и G4 (удалённое
          хранилище), книга описывает его как актуальный компонент. Но в Dynamo 1.5.0 (сентябрь 2026) KVBM объявлен устаревшим, а к версии
          1.7.0 его убирают: из основной ветки он уже удалён. Вместо него документация предлагает встроенный оффлоад движков, LMCache (в том числе DaemonSet через оператор), FlexKV и
          SGLang HiCache. Зато роутер Dynamo научился учитывать уровень: префикс в HBM даёт полный кредит, в RAM 0,75, на диске 0,25 (подробнее
          в главе <ChapterLink to="routing">про KV-aware routing</ChapterLink>).
        </p>
        <h4>llm-d</h4>
        <p>
          Стек для Kubernetes вокруг vLLM и SGLang. Его роутер ведёт карту блоков по подам и уровням и выбирает под, где готово больше всего
          начала промпта. Гайд по многоуровневому кэшу рекомендует начинать со встроенного оффлоада движка (HBM и RAM), добавлять диск, когда
          рабочее множество не влезает, и брать LMCache или Mooncake, если нужна возможность, которой нет во встроенном варианте.
        </p>
      </Deeper>

      <Takeaways
        items={[
          'HBM заполняется быстро, но вытесненный KV можно не выбрасывать, а увезти ниже: в RAM хоста, на NVMe, в сетевое хранилище.',
          'Загрузка выигрывает, когда полоса уровня выше порога «KV на токен × скорость prefill». Чем длиннее префикс, тем ниже порог: RAM по PCIe обгоняет пересчёт в десять и больше раз, медленная сеть только на длинных контекстах.',
          'Кэш ноды в виде DaemonSet делят все движки ноды, он переживает рестарты воркеров и сам появляется на новых нодах (LMCache Operator, llm-d с Mooncake). Нода при этом остаётся доменом отказа.',
          'Названия уровней у всех свои: G1–G4 у книги и KVBM, L1–L3 у SGLang, L1 и L2 у LMCache. KVBM в Dynamo объявлен устаревшим в 1.5 и удаляется в 1.7.',
        ]}
      />
    </Chapter>
  )
}
