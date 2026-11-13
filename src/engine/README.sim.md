# Движок «engine-sim» (агент A2): API

Системная часть движка: данные о железе и моделях, формулы производительности, KV cache и prefix
caching, KV-aware роутер, симуляции батчинга и (дез)агрегации, игрушечный attention, ГПСЧ.
Всё чистое (без React и DOM) и детерминированное (никакого `Math.random`, только seed).
Контракты типов: `src/engine/types.ts`; здесь расширения помечены суффиксом `Ext`.

```ts
import { getModel } from '@/engine/data/models'
import { getGpu } from '@/engine/data/gpus'
import { kvBytesPerToken, decodeStepMs, prefillMs } from '@/engine/perf'

const m = getModel('qwen3-8b'), g = getGpu('h100-sxm')
kvBytesPerToken(m)                                   // 147456 (144 КиБ)
prefillMs(m, g, 2000)                                // ≈ 58,9 мс (MFU 0,5)
decodeStepMs(m, g, { batch: 1, avgContext: 2000 })   // ≈ 7,4 мс (MBU 0,7 + 0,3 мс накладных)
```

## Соглашения

- **Единицы.** Байты (не ГБ), FLOPs, миллисекунды. ГБ = 10⁹ байт, ГиБ = 2³⁰ (`GB`, `GiB`, `MiB` в `perf.ts`).
  Полосы каналов `GBs` — ГБ/с **в одну сторону** (NVIDIA пишет NVLink и PCIe как сумму двух направлений).
  TFLOPS — только **dense** (без 2:4 sparsity). Память GPU для бюджета KV — `usableMiB` из nvidia-smi.
- **Источники.** Каждое число в `data/*` и каждая константа в `perf.ts`/`router.ts`/`sched.ts` имеет
  комментарий-источник (страница книги, URL, раздел `research/notes-*.md`). Данные перенесены из
  `research/numbers.json` (генератор `research/scripts/numbers_calc.py`), тесты сверяют их с ним.
- **Идеализации roofline.** Шаг = max(байты / (BW·MBU), FLOPs / (пик·MFU)) + накладные. TP делит
  байты и FLOPs поровну без all-reduce; MoE маршрутизирует равномерно; attention «фьюзенный».
  Чтобы получить «нижнюю границу» из заметок, передайте `{ mbu: 1, mfu: 1, overheadMs: 0 }`.
- **Тексты для пользователя** (объяснения роутера, `fitsCheck().message`, имена паттернов голов,
  пояснения к игрушечному attention) — по-русски, числа как `Intl.NumberFormat('ru-RU')` (разряды
  неразрывным пробелом, десятичная запятая: «131 072 токена», «4,72 ТБ»), слово согласовано с числом,
  имя воркера в именительном падеже («Воркер 2 держит больше всего кэша»).

### Константы по умолчанию (`perf.ts`)

| Константа | Значение | Откуда |
|---|---|---|
| `DEFAULT_MFU` | 0.5 | допущение заметок: TTFT «при MFU около 50 %» (notes-models §3.12; notes-kvcache-tiers F6) |
| `DEFAULT_MBU` | 0.7 | между 50 % (Databricks) и 80 % (8B на H100), notes-hardware §3.16 |
| `DEFAULT_MEM_UTIL` | 0.9 | vLLM `gpu_memory_utilization` до v0.18 (сейчас 0.92), notes-engines §2.3; умолчание движка, на нём сверочные тесты с numbers_calc.py |
| `VLLM_MEM_UTIL` | 0.92 | умолчание vLLM v0.20 и новее (SPEC §8.8): главы, которые показывают ёмкость vLLM «из коробки», передают `memUtil: VLLM_MEM_UTIL` |
| `DEFAULT_RESERVE_GIB` | 2 | резерв на активации/CUDA graphs на GPU, notes-numbers §1.8, §3.15 |
| `STEP_OVERHEAD_MS` | 0.3 | **допущение**: планирование + запуск CUDA graph на шаг (notes-hardware §3.9) |
| `BYTES_PER_PARAM` | bf16 2, fp8 1, fp4 0.5625 | FP4 = NVFP4, 4.5 бита (NVIDIA NVFP4 blog; книга с. 124) |
| `KV_BYTES_PER_ELEM` | bf16 2, fp8 1, fp4 0.5625 | `--kv-cache-dtype` (notes-numbers §2.8) |

---

## `data/gpus.ts`

| Экспорт | Что это |
|---|---|
| `interface GpuSpecExt extends GpuSpec` | + `aliases`, `arch`, `memType`, `usableMiB`, `tdpW`, `nvlinkGen`, `hostLink`, `notes`, `featured` |
| `GPUS: GpuSpecExt[]` | a100-80g, h100-sxm, h200, b200, gb200, b300, l40s, l4, h100-nvl, gb300, rtx-4090, rtx-5090, mi300x, mi355x |
| `FEATURED_GPU_IDS` | `['h100-sxm', 'h200', 'b200']` — для селектора симулятора |
| `getGpu(id): GpuSpecExt` | по id или алиасу из numbers.json (`'h200-sxm'`, `'b200-hgx'`, `'a100-sxm-80gb'`); бросает ошибку |
| `findGpu(id): GpuSpecExt \| undefined` | то же без ошибки |

Если формат не поддержан тензорными ядрами (FP8 на A100, FP4 на Hopper), `tflops[dtype]` отсутствует,
и `perf.peakTflops` считает математику в BF16 (weight-only квантизация).

## `data/models.ts`

| Экспорт | Что это |
|---|---|
| `type ModelFamilyId` | `'qwen3' \| 'llama3' \| 'deepseek-v3' \| 'gpt-oss' \| 'mistral' \| 'gemma3' \| 'kimi-k2'` |
| `interface ModelSpecExt extends ModelSpec` | + `family`, `hfId`, `intermediate`, `activeMatmulParamsB` (для FLOPs), `lmHeadParamsB`, `expertParamsB` (MoE), `moeIntermediate`, `sharedExperts`, `vHeadDim` (MLA: 128), `kvLoraRank`/`qkRopeHeadDim` (MLA: 512/64), `native` (смешанный формат gpt-oss), `nativeContext`, `featured`, `notes` |
| `MODELS: ModelSpecExt[]` | обязательные: qwen3-8b, qwen3-32b, qwen3-30b-a3b, qwen3-235b-a22b, llama-3.1-8b, llama-3.1-70b, deepseek-v3, gpt-oss-120b; дополнительно llama-3.1-405b, gpt-oss-20b, kimi-k2, mistral-small-24b, gemma-3-27b |
| `FEATURED_MODEL_IDS` | `['qwen3-8b', 'qwen3-32b', 'llama-3.1-70b']` (SPEC §6.0) |
| `getModel(id)` / `findModel(id)` | по id (ошибка / undefined) |
| `slidingLayers(model): number` | сколько слоёв со скользящим окном (gpt-oss-120b: 18) |

Поля: `paramsB` — все параметры (так лежит в памяти); `activeParamsB` — как в карточке (у gpt-oss без
таблицы эмбеддингов, как у OpenAI); `vocab` — ширина логитов (Qwen3: 151 936, токенизатор 151 669).
DeepSeek-V3: `attention: 'mla'`, `mlaDim: 576`, `kvHeads: 1` условно. gpt-oss-120b: `attention: 'gqa-swa'`,
`slidingWindow: { window: 128, fullLayers: 18 }`, `native` = эксперты MXFP4 (0.53125 Б/пар.) + BF16.

## `data/interconnects.ts`

| Экспорт | Что это |
|---|---|
| `interface InterconnectSpecExt` | + `GBsBidirectional`, `practicalGBs` (PCIe: 25 / 50 ГБ/с реально), `kind`, `aliases` |
| `INTERCONNECTS` | nvlink5 (900), nvlink4 (450), nvlink3, nvlink-c2c, pcie6, pcie5 (63), pcie4 (31.5), ib-xdr (100), ib-ndr (50), ib-ndr-x8 (400), roce-400 (50), tcp-25g (3.125), nvme5 (14), nvme5-x8 (112), nvme4 (7), host-dram (307.2), hbm3-h100, hbm3e-b200 |
| `getInterconnect(id)` / `findInterconnect(id)` | по id или алиасу (`'pcie5-x16'`, `'roce-400g'`, `'nvme-gen5'`) |
| `KV_TRANSFER_LINK_IDS` | каналы для калькулятора передачи KV (SPEC §6.16, фигура 3) |
| `interface KvTier`, `KV_TIERS` | лестница G1 HBM → G2 RAM (по `pcie5`) → G3 NVMe → G4 сеть: `linkId`, ёмкость ноды DGX H100, подписи по-русски (глава 14) |

`host-dram` — полоса самой DRAM; GPU читает её через PCIe, поэтому время загрузки KV с уровня L2 считайте по `pcie5`.

---

## `perf.ts`

Все функции принимают `ModelSpec`/`GpuSpec` из `types.ts`; расширенные поля из `data/*` используются, если есть.

### Базовое

| Функция | Что делает |
|---|---|
| `bytesPerParam(dtype)` | байт на параметр |
| `peakTflops(gpu, dtype)` | пиковые dense TFLOPS (нет формата → BF16) |
| `usableMemBytes(gpu)` | память GPU по nvidia-smi (иначе `memGB` как ГиБ) |

### KV cache

| Функция | Что делает |
|---|---|
| `kvLayout(model, kvDtype='bf16'): KvLayout` | раскладка: байт на слой/токен, полные и sliding-слои, окно, константа sliding-слоёв |
| `kvBytesPerToken(model, kvDtype='bf16')` | байт KV на токен по всем слоям («наивно»): Qwen3-8B 147 456; Llama-3.1-8B 131 072; Llama-3.1-70B 327 680; DeepSeek-V3 (MLA, без ×2) 70 272 |
| `kvBytesForSeq(model, tokens, kvDtype='bf16', { swaAware=true })` | байт KV последовательности; sliding window: S·full + min(S, W)·sliding (gpt-oss-120b, 8k: 306 708 480) |
| `tpKvReplication(model, tp, dpAttention=false)` | дублирование KV при TP: TP > kv_heads → TP/kv_heads; MLA → ×TP; с DP-attention ×1 у любой модели (Qwen3-235B TP8: 115 → 230 запросов по 8k) |
| `effectiveCachedTokens(tokens, cached)` | сколько токенов реально берётся из prefix cache: не больше tokens − 1 (весь промпт в кэше → последний токен всё равно считается, как в vLLM) |

### Веса и FLOPs

| Функция | Что делает |
|---|---|
| `weightBytes(model, dtype='bf16')` | байт весов в памяти (gpt-oss с `'fp4'` — нативный чекпоинт ≈ 65,25 ГБ) |
| `stepWeightBytes(model, dtype, tokens=1)` | байт весов, читаемых одним шагом; MoE: всегда активная часть + задетые эксперты |
| `moeExpertsTouched(model, tokens)` | E·(1 − (1 − k/E)^B): сколько экспертов задето на слой |
| `attentionPairs(model, newTokens, context=0): AttnPairs` | пары (запрос, ключ) на слой: полный causal и sliding |
| `attentionFlopsPerPair(model, phase='prefill')` | 2·heads·(d_qk + d_v); MLA decode — absorbed-форма (278 528 у DeepSeek) |
| `prefillFlopsBreakdown(model, tokens, { cachedTokens })` | `{ linear, attention, total }`: 2·(N − N_lm)·S + 2·N_lm + attention |
| `prefillFlops(model, tokens, { cachedTokens })` | то же, итог (Llama-3.1-8B, 8192: 1,32·10¹⁴) |

### Время шага

| Функция | Что делает |
|---|---|
| `stepTime(model, gpu, work: StepWork, opts?: PerfOptions): StepBreakdown` | шаг со смесью prefill-кусков `{tokens, context, last}` и decode (`decode: number[]` длин или `decodeUniform: {batch, context}`); основа `sched.ts` |
| `prefillBreakdown(model, gpu, tokens, opts?: PrefillOptions)` | разбивка prefill одного промпта (`tokens` — полная длина, `cachedTokens` — сколько в кэше; полный хит → считается 1 токен, TTFT не 0) |
| `prefillMs(model, gpu, tokens, opts?)` | время prefill, мс: max(compute, память) + накладные |
| `decodeStepBreakdown(model, gpu, opts?: DecodeOptions)` | разбивка decode-шага (`batch`, `avgContext`) |
| `decodeStepMs(model, gpu, opts?)` | время decode-шага (= ITL), мс |
| `decodeLowerBoundMs(model, gpu, dtype='bf16', gpus=1)` | веса / полоса HBM: Llama-3.1-8B BF16 на H100 = 4,79 мс |
| `decodeBatchSweep(model, gpu, batches, opts?)` | точки «batch → шаг, ток/с на пользователя, ток/с всего, bound, влезает ли» (глава 9, фигура 2) |

`PerfOptions`: `dtype`, `kvDtype`, `mfu`, `mbu`, `gpus` (TP), `overheadMs`, `dpAttention`, `swaAware`.
`StepBreakdown`: `ms`, `memoryMs`, `computeMs`, `overheadMs`, `bytes`, `weightBytes`, `kvBytes`, `flops`,
`linearFlops`, `attentionFlops`, `intensity`, `bound: 'memory' | 'compute'`, `tokens`.

### Roofline

| Функция | Что делает |
|---|---|
| `arithmeticIntensity(model, { phase, batch, avgContext, tokens, dtype, kvDtype })` | `{ flops, bytes, intensity }` целого шага decode/prefill |
| `linearLayerIntensity(tokens, dIn, dOut, bytesW=2, bytesA=2)` | AI линейного слоя ≈ T (d = 4096, T = 64 → 62,1) |
| `decodeAttentionIntensity(model, kvDtype='bf16')` | AI attention в decode: ≈ heads/kv_heads (Llama-3.1-8B: 4), MLA ≈ 242 |
| `ridgePoint(gpu, dtype='bf16')` | FLOP/байт: H100 BF16 295,2; H200 206; B200 FP4 1125 |
| `attainableTflops(gpu, dtype, intensity)` | min(пик, AI × BW) |

### Передача и память

| Функция | Что делает |
|---|---|
| `transferMs(bytes, GBs, latencyMs=0)` | время передачи (8k KV Llama-3.1-70B по NDR 400G ≈ 53,7 мс) |
| `loadVsRecompute(model, gpu, tokens, linkGBs, opts?)` | `{ kvBytes, loadMs, recomputeMs, breakEvenGBs, loadIsFaster }` (глава 14); `linkGBs` — суммарная полоса всех каналов (TP8 по PCIe Gen5: 8 × 63) |
| `kvBudgetBytes(model, gpu, opts?: MemoryOptions)` | gpus·(память·memUtil − резерв) − веса |
| `maxConcurrentSeqs(model, gpu, { contextLen, ... })` | сколько последовательностей влезает (Llama-3.1-8B BF16 на H100, 8k: 54; FP8 KV: 109) |
| `kvCapacityTokens(model, gpu, opts?)` | ёмкость KV в токенах |
| `fitsCheck(model, gpu, { contextLen, seqs, ... }): FitsResult` | полоса «веса / резерв / KV / свободно», `fits`, `maxSeqs`, `message` по-русски |

`MemoryOptions`: `dtype`, `kvDtype`, `memUtil`, `gpus`, `reserveGiB`, `dpAttention`, `swaAware`.

---

## `kv.ts`

### Хэши блоков

| Экспорт | Что делает |
|---|---|
| `DEFAULT_BLOCK_SIZE = 16` | vLLM, все релизы ≤ v0.33 (в main 64) |
| `CHAIN_SEED = 1337` | как seed XXH3 в Dynamo (сама функция наша) |
| `blockHashes(tokenIds, blockSize=16, salt?): BlockInfoExt[]` | цепочка hash_i = H(hash_{i−1}, токены блока); `hash` — 8 hex; хвост короче блока: `full: false`, `hash: ''` (не хэшируется, как в vLLM и Dynamo); `fullHash` (16 hex), `parentHash`, `start` |
| `commonPrefixBlocks(a, b)` | сколько первых полных блоков совпадает |
| `cachedPrefixBlocks(blocks, set)` | сколько первых полных блоков есть в наборе хэшей |

Функция хэша — быстрый детерминированный 64-битный `hash64Words` (rng.ts). Настоящие системы:
Dynamo XXH3_64 seed 1337 (local + sequence hash), vLLM sha256 (`--prefix-caching-hash-algo`, с v0.11).
Значения у нас для показа, принцип (цепочка, только полные блоки, соль меняет всё) тот же.

### `class PagedAllocator` (PagedAttention)

`new PagedAllocator({ numBlocks, blockSize=16, freeOrder: 'ascending' | 'scattered' })`

| Метод | Что делает |
|---|---|
| `allocate(seqId, tokens): boolean` | ceil(tokens / B) блоков; false при нехватке |
| `append(seqId, n=1): boolean` | decode-токены; новый блок, когда последний полон; запись в общий неполный блок → copy-on-write; атомарно |
| `free(seqId)` | ref count −1, блоки с нулём в пул (в обратном порядке, как vLLM) |
| `fork(parentId, childId)` | делит все блоки (parallel sampling), ref +1 |
| `has`, `blockTable(seqId)`, `slotOf(seqId, pos)` | block table; слот = table[pos/B]·B + pos mod B |
| `sequences(): PagedSeqView[]`, `physicalBlocks(): PhysicalBlockView[]` | состояние для отрисовки (ref count, filled, owners) |
| `stats(): AllocatorStats` | утилизация, внутренние/внешние потери, `sharedSavedSlots` |
| `freeBlocks`, `lastEvents: AllocEvent[]` | свободные блоки; события последней операции (`note` по-русски) |

### `class ContiguousAllocator` (до PagedAttention)

`new ContiguousAllocator({ capacity, maxLen })`: `allocate(seqId, tokens, reserve=maxLen)` (first-fit,
непрерывно; бронь нулевой длины места не занимает), `append`, `free`, `has`, `regions(): ContiguousRegion[]` (занятые + дыры), `stats()`
(внешние потери = свободные слоты в дырах меньше `maxLen`), `lastEvents`.

`AllocatorStats`: `capacitySlots`, `allocatedSlots`, `usedSlots`, `logicalTokens`, `freeSlots`,
`internalWasteSlots`, `externalWasteSlots`, `utilization` (= used/allocated), `occupancy`,
`internalFragmentation`, `externalFragmentation`, `sequences`, `sharedSavedSlots`.

### `class RadixTree` (SGLang RadixAttention)

| Метод | Что делает |
|---|---|
| `insert(tokens): RadixInsertResult` | `{ matchedTokens, insertedTokens, totalTokens, path }`; совпавший префикс переиспользуется |
| `match(tokens): PrefixMatch` | самый длинный префикс; узлы пути `hit`, «трогаются» для LRU; совпадение посреди ребра **расщепляет** узел (как SGLang) |
| `lock(matchOrNodeId)` / `unlock(...)` | ref +1/−1 на узле и предках (активный запрос) |
| `evictLRU(capacityTokens): RadixEvictResult` | вытесняет LRU-листья без ссылок, пока размер > ёмкости; предки становятся листьями позже |
| `view(): RadixNodeView` | снимок дерева (дети по первому токену), `hit` — участие в последней операции |
| `size`, `now`, `nodeCount()`, `protectedTokens()` | токенов в дереве; логические часы; узлов; заблокировано |

Размер страницы 1 токен (SGLang `--page-size` по умолчанию 1).

---

## `router.ts`

Формула Dynamo ≥ 1.2 (см. `RouterOptions` в types.ts, notes-dynamo §1.8):
`cost = prefillLoadScale·max(0, (queued + ISL)/B − credit) + (activeBlocks + newBlocks) + w·activeRequests`,
`credit = c·device + 0.75·host + 0.25·disk (+ 0.5·shared_beyond при общем пуле)`. Перекрытие — самый длинный
префикс цепочки хэшей запроса, лежащий у воркера (каждый блок на лучшем уровне: GPU, затем host, затем disk).
`newBlocks` (additional_active_blocks) — **точно по коду Dynamo** (`lib/router/slot-tracker/src/prompt_registry.rs`):
хэши ПОЛНЫХ блоков запроса (floor(ISL/B), хвост не хэшируется) минус совпадение по префиксу с блоками
**активных** запросов воркера (`RouterWorker.activeBlockHashes`). Prefix cache сам по себе decode-слагаемое
не уменьшает: кэш — скидка только на prefill (notes-dynamo §1.8 «а не промахи кэша»). Активные блоки ведут
`applyRoute` / `releaseRoute` с ref count (`activeBlockRefs`): блок освобождается, когда закончился последний
державший его запрос. Пример доков §3.3 (potential decode 10/5/9 при 10 блоках запроса) воспроизводится,
только если закэшированный префикс держат активные запросы — передайте `activeBlockHashes`.
T = 0 → argmin (ничья: первый по порядку; в Dynamo случайно); T > 0 → softmax по min-max нормированным
стоимостям. `'legacy'` (≤ v1.1): `legacyOverlapWeight·(ISL/B − device) + activeBlocks`.

| Экспорт | Что делает |
|---|---|
| `DEFAULT_ROUTER_OPTIONS` | c = 1, T = 0, scale = 1, w = 0, `'v1.2'` (дефолты Dynamo) |
| `HOST_CACHE_HIT_WEIGHT = 0.75`, `DISK_CACHE_HIT_WEIGHT = 0.25`, `SHARED_CACHE_MULTIPLIER = 0.5` | кредиты уровней и общего пула |
| `interface RouterWorker extends WorkerState` | + `activeBlockHashes?: Set<string>` (блоки активных запросов), `activeBlockRefs?: Map<string, number>` (их ref count) |
| `interface KvRouterOptions extends RouterOptions` | + `blockSize`, `assumeKvReuse`, `hostCacheHitWeight`, `diskCacheHitWeight`, `sharedPoolBlocks` (общий пул кластера, §3.5б), `sharedCacheMultiplier`, `rng` |
| `interface KvWorkerScore extends WorkerScore` | + `name`, `hostOverlapBlocks`, `diskOverlapBlocks`, `sharedBeyondBlocks`, `rawPrefillBlocks`, `adjustedPrefillBlocks`, `newBlocks`, `activeBlocks`, `queuedPrefillTokens`, `activeRequests`, `activeRequestsTerm`, `cachedTokens`, `prefillTokens` |
| `interface KvRouteDecision extends RouteDecision` | + `isl`, `blockSize`, `formula`, `temperature`, `fullBlockHashes` (для releaseRoute), `assumeKvReuse` |
| `routeKV(blocks, workers, opts?): KvRouteDecision` | KV-aware выбор + `explanation` по-русски. Сравнение с воркером, у которого больше кэша, печатается, только если он дороже выбранного; при T > 0, если выпал не самый дешёвый, объяснение называет самого дешёвого. Шансы в процентах как `fmt.pct` («<0,1 %», «5,4 %», «44 %») |
| `routeRoundRobin(blocks, workers, counter, opts?)` | запрос №counter → воркер counter mod N (стоимости в scores для сравнения) |
| `routeRandom(blocks, workers, seedOrRng, opts?)` | равновероятно, детерминированно |
| `scoreWorker(blocks, worker, opts?)`, `overlapByTier(blocks, worker)` | стоимость одного воркера; перекрытие по уровням |
| `selectionProbs(costs, temperature)` | вероятности выбора ([18, 10, 11], T = 1 → [0,1635; 0,4444; 0,3922]) |
| `applyRoute(workers, decision, blocks, { capacityBlocks })` | новые состояния: блоки в кэш (LRU-порядок Set), блоки запроса становятся активными (ref +1, `activeBlocks` += ранее не занятые), +prefill-токены, +1 запрос; вытеснение в host |
| `releaseRoute(workers, decision, 'prefill' \| 'finish')` | снять prefill-токены / ref активных блоков (освобождаются только никем не занятые) и слот |
| `cacheBlocks(worker, blocks)` | тёплый кэш без нагрузки (системный промпт уже лежит) |
| `makeWorkers(n, { capacityBlocks, withHostTier })` | пустые воркеры `worker-0…`, имена «Воркер 1…», с пустым учётом активных блоков |
| `simulateRouting(requests, workers, opts): RoutingSimResult` | один поток через `kv` / `round-robin` / `random`: hit rate, mean/p50/p90/p99 TTFT (perf.ts), GPU-время prefill, нагрузка по воркерам, дисбаланс |
| `makeTenantWorkload(opts): RoutingRequest[]` | поток «у каждого тенанта свой системный промпт» (настоящие блоки и хэши, синтетические id) |

`RoutingSimOptions`: `router`, `model`, `gpu`, `dtype`, `kvDtype`, `mfu`, `mbu`, `gpus`, `capacityBlocks`, `strategies`, `seed`.
Модель воркера в симуляции: FIFO-очередь prefill (perf.prefillMs с кэшем), decode = outputTokens × decodeStepMs.

---

## `sched.ts`

| Экспорт | Что делает |
|---|---|
| `quantile(values, q, method='linear' \| 'nearest-rank')`, `percentile(values, p)` | перцентили (numpy-интерполяция по умолчанию) |
| `sortedValues(values)`, `quantileSorted(sorted, q, method?)` | отсортировать выборку один раз (Float64Array) и брать из неё несколько перцентилей |
| `mean(values)`, `histogram(values, { bins, min, max })` | среднее; гистограмма (глава 10) |
| `goodput(result, { ttftMs, tpotMs })` | доля и req/s в SLO |
| `type Dist`, `sampleDist(dist, rng)` | распределения длин: число, const, uniform, normal, lognormal, choice |
| `makeWorkload({ rate, promptDist, outputDist, seed, count, durationMs, sharedPrefixes })` | пуассоновский поток; общие префиксы → `cachedTokens` (целые блоки); при том же seed длины одинаковы, а моменты прихода масштабируются как 1 / rate |
| `vllmDefaultLimits(gpu)` | `{ tokenBudget, maxSeqs }` как `vllm serve`: H100 8192/1024, B200 16384/1024, A100/L40S 2048/256 |
| `simulateContinuous(reqs, ContinuousOptions): SimResultExt` | vLLM V1: RUNNING первыми, WAITING в остаток бюджета, chunked prefill, смешанные шаги, recompute-preemption; `workers > 1` — агрегированный serving |
| `simulateStatic(reqs, StaticOptions)` | static batching (ждём batch, слоты простаивают до конца самого длинного); `maxWaitMs` — dynamic batching |
| `simulateDisagg(reqs, DisaggOptions)` | xPyD: prefill-воркеры → передача KV по `link` (по очереди с каждого воркера) → decode-воркеры (при нехватке KV тоже recompute-preemption, пересчёт идёт на decode-воркере); первый токен на decode (vLLM/Dynamo) или `firstTokenOn: 'prefill'` (SGLang) |
| `simulateQueue(reqs, QueueOptions)` | FCFS-очередь из `servers` реплик для хвостов TTFT; + `utilization` |
| `summarize(requests, gpuCount)` | сводка (`SimSummaryExt`) |

`ContinuousOptions`: `model`, `gpu`, `tokenBudget`, `maxSeqs`, `chunkedPrefill` (true), `workers`, `kvCapacityTokens`
(по умолчанию из памяти GPU), плюс `PerfOptions`. Без chunked prefill длинный промпт идёт **целиком** одним
шагом (если это единственный prefill в шаге), отсюда всплеск ITL.
`DisaggOptions`: `prefillWorkers`, `decodeWorkers`, `link` (InterconnectSpec или ГБ/с), `linkLatencyMs`, `tokenBudget`, `maxSeqs`, `firstTokenOn`.

Результат `SimResultExt`: `steps: SimStepRecord[]` (для Gantt: `prefill` куски, `decode` id, `worker`, `kvTokens` — KV движка к концу шага, ≤ ёмкости, если в шаге больше одного запроса),
`requests: SimRequestStats[]` (+ `promptTokens`, `outputTokens`, `cachedTokens`, `worker`, `prefillWorker`,
`e2eMs`, `queueMs`, `preemptions`), `summary` (+ `ttftMean/P90`, `itlMean/P90`, `e2eP50/P99`, `tpotMean`,
`gpuCount`, `throughputPerGpuTokS`, `preemptions`), `transfers: KvTransferRecord[]`, `label` по-русски.
ITL всех запросов сливаются в один список (как `vllm bench serve`).

---

## `attention.ts`

| Экспорт | Что делает |
|---|---|
| `zeros`, `shape`, `matmul`, `transpose`, `add`, `scale`, `mapMatrix`, `dot`, `dotTerms`, `roundMatrix`, `rowSums` | матричные помощники; `dotTerms(a, b)` → слагаемые для подсказки «как посчитана ячейка» |
| `causalMask(n)`, `applyMask(A, mask)`, `softmaxRows(A, mask?)` | маска (true при j ≤ i), −∞ вне маски, устойчивый softmax |
| `mat` | все помощники одним объектом (`mat.matmul`, `mat.softmaxRows`, …) |
| `attention(X, Wq, Wk, Wv, { causal=true, scale=true }): AttentionTrace` | одна голова по шагам: Q, K, V, rawScores, scaled, masked, weights, out |
| `TOY_ATTENTION: ToyAttention` | игрушечный пример главы 6 (ниже) |
| `headPatterns(tokens): HeadPattern[]` | паттерны голов для любого промпта: `previous-token`, `attention-sink`, `local-window`, `duplicate-token`, `induction`, `delimiter`, `broad`; causal, строки в сумме 1, имена и описания по-русски |
| `gqaGroups(heads, kvHeads): GqaGroup[]`, `attentionKind(heads, kvHeads)` | раскладка Q-голов по KV-головам (32/8 → 8 групп по 4); `'MHA' \| 'GQA' \| 'MQA'` |

### `TOY_ATTENTION`

«Кошка не перепрыгнула забор, потому что она устала» (9 слов, запятая оставлена отдельным токеном).
Признаки `featureNames`: существительное, женский род, местоимение, глагол. d_k = 2:
`queryDimNames` ищу предмет / ищу женский род; `keyDimNames` я предмет / я женского рода;
`valueDimNames` существительное / женский род. Все числа матриц с одним знаком после запятой максимум.
Поля: `tokens`, `X`, `Wq`, `Wk`, `Wv`, `dK`, `focus` (`query` = «она», `target` = «Кошка»,
`distractor` = «забор»), `explanations` (по шагу: X, Wq, Wk, Wv, Q, K, V, scores, scale, mask, softmax, out),
`note`, `trace` (causal, /√2).

Строка «она» (causal, /√2): Кошка 0,650 · не 0,019 · перепрыгнула 0,038 · **забор 0,078** · «,» 0,019 ·
потому 0,019 · что 0,019 · она 0,158 · устала 0 (маска). «Устала»: Кошка 0,370, она 0,182.
«Перепрыгнула»: Кошка 0,734. Выход «она» = [0,73; 0,85]: признак «существительное» пришёл от «Кошки».

---

## `rng.ts`

| Экспорт | Что делает |
|---|---|
| `type Rng = () => number` | генератор чисел в [0, 1) |
| `mulberry32(seed)`, `makeRng(seed: number \| string)` | ГПСЧ (mulberry32(1)() = 0,62707…) |
| `hashString(s)`, `fmix32(h)` | 32-битный хэш строки (FNV-1a + murmur fmix32) |
| `hash64Words(words, seed=0): [hi, lo]` | 64-битный хэш 32-битных слов (цепочка блоков KV) |
| `hex32(n)`, `shortHash(string \| number \| number[])` | 8 hex |
| `randInt(rng, lo, hi)`, `pick(rng, arr)`, `pickWeighted(rng, weights)`, `shuffle(rng, arr)` | выборки (shuffle возвращает новый массив) |
| `normal(rng, mean, sd)`, `logNormal(rng, median, sigma)`, `exponential(rng, rate)` | распределения |

---

## Рецепты по главам

| Глава / фигура | Что вызвать |
|---|---|
| Симулятор: route | `blockHashes(promptIds, 16)`, `makeWorkers(4)`, `cacheBlocks` (тёплый кэш), `routeKV(blocks, workers, { overlapCredit: 1, temperature: T, seed })` |
| Симулятор: prefill / transfer / decode | `prefillMs(m, g, total, { cachedTokens })`, `kvBytesForSeq(m, total)`, `transferMs(bytes, getInterconnect('ib-ndr').GBs)`, `decodeStepMs(m, g, { batch: 1, avgContext: total + i })` |
| 8 prefill-decode, 9 roofline | `prefillBreakdown`, `decodeStepBreakdown`, `ridgePoint`, `attainableTflops`, `arithmeticIntensity`, `decodeBatchSweep`, `linearLayerIntensity`, `decodeAttentionIntensity` |
| 10 metrics | `makeWorkload` + `simulateQueue` / `simulateContinuous`, `histogram`, `percentile`, `goodput` |
| 11 kv-cache | `kvLayout`, `kvBytesPerToken`, `kvBytesForSeq`, `fitsCheck`, `maxConcurrentSeqs` |
| 12 paged-attention | `PagedAllocator` (`freeOrder: 'scattered'`), `ContiguousAllocator`, `stats()`, `lastEvents` |
| 13 prefix-caching | `blockHashes`, `commonPrefixBlocks`, `RadixTree` (`insert`/`match`/`evictLRU`/`view`), `prefillMs(..., { cachedTokens })` |
| 14 kv-tiers | `KV_TIERS`, `loadVsRecompute(m, g, tokens, getInterconnect(tier.linkId).GBs)` |
| 15 batching | `simulateStatic`, `simulateContinuous({ chunkedPrefill, tokenBudget })`, `steps` для Gantt |
| 16 disaggregation | `simulateContinuous({ workers: N })` против `simulateDisagg({ prefillWorkers, decodeWorkers, link })`, `KV_TRANSFER_LINK_IDS` |
| 17 parallelism | `fitsCheck(m, g, { gpus, dtype })`, `tpKvReplication`, `decodeLowerBoundMs(m, g, dtype, gpus)` |
| 20 routing | `routeKV` / `routeRoundRobin` / `routeRandom`, `selectionProbs`, `makeTenantWorkload` + `simulateRouting` |
| 6 attention | `TOY_ATTENTION`, `attention`, `dotTerms`, `headPatterns(tokens)`, `gqaGroups` |

## Проверено тестами (числа из заметок)

KV/токен: Qwen3-8B 147 456, Llama-3.1-8B 131 072, Llama-3.1-70B 327 680, DeepSeek-V3 70 272, gpt-oss-120b 8k SWA
306 708 480. Decode при batch 1 (идеально): Llama-3.1-8B BF16 H100 4,794 мс, H200 3,346, B200 2,008, FP8 H100 2,397;
70B FP8 TP2 10,53; Qwen3-30B-A3B 2,002; gpt-oss-120b нативно ≈ 1,49. Throughput/latency S = 2048: batch 1/8/32/64/128/200 →
4,87/5,44/7,36/9,92/15,05/20,82 мс; S = 256, batch 1024 → compute-bound 15,68 мс. Prefill Llama-3.1-8B 8k: 1,1435·10¹⁴ +
1,7594·10¹³ FLOPs, 133,41 мс на 100 % пика; 70B FP8 8k 611 мс; DeepSeek-V3 FP8 H200 1k 38,3 мс; gpt-oss-120b 8k 85,7 мс.
Ridge H100 BF16 295,2. Сколько 8k-запросов влезает: 54 / 109 (FP8 KV) / 70B TP2 29 / H200 23 / Qwen3-235B TP8 115 /
DeepSeek-V3 TP8 86, DP-attention 690 / gpt-oss-120b 31 (15 без SWA). Роутер: пример доков 18/10/11 → W2 и softmax;
3.4 (720/564/676), 3.5а (396/380), legacy 56/74 и 48/42, перекрытие 6/4/0. Перцентили на 20 замерах (468/975/2115).

Независимая проверка (`*.verify.test.ts`): все 75 тест-векторов `research/numbers.json` → `checks` (23 нижних
границы decode, 28 конфигураций «сколько 8k-запросов влезает», 24 prefill с FLOPs по частям), ridge point всех GPU;
роутер: вся таблица softmax §3.3 (T = 0,1…2), §3.4 с вариациями, §3.5а, §3.5б (общий пул), notes-kvcache-tiers §3.9
(844/922/1078/1156, занятый A → 1244, выигрывает B); property-тесты RadixTree (совпадение = брутфорс LCP, refs =
блокировки в поддереве, вытеснение не трогает заблокированное), PagedAllocator (ref count, CoW, атомарность, нет утечек),
цепочки хэшей (правка в блоке k меняет все блоки ≥ k); sched: детерминизм, монотонность TTFT по rate (FCFS — по каждому
запросу), сохранение токенов, ёмкость KV в agg и disagg; TOY_ATTENTION и headPatterns на 200 случайных промптах.

## Ограничения и упрощения

- TP без all-reduce; PP/EP не моделируются (кроме дублирования KV при TP).
- Prefill одного промпта считается одним шагом (если не через `sched`), шаги без overlap scheduling.
- В `simulateRouting` учитываются только попадания в GPU-кэш; host/disk-кредиты работают в `routeKV`.
- `decay` кредита и preferred taint роутера Dynamo не моделируются; общий пул (×0.5) — через `sharedPoolBlocks`.
- `kvCapacityTokens` (ёмкость KV в симуляциях) считает «наивные» байты на токен: у SWA-моделей (gpt-oss, Gemma 3) ёмкость занижена.
- `overlapByTier`: блок засчитывается на лучшем уровне, где он есть; блок на GPU после блока, лежащего только в host, считается device (у Dynamo точная семантика уровней в доках не описана).
- Дубликаты блоков vLLM V1 и page size > 1 в `RadixTree` не моделируются.
