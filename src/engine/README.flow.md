# Движок: прогон запроса для симулятора (агент S1a «flow»)

`buildRun(settings, tokenizer)` собирает всё, что показывает симулятор «Путь запроса», из настоящих модулей движка.
Симулированы только текст ответа модели (пресеты и заготовки по темам) и логиты кандидатов; всё остальное посчитано
по-настоящему: шаблон Qwen3, токены словаря Qwen3, хэши блоков, решение роутера Dynamo, время prefill и шагов decode
из `perf.ts`, маски structured output, детокенизация, парсеры, SSE. Чистая детерминированная функция, без React и DOM.

```ts
import { useMemo } from 'react'
import { useSettings } from '@/state/settings'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { buildRun, STAGE_INFO, pipelineStages } from '@/engine/flow'
import { SCENARIOS, scenarioPatch, resolveScenario } from '@/engine/scenarios'

const [settings, patch] = useSettings()
const { tokenizer, status } = useTokenizer()          // пока null: скелетон
const run = useMemo(() => (tokenizer ? buildRun(settings, tokenizer) : null), [settings, tokenizer])
```

Скорость: около 3 мс на прогон после прогрева (первый вызов около 30 мс: пулы кандидатов, промпты тенантов, JIT).
Пересчитывать на каждое изменение настроек можно; ввод текста можно не дебаунсить.

| Файл | Что внутри |
|---|---|
| `flow.ts` | `buildRun`, модели и GPU симулятора, константы времени, `STAGE_INFO`, помощники для UI |
| `scenarios.ts` | вкладки, заскриптованные ответы, темы своего текста, `resolveScenario`, `WEATHER_TOOL` |
| `flow-candidates.ts` | кандидаты на шаге decode: пулы, классы токенов, синтетические логиты |
| `flow-cluster.ts` | кластер: тенанты в кэше, нагрузка, тёплый воркер, роутинг agg и disagg |
| `flow.test.ts` | 123 теста: все сценарии × agg/disagg × холодный/тёплый × 4 модели, инварианты, длинный документ, темы, скорость |

---

## API `flow.ts`

| Экспорт | Что это |
|---|---|
| `buildRun(settings, tokenizer, opts?): FlowRunEx` | полный прогон; `opts.maxTokens` (по умолчанию 512) задаёт `max_tokens` запроса, меньше длины ответа даёт `finish_reason: 'length'` |
| `SIM_MODEL_IDS`, `SIM_MODELS` | `qwen3-8b`, `qwen3-32b`, `qwen3-30b-a3b`, `qwen3-235b-a22b` (только Qwen3: у них настоящие токенизатор и шаблон) |
| `SIM_GPU_IDS`, `SIM_GPUS` | `h100-sxm`, `h200`, `b200`, `a100-80g` (для селектора берите первые три) |
| `DEFAULT_SIM_MODEL_ID`, `DEFAULT_SIM_GPU_ID` | `'qwen3-8b'`, `'h100-sxm'` |
| `resolveSimModel(id)` | модель симулятора; не Qwen3 (например, `llama-3.1-70b` из другой главы) → Qwen3-8B, `fallback: true`, в `run.notes` пояснение |
| `resolveSimGpu(id)` | любой GPU из `data/gpus` (по id или алиасу), неизвестный → H100 |
| `simTensorParallel(model, gpu)` | `{ tp, fits, message }`: наименьший TP из {1, 2, 4, 8}, при котором влезают веса BF16 и бюджет KV `SIM_KV_REQUIREMENT` (8 последовательностей по 8k, условно), через `perf.fitsCheck` |
| `DEFAULT_MAX_TOKENS` | 512 |
| `QWEN3_GENERATION_CONFIG` | `{ temperature: 0.6, topP: 0.95, topK: 20 }`: рекомендованный сэмплинг из generation_config.json Qwen3; поля, которых нет в запросе, воркер vLLM берёт отсюда |
| `FLOW_BLOCK_SIZE` | 16 (vLLM по умолчанию, SPEC §8.8) |
| `SIM_TP_OPTIONS`, `SIM_KV_REQUIREMENT` | см. выше |
| `FLOW_TIMING` | условные константы времени (таблица ниже) |
| `STAGE_INFO[stage]` | `{ title, what, chapters }`: подпись узла, одна фраза «что происходит», id глав для «Подробнее» |
| `pipelineStages(deployment)` | узлы схемы по порядку; в agg без `transfer` (prefill и decode на одном воркере) |
| `sseUpTo(run, n)` | первые n SSE-чанков |
| `sseCountAt(run, tMs)` | сколько чанков клиент получил к моменту tMs |
| `tokenIndexAt(run, tMs)` | последний выходной токен, дошедший до клиента к tMs (−1, если ни одного) |
| `cachedBlockFlags(run)` | для каждого блока промпта: лежит ли он в кэше выбранного (prefill-)воркера |
| типы `DecodeStepEx`, `FlowFrontendInfoEx`, `FlowInjection` | расширения для `FlowRunEx` (G0, ниже) |

TP по факту: Qwen3-8B и 30B-A3B везде 1; Qwen3-32B: H100 и A100 по 2, H200 и B200 по 1; Qwen3-235B-A22B: H100 и A100 по 8,
H200 и B200 по 4.

## API `scenarios.ts`

| Экспорт | Что это |
|---|---|
| `SCENARIOS: ScenarioInfo[]` | вкладки по порядку: `simple`, `thinking`, `tool`, `json`, `custom`; `{ id, label, description, userText, systemText, toggles }` |
| `getScenario(id)` | вкладка по id |
| `scenarioPatch(id)` | что записать в настройки при клике по вкладке: у пресета текст, system prompt и переключатели; у `custom` только `{ scenarioId: 'custom' }` |
| `resolveScenario(settings)` | сценарий, по которому реально строится ответ: пресет, пока текст пользователя совпадает с его текстом (без учёта регистра, лишних пробелов, финальной `?!.`); изменённый текст → `'custom'`; текст пресета, набранный на вкладке «Свой текст», → этот пресет |
| `scriptFor(settings): Script` | заскриптованный ответ: рассуждение (без `<think>`), тело (`text` / `tool_call` / `json`), JSON-схема, заметки |
| `WEATHER_TOOL` | инструмент `get_weather(city)`, его получает модель при `tools: true` |
| `detectTopic(text)`, `extractTopicPhrase(text)`, `findCity(text)`, `guessCity(text)`, `sanitizeEcho(text)` | темы своего текста, фраза темы, город для погоды (словарь основ; `guessCity`: слово с большой буквы после «в» / «in», как в тексте), очистка эха от литералов спецтокенов |
| `LONG_DOC_TAB`, `LONG_DOC_SYSTEM`, `longDocPatch()`, `isLongDoc(settings)` | вкладка «Длинный документ»: system prompt-регламент и вопрос по нему. Отдельного `ScenarioId` нет: в настройках это `custom` с длинным system prompt, `scriptFor` узнаёт его по `isLongDoc` и даёт заготовленный ответ (`answerSource: 'preset'`). Описание вкладки честно делит модели: у плотных холодный prefill здесь упирается в вычисления, у MoE нет |
| `LONG_DOC_PROMPT_TOKENS` | 1 167: токенов в промпте вкладки с её переключателями (закреплено тестом). С ним сравнивают `run.prefillThreshold`: если порог выше, вкладка упор prefill не сменит (так у MoE на H100, H200 и B200) |
| `piecesText(pieces)` | склейка кусков скрипта |

Вкладки: «Простой вопрос» («Сколько будет 2+2?»), «С размышлением» (вопрос про KV cache из настроек по умолчанию),
«Вызов инструмента» («Какая погода сейчас в Берлине?», tools), «JSON по схеме» («Извлеки данные: Анна, 29 лет, живёт в Казани»,
json), «Свой текст». У всех system prompt «Ты полезный ассистент.».

---

## FlowRun: поля

Контракт в `src/engine/types.ts` §11. Поля с пометкой (S1a) добавлены агентом S1a: все необязательные в типе, но `buildRun`
заполняет их всегда (кроме помеченных «только disagg» / «только json»).

Что добавлено в `types.ts` §11 (только необязательные поля и новые типы для них, существующие поля не менялись):
- новые типы: `DecodeCandidate` (кандидат шага: прежние `token`, `logit`, `prob`, `allowed` + `rawProb?`, `preview?`; `DecodeStep.candidates`
  теперь `DecodeCandidate[]`, структурно то же самое), `FlowStageSummary`, `FlowPlaybackStep`, `FlowWorkerInfo`, `FlowClusterInfo`,
  `FlowFrontendInfo`, `FlowPreprocessInfo`;
- `TimelineItem`: `detail?`, `simulated?`;
- `DecodeStep`: `phase?`, `chosenIndex?`, `entropyBits?`, `maskedCount?`, `grammarExpecting?`, `kvTokens?`, `kvBytes?`, `batch?`, `emitMs?`,
  `clientMs?`, `stop?`, `note?`;
- `FlowRun.prefill`: `workerName?`, `cachedBlocks?`, `kvTotalBytes?`, `layers?`, `layerMs?`, `bound?`, `computeMs?`, `memoryMs?`, `flops?`;
- `FlowRun.transfer`: `linkId?`, `GBs?`, `links?`, `setupMs?`, `params?`, `fromName?`, `toName?`;
- `FlowRun.metrics`: `ttftVisibleMs?`, `perUserTps?`, `prefillMs?`, `decodeMs?`;
- `FlowRun`: `effectiveScenarioId?`, `answerSource?`, `topic?`, `simulatedNote?`, `modelId?`, `gpuId?`, `gpus?`, `fits?`, `fitsMessage?`,
  `maxTokens?`, `seed?`, `notes?`, `frontend?`, `preprocess?`, `cluster?`, `decodeWorkers?`, `decodeRouting?`, `output?`, `sseStep?`,
  `sseMs?`, `stages?`, `playback?`.

`buildRun` возвращает `FlowRunEx` (экспорт `flow.ts`, совместим с `FlowRun`) с точными типами модулей: `routing` и
`decodeRouting` это `KvRouteDecision` (у очков `queuedPrefillTokens`, `rawPrefillBlocks`, `newBlocks`, `cachedTokens`; у решения `isl`),
`workers` / `decodeWorkers` это `RouterWorker[]`, `detok` это `DetokStepEx[]` (`heldBytes`, `output`, `stopReason`), `parser` это
`ParserEventEx[]` (`out`, `toolDeltas`, `channelAfter`, `toolCalls`). Приведения типов не нужны.

### Вход и запрос

| Поле | Что |
|---|---|
| `settings` | настройки, как пришли (тот же объект) |
| `request` | JSON OpenAI chat.completions: `model` (HF id, `Qwen/Qwen3-8B`), `messages` (system опускается, если пустой), `tools` + `tool_choice: "auto"` (при tools), `response_format: {type: "json_schema", json_schema: {name, schema, strict: true}}` (при json), `temperature`, `max_tokens`, `stream: true`, `stream_options: {include_usage: true}`, `chat_template_kwargs: {enable_thinking}` |
| `effectiveScenarioId` (S1a) | итог `resolveScenario` |
| `answerSource` (S1a) | `'preset'` / `'topic'` (заготовка по теме своего текста) / `'generic'` (общий ответ) |
| `topic` (S1a) | тема своего текста («batching», «погода», фраза из вопроса) |
| `simulatedNote` (S1a) | честная подпись: что симулировано, что посчитано (показывайте рядом с ответом) |
| `modelId`, `gpuId`, `gpus` (S1a) | реально использованные модель, GPU и TP (GPU на воркер) |
| `fits`, `fitsMessage` (S1a) | влезает ли модель при этом TP, вывод `fitsCheck` по-русски |
| `maxTokens`, `seed` (S1a) | `max_tokens` запроса; seed синтетических логитов (из сценария, текста и переключателей) |
| `prefillThreshold` (X-sim-r2-1, `FlowRunEx`) | порог упора prefill в вычисления, токенов: `total × memoryMs / computeMs` для всего промпта без кэша (у холодного и тёплого прогона одно число), до десятков. Плотные модели: 150–250 токенов; MoE: 1 480–2 270 (токены промпта вместе задевают почти всех экспертов, и prefill читает почти все веса, а считает только по активным экспертам) |
| `notes` (S1a) | заметки по-русски к этому прогону: замена модели, «tools в промпте, но вызова нет», «JSON перекрывает вызов инструмента», «грамматика включается после `</think>`», «первый токен в disagg выбирает decode-воркер», обрезка по max_tokens |

### frontend (S1a): `run.frontend`

`checks[]` (`{ label, ok, detail }`: тело, модель, сообщения, temperature, стрим, бюджет контекста `prompt + max_tokens ≤ 40 960`,
tools, response_format), `sampling` (`temperature` из запроса, `topP: 0.95`, `topK: 20`, `seed: null`), (G0) `samplingFromModel: ['top_p', 'top_k']`
и `generationConfig`: top_p и top_k в запросе нет, frontend Dynamo передаёт их пустыми, а воркер vLLM подставляет из generation_config.json
модели (`--generation-config auto` по умолчанию). Кандидатам симулятор применяет только temperature. `stop` (`maxTokens`,
`stopTokenIds [151645, 151643]`, их тексты, `stopStrings: []`), `guidedJson` (схема, только json), `parsers` (`qwen3` / `hermes`), `chatTemplateKwargs`.

### preprocess: `template`, `promptTokens`, `run.preprocess` (S1a)

| Поле | Что |
|---|---|
| `template` | `renderTemplate` Qwen3 (`text`, `segments` с видами special / role / content / tools / gen-prompt / format, `warnings`) |
| `promptTokens` | `tokenizer.encode(template.text, { allowSpecial: true })` (со смещениями `start`/`end` в `template.text`) |
| `preprocess.segmentOfToken[i]` | индекс сегмента шаблона для токена i: раскраска `TokenStrip` по сегментам |
| `preprocess.templateMs`, `tokenizeMs`, `chars` | условное время CPU и длина промпта в символах |
| `preprocess.toolsTokens` | сколько токенов добавило описание tools (около 150 у `get_weather`), 0 без tools |
| `preprocess.contextLimit` | контекст модели (40 960 у Qwen3) |
| `preprocess.preprocessedRequest` | упрощённый `PreprocessedRequest` Dynamo: `model`, `token_ids` (все), `stop_conditions`, `sampling_options` (только `temperature`; при json внутри `guided_decoding: { json }`, как в Dynamo), `output_options`, `eos_token_ids` |
| `injection` (G0) | `FlowInjection[]`: спецтокены, пришедшие из литералов в тексте сообщения (`<\|im_end\|>`, `<think>` и т. п.), с индексом в `promptTokens` и ролью сообщения; токены шаблона и описания tools сюда не попадают. При непустом списке в `notes` первым идёт пояснение про инъекцию спецтокенов |

### route: `blockSize`, `blocks`, `workers`, `routing`, `cluster`, `decodeWorkers`, `decodeRouting`

| Поле | Что |
|---|---|
| `blockSize` | 16 |
| `blocks` | `blockHashes(promptIds, 16)`: цепочка, 8 hex; неполный хвост `full: false`, `hash: ''` (не хэшируется) |
| `workers` | agg: 4 агрегированных воркера `worker-0…3` («Воркер 1…4»); disagg: пул prefill `prefill-0…3` («Prefill 1…4»). `RouterWorker`: `cachedBlocks` (Set хэшей), `activeBlocks`, `queuedPrefillTokens`, `activeRequests`, `capacityBlocks` |
| `routing` | `KvRouteDecision` (router.ts): `scores[]` с `overlapBlocks`, `prefillBlocks`, `decodeBlocks`, `credit`, `prefillTerm`, `decodeTerm`, `cost`, `prob`, `rawPrefillBlocks`, `queuedPrefillTokens`, `cachedTokens`; `chosen`; `explanation` по-русски (объяснение роутера + история кэша); `isl`, `fullBlockHashes`. agg: `routeKV` с дефолтами Dynamo (c = 1, T = 0); disagg: PrefillRouter, стоимость без decode-слагаемого (`decodeTerm = 0`, track_active_blocks = false) |
| `decodeWorkers`, `decodeRouting` (S1a, только disagg) | пул decode `decode-0…1` («Decode 1…2») и выбор только по нагрузке: overlap_score_credit = 0, assume_kv_reuse = false, track_prefill_tokens = false, поэтому `prefillTerm = 0` и `cost = decodeTerm` |
| `cluster` (S1a) | `{ deployment, tp, gpuCount, workers: FlowWorkerInfo[], warmWorkerId?, warmBlocks }`; `FlowWorkerInfo`: `id, name, role ('aggregated' / 'prefill' / 'decode'), gpus, capacityBlocks, activeRequests, activeBlocks, queuedPrefillTokens, cached[] ({ label, blocks, matchesPrompt })` |

История кэша (одинакова в agg и disagg, зависит только от `cacheWarm` и длины промпта):
- В кэше воркеров лежат настоящие блоки пяти тенантов (поддержка магазина, код-ревью, помощник юриста, поиск по документации,
  переводчик): их промпты прогнаны тем же шаблоном и токенизатором, поэтому с вашим промптом они не совпадают с первого блока.
- Холодный: префикса промпта нет ни у кого, побеждает наименее загруженный Воркер 3 (`worker-2` / `prefill-2`).
- Тёплый: у Воркера 2 (`worker-1` / `prefill-1`) лежат все полные блоки промпта, кроме последнего (при одном полном блоке: этот блок).
  Он нагружен на c/2 + 0,25 блока больше Воркера 3 (c = блоков в тёплом кэше), но скидка за кэш перевешивает: роутер выбирает его.
  Нагрузка одинакова в холодном и тёплом режиме, меняется только кэш.
- Промпт короче 16 токенов: полных блоков нет, тёплому кэшу нечего хранить (заметка в `notes`).

### prefill: `run.prefill`

| Поле | Что |
|---|---|
| `totalTokens`, `cachedTokens`, `computedTokens` | `cachedTokens = effectiveCachedTokens(total, overlap × 16)`: всегда меньше total (последний токен считается) |
| `ms` | `prefillMs(model, gpu, total, { cachedTokens, gpus: tp })` |
| `worker`, `workerName` (S1a) | id и имя воркера |
| `kvBytes` | KV, записанный prefill: `kvBytesForSeq(model, computedTokens)` |
| `kvTotalBytes` (S1a) | KV всего промпта |
| `cachedBlocks` (S1a) | сколько полных блоков из кэша |
| `layers`, `layerMs` (S1a) | для прогресса по слоям: `model.layers` (36 у Qwen3-8B) и `ms / layers` (поровну) |
| `bound`, `computeMs`, `memoryMs`, `flops` (S1a) | разбивка `prefillBreakdown`: короткий промпт упирается в чтение весов (`'memory'`) |

### transfer (только disagg): `run.transfer`

`bytes` (KV всего промпта × дублирование при TP > kv_heads), `ms` = `nixlSetupMs` + `transferMs(bytes / links, GBs)`,
`link` (имя канала), `from` / `to` (id воркеров) и (S1a) `linkId: 'ib-ndr'`, `GBs: 50`, `links` (= TP: каждый ранг своей картой NDR 400G,
как в DGX), `setupMs`, `fromName` / `toName`, `params` (иллюстративные `kv_transfer_params` vLLM NixlConnector: `do_remote_prefill`,
`remote_block_ids`, `remote_engine_id`, `remote_host`, `remote_port`, `tp_size`; значения условные).

### decode: `run.decode[k]` (по шагу на выходной токен, включая `<|im_end|>`)

| Поле | Что |
|---|---|
| `step`, `token` | индекс и токен (настоящий, со смещениями в тексте ответа) |
| `channel` | `'reasoning'` (включая `<think>` и `</think>`), `'content'` (включая `\n\n` после `</think>` и `<|im_end|>`), `'tool_call'` (от `<tool_call>` до `</tool_call>`) |
| `candidates[]` | 5–8 кандидатов, отсортированы по логиту: `token`, `logit` (синтетический, 2 знака), `prob` (после маски и перенормировки: из этого выбирает сэмплер, 0 у замаскированных), `allowed`, (S1a) `rawProb` (до маски, после temperature: что «хотела» модель), (S1a) `preview` (у ручных альтернатив: как модель продолжила бы, например «Конечно, вот JSON:») |
| `chosenIndex` (S1a) | индекс выбранного токена в `candidates` (в json-режиме выбранный может быть не первым: первым стоит замаскированная болтовня) |
| `masked` | грамматика активна на этом шаге (json, после рассуждения) |
| `maskedCount`, `grammarExpecting` (S1a) | сколько показанных кандидатов отрезано; что грамматика ждёт перед шагом («ключ "name"», «значение поля "age": целое число») |
| `startMs`, `durationMs` | время шага. agg, k = 0: интервал prefill (первый токен выбран в конце prefill); остальные: `decodeStepMs(model, gpu, { batch, avgContext, gpus })` |
| `phase` (S1a) | `'prefill'` у первого токена в agg, иначе `'decode'` (в disagg первый токен выбирает decode-воркер, пересчитав последний токен промпта) |
| `entropyBits`, `batch`, `kvTokens`, `kvBytes` (S1a) | энтропия распределения; batch шага (наш запрос + активные запросы воркера); длина и размер KV нашей последовательности после шага |
| `breakdown` (G0, `DecodeStepEx`) | разбивка `perf.decodeStepBreakdown`: `memoryMs`, `computeMs`, `overheadMs`, `weightBytes` (у MoE только задетые эксперты), `kvReadBytes` (KV всех запросов batch), `bound`; у первого токена в agg нет (он выбран в конце prefill) |
| `emitMs`, `clientMs` (S1a) | когда frontend отправил чанк этого токена и когда он дошёл до клиента |
| `stop`, `note` (S1a) | стоп-токен; пояснение к шагу по-русски (первый токен, начало грамматики, `</think>`, `<tool_call>`, стоп) |
| `neighborPrefill` (X-sim-7, только agg) | `{ tokens, doneBefore, promptTokens, budget, plainMs }`: в этот шаг chunked prefill поставил кусок промпта соседа; `durationMs` из `perf.stepTime` (prefill + decode в одном шаге), `plainMs` тот же шаг без него |

### Сосед с длинным промптом (X-sim-7): `run.neighbor`

Чтобы переключатель «Деплой» показывал не только цену дезагрегации, на воркер нашего запроса на шаге `neighborArrivalStep(N)` (треть ответа,
не позже 12-го токена) приходит условный промпт соседа на `AGG_NEIGHBOR_PREFILL_TOKENS` = 6 000 токенов. Только в ответах от
`NEIGHBOR_MIN_COMPLETION_TOKENS` = 40 токенов (X-sim-r2-4): в коротком ответе один долгий шаг соседа задавал бы почти всё среднее ITL и E2E,
и обычный шаг decode выглядел бы в разы медленнее. Интерфейс рядом со средним ITL показывает обычный шаг (`neighbor.plainMs`). Бюджет шага
`servingTokenBudget(gpuId)` как у `vllm serve` по умолчанию (v0.32): B200 16 384, H100 и H200 8 192, остальные 2 048.
agg: chunked prefill ставит куски промпта в шаги decode (`decode[k].neighborPrefill`), шаги упираются в вычисления и длятся в десятки раз дольше
(Qwen3-8B на H100: около 190 мс вместо 7); заметка «Под нагрузкой…». disagg: промпт считает prefill-воркер, шаги decode ровные,
`run.neighbor = { applied: false, stepMs }` говорит, сколько длился бы такой шаг в agg; заметка «Выигрыш дезагрегации…».

### postprocess: `detok`, `parser`, `output`

| Поле | Что |
|---|---|
| `detok[k]` | `incrementalDetokenize(outTokens, tokenizer, { skipSpecial: true, promptTokens, stopTokenIds: [151645, 151643] })`: `DetokStepEx` с `emitted`, `held`, `heldBytes`, `output`, `prefixOffset`, `readOffset`, `stopped` (на `<|im_end|>`) |
| `parser[k]` | `runStreamingParser(detok.map(d => d.emitted), { reasoning: 'qwen3', tools: tools ? 'hermes' : 'none' })`: `ParserEventEx` с `stateBefore` / `stateAfter` (`maybe_tag` при придержанном «<thi»), `buffer`, `deltas`, `out`, `toolDeltas`, `toolCall`, `note`. Событие k соответствует токену k; может быть одно финальное событие сброса (`step = N`) |
| `output` (S1a) | итог глазами клиента: `text` (всё отданное детокенизатором), `reasoning`, `content` (как склеит клиент; у thinking начинается с `\n\n`), `toolCalls`, `json` (разобранный JSON, только json) |

### stream: `sse`, `sseStep`, `sseMs`

`sse` = `toSseChunks(parser, { reasoningField: 'reasoning_content', includeUsage: true, usage: { promptTokens, completionTokens, cachedTokens } })`:
чанк с ролью, дельты `reasoning_content` / `content` / `tool_calls`, финальный чанк с `finish_reason`, чанк `usage` (с
`prompt_tokens_details.cached_tokens`), `data: [DONE]`. (S1a) `sseStep[i]`: после какого токена отправлен чанк i (роль приходит
с токеном 0, последние три чанка получают `N` = «после последнего токена»); `sseMs[i]`: когда чанк дошёл до клиента.

### timeline, stages, playback, metrics

| Поле | Что |
|---|---|
| `timeline` | крупные отрезки без `step`: `client`, `frontend`, `preprocess`, `route`, `prefill`, (disagg) второй `route` (выбор decode-воркера) и `transfer`; затем по токену: `decode` (кроме токена 0 в agg), `postprocess`, `stream` с `step`. (S1a) `detail` (пояснение), `simulated: true` у условных констант |
| `stages` (S1a) | по записи на стадию: `startMs`, `endMs`, `ms` (сумма отрезков), `count` (сколько отрезков), `label`, `simulated` |
| `playback` (S1a) | шаги плеера: `kind: 'stage'` (каждый крупный отрезок), `'token'` (по токену, `token: k`), `'done'`; `tMs` (позиция плейхеда), `sseCount` (сколько чанков клиент уже получил) |
| `metrics` | `ttftMs` (клиент получил первый чанк после токена 0), `itlMs` = (E2E − TTFT)/(N − 1), `e2eMs` (финальный чанк у клиента), `tokensPerSec` = 1000 / ITL (perceived TPS книги), `promptTokens`, `completionTokens` (с `<|im_end|>`, как vLLM), `cachedTokens`, `finishReason`; (S1a) `ttftVisibleMs` (первый чанк с непустой дельтой: при thinking токен `<think>` сам ничего не показывает), `perUserTps` (N / E2E, определение NVIDIA), `prefillMs`, `decodeMs`; (G0) `firstAnswerMs`: первый чанк с ответом (непустой content или tool_calls), а не с рассуждением |

Инварианты (проверены тестами): `ttftMs` = сумма крупных отрезков (+ шаг 0 в disagg) + `postprocessMs` + `networkMs`;
`e2eMs = ttftMs + itlMs × (N − 1)`; шаги decode идут встык; выбранный токен разрешён маской и самый вероятный среди разрешённых;
склейка `detok.emitted` равна тексту ответа без скрытых спецтокенов; `reasoning` / `content` в SSE равны дорожкам парсера.

---

## Что настоящее, а что симулировано

| Настоящее (модули движка) | Симулировано |
|---|---|
| шаблон Qwen3 (`renderTemplate`), токены (`tokenizer.encode`), сегменты | текст ответа модели: пресеты и заготовки по темам (`scenarios.ts`) |
| хэши блоков (`blockHashes`), формула роутера Dynamo (`routeKV`, `scoreWorker`, `selectionProbs`) | состояние кластера: тенанты, нагрузка, тёплый воркер (`flow-cluster.ts`) |
| prefill, шаги decode, KV, передача (`perf.ts`, `data/*`), TP (`fitsCheck`) | логиты и альтернативы кандидатов (`flow-candidates.ts`): настоящие токены словаря, синтетические числа |
| вероятности (`sampling.distribution` при `settings.temperature`) | выбор: всегда заскриптованный токен (самый вероятный среди разрешённых) |
| маски JSON (`allowedMaskAfterTokens`, `whitespace: 'xgrammar'`), состояние грамматики | константы времени `FLOW_TIMING` (сеть, HTTP, CPU-стадии, NIXL setup) |
| детокенизация, парсеры, SSE (`detok.ts`, `parsers.ts`) | `kv_transfer_params`, id ответа и вызовов, `created` |

`FLOW_TIMING` (мс, условно; на таймлайне `simulated: true`):

| Константа | Значение | Что |
|---|---|---|
| `networkMs` | 5 | сеть клиент ↔ frontend в одну сторону (клиент в том же регионе) |
| `frontendMs` | 0,2 | разбор HTTP и JSON, проверка полей |
| `templateBaseMs` + `templatePerCharMs` | 0,05 + 0,0002 на символ | рендер шаблона |
| `tokenizeBaseMs` + `tokenizePerTokenMs` | 0,03 + 0,002 на токен | токенизация (Rust HF tokenizers порядка 1 мкс на токен, с запасом) |
| `routeBaseMs` + `routePerBlockMs` + `dispatchMs` | 0,05 + 0,002 на блок + 0,15 | хэши и поиск в индексе, отправка на воркер по request plane |
| `prefillReturnMs` + `decodeRouteMs` | 0,15 + 0,05 | disagg: ответ prefill-воркера с метаданными и выбор decode-воркера |
| `nixlSetupMs` | 0,1 | disagg: регистрация передачи и уведомление о завершении |
| `postprocessMs` | 0,08 на токен | доставка токена с воркера во frontend, детокенизация, парсеры, сборка чанка |

## Как устроены сценарии

`scriptFor(settings)` берёт `resolveScenario(settings)` и переключатели. Любое сочетание даёт осмысленный ответ:
- `json`: тело это JSON по схеме сценария (`simple` → `{"answer": 4}`, `thinking` → `{term, what, why}`, `tool` → `{city, answer}`,
  `json` → `{name, age, city}`, свой текст → `{topic, answer}`); вызов инструмента невозможен (маска не пустит `<tool_call>`, заметка);
- `tools` и вопрос, которому нужен инструмент (погода с городом): вызов `get_weather` в формате hermes, `finish_reason: tool_calls`;
- иначе текст (при tools модель видит описание инструмента, но не вызывает его, заметка); погода без tools: «нет доступа к данным»;
- `thinking`: перед телом рассуждение; flow.ts пишет `<think>\n` + рассуждение + `\n</think>` + `\n\n` (без `\n\n` перед JSON:
  грамматика включается сразу после `</think>`).

Скрипт нарезан на куски `ScriptPiece`; кусок может нести `alts: AltSpec[]` (`text`, `preview`, `delta`: логит относительно
выбранного) и `confidence`. Ручные альтернативы стоят в ключевых местах: первый токен ответа («Представьте…», «Это…»), `4` в
«2 + 2 = 4», `<tool_call>` против «Сейчас посмотрю…» и выдуманной погоды, «Berlin», первый шаг JSON («Конечно, вот JSON:»,
«```json», «Вот…» под маской), возраст строкой `"` под маской, «Казани» вместо «Казань». У служебных токенов (`<think>`, `</think>`,
`<|im_end|>`, `</tool_call>`) общие альтернативы (`specialAlts`). Остальное подбирает `buildCandidates` по классу выбранного
токена: начало слова (частые слова языка ответа и слова из ответа и промпта), окончание (суффиксы), цифры, пунктуация, разметка JSON,
ключ JSON (начала самого ключа, которые маска пропускает, и чужие ключи, которые она режет), идентификаторы.

Свой текст: 20 тем по ключевым словам (список и порядок в разделе «Заметки прогона») плюс приветствие, погода (город по словарю основ:
«в Казани» → «Казань»; город не из словаря берётся как в тексте, с заметкой), арифметика (пример с одним действием считается
по-настоящему), пустое сообщение и общий ответ, который повторяет тему вопроса («что такое X» → X) и прямо говорит, что это симуляция.
Эхо текста пользователя чистится от литералов спецтокенов. Вкладка «Длинный документ» даёт свой ответ по регламенту (`isLongDoc`).

## Как вести анимацию (S1b)

1. Плеер: `usePlayer(run.playback.length)`. Шаг плеера → `run.playback[i]`: `stage` для подсветки узла (`pipelineStages`
   даёт порядок узлов), `label` для подписи, `tMs` для плейхеда на таймлайне, `sseCount` для ленты SSE (`run.sse.slice(0, sseCount)`).
2. Шаг `kind: 'token'` с `token = k`: инспектор decode показывает `run.decode[k]` (кандидаты, маска, KV, ITL = `durationMs`),
   postprocess показывает `run.detok[k]` и `run.parser[k]` (а на последнем токене и финальное событие сброса, если есть),
   накопленные дорожки берите из `run.parser.slice(0, k + 1)` (`deltas`) или из SSE до `sseCount`.
3. Пакет по рёбрам: на шаге-стадии он идёт к узлу `stage`; на шаге-токене от decode через postprocess к клиенту.
   В disagg перед первым токеном есть два шага-стадии `route` (prefill-роутер и выбор decode-воркера) и шаг `transfer`.
4. Время: реальные интервалы от 0,08 мс до секунд, поэтому анимацию не масштабируйте пропорционально: дайте шагам-стадиям
   фиксированную длительность (например 500–700 мс), токенам 40–80 мс, а реальные мс подписывайте (`timeline`, `stages`,
   `decode[k].durationMs`). Для таймлайна внизу удобна логарифмическая или разрывная ось: крупные отрезки (`step === undefined`)
   плюс ITL-тики `decode[k].clientMs`, маркер TTFT `metrics.ttftMs`, плейхед `playback[i].tMs`.
5. При reduced motion плеер без автоплея, переходы мгновенные (usePlayer это делает).
6. Ссылка «Подробнее»: `STAGE_INFO[stage].chapters[0]` через `ChapterLink`.

## Рецепты по стадиям инспектора

| Стадия | Что показать |
|---|---|
| client | `run.request` (`CodeBlock lang="json"`) |
| frontend | `run.frontend.checks` (иконка + слово), `sampling`, `stop`, `guidedJson`, `parsers`; `STAGE_INFO.frontend.what` |
| preprocess | `template.segments` с раскраской, `TokenStrip tokens={run.promptTokens}` + `preprocess.segmentOfToken`, `metrics.promptTokens`, `preprocess.toolsTokens`, `preprocess.preprocessedRequest` |
| route | `blocks` (хэши, `full`), `cachedBlockFlags(run)`, таблица `routing.scores` (overlap, prefill blocks, decode blocks, cost, prob), `routing.chosen`, `routing.explanation`, `cluster.workers[].cached`; в disagg ещё `decodeRouting` |
| prefill | слои `prefill.layers` × `prefill.layerMs`, `cachedTokens` / `computedTokens`, `kvBytes`, `ms`, `bound` |
| transfer | `transfer.bytes`, `link`, `links`, `GBs`, `ms`, `params` |
| decode | `decode[k].candidates` (prob, rawProb, allowed, preview), `chosenIndex`, `grammarExpecting`, `kvTokens` / `kvBytes`, `durationMs`, `batch`, `note` |
| postprocess | `detok[k]` (`emitted`, `held`, `heldBytes`), `parser[k]` (`stateBefore` → `stateAfter`, `buffer`, `note`), дорожки `output.reasoning` / `output.content` / `output.toolCalls` |
| stream | `run.sse.slice(0, sseCount)` (`line`), в конце `finish_reason`, usage, `[DONE]` |
| метрики | `metrics` (`ttftMs`, `itlMs`, `e2eMs`, `tokensPerSec`, токены, `finishReason`), `simulatedNote` |

## Упрощения

- Выбор токена не случайный: всегда заскриптованный токен, логиты подобраны так, что он самый вероятный среди разрешённых.
  Вероятности считаются только по показанным кандидатам, поэтому они выше, чем дал бы softmax по всему словарю (151 936 логитов, по одному
  на строку выходного слоя); top-p и top-k в симуляторе не применяются, учитывается только temperature. temperature меняет вероятности, но не ответ.
- Маска JSON в режиме `whitespace: 'xgrammar'` (ровно один пробел после `:` и `,`, как XGrammar с any_whitespace=False);
  в vLLM по умолчанию пробелы гибкие (сотни разрешённых токенов-пробелов в структурных местах).
- Нагрузка соседей по batch условна: их контекст = активные блоки воркера × 16 и растёт на токен за шаг; prefill считается
  отдельным шагом без смешивания с чужим decode; очередь prefill воркера на наш TTFT не влияет (её токены попали бы в тот же шаг).
  Из чужих prefill в шаги decode попадает только условный сосед на 6 000 токенов (agg, ответы от 40 токенов, см. «Сосед с длинным промптом»);
  после своего prefill он в batch не остаётся.
- disagg: передаётся KV всего промпта (у decode-воркера нет своего кэша этого промпта); первый токен выбирает decode-воркер
  после пересчёта последнего токена промпта (vLLM; в SGLang первый токен сэмплирует prefill); токен prefill-воркера отброшен.
- Dynamo по умолчанию работает в режиме round-robin, KV-aware режим включают явно (`--router-mode kv`); симулятор показывает
  KV-aware режим.
- Ответ модели одинаков для всех моделей и GPU (меняются только время, размер KV и TP).
- Ёмкость KV воркера (`cluster.workers[].capacityBlocks`) считается с gpu_memory_utilization 0,92 (vLLM с v0.20, SPEC §8.8).

## Заметки прогона (G0)

`notes` по порядку: замена модели или GPU, инъекция спецтокенов, заметки сценария, заметки кластера, в disagg «клон с max_tokens = 1»,
«цена дезагрегации» и (у MoE) почему ITL выше при большом batch, обрезка по max_tokens, затем откуда TP (если у модели на разных GPU
симулятора разное число GPU на воркер, например Qwen3-32B: 2 × H100 или 1 × H200). Если веса BF16 больше памяти одного GPU (Qwen3-235B-A22B,
около 470 ГБ), заметка называет причиной веса, а не запас под KV, и говорит, чего не хватает на TP вдвое меньше (G0-r2-10).
Запас под KV в заметке: «8 запросов по 8 192 токена».

Темы своего текста (порядок проверки): детокенизация, PagedAttention, уровни KV cache (L3, DaemonSet, LMCache, Mooncake), prefix caching (и RadixAttention), KV cache, дезагрегация,
compute- и memory-bound, speculative decoding, prefill и decode, метрики (и «первый токен»), сэмплинг, attention (и QKV, «Q, K и V»),
шаблон чата, парсеры, structured output, токенизация (без голого «токен»), батчинг, квантизация, роутинг, Dynamo. Приветствие в начале
сообщения не перекрывает вопрос после него: вопрос без заготовки после приветствия («Привет! Что такое vLLM?») получает общий ответ
с темой вопроса (X-sim-r2-8). Заготовка приветствия остаётся для одного приветствия и светской беседы («Привет, как дела?», «меня зовут…»). Погода только по «погод» / «weather». Арифметика только с одним действием; пример из нескольких
действий получает ответ «Заготовка считает только одно действие». Вопрос латиницей получает пометку, что заготовки русские.
