# Движок: текст и токены (A1 «engine-text»)

Чистый TypeScript без React и DOM (кроме хука `useTokenizer`). Всё детерминировано: случайность только
через `rng`, который передаёт вызывающий. Контракты — в `src/engine/types.ts`; здесь у некоторых функций
есть расширенные типы (`…Ex`) с дополнительными полями, они совместимы с контрактами.

Модули:

| Файл | Что внутри |
|---|---|
| `tokenizer/bpe.ts` | `ByteLevelBpeTokenizer` — byte-level BPE Qwen3 (интерфейс `Tokenizer`) |
| `tokenizer/loader.ts` | `loadTokenizer()` и состояние загрузки (без React) |
| `tokenizer/useTokenizer.ts` | React-хук `useTokenizer()` |
| `tokenizer/o200k.ts` | сравнительный токенизатор o200k (gpt-oss / GPT-4o) через `gpt-tokenizer` |
| `tokenizer/bytes.ts` | byte-level алфавит (Ġ, Ċ), UTF-8 утилиты |
| `tokenizer/asset.ts` | формат `public/tokenizer/qwen3.json` |
| `tokenizer/index.ts` | реэкспорт всего чистого (без хука) + константы Qwen3 |
| `templates.ts` | чат-шаблоны qwen3 / llama3 / deepseek-r1 / gpt-oss, `FAMILIES` |
| `sampling.ts` | temperature, top-k, top-p, min-p, маска, выбор токена, энтропия |
| `detok.ts` | инкрементальная детокенизация (prefix/read offset), stop-строки, skip_special_tokens |
| `parsers.ts` | стриминговый парсер reasoning / tool calls, SSE-чанки, Harmony по каналам |
| `structured.ts` | JSON-схема → автомат префиксов, маски, jump-forward |

Импорты: `@/engine/tokenizer` (чистое), `@/engine/tokenizer/useTokenizer` (хук), `@/engine/templates` и т. д.

---

## Токенизатор Qwen3

### Словарь `public/tokenizer/qwen3.json`

Собирается `npm run build:tokenizer` (`scripts/build-tokenizer.mjs`) из `research/vendor/qwen3/tokenizer.json`
и `tokenizer_config.json`. Формат `bpe-merges-v1`:

- `base` — 256 символов byte-level алфавита в порядке id 0..255;
- `merges` — плоский массив пар id в порядке ранга. У Qwen3 склейка ранга r всегда порождает токен
  `256 + r` (скрипт это проверяет), поэтому строки словаря не хранятся: `raw[256 + r] = raw[l] + raw[r]`;
- `added` — 26 added tokens с флагом `special` из HF;
- `pattern` — регулярка пре-токенизации, переписанная под JS (флаг `u`): `(?i:'s|…)` развёрнута в явные
  варианты регистра (включая `ſ`, которую Oniguruma сворачивает в `s`; других складок нет — проверено перебором),
  а `\p{L}`, `\p{N}` и `\s` заменены явными диапазонами code point из снимка классов Oniguruma
  (`scripts/data/onig-unicode-classes.json`, генерирует `scripts/gen-onig-classes.py` через HF `tokenizers`).
  Зачем: `\p{L}` в JS берётся из ICU браузера (Node 24 — Unicode 17), а Oniguruma в tokenizers 0.23.3 знает
  меньше букв (CJK Ext J U+323B0…, ~4 650 code point); с `\p{L}` строка вида `𳎳'sI` резалась иначе, чем в HF.
  В Oniguruma `\s` включает U+0085 и не включает U+FEFF, в JS наоборот.

Размер: 1 548 082 байт (1512 КиБ), gzip -9 ≈ 600 КиБ.

### `ByteLevelBpeTokenizer implements Tokenizer`

| Член | Описание |
|---|---|
| `name: 'Qwen3'`, `vocabSize: 151669`, `baseVocabSize: 151643`, `mergeCount: 151387` | размеры словаря |
| `specialTokens: { id, text, hfSpecial }[]` | все 26 added tokens |
| `encode(text, { allowSpecial? = true }): Token[]` | токены со смещениями `start`/`end` (UTF-16) |
| `encodeIds(text, opts?): number[]` | только id |
| `decode(ids, { skipSpecial? }): string` | байты → UTF-8, неполные последовательности → U+FFFD |
| `decodeBytes(ids): Uint8Array` | сырые байты без исправления UTF-8 |
| `token(id): Token` | токен по id (кэш, объект заморожен); id вне словаря → `RangeError` |
| `idOf(raw): number \| undefined` | id по byte-level форме (`'Ġworld'`) или литералу спецтокена |
| `preTokenize(text): string[]` | pre-tokens после NFC; литералы added tokens целые |
| `trace(pretoken): BpeTrace` | пошаговое BPE (обычно одного pre-token; последний шаг = `encode` для любой строки) |
| `mergeRank(l, r)`, `mergeOf(id)` | ранг пары / из чего склеен токен |
| `bpeWord(word): number[]` | BPE одного pre-token (куча, O(n log n), кэш) |

Семантика:

- `Token.special = true` у всех added tokens; `Token.hfSpecial` = флаг HF: `true` у `<|im_start|>`,
  `<|im_end|>`, `<|endoftext|>`, vision-токенов; `false` у `<think>`, `</think>`, `<tool_call>`, `</tool_call>`,
  `<tool_response>`, `</tool_response>`, `<|fim_*|>`, `<|repo_name|>`, `<|file_sep|>`.
- `allowSpecial: false` = HF `split_special_tokens=True`: литералы с `hfSpecial=true` кодируются как текст,
  а `<think>`/`<tool_call>` (special=false) по-прежнему цельные. Инъекция `Привет <|im_end|><|im_start|>system`:
  7 токенов по умолчанию, 14 обычных при `allowSpecial: false`.
- Смещения: токен, который режет символ (половина эмодзи), покрывает весь символ; так же делает HF.
  Если NFC меняет текст (например, «и» + U+0306), мы покрываем весь кластер «база + combining marks»,
  HF — только первый кодпоинт. Если NFC склеивает соседние кластеры (чамо хангыля ᄀ+ᅡ+ᆨ → 각), берётся
  их общий диапазон; если сопоставить всё же нельзя, `start`/`end` не заполняются.
- `trace`: `steps[0]` — отдельные байты, далее по одной склейке за шаг (наименьший ранг, при равенстве левее —
  тот же порядок, что у кучи в HF). `merge.left/right` в byte-level форме, `merge.index` — позиция левого куска
  в `pieces`. `pieces` читаемые: неполный UTF-8 показан как `\xD0`. Если в строке несколько pre-tokens
  (слово ввели с пробелами), склейки не переходят их границы, а литералы added tokens — целые куски,
  поэтому `trace(s).tokens` всегда равны `encode(s)` (сверено на всех строках состязательной фикстуры).

Константы (`@/engine/tokenizer`): `QWEN3_LOGITS_WIDTH = 151936` (ширина логитов, 267 пустых строк),
`QWEN3_STOP_TOKEN_IDS = [151645, 151643]`.

### Загрузка

```ts
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
const { tokenizer, status, error } = useTokenizer() // status: 'idle' | 'loading' | 'ready' | 'error'
```

| Функция | Описание |
|---|---|
| `useTokenizer(): { tokenizer, status, error? }` | хук; при первом вызове качает `${BASE_URL}tokenizer/qwen3.json`, состояние общее на страницу; до первого эффекта `status = 'idle'` — рисуйте скелетон и для него |
| `loadTokenizer(): Promise<Tokenizer>` | то же без React; повторные вызовы — тот же Promise, после ошибки — новая попытка |
| `loadQwen3Tokenizer(): Promise<ByteLevelBpeTokenizer>` | с конкретным типом (`trace`, `mergeOf`, `encodeIds`) |
| `getTokenizerState()`, `subscribeTokenizer(fn)`, `getLoadedTokenizer()` | для `useSyncExternalStore` и не-React кода |
| `tokenizerUrl()` | URL словаря с учётом `import.meta.env.BASE_URL` |
| `createBpeTokenizer(asset)`, `parseBpeAsset(json)` | собрать токенизатор из уже загруженного JSON |

### Сравнительный o200k (`tokenizer/o200k.ts`)

| Функция | Описание |
|---|---|
| `loadO200kTokenizer(variant = 'o200k_harmony'): Promise<O200kTokenizer>` | динамический импорт `gpt-tokenizer/encoding/o200k_harmony` (или `o200k_base`) + `gpt-tokenizer/bpeRanks/o200k_base` + регулярки; отдельный чанк ~2 МБ (≈1 МБ gzip) |
| `O200kTokenizer` | тот же интерфейс `Tokenizer`: `encode` (с `bytes`, `text` с U+FFFD у неполных токенов, `raw` в byte-level форме, смещения), `decode`, `decodeBytes`, `token`, `idOf`, `preTokenize`, `trace` (BPE в стиле tiktoken: ранг = id) |
| `O200K_HARMONY_LOGITS_WIDTH = 201088` | ширина логитов gpt-oss |

`vocabSize = 200019`. Спецтокены Harmony: `<|start|>` 200006, `<|end|>` 200007, `<|message|>` 200008,
`<|channel|>` 200005, `<|return|>` 200002, `<|call|>` 200012 и т. д. У `o200k_base` номера спецтокенов
как в `gpt-tokenizer` (у tiktoken их меньше). `allowSpecial: false` кодирует литералы как текст.

### Байтовые утилиты (`tokenizer/bytes.ts`)

`BYTE_TO_CHAR`, `byteOfChar(ch)`, `bytesToRaw(bytes)`, `rawToBytes(raw)`, `utf8Bytes(text)`,
`utf8DecodeLossy(bytes)` (U+FFFD на неполных, BOM сохраняется), `utf8DecodeSoft(bytes)` (неполные как `\xNN`),
`isCompleteUtf8(bytes)`, `byteToUtf16Spans(s)`.

---

## Чат-шаблоны (`templates.ts`)

| Экспорт | Описание |
|---|---|
| `renderTemplate(input: TemplateInputEx): RenderedTemplateEx` | `{ family, text, segments, warnings }`, `text === segments.map(s => s.text).join('')` |
| `FAMILIES: FamilyInfoEx[]` | qwen3, llama3, deepseek-r1, gpt-oss: имена, пример модели, парсеры, заметка, стоп-токены, generation prompt, флаги vLLM/Dynamo |
| `familyInfo(family)` | одно семейство |
| `FAMILY_SPECIAL_TOKENS` | литералы спецтокенов каждого семейства |
| `pyJson(value, indent?)` | JSON как `tojson` в transformers (Python `json.dumps(ensure_ascii=False)`: `', '`, `': '`) |

`TemplateInputEx` = `TemplateInput` + `llamaDateString?` (по умолчанию `"26 Jul 2024"` — дефолт шаблона Llama 3.1),
`harmonyDate?` (по умолчанию `"2026-11-12"` — день замеров), `reasoningEffort?` (gpt-oss; `enableThinking: false` → `low`).

Маппинг полей: `ChatMessage.reasoning` → `reasoning_content` (Qwen3) / `thinking` (gpt-oss), остальные шаблоны его
не рендерят. `tool_calls` с пустым массивом считаются отсутствующими. Аргументы вызовов сериализуются `pyJson`
(DeepSeek склеивает их строкой).

Сегменты: `special` (литерал спецтокена, в том числе внутри текста — так видна инъекция), `role` (имя роли;
у gpt-oss также канал и получатель), `content` (текст сообщения, с `messageIndex`), `tools` (описание инструментов),
`gen-prompt` (весь хвост add_generation_prompt, включая его спецтокены), `format` (переводы строк и обвязка шаблона,
в том числе строки про даты Llama и system-блок gpt-oss).

Каждый рендерер повторяет официальный Jinja построчно, включая странности: DeepSeek-R1 не сбрасывает `is_first`
между сообщениями и `is_output_first` между блоками ответов инструментов; gpt-oss печатает перед вложенным типом
объекта перевод строки и 16 пробелов; Llama 3.1 пропускает строковый ответ инструмента через `tojson` (кавычки).
Там, где шаблон бросил бы `raise_exception`, рендер продолжает и пишет предупреждение в `warnings`.

---

## Сэмплинг (`sampling.ts`)

Порядок как в vLLM V1: маска → temperature → softmax → min-p → top-k → top-p (по распределению после top-k) → выбор.

| Функция | Описание |
|---|---|
| `sample(cands, params, rng, allowed?): SamplingResultEx` | выбор: обратная функция распределения по `finalProbs` с `u = rng()`; T = 0 → argmax среди разрешённых; нет разрешённых → `chosen = -1` |
| `distribution(cands, params, allowed?): SamplingResultEx` | то же без случайности (`chosen` = самый вероятный) |
| `softmax(logits, T = 1)` | `−Infinity` → 0 |
| `topKMask(probs, k, keep?)`, `topPMask(probs, p, keep?)`, `minPMask(probs, minP, keep?)` | маски отсечений (k ≤ 0, p ≥ 1, minP ≤ 0 — выключено) |
| `entropy(probs, base = 2)` | в битах (или натах) |
| `argmax(logits, allowed?)` | ничьи → меньший индекс |

`SamplingParamsEx` = `SamplingParams` + `minP?`. `params.seed` игнорируется: ГПСЧ создаёт вызывающий.
`SamplingResultEx`: `probs` (после temperature, до маски и отсечений — видно, что «хотела» модель), `maskedProbs`,
`allowed`, `kept`, `finalProbs`, `chosen`, `dropReason[i]` (`'mask' | 'min-p' | 'top-k' | 'top-p' | 'greedy' | null`),
`entropyBits`, `u`. При T = 0 `probs` вырождено в one-hot у argmax всех кандидатов.

---

## Детокенизация (`detok.ts`)

| Функция | Описание |
|---|---|
| `incrementalDetokenize(tokens, tokenizer, opts?): DetokStepEx[]` | по шагу на токен; после stop массив обрывается |
| `naiveDetokenize(tokens, tokenizer, { skipSpecial? }): string` | «каждый токен отдельно» — с «�» |
| `stopPrefixHoldback(text, stops)` | длина хвоста, похожего на начало stop-строки (jail) |
| `INITIAL_DETOK_OFFSET = 5` | стартовый сдвиг окна vLLM |

`DetokOptions`: `stop?`, `skipSpecial?` (**по умолчанию true**, как в vLLM: `<|im_end|>` не даёт текста, `<think>`
остаётся), `stopMode?: 'jail' | 'vllm'` (по умолчанию `'jail'`, как Dynamo: держим только возможное начало
stop-строки; `'vllm'` держит фиксированный хвост `max(len(stop)) − 1` **code point**, как `len()` в Python, и не режет
суррогатные пары), `includeStopStr?` (тогда ничего не придерживается: в vLLM `stop_buffer_length = 0`, в Dynamo такие
stop-строки «visible»; текст стоп-токена тоже выводится, если его не выкинул `skipSpecial`), `promptTokens?`
(последние 7 «заряжают» окно), `stopTokenIds?` (EOS заканчивает поток без текста), `finished?` (по умолчанию true:
на последнем токене придержанный под stop-строку хвост выдаётся; неполные байты в конце теряются, как в vLLM).
Пошаговый вывод совпадает с HF `tokenizers.decoders.DecodeStream` (`detok.adversarial.test.ts`).

`DetokStepEx`: `emitted` (ушло клиенту), `held` (придержанное: хвост под stop-строку + неполный символ как «�»),
`heldBytes`, `stopHeld`, `output` (всё отданное к концу шага), `naiveText`, `prefixOffset`/`readOffset`
(в индексах сгенерированных токенов; отрицательные указывают в окно промпта), `stopped`, `stopReason`.

---

## Парсеры и SSE (`parsers.ts`)

| Функция | Описание |
|---|---|
| `runStreamingParser(chunks, opts): ParserEventEx[]` | событие на каждый кусок (+ финальный сброс буфера) |
| `toSseChunks(events, meta): SseChunk[]` | строки `data: {...}` chat.completion.chunk и `data: [DONE]` |
| `toSseText(chunks)` | как это выглядит на проводе (пустая строка после каждого события) |
| `harmonyToChannels(text): HarmonyMessage[]` | упрощённый разбор вывода gpt-oss по каналам (analysis → reasoning, final → content, `to=` → tool_call) |
| `partialTagOverlap(text, tags)`, `streamableArguments(body)` | помощники (частичный тег; часть аргументов, которую можно отправить) |

`ParserOptionsEx` = `ParserOptions` + `toolCallIdPrefix?` (`'chatcmpl-tool-'`), `idSeed?`, `final?` (true).
Для `reasoning: 'deepseek-r1'` по умолчанию `startInReasoning = true` (forced reasoning).

Поведение: сначала reasoning, потом tool calls по content (внутри рассуждения вызовы не ищутся). Рассуждение
открывается один раз (после `</think>` новый `<think>` — обычный текст). При forced reasoning первый `<think>`,
который модель всё же напечатала, пропускается, где бы он ни стоял (как `stripped_think_start` в SGLang). Частичный тег (`<thi`, `<tool_`)
придерживается: `stateAfter = 'maybe_tag'`, `buffer`, реальная дорожка в `channelAfter`. Вызов стримится как в vLLM
hermes: один фрагмент `{index, id, type, name}`, когда имя закрыто кавычкой, затем приращения строки после
`"arguments":` без закрывающей `}` внешнего объекта (токен `"}}\n` даёт фрагмент `"}`). На `</tool_call>` —
`toolCall` (и `toolCalls`, если в куске закрылось несколько). Битый JSON без отправленных фрагментов возвращается
в content; строка-JSON в `arguments` разворачивается в объект. После каждого вызова ведущие пробелы до следующего
текста не отправляются, содержательный текст идёт в content. Результат (склейка дорожек, аргументы, вызовы)
не зависит от разбиения потока на куски — проверено на каждом офсете (`parsers.adversarial.test.ts`).
`ParserEventEx.out` — всё, что ушло клиенту, по порядку.

`SseMeta`: `id`, `created`, `model`, `finishReason?` (`'stop' | 'length'`, вызовы переписывают в `'tool_calls'`),
`includeUsage?` + `usage?` (`{ promptTokens, completionTokens, cachedTokens? }` → чанк с `choices: []`),
`reasoningField?` (`'reasoning_content'` по умолчанию, как Dynamo/SGLang; `'reasoning'` — vLLM ≥ 0.16),
`stopReason?` (расширение vLLM, добавляется только если задано). Первый чанк: `delta: {role: "assistant", content: ""}`.

---

## Structured output (`structured.ts`)

| Функция | Описание |
|---|---|
| `grammarState(schema, text, opts?): GrammarStateEx` | `{ validPrefix, complete, expecting, failedAt? }`, `expecting` по-русски: «ключ "name"», «целое число», «запятая или }» |
| `allowedMask(schema, textSoFar, candidates, opts?): boolean[]` | кандидат разрешён, если `textSoFar + candidate` — валидный префикс |
| `allowedMaskTokens(schema, textSoFar, tokens, opts?)` | то же по `Token`, **побайтно как XGrammar**: битый UTF-8 запрещён везде, недописанный символ — только там, где его можно дописать; стоп-токены (`eosIds`, по умолчанию `MASK_EOS_IDS = [151645]`) — только у полного документа; остальные added tokens запрещены (`addedTokensAsText: true` — считать литералом, как XGrammar 0.2.9); `pendingBytes` — недописанный символ в конце префикса |
| `allowedMaskAfterTokens(schema, prefix: Token[], tokens, opts?)` | то же, но префикс — токены (сам находит недописанный UTF-8 хвост, например половину эмодзи внутри строки) |
| `forcedContinuation(schema, textSoFar, opts?): string` | jump-forward: строка, которая следует однозначно |
| `grammarCursor(schema, textSoFar, opts?)`, `cursorAccepts(cursor, text)` | разобрать префикс один раз и проверять много кандидатов |
| `QWEN3_EOS_TEXTS` | `['<|im_end|>', '<|endoftext|>']` (стоп-токены генерации из generation_config) |
| `MASK_EOS_IDS` | `[151645]`: XGrammar берёт stop_token_ids = [eos_token_id], поэтому в конце JSON разрешён только `<|im_end|>` (notes-tokens §3.12, шаг 14) |

Подмножество схемы: object (свойства строго в порядке объявления, required обязательны, необязательные можно
пропустить, лишних нет), string (+ enum), integer (`0 | -?[1-9][0-9]*`, как в XGrammar), number, boolean, enum,
вложенные объекты, массивы. Проверка идёт по символам токена, поэтому токены через границы структуры
(`'{"'`, `'":'`, `'",'`, `'"},'`, `'"}}'`) работают.

Пробелы (`opts.whitespace`):
- `'compact'` (**по умолчанию**): никаких пробелов, кроме необязательного одного после `:` и `,`;
  `forcedContinuation` в начале схемы `{name, age}` даёт `{"name":` (дальше пробел необязателен);
- `'xgrammar'`: ровно один пробел после `:` и `,` — как XGrammar с `any_whitespace=False`; даёт `{"name": "`
  и `, "age": ` (как `find_jump_forward_string` в заметках) и маски шагов из notes-tokens §3.12;
- `'minified'`: пробелов нет совсем.
По умолчанию XGrammar разрешает любые пробелы (`any_whitespace=True`), отсюда сотни разрешённых
токенов-пробелов в структурных местах; этот режим не моделируется.

`allowedMask` (по строкам) по умолчанию считает EOS текст `<|im_end|>` (разрешён у полного документа)
и запрещает остальные added tokens Qwen3 (`eosTexts`, `forbiddenTexts` переопределяются). Строка «�» — обычный
символ U+FFFD; для точной проверки токенов с неполным UTF-8 берите `allowedMaskTokens` / `allowedMaskAfterTokens`.

Маски `whitespace: 'xgrammar'` сверены с XGrammar 0.2.9 (`compile_json_schema(any_whitespace=False)`) на всём
словаре Qwen3 на каждом шаге шести целевых JSON (`structured.xgrammar.test.ts`, эталон —
`tokenizer/fixtures/gen_xgrammar_masks.py`): число разрешённых токенов и хэш их списка совпадают, включая шаги
внутри строк (147 061), после половины эмодзи (101/143), не-ASCII ключи и enum. Property-тесты
(`structured.property.test.ts`): случайные схемы и экземпляры валидны на каждом разрезе, 14 видов порчи ловятся,
`forcedContinuation` — каждый символ единственный допустимый и строка максимальна.

---

## Проверка

- `npx vitest run src/engine` — тесты всех модулей.
- Эталоны сняты с настоящих реализаций скриптом `src/engine/tokenizer/fixtures/gen_fixtures.py`:
  ```
  python3 -m venv /tmp/tokvenv && /tmp/tokvenv/bin/pip install tokenizers jinja2
  /tmp/tokvenv/bin/python -I src/engine/tokenizer/fixtures/gen_fixtures.py . /tmp/inferstudy_dl
  ```
  - `tokenizer/fixtures/qwen3-encode.json`: 59 случаев (`tokenizers` 0.23.3): id, смещения, decode
    (с skip_special и без), pre-tokens; русский, английский, цифры, эмодзи и ZWJ, CJK, арабский, код с отступами,
    `\r\n`, серии пробелов, пробельные символы, по которым расходятся Oniguruma и JS, сокращения, NFD, BOM,
    спецтокены (в том числе `split_special_tokens`), целый отрендеренный шаблон;
  - `templates.fixture.json`: 24 рендера jinja2 в окружении как у transformers (11 qwen3 из
    `research/vendor`, 4 llama3, 4 deepseek-r1, 5 gpt-oss из копий HF-шаблонов исследователя).
- Независимая проверка (состязательные эталоны, Python с `-I`, venv `/tmp/inferstudy_venv`: tokenizers 0.23.3,
  transformers 5.22.2, xgrammar 0.2.9, tiktoken 0.14):
  - `tokenizer/fixtures/gen_adversarial.py` → `qwen3-adversarial.json`: 333 строки (случайные смеси Unicode, фузз
    по веткам регулярки, серии пробелов, цифры, ZWJ-эмодзи, `\r\n`, спецлитералы вплотную к тексту, NFC, Unicode 17,
    управляющие символы, код) — id (и `split_special_tokens`), смещения, decode, pre-tokens, o200k-эталон tiktoken;
    80 случайных последовательностей id для decode; 39 пошаговых трасс `DecodeStream`;
  - `tokenizer/fixtures/gen_templates_adversarial.py` → `templates.adversarial.fixture.json`: 132 рендера настоящим
    transformers (`apply_chat_template` для Qwen3: пустой system, ответы инструментов, несколько вызовов, история
    с мыслями при enable_thinking true/false, `<tool_response>` в user, `content: null` и т. д., плюс 100 случайных
    диалогов Qwen3; 8 кейсов Llama 3.1 / DeepSeek-R1 / gpt-oss);
  - `tokenizer/fixtures/gen_xgrammar_masks.py` → `structured.xgrammar.fixture.json`: маски XGrammar на всём словаре;
  - `scripts/gen-onig-classes.py` → `scripts/data/onig-unicode-classes.json`: классы Oniguruma для регулярки.
