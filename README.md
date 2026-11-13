# Инференс LLM наглядно

**Сайт: https://g1phy.github.io/inference-study/**

Интерактивный учебник по инференсу LLM для тех, кто входит в MLOps и inference engineering. Вы пишете свой запрос и
смотрите весь путь от Enter до последнего токена: шаблон чата, токенизацию, KV-aware роутинг, prefill и decode, передачу
KV между воркерами, детокенизацию, парсеры reasoning и tool calls, structured output, SSE, метрики TTFT и ITL.

Дальше 28 глав с интерактивными фигурами:
- **Вход:** запрос, шаблон чата, токенизация (настоящий BPE Qwen3);
- **Внутри модели:** эмбеддинги, слои трансформера, attention и Q/K/V на конкретном примере, логиты и сэмплинг;
- **Две фазы:** prefill и decode, compute-bound и memory-bound (roofline), метрики;
- **Память:** KV cache, PagedAttention, prefix caching, иерархия KV cache и L3 как DaemonSet на каждой ноде;
- **Много запросов:** continuous batching, дезагрегация prefill и decode, параллелизм TP/PP/EP/DP;
- **Движки и оркестрация:** vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, KV-aware routing;
- **Выход:** детокенизация, парсеры, structured output;
- **Ускорения и прод:** speculative decoding, квантизация, прод, шпаргалка всего пути, глоссарий.

Шаблон, токены, хэши блоков, решение роутера, маски structured output, детокенизация, парсеры, SSE и тайминги
считаются по-настоящему, по параметрам модели и GPU. Заготовлен только текст ответа модели: живая LLM в браузере не
запускается, и сайт это подписывает.

## Запуск

```bash
npm ci
npm run dev          # http://localhost:5173/
npm test             # vitest: движок (токенизатор Qwen3, шаблоны, роутер, планировщики…) и главы
npm run typecheck
npm run build        # dist/
```

Нужен Node.js 20.19+ или 22.12+. Каждый пуш в `main` собирается и выкладывается на GitHub Pages
(`.github/workflows/pages.yml`).

Проверка главы в headless Chromium (Chromium от Playwright или `CHROME_PATH`): скриншоты и JSON-отчёт в `/tmp/qa`.

```bash
node tools/qa/check.mjs attention --theme dark --mobile
node tools/qa/check.mjs hero --origin https://g1phy.github.io --base /inference-study/
```

## Устройство

- `src/engine`: byte-level BPE-токенизатор Qwen3 (словарь в `public/tokenizer/qwen3.json`, совпадает с Hugging Face
  `tokenizers`), шаблон чата Qwen3 (совпадает с `transformers`), сэмплинг, инкрементальная детокенизация, стриминговые
  парсеры, маски structured output (совпадают с XGrammar), формула KV-aware роутера Dynamo, модели производительности
  GPU, планировщики, симулятор пути запроса. Описание в `src/engine/README.*.md`.
- `src/ui`: UI-кит (`src/ui/README.md`); визуальная система в `DESIGN.md`.
- `src/chapters/<id>`: главы; `src/chapters/hero`: симулятор.
- `research/numbers.json`: справочные числа по GPU и моделям, с ними сверяются тесты движка.

Факты сверены с официальной документацией vLLM, SGLang, TensorRT-LLM и NVIDIA Dynamo на ноябрь 2026 года. Для
углублённого чтения: книга Philip Kiely «Inference Engineering» (Baseten Books, 2026).

Сторонние компоненты и их лицензии: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
