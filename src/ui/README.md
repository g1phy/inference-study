# ui-kit: «Инференс LLM наглядно»

Всё, из чего собираются главы. Визуальные правила в `DESIGN.md` (обязательно), здесь API.
Живые примеры всех компонентов: **`/#kit`** (переключатель «Обе рядом» показывает светлую и тёмную тему одновременно).

```tsx
import { Chapter, Figure, Player, usePlayer, TokenStrip, fmt } from '@/ui'
import { useSetting, selectStage } from '@/state/settings'
```

Алиас `@/` → `src/` настроен в `vite.config.ts`, `vitest.config.ts` и `tsconfig.json`.

---

## 0. Договорённости

**Глава.** Файл `src/chapters/<id>/index.tsx`, `export default` компонент, корень всегда `<Chapter id="<id>">`.
Номер, заголовок, короткое имя и черновой лид берутся из реестра `src/content/chapters.ts` (его не правим; свой лид передаём пропсом `lead`).
Главы грузятся лениво и монтируются прогрессивно: пока глава не смонтирована, вместо неё стоит её шапка и резерв высоты.
Поэтому:
- тяжёлое считаем в `useMemo`, анимации и таймеры останавливаем вне экрана (`useInView`, `usePlayer` делает это сам);
- глобальные слушатели только в `useEffect` с очисткой; `window.addEventListener('scroll')` запрещён (DESIGN §1);
- якоря внутри главы называем с префиксом id главы: `id="attention-figure-qkv"` (тогда ссылка `#attention-figure-qkv` сработает даже до монтирования главы).

**Раскладка главы.** Прямые потомки `<Chapter>` встают в колонку прозы (60ch, около 70-75 знаков кириллицы в строке). Шире выходят `<Figure width="wide|full">` и `<Block width="wide|full">`.
На экранах 1200-1599 px `full` равна `wide` (разница в 40 px читалась как сдвиг); шире она от 1600 px.
Абзацы, списки, `h3`, `h4` можно писать прямо в `<Chapter>`: стили прозы применяются к прямым потомкам. Отступы между блоками ставятся сами.
Внутри своих сеток ставьте `min-w-0` детям или `grid-cols-[minmax(0,1fr)…]`, иначе широкий контент (лента токенов, матрица) распирает колонку.

**Цвета.** Только токены. Стандартная палитра Tailwind выключена (`bg-gray-100`, `text-white` не существуют).

| Что | Tailwind | JS / inline |
|---|---|---|
| нейтральные | `bg-bg bg-surface bg-surface-2 bg-surface-3 text-ink text-ink-2 text-ink-3 border-line border-line-strong` | `var(--ink)` и т.д. |
| акцент (только UI) | `bg-accent text-accent bg-accent-soft text-accent-ink hover:bg-accent-hover` | `cssVar('accent')` |
| сущности данных | `bg-prefill bg-decode bg-q bg-k bg-v bg-tool bg-reasoning border-special` (+ `/20` для прозрачности) | `cssVar('k')`, `tint('v', 14)`, `mix('q', 40)` |
| статусы (с иконкой и словом) | `text-good text-warn text-bad` | `cssVar('bad')` |
| шкалы heatmap | | `seqColor(t)`, `seqInk(t)`, `divColor(t)`, `divInk(t)`, `legendGradient('seq')` |

Тексты подписей и значений всегда цветом текста (`ink`, `ink-2`, `ink-3`), цвет сущности несёт метка рядом (квадрат, штрих, подложка).
Мелкий текст на подложке `accent-soft` пишите `text-accent-hover`, а не `text-accent`: сам акцент даёт там 4,3:1 (ниже AA).
`--ink-3` в светлой теме `#6e6c66`, а не `#7d7b75` из таблицы DESIGN §3.1: исходный цвет не проходит AA (§8) для 12-14px текста (3,9:1 на `--bg`).
Тёмная тема: переменные переключаются сами (атрибут `data-theme` на `<html>` или на любом поддереве). Вариант `dark:` есть, но почти не нужен.

**Шрифты и размеры.** `font-sans` = Onest, `font-mono` = JetBrains Mono (программистские лигатуры выключены, `<|im_start|>` читается как есть).
Шкала размеров (не выдумывайте промежуточные 12,5/13,5/14,5 px):

| px | класс | что |
|---|---|---|
| 17 | `text-base` | проза |
| 14 | `text-sm` (`text-caption`) | пояснения в фигурах, подписи под фигурами, таблицы |
| 13 | `text-fig` | подписи и значения внутри фигур, легенды |
| 12 | `text-xs` (`text-axis`) | оси, мелкие подписи, бейджи |
| 11 | `text-2xs` | только плотные ячейки и ID токенов, у которых есть тултип |

Заголовок фигуры 17px, внутренние заголовки шагов и панелей в фигуре не крупнее 15px (`text-[15px] font-semibold`). Числа в колонках `tabular-nums`.
`tabular-nums` только для таблиц, осей, матриц и меняющихся счётчиков: у Onest табличная «1» широкая, и номер «11» в заголовке читается как «1 1».
Радиусы: `rounded-panel` (12px, панели и фигуры), `rounded-control` (8px, кнопки и поля), `rounded-chip` (6px, токены и бейджи).
Брейкпоинты: `md:` от 768px, `desk:` от 1200px (раскладка с левой навигацией). Тень только у поповеров: `shadow-pop`.

**Z-index** (`src/ui/z.ts`): `Z.nav = 40`, `Z.popover = Z.tooltip = 50`, `Z.modal = 60`. Других нет.

**Движение.** `motion` (`import { motion } from 'motion/react'`), анимируем `transform` и `opacity`. Приложение обёрнуто в `<MotionConfig reducedMotion="user">`,
для своей логики есть `useReducedMotion()`. Глобальный CSS при reduced motion гасит transition и animation.

**Тексты.** Без эмодзи (кроме данных в демо токенизации), без тире-декораций в подписях и кнопках, кавычки «ёлочки», числа только через `fmt`.

---

## 1. Состояние (`src/state/*`)

### settings.ts
```ts
const modelId = useSetting('modelId')           // ПО УМОЛЧАНИЮ: подписка на одно поле
const [settings, patch] = useSettings()        // объект целиком: для симулятора и редких мест
patch({ userText: 'Привет', temperature: 0.2 }) // частичное обновление, проходит валидацию
getSettings(); patchSettings(p); resetSettings(); subscribeSettings(fn) // вне React (живые настройки)
DEFAULT_SETTINGS, SCENARIO_IDS, STAGE_IDS, SETTINGS_LIMITS, sanitizeSettings(raw)
```
Хранится в `localStorage['inference-study:settings:v1']`, мусор и чужие значения откатываются к умолчаниям по полю.

**Живые и принятые настройки.** Читатель печатает в симуляторе, а текст запроса читают десятки фигур на всём сайте.
Чтобы каждая буква не перерисовывала и не перетокенизировала весь сайт:
- главы вне hero обёрнуты в App в `<SettingsScope mode="committed">`: там `useSetting` и `useSettings` отдают **принятые** настройки:
  `userText` и `systemText` приходят через `SETTINGS_COMMIT_DELAY` (250 мс) после последнего нажатия, остальные поля сразу
  (смена сценария, флагов, модели применяется одним обновлением вместе с текстом);
- глава hero (симулятор) читает **живые** настройки;
- `useCommittedSetting(key)` и `useLiveSetting(key)` выбирают явно (например, чип «Ваш запрос» в навигации берёт принятый текст);
  `getCommittedSettings()`, `subscribeCommittedSettings(fn)`, `flushCommittedSettings()` вне React.

**`useSettings()` отслеживает чтение.** Снимок-прокси перерисовывает компонент только при изменении тех полей, которые он прочитал
(в рендере, эффекте или обработчике). Глава, читающая `settings.modelId`, не перерисовывается от набора текста.
Перебор ключей (`{...settings}`, `Object.keys`, `JSON.stringify`) подписывает на все поля. Снимок только для чтения (запись бросает TypeError).
Всё равно предпочитайте `useSetting('modelId')`: это явно и дешевле. В инициализаторе `useState` можно взять `getSettings().temperature`, тогда подписки нет вовсе.

### prompt.ts
```ts
const { tokens, status } = useReaderPromptTokens() // токены промпта читателя: шаблон Qwen3 + токенизатор Qwen3
const n = useReaderPromptTokenCount()               // только длина (null, пока словарь грузится): для корней глав
readerPromptTokens(tokenizer, system, user, thinking); renderReaderPrompt(system, user, thinking)
```
Один общий кэш на страницу: для одного текста токены считаются один раз, сколько бы фигур ни спросили.
Умолчания (SPEC §3): сценарий `thinking`, system «Ты полезный ассистент.», вопрос про KV cache, `agg`, холодный кэш, `qwen3-8b`, `h100-sxm`, temperature 0,7.

### Выбранная стадия симулятора
```ts
selectStage('route')                  // открыть стадию в инспекторе И проскроллить к симулятору (ссылки из глав)
selectStage('decode', { scroll: false }) // внутри самого симулятора
const stage = useSelectedStage()      // StageId | null
const { stage, nonce, fromLink } = useStageSelection() // nonce растёт на каждый вызов: можно реагировать на повторный клик
```
Готовая кнопка для глав: `<SimulatorLink stage="route" />`.

### theme.ts
```ts
const { theme, resolved, setTheme, toggle } = useTheme() // theme: 'light' | 'dark' | 'system', resolved: 'light' | 'dark'
```
Ключ `localStorage['inference-study:theme']` читается в `index.html` до отрисовки. Цвета от темы в JS почти никогда не нужны.

### nav.ts
```ts
navigateTo('kv-cache')            // смонтировать главу, прыгнуть, удержать позицию, pushState, фокус на заголовок
scrollToSimulator({ focusPrompt: true })          // издалека: запись в истории, «Назад» вернёт к месту чтения
scrollToSimulator({ history: 'none' })            // только прокрутить, без записи в истории
useActiveChapter()                // id главы под линией 30% экрана (scrollspy)
chapterFromHash()                 // id главы из адресной строки или null
useSimulatorReturn()              // глава, из которой прыгнули к симулятору (ссылка, чип), или null
returnFromSimulator()             // вернуться к месту чтения: то же, что «Назад» в браузере
```
**Договорённость с симулятором (S1):** блок симулятора имеет `id="simulator"`, поле сообщения пользователя `id="prompt-input"`.
К ним скроллят `selectStage()` и чип «Ваш запрос» в навигации.

**«Назад» возвращает к абзацу, а не к началу главы.** Перед прыжком (`navigateTo`, `scrollToSimulator` дальше экрана)
место чтения запоминается в текущей записи истории: глава под верхним краем окна со смещением, элемент на линии
чтения и элемент в фокусе. App слушает `popstate` (`handlePopState`), возвращает окно на место, держит его, пока выше
догружаются главы, и возвращает фокус на ссылку, по которой ушли. Своя `history.pushState`/`replaceState` в главах не нужна;
если всё же понадобится, сохраняйте `history.state` (в нём ключ записи и место), а не `null`.

**Якорь внутри «Глубже»** (`#sampling-figure-pipeline`) работает: `scrollToAnchor(el)` (его зовут `navigateTo`, `pinScrollTo`
и App) шлёт `beforematch` закрытым блокам вокруг цели (`revealHidden`), встаёт на сам блок, а когда он раскрылся, удержание
позиции докручивает до фигуры. Исключение: `<Deeper lazy>` монтирует детей только после открытия, так что якоря внутри
него до открытия нет.

### store.ts
`createStore(initial)`, `useStore(store)`, `useStoreSelector(store, sel)`, `shallowEqual`: маленький внешний стор, если главе нужно своё общее состояние между фигурами.
`useTrackedStore(store)` / `createTracker(store)`: подписка на объектный стор по факту чтения полей (так устроен `useSettings`).

---

## 2. Реестр глав (`src/content/chapters.ts`)

```ts
CHAPTERS, CHAPTER_IDS, PARTS               // порядок как в SPEC §2
type ChapterId = 'hero' | 'request' | …
getChapter(id), findChapter(str), chapterIndex(id), nextChapter(id), prevChapter(id)
getPart(partId), chaptersOfPart(partId), isFirstInPart(id), isChapterId(str)
```
`ChapterInfo`: `{ id, part, number, title, nav, lead, estimatedHeight?, load }`.

Глоссарий: `src/content/glossary.ts` → `GLOSSARY`, `GLOSSARY_CATEGORIES`, `findTerm(t)`, `normalizeTerm(s)`; данные в `glossary.json` (копия `research/glossary.json`, поля `term, aliases, ru, why, category`, необязательное `chapter`).

---

## 3. Структура главы

### Chapter
```tsx
<Chapter id="attention" lead={<>Attention даёт каждому токену…</>}>…</Chapter>
```
| prop | тип | |
|---|---|---|
| `id` | `ChapterId` | обязательно; рендерит `<section id>` |
| `lead` | `ReactNode` | лид; по умолчанию черновой из реестра |
| `title` | `ReactNode` | заголовок вместо реестрового (редко) |
| `header` | `'default' \| 'hero' \| 'none'` | `hero` у главы hero (h1), `none` для своей вёрстки шапки |
| `next` | `boolean = true` | ссылка «Дальше: …» в конце |

Номер главы показывается обычным начертанием рядом с заголовком (без «бровей»). `ChapterHeader` экспортирован отдельно.

### Prose, Block, Panel
```tsx
<Prose><p>…</p><ul><li>…</li></ul></Prose>
<Block width="wide"><div className="grid gap-4 md:grid-cols-3">…</div></Block>
<Panel>Пояснение к шагу внутри фигуры</Panel>   <Panel tone="outline">Пример, справка</Panel>
```
`Prose`: абзацы, списки, `h3`/`h4`, `code`, ссылки (DESIGN §2). `Block({ width: 'narrow' | 'wide' | 'full', id?, className? })`: блок без рамки нужной ширины.
`Panel({ tone: 'inset' | 'outline' = 'inset', as?, className? })`: панель пояснения в фигуре. Два тона на весь сайт, без полупрозрачных вариаций `bg-surface-2/60`.

### Figure
```tsx
<Figure title="Attention по шагам" width="wide" note="упрощённо"
  inputs={<Segmented label="Пример" … />}
  player={<Player {...p.bind} labels={labels} />}
  controls={<Slider … />}
  caption="Попробуйте: нажмите на ячейку, чтобы увидеть слагаемые."
  source="Веса условные: модель с одной головой, слова вместо токенов.">
  …
</Figure>
```
Анатомия одна на весь сайт: **сценарий сверху, проигрывание снизу**. `inputs` (выбор примера, пресета, сценария) встаёт строкой под заголовком,
`player` всегда первая строка нижней панели на всю ширину, `controls` (настройка демо: слайдеры, переключатели) под плеером.

| prop | тип | |
|---|---|---|
| `title` | `ReactNode` | заголовок фигуры (17px) |
| `width` | `'narrow' \| 'wide' \| 'full'` = `'wide'` | 60ch / 1040px / 1280px (до 1600px экрана `full` = `wide`) |
| `inputs` | `ReactNode` | выбор сценария или пресета: строка под заголовком, до тела |
| `player` | `ReactNode` | `<Player {...p.bind} />`: первая строка нижней панели |
| `controls` | `ReactNode` | панель контролов (авто-сетка колонок от 200px); с `player` встаёт под ним |
| `controlsPosition` | `'bottom' \| 'top' \| 'side'` = `'bottom'` | `side`: справа, если фигура шире 760px, иначе снизу. `player` всегда снизу |
| `stickyControls` | `boolean` | нижняя панель липнет к низу экрана, пока фигура на экране. По умолчанию сама липнет на телефоне, если в ней только Player (высокие пошаговые фигуры; скорость тогда скрыта); `true` на любой ширине; `false` никогда |
| `caption` | `ReactNode` | подпись; «Попробуйте:» в начале строки выделяется само. Подпись про эксперимент |
| `source` | `ReactNode` | откуда данные: строка 12px под подписью («Модель условная: …») |
| `note` | `ReactNode` | пометка о точности в шапке справа: «упрощённо», «иллюстрация», «условные числа» |
| `actions` | `ReactNode` | элементы справа в шапке (переключатели вида, бейджи) |
| `id`, `padded = true`, `className`, `bodyClassName`, `aria-label` | | |

Тело фигуры является container query контейнером (`@sm: @md: @lg:` работают по ширине фигуры). Фигура фокусируется кликом,
поэтому клавиши `Player` работают, когда пользователь кликнул внутрь. Ошибка рендера внутри фигуры не роняет главу: на месте фигуры сообщение «Эта фигура не загрузилась».
Содержимое шире фигуры обрезается рамкой: широкие вещи кладите в `<ScrollX>` (MatrixView и Heatmap делают это сами).

Живые регионы: `<FigureLive as="p">…</FigureLive>` или `aria-live={useFigureLive()}` в дочернем компоненте. Они озвучиваются, только пока
фокус внутри фигуры (клик тоже ставит фокус): текст, который зависит от запроса читателя, иначе зачитывался бы из далёких фигур, пока он
печатает в симуляторе. `useFigureFocusWithin()` даёт то же как boolean; подпись `Player` уже так устроена.

### ScrollX, ScrollHint
```tsx
<ScrollX label="Таблица стоимости"><table className="data-table min-w-[560px]">…</table></ScrollX>
```
Горизонтальная прокрутка для широкого содержимого: если не влезает, обрезанный край затухает, под блоком подсказка «шире экрана: прокрутите вбок»
(на телефоне полос прокрутки не видно). `hint` (по умолчанию есть, можно свой текст), `label` для скринридера, `scrollerClassName`.
Своя прокрутка: `const [ref, edges] = useScrollEdges()` + класс `scroll-fade` + `{edges !== 'none' && <ScrollHint />}`.

### Таблица данных
`<table className="data-table">`: один вид таблиц на весь сайт (14px, заголовки 13px `--ink-3`, линии `--line`, `tabular-nums`). Числовые столбцы `className="num"` (вправо).
Широкую таблицу оборачивайте в `<ScrollX>` и задавайте ей `min-w-[…]`.

### Callout
```tsx
<Callout kind="production" title="таймауты">Стриминг улучшает воспринимаемую задержку…</Callout>
```
`kind: 'intuition' | 'production' | 'pitfall' | 'precise'` (Интуиция, В проде, Подвох, Точнее), `title?`, `children`. Тело врезки стилизовано как проза.

### Deeper
```tsx
<Deeper title="зачем делить на √d_k">…</Deeper>
```
`title = 'Глубже'` (свой выводится как «Глубже: …»), `defaultOpen = false`, `lazy = false` (монтировать детей только после первого открытия).
Кнопка с `aria-expanded`; закрытый текст находит поиск по странице и блок раскрывается сам. Ссылка на фигуру внутри
(`#<глава>-figure-…`, `navigateTo`) тоже раскрывает блок: nav.ts шлёт ему `beforematch`. С `lazy` такой ссылки до открытия нет.

### Takeaways
`<Takeaways items={['…', <>…</>]} title?="Запомнить" simulator?="decode" />`: 2-4 пункта в конце главы.
`simulator`: ссылка «Где это в симуляторе» справа в шапке. Это лучшее место для неё: одно и то же во всех главах.

### ChapterLink, SimulatorLink, linkClass
```tsx
Подробнее в главе <ChapterLink to="kv-cache" />.        // текст: короткое имя главы
<ChapterLink to="routing" variant="card" />              // карточка: номер, заголовок, лид
<SimulatorLink stage="prefill" />                        // «Где это в симуляторе»
```
Переход монтирует цель, если она ещё не загружена. `linkClass`: классы текстовой ссылки для своих элементов.
`SimulatorLink({ stage, children?, variant: 'inline' | 'chip' = 'inline', prepare? })`: отдельной строкой в главе (прямой потомок `<Chapter>`) сама рисуется чипом с рамкой, внутри абзаца остаётся текстовой ссылкой.
`prepare` вызывается до перехода: включить сценарий, без которого на стадии нечего показать (structured-output включает JSON через `patchSettings`).

### Term
```tsx
Здесь работает <Term t="prefill" />, дальше <Term t="KV cache">кэш ключей и значений</Term>.
```
`t`: термин или alias (регистр и ё не важны). Поповер с определением, категорией, «почему важно» и главой (если задана и это не текущая глава). Открывается наведением, фокусом с клавиатуры, тапом; Escape закрывает.
Нет в глоссарии: обычный текст и `console.warn` в dev. Оборачивайте термин при первом появлении в главе.
Глоссарий грузится отдельным чанком (`loadGlossary()`): App подгружает его в простое, первое наведение на термин тоже.

---

## 4. Пошаговые демо

### usePlayer + Player
```tsx
const p = usePlayer(STEPS.length, { autoplay: true, intervalMs: 1400 })
<Figure controls={<Player {...p.bind} labels={STEPS.map((s) => s.label)} />}>{render(STEPS[p.step])}</Figure>
```
`usePlayer(steps, { autoplay?, intervalMs? (число или (step) => мс), loop?, initialStep? })` →
`{ step, setStep, playing, setPlaying, toggle, next, prev, reset, speed, setSpeed, atStart, atEnd, steps, bind }`.

`Player({ steps, step, onStep, playing, onPlayingChange, speed?, onSpeedChange?, labels?, speeds = [0.5, 1, 2], viewRef?, counter?, aria-label? })`.

- Кладите плеер в слот `player` у `Figure` (или в `controls`): он всегда занимает строку сетки целиком (`col-span-full`), обёртки не нужны.
- Автоплей стартует, когда на экране пятая часть фигуры (или половина экрана у высокой фигуры), и не стартует при `prefers-reduced-motion`.
- Пауза, только когда вся фигура ушла с экрана, и продолжение при возврате (через `bind.viewRef`; наблюдается фигура, а не панель плеера:
  в высокой фигуре читатель смотрит на схему над плеером).
- `counter={(step, total) => …}`: свой счётчик вместо «Шаг N из M», когда кадры означают время, а «шаг» в главе значит проход модели.
- Клавиши, когда фокус внутри фигуры: пробел пуск/пауза, ← → шаг. Ручной шаг ставит паузу. Один Player на фигуру.
- Пуск на последнем шаге начинает сначала. Анимации шагов делайте с учётом `useReducedMotion()` (мгновенно при reduce).
- Число шагов уменьшилось (читатель укоротил текст): `p.step` уже в новых пределах на том же рендере, `views[p.step]` не выходит за массив.

---

## 5. Данные и визуализации

### TokenChip, TokenStrip, TokenTip, tokenSegments
```tsx
<TokenStrip tokens={tokens} showIds variantOf={(t, i) => (i < promptLen ? 'prompt' : 'generated')}
  highlightIndex={step} onClickIndex={setPicked} />
<TokenChip token={t} variant="generated" showId />
```
`TokenStrip({ tokens, showIds?, showBytes?, wrap = true, highlightIndex?, onHoverIndex?, onClickIndex?, variantOf?, variant = 'default', limit?, tooltip = true, size = 'md', label?, empty? })`

`TokenChip({ token, variant?, showId?, showBytes?, highlight?, alt?, onHover?, onClick?, tooltip = true, size? })`

- Варианты: `default` (два нейтральных фона чередуются), `prompt` (тон prefill), `generated` (тон decode), `muted`.
- Пробел `·`, перевод строки `↵` и реальный перенос, таб `→`, неполный UTF-8 байтами `\xD0`; спецтокены в пунктирной рамке без заливки.
- Тултип: ID, форма в словаре (`Ġ…`), байты hex. В ленте один общий тултип. С `onClickIndex` чипы становятся кнопками, стрелки двигают фокус.
- `wrap={false}`: одна строка с прокруткой, лента берёт ширину родителя. `limit`: первые N и «ещё K токенов».
- `TokenStripSkeleton({ count })`: заглушка, пока грузится токенизатор (`useTokenizer().status === 'loading'`).

### MatrixView
```tsx
<MatrixView label="Веса" data={weights} rowLabels={words} colLabels={words} heat="seq" domain={[0, 1]}
  highlight={{ rows: [7] }} onCellClick={(c) => setRow(c.row)} cellTooltip={(c) => <>…</>} />
```
| prop | |
|---|---|
| `data: number[][]`, `rowLabels?`, `colLabels?` (ReactNode) | длинные подписи столбцов встают вертикально |
| `digits = 2`, `format?` | формат ячейки (по умолчанию `fmt.num(v, digits)`) |
| `heat: 'seq' \| 'diverging' \| false` = `false`, `domain?` | заливка: синяя шкала или знак (минус красный, плюс синий) |
| `colorRole?: 'q' \| 'k' \| 'v'` | матрица сущности: подложка её цвета; с `heat` интенсивность по модулю |
| `highlight?: { rows?, cols?, cells? }`, `dimOthers = true` | рамка акцентом; остальные ячейки бледнеют, но числа в них читаются (`--ink-2`) |
| `onCellHover?`, `onCellClick?`, `cellTooltip?` | интерактив: ячейки доступны с клавиатуры (стрелки, Enter) |
| `maskInfinity = true` | `-Infinity` рисуется «−∞» на штриховке |
| `isMasked?: (row, col) => boolean`, `maskedText = '0'` | маска независимо от значения: нули после softmax над диагональю остаются на штриховке, не приглушаются |
| `cellLabel?: (cell) => string` | подпись ячейки для скринридера; по умолчанию «строка · столбец: значение», если подписи строками |
| `cellSize: 'sm' \| 'md' \| 'lg' \| number` = `'md'` | 44×28, 56×32, 68×40 |
| `label?` | подпись над матрицей (с цветной меткой при `colorRole`) |

Широкая матрица прокручивается внутри: обрезанный край затухает, под ней подсказка «шире экрана: прокрутите вбок».

### Heatmap
```tsx
<Heatmap data={w} rowLabels={toks} colLabels={toks} domain={[0, 1]} valueFormat={(v) => fmt.pct(v)}
  rowAxisLabel="кто смотрит" colAxisLabel="на кого смотрит" isMasked={(r, c) => c > r} />
```
`Heatmap({ data, rowLabels, colLabels, valueFormat?, onHover?, onClick?, domain?, scale = 'seq', highlight?, maxCellSize = 40, minCellSize = 14, showValues = 'auto', legend = true, legendLabel?, rowAxisLabel?, colAxisLabel?, tooltip?, isMasked?, labelFont = 'mono', aria-label? })`.
SVG, ширина по контейнеру; если не влезает даже с минимальной ячейкой, прокручивается внутри (затухание края и подсказка). Легенда шкалы, тултип на наведении и стрелках, скрытая таблица для скринридеров.

### BarChart
```tsx
<BarChart data={cands.map((c) => ({ label: c.text, value: c.p, muted: !c.kept, struck: !c.allowed }))}
  max={1} valueFormat={(v) => fmt.pct(v)} labelFont="mono" color="decode" />
```
`BarChart({ data: { label, value, color?, muted?, struck?, note?, id? }[], valueFormat?, max?, min?, horizontal = true, color = 'prefill', height = 200, barSize?, highlightIndex?, onHover?, onClick?, labelFont = 'sans', showValues = true, aria-label? })`.
Столбики до 24px, скругление только на конце данных, значение у конца. `muted`: приглушён (отсечён сэмплером), `struck`: зачёркнут с иконкой (маска грамматики).
Отрицательные значения растут от нуля в другую сторону. Изменение данных анимируется (передайте стабильный `id`, если порядок меняется).
Пути столбиков (`barPath`) всегда одной структуры и без экспоненты в числах: анимация от нуля не даёт ошибок «Expected arc flag».

### LineChart
```tsx
<LineChart xLabel="Арифметическая интенсивность, FLOP/байт" yLabel="TFLOPS" xScale="log" yScale="log"
  series={[{ id: 'h100', label: 'H100', color: 'prefill', points }]}
  annotations={[{ kind: 'x', x: 295, label: 'ridge point' }, { kind: 'y', y: 500, label: 'SLO 500 мс' }]} />
```
`LineChart({ series: { id, label, color?, points: [x, y][], dashed?, area?, markers? }[], xLabel, yLabel, xScale = 'linear', yScale = 'linear', markers?, annotations?, height = 260, xDomain?, yDomain?, xFormat?, yFormat?, xTicks?, yTicks?, legend?, directLabels?, onHoverX?, aria-label? })`.

- Одна ось Y (двух осей не бывает: нужен второй масштаб, делайте второй график).
- Легенда при 2+ сериях, прямые подписи у концов линий при 2-4 сериях и ширине от 420px (с выносками, если подписи сдвинуты).
- Crosshair: тултип по всем сериям в ближайшей точке X, стрелки с клавиатуры.
- Аннотации: `x` (вертикаль), `y` (горизонталь, порог; `labelAlign: 'start' | 'end'`), `point` (`labelPosition: 'above' | 'below'`), `band` (`x`..`x2`). Пунктир у аннотаций означает порог или отметку.
  Подпись полосы стоит у её правого края; если её задевает подпись отметки `x`, она уходит на вторую строку.
- `NaN` в y разрывает линию. Линейная Y начинается с нуля и заканчивается на тике; задайте `yDomain`, если нужно иначе.

### Formula
```tsx
<Formula block tex="\mathrm{softmax}\!\left(\frac{\cq{Q}\ck{K}^\top}{\sqrt{d_k}}\right)\cv{V}" label="…" />
Размер KV на токен: <Formula tex="2 \cdot L \cdot h_{kv} \cdot d_{head} \cdot b" />.
```
`Formula({ tex, block = false, label?, macros?, className?, narrowTex? })`. `narrowTex`: та же блочная формула по строкам (`\begin{aligned}`), показывается сама, когда основная строка не влезает в колонку (телефон). KaTeX грузится отдельным чанком (App подгружает его в простое после словаря токенизатора); до загрузки на месте формулы заглушка того же размера.
Если KaTeX не загрузился (сеть), вместо формулы её TeX и «Формула не загрузилась, обновите страницу».
Блочная формула шире колонки прокручивается внутри, обрезанный край затухает. На телефоне всё равно делите длинную выкладку на строки
(`narrowTex` с `\begin{aligned}` или две формулы): итог справа читатель иначе не увидит.
Макросы сущностей: `\cq{} \ck{} \cv{} \cprefill{} \cdecode{} \ctool{} \cmuted{}` (подложка цвета сущности). `\text{…}` набирается шрифтом сайта (кириллица).
`renderTex(tex, block)` возвращает HTML или `null`, пока KaTeX не загружен; `loadKatex()` загружает явно.

### CodeBlock
```tsx
<CodeBlock lang="bash" title="запуск" code={'vllm serve Qwen/Qwen3-8B \\\n  --enable-prefix-caching'} highlightLines={[2]} />
```
`CodeBlock({ code, lang: 'json' | 'yaml' | 'bash' | 'jinja' | 'python' | 'text' = 'text', highlightLines?, wrap?, title?, lineNumbers?, maxHeight?, copy = true })`.
С `wrap` продолжение длинной строки встаёт под её текст (висячий отступ), а не в нулевую колонку.
Своя лёгкая подсветка (`highlight(code, lang)` экспортирован). В `text` и `jinja` спецтокены (`<|im_start|>`, `<think>`) обводятся пунктиром. Кнопка «Копировать».

### Stat
`<Stat label="TTFT" value="182" unit="мс" hint="очередь + prefill" swatch="var(--c-prefill)" size="md|lg" />`. Число подписью сверху, без двоеточий; `value` числом форматируется `fmt.num`.

---

## 6. Контролы

Все поля с видимой подписью над контролом, фокус акцентом, доступны с клавиатуры.

| Компонент | Главные props |
|---|---|
| `Button` | `variant: 'primary' \| 'secondary' \| 'ghost'`, `size: 'sm' \| 'md' \| 'lg'`, `icon`, `iconRight` (компонент Phosphor или элемент), `iconOnly` (+ `aria-label`), `fullWidth`, все props `<button>` |
| `ButtonLink` | то же для `<a href>`; `buttonClass({ variant, size })` для своих элементов |
| `Slider` | `label, value, min, max, step = 1, onChange, format?, hint?, scale: 'linear' \| 'log', disabled?, id?` (см. ниже) |
| `Toggle` | `label, checked, onChange, hint?, disabled?` (role="switch") |
| `Segmented` | `label?, options: { value, label, icon?, disabled?, title? }[], value, onChange, size: 'sm' \| 'md', fullWidth?, wrap?` (radiogroup, стрелки; без `wrap` лишнее прокручивается вбок) |
| `Select` | `label, value, onChange, options: { value, label, group?, disabled? }[], hint?, size?` (нативный select) |
| `Tabs` | `tabs: { id, label, icon?, disabled? }[], value, onChange, children?: (active) => ReactNode, aria-label?, size?, wrap?` (`wrap`: вкладки переносятся на вторую строку; без него ряд прокручивается, обрезанный край затухает). Без `children` нет `tabpanel`: если вкладки ничего не показывают сами, берите `Segmented` |
| `TextArea` | `label, value, onChange, rows = 3, hint?, placeholder?, maxLength?, counter: boolean \| 'auto' = 'auto', autoGrow?, maxRows = 10, mono?, id?, labelClassName?, onKeyDown?`: `autoGrow` растёт до `maxRows` строк, дальше прокрутка внутри; `counter='auto'` показывает «3 416 / 4 000», когда текст занял больше половины `maxLength`; `labelClassName` для строки подписи (например, `pr-9` под свою кнопку справа, как крестик в лаборатории шаблона) |
| `Badge` | `tone: 'neutral' \| 'accent' \| 'good' \| 'warn' \| 'bad' \| 'prefill' \| 'decode' \| 'q' \| 'k' \| 'v' \| 'tool' \| 'reasoning' \| 'special'`, `icon?` (у статусов иконка автоматически), `size?` |
| `StatusIcon` | `status: 'good' \| 'warn' \| 'bad' \| 'neutral', size = 16`: иконка статуса для своих таблиц и списков (regular, цвет токена). Рядом всегда слово |
| `Tooltip` | `content, children, side?, align?, delay = 150, wide?, disabled?` (наведение и фокус, тап на тач-экране) |
| `Popover` | `label, content \| (close) => ReactNode, children: (triggerProps) => ReactNode, side?, align?, open?, onOpenChange?` |
| `Kbd` | `<Kbd>Пробел</Kbd>` |
| `Skeleton` | `width?, height?, radius?, lines?, label?, still?`; `TokenStripSkeleton({ count, still? })`. Перелив только на экране; `still` без перелива (заглушка ждёт подъезда фигуры, а не загружается) |
| `FieldLabel`, `FieldHint` | подпись и пояснение для своих полей |

Иконки только `@phosphor-icons/react` с суффиксом `Icon` (`PlayIcon`, `ArrowRightIcon`), weight regular, размеры 16/18/20.

**Slider на log-шкале.** Стрелки двигают значение на каждом нажатии (раньше на малых значениях 1/1000 шкалы округлялась обратно, и бегунок стоял),
PageUp/PageDown прыгают на десятую часть шкалы, Home/End дают ровно `min`/`max`. При `step` 1, 10, 0,1, 0,01… значения прилипают к круглым числам
(8 000, 16 000, 32 000) и степеням двойки (512, 4 096, 8 192), поэтому просьбу подписи «поставьте 16 000» можно выполнить.
**Округляйте шагом, а не в `onChange`:** `step={256}` вместо `onChange={(v) => setCtx(Math.round(v / 256) * 256)}`, иначе слайдер не знает шага.
Значение вне `[min, max]` показывается зажатым в пределы.

---

## 7. Утилиты

### fmt (ru-RU)
| вызов | результат |
|---|---|
| `fmt.num(3.3456)` / `fmt.num(0.5, 2)` | `3,35` / `0,50` (с `digits` ровно столько знаков) |
| `fmt.int(151936)` | `151 936` (неразрывный пробел) |
| `fmt.compact(622e6)` | `622 млн` |
| `fmt.signed(1.2)`, `fmt.multiplier(2.5)` | `+1,2`, `×2,5` |
| `fmt.bytes(1.24e9)` / `fmt.bytes(147456, { mode: 'binary' })` | `1,24 ГБ` / `144 КиБ` |
| `fmt.ms(0.85)`, `fmt.ms(4.93)`, `fmt.ms(1234)` | `850 мкс`, `4,9 мс`, `1,2 с` |
| `fmt.pct(0.42)`, `fmt.pct(0.042)` | `42 %`, `4,2 %` (на вход доля) |
| `fmt.rate(205.3)`, `fmt.rate(21, { long: true })` | `205 ток/с`, `21 токен/с` |
| `fmt.tokens(151669)`, `fmt.count(n, forms)`, `fmt.plural(n, forms)` | `151 669 токенов` (русские формы) |
| `fmt.tokensGen(21)`, `fmt.tokensGen(43)` | `21 токена`, `43 токенов` (родительный падеж: «из …», «около …») |
| `fmt.bandwidth(3350)` | `3,35 ТБ/с` (на вход ГБ/с) |
| `fmt.flops(989e12, { perSecond: true })` | `989 TFLOPS` |
| `fmt.hex([0xd0, 0x9f])` | `D0 9F` |

Минус настоящий (`−`), `NaN` → `н/д`, `±Infinity` → `±∞`.

### Цвета (`color.ts`)
`cssVar(role)`, `tint(role, pct = 14)` (полупрозрачная подложка), `mix(role, pct, base = 'var(--surface)')` (непрозрачная смесь для ячеек с текстом),
`resolveColor(roleOrCss)`, `seqColor(t)` / `seqInk(t)` (t ∈ [0, 1]), `divColor(t)` / `divInk(t)` (t ∈ [−1, 1]), `legendGradient('seq' | 'diverging')`, `ROLE_LABELS`, `DATA_ROLES`.

### Хуки (`hooks.ts`)
| хук | |
|---|---|
| `useInView({ rootMargin?, threshold?, once?, initial? })` → `[ref, inView]` | IntersectionObserver |
| `useMediaQuery(q)`, `MEDIA.tablet / MEDIA.desk / MEDIA.hover` | |
| `useElementSize()` → `[ref, { width, height }]` | первое измерение до отрисовки; все замеры по border-box. В главе за экраном (content-visibility) и в закрытом «Глубже» размер приходит, когда блок подъезжает к экрану: до того `width = 0`, рисуйте по `width > 0` |
| `useScrollEdges(contentKey?)` → `[ref, edges]` | есть ли скрытое содержимое у горизонтальной прокрутки: `'none' \| 'start' \| 'end' \| 'both'`; ставит `data-overflow` для класса `scroll-fade` |
| `useTheme()`, `useReducedMotion()`, `useFontsReady()`, `useLatest(v)` | |
| `measureText(text, font)`, `truncateText(text, maxWidth, font)` | ширина текста для SVG |

### Плавающие слои (`floating.tsx`)
`useTip({ scopeEl?, onOutside?, side?, align?, wide? })` → `{ show(rect, content, side?), hide, node, visible }`: один тултип на компонент (рендерите `{tip.node}`).
Тач: тап показывает тултип (тап не даёт `pointermove`, слушайте ещё `pointerdown`), а `pointerleave` после тапа его закрывать не должен:
`onPointerLeave={(e) => e.pointerType === 'mouse' && clear()}`. Закрывает тап вне `scopeEl` (корень компонента), после чего вызывается `onOutside` для сброса наведения.
Все графики, матрицы и ленты токенов kit так и работают.
`TipCard`, `TipRow({ color, value, label, dashed? })` (значение крупно, подпись следом, штрих цвета серии), `FloatingLayer`, `computePosition`, `pointRect(x, y)`.
Слой ставится в координатах документа и едет вместе со страницей без scroll-слушателей.

### Прочее
`cn(...classes)`, `Z`, `renderIcon(icon, size)`, `ErrorBoundary({ fallback?, label? })` (каждая глава и каждая фигура уже обёрнуты им).

---

## 8. Подвохи

- Внутри своих `grid`/`flex` ставьте `min-w-0` детям с широким содержимым, иначе колонка распирается и фигура обрезается.
- `outline-none` и `outline-hidden` в Tailwind 4.3 ставят `--tw-outline-style: none`, и `focus-visible:outline-2` после них кольца не рисует.
  Не гасите outline совсем: глобальное кольцо `:focus-visible` уже есть, или добавьте `focus-visible:outline-solid`.
- Кольцо фокуса рисуется снаружи элемента (2px + отступ 2px). Внутри `overflow-auto`/`overflow-hidden` (и у краёв фигуры с `padded={false}`) оно обрезается:
  дайте контейнеру поле 4px или рисуйте кольцо внутри (`focus-visible:-outline-offset-2`, так сделано в `Segmented`, `Tabs`, `CodeBlock`).
- Эффекты возвращают только функцию очистки: `useEffect(() => { el.scrollIntoView() }, [])` с фигурными скобками.
  В новых Chromium `scrollTo`/`scrollIntoView` возвращают Promise, и `useEffect(() => el.scrollIntoView())` роняет компонент («destroy is not a function»).
- Не задавайте фигурам фиксированную высоту без учёта подписей осей.
- Цвет никогда не единственный носитель смысла: статус = иконка + слово, серии = легенда + подписи, спецтокены = пунктир.
- Ничего тяжёлого на каждом рендере: симуляции и расчёты в `useMemo` по параметрам.
- Настройки читайте полем: `useSetting('modelId')`, а не весь объект. Текст запроса в главах приходит с задержкой 250 мс после набора (это нормально).
  Тяжёлых потребителей текста (ленты токенов, матрицы) заворачивайте в `memo` и `useDeferredValue`.
- Слайдер округляет сам: передайте `step`, не округляйте в `onChange`.
- Таблицы: класс `data-table`, а не свои размеры и линии. Панели пояснений: `<Panel>`.
- Таймеры и анимации за экраном останавливайте (`useInView`), `usePlayer` делает это сам.
- `motion` и частые перерисовки: анимация с массивом ключевых кадров в `animate` (`animate={{ opacity: [1, 1, 0] }}`) у элемента, который перерисовывается с частотой плеера (около 20 Гц), может перезапускаться и не доходить до конца (так было с точками токенов в симуляторе). Такое движение надёжнее запускать через Web Animations API (`el.animate(...)`) в эффекте.
