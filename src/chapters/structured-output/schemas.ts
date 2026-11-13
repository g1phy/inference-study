/**
 * Схемы для главы «Structured output»: пресеты, разбор своей схемы, пример ответа по схеме,
 * сериализация с разметкой кусков (ключ, строка, число…) и проверка готового JSON по схеме.
 * Чистые функции без React (structured.test.ts).
 */
import type { JsonSchema } from '@/engine/types'

export type PresetId = 'person' | 'enum' | 'nested'

export interface Preset {
  id: PresetId
  label: string
  /** Запрос пользователя (иллюстрация). */
  prompt: string
  schema: JsonSchema
  /** Ответ, который модель пишет под маской (ключи в порядке схемы). */
  instance: Record<string, unknown>
  /** Ответ той же модели без ограничений (иллюстрация «болтовни»). */
  chatter: string
  /** С какого слова модель хочет начать ответ (первое слово chatter). */
  opener: string
}

export const PRESETS: readonly Preset[] = [
  {
    id: 'person',
    label: 'Человек',
    prompt: 'Извлеки данные: «Анна, 29 лет, живёт в Казани». Ответь в JSON.',
    schema: {
      type: 'object',
      properties: { name: { type: 'string' }, age: { type: 'integer' }, city: { type: 'string' } },
      required: ['name', 'age', 'city'],
      additionalProperties: false,
    },
    instance: { name: 'Анна', age: 29, city: 'Казань' },
    chatter: 'Конечно! Вот JSON:\n```json\n{"имя": "Анна", "возраст": "29 лет", "город": "Казань"}\n```',
    opener: 'Конечно',
  },
  {
    id: 'enum',
    label: 'Enum',
    prompt: 'Классифицируй обращение: «Деньги списали дважды, верните срочно!»',
    schema: {
      type: 'object',
      properties: {
        category: { type: 'string', enum: ['оплата', 'доставка', 'возврат', 'другое'] },
        priority: { type: 'string', enum: ['низкий', 'средний', 'высокий'] },
      },
      required: ['category', 'priority'],
      additionalProperties: false,
    },
    instance: { category: 'оплата', priority: 'высокий' },
    chatter: 'Похоже на проблему с оплатой.\n{"category": "двойное списание", "priority": "срочно"}',
    opener: 'Похоже',
  },
  {
    id: 'nested',
    label: 'Вложенный объект',
    prompt: 'Подними Qwen3-8B на двух H100 и включи стриминг.',
    schema: {
      type: 'object',
      properties: {
        model: { type: 'string' },
        gpu: {
          type: 'object',
          properties: { type: { type: 'string', enum: ['H100', 'H200', 'B200'] }, count: { type: 'integer' } },
          required: ['type', 'count'],
          additionalProperties: false,
        },
        stream: { type: 'boolean' },
      },
      required: ['model', 'gpu', 'stream'],
      additionalProperties: false,
    },
    instance: { model: 'Qwen3-8B', gpu: { type: 'H100', count: 2 }, stream: true },
    chatter: 'Вот конфигурация:\n{"model": "Qwen3-8B", "gpu": "2x H100", "stream": "да"}',
    opener: 'Вот',
  },
]

/* ---------------------------------------------------------------------------------------------
 * Своя схема: разбор и проверка поддерживаемого подмножества
 * ------------------------------------------------------------------------------------------- */

const TYPES = ['object', 'string', 'integer', 'number', 'boolean', 'array'] as const
const MAX_PROPS = 6
const MAX_DEPTH = 3
/** Ключевые слова, которые понимает автомат сайта. */
const KNOWN = new Set(['type', 'properties', 'required', 'enum', 'items', 'additionalProperties'])
/** Пояснения без ограничений: молча пропускаем. */
const ANNOTATIONS = new Set(['description', 'title', '$schema', '$id', '$comment', 'default', 'examples'])

/**
 * ignored: ключевые слова схемы, которые маска не проверяет (format, minimum, pattern…).
 * Схема при этом принимается, но читатель должен знать, что эти правила не действуют.
 * extraKeys: где-то в схеме additionalProperties равен true или задан схемой, то есть лишние ключи разрешены.
 * Автомат сайта их не пишет никогда: здесь маска строже схемы, и нарушить это правило ответ не может.
 */
export type SchemaResult = { ok: true; schema: JsonSchema; ignored: string[]; extraKeys: boolean } | { ok: false; error: string }

/** Что разбор схемы узнал по дороге. */
interface Notes {
  ignored: Set<string>
  extraKeys: boolean
}

/** Схема для значений лишних ключей (additionalProperties, заданный схемой): нужна только проверке ответа, автомат её не читает. */
const EXTRA_SCHEMA = new WeakMap<JsonSchema, JsonSchema>()

/** Разбирает текст схемы и проверяет, что она в подмножестве, которое понимает автомат сайта. */
export function parseSchema(text: string): SchemaResult {
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch {
    return { ok: false, error: 'Это не JSON: проверьте кавычки и запятые.' }
  }
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return { ok: false, error: 'Схема должна быть объектом.' }
  if ((raw as JsonSchema).type !== 'object') return { ok: false, error: 'В корне нужен "type": "object".' }
  const notes: Notes = { ignored: new Set<string>(), extraKeys: false }
  const r = normalize(raw as Record<string, unknown>, 'корень', 1, notes)
  return typeof r === 'string' ? { ok: false, error: r } : { ok: true, schema: r, ignored: [...notes.ignored], extraKeys: notes.extraKeys }
}

/** Проверка и нормализация узла схемы: возвращает узел в подмножестве движка или текст ошибки. */
function normalize(s: Record<string, unknown>, path: string, depth: number, notes: Notes): JsonSchema | string {
  if (depth > MAX_DEPTH) return `${path}: слишком глубокая вложенность (не больше ${MAX_DEPTH}).`
  for (const k of Object.keys(s)) if (!KNOWN.has(k) && !ANNOTATIONS.has(k)) notes.ignored.add(k)
  const extra = s.additionalProperties
  if (extra !== undefined && typeof extra !== 'boolean' && (!extra || typeof extra !== 'object' || Array.isArray(extra))) {
    return `${path}: additionalProperties должен быть true, false или схемой.`
  }
  let type = s.type
  if (s.enum !== undefined) {
    if (!Array.isArray(s.enum) || s.enum.length === 0 || !s.enum.every((v) => typeof v === 'string' || typeof v === 'number')) {
      return `${path}: enum должен быть непустым списком строк или чисел.`
    }
    if (type === undefined) {
      // стандартный {"enum": [...]} без type: тип выводим из значений
      const vals = s.enum as Array<string | number>
      if (vals.every((v) => typeof v === 'string')) type = 'string'
      else if (vals.every((v) => typeof v === 'number')) type = vals.every((v) => Number.isInteger(v)) ? 'integer' : 'number'
      else return `${path}: в enum и строки, и числа: укажите "type".`
    }
  }
  if (typeof type !== 'string' || !(TYPES as readonly string[]).includes(type)) {
    return `${path}: поддерживаются типы ${TYPES.join(', ')}.`
  }
  const out: JsonSchema = { type: type as JsonSchema['type'] }
  if (Array.isArray(s.enum)) out.enum = s.enum as Array<string | number>
  if (type === 'object') {
    const props = s.properties
    if (!props || typeof props !== 'object' || Array.isArray(props)) return `${path}: у объекта нужны properties.`
    const keys = Object.keys(props)
    if (keys.length === 0) return `${path}: нужно хотя бы одно свойство.`
    if (keys.length > MAX_PROPS) return `${path}: не больше ${MAX_PROPS} свойств, чтобы пример оставался коротким.`
    if (s.required !== undefined && (!Array.isArray(s.required) || !s.required.every((k) => typeof k === 'string' && keys.includes(k)))) {
      return `${path}: required должен перечислять существующие свойства.`
    }
    out.properties = {}
    for (const k of keys) {
      const child = (props as Record<string, unknown>)[k]
      if (!child || typeof child !== 'object' || Array.isArray(child)) return `${path}.${k}: описание свойства должно быть объектом.`
      const e = normalize(child as Record<string, unknown>, `${path === 'корень' ? '' : path + '.'}${k}`, depth + 1, notes)
      if (typeof e === 'string') return e
      out.properties[k] = e
    }
    if (Array.isArray(s.required)) out.required = s.required as string[]
    // лишние ключи схема разрешает, а автомат сайта их не пишет никогда (маска строже схемы).
    // Флаг нужен проверке ответа: лишний ключ тогда не ошибка, а его значение сверяется со схемой additionalProperties
    if (extra !== undefined && extra !== false) {
      notes.extraKeys = true
      out.additionalProperties = true
      // пустая схема {} разрешает любое значение, как true
      if (extra !== true && Object.keys(extra as object).length > 0) {
        const e = normalize(extra as Record<string, unknown>, `${path === 'корень' ? '' : path + '.'}additionalProperties`, depth + 1, notes)
        if (typeof e === 'string') return e
        EXTRA_SCHEMA.set(out, e)
      }
    }
  }
  if (type === 'array') {
    const items = s.items
    if (!items || typeof items !== 'object' || Array.isArray(items)) return `${path}: у массива нужен items.`
    const e = normalize(items as Record<string, unknown>, `${path}[]`, depth + 1, notes)
    if (typeof e === 'string') return e
    out.items = e
  }
  return out
}

/* ---------------------------------------------------------------------------------------------
 * Пример ответа по схеме (для своей схемы)
 * ------------------------------------------------------------------------------------------- */

const STRING_BY_KEY: Record<string, string> = {
  name: 'Анна',
  city: 'Казань',
  model: 'Qwen3-8B',
  title: 'Отчёт',
  country: 'Россия',
  email: 'anna@example.com',
  text: 'Привет',
  summary: 'Кратко',
  language: 'ru',
  type: 'H100',
  status: 'готово',
}
const INT_BY_KEY: Record<string, number> = { age: 29, count: 2, year: 2026, tokens: 512, id: 7, price: 990 }

/** Правдоподобное значение по схеме (детерминированно). */
export function exampleValue(schema: JsonSchema, key = ''): unknown {
  if (schema.enum && schema.enum.length > 0) return schema.enum[0]
  switch (schema.type) {
    case 'string':
      return STRING_BY_KEY[key.toLowerCase()] ?? 'пример'
    case 'integer':
      return INT_BY_KEY[key.toLowerCase()] ?? 42
    case 'number':
      return 3.5
    case 'boolean':
      return true
    case 'array':
      return schema.items ? [exampleValue(schema.items, key), exampleValue(schema.items, key + '2')] : []
    case 'object': {
      const out: Record<string, unknown> = {}
      for (const [k, s] of Object.entries(schema.properties ?? {})) out[k] = exampleValue(s, k)
      return out
    }
  }
}

/** Та же модель без ограничений: болтовня, русские ключи и строки вместо чисел (иллюстрация). */
export function chatterFor(schema: JsonSchema): string {
  const bad: Record<string, unknown> = {}
  for (const [k, s] of Object.entries(schema.properties ?? {})) {
    const v = exampleValue(s, k)
    bad[RU_KEYS[k] ?? k] = typeof v === 'number' ? `${v} шт.` : typeof v === 'boolean' ? 'да' : v
  }
  return 'Конечно! Вот JSON:\n```json\n' + JSON.stringify(bad).replace(/","/g, '", "').replace(/":/g, '": ') + '\n```'
}

/** Как модель могла бы назвать ключ по-русски (соблазн, который маска запретит). */
export const RU_KEYS: Record<string, string> = {
  name: 'имя',
  age: 'возраст',
  city: 'город',
  category: 'категория',
  priority: 'приоритет',
  model: 'модель',
  gpu: 'видеокарта',
  type: 'тип',
  count: 'количество',
  stream: 'стрим',
  title: 'заголовок',
  email: 'почта',
}

/* ---------------------------------------------------------------------------------------------
 * Сериализация с разметкой кусков (разделители как у json.dumps: ', ' и ': ')
 * ------------------------------------------------------------------------------------------- */

export type SpanKind = 'punct' | 'key' | 'string' | 'number' | 'bool' | 'enum'

export interface Span {
  start: number
  end: number
  kind: SpanKind
  /** Ключ поля, к которому относится кусок. */
  key?: string
  schema?: JsonSchema
}

export function serialize(value: unknown, schema: JsonSchema): { text: string; spans: Span[] } {
  let text = ''
  const spans: Span[] = []
  const push = (s: string, kind: SpanKind, key?: string, sch?: JsonSchema) => {
    if (!s) return
    const last = spans[spans.length - 1]
    if (kind === 'punct' && last && last.kind === 'punct') last.end += s.length
    else spans.push({ start: text.length, end: text.length + s.length, kind, key, schema: sch })
    text += s
  }
  const walk = (v: unknown, s: JsonSchema | undefined, key?: string) => {
    if (s?.enum && s.enum.length > 0) {
      push(JSON.stringify(v), 'enum', key, s)
      return
    }
    if (Array.isArray(v)) {
      push('[', 'punct')
      v.forEach((x, i) => {
        if (i > 0) push(', ', 'punct')
        walk(x, s?.items, key)
      })
      push(']', 'punct')
    } else if (v && typeof v === 'object') {
      push('{', 'punct')
      Object.entries(v as Record<string, unknown>).forEach(([k, x], i) => {
        if (i > 0) push(', ', 'punct')
        push(JSON.stringify(k), 'key', k)
        push(': ', 'punct')
        walk(x, s?.properties?.[k], k)
      })
      push('}', 'punct')
    } else if (typeof v === 'string') push(JSON.stringify(v), 'string', key, s)
    else if (typeof v === 'number') push(String(v), 'number', key, s)
    else if (typeof v === 'boolean') push(String(v), 'bool', key, s)
    else push('null', 'punct')
  }
  walk(value, schema)
  return { text, spans }
}

/* ---------------------------------------------------------------------------------------------
 * Проверка готового JSON по схеме (чтобы показать, что именно нарушила «болтовня»)
 * ------------------------------------------------------------------------------------------- */

export function validate(value: unknown, schema: JsonSchema, path = ''): string[] {
  const at = path || 'ответ'
  const issues: string[] = []
  if (schema.enum && schema.enum.length > 0) {
    if (!schema.enum.includes(value as string | number)) issues.push(`${at}: значения ${JSON.stringify(value)} нет в enum`)
    return issues
  }
  switch (schema.type) {
    case 'object': {
      if (!value || typeof value !== 'object' || Array.isArray(value)) return [`${at}: ожидался объект`]
      const obj = value as Record<string, unknown>
      const props = schema.properties ?? {}
      for (const k of schema.required ?? []) if (!(k in obj)) issues.push(`нет обязательного ключа "${k}"`)
      for (const k of Object.keys(obj)) {
        const at = path ? `${path}.${k}` : k
        if (k in props) issues.push(...validate(obj[k], props[k], at))
        else if (schema.additionalProperties !== true) issues.push(`лишний ключ "${k}"`)
        else {
          // лишние ключи схема разрешает; если для них задана схема, значение сверяем с ней
          const extra = EXTRA_SCHEMA.get(schema)
          if (extra) issues.push(...validate(obj[k], extra, at))
        }
      }
      return issues
    }
    case 'array':
      if (!Array.isArray(value)) return [`${at}: ожидался массив`]
      value.forEach((x, i) => schema.items && issues.push(...validate(x, schema.items, `${at}[${i}]`)))
      return issues
    case 'integer':
      return typeof value === 'number' && Number.isInteger(value) ? [] : [`${at}: ожидалось целое число, пришло ${JSON.stringify(value)}`]
    case 'number':
      return typeof value === 'number' ? [] : [`${at}: ожидалось число, пришло ${JSON.stringify(value)}`]
    case 'boolean':
      return typeof value === 'boolean' ? [] : [`${at}: ожидалось true или false, пришло ${JSON.stringify(value)}`]
    case 'string':
      return typeof value === 'string' ? [] : [`${at}: ожидалась строка, пришло ${JSON.stringify(value)}`]
  }
  return issues
}

/** Пытается разобрать ответ: сначала целиком, потом самый внешний {…} внутри текста. */
export function tryParse(text: string): { whole: boolean; value?: unknown; error?: string } {
  try {
    return { whole: true, value: JSON.parse(text) }
  } catch (e) {
    const a = text.indexOf('{')
    const b = text.lastIndexOf('}')
    const error = e instanceof Error ? e.message : String(e)
    if (a >= 0 && b > a) {
      try {
        return { whole: false, value: JSON.parse(text.slice(a, b + 1)), error }
      } catch {
        /* внутри тоже не JSON */
      }
    }
    return { whole: false, error }
  }
}
