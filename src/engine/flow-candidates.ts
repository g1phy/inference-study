/**
 * Кандидаты на шаге decode для симулятора (агент S1a): выбранный токен (из скрипта ответа) плюс 4–7
 * правдоподобных альтернатив с синтетическими логитами. Альтернативы это настоящие токены словаря Qwen3:
 * ручные (AltSpec в scenarios.ts) в ключевых местах, иначе подобранные по классу выбранного токена
 * (начало слова, окончание, цифра, пунктуация, JSON-разметка) из небольших пулов и из слов самого ответа.
 *
 * Логиты условные: выбранный токен получает 19–24, альтернативы ниже с разрывом, который зависит от того,
 * насколько модель «уверена» (середина слова и служебные токены уверенно, начало слова и пунктуация нет).
 * Всё детерминировано: случайность только через переданный rng.
 */
import type { Token, Tokenizer } from './types'
import type { Rng } from './rng'
import type { AltSpec } from './scenarios'

/** Класс выбранного токена: от него зависят пул альтернатив и разрыв логитов. */
export type TokenClass =
  | 'special'
  | 'wordStart'
  | 'wordStartNoSpace'
  | 'continuation'
  | 'digit'
  | 'punct'
  | 'whitespace'
  | 'jsonStructural'

/** Кандидат до сэмплинга. */
export interface RawCandidate {
  token: Token
  logit: number
  preview?: string
}

interface Pools {
  wordRu: Token[]
  wordEn: Token[]
  suffixRu: Token[]
  suffixEn: Token[]
  startsRu: Token[]
  startsEn: Token[]
  punct: Token[]
  digits: Token[]
  json: Token[]
  keys: Token[]
  ident: Token[]
}

const WORD_RU = [
  ' и', ' в', ' не', ' на', ' что', ' это', ' с', ' по', ' для', ' как', ' он', ' она', ' его', ' при', ' так', ' то', ' из', ' у',
  ' к', ' за', ' от', ' также', ' может', ' можно', ' нужно', ' только', ' который', ' когда', ' если', ' чтобы', ' все', ' уже',
  ' ещё', ' есть', ' будет', ' очень', ' более', ' один', ' этот', ' такой', ' поэтому', ' здесь', ' сейчас', ' просто', ' каждый',
  ' их', ' мы', ' вы', ' я', ' они', ' без', ' до', ' после', ' через', ' между', ' о', ' а', ' но', ' или', ' же', ' бы', ' модель',
  ' токен', ' память', ' данные', ' ответ', ' текст', ' шаг', ' время', ' запрос',
]
const WORD_EN = [' the', ' a', ' of', ' to', ' and', ' is', ' in', ' cache', ' model', ' token', ' memory', ' data', ' GPU', ' KV', ' attention', ' step']
const SUFFIX_RU = [
  'а', 'ы', 'и', 'е', 'о', 'у', 'ю', 'я', 'ь', 'й', 'ой', 'ый', 'ий', 'ая', 'ое', 'ые', 'ие', 'ов', 'ев', 'ам', 'ами', 'ах', 'ом', 'ем',
  'ей', 'ть', 'ет', 'ют', 'ит', 'ят', 'ся', 'ия', 'ость', 'ение', 'ного', 'ными', 'ные', 'ный', 'ная', 'ться', 'ал', 'ла', 'ли', 'ло',
  'ка', 'ки', 'ку',
]
const SUFFIX_EN = ['s', 'ing', 'ed', 'er', 'ly', 'ion', 'es', 'al', 'ic', 'ive', 'ment']
const STARTS_RU = ['В', 'К', 'П', 'Э', 'Н', 'О', 'С', 'Т', 'М', 'Д', 'И', 'А', 'Это', 'Но', 'Так']
const STARTS_EN = ['The', 'A', 'In', 'This', 'It']
const PUNCT = [',', '.', ':', ';', '!', '?', '\n', '\n\n', ' (', ')', ' -', '.\n', ' «', '»']
const DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
const JSON_POOL = ['{"', '{', '"', '",', '":', ' "', ',', '}', '"}', ' {"', '\n', ' ', '}\n', '"\n', ' }']
/** Ключи-соседи для позиции ключа JSON: под маской видно, что порядок ключей задан схемой. */
const KEY_POOL = ['Name', 'id', 'type', 'value', 'data', 'text', 'result', 'answer', 'name', 'age', 'city', 'location', 'query']
/** Продолжения идентификатора внутри JSON (имя функции и т. п.). */
const IDENT_POOL = ['_forecast', '_info', '_data', '_temperature', 'Weather', '_city', '_current', '_search']

const poolCache = new WeakMap<Tokenizer, Pools>()

/** Только строки, которые кодируются ровно одним токеном (так кандидат честно показывает настоящий id). */
function singleTokens(tokenizer: Tokenizer, list: readonly string[]): Token[] {
  const out: Token[] = []
  const seen = new Set<number>()
  for (const s of list) {
    const enc = tokenizer.encode(s, { allowSpecial: true })
    if (enc.length !== 1 || seen.has(enc[0].id)) continue
    seen.add(enc[0].id)
    out.push(tokenizer.token(enc[0].id))
  }
  return out
}

/** Пулы альтернатив для токенизатора (считаются один раз). */
export function candidatePools(tokenizer: Tokenizer): Pools {
  let pools = poolCache.get(tokenizer)
  if (!pools) {
    pools = {
      wordRu: singleTokens(tokenizer, WORD_RU),
      wordEn: singleTokens(tokenizer, WORD_EN),
      suffixRu: singleTokens(tokenizer, SUFFIX_RU),
      suffixEn: singleTokens(tokenizer, SUFFIX_EN),
      startsRu: singleTokens(tokenizer, STARTS_RU),
      startsEn: singleTokens(tokenizer, STARTS_EN),
      punct: singleTokens(tokenizer, PUNCT),
      digits: singleTokens(tokenizer, DIGITS),
      json: singleTokens(tokenizer, JSON_POOL),
      keys: singleTokens(tokenizer, KEY_POOL),
      ident: singleTokens(tokenizer, IDENT_POOL),
    }
    poolCache.set(tokenizer, pools)
  }
  return pools
}

/** Первый токен строки (для ручных альтернатив: «Конечно» → «К»). */
export function firstToken(tokenizer: Tokenizer, text: string): Token | null {
  const enc = tokenizer.encode(text, { allowSpecial: true })
  return enc.length > 0 ? tokenizer.token(enc[0].id) : null
}

const LETTER_START = /^\p{L}/u
const LETTER_END = /\p{L}$/u
const LATIN = /[A-Za-z]/

/** Класс выбранного токена по его тексту и предыдущему токену. */
export function classifyToken(tok: Token, prev: Token | undefined, inJson: boolean): TokenClass {
  if (tok.special) return 'special'
  const text = tok.text
  if (inJson && (/[{}":,]/.test(text) || /^\s+$/u.test(text))) return 'jsonStructural'
  if (/^\s+$/u.test(text)) return 'whitespace'
  if (/^\s?\d/u.test(text)) return 'digit'
  if (/^ \p{L}/u.test(text)) return 'wordStart'
  if (LETTER_START.test(text)) {
    return prev && !prev.special && LETTER_END.test(prev.text) ? 'continuation' : 'wordStartNoSpace'
  }
  return 'punct'
}

/** Начальный разрыв логитов между выбранным токеном и лучшей автоматической альтернативой. */
function initialGap(cls: TokenClass, confidence: 'high' | 'medium' | 'low' | undefined, rng: Rng): number {
  if (confidence === 'high') return 3.5 + 2.5 * rng()
  if (confidence === 'low') return 0.15 + 0.6 * rng()
  switch (cls) {
    case 'special':
    case 'continuation':
    case 'jsonStructural':
      return 3 + 2.5 * rng()
    case 'digit':
      return 1.5 + 2.5 * rng()
    case 'wordStart':
      return 0.4 + 1.6 * rng()
    case 'wordStartNoSpace':
      return 0.6 + 1.8 * rng()
    default:
      return 0.8 + 2.2 * rng()
  }
}

export interface BuildCandidatesInput {
  tokenizer: Tokenizer
  chosen: Token
  prev?: Token
  /** Следующий токен ответа (чтобы узнать позицию ключа JSON: перед ним «"», после него «":»). */
  next?: Token
  /** Внутри JSON (тело structured output или аргументы вызова): разметка берётся из JSON-пула. */
  inJson: boolean
  curated?: readonly AltSpec[]
  confidence?: 'high' | 'medium' | 'low'
  /** Слова из ответа и промпта (токены с ведущим пробелом): правдоподобные замены в начале слова. */
  context: readonly Token[]
  /** Язык ответа: из него берутся частые слова и начала слов (латинский токен в русском ответе получает русские замены). */
  lang: 'ru' | 'en'
  rng: Rng
}

/**
 * Выбранный токен + 4–7 альтернатив (без дублей id), логиты синтетические. Порядок: выбранный, ручные,
 * автоматические; сортирует и проверяет маску вызывающий.
 */
export function buildCandidates(input: BuildCandidatesInput): RawCandidate[] {
  const { tokenizer, chosen, rng } = input
  const pools = candidatePools(tokenizer)
  const cls = classifyToken(chosen, input.prev, input.inJson)
  const latin = LATIN.test(chosen.text) && !chosen.special
  // Ключ JSON: перед ним открывающая кавычка, после него закрывающая («":», «",»).
  const keyContext =
    input.inJson && LETTER_START.test(chosen.text) && !!input.prev?.text.endsWith('"') && !!input.next?.text.startsWith('"')
  const zc = 19 + 5 * rng()
  // Внутри JSON модель почти всегда уверена (копирует значения из запроса, разметка однозначна).
  let gap = initialGap(cls, input.confidence ?? (input.inJson ? 'high' : undefined), rng)
  const step = () => 0.35 + 1.2 * rng()

  const out: RawCandidate[] = [{ token: chosen, logit: zc }]
  const used = new Set<number>([chosen.id])

  let lowestCurated = 0
  for (const alt of input.curated ?? []) {
    const tok = firstToken(tokenizer, alt.text)
    if (!tok || used.has(tok.id)) continue
    used.add(tok.id)
    let logit: number
    if (alt.delta !== undefined) {
      logit = zc + alt.delta
      lowestCurated = Math.min(lowestCurated, alt.delta)
    } else {
      logit = zc - gap
      lowestCurated = Math.min(lowestCurated, -gap)
      gap += step()
    }
    const c: RawCandidate = { token: tok, logit }
    if (alt.preview) c.preview = alt.preview
    out.push(c)
  }

  // Автоматические альтернативы: ниже ручных (те осмысленнее), всего 5–8 кандидатов вместе с выбранным.
  const target = Math.max(out.length, 5 + Math.floor(rng() * 4))
  if (out.length > 1) gap = Math.max(gap, -lowestCurated + 0.3 + 0.4 * rng())

  const ru = input.lang === 'ru'
  let main: Token[]
  let extra: Token[] = []
  if (keyContext) {
    // начала самого ключа («n», «na», «nam» для «name») под маской разрешены, другие ключи нет
    const prefixes: string[] = []
    for (let i = 1; i < chosen.text.length; i++) prefixes.push(chosen.text.slice(0, i))
    main = [...singleTokens(tokenizer, prefixes), ...pools.keys]
  } else if (input.inJson && !chosen.special && (chosen.text.startsWith('_') || (latin && cls !== 'jsonStructural'))) {
    // идентификаторы внутри JSON: имя функции, латинские значения
    main = chosen.text.startsWith('_') || cls === 'continuation' ? pools.ident : pools.keys
  } else switch (cls) {
    case 'continuation':
      main = latin ? pools.suffixEn : pools.suffixRu
      break
    case 'wordStart':
      main = ru ? pools.wordRu : pools.wordEn
      extra = input.context.slice()
      break
    case 'wordStartNoSpace':
      main = ru ? pools.startsRu : pools.startsEn
      break
    case 'digit':
      main = [...pools.digits, ...pools.punct.slice(0, 2)]
      break
    case 'jsonStructural':
      main = pools.json
      break
    case 'special':
      main = [...pools.punct, ...pools.startsRu]
      break
    default:
      main = pools.punct
  }

  // Случайная стартовая позиция и обход по кругу: берём первый ещё не использованный токен списка.
  const tryPick = (list: readonly Token[]): boolean => {
    if (list.length === 0) return false
    const from = Math.floor(rng() * list.length)
    for (let j = 0; j < list.length; j++) {
      const tok = list[(from + j) % list.length]
      if (used.has(tok.id)) continue
      used.add(tok.id)
      out.push({ token: tok, logit: zc - gap })
      gap += step()
      return true
    }
    return false
  }

  let guard = 0
  while (out.length < target && guard++ < 40) {
    const useExtra = extra.length > 0 && rng() < 0.5
    if (useExtra && tryPick(extra)) continue
    if (tryPick(main)) continue
    // пул исчерпан: добираем частыми словами
    if (!tryPick(ru ? pools.wordRu : pools.wordEn)) tryPick(pools.punct)
  }
  return out
}

/* ============================================================================
 * Ручные альтернативы для служебных токенов (общие для всех сценариев)
 * ========================================================================== */

/** Альтернативы на месте тега или стоп-токена: что модель могла сделать вместо него. */
export function specialAlts(tok: Token, ctx: { nextText: string; jsonBody: boolean }): AltSpec[] | undefined {
  const head = ctx.nextText.replace(/^\s+/, '').slice(0, 24)
  switch (tok.text) {
    case '<think>':
      return [
        ...(head ? [{ text: head, preview: 'ответ сразу, без рассуждения', delta: -4.2 }] : []),
        { text: 'Хорошо', preview: 'Хорошо, … (рассуждение без тега <think>)', delta: -5.0 },
        { text: '\n', delta: -5.6 },
      ]
    case '</think>':
      return [
        { text: '\n', preview: 'рассуждение продолжилось бы с новой строки', delta: -1.6 },
        { text: ' Итак', preview: 'Итак, … (ещё одна мысль)', delta: -2.2 },
        { text: ' Ещё', preview: 'Ещё стоит учесть…', delta: -2.7 },
      ]
    case '<|im_end|>':
      return ctx.jsonBody
        ? [
            { text: '\n', delta: -1.5 },
            { text: ' ', delta: -2.1 },
            { text: ',', delta: -3.0 },
          ]
        : [
            { text: '\n\n', preview: 'ответ продолжился бы новым абзацем', delta: -1.8 },
            { text: ' Если', preview: 'Если хотите, расскажу подробнее.', delta: -2.4 },
            { text: '\n', delta: -3.0 },
          ]
    case '</tool_call>':
      return [
        { text: '\n', delta: -3.2 },
        { text: ' ', delta: -4.1 },
      ]
    default:
      return undefined
  }
}
