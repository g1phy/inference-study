/**
 * Формат компактного словаря byte-level BPE, который пишет scripts/build-tokenizer.mjs
 * в public/tokenizer/qwen3.json.
 */

/** Added token из tokenizer.json с флагом special (true = выкидывается при skip_special_tokens). */
export interface AddedTokenSpec {
  id: number
  content: string
  special: boolean
}

/** Компактный словарь: база из 256 байтовых символов + merges парами id (результат склейки r = 256 + r). */
export interface BpeAsset {
  format: 'bpe-merges-v1'
  name: string
  source?: string
  /** BPE-база + added tokens (Qwen3: 151 643 + 26 = 151 669). */
  vocabSize: number
  /** Нормализация перед пре-токенизацией (Qwen3: NFC). */
  normalizer: 'NFC' | null
  /** Регулярка пре-токенизации, переписанная под JS. */
  pattern: string
  patternFlags: string
  /** Исходная регулярка из tokenizer.json (синтаксис Oniguruma), для справки. */
  patternOriginal?: string
  /** 256 символов byte-level алфавита в порядке id 0..255. */
  base: string
  added: AddedTokenSpec[]
  /** Плоский массив [l0, r0, l1, r1, ...]: склейка ранга r соединяет id l_r и r_r в токен 256 + r. */
  merges: number[]
}

/** Проверяет форму JSON словаря и возвращает его как BpeAsset (бросает Error с описанием по-русски). */
export function parseBpeAsset(json: unknown): BpeAsset {
  const a = json as Partial<BpeAsset> | null
  if (!a || typeof a !== 'object') throw new Error('Словарь токенизатора: ожидался JSON-объект')
  if (a.format !== 'bpe-merges-v1') throw new Error(`Словарь токенизатора: неизвестный формат ${String(a.format)}`)
  if (typeof a.base !== 'string' || [...a.base].length !== 256) throw new Error('Словарь токенизатора: base должен содержать 256 символов')
  if (!Array.isArray(a.merges) || a.merges.length % 2 !== 0) throw new Error('Словарь токенизатора: merges должен быть массивом пар id')
  if (!Array.isArray(a.added)) throw new Error('Словарь токенизатора: нет added')
  if (typeof a.pattern !== 'string') throw new Error('Словарь токенизатора: нет pattern')
  const expected = 256 + a.merges.length / 2 + a.added.length
  if (a.vocabSize !== expected) throw new Error(`Словарь токенизатора: vocabSize ${a.vocabSize} не сходится с ${expected}`)
  return a as BpeAsset
}
