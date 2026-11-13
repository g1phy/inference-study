/**
 * Токенизаторы движка (без React). Хук useTokenizer импортируйте из '@/engine/tokenizer/useTokenizer'.
 */
export { ByteLevelBpeTokenizer, createBpeTokenizer } from './bpe'
export type { EncodeOptions, DecodeOptions } from './bpe'
export { parseBpeAsset } from './asset'
export type { BpeAsset, AddedTokenSpec } from './asset'
export {
  loadTokenizer,
  loadQwen3Tokenizer,
  getTokenizerState,
  getLoadedTokenizer,
  subscribeTokenizer,
  tokenizerUrl,
} from './loader'
export type { TokenizerState } from './loader'
export { loadO200kTokenizer, O200kTokenizer, O200K_HARMONY_LOGITS_WIDTH } from './o200k'
export type { O200kVariant } from './o200k'
export {
  BYTE_TO_CHAR,
  byteOfChar,
  bytesToRaw,
  rawToBytes,
  utf8Bytes,
  utf8DecodeLossy,
  utf8DecodeSoft,
  isCompleteUtf8,
} from './bytes'

/** Ширина логитов Qwen3 (config.vocab_size): 151 936, из них 267 пустых строк (notes-tokens §1.9). */
export const QWEN3_LOGITS_WIDTH = 151936
/** Стоп-токены генерации Qwen3 из generation_config.json: <|im_end|>, <|endoftext|> (notes-tokens §1.3). */
export const QWEN3_STOP_TOKEN_IDS: readonly number[] = [151645, 151643]
