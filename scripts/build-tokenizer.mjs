#!/usr/bin/env node
/**
 * Сборка компактного словаря Qwen3 для браузера: public/tokenizer/qwen3.json.
 *
 * Вход (скачано с Hugging Face, Qwen/Qwen3-8B):
 *   research/vendor/qwen3/tokenizer.json         (модель BPE, merges, added tokens, regex)
 *   research/vendor/qwen3/tokenizer_config.json  (флаги special у added tokens для transformers)
 *
 * Формат выхода "bpe-merges-v1" (см. src/engine/tokenizer/asset.ts):
 *   base    — 256 символов byte-level алфавита в порядке id 0..255 (байт → символ как в GPT-2 bytes_to_unicode);
 *   merges  — плоский массив пар id [l0, r0, l1, r1, ...] в порядке ранга.
 *             Склейка ранга r порождает токен с id 256 + r (скрипт это проверяет), поэтому
 *             словарь восстанавливается без хранения строк: raw[256 + r] = raw[l] + raw[r];
 *   added   — added tokens с флагом special из HF;
 *   pattern — регулярка пре-токенизации, переписанная под JS (флаг u).
 *
 * Запуск: npm run build:tokenizer
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { gzipSync } from 'node:zlib'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const SRC_TOKENIZER = resolve(root, 'research/vendor/qwen3/tokenizer.json')
const SRC_CONFIG = resolve(root, 'research/vendor/qwen3/tokenizer_config.json')
const OUT = resolve(root, 'public/tokenizer/qwen3.json')

/** Регулярка Qwen2/Qwen3 ровно в том виде, как она лежит в tokenizer.json (синтаксис Oniguruma). */
const QWEN2_PATTERN =
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

/**
 * Классы Oniguruma из HF tokenizers (снимок scripts/data/onig-unicode-classes.json, генерирует
 * scripts/gen-onig-classes.py): \p{L}, \p{N} и \s как явные диапазоны code point.
 * \p{L}/\p{N} в JS зависят от версии Unicode в ICU браузера (Node 24 — Unicode 17), а Oniguruma в
 * tokenizers 0.23.3 знает меньше: CJK Ext J (U+323B0…) и ещё ~350 символов для неё не буквы. С \p{L}
 * строка "𳎳'sI" резалась иначе, чем в HF (найдено фузз-тестом, fixtures/gen_adversarial.py).
 */
const SRC_CLASSES = resolve(root, 'scripts/data/onig-unicode-classes.json')
const ONIG = JSON.parse(readFileSync(SRC_CLASSES, 'utf8'))
const hexCp = (c) => `\\u{${c.toString(16).toUpperCase()}}`
const cls = (ranges) => ranges.map(([a, b]) => (a === b ? hexCp(a) : `${hexCp(a)}-${hexCp(b)}`)).join('')
const L = cls(ONIG.L)
const N = cls(ONIG.N)

/**
 * Та же регулярка для JS (флаги gu). Отличия от исходника:
 * 1. (?i:'s|...) → явные альтернативы регистра. ſ (U+017F) добавлена к s: в Oniguruma
 *    (?i) сравнивает по Unicode case folding, а ſ сворачивается в s (других складок для
 *    s/t/re/ve/m/ll/d нет — проверено перебором всех code point через HF).
 * 2. \s и \S заменены явным классом: \s в Oniguruma (Unicode) = \t \n \v \f \r, U+0085 и
 *    Zs/Zl/Zp; в JS \s ещё включает U+FEFF и не включает U+0085.
 *    Источник: Oniguruma doc/RE, раздел «\s»; ECMA-262 WhiteSpace/LineTerminator; снимок ONIG.S.
 * 3. \p{L} и \p{N} заменены явными диапазонами Oniguruma (см. выше): регулярка не зависит
 *    от Unicode-таблиц браузера.
 */
const WS = cls(ONIG.S)
const JS_PATTERN =
  "'(?:[sSſ]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])" +
  `|[^\\r\\n${L}${N}]?[${L}]+` +
  `|[${N}]` +
  `| ?[^${WS}${L}${N}]+[\\r\\n]*` +
  `|[${WS}]*[\\r\\n]+` +
  `|[${WS}]+(?![^${WS}])` +
  `|[${WS}]+`

function fail(msg) {
  console.error(`build-tokenizer: ${msg}`)
  process.exit(1)
}

function assert(cond, msg) {
  if (!cond) fail(msg)
}

/** GPT-2 bytes_to_unicode: 188 «печатных» байтов в себя, остальные 68 в U+0100…U+0143. */
function bytesToUnicodeOrder() {
  const bs = []
  for (let b = 0x21; b <= 0x7e; b++) bs.push(b)
  for (let b = 0xa1; b <= 0xac; b++) bs.push(b)
  for (let b = 0xae; b <= 0xff; b++) bs.push(b)
  const cs = bs.slice()
  let n = 0
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b)
      cs.push(256 + n)
      n++
    }
  }
  // Порядок как в vocab.json GPT-2 / Qwen2: сначала печатные, потом U+0100…
  return cs.map((c) => String.fromCodePoint(c))
}

const tok = JSON.parse(readFileSync(SRC_TOKENIZER, 'utf8'))
const cfg = JSON.parse(readFileSync(SRC_CONFIG, 'utf8'))
const model = tok.model

// --- Проверяем, что runtime-реализация покрывает именно эту конфигурацию -----------------
assert(model.type === 'BPE', 'ожидалась модель BPE')
assert(model.byte_fallback === false, 'byte_fallback не поддерживается')
assert(model.ignore_merges === false, 'ignore_merges=true не поддерживается')
assert(!model.continuing_subword_prefix && !model.end_of_word_suffix, 'префиксы/суффиксы подслов не поддерживаются')
assert(model.dropout == null, 'dropout не поддерживается')
assert(tok.normalizer && tok.normalizer.type === 'NFC', 'ожидался normalizer NFC')
const pre = tok.pre_tokenizer
assert(pre.type === 'Sequence' && pre.pretokenizers.length === 2, 'ожидался Sequence из двух пре-токенизаторов')
const [split, bl] = pre.pretokenizers
assert(split.type === 'Split' && split.behavior === 'Isolated' && split.invert === false, 'ожидался Split Isolated')
assert(split.pattern.Regex === QWEN2_PATTERN, `регулярка отличается от Qwen2: ${split.pattern.Regex}`)
assert(bl.type === 'ByteLevel' && bl.use_regex === false && bl.add_prefix_space === false, 'ожидался ByteLevel без своей регулярки')
assert(tok.decoder.type === 'ByteLevel', 'ожидался ByteLevel decoder')

// --- База: 256 byte-level символов --------------------------------------------------------
const vocab = model.vocab
const inv = []
for (const [s, id] of Object.entries(vocab)) inv[id] = s
assert(inv.length === Object.keys(vocab).length, 'id словаря идут не подряд')
const base = bytesToUnicodeOrder()
for (let i = 0; i < 256; i++) assert(inv[i] === base[i], `id ${i}: ожидался байтовый символ ${base[i]}, а не ${inv[i]}`)

// --- Merges: пара → id 256 + rank ------------------------------------------------------
const merges = new Array(model.merges.length * 2)
model.merges.forEach((m, r) => {
  const [a, b] = Array.isArray(m) ? m : m.split(' ')
  const la = vocab[a]
  const rb = vocab[b]
  const out = vocab[a + b]
  assert(la !== undefined && rb !== undefined, `merge ${r}: части не в словаре`)
  assert(out === 256 + r, `merge ${r}: результат ${a + b} имеет id ${out}, ожидался ${256 + r}`)
  assert(la < out && rb < out, `merge ${r}: часть с id больше результата`)
  merges[2 * r] = la
  merges[2 * r + 1] = rb
})
assert(256 + model.merges.length === inv.length, 'каждый токен ≥ 256 должен порождаться ровно одной склейкой')

// Восстановление словаря из merges должно дать исходные строки.
const raw = base.slice()
for (let r = 0; r < model.merges.length; r++) raw.push(raw[merges[2 * r]] + raw[merges[2 * r + 1]])
for (let i = 0; i < inv.length; i++) assert(raw[i] === inv[i], `восстановление словаря расходится на id ${i}`)

// --- Added tokens ------------------------------------------------------------------------
const decoderFlags = cfg.added_tokens_decoder || {}
const added = tok.added_tokens.map((t) => {
  assert(!t.lstrip && !t.rstrip && !t.single_word, `${t.content}: lstrip/rstrip/single_word не поддерживаются`)
  assert(t.normalized === false, `${t.content}: normalized=true не поддерживается`)
  const c = decoderFlags[String(t.id)]
  assert(c && c.content === t.content && c.special === t.special, `${t.content}: флаг special расходится с tokenizer_config.json`)
  return { id: t.id, content: t.content, special: t.special }
})
added.sort((a, b) => a.id - b.id)
added.forEach((t, i) => assert(t.id === inv.length + i, 'added tokens идут не подряд после BPE-словаря'))

// Проверяем, что JS-регулярка компилируется, и снимок классов Oniguruma правдоподобен.
new RegExp(JS_PATTERN, 'gu')
{
  const inCls = (ranges, ch) => ranges.some(([a, b]) => ch.codePointAt(0) >= a && ch.codePointAt(0) <= b)
  for (const ch of ['a', 'Z', 'Я', 'ё', '中', 'ſ', 'ª']) assert(inCls(ONIG.L, ch), `снимок \\p{L} не содержит ${ch}`)
  for (const ch of ['0', '9', '٣', 'Ⅻ', '½']) assert(inCls(ONIG.N, ch), `снимок \\p{N} не содержит ${ch}`)
  for (const ch of [' ', '\t', '\n', '\r', '\u0085', ' ', '　', ' ']) assert(inCls(ONIG.S, ch), `снимок \\s не содержит U+${ch.codePointAt(0).toString(16)}`)
  for (const ch of ['﻿', '​', 'a']) assert(!inCls(ONIG.S, ch), `снимок \\s содержит лишний U+${ch.codePointAt(0).toString(16)}`)
  // Справка: насколько таблицы Oniguruma расходятся с \p{L}/\p{N} текущего движка JS.
  const onigLN = new Uint8Array(0x110000)
  for (const [a, b] of [...ONIG.L, ...ONIG.N]) onigLN.fill(1, a, b + 1)
  let onlyJs = 0
  let onlyOnig = 0
  const jsLN = /\p{L}|\p{N}/u
  for (let c = 0; c < 0x110000; c++) {
    if (c >= 0xd800 && c <= 0xdfff) continue
    const js = jsLN.test(String.fromCodePoint(c))
    if (js && !onigLN[c]) onlyJs++
    if (onigLN[c] && !js) onlyOnig++
  }
  console.log(`build-tokenizer: классы Oniguruma (${ONIG.source}) vs \\p{L}|\\p{N} в Node ${process.versions.unicode ?? '?'}: только JS ${onlyJs}, только Oniguruma ${onlyOnig}`)
}

const asset = {
  format: 'bpe-merges-v1',
  name: 'Qwen3',
  source:
    'Hugging Face Qwen/Qwen3-8B: tokenizer.json + tokenizer_config.json (research/vendor/qwen3), собрано scripts/build-tokenizer.mjs',
  vocabSize: inv.length + added.length,
  normalizer: 'NFC',
  pattern: JS_PATTERN,
  patternFlags: 'gu',
  patternOriginal: QWEN2_PATTERN,
  base: base.join(''),
  added,
  merges,
}

mkdirSync(dirname(OUT), { recursive: true })
const json = JSON.stringify(asset)
writeFileSync(OUT, json)
const rawBytes = Buffer.byteLength(json)
const gz = gzipSync(json, { level: 9 }).length
console.log(
  `build-tokenizer: ${OUT}\n  BPE ${inv.length} + added ${added.length} = ${asset.vocabSize} токенов, merges ${model.merges.length}\n` +
    `  размер ${(rawBytes / 1024).toFixed(0)} КиБ, gzip -9 ${(gz / 1024).toFixed(0)} КиБ`,
)
