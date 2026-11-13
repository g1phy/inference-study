"""
Состязательные эталоны для токенизатора и детокенизатора Qwen3 (независимая проверка модулей A1).
Всё снимается с настоящих реализаций HF `tokenizers` (Rust):
  * encode: 250+ строк — случайные смеси Unicode, длинные серии пробелов, цифры, ZWJ-эмодзи, \\r\\n,
    литералы спецтокенов вплотную к тексту, NFC-меняющиеся строки, буквы/цифры Unicode 17, управляющие
    символы, «фузз» по алфавиту, который различает ветки регулярки пре-токенизации;
  * decode: случайные последовательности id (в т. ч. неполный UTF-8 и спецтокены) с skip_special и без;
  * stream: tokenizers.decoders.DecodeStream пошагово (эталон инкрементальной детокенизации).
Выход: fixtures/qwen3-adversarial.json (читает qwen3.adversarial.test.ts и detok.adversarial.test.ts).

Запуск (Python с -I, venv во /tmp):
  /tmp/inferstudy_venv/bin/python -I src/engine/tokenizer/fixtures/gen_adversarial.py <корень проекта>
"""
import json
import os
import random
import sys

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import DecodeStream
from tokenizers.pre_tokenizers import Split

ROOT = sys.argv[1]
VENDOR = os.path.join(ROOT, "research/vendor/qwen3")
OUT = os.path.join(ROOT, "src/engine/tokenizer/fixtures/qwen3-adversarial.json")

tok = Tokenizer.from_file(os.path.join(VENDOR, "tokenizer.json"))
cfg = json.load(open(os.path.join(VENDOR, "tokenizer_config.json"), encoding="utf-8"))
pattern = json.load(open(os.path.join(VENDOR, "tokenizer.json"), encoding="utf-8"))["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
split = Split(Regex(pattern), behavior="isolated", invert=False)
ADDED = [t["content"] for t in cfg["added_tokens_decoder"].values()]
SPECIAL = [t["content"] for t in cfg["added_tokens_decoder"].values() if t["special"]]
NON_SPECIAL = [t["content"] for t in cfg["added_tokens_decoder"].values() if not t["special"]]

rnd = random.Random(20261112)


def utf16_index(s):
    out = [0]
    for ch in s:
        out.append(out[-1] + (2 if ord(ch) > 0xFFFF else 1))
    return out


# ---------------------------------------------------------------------------------------------
# Строки
# ---------------------------------------------------------------------------------------------
POOLS = [
    (0x20, 0x7E, 30),  # ASCII
    (0xA0, 0xFF, 4),  # Latin-1
    (0x100, 0x24F, 3),  # Latin Ext-A/B (ſ, Ŀ, ŀ ...)
    (0x250, 0x2FF, 1),  # IPA, modifier letters (ʼ)
    (0x300, 0x36F, 3),  # combining diacriticals
    (0x370, 0x3FF, 2),  # Greek (U+037E -> ';' под NFC)
    (0x400, 0x4FF, 20),  # Cyrillic
    (0x530, 0x58F, 1),  # Armenian
    (0x590, 0x5FF, 1),  # Hebrew
    (0x600, 0x6FF, 2),  # Arabic (+ арабские цифры)
    (0x900, 0x97F, 2),  # Devanagari (U+0958 разлагается под NFC)
    (0xE00, 0xE7F, 1),  # Thai
    (0x1100, 0x11FF, 1),  # Hangul Jamo (собираются NFC)
    (0x2000, 0x206F, 3),  # general punctuation, пробелы U+2000-200A, ZWSP/ZWJ, U+2028/2029
    (0x2150, 0x218F, 1),  # number forms (Ⅻ — Nl)
    (0x2460, 0x24FF, 1),  # enclosed alnum (① — No)
    (0x2600, 0x27BF, 2),  # misc symbols, dingbats
    (0x3000, 0x30FF, 2),  # CJK punct, kana
    (0x4E00, 0x9FFF, 4),  # CJK
    (0xAC00, 0xD7A3, 2),  # Hangul syllables
    (0xF900, 0xFAFF, 1),  # CJK compatibility (меняются NFC)
    (0xFF00, 0xFFEF, 2),  # fullwidth
    (0x1D400, 0x1D7FF, 1),  # math alnum (L / Nd, астральные)
    (0x1F300, 0x1FAFF, 6),  # emoji
    (0x323B0, 0x3347F, 1),  # CJK Ext J (Unicode 17: в JS \\p{L}, в Oniguruma tokenizers 0.23.3 — нет)
]
POOL_W = [w for _, _, w in POOLS]
SPACES = ["\u0085", "\u00a0", "\u1680", "\u2000", "\u2002", "\u2009", "\u200a", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000", "\ufeff", "\u180e", "\u200b", "\u200d", "\t", "\n", "\r", "\x0b", "\x0c", " "]


def rand_cp():
    lo, hi, _ = rnd.choices(POOLS, weights=POOL_W)[0]
    c = rnd.randint(lo, hi)
    if 0xD800 <= c <= 0xDFFF:
        c = 0x41
    return chr(c)


def random_mix(n):
    out = []
    for _ in range(n):
        r = rnd.random()
        if r < 0.12:
            out.append(rnd.choice(SPACES))
        elif r < 0.2:
            out.append(rnd.choice([" the", " и", " 42", "'s", "'LL", "...", "\r\n", "  ", "→"]))
        else:
            out.append(rand_cp())
    return "".join(out)


# Алфавит, который бьёт по всем веткам регулярки Qwen2: буквы, цифры, апостроф+s/t, пунктуация,
# \r \n, обычные и «экзотические» пробелы (U+0085 и U+3000 — пробелы в Oniguruma; U+FEFF — нет).
FUZZ = [" ", " ", "\t", "\n", "\r", "a", "S", "я", "1", "٣", "!", ".", "'", "s", "t", "ll", "\u00a0", "\u0085", "\u3000", "\ufeff", "\u2028", "\u200b", "é", "\u0301", "🙂", "<"]

strings = []
cats = []


def add(cat, s):
    strings.append(s)
    cats.append(cat)


for _ in range(70):
    add("mix", random_mix(rnd.randint(3, 60)))
for _ in range(45):
    add("fuzz", "".join(rnd.choice(FUZZ) for _ in range(rnd.randint(1, 24))))

# Длинные серии пробелов
for n in [1, 2, 3, 7, 16, 33, 64, 100, 257]:
    add("ws", " " * n + "x")
for s in [
    " " * 40,
    "\t" * 9 + "def",
    "a" + " " * 12 + "b",
    "a" + " " * 12 + "1",
    "a" + " " * 12 + "!",
    "a" + " " * 12 + "\n",
    "x" + " \t" * 10 + "y",
    " \n" * 8,
    "\n" * 20 + "end",
    "\u3000" * 5 + "全角",
    "\u00a0" * 6 + "nbsp",
    "  \u00a0  word",
    "end" + " " * 30,
    " " * 5 + "\t" * 5 + "\n" * 5 + "z",
    "\u2003\u2003\u2003em",
    "a  \u0085  b",
]:
    add("ws", s)

# Цифры
for s in [
    "1234567890" * 3,
    "3.14159265358979323846",
    "-0.0001e-10",
    "1,234,567.89",
    "abc123def456",
    "2025-11-12T15:34:53.123Z",
    "+7 (912) 345-67-89",
    "٠١٢٣٤٥٦٧٨٩",
    "०१२३४५",
    "１２３４５",
    "x² + y³ = z⁴",
    "½ ⅓ ¾ ⅞",
    "Ⅰ Ⅱ Ⅻ ⅿ",
    "①②③⑩",
    "𝟎𝟏𝟐𝟗",
    "0x1F600 0b1010 0o777",
    "v1.2.3-rc.4+build.5",
    " 1 2  3   4",
    "99 bottles",
    "\U00011DE0\U00011DE1\U00016FF4",  # цифры Unicode 17
]:
    add("digits", s)

# ZWJ-эмодзи и прочие последовательности
for s in [
    "👨‍👩‍👧‍👦",
    "👩‍💻👨‍🔬🧑‍🚀",
    "🏳️‍🌈🏴‍☠️",
    "🇷🇺🇺🇸🇯🇵🇩🇪",
    "1️⃣#️⃣*️⃣",
    "👍🏻👍🏼👍🏽👍🏾👍🏿",
    "🧑🏽‍🚀 летит",
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 и 🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "❤️‍🔥❤️‍🩹",
    "🫠🫨🩷🫎🪿🫩🫆",
    "☺︎ ☺️ ✌︎ ✌️",
    "привет🙂мир🙂",
    "🙂🙂🙂🙂🙂🙂🙂🙂",
    "😀😃😄😁😆😅🤣😂",
    "🤖 ответ: 🫡",
    "👁️‍🗨️",
    "🧔🏻‍♀️🧑‍🤝‍🧑",
    "🐈‍⬛🐻‍❄️",
    " 🙂",
    "a🙂b🙂c",
    "\u200d\u200d",
    "🏻",
    "\ufe0f",
    "©️®️™️",
]:
    add("emoji", s)

# Windows-переводы строк и одиночные \r
for s in [
    "a\r\nb",
    "\r\n\r\n",
    " \r\n",
    "\r",
    "\r\r\n",
    "text\r\n  indented\r\n\tTab\r\n",
    "a\r \nb",
    "x.\r\n",
    "!!\r\n\r\nПривет",
    "\r\n" * 10,
    "\n\r\n\r",
    "def f():\r\n    return 1\r\n",
    "line1\rline2\rline3",
    "  \r\n  \r\n",
    "end.\r\n\r\n\r\n",
    "a\r\n\u00a0b",
    "):\r\n",
    "'s\r\n",
    "1\r\n2",
    "\r\n🙂\r\n",
]:
    add("crlf", s)

# Литералы added tokens вплотную к тексту
for t in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<tool_response>", "<|vision_pad|>"]:
    add("special", "abc" + t + "def")
    add("special", "привет" + t + "мир")
    add("special", " " + t + " ")
    add("special", t + t)
for s in [
    "1<|im_end|>2",
    "🙂<think>🙂",
    "<|im_start|",
    "im_start|>",
    "<<|im_end|>>",
    "<|<|im_end|>|>",
    "<think></think>",
    "</think><think>",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
    "<thi<think>nk>",
    "<THINK>",
    "<|IM_START|>",
    "< think>",
    "https://x.ru/<|endoftext|>/a",
    "\n<|im_end|>\n",
    "<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nТы полезный ассистент.<|im_end|>\n<|im_start|>user\n<|im_end|>",
    "  <think>\n\n</think>\n\n",
    "<tool_response>\n{\"t\": 1}\n</tool_response>",
    "<|fim_prefix|>def<|fim_suffix|>\n<|fim_middle|>",
    "<|endoftext|><|endoftext|><|endoftext|>",
    "x<|im_start|>",
    "<|im_start|>x",
    "'s<|im_end|>'s",
    "\r\n<|im_end|>\r\n",
    "<|im_end",
    "<|im_end|",
    "|>",
]:
    add("special", s)

# Сокращения и регистр (Oniguruma (?i) сворачивает ſ в s)
for s in [
    "'s 'S 'ſ 'ſſ",
    "I'M HE'LL THEY'RE WE'VE SHE'D IT'S",
    "'ll'll'll",
    "''s",
    "don't won't can't",
    "1's 2'd",
    "'sa 'tion 'lly",
    "ʼs ’s ‘s ＇s",
    "'K 'ß 'İ",
    "'Re 'rE 'Ve 'vE 'Ll 'lL",
    "rock'n'roll",
    "'",
    "''''",
    "' s",
    "'\ns",
]:
    add("contraction", s)

# NFC меняет строку (смещения в тестах для них не сравниваются, только id и decode)
for s in [
    "Cafe\u0301 и e\u0301\u0301",
    "A\u030a \u212b \u2126 \u212a",
    "\uf900\uf901\ufa10",
    "\u1100\u1161\u11a8 \u1100\u1161",
    "\u0344 \u0340 \u0341",
    "\u0958\u0959 \u09dc",
    "\u2000\u2001x",
    "a\u037eb \u0387",
    "a\u0323\u0302 a\u0302\u0323",
    "\u1e9b\u0323",
    "\u2329x\u232a",
    "й из \u0438\u0306",
    "\u0415\u0308\u043b\u043a\u0430",
    "\u00c5\u212b",
    "\u0301начало",
]:
    add("nfc", s)

# Unicode 17: буквы, которые JS (ICU 78) считает \\p{L}, а Oniguruma в tokenizers 0.23.3 — нет
for s in [
    "\u088f\u0c5c\u0cdc",
    "\ua7ce\ua7cf\ua7d2",
    "\U00010940\U00010941\U00010942",
    "\U000323B0\U000323B1 \U000323B2",
    "x\U00011DB0y",
    " \U00016EA0\U00016EA1",
]:
    add("unicode17", s)

# Управляющие символы
for s in ["\x00", "\x01abc", "\x7f", "\x1b[31mred\x1b[0m", "a\x0bb", "a\x0cb", "\x85", "\x08\x08", "\x00\x00 \x00", "tab\x09\x09end"]:
    add("control", s)

# Код
for s in [
    "for (let i = 0; i < n; i++) {\n\tsum += a[i];\n}",
    "SELECT * FROM t WHERE id = 42;",
    "<div class=\"a\">Привет</div>",
    "regex: /^[a-z]+\\d{2,}$/gi",
    "$ curl -s https://api.example.com/v1/chat/completions | jq '.choices[0]'",
    '{"a":[1,2,{"b":null}],"c":"\\u0416"}',
    "x = {'k': [1, 2.5, -3e4]}  # comment",
    "    if x:\n        pass\n    else:\n        raise",
    "a&&b||c!=d<=e>=f",
    "@decorator\ndef f(*args, **kwargs): ...",
]:
    add("code", s)

# Длинные тексты
add("long", ("Модель Qwen3-8B: 36 слоёв, 32 головы, KV 144 КиБ/токен. " * 12).strip())
add("long", ("The quick brown fox jumps over the lazy dog. " * 20) + "\n")
add("long", "".join(random_mix(40) + "\n" for _ in range(6)))
add("long", "ab" * 600)
add("long", "Ё" * 333 + "ё" * 333)

assert len(strings) >= 250, len(strings)


def has_added(s):
    return any(t in s for t in ADDED)


encode_cases = []
for cat, s in zip(cats, strings):
    tok.encode_special_tokens = False
    enc = tok.encode(s, add_special_tokens=False)
    rec = {"cat": cat, "text": s, "ids": enc.ids}
    if has_added(s):
        tok.encode_special_tokens = True
        rec["splitIds"] = tok.encode(s, add_special_tokens=False).ids
        tok.encode_special_tokens = False
    norm = tok.normalizer.normalize_str(s)
    if norm == s:
        idx = utf16_index(s)
        flat = []
        for a, b in enc.offsets:
            flat += [idx[a], idx[b]]
        rec["offsets"] = flat
    rec["decoded"] = tok.decode(enc.ids, skip_special_tokens=False)
    rec["decodedSkip"] = tok.decode(enc.ids, skip_special_tokens=True)
    if not has_added(s):
        rec["pretokens"] = [p for p, _ in split.pre_tokenize_str(norm)]
    encode_cases.append(rec)

# ---------------------------------------------------------------------------------------------
# decode: случайные последовательности id
# ---------------------------------------------------------------------------------------------
vocab = tok.get_vocab()
inv = {v: k for k, v in vocab.items()}
BYTE_DECODER = {}
bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
for b, c in zip(bs, cs):
    BYTE_DECODER[chr(c)] = b


def token_bytes(i):
    s = inv[i]
    if i >= 151643:
        return s.encode("utf-8")
    return bytes(BYTE_DECODER[ch] for ch in s)


partial = []
for i in range(151643):
    try:
        token_bytes(i).decode("utf-8")
    except UnicodeDecodeError:
        partial.append(i)

decode_cases = []
for k in range(80):
    seq = []
    for _ in range(rnd.randint(1, 14)):
        r = rnd.random()
        if r < 0.4:
            seq.append(rnd.choice(partial))
        elif r < 0.55:
            seq.append(rnd.randint(0, 255))
        elif r < 0.7:
            seq.append(rnd.randint(151643, 151668))
        else:
            seq.append(rnd.randint(0, 151642))
    decode_cases.append(
        {
            "ids": seq,
            "decoded": tok.decode(seq, skip_special_tokens=False),
            "decodedSkip": tok.decode(seq, skip_special_tokens=True),
        }
    )

# ---------------------------------------------------------------------------------------------
# stream: DecodeStream пошагово (эталон для incrementalDetokenize)
# ---------------------------------------------------------------------------------------------
STREAM_TEXTS = [
    "<think>\nПользователь спрашивает 2+2. Это 4.\n</think>\n\n2 + 2 = 4.<|im_end|>",
    " Ответ: 4 🫠 готово",
    "👨‍👩‍👧‍👦 семья, 🇷🇺 флаг, 🧑🏽‍🚀 космонавт<|im_end|>",
    "Ѣѣ Ꙗꙗ 𝟎𝟏 🫨🩷",
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Москва\"}}\n</tool_call><|im_end|>",
    "日本語のテキスト 中文 한국어 🙂",
    "a\r\nb\r\n\r\nend<|endoftext|>",
]
stream_cases = []
for s in STREAM_TEXTS:
    ids = tok.encode(s, add_special_tokens=False).ids
    for skip in (True, False):
        ds = DecodeStream(skip_special_tokens=skip)
        stream_cases.append({"ids": ids, "skip": skip, "steps": [ds.step(tok, i) for i in ids]})
# случайные последовательности id с неполным UTF-8 (без спецтокенов в середине)
for k in range(25):
    seq = []
    for _ in range(rnd.randint(2, 20)):
        r = rnd.random()
        seq.append(rnd.choice(partial) if r < 0.5 else rnd.randint(0, 151642))
    ds = DecodeStream(skip_special_tokens=True)
    steps = []
    ok = True
    for i in seq:
        try:
            steps.append(ds.step(tok, i))
        except Exception:
            ok = False
            break
    if ok:
        stream_cases.append({"ids": seq, "skip": True, "steps": steps})

# ---------------------------------------------------------------------------------------------
# o200k (эталон tiktoken) для сравнительного токенизатора: те же строки
# ---------------------------------------------------------------------------------------------
o200k_source = None
try:
    import tiktoken

    base = tiktoken.get_encoding("o200k_base")
    harm = tiktoken.get_encoding("o200k_harmony")
    for rec in encode_cases:
        rec["o200kBase"] = base.encode_ordinary(rec["text"])
        rec["o200kHarmony"] = harm.encode(rec["text"], allowed_special="all")
    o200k_source = "tiktoken %s" % tiktoken.__version__
except Exception as e:  # tiktoken необязателен
    print("o200k skipped:", e)

out = {
    "source": "tokenizers %s, research/vendor/qwen3/tokenizer.json; gen_adversarial.py, seed 20261112"
    % __import__("tokenizers").__version__,
    "o200kSource": o200k_source,
    "encode": encode_cases,
    "decode": decode_cases,
    "stream": stream_cases,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
print("encode", len(encode_cases), "decode", len(decode_cases), "stream", len(stream_cases), "->", OUT, os.path.getsize(OUT), "bytes")
