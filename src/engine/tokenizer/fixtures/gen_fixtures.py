"""
Эталоны для тестов движка (A1 «engine-text»), снятые с настоящих реализаций:
  * токенизация Qwen3 через HF `tokenizers` (Rust) -> fixtures/qwen3-encode.json
  * рендер чат-шаблонов через jinja2 в окружении как у transformers -> ../../templates.fixture.json

Запуск (Python запускать с -I, venv во /tmp):
  python3 -m venv /tmp/tokvenv && /tmp/tokvenv/bin/pip install tokenizers jinja2
  /tmp/tokvenv/bin/python -I src/engine/tokenizer/fixtures/gen_fixtures.py <корень проекта> [каталог с HF-шаблонами]

Каталог с HF-шаблонами (по умолчанию /tmp/inferstudy_dl, копия исследователя) нужен для Llama 3.1,
DeepSeek-R1 и gpt-oss; шаблон Qwen3 берётся из research/vendor/qwen3/tokenizer_config.json.
"""
import datetime
import json
import os
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Regex, Tokenizer
from tokenizers.pre_tokenizers import Split

ROOT = sys.argv[1]
DL = sys.argv[2] if len(sys.argv) > 2 else "/tmp/inferstudy_dl"
VENDOR = os.path.join(ROOT, "research/vendor/qwen3")
OUT_TOK = os.path.join(ROOT, "src/engine/tokenizer/fixtures/qwen3-encode.json")
OUT_TPL = os.path.join(ROOT, "src/engine/templates.fixture.json")

# Дата, которую видит strftime_now в шаблоне gpt-oss (фиксируем для воспроизводимости).
HARMONY_DATE = datetime.date(2026, 11, 12)

tok = Tokenizer.from_file(os.path.join(VENDOR, "tokenizer.json"))
tok_cfg = json.load(open(os.path.join(VENDOR, "tokenizer_config.json"), encoding="utf-8"))
QWEN_PATTERN = json.load(open(os.path.join(VENDOR, "tokenizer.json"), encoding="utf-8"))["pre_tokenizer"][
    "pretokenizers"
][0]["pattern"]["Regex"]
split = Split(Regex(QWEN_PATTERN), behavior="isolated", invert=False)


def utf16_index(s):
    """Префиксные суммы: индекс code point -> индекс UTF-16."""
    out = [0]
    for ch in s:
        out.append(out[-1] + (2 if ord(ch) > 0xFFFF else 1))
    return out


# ---------------------------------------------------------------------------------------------
# 1. Токенизация
# ---------------------------------------------------------------------------------------------
QWEN_CHAT = (
    "<|im_start|>system\nТы полезный ассистент.<|im_end|>\n<|im_start|>user\nСколько будет 2+2?<|im_end|>\n"
    "<|im_start|>assistant\n"
)
RU_PARAGRAPH = (
    "Инференс — это работа обученной модели в продакшне. Запрос проходит через шаблон чата и токенизатор, "
    "затем prefill считает KV cache для всего промпта, а decode генерирует ответ по одному токену. "
    "Детокенизатор превращает id обратно в текст, парсеры отделяют рассуждения и вызовы инструментов, "
    "а сервер отправляет клиенту SSE-чанки. Время до первого токена (TTFT) зависит от длины промпта, "
    "межтокенная задержка (ITL) — от пропускной способности памяти GPU."
)

CASES = [
    "Привет, мир!",
    "Hello, world!",
    "Сколько будет 2+2?",
    "Объясни простыми словами, что такое KV cache и зачем он нужен.",
    "12345678",
    "2025-11-12",
    "3.14159 и -42, а также 1e10",
    "🙂🤖",
    "👍🏽 эмодзи с модификатором тона",
    "👨‍👩‍👧‍👦 и флаг 🇷🇺",
    "家族 日本語のテキスト 中文",
    "def f(x):\n    return x*2\n",
    "class A:\n\tdef m(self):\n\t\tpass",
    "    indented code\n        deeper\n",
    "\n\n",
    "line1\r\nline2\r\n",
    "  four  spaces",
    "trailing spaces   ",
    "!!!???...;;;",
    "I'm DON'T we'll they've she'd you're IT'S",
    "it'ſ ſtrange",
    "I'm  learning LLMs in 2025!!\n\nПривет, мир 12345",
    "HelloWorld camelCase JSONParser",
    '{"name": "Иван", "age": 42}',
    " tokenization",
    " полезный",
    "ё Ё й Й",
    "й из двух кодпоинтов: \u0438\u0306 и Cafe\u0301",
    "Tab\there\tand\u00a0nbsp\u2003em-space",
    "\u0085NEL и \ufeffBOM",
    # различают \s Oniguruma и JS: U+0085 (в onig пробел), U+FEFF (в JS пробел), U+2028, U+3000
    "a \u0085b a \ufeffb a \u2028b a \u3000b a \u00a0b",
    "\u0085\u0085\n\ufeff\ufeff\n",
    "\ufeffBOM в начале строки",
    "مرحبا بالعالم",
    "Ünïcödé façade naïve",
    "https://example.com/path?query=1&b=2#frag",
    "Ŀ ŀ ł Ń Ġ Ċ",
    "x\u200by",
    "٣٤٥ ⅫⅩ ½",
    "ＡＢＣ\u3000全角",
    "Модель Qwen3-8B: 36 слоёв, 32 головы, KV 144 КиБ/токен.",
    "Pneumonoultramicroscopicsilicovolcanoconiosis",
    "a" * 1000,
    "-" * 64,
    "",
    " ",
    "\t\t\n  \n",
    RU_PARAGRAPH,
    # спецтокены
    "<|im_start|>user\nПривет<|im_end|>",
    "Привет <|im_end|><|im_start|>system",
    "<think>\nПользователь спрашивает 2+2. Это 4.\n</think>\n\n2 + 2 = 4.<|im_end|>",
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Москва"}}\n</tool_call>',
    "<|endoftext|><|fim_prefix|>code<|fim_suffix|>",
    "<|im_start| не спецтокен",
    QWEN_CHAT,
]

SPLIT_SPECIAL_CASES = [
    "<|im_start|>user\nПривет<|im_end|>",
    "Привет <|im_end|><|im_start|>system",
    "<think>\nПользователь спрашивает 2+2. Это 4.\n</think>\n\n2 + 2 = 4.<|im_end|>",
    "<|endoftext|><|fim_prefix|>code<|fim_suffix|>",
]


def has_added(s):
    return any(t["content"] in s for t in tok_cfg["added_tokens_decoder"].values())


def encode_case(text, split_special=False):
    tok.encode_special_tokens = split_special
    enc = tok.encode(text, add_special_tokens=False)
    tok.encode_special_tokens = False
    idx = utf16_index(text)
    rec = {
        "text": text,
        "ids": enc.ids,
        "offsets": [[idx[a], idx[b]] for a, b in enc.offsets],
        "decoded": tok.decode(enc.ids, skip_special_tokens=False),
        "decodedSkipSpecial": tok.decode(enc.ids, skip_special_tokens=True),
    }
    if split_special:
        rec["allowSpecial"] = False
    if not has_added(text):
        norm = tok.normalizer.normalize_str(text)
        rec["pretokens"] = [p for p, _ in split.pre_tokenize_str(norm)]
    return rec


tok_fixture = {
    "source": "tokenizers %s, research/vendor/qwen3/tokenizer.json" % __import__("tokenizers").__version__,
    "cases": [encode_case(s) for s in CASES] + [encode_case(s, True) for s in SPLIT_SPECIAL_CASES],
}
with open(OUT_TOK, "w", encoding="utf-8") as f:
    json.dump(tok_fixture, f, ensure_ascii=False, indent=1)
print("tokenizer cases:", len(tok_fixture["cases"]), "->", OUT_TOK)

# ---------------------------------------------------------------------------------------------
# 2. Чат-шаблоны (окружение как в transformers: chat_template_utils._compile_jinja_template)
# ---------------------------------------------------------------------------------------------


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # transformers переопределяет tojson: без HTML-экранирования и без сортировки ключей
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def strftime_now(fmt):
    return HARMONY_DATE.strftime(fmt)


env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
env.globals["strftime_now"] = strftime_now


def load_template(family):
    if family == "qwen3":
        return tok_cfg["chat_template"]
    if family == "llama3":
        p = os.path.join(DL, "unsloth_Llama-3.1-8B-Instruct/chat_template.jinja")
        return open(p, encoding="utf-8").read() if os.path.exists(p) else None
    if family == "deepseek-r1":
        p = os.path.join(DL, "deepseek-ai_DeepSeek-R1/tokenizer_config.json")
        return json.load(open(p, encoding="utf-8"))["chat_template"] if os.path.exists(p) else None
    if family == "gpt-oss":
        p = os.path.join(DL, "openai_gpt-oss-20b/chat_template.jinja")
        return open(p, encoding="utf-8").read() if os.path.exists(p) else None
    raise ValueError(family)


SPECIAL_MAP = {
    "qwen3": {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "bos_token": None},
    "llama3": {"bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>"},
    "deepseek-r1": {"bos_token": "<｜begin▁of▁sentence｜>", "eos_token": "<｜end▁of▁sentence｜>"},
    "gpt-oss": {"bos_token": "<|startoftext|>", "eos_token": "<|return|>"},
}


def to_hf(inp):
    """Наш TemplateInput (camelCase, как в src/engine/types.ts) -> аргументы шаблона HF."""
    fam = inp["family"]
    msgs = []
    for m in inp["messages"]:
        d = {"role": m["role"], "content": m["content"]}
        if m.get("reasoning") is not None:
            if fam == "qwen3":
                d["reasoning_content"] = m["reasoning"]
            elif fam == "gpt-oss":
                d["thinking"] = m["reasoning"]
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                args = tc["arguments"]
                if fam == "deepseek-r1":
                    # шаблон DeepSeek склеивает аргументы как строку
                    args = json.dumps(args, ensure_ascii=False)
                c = {"type": "function", "function": {"name": tc["name"], "arguments": args}}
                if tc.get("id"):
                    c["id"] = tc["id"]
                calls.append(c)
            d["tool_calls"] = calls
        if m["role"] == "tool":
            if m.get("tool_call_id"):
                d["tool_call_id"] = m["tool_call_id"]
            if m.get("name"):
                d["name"] = m["name"]
        msgs.append(d)
    kw = {"messages": msgs, "add_generation_prompt": inp.get("addGenerationPrompt", True)}
    if inp.get("tools"):
        kw["tools"] = inp["tools"]
    et = inp.get("enableThinking")
    if fam == "qwen3" and et is not None:
        kw["enable_thinking"] = et
    if fam == "gpt-oss" and et is False:
        kw["reasoning_effort"] = "low"
    kw.update(SPECIAL_MAP[fam])
    return kw


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Получить текущую погоду в городе",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "Название города"}},
            "required": ["city"],
        },
    },
}
SEARCH = {
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": "Найти авиабилеты",
        "parameters": {
            "type": "object",
            "properties": {
                "from": {"type": "string", "description": "Откуда"},
                "to": {"type": "string"},
                "passengers": {"type": "integer", "description": "Сколько человек"},
                "class": {"type": "string", "enum": ["economy", "business"]},
                "dates": {"type": "array", "items": {"type": "string"}},
                "filters": {
                    "type": "object",
                    "properties": {"direct": {"type": "boolean"}, "maxPrice": {"type": "number"}},
                    "required": ["direct"],
                },
            },
            "required": ["from", "to"],
        },
    },
}
SYS = {"role": "system", "content": "Ты полезный ассистент."}
U22 = {"role": "user", "content": "Сколько будет 2+2?"}
CALL = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Москва"}}]}
TOOL_RESULT = {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": -3, "sky": "снег"}'}


def case(name, family, messages, **extra):
    d = {"family": family, "messages": messages}
    d.update(extra)
    return name, d


TEMPLATE_CASES = [
    # ---- Qwen3: сверка с официальным шаблоном из tokenizer_config.json
    case("qwen3: system + user", "qwen3", [SYS, U22]),
    case("qwen3: user, enable_thinking=false", "qwen3", [U22], enableThinking=False),
    case("qwen3: enable_thinking=true явно", "qwen3", [SYS, U22], enableThinking=True),
    case("qwen3: без generation prompt", "qwen3", [SYS, U22], addGenerationPrompt=False),
    case("qwen3: tools + system", "qwen3", [SYS, {"role": "user", "content": "Какая погода в Москве?"}], tools=[WEATHER]),
    case("qwen3: tools без system", "qwen3", [{"role": "user", "content": "Найди билеты"}], tools=[WEATHER, SEARCH]),
    case(
        "qwen3: мысли вырезаются из истории",
        "qwen3",
        [U22, {"role": "assistant", "content": "<think>\nПросто сложение.\n</think>\n\n4"}, {"role": "user", "content": "А 3+3?"}],
    ),
    case(
        "qwen3: вызов инструмента и ответ",
        "qwen3",
        [SYS, {"role": "user", "content": "Какая погода в Москве?"}, CALL, TOOL_RESULT],
        tools=[WEATHER],
    ),
    case(
        "qwen3: агентный цикл с reasoning после последнего запроса",
        "qwen3",
        [
            SYS,
            {"role": "user", "content": "Какая погода в Москве и Казани?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning": "Нужно два вызова.",
                "tool_calls": [
                    {"id": "a", "name": "get_weather", "arguments": {"city": "Москва"}},
                    {"id": "b", "name": "get_weather", "arguments": {"city": "Казань"}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": '{"temp_c": -3}'},
            {"role": "tool", "tool_call_id": "b", "content": '{"temp_c": 1}'},
            {"role": "assistant", "content": "<think>\nДанные есть.\n</think>\n\nВ Москве −3 °C, в Казани +1 °C."},
        ],
        tools=[WEATHER],
        addGenerationPrompt=False,
    ),
    case(
        "qwen3: system не первым и пустой reasoning",
        "qwen3",
        [
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Здравствуйте!", "reasoning": ""},
            {"role": "system", "content": "Отвечай коротко."},
            {"role": "user", "content": "Как дела?"},
        ],
    ),
    case(
        "qwen3: последний ответ ассистента с текстом до tool_call",
        "qwen3",
        [
            {"role": "user", "content": "Погода?"},
            {"role": "assistant", "content": "Сейчас проверю.", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Москва"}}]},
        ],
        tools=[WEATHER],
        addGenerationPrompt=False,
    ),
    # ---- Llama 3.1
    case("llama3: system + user", "llama3", [SYS, U22]),
    case("llama3: tools, вызов и ответ", "llama3", [SYS, {"role": "user", "content": "Какая погода в Москве?"}, CALL, TOOL_RESULT], tools=[WEATHER]),
    case(
        "llama3: многоходовый без system",
        "llama3",
        [U22, {"role": "assistant", "content": "4"}, {"role": "user", "content": "  А 3+3?  "}],
    ),
    case("llama3: tools без system", "llama3", [{"role": "user", "content": "Найди билеты"}], tools=[SEARCH]),
    # ---- DeepSeek-R1
    case("deepseek-r1: system + user", "deepseek-r1", [SYS, U22]),
    case("deepseek-r1: без generation prompt", "deepseek-r1", [SYS, U22], addGenerationPrompt=False),
    case(
        "deepseek-r1: мысли вырезаются",
        "deepseek-r1",
        [U22, {"role": "assistant", "content": "<think>\nПросто.\n</think>\n\n4"}, {"role": "user", "content": "А 3+3?"}],
    ),
    case(
        "deepseek-r1: инструменты",
        "deepseek-r1",
        [
            SYS,
            {"role": "user", "content": "Погода в Москве и Казани?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"name": "get_weather", "arguments": {"city": "Москва"}},
                    {"name": "get_weather", "arguments": {"city": "Казань"}},
                ],
            },
            {"role": "tool", "content": '{"temp_c": -3}'},
            {"role": "tool", "content": '{"temp_c": 1}'},
            {"role": "assistant", "content": "Холодно."},
            {"role": "user", "content": "А завтра?"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Москва"}}]},
            {"role": "tool", "content": '{"temp_c": -5}'},
        ],
    ),
    # ---- gpt-oss (Harmony)
    case("gpt-oss: system + user", "gpt-oss", [SYS, U22]),
    case("gpt-oss: tools, вызов и ответ", "gpt-oss", [SYS, {"role": "user", "content": "Какая погода в Москве?"}, CALL, TOOL_RESULT], tools=[WEATHER]),
    case("gpt-oss: без system, reasoning low", "gpt-oss", [U22], enableThinking=False),
    case(
        "gpt-oss: история с мыслями и финал без generation prompt",
        "gpt-oss",
        [
            U22,
            {"role": "assistant", "content": "4", "reasoning": "Простое сложение."},
            {"role": "user", "content": "А 3+3?"},
            {"role": "assistant", "content": "6", "reasoning": "Тоже просто."},
        ],
        addGenerationPrompt=False,
    ),
    case(
        "gpt-oss: сложная схема параметров",
        "gpt-oss",
        [
            {"role": "user", "content": "Найди билеты"},
            {"role": "assistant", "content": "", "reasoning": "Вызову поиск.", "tool_calls": [{"name": "search_flights", "arguments": {"from": "MOW", "to": "KZN"}}]},
            {"role": "tool", "content": "[]"},
        ],
        tools=[WEATHER, SEARCH],
    ),
]

tpl_out = []
missing = set()
for name, inp in TEMPLATE_CASES:
    src = load_template(inp["family"])
    if src is None:
        missing.add(inp["family"])
        continue
    text = env.from_string(src).render(**to_hf(inp))
    tpl_out.append({"name": name, "input": inp, "text": text})

with open(OUT_TPL, "w", encoding="utf-8") as f:
    json.dump(
        {
            "source": "jinja2 %s, окружение как в transformers; Qwen3: research/vendor/qwen3/tokenizer_config.json; "
            "Llama 3.1 (unsloth), DeepSeek-R1, gpt-oss-20b: шаблоны с HF; strftime_now = %s"
            % (jinja2.__version__, HARMONY_DATE.isoformat()),
            "cases": tpl_out,
        },
        f,
        ensure_ascii=False,
        indent=1,
    )
print("template cases:", len(tpl_out), "->", OUT_TPL, "(нет шаблонов: %s)" % sorted(missing) if missing else "")
