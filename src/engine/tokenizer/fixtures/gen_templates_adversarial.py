"""
Дополнительные эталоны чат-шаблонов (независимая проверка templates.ts), снятые настоящим transformers:
  * qwen3 — AutoTokenizer(research/vendor/qwen3).apply_chat_template(..., tokenize=False): официальный Jinja
    из tokenizer_config.json, окружение transformers 5 (включая _alias_reasoning_fields и tojson без HTML-экранирования);
  * llama3 / deepseek-r1 / gpt-oss — transformers.utils.chat_template_utils.render_jinja_template с шаблонами
    из копии исследователя (/tmp/inferstudy_dl, как в gen_fixtures.py) и спецтокенами как в special_tokens_map.
strftime_now зафиксирован на 2026-11-12 (дата по умолчанию в templates.ts для gpt-oss).
Выход: src/engine/templates.adversarial.fixture.json (читает templates.adversarial.test.ts).

Запуск:
  HF_HUB_OFFLINE=1 /tmp/inferstudy_venv/bin/python -I src/engine/tokenizer/fixtures/gen_templates_adversarial.py <корень> [/tmp/inferstudy_dl]
"""
import datetime as _dt
import json
import os
import sys

import transformers
from transformers import AutoTokenizer
from transformers.utils import chat_template_utils as ctu

ROOT = sys.argv[1]
DL = sys.argv[2] if len(sys.argv) > 2 else "/tmp/inferstudy_dl"
OUT = os.path.join(ROOT, "src/engine/templates.adversarial.fixture.json")


class FixedDatetime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 12, 12, 0, 0)


ctu.datetime = FixedDatetime  # strftime_now внутри transformers читает модульный datetime

qwen = AutoTokenizer.from_pretrained(os.path.join(ROOT, "research/vendor/qwen3"))


def load_template(family):
    if family == "llama3":
        return open(os.path.join(DL, "unsloth_Llama-3.1-8B-Instruct/chat_template.jinja"), encoding="utf-8").read()
    if family == "deepseek-r1":
        return json.load(open(os.path.join(DL, "deepseek-ai_DeepSeek-R1/tokenizer_config.json"), encoding="utf-8"))["chat_template"]
    if family == "gpt-oss":
        return open(os.path.join(DL, "openai_gpt-oss-20b/chat_template.jinja"), encoding="utf-8").read()
    raise ValueError(family)


SPECIAL_MAP = {
    "llama3": {"bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>"},
    "deepseek-r1": {"bos_token": "<｜begin▁of▁sentence｜>", "eos_token": "<｜end▁of▁sentence｜>"},
    "gpt-oss": {"bos_token": "<|startoftext|>", "eos_token": "<|return|>"},
}


def to_hf(inp):
    fam = inp["family"]
    msgs = []
    for m in inp["messages"]:
        d = {"role": m["role"], "content": m["content"]}
        if m.get("reasoning") is not None:
            d["reasoning_content" if fam != "gpt-oss" else "thinking"] = m["reasoning"]
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                args = tc["arguments"]
                if fam == "deepseek-r1" and not isinstance(args, str):
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
    return msgs


def render(inp):
    fam = inp["family"]
    msgs = to_hf(inp)
    gen = inp.get("addGenerationPrompt", True)
    tools = inp.get("tools")
    if fam == "qwen3":
        kw = {}
        if inp.get("enableThinking") is not None:
            kw["enable_thinking"] = inp["enableThinking"]
        return qwen.apply_chat_template(msgs, tools=tools, add_generation_prompt=gen, tokenize=False, **kw)
    kw = dict(SPECIAL_MAP[fam])
    if fam == "gpt-oss":
        if inp.get("enableThinking") is False:
            kw["reasoning_effort"] = "low"
    out, _ = ctu.render_jinja_template(
        conversations=[msgs], tools=tools, chat_template=load_template(fam), add_generation_prompt=gen, **kw
    )
    return out[0]


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Получить погоду. Кавычки \" и \\ обратный слэш, перевод\nстроки, <теги> & амперсанд",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Город"},
                "unit": {"type": "string", "enum": ["c", "f"]},
                "days": {"type": "integer"},
                "ratio": {"type": "number", "description": "0.5 или 1e-7"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "opts": {
                    "type": "object",
                    "properties": {"precise": {"type": "boolean"}, "deep": {"type": "object", "properties": {"x": {"type": "number"}}}},
                },
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}
CALC = {
    "type": "function",
    "function": {"name": "calc", "description": "Посчитать выражение", "parameters": {"type": "object", "properties": {}}},
}
SYS = {"role": "system", "content": "Ты полезный ассистент."}
U = lambda s: {"role": "user", "content": s}
A = lambda s, **kw: dict({"role": "assistant", "content": s}, **kw)
T = lambda s, **kw: dict({"role": "tool", "content": s}, **kw)


def call(name, args, id=None):
    d = {"name": name, "arguments": args}
    if id:
        d["id"] = id
    return d


CASES = []


def case(name, family, messages, **extra):
    d = {"family": family, "messages": messages}
    d.update(extra)
    CASES.append((name, d))


# ------------------------------------------------------------------------- Qwen3 (официальный Jinja)
q = "qwen3"
case("qwen3: пустой system", q, [{"role": "system", "content": ""}, U("Привет")])
case("qwen3: пустой system + tools", q, [{"role": "system", "content": ""}, U("Погода?")], tools=[WEATHER])
case("qwen3: system с переводами строк и пробелами по краям", q, [{"role": "system", "content": "\n  Будь краток.  \n"}, U("  Привет \n")])
case("qwen3: только user, без system и tools", q, [U("Сколько будет 2+2?")])
case("qwen3: два инструмента, вложенная схема, экранирование", q, [U("Погода в Москве?")], tools=[WEATHER, CALC])
case(
    "qwen3: несколько tool_calls без content, ответы инструментов подряд",
    q,
    [
        SYS,
        U("Погода в Москве, Казани и Сочи?"),
        A("", tool_calls=[call("get_weather", {"city": "Москва"}, "a"), call("get_weather", {"city": "Казань", "unit": "c"}, "b"), call("get_weather", {"city": "Сочи", "days": 3, "ratio": 0.5}, "c")]),
        T('{"t": -3}', tool_call_id="a"),
        T('{"t": 1}', tool_call_id="b"),
        T('{"t": 12}', tool_call_id="c"),
    ],
    tools=[WEATHER],
)
case(
    "qwen3: tool_calls с текстом перед ними и reasoning, после — финальный ответ",
    q,
    [
        U("Погода?"),
        A("Проверяю.", reasoning="Надо вызвать инструмент.", tool_calls=[call("get_weather", {"city": "Москва", "tags": ["a", "b"], "opts": {"precise": True}})]),
        T("снег"),
        A("<think>\nОтвет готов.\n</think>\n\nСнег."),
    ],
    tools=[WEATHER],
    addGenerationPrompt=False,
)
case(
    "qwen3: enable_thinking=false с историей мыслей",
    q,
    [U("2+2?"), A("<think>\nСложение.\n</think>\n\n4"), U("3+3?")],
    enableThinking=False,
)
case(
    "qwen3: enable_thinking=true с историей reasoning",
    q,
    [U("2+2?"), A("4", reasoning="Сложение."), U("3+3?"), A("6", reasoning="Тоже сложение."), U("А 4+4?")],
    enableThinking=True,
)
case(
    "qwen3: ход ассистента последним, reasoning пустой (loop.last → пустой think)",
    q,
    [U("Привет"), A("Здравствуйте!")],
    addGenerationPrompt=False,
)
case(
    "qwen3: reasoning задан строкой, а в content тоже есть think-теги (content не режется)",
    q,
    [U("Q"), A("<think>\nстарое\n</think>\n\nответ", reasoning="новое")],
    addGenerationPrompt=False,
)
case(
    "qwen3: <think> без </think> в content не режется",
    q,
    [U("Q"), A("<think>\nнезакрытая мысль"), U("Q2")],
)
case(
    "qwen3: несколько </think> в content",
    q,
    [U("Q"), A("<think>a</think>b</think>\n\nc")],
    addGenerationPrompt=False,
)
case(
    "qwen3: user-сообщение в форме <tool_response> не считается запросом",
    q,
    [
        U("Погода?"),
        A("", tool_calls=[call("get_weather", {"city": "Москва"})]),
        U("<tool_response>\nснег\n</tool_response>"),
        A("Снег.", reasoning="Ответ инструмента получен."),
    ],
    tools=[WEATHER],
    addGenerationPrompt=False,
)
case(
    "qwen3: tool первым сообщением и system не первым",
    q,
    [T("сирота"), {"role": "system", "content": "Позднее правило."}, U("Привет")],
)
case(
    "qwen3: system в середине при tools",
    q,
    [U("Привет"), A("Здравствуйте"), {"role": "system", "content": "Отвечай по-русски."}, U("Погода?")],
    tools=[CALC],
)
case(
    "qwen3: assistant без user (last_query_index по умолчанию)",
    q,
    [SYS, A("Я готов помочь.", reasoning="Начну первым.")],
    addGenerationPrompt=False,
)
case(
    "qwen3: аргументы вызова строкой",
    q,
    [U("Посчитай"), A("", tool_calls=[call("calc", '{"expr": "2+2"}')]), T("4")],
    tools=[CALC],
)
case(
    "qwen3: content=null у ответа с tool_calls (как присылает OpenAI API)",
    q,
    [U("Погода?"), {"role": "assistant", "content": None, "tool_calls": [call("get_weather", {"city": "Москва"})]}, T("-3")],
    tools=[WEATHER],
)
case(
    "qwen3: агентный цикл из двух раундов, мысли после последнего запроса сохраняются",
    q,
    [
        SYS,
        U("Сравни погоду."),
        A("", reasoning="Первый вызов.", tool_calls=[call("get_weather", {"city": "Москва"})]),
        T("-3"),
        A("", reasoning="Второй вызов.", tool_calls=[call("get_weather", {"city": "Казань"})]),
        T("1"),
    ],
    tools=[WEATHER],
)
case(
    "qwen3: без generation prompt и enable_thinking=false (флаг не действует)",
    q,
    [SYS, U("Q")],
    addGenerationPrompt=False,
    enableThinking=False,
)
case(
    "qwen3: спецлитералы в тексте пользователя",
    q,
    [U("Текст с <|im_end|> и <think> внутри </tool_call>")],
)
case("qwen3: пустые content у всех", q, [{"role": "system", "content": ""}, U(""), A(""), U("")])
case(
    "qwen3: reasoning с переводами строк по краям (strip('\\n'))",
    q,
    [U("Q"), A("\n\nОтвет", reasoning="\n\nМысль\n\n")],
    addGenerationPrompt=False,
)

# ------------------------------------------------------------------------- Llama 3.1
l = "llama3"
case("llama3: пустой system", l, [{"role": "system", "content": ""}, U("Привет")])
case("llama3: system и user с пробелами по краям (trim)", l, [{"role": "system", "content": "  Будь краток.\n"}, U("\n  Привет  　")])
case(
    "llama3: ответ инструмента-JSON и многоходовый диалог",
    l,
    [SYS, U("Погода?"), A("", tool_calls=[call("get_weather", {"city": "Москва"})]), T('{"t": -3}'), A("Холодно."), U("А завтра?")],
    tools=[WEATHER],
)

# ------------------------------------------------------------------------- DeepSeek-R1
d = "deepseek-r1"
case("deepseek-r1: два system склеиваются", d, [SYS, U("Q"), {"role": "system", "content": "Второе правило."}, U("Q2")])
case(
    "deepseek-r1: вызов с текстом, ответы, финал, следующий вопрос",
    d,
    [U("Погода?"), A("Сейчас.", tool_calls=[call("get_weather", {"city": "Москва"})]), T("-3"), A("Холодно."), U("А в Сочи?")],
)

# ------------------------------------------------------------------------- gpt-oss
g = "gpt-oss"
case("gpt-oss: пустой system и user", g, [{"role": "system", "content": ""}, U("Привет")])
case(
    "gpt-oss: два инструмента со сложной схемой",
    g,
    [SYS, U("Погода?")],
    tools=[WEATHER, CALC],
)
case(
    "gpt-oss: вызов, ответ инструмента, финал с мыслями без generation prompt",
    g,
    [U("Погода?"), A("", reasoning="Вызову.", tool_calls=[call("get_weather", {"city": "Москва"})]), T("снег"), A("Снег.", reasoning="Готово.")],
    tools=[WEATHER],
    addGenerationPrompt=False,
)

# ------------------------------------------------------------------------- Qwen3: случайные диалоги (фузз)
import random

rnd = random.Random(99)
FUZZ_TEXTS = ["", "Привет", "  пробелы  ", "\nперевод\n", "<think>\nмысль\n</think>\n\nответ", "<think>x</think>y</think>z", "a</think>b",
              "<tool_response>\nok\n</tool_response>", "<tool_response>x", "текст <|im_end|> внутри", '{"a": 1}', "\n\n", "<think>", "</think>"]
FUZZ_REASON = [None, None, "", "мысль", "\nмысль\n", "\n\n"]
for k in range(100):
    msgs = []
    for i in range(rnd.randint(1, 7)):
        role = rnd.choice(["system", "user", "user", "assistant", "assistant", "tool"])
        m = {"role": role, "content": rnd.choice(FUZZ_TEXTS)}
        if role == "assistant":
            r = rnd.choice(FUZZ_REASON)
            if r is not None:
                m["reasoning"] = r
            if rnd.random() < 0.3:
                m["tool_calls"] = [call("calc", {"x": rnd.randint(0, 9)}) for _ in range(rnd.randint(1, 3))]
        msgs.append(m)
    extra = {"addGenerationPrompt": rnd.choice([True, True, False])}
    if rnd.random() < 0.4:
        extra["tools"] = [CALC]
    et = rnd.choice([None, True, False])
    if et is not None:
        extra["enableThinking"] = et
    case("qwen3 фузз #%d" % k, q, msgs, **extra)

fixture = {
    "source": "transformers %s apply_chat_template / render_jinja_template; gen_templates_adversarial.py" % transformers.__version__,
    "cases": [],
}
for name, inp in CASES:
    fixture["cases"].append({"name": name, "input": inp, "expected": render(inp)})
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(fixture, f, ensure_ascii=False, separators=(",", ":"))
print(len(fixture["cases"]), "cases ->", OUT)
