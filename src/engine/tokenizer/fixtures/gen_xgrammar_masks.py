"""
Эталон масок structured output: XGrammar 0.2.9 (бэкенд по умолчанию в vLLM / SGLang) на всём словаре Qwen3.
Для каждой пары (схема, целевой JSON) идём по токенам цели и на каждом шаге снимаем bitmask
GrammarMatcher.fill_next_token_bitmask (compile_json_schema(any_whitespace=False) ↔ whitespace: 'xgrammar').
Маски огромные (внутри строки ~147 тыс. токенов), поэтому храним число разрешённых id и FNV-1a хэш
их отсортированного списка (каждый id — 4 байта little-endian). Тест: structured.xgrammar.test.ts.

Запуск:
  HF_HUB_OFFLINE=1 /tmp/inferstudy_venv/bin/python -I src/engine/tokenizer/fixtures/gen_xgrammar_masks.py <корень>
"""
import json
import os
import sys
from importlib.metadata import version

import xgrammar as xgr
from transformers import AutoTokenizer

ROOT = sys.argv[1]
OUT = os.path.join(ROOT, "src/engine/structured.xgrammar.fixture.json")
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "research/vendor/qwen3"))
V = 151936  # config.vocab_size Qwen3 (ширина логитов, notes-tokens §1.9)
info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=V)
compiler = xgr.GrammarCompiler(info)

NA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"], "additionalProperties": False}
BIG = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["c", "f"]},
        "days": {"type": "integer"},
        "ratio": {"type": "number"},
        "ok": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "loc": {"type": "object", "properties": {"lat": {"type": "number"}, "lon": {"type": "number"}}, "required": ["lat", "lon"], "additionalProperties": False},
    },
    "required": ["city", "unit"],
    "additionalProperties": False,
}
RU = {
    "type": "object",
    "properties": {"город": {"type": "string", "enum": ["Москва", "Казань"]}, "n": {"type": "array", "items": {"type": "integer"}}},
    "required": ["город"],
    "additionalProperties": False,
}
CASES = [
    ("notes §3.12: {name, age}", NA, '{"name": "Иван", "age": 42}'),
    ("пустая строка и отрицательное число", NA, '{"name": "", "age": -7}'),
    ("эмодзи и Ѣ режутся на несколько токенов внутри строки", NA, '{"name": "Ѣ🫠x", "age": 0}'),
    (
        "все типы, экранирование, вложенный объект, массив",
        BIG,
        '{"city": "Москва \\"центр\\" \\\\ 🙂\\u0416", "unit": "c", "ratio": -0.5e-3, "ok": true, "tags": ["a", "б"], "loc": {"lat": 55.75, "lon": 37}}',
    ),
    ("необязательные поля пропущены, пустой массив", BIG, '{"city": "x", "unit": "f", "days": 0, "tags": []}'),
    ("не-ASCII ключ и enum", RU, '{"город": "Казань", "n": [1, -20, 300]}'),
]


def fnv(ids):
    h = 2166136261
    for i in ids:
        for b in i.to_bytes(4, "little"):
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
    return h


out = []
for name, schema, target in CASES:
    json.loads(target)
    cg = compiler.compile_json_schema(json.dumps(schema), any_whitespace=False)
    m = xgr.GrammarMatcher(cg)
    ids = tok.encode(target, add_special_tokens=False)
    bm = xgr.allocate_token_bitmask(1, V)
    steps = []
    for k in range(len(ids) + 1):
        xgr.reset_token_bitmask(bm)
        m.fill_next_token_bitmask(bm)
        words = bm[0].numpy().view("uint32")
        allowed = [t for t in range(V) if (int(words[t >> 5]) >> (t & 31)) & 1]
        assert all(t < 151669 for t in allowed), "в маску попала пустая строка паддинга"
        steps.append({"count": len(allowed), "hash": fnv(allowed)})
        if k < len(ids):
            assert m.accept_token(ids[k]), (name, k)
    out.append({"name": name, "schema": schema, "target": target, "ids": ids, "steps": steps})
    print(name, [s["count"] for s in steps])

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(
        {"source": "xgrammar %s, Qwen3 TokenizerInfo(vocab_size=151936), compile_json_schema(any_whitespace=False)" % version("xgrammar"), "cases": out},
        f,
        ensure_ascii=False,
        indent=1,
    )
print("->", OUT)
