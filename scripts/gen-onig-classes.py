"""
Снимок классов символов Oniguruma, которыми пользуется регулярка пре-токенизации Qwen2/Qwen3 в HF `tokenizers`:
\\p{L}, \\p{N} и \\s. Пишет scripts/data/onig-unicode-classes.json (диапазоны code point), который читает
scripts/build-tokenizer.mjs.

Зачем: \\p{L} и \\p{N} в JS берутся из ICU движка браузера, а их версия Unicode плавает (Node 24 / ICU 78 —
Unicode 17, Safari и старые Chrome — 15–16). Oniguruma в tokenizers 0.23.3 знает меньше букв: например,
CJK Extension J (U+323B0…U+3347F) и цифры U+11DE0… для неё не \\p{L}/\\p{N}. Из-за этого строка вида
"𳎳'sI" резалась по-разному («'s» прилипало к букве) и id расходились с HF. С явными диапазонами
регулярка не зависит от Unicode-таблиц браузера и совпадает с Oniguruma посимвольно.

Метод: Split(Regex(класс), behavior="removed") на строке из всех code point (без суррогатов): всё, что удалено, —
члены класса. Нормализатор не применяется (это свойство самой регулярки).

Запуск (venv во /tmp, Python с -I):
  /tmp/inferstudy_venv/bin/python -I scripts/gen-onig-classes.py <корень проекта>
"""
import json
import os
import sys

import tokenizers
from tokenizers import Regex
from tokenizers.pre_tokenizers import Split

ROOT = sys.argv[1]
OUT = os.path.join(ROOT, "scripts/data/onig-unicode-classes.json")

CPS = [c for c in range(0x110000) if not (0xD800 <= c <= 0xDFFF)]
CHUNK = 20000


def members(pattern):
    sp = Split(Regex(pattern), behavior="removed", invert=False)
    out = []
    for i in range(0, len(CPS), CHUNK):
        chunk = CPS[i : i + CHUNK]
        rest = set(ord(ch) for p, _ in sp.pre_tokenize_str("".join(map(chr, chunk))) for ch in p)
        out.extend(c for c in chunk if c not in rest)
    return out


def ranges(cps):
    out = []
    for c in cps:
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return out


data = {
    "source": "HF tokenizers %s (Oniguruma), scripts/gen-onig-classes.py" % tokenizers.__version__,
    "L": ranges(members(r"\p{L}")),
    "N": ranges(members(r"\p{N}")),
    "S": ranges(members(r"\s")),
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(data, f, separators=(",", ":"))
    f.write("\n")
print({k: len(v) for k, v in data.items() if k != "source"}, "->", OUT)
