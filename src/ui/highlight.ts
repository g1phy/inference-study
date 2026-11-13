/**
 * Лёгкая подсветка синтаксиса без зависимостей: json, yaml, bash, jinja, python, text.
 * Цель: читабельные конфиги и шаблоны, а не полноценный парсер.
 * Результат разбит по строкам (для подсветки строк и номеров).
 */

export type CodeLang = 'json' | 'yaml' | 'bash' | 'jinja' | 'python' | 'text'

export type TokType = 'plain' | 'key' | 'str' | 'num' | 'kw' | 'com' | 'punct' | 'fn' | 'var' | 'flag' | 'special' | 'tag'

export interface CodeToken {
  t: TokType
  v: string
}

type Rule = { re: RegExp; t: TokType | ((m: string, src: string, at: number) => TokType | null) }

/** Спецтокены шаблонов чата: <|im_start|>, <|eot_id|>, полноширинные DeepSeek <｜User｜>, <think>, </tool_call> … */
const SPECIAL_RE = /<[|｜][^|｜<>\n]{1,40}[|｜]>|<\/?(?:think|tool_call|tool_response|answer)>/y

function lineStartBefore(src: string, at: number): string {
  const nl = src.lastIndexOf('\n', at - 1)
  return src.slice(nl + 1, at)
}

function prevChar(src: string, at: number): string {
  return at > 0 ? src[at - 1] : '\n'
}

function runRules(src: string, rules: Rule[]): CodeToken[] {
  const out: CodeToken[] = []
  let i = 0
  const push = (t: TokType, v: string) => {
    const last = out[out.length - 1]
    if (last && last.t === t && (t === 'plain' || t === 'punct')) last.v += v
    else out.push({ t, v })
  }
  outer: while (i < src.length) {
    for (const r of rules) {
      r.re.lastIndex = i
      const m = r.re.exec(src)
      if (m && m[0].length > 0) {
        const t = typeof r.t === 'function' ? r.t(m[0], src, i) : r.t
        if (t === null) continue
        push(t, m[0])
        i += m[0].length
        continue outer
      }
    }
    push('plain', src[i])
    i++
  }
  return out
}

const JSON_RULES: Rule[] = [
  { re: /\s+/y, t: 'plain' },
  { re: /"(?:\\.|[^"\\\n])*"(?=\s*:)/y, t: 'key' },
  { re: /"(?:\\.|[^"\\\n])*"?/y, t: 'str' },
  { re: /-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/y, t: 'num' },
  { re: /\b(?:true|false|null)\b/y, t: 'kw' },
  { re: /[{}[\],:]/y, t: 'punct' },
]

const YAML_RULES: Rule[] = [
  { re: /[ \t]+|\n/y, t: 'plain' },
  { re: /#[^\n]*/y, t: (_m, src, at) => (/\s/.test(prevChar(src, at)) ? 'com' : null) },
  {
    re: /(?:"[^"\n]*"|'[^'\n]*'|[A-Za-z0-9_./\-$]+)(?=:(?:[ \t]|\n|$))/y,
    t: (_m, src, at) => (/^[ \t]*(?:-[ \t]+)*$/.test(lineStartBefore(src, at)) ? 'key' : null),
  },
  { re: /:(?=[ \t]|\n|$)/y, t: 'punct' },
  { re: /-(?=[ \t]|\n)/y, t: 'punct' },
  { re: /"(?:\\.|[^"\\\n])*"|'(?:''|[^'\n])*'/y, t: 'str' },
  { re: /-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?=[ \t]*(?:#|\n|$|,|\]|\}))/y, t: 'num' },
  { re: /(?:true|false|null|yes|no|on|off|True|False|Null)(?=[ \t]*(?:#|\n|$|,|\]|\}))/y, t: 'kw' },
  { re: /[[\]{},|>]/y, t: 'punct' },
  { re: /[&*][A-Za-z0-9_-]+/y, t: 'var' },
  { re: /[^\s#,[\]{}]+/y, t: 'plain' },
]

const PY_KEYWORDS =
  'False|None|True|and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield'

const PY_RULES: Rule[] = [
  { re: /[ \t]+|\n/y, t: 'plain' },
  { re: /#[^\n]*/y, t: 'com' },
  { re: /[rbfuRBFU]{0,2}("""|''')[\s\S]*?(?:\1|$)/y, t: 'str' },
  { re: /[rbfuRBFU]{0,2}(?:"(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?)/y, t: 'str' },
  { re: /@[A-Za-z_][\w.]*/y, t: 'fn' },
  { re: /\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b/y, t: 'num' },
  { re: /(?<=\b(?:def|class)\s+)[A-Za-z_]\w*/y, t: 'fn' },
  { re: new RegExp(`\\b(?:${PY_KEYWORDS})\\b`, 'y'), t: 'kw' },
  { re: /[A-Za-z_]\w*(?=\()/y, t: 'fn' },
  { re: /[A-Za-z_]\w*/y, t: 'plain' },
  { re: /[()[\]{}:.,;=+\-*/%<>!&|^~]+/y, t: 'punct' },
]

const TEXT_RULES: Rule[] = [
  { re: SPECIAL_RE, t: 'special' },
  { re: /[^<]+|</y, t: 'plain' },
]

/* ---------- bash: команда в начале строки, флаги, строки, переменные ---------- */
function tokenizeBash(src: string): CodeToken[] {
  const out: CodeToken[] = []
  let i = 0
  let expectCmd = true
  const push = (t: TokType, v: string) => {
    const last = out[out.length - 1]
    if (last && last.t === t && (t === 'plain' || t === 'punct')) last.v += v
    else out.push({ t, v })
  }
  const at = (re: RegExp) => {
    re.lastIndex = i
    const m = re.exec(src)
    return m ? m[0] : null
  }
  while (i < src.length) {
    let m: string | null
    if ((m = at(/\n/y))) {
      push('plain', m)
      expectCmd = !/\\\s*$/.test(lineStartBefore(src, i))
    } else if ((m = at(/[ \t]+/y))) push('plain', m)
    else if ((m = at(/\\(?=\n)/y))) push('punct', m)
    else if (/[\s]/.test(prevChar(src, i)) && (m = at(/#[^\n]*/y))) push('com', m)
    else if (expectCmd && /^\s*$/.test(lineStartBefore(src, i)) && (m = at(/\$(?= )/y))) push('punct', m)
    else if ((m = at(/"(?:\\.|[^"\\])*"?|'[^']*'?/y))) {
      push('str', m)
      expectCmd = false
    } else if ((m = at(/\$\{[^}\n]*\}|\$[A-Za-z_][A-Za-z0-9_]*/y))) {
      push('var', m)
      expectCmd = false
    } else if ((m = at(/&&|\|\||[|;]/y))) {
      push('punct', m)
      expectCmd = true
    } else if ((m = at(/[<>=]+/y))) push('punct', m)
    else if (/[\s]/.test(prevChar(src, i)) && (m = at(/--?[A-Za-z0-9][\w.-]*/y))) push('flag', m)
    else if ((m = at(/[^\s"'$|;&<>=\\]+/y))) {
      if (expectCmd && /^[A-Za-z_][\w./-]*$/.test(m) && !/=/.test(m)) {
        push('fn', m)
        expectCmd = false
      } else push(/^\d+(?:\.\d+)?$/.test(m) ? 'num' : 'plain', m)
    } else {
      push('plain', src[i])
      i++
      continue
    }
    i += m.length
  }
  return out
}

/* ---------- jinja: текст шаблона + теги {% %}, {{ }}, {# #} ---------- */
const JINJA_KW =
  'if|elif|else|endif|for|in|endfor|set|endset|macro|endmacro|call|endcall|filter|endfilter|not|and|or|is|true|false|none|True|False|None|loop|namespace|raise_exception|defined'

const JINJA_INNER: Rule[] = [
  { re: /[ \t\n]+/y, t: 'plain' },
  { re: /"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/y, t: 'str' },
  { re: /\b\d+(?:\.\d+)?\b/y, t: 'num' },
  { re: new RegExp(`\\b(?:${JINJA_KW})\\b`, 'y'), t: 'kw' },
  { re: /(?<=\|\s*)[A-Za-z_]\w*/y, t: 'fn' },
  { re: /[A-Za-z_]\w*(?=\()/y, t: 'fn' },
  { re: /[A-Za-z_]\w*/y, t: 'var' },
  { re: /[()[\]{}.,:|=+\-*/%<>!~]+/y, t: 'punct' },
]

function tokenizeJinja(src: string): CodeToken[] {
  const out: CodeToken[] = []
  let i = 0
  const open = /\{#[\s\S]*?(?:#\}|$)|\{\{-?|\{%-?/g
  while (i < src.length) {
    open.lastIndex = i
    const m = open.exec(src)
    const textEnd = m ? m.index : src.length
    if (textEnd > i) out.push(...runRules(src.slice(i, textEnd), TEXT_RULES))
    if (!m) break
    if (m[0].startsWith('{#')) {
      out.push({ t: 'com', v: m[0] })
      i = m.index + m[0].length
      continue
    }
    const closer = m[0].startsWith('{{') ? /-?\}\}/g : /-?%\}/g
    closer.lastIndex = m.index + m[0].length
    const c = closer.exec(src)
    const innerEnd = c ? c.index : src.length
    out.push({ t: 'tag', v: m[0] })
    out.push(...runRules(src.slice(m.index + m[0].length, innerEnd), JINJA_INNER))
    if (c) out.push({ t: 'tag', v: c[0] })
    i = c ? c.index + c[0].length : src.length
  }
  return out
}

function splitLines(tokens: CodeToken[]): CodeToken[][] {
  const lines: CodeToken[][] = [[]]
  for (const tok of tokens) {
    const parts = tok.v.split('\n')
    parts.forEach((p, idx) => {
      if (idx > 0) lines.push([])
      if (p) lines[lines.length - 1].push({ t: tok.t, v: p })
    })
  }
  return lines
}

/** Разбить код на строки токенов для подсветки. */
export function highlight(code: string, lang: CodeLang = 'text'): CodeToken[][] {
  const src = code.replace(/\r\n?/g, '\n')
  let tokens: CodeToken[]
  switch (lang) {
    case 'json':
      tokens = runRules(src, JSON_RULES)
      break
    case 'yaml':
      tokens = runRules(src, YAML_RULES)
      break
    case 'python':
      tokens = runRules(src, PY_RULES)
      break
    case 'bash':
      tokens = tokenizeBash(src)
      break
    case 'jinja':
      tokens = tokenizeJinja(src)
      break
    default:
      tokens = runRules(src, TEXT_RULES)
  }
  return splitLines(tokens)
}

export const LANG_LABELS: Record<CodeLang, string> = {
  json: 'JSON',
  yaml: 'YAML',
  bash: 'bash',
  jinja: 'Jinja',
  python: 'Python',
  text: 'текст',
}
