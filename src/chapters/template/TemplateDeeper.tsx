import { useMemo } from 'react'
import { TokenStrip, TokenStripSkeleton, fmt } from '@/ui'
import { renderTemplate } from '@/engine/templates'
import type { ChatMessage, Tokenizer } from '@/engine/types'
import { WEATHER_TOOL } from '../request/requestModel'

/**
 * Короткий кусок официального chat_template Qwen3 (research/vendor/qwen3/tokenizer_config.json, лицензия Apache 2.0).
 * Длинные строки сокращены «…», комментарии {#- … #} наши.
 */
export const QWEN3_JINJA_EXCERPT = `{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions … <tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return … </tool_call><|im_end|>\\n" }}
{%- endif %}
{#- … реплики; у ассистента (упрощённо): #}
{%- if loop.index0 > ns.last_query_index %}
    {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}
{%- else %}
    {{- '<|im_start|>' + message.role + '\\n' + content }}
{%- endif %}
{#- … #}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}`

/** Хвост промпта с enable_thinking=false: настоящие id. */
export function ThinkTail({ tk }: { tk: Tokenizer | null }) {
  const toks = useMemo(() => (tk ? tk.encode('<|im_start|>assistant\n<think>\n\n</think>\n\n') : null), [tk])
  if (!toks) return <TokenStripSkeleton count={7} />
  return <TokenStrip tokens={toks} showIds size="sm" variant="prompt" label="Хвост промпта с пустым блоком рассуждения" />
}

/** Сколько токенов стоит описание get_weather в шаблоне Qwen3. */
export function useToolsOverhead(tk: Tokenizer | null, messages: ChatMessage[]): number | null {
  return useMemo(() => {
    if (!tk) return null
    const a = tk.encode(renderTemplate({ family: 'qwen3', messages }).text).length
    const b = tk.encode(renderTemplate({ family: 'qwen3', messages, tools: [WEATHER_TOOL] }).text).length
    return b - a
  }, [tk, messages])
}

/**
 * Второй ход диалога: шаблон вырезает рассуждение прошлого ответа, и KV cache первого хода
 * совпадает с новым промптом только до <|im_start|>assistant (notes-tokens §3.3).
 */
export function ReasoningStripDemo({ tk, messages }: { tk: Tokenizer | null; messages: ChatMessage[] }) {
  const data = useMemo(() => {
    if (!tk) return null
    const reasoning = 'Нужно объяснить без формул и коротко.'
    const answer = 'KV cache хранит ключи и значения прошлых токенов, чтобы не считать их заново.'
    const generated = `<think>\n${reasoning}\n</think>\n\n${answer}<|im_end|>`
    // ход 1: промпт + то, что сгенерировала модель (KV для этих токенов уже лежит в кэше)
    const turn1 = tk.encode(renderTemplate({ family: 'qwen3', messages }).text + generated).map((t) => t.id)
    // ход 2: та же история + ответ ассистента + новый вопрос
    const turn2Msgs: ChatMessage[] = [...messages, { role: 'assistant', content: answer, reasoning }, { role: 'user', content: 'А сколько памяти он занимает?' }]
    const turn2 = tk.encode(renderTemplate({ family: 'qwen3', messages: turn2Msgs }).text).map((t) => t.id)
    let common = 0
    while (common < turn1.length && common < turn2.length && turn1[common] === turn2[common]) common++
    const cached = Math.floor(common / 16) * 16
    return { turn1: turn1.length, turn2: turn2.length, common, cached }
  }, [tk, messages])
  if (!data) return <TokenStripSkeleton count={10} />
  return (
    <div className="my-3 grid gap-2 rounded-panel border border-line px-4 py-3 text-[15px] leading-relaxed">
      <div>
        После первого хода в KV cache лежат <b>{fmt.tokens(data.turn1)}</b>: промпт и ответ модели вместе с рассуждением.
      </div>
      <div>
        Промпт второго хода: <b>{fmt.tokens(data.turn2)}</b>, рассуждения в нём уже нет.
      </div>
      <div>
        Совпадает только начало: {fmt.tokens(data.common)}, включая <code>{'<|im_start|>assistant'}</code>. Из кэша берутся{' '}
        <b>{fmt.tokens(data.cached)}</b> (целые блоки по 16), остальные {fmt.int(data.turn2 - data.cached)} считаются заново, включая
        ответ, который модель уже однажды прочитала.
      </div>
    </div>
  )
}
