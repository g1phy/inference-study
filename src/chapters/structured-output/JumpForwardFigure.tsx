import { useMemo, useState } from 'react'
import { Figure, Segmented, Stat, TokenStrip, TokenStripSkeleton, fmt, tint } from '@/ui'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import type { Token } from '@/engine/types'
import { PRESETS, serialize, type PresetId } from './schemas'
import { forcedPieces, jumpForwardCost } from './jumpforward'

export function JumpForwardFigure() {
  const { tokenizer } = useTokenizer()
  const [preset, setPreset] = useState<PresetId>('person')
  const p = PRESETS.find((x) => x.id === preset)!

  const data = useMemo(() => {
    const { text } = serialize(p.instance, p.schema)
    const pieces = forcedPieces(p.schema, text)
    if (!tokenizer) return { text, pieces, tokens: null, cost: null }
    const tokens = tokenizer.encode(text, { allowSpecial: false })
    return { text, pieces, tokens, cost: jumpForwardCost(tokens, pieces) }
  }, [p, tokenizer])

  const isForcedToken = (t: Token, i: number, list: Token[]) => {
    let pos = 0
    for (let k = 0; k < i; k++) pos = list[k].end ?? pos + list[k].text.length
    const s = t.start ?? pos
    const e = t.end ?? s + t.text.length
    for (let c = s; c < e; c++) if (!data.pieces.some((pc) => pc.forced && c >= pc.start && c < pc.end)) return false
    return true
  }

  return (
    <Figure
      id="structured-output-figure-jump-forward"
      title="Jump-forward: какие куски грамматика знает заранее"
      width="wide"
      caption="Попробуйте: переключите схему на «Вложенный объект» и сравните число проходов. Синим отмечено то, что грамматика допускает единственным образом: его можно дописать одним проходом, как маленький prefill. Оценка упрощённая, токены настоящие."
    >
      <div className="grid gap-5">
        <Segmented<PresetId> label="Схема" value={preset} onChange={setPreset} wrap options={PRESETS.map((x) => ({ value: x.id, label: x.label }))} />
        <div className="grid gap-2">
          <div className="text-sm font-medium text-ink-2">Ответ: принудительные куски и свободный выбор модели</div>
          <div className="rounded-control bg-surface-2 px-3 py-2.5 font-mono text-[14px] leading-loose break-words text-ink">
            {data.pieces.map((pc, i) => (
              <span
                key={i}
                className="rounded-[3px] px-px"
                style={pc.forced ? { background: tint('prefill', 22), boxShadow: 'inset 0 -2px 0 var(--c-prefill)' } : { background: tint('decode', 16) }}
              >
                {data.text.slice(pc.start, pc.end)}
              </span>
            ))}
          </div>
          <div className="flex flex-wrap gap-x-5 gap-y-1 text-[13px] text-ink-2">
            <span className="inline-flex items-center gap-1.5">
              <span aria-hidden className="h-2.5 w-4 rounded-[2px]" style={{ background: tint('prefill', 30), boxShadow: 'inset 0 -2px 0 var(--c-prefill)' }} />
              принудительно: дописываем без сэмплинга
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span aria-hidden className="h-2.5 w-4 rounded-[2px]" style={{ background: tint('decode', 24) }} />
              свободно: выбирает модель
            </span>
          </div>
        </div>
        {!data.tokens && <TokenStripSkeleton count={24} />}
        {data.tokens && data.cost && (
          <>
            <div className="grid gap-2">
              <div className="text-sm font-medium text-ink-2">Те же токены Qwen3: синие целиком лежат в принудительных кусках</div>
              <TokenStrip tokens={data.tokens} size="sm" variantOf={(t, i) => (isForcedToken(t, i, data.tokens!) ? 'prompt' : 'generated')} />
            </div>
            <div className="grid grid-cols-2 gap-x-6 gap-y-4 @3xl:grid-cols-4">
              <Stat label="Без jump-forward" value={fmt.int(data.cost.plain)} hint="проходов: по токену за шаг и EOS" swatch="var(--c-decode)" />
              <Stat label="С jump-forward" value={fmt.int(data.cost.jf)} hint="оценка: свободные токены и проход на кусок" swatch="var(--c-prefill)" />
              <Stat label="Принудительных кусков" value={fmt.int(data.cost.chunks)} />
              <Stat label="Выигрыш" value={fmt.multiplier(data.cost.plain / data.cost.jf)} hint="по числу проходов" />
            </div>
          </>
        )}
      </div>
    </Figure>
  )
}
