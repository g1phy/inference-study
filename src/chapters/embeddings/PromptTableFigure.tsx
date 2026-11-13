import { memo, useMemo, useState, type PointerEvent } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import {
  Figure,
  Segmented,
  Select,
  Stat,
  TokenChip,
  TokenStrip,
  TokenStripSkeleton,
  Skeleton,
  divColor,
  fmt,
  fontCss,
  truncateText,
  useElementSize,
} from '@/ui'
import { useSettings } from '@/state/settings'
import { scrollToSimulator } from '@/state/nav'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { renderTemplate } from '@/engine/templates'
import { getModel } from '@/engine/data/models'
import type { ChatMessage, Token } from '@/engine/types'
import { Barcode, BarcodeLegend } from './Barcode'
import { BARCODE_DIMS, VALUE_SCALE, embeddingRow } from './vectors'
import { tokenLabel } from './tokenText'

/** Модели с токенизатором Qwen3: ID на странице считает настоящий словарь Qwen3. */
export const QWEN3_MODEL_IDS = ['qwen3-8b', 'qwen3-32b', 'qwen3-30b-a3b', 'qwen3-235b-a22b'] as const
const MAX_ROWS = 48

type Mode = 'message' | 'template'

export function PromptTableFigure() {
  const [settings] = useSettings()
  const { tokenizer, status, error } = useTokenizer()
  const [mode, setMode] = useState<Mode>('template')
  const [pickedModel, setPickedModel] = useState<string | null>(null)
  const modelId =
    pickedModel ?? ((QWEN3_MODEL_IDS as readonly string[]).includes(settings.modelId) ? settings.modelId : 'qwen3-8b')
  const model = getModel(modelId)

  const tokens = useMemo<Token[] | null>(() => {
    if (!tokenizer) return null
    if (mode === 'message') return tokenizer.encode(settings.userText)
    const messages: ChatMessage[] = []
    if (settings.systemText.trim()) messages.push({ role: 'system', content: settings.systemText })
    messages.push({ role: 'user', content: settings.userText })
    const rendered = renderTemplate({ family: 'qwen3', messages, enableThinking: settings.thinking, addGenerationPrompt: true })
    return tokenizer.encode(rendered.text)
  }, [tokenizer, mode, settings.userText, settings.systemText, settings.thinking])

  // индексы selected / hovered абсолютные (позиция в промпте); матрица показывает окно из MAX_ROWS строк
  const [selected, setSelected] = useState<number | null>(null)
  const [hovered, setHovered] = useState<number | null>(null)
  const [offsetRaw, setOffset] = useState(0)

  const total = tokens?.length ?? 0
  const offset = Math.max(0, Math.min(offsetRaw, total - MAX_ROWS))
  const inWindow = (i: number) => i >= offset && i < offset + MAX_ROWS
  const shown = useMemo(() => (tokens ? tokens.slice(offset, offset + MAX_ROWS) : []), [tokens, offset])
  // по умолчанию выбран «·KV» (есть в запросе по умолчанию) или первый обычный токен окна
  const defaultIndex = useMemo(() => {
    if (!shown.length) return null
    const kv = shown.findIndex((t) => t.text === ' KV')
    if (kv >= 0) return offset + kv
    const plain = shown.findIndex((t) => !t.special)
    return offset + (plain >= 0 ? plain : 0)
  }, [shown, offset])
  const sel = selected !== null && selected < total && inWindow(selected) ? selected : defaultIndex
  const active = hovered !== null && hovered < total && inWindow(hovered) ? hovered : sel
  const activeToken = active !== null && tokens ? tokens[active] : null
  // нажатие на токен за пределами окна сдвигает окно к нему
  const pickToken = (i: number) => {
    if (!inWindow(i)) setOffset(Math.max(0, Math.min(i - Math.floor(MAX_ROWS / 2), total - MAX_ROWS)))
    setSelected(i)
    setHovered(null)
  }

  const tableParams = model.vocab * model.hidden
  const promptNumbers = (tokens?.length ?? 0) * model.hidden

  const controls = (
    <>
      <Segmented<Mode>
        label="Что токенизируем"
        value={mode}
        onChange={(m) => {
          setMode(m)
          setSelected(null)
          setHovered(null)
          setOffset(0)
        }}
        options={[
          { value: 'template', label: 'Весь промпт' },
          { value: 'message', label: 'Только сообщение' },
        ]}
      />
      <Select
        label="Модель (словарь Qwen3)"
        value={modelId}
        onChange={setPickedModel}
        options={QWEN3_MODEL_IDS.map((id) => ({ value: id, label: getModel(id).name }))}
      />
    </>
  )

  const empty = tokens !== null && tokens.length === 0

  return (
    <Figure
      id="embeddings-figure-lookup"
      title="Ваш промпт: ID токенов выбирают строки таблицы эмбеддингов"
      controls={controls}
      controlsPosition="top"
      caption={
        <>
          <span className="font-medium text-ink">Попробуйте:</span> нажмите на любой <code>&lt;|im_start|&gt;</code> в ленте: подсветятся все его
          строки в матрице, и все они ведут к одной строке таблицы, поэтому векторы одинаковые. Затем выберите «Только сообщение», чтобы оставить
          только ваш текст. Числа в векторах условные (иллюстрация): настоящие веса мы не загружаем, но размеры таблицы посчитаны по config
          модели.
        </>
      }
    >
      <div className="grid min-w-0 gap-5">
        <div className="min-w-0">
          <div className="mb-2 text-sm text-ink-2">
            {mode === 'template' ? 'Промпт после шаблона чата Qwen3, настоящие токены' : 'Ваше сообщение, настоящие токены Qwen3'}
            {tokens && tokens.length > 0 && <span className="text-ink-3"> · {fmt.tokens(tokens.length)}</span>}
          </div>
          {status === 'error' ? (
            <p className="text-sm text-bad">Не удалось загрузить токенизатор: {error}</p>
          ) : !tokens ? (
            <TokenStripSkeleton count={30} />
          ) : empty ? (
            <p className="text-sm text-ink-2">
              Сообщение пустое.{' '}
              <button type="button" className="text-accent underline underline-offset-2" onClick={() => scrollToSimulator({ focusPrompt: true })}>
                Напишите запрос в симуляторе
              </button>
              , и здесь появятся его токены.
            </p>
          ) : (
            <TokenStrip
              tokens={tokens}
              variant="prompt"
              limit={120}
              highlightIndex={activeToken ? tokens.flatMap((t, i) => (t.id === activeToken.id ? [i] : [])) : null}
              onClickIndex={pickToken}
              onHoverIndex={(i) => setHovered(i !== null && inWindow(i) ? i : null)}
              label="Токены промпта"
            />
          )}
        </div>

        {!empty && (
          <div className="grid min-w-0 gap-5 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,300px)]">
            <div className="min-w-0">
              {!tokens ? (
                <Skeleton height={300} radius={8} label="Загружаем токенизатор" />
              ) : (
                <TableMatrix
                  tokens={shown}
                  offset={offset}
                  total={total}
                  vocab={model.vocab}
                  dModel={model.hidden}
                  active={active !== null ? active - offset : null}
                  onHover={(r) => setHovered(r === null ? null : offset + r)}
                  onSelect={(r) => setSelected(offset + r)}
                />
              )}
              {tokens && total > MAX_ROWS && (
                <p className="mt-2 text-xs text-ink-3">
                  Показаны строки с {fmt.int(offset + 1)} по {fmt.int(offset + shown.length)} из {fmt.int(total)}: нажмите на токен в ленте, чтобы
                  перейти к нему.
                </p>
              )}
            </div>
            <div className="min-w-0">
              {activeToken && active !== null && tokens ? (
                <TokenDetail token={activeToken} index={active} tokens={tokens} dModel={model.hidden} vocab={model.vocab} />
              ) : (
                <Skeleton lines={5} />
              )}
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-x-6 gap-y-5 border-t border-line pt-4 @2xl:grid-cols-4">
          <Stat
            label="Строк в таблице"
            value={fmt.int(model.vocab)}
            hint={
              tokenizer
                ? `в токенизаторе ${fmt.int(tokenizer.vocabSize)}, ещё ${fmt.int(model.vocab - tokenizer.vocabSize)} ${fmt.plural(model.vocab - tokenizer.vocabSize, ['строка не занята', 'строки не заняты', 'строк не заняты'])} ни одним токеном`
                : 'ширина логитов модели'
            }
          />
          <Stat label="Длина вектора" value={fmt.int(model.hidden)} hint={`d_model, в config ${model.name} это hidden_size`} />
          <Stat
            label="Параметров"
            value={fmt.compact(tableParams)}
            hint={`${fmt.int(model.vocab)} × ${fmt.int(model.hidden)}, это ${fmt.pct(tableParams / (model.paramsB * 1e9))} модели`}
          />
          <Stat
            label="Вес в BF16"
            value={fmt.bytes(tableParams * 2)}
            hint={
              !tokens
                ? '2 байта на число'
                : empty
                  ? 'промпт пустой'
                  : `а сам промпт: ${fmt.int(tokens.length)} × ${fmt.int(model.hidden)} = ${fmt.compact(promptNumbers)} чисел, ${fmt.bytes(promptNumbers * 2)}`
            }
          />
        </div>
      </div>
    </Figure>
  )
}

/* ---------------------------------------------------------------------------
 * Таблица-«миникарта» слева, матрица векторов промпта справа, связь выбранной строки.
 * ------------------------------------------------------------------------- */

function TableMatrix({
  tokens,
  offset,
  total,
  vocab,
  dModel,
  active,
  onHover,
  onSelect,
}: {
  /** Строки окна (не больше MAX_ROWS). */
  tokens: Token[]
  /** Позиция первой строки окна в промпте и длина всего промпта. */
  offset: number
  total: number
  vocab: number
  dModel: number
  /** Индекс строки внутри окна. */
  active: number | null
  onHover: (i: number | null) => void
  onSelect: (i: number) => void
}) {
  const [ref, { width }] = useElementSize<HTMLDivElement>()
  const reduce = useReducedMotion()
  const n = tokens.length
  const W = Math.max(260, width)
  // на широком экране строки не ниже 12px, чтобы у каждой была подпись токена; на узком матрица компактнее
  const rowH = W >= 400 ? Math.max(12, Math.min(16, Math.round(560 / Math.max(1, n)))) : Math.max(7, Math.min(16, Math.round(336 / Math.max(1, n))))
  const top = 30
  const bodyH = n * rowH
  const H = top + bodyH + 30
  const mapW = 26
  const gap = W >= 560 ? 72 : 44
  const labelW = rowH >= 12 && W >= 400 ? 96 : 0
  const mx0 = mapW + gap
  const mx1 = W - (labelW ? labelW + 8 : 0)
  const cellW = (mx1 - mx0) / BARCODE_DIMS

  const rows = useMemo(() => tokens.map((t) => embeddingRow(t.id)), [tokens])
  const activeId = active !== null ? tokens[active]?.id : undefined
  const tickY = (id: number) => top + (id / Math.max(1, vocab - 1)) * bodyH
  const uniqueIds = useMemo(() => [...new Set(tokens.map((t) => t.id))], [tokens])
  const firstIndexOf = (id: number) => tokens.findIndex((t) => t.id === id)

  const onMapMove = (e: PointerEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    const y = top + ((e.clientY - box.top) / box.height) * bodyH
    let best = -1
    let bestD = Infinity
    for (const id of uniqueIds) {
      const d = Math.abs(tickY(id) - y)
      if (d < bestD) {
        bestD = d
        best = id
      }
    }
    if (best >= 0 && bestD < 14) onHover(firstIndexOf(best))
  }

  const labelFont = '11px "JetBrains Mono Variable", monospace'

  return (
    <div ref={ref} className="min-w-0">
      {width > 0 && (
        <svg
          width={W}
          height={H}
          role="img"
          aria-label={`Таблица эмбеддингов из ${fmt.int(vocab)} строк: каждый из ${n} показанных токенов промпта выбирает строку по своему ID. Справа матрица промпта ${total} × ${fmt.int(dModel)}, показаны строки с ${offset + 1} по ${offset + n} и первые ${BARCODE_DIMS} столбцов.`}
          className="block select-none"
          onPointerLeave={(e) => e.pointerType === 'mouse' && onHover(null)}
        >
          {/* подписи колонок */}
          <text x={0} y={12} style={{ font: '600 12px "Onest Variable", sans-serif', fill: 'var(--ink-2)' }}>
            таблица
          </text>
          <text x={mx0} y={12} style={{ font: '600 12px "Onest Variable", sans-serif', fill: 'var(--ink-2)' }}>
            {`ваш промпт: ${fmt.int(total)} × ${fmt.int(dModel)}`}
          </text>
          {W >= 440 && (
            <text x={mx1} y={12} textAnchor="end" style={{ font: '12px "Onest Variable", sans-serif', fill: 'var(--ink-3)' }}>
              {`первые ${BARCODE_DIMS} из ${fmt.int(dModel)} столбцов`}
            </text>
          )}

          {/* миникарта таблицы: 0 сверху, последняя строка снизу */}
          <rect x={0} y={top} width={mapW} height={bodyH} rx={4} style={{ fill: 'var(--surface-2)', stroke: 'var(--line-strong)', strokeWidth: 1 }} />
          <text x={mapW / 2} y={top - 5} textAnchor="middle" style={{ font: '11px "Onest Variable", sans-serif', fill: 'var(--ink-3)' }}>
            0
          </text>
          <text x={0} y={top + bodyH + 16} style={{ ...fontCss('11px "Onest Variable", sans-serif'), fill: 'var(--ink-3)', fontVariantNumeric: 'tabular-nums' }}>
            {fmt.int(vocab - 1)}
          </text>
          {uniqueIds.map((id) => {
            const special = tokens.find((t) => t.id === id)?.special
            const on = id === activeId
            const y = tickY(id)
            return (
              <line
                key={id}
                x1={on ? -2 : 2}
                x2={on ? mapW + 2 : mapW - 2}
                y1={y}
                y2={y}
                style={{
                  stroke: on ? 'var(--accent)' : special ? 'var(--c-special)' : 'var(--c-prefill)',
                  strokeWidth: on ? 2.5 : 1.5,
                  strokeDasharray: special && !on ? '3 2' : undefined,
                }}
              />
            )
          })}
          <rect
            x={0}
            y={top}
            width={mapW}
            height={bodyH}
            fill="transparent"
            onPointerMove={onMapMove}
            onPointerDown={onMapMove}
            style={{ cursor: 'crosshair' }}
          />

          {/* связь: строка таблицы → позиции в промпте (у повторов одна строка на несколько позиций) */}
          {activeId !== undefined &&
            tokens.map((t, r) =>
              t.id === activeId ? (
                <motion.path
                  key={`c-${r}`}
                  initial={reduce ? false : { opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.18 }}
                  d={`M ${mapW + 3} ${tickY(activeId)} C ${mapW + gap / 2} ${tickY(activeId)}, ${mx0 - gap / 2} ${top + r * rowH + rowH / 2}, ${mx0 - 3} ${top + r * rowH + rowH / 2}`}
                  style={{ fill: 'none', stroke: 'var(--accent)', strokeWidth: 1.5 }}
                />
              ) : null,
            )}

          {/* матрица: строка на токен (ячейки мемоизированы, при наведении меняется только прозрачность строк) */}
          {rows.map((vals, r) => {
            const dim = activeId !== undefined && tokens[r].id !== activeId
            return (
              <g key={r} opacity={dim ? 0.42 : 1} style={{ transition: 'opacity 150ms' }}>
                <MatrixRow vals={vals} x0={mx0} y={top + r * rowH + 0.5} cellW={cellW} h={rowH - (rowH >= 10 ? 1.5 : 1)} />
              </g>
            )
          })}
          {activeId !== undefined &&
            tokens.map((t, r) =>
              t.id === activeId ? (
                <rect
                  key={`o-${r}`}
                  x={mx0 - 2}
                  y={top + r * rowH - 1}
                  width={mx1 - mx0 + 4}
                  height={rowH + 1}
                  rx={3}
                  style={{ fill: 'none', stroke: 'var(--accent)', strokeWidth: 1.75 }}
                />
              ) : null,
            )}

          {/* подписи строк */}
          {labelW > 0 &&
            tokens.map((t, r) => {
              const on = active === r || (activeId !== undefined && t.id === activeId)
              return (
                <text
                  key={`l-${r}`}
                  x={mx1 + 8}
                  y={top + r * rowH + rowH / 2}
                  dominantBaseline="central"
                  style={{ ...fontCss(labelFont), fill: on ? 'var(--ink)' : 'var(--ink-3)', fontWeight: on ? 600 : 400 }}
                >
                  {truncateText(tokenLabel(t), labelW, labelFont)}
                </text>
              )
            })}

          {/* зоны наведения по строкам */}
          {tokens.map((_, r) => (
            <rect
              key={`h-${r}`}
              x={mx0 - 4}
              y={top + r * rowH}
              width={W - mx0 + 4}
              height={rowH}
              fill="transparent"
              onPointerEnter={(e) => e.pointerType === 'mouse' && onHover(r)}
              onClick={() => {
                onSelect(r)
                onHover(null)
              }}
              style={{ cursor: 'pointer' }}
            />
          ))}
        </svg>
      )}
      <div className="mt-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-3">
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="h-0 w-3 border-t-2" style={{ borderColor: 'var(--c-prefill)' }} />
            строка обычного токена
          </span>
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="h-0 w-3 border-t-2 border-dashed" style={{ borderColor: 'var(--c-special)' }} />
            строка спецтокена
          </span>
        </div>
        <BarcodeLegend />
      </div>
    </div>
  )
}

const MatrixRow = memo(function MatrixRow({ vals, x0, y, cellW, h }: { vals: number[]; x0: number; y: number; cellW: number; h: number }) {
  return (
    <>
      {vals.map((v, c) => (
        <rect key={c} x={x0 + c * cellW} y={y} width={cellW + 0.35} height={h} style={{ fill: divColor(v / 3) }} shapeRendering="crispEdges" />
      ))}
    </>
  )
})

function TokenDetail({ token, index, tokens, dModel, vocab }: { token: Token; index: number; tokens: Token[]; dModel: number; vocab: number }) {
  const row = embeddingRow(token.id)
  const repeats = tokens.filter((t) => t.id === token.id).length
  const first = row.slice(0, 4).map((v) => fmt.num(v * VALUE_SCALE, 3))
  return (
    <div className="grid content-start gap-3 rounded-panel bg-surface-2 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <TokenChip token={token} variant={token.special ? 'default' : 'prompt'} showId />
        <span className="text-sm text-ink-2">позиция {index + 1}</span>
      </div>
      <p className="text-[15px] leading-snug text-ink">
        ID <span className="font-semibold tabular-nums">{fmt.int(token.id)}</span> → строка таблицы с тем же номером
        <span className="text-ink-3"> (из {fmt.int(vocab)})</span>.
      </p>
      <div className="grid gap-1.5">
        <Barcode values={row} height={30} label={`Условный вектор токена ${tokenLabel(token)}`} />
        <div className="font-mono text-[12px] leading-snug text-ink-2 tabular-nums">
          {first.join(' · ')} · … <span className="text-ink-3">ещё {fmt.int(dModel - 4)} {fmt.plural(dModel - 4, ['число', 'числа', 'чисел'])} (условно)</span>
        </div>
      </div>
      <p className="text-sm leading-snug text-ink-2">
        {token.special
          ? 'Спецтокен шаблона: у него тоже своя строка, выученная при обучении. Так модель «видит» границы реплик.'
          : repeats > 1
            ? `Токен встречается в промпте ${repeats} ${fmt.plural(repeats, ['раз', 'раза', 'раз'])}: каждый раз берётся одна и та же строка.`
            : 'Таблица не знает, где стоит токен: строка одна и та же на любой позиции. Порядок добавит позже RoPE внутри attention.'}
        {token.special && repeats > 1 && ` Здесь он встречается ${repeats} ${fmt.plural(repeats, ['раз', 'раза', 'раз'])}, векторы одинаковые.`}
      </p>
    </div>
  )
}
