import { useState, type KeyboardEvent } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { Figure, Segmented, useElementSize } from '@/ui'
import { KEY_INDEX, MAP_CONTEXTS, MAP_GROUP_LABELS, MAP_WORDS, mapLayout, nearestTo, project, type MapContextId } from './meaningMap'

/**
 * Иллюстративная 2D-карта смыслов: координаты подобраны вручную, чтобы показать идею
 * «похожие по смыслу токены лежат рядом». Настоящие векторы имеют тысячи измерений.
 */
type Ctx = 'table' | MapContextId

const quoteList = (idx: number[]) => idx.map((j) => `«${MAP_WORDS[j].w}»`).join(', ')

export function MeaningMapFigure() {
  const [ref, { width }] = useElementSize<HTMLDivElement>()
  const reduce = useReducedMotion()
  const [sel, setSel] = useState(KEY_INDEX)
  const [ctx, setCtx] = useState<Ctx>('table')

  const L = mapLayout(width)
  const { W, H } = L
  const narrow = L.kind === 'narrow'
  const pts = MAP_WORDS.map((w) => project(L, w))
  const ctxInfo = ctx === 'table' ? null : MAP_CONTEXTS[ctx]
  const ctxPoint = ctxInfo ? project(L, ctxInfo) : null
  // «ключ» в предложении: соседей показываем у сдвинутого вектора, строку таблицы не считаем
  const fromCtx = ctxPoint !== null && sel === KEY_INDEX
  const from = fromCtx && ctxPoint ? ctxPoint : pts[sel]
  const near = nearestTo(pts, from, fromCtx ? [KEY_INDEX] : [sel])

  const pickCtx = (c: Ctx) => {
    setCtx(c)
    setSel(KEY_INDEX)
  }

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') setSel((s) => (s + 1) % MAP_WORDS.length)
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') setSel((s) => (s - 1 + MAP_WORDS.length) % MAP_WORDS.length)
    else return
    e.preventDefault()
  }

  const fontPx = narrow ? 12 : 13
  const font = `${fontPx}px "Onest Variable", sans-serif`
  const groupFont = 'italic 12px "Onest Variable", sans-serif'
  const halo = { paintOrder: 'stroke', stroke: 'var(--surface)', strokeWidth: 3, strokeLinejoin: 'round' } as const

  // стрелка от строки таблицы к вектору после слоёв: наконечник у края кружка, а не под ним
  const keyPt = pts[KEY_INDEX]
  const arrow = ctxPoint
    ? (() => {
        const dx = ctxPoint.x - keyPt.x
        const dy = ctxPoint.y - keyPt.y
        const len = Math.hypot(dx, dy) || 1
        return { x1: keyPt.x + (dx / len) * 7, y1: keyPt.y + (dy / len) * 7, x2: ctxPoint.x - (dx / len) * 9, y2: ctxPoint.y - (dy / len) * 9 }
      })()
    : null

  const selWord = MAP_WORDS[sel].w

  return (
    <Figure
      id="embeddings-figure-map"
      title="Карта смыслов (иллюстрация): похожие токены лежат рядом"
      controls={
        <Segmented<Ctx>
          label="«Ключ» в предложении"
          size="sm"
          wrap
          value={ctx}
          onChange={pickCtx}
          options={[
            { value: 'table', label: 'без контекста' },
            { value: 'door', label: MAP_CONTEXTS.door.label },
            { value: 'spring', label: MAP_CONTEXTS.spring.label },
          ]}
        />
      }
      caption="Попробуйте: наведите на слово или нажмите на него, чтобы увидеть ближайших соседей, затем поставьте «ключ» в предложение. Это иллюстрация: настоящие векторы имеют тысячи измерений, координаты здесь подобраны вручную, а слова для простоты считаются целыми токенами."
    >
      <div
        ref={ref}
        tabIndex={0}
        role="group"
        aria-label="Карта смыслов: стрелки выбирают слово"
        onKeyDown={onKey}
        className="mx-auto w-full max-w-[860px] min-w-0 rounded-control focus-visible:outline-offset-4"
      >
        {width > 0 && (
          <svg width={W} height={H} role="img" aria-label="Двумерная иллюстрация пространства эмбеддингов: группы слов по смыслу" className="block select-none">
            {MAP_GROUP_LABELS.map((g) => {
              if ('corner' in g) {
                const right = g.corner === 'tr' || g.corner === 'br'
                const top = g.corner === 'tl' || g.corner === 'tr'
                return (
                  <text key={g.label} x={right ? W - 6 : 6} y={top ? 14 : H - 6} textAnchor={right ? 'end' : 'start'} style={{ font: groupFont, fill: 'var(--ink-3)' }}>
                    {g.label}
                  </text>
                )
              }
              const p = project(L, g)
              return (
                <text key={g.label} x={p.x} y={p.y} textAnchor={g.anchor} style={{ font: groupFont, fill: 'var(--ink-3)', ...halo }}>
                  {g.label}
                </text>
              )
            })}
            {/* связи с ближайшими соседями: от выбранного слова или от «ключа» после слоёв */}
            {near.map((j) => (
              <line
                key={`n-${sel}-${ctx}-${j}`}
                x1={from.x}
                y1={from.y}
                x2={pts[j].x}
                y2={pts[j].y}
                style={{ stroke: 'var(--accent)', strokeWidth: 1.25, strokeDasharray: '4 3', opacity: 0.85 }}
              />
            ))}
            {/* «ключ» в контексте: вектор после слоёв сдвигается к нужному смыслу */}
            {ctxPoint && arrow && (
              <g>
                <defs>
                  <marker id="emb-map-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                    <path d="M 0 0 L 10 5 L 0 10 z" style={{ fill: 'var(--ink)' }} />
                  </marker>
                </defs>
                <motion.line
                  key={`${ctx}-${L.kind}`}
                  x1={arrow.x1}
                  y1={arrow.y1}
                  x2={arrow.x2}
                  y2={arrow.y2}
                  initial={reduce ? false : { opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.25 }}
                  markerEnd="url(#emb-map-arrow)"
                  style={{ stroke: 'var(--ink)', strokeWidth: 1.75 }}
                />
                <motion.circle
                  key={`${ctx}-${L.kind}-c`}
                  initial={reduce ? false : { scale: 0.4, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.3, delay: reduce ? 0 : 0.15 }}
                  cx={ctxPoint.x}
                  cy={ctxPoint.y}
                  r={6}
                  style={{ fill: 'var(--surface)', stroke: 'var(--ink)', strokeWidth: 2, transformBox: 'fill-box', transformOrigin: 'center' }}
                />
              </g>
            )}
            {MAP_WORDS.map((p, i) => {
              const isSel = i === sel
              const isNear = near.includes(i)
              const isKey = i === KEY_INDEX
              const { x, y } = pts[i]
              const side = p.label ?? 'right'
              return (
                <g key={p.w} onPointerEnter={() => setSel(i)} onPointerDown={() => setSel(i)} style={{ cursor: 'pointer' }}>
                  <circle cx={x} cy={y} r={14} fill="transparent" />
                  <circle
                    cx={x}
                    cy={y}
                    r={isSel ? 6 : 4.5}
                    style={{
                      fill: isSel ? 'var(--accent)' : isNear ? 'var(--ink)' : 'var(--ink-3)',
                      stroke: isKey && !isSel ? 'var(--ink)' : 'none',
                      strokeWidth: 1.5,
                    }}
                  />
                  <text
                    x={side === 'left' ? x - 9 : side === 'top' ? x : x + 9}
                    y={side === 'top' ? y - 14 : y}
                    dominantBaseline="central"
                    textAnchor={side === 'left' ? 'end' : side === 'top' ? 'middle' : 'start'}
                    style={{
                      font: isSel || isKey ? `600 ${font}` : font,
                      fill: isSel || isNear || isKey ? 'var(--ink)' : 'var(--ink-2)',
                      ...halo,
                    }}
                  >
                    {p.w}
                  </text>
                </g>
              )
            })}
          </svg>
        )}
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-3">
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-accent" />
            выбранное слово
          </span>
          <span className="flex items-center gap-1.5">
            <span aria-hidden className="h-0 w-4 border-t-[1.5px] border-dashed border-accent" />
            три ближайших соседа
          </span>
          {ctxPoint && (
            <span className="flex items-center gap-1.5">
              <span aria-hidden className="h-3 w-3 rounded-full border-2 border-ink bg-surface" />
              «ключ» после слоёв
            </span>
          )}
        </div>
      </div>
      <p className="mt-3 text-sm text-ink-2" aria-live="polite">
        {ctxInfo && fromCtx ? (
          <>
            В предложении {ctxInfo.phrase} слои attention подмешивают к вектору «ключа» информацию от слов, которые стоят перед ним, и вектор
            сдвигается к словам {quoteList(near)}. Сама строка таблицы при этом не меняется. Смотреть можно только назад: во фразе «ключ от
            двери» вектор самого «ключа» о двери ещё не знает, этот смысл соберут следующие токены.
          </>
        ) : (
          <>
            Ближайшие соседи слова «{selWord}»: {quoteList(near)}.
            {ctxInfo ? (
              <> Нажмите на «ключ», чтобы снова увидеть его вектор в предложении {ctxInfo.phrase}.</>
            ) : (
              <> У «ключа» несколько смыслов, и его строка в таблице лежит между группами: таблица не знает, о каком ключе речь.</>
            )}
          </>
        )}
      </p>
    </Figure>
  )
}
