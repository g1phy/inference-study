import { useMemo, useState } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import {
  BarChart,
  Figure,
  LineChart,
  Player,
  Select,
  Slider,
  Stat,
  TokenChip,
  TokenStrip,
  TokenStripSkeleton,
  fmt,
  seqColor,
  seqInk,
  usePlayer,
} from '@/ui'
import { cn } from '@/ui/cn'
import { useSettings } from '@/state/settings'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { getModel } from '@/engine/data/models'
import { moeExpertsTouched, stepWeightBytes, weightBytes } from '@/engine/perf'
import { tokenLabel } from '../embeddings/tokenText'
import { routeToken, routerOf, touchedExperts, type RouterSpec } from './moe'

const MOE_IDS = ['qwen3-30b-a3b', 'qwen3-235b-a22b', 'gpt-oss-120b', 'deepseek-v3'] as const

/** «после 1 токена», «после 5 токенов» (родительный падеж). */
const afterTokens = (n: number) => `после ${fmt.int(n)} ${n % 10 === 1 && n % 100 !== 11 ? 'токена' : 'токенов'}`
/** «при 1 токене», «при 5 токенах» (предложный падеж). */
const atTokens = (n: number) => `при ${fmt.int(n)} ${n % 10 === 1 && n % 100 !== 11 ? 'токене' : 'токенах'}`
const MAX_TOKENS = 24
const FALLBACK_TEXT = 'Объясни простыми словами, что такое KV cache и зачем он нужен.'

/** Как роутер этой модели превращает оценки в выбор: строка под лентой. */
function routerSteps(r: RouterSpec, E: number, k: number): string {
  if (r.scheme === 'topk-softmax') return `top-${k} → softmax по выбранным ${k}`
  if (r.scheme === 'sigmoid-group' && r.groups && r.topGroups) {
    return `sigmoid → ${r.topGroups} лучшие группы из ${r.groups} (по ${E / r.groups} ${fmt.plural(E / r.groups, ['эксперту', 'эксперта', 'экспертов'])}) → top-${k} внутри них, веса делятся на свою сумму`
  }
  return `softmax по ${fmt.int(E)} → top-${k} → веса делятся на свою сумму`
}

export function MoeFigure() {
  const [settings] = useSettings()
  const { tokenizer } = useTokenizer()
  const reduce = useReducedMotion()
  const [moeId, setMoeId] = useState<string>('qwen3-30b-a3b')
  const m = getModel(moeId)
  const router = routerOf(moeId)
  const E = m.experts?.total ?? 128
  const k = m.experts?.active ?? 8
  const shared = m.sharedExperts ?? 0
  const [layerRaw, setLayer] = useState(1)
  const layer = Math.max(router.firstMoeLayer, Math.min(layerRaw, m.layers))
  const [hoverExpert, setHoverExpert] = useState<number | null>(null)

  const encoded = useMemo(() => {
    if (!tokenizer) return null
    const text = settings.userText.trim() ? settings.userText : FALLBACK_TEXT
    return tokenizer.encode(text)
  }, [tokenizer, settings.userText])
  const tokens = useMemo(() => (encoded ? encoded.slice(0, MAX_TOKENS) : null), [encoded])
  const n = tokens?.length ?? MAX_TOKENS
  const p = usePlayer(Math.max(1, n), { intervalMs: 1300 })
  const step = Math.min(p.step, n - 1)
  const token = tokens?.[step]

  const route = useMemo(() => (token ? routeToken(token.id, layer, E, k, router) : null), [token, layer, E, k, router])
  const touched = useMemo(
    () => (tokens ? touchedExperts(tokens.map((t) => t.id), step, layer, E, k, router) : new Set<number>()),
    [tokens, step, layer, E, k, router],
  )
  const chosenMap = useMemo(() => new Map(route?.chosen.map((c, rank) => [c.expert, { ...c, rank }]) ?? []), [route])
  const wMax = route?.chosen[0]?.weight ?? 1
  const cols = E <= 64 ? 8 : 16
  const processed = step + 1
  const grouped = router.scheme === 'sigmoid-group' && router.groups ? router.groups : 0
  const groupSize = grouped ? E / grouped : E

  const curve = useMemo(() => {
    const xs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    return [{ id: 'touched', label: 'задето экспертов (ожидание)', color: 'var(--ink-2)', points: xs.map((x) => [x, moeExpertsTouched(m, x)] as [number, number]) }]
  }, [m])

  // gpt-oss хранит экспертов в MXFP4: считаем в нативном формате, остальные модели в BF16
  const dtype = m.native?.dtype ?? 'bf16'
  const fmtLabel = m.native ? 'в MXFP4 + BF16' : 'в BF16'
  const total = weightBytes(m, dtype)
  const perToken = stepWeightBytes(m, dtype, 1)
  const perStep = stepWeightBytes(m, dtype, processed)
  const expectedNow = moeExpertsTouched(m, processed)
  const allCount = encoded?.length ?? 0

  const cell = (e: number) => {
    const c = chosenMap.get(e)
    const was = touched.has(e)
    const hot = hoverExpert === e
    // ровные ступени шкалы, на которых цифра проходит AA в обеих темах (на смеси ступеней и на 4-й ступени контраста не хватает)
    const ratio = c ? c.weight / wMax : 0
    const t = (ratio > 0.85 ? 7 : ratio > 0.65 ? 6 : ratio > 0.45 ? 5 : 3) / 7
    return (
      <div
        key={e}
        className={cn(
          'relative aspect-square min-w-0 rounded-[3px] bg-surface-2 shadow-[inset_0_0_0_1px_var(--line)]',
          hot && 'outline-2 outline-offset-1 outline-accent',
        )}
      >
        {was && !c && <div className="absolute inset-0 rounded-[3px] bg-surface-3 shadow-[inset_0_0_0_1.5px_var(--ink-3)]" />}
        {c && (
          <motion.div
            key={`${step}-${layer}-${e}`}
            className="absolute inset-0 grid place-items-center rounded-[3px] font-mono text-[12px] leading-none font-semibold"
            initial={reduce ? false : { opacity: 0, scale: 0.4 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: reduce ? 0 : 0.28, delay: reduce ? 0 : c.rank * 0.05 }}
            style={{ background: seqColor(t), color: seqInk(t) }}
          >
            {c.rank + 1}
          </motion.div>
        )}
      </div>
    )
  }

  const grid = (from: number, count: number) => (
    <div className="grid w-full gap-[3px]" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
      {Array.from({ length: count }, (_, i) => cell(from + i))}
    </div>
  )

  return (
    <Figure
      id="transformer-figure-moe"
      title={`MoE: роутер выбирает ${k} ${fmt.plural(k, ['эксперта', 'эксперта', 'экспертов'])} из ${fmt.int(E)} для каждого токена`}
      controls={<Player {...p.bind} labels={tokens?.map((t, i) => `токен ${i + 1}: «${tokenLabel(t)}»`)} aria-label="Токены промпта по очереди" />}
      caption="Попробуйте: запустите проигрывание и следите, сколько экспертов слоя задето к концу промпта. Смените слой: тот же токен выбирает других экспертов. Иллюстрация: выбор псевдослучайный, но детерминированный; размеры моделей и память посчитаны по данным."
    >
      <div className="grid min-w-0 gap-5">
        <div className="grid gap-x-6 gap-y-4 @xl:grid-cols-2">
          <Select
            label="MoE-модель"
            value={moeId}
            onChange={(id) => setMoeId(id)}
            options={MOE_IDS.map((id) => ({ value: id, label: getModel(id).name }))}
          />
          <Slider
            label="Слой"
            value={layer}
            min={router.firstMoeLayer}
            max={m.layers}
            onChange={setLayer}
            format={(v) => `${fmt.int(v)} из ${fmt.int(m.layers)}`}
            hint={
              router.firstMoeLayer > 1
                ? `Первые ${router.firstMoeLayer - 1} слоя у ${m.name} плотные (обычный MLP), MoE начинается с ${router.firstMoeLayer}-го`
                : undefined
            }
          />
        </div>

        <div className="min-w-0">
          <div className="mb-2 text-sm text-ink-2">
            {allCount > MAX_TOKENS
              ? `Ваш запрос, первые ${MAX_TOKENS} токена: в prefill они идут через слой вместе`
              : `Ваш запрос, ${allCount === 1 ? 'один токен' : `все ${fmt.tokens(n)}`}: в prefill они идут через слой вместе`}
          </div>
          {tokens ? (
            <TokenStrip tokens={tokens} variant="prompt" highlightIndex={step} onClickIndex={(i) => p.setStep(i)} label="Токены запроса" />
          ) : (
            <TokenStripSkeleton count={20} />
          )}
        </div>

        <div className="grid min-w-0 gap-6 @3xl:grid-cols-[minmax(0,1fr)_minmax(0,320px)]">
          <div className="grid min-w-0 content-start gap-3">
            <div className="flex flex-wrap items-center gap-2 rounded-control bg-surface-2 px-3 py-2 text-sm text-ink-2">
              {token && <TokenChip token={token} variant="prompt" showId />}
              <span>
                → роутер: x · W<sub>router</sub> ({fmt.int(m.hidden)} × {fmt.int(E)}) → {routerSteps(router, E, k)}
              </span>
            </div>
            <div
              className={cn('grid w-full max-w-[560px]', grouped ? 'gap-[7px]' : 'gap-[3px]')}
              role="img"
              aria-label={`Сетка из ${E} экспертов слоя ${layer}: для текущего токена выбраны ${k}, всего задето ${touched.size}`}
            >
              {grouped
                ? Array.from({ length: grouped }, (_, g) => {
                    const on = route?.groups?.includes(g)
                    return (
                      <div key={g} className={cn('rounded-[5px] p-[2px]', on ? 'shadow-[inset_0_0_0_1.5px_var(--ink-2)]' : 'shadow-[inset_0_0_0_1px_transparent]')}>
                        {grid(g * groupSize, groupSize)}
                      </div>
                    )
                  })
                : grid(0, E)}
            </div>
            {shared > 0 && (
              <div className="flex items-center gap-2 text-sm text-ink-2">
                <span aria-hidden className="h-4 w-8 rounded-[3px]" style={{ background: seqColor(0.6) }} />
                плюс {shared} общий эксперт: работает для каждого токена всегда
              </div>
            )}
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-3">
              <span className="flex items-center gap-1.5">
                <span aria-hidden className="h-3 w-3 rounded-[2px]" style={{ background: seqColor(0.9) }} />
                выбран для текущего токена: цифра = место по весу, насыщеннее = больше вес
              </span>
              <span className="flex items-center gap-1.5">
                <span aria-hidden className="h-3 w-3 rounded-[2px] bg-surface-3 shadow-[inset_0_0_0_1.5px_var(--ink-3)]" />
                уже задет другим токеном
              </span>
              <span className="flex items-center gap-1.5">
                <span aria-hidden className="h-3 w-3 rounded-[2px] bg-surface-2 shadow-[inset_0_0_0_1px_var(--line)]" />
                не задет
              </span>
              {grouped > 0 && (
                <span className="flex items-center gap-1.5">
                  <span aria-hidden className="h-3 w-5 rounded-[3px] shadow-[inset_0_0_0_1.5px_var(--ink-2)]" />
                  группа, которую оставил роутер
                </span>
              )}
            </div>
          </div>

          <div className="grid min-w-0 content-start gap-4">
            <div className="grid gap-2">
              <div className="text-sm font-medium text-ink">Веса выбранных экспертов</div>
              {route && (
                <BarChart
                  data={route.chosen.map((c, rank) => ({ label: `${rank + 1} · эксперт ${c.expert}`, value: c.weight, id: `e${c.expert}` }))}
                  max={Math.max(0.3, wMax)}
                  valueFormat={(v) => fmt.pct(v)}
                  color={seqColor(0.8)}
                  barSize={14}
                  onHover={(i) => setHoverExpert(i === null ? null : (route.chosen[i]?.expert ?? null))}
                  aria-label="Веса выбранных экспертов"
                />
              )}
            </div>
            <Stat
              label="Задето экспертов в слое"
              value={`${touched.size} из ${E}`}
              hint={`${afterTokens(processed)}; при равномерном выборе в среднем ${fmt.num(expectedNow, 1)}`}
            />
          </div>
        </div>

        <div className="grid gap-x-6 gap-y-4 border-t border-line pt-4 @md:grid-cols-2 @2xl:grid-cols-3">
          <Stat label="Параметров в памяти" value={fmt.compact(m.paramsB * 1e9)} hint={`${fmt.bytes(total)} ${fmtLabel}: все эксперты всех слоёв`} />
          <Stat label="Работает на один токен" value={fmt.compact(m.activeParamsB * 1e9)} hint={`параметров; при batch 1 за шаг читается ${fmt.bytes(perToken)}`} />
          <Stat
            label="Читается весов за шаг"
            value={fmt.bytes(perStep)}
            hint={`${atTokens(processed)} в шаге: ${fmt.pct(perStep / total)} всех весов, разные токены зовут разных экспертов`}
          />
        </div>

        <div className="grid min-w-0 gap-2">
          <div className="text-sm font-medium text-ink">Сколько экспертов слоя задето, если в шаге много токенов (равномерный выбор, ожидание)</div>
          <LineChart
            height={170}
            xLabel="Токенов в одном шаге (batch или промпт)"
            yLabel="Экспертов"
            xScale="log"
            series={curve}
            yDomain={[0, E]}
            xFormat={(v) => fmt.int(v)}
            yFormat={(v) => fmt.int(v)}
            markers
            annotations={[
              { kind: 'y', y: E, label: `все ${E}`, labelAlign: 'end' },
              { kind: 'point', x: processed, y: expectedNow },
            ]}
            aria-label="Ожидаемое число задетых экспертов от числа токенов в шаге"
          />
          <div className="flex items-center gap-1.5 text-xs text-ink-3">
            <span aria-hidden className="h-2.5 w-2.5 rounded-full bg-ink" />
            сейчас: {fmt.tokens(processed)} в шаге, в среднем задето {fmt.num(expectedNow, 1)} из {fmt.int(E)}
          </div>
        </div>
      </div>
    </Figure>
  )
}
