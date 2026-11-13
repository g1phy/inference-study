import { Fragment } from 'react'
import { ArrowDownIcon, ArrowRightIcon } from '@phosphor-icons/react'
import { BarChart, Figure, fmt, useElementSize } from '@/ui'
import { cn } from '@/ui/cn'
import { useReaderPromptTokenCount } from '@/state/prompt'
import { prefillFlopsBreakdown } from '@/engine/perf'
import { type Arch, attentionCrossover, modelFlops } from './arch'
import { ArchControls } from './ArchControls'

export function ModelFigure({
  arch,
  ctx,
  onCtx,
  modelId,
  onModel,
}: {
  arch: Arch
  ctx: number
  onCtx: (n: number) => void
  modelId: string
  onModel: (id: string) => void
}) {
  const m = arch.model
  const L = m.layers
  const d = m.hidden
  const total = arch.totalParams
  // на узком экране у столбиков только доли, абсолютные числа в тултипе
  const [chartsRef, { width: chartsW }] = useElementSize<HTMLDivElement>()
  const perChart = chartsW >= 768 ? (chartsW - 24) / 2 : chartsW
  const compact = perChart > 0 && perChart < 400

  const paramRows = arch.moe && m.experts && m.moeIntermediate
    ? [
        { label: 'Эксперты', value: L * m.experts.total * 3 * d * m.moeIntermediate },
        { label: 'Attention', value: L * arch.attn.params },
        { label: 'Эмбеддинги', value: arch.embed.params },
        { label: 'LM head', value: arch.head.params },
        { label: 'Роутеры', value: L * d * m.experts.total },
        { label: 'RMSNorm', value: L * 2 * d + d },
      ]
    : [
        { label: 'MLP', value: L * arch.mlp.params },
        { label: 'Attention', value: L * arch.attn.params },
        { label: 'Эмбеддинги', value: arch.embed.params },
        { label: 'LM head', value: arch.head.params },
        { label: 'RMSNorm', value: L * 2 * d + d },
      ]

  const flopsTotal = modelFlops(arch, ctx)
  const flopRows = [
    { label: arch.moe ? 'Эксперты и роутер' : 'MLP', value: L * arch.mlp.flopsFixed },
    { label: 'Проекции Q, K, V, O', value: L * arch.attn.flopsFixed },
    { label: 'Q·Kᵀ и сумма V', value: L * arch.attn.flopsPerCtx * ctx },
    { label: 'LM head', value: arch.head.flopsFixed },
    { label: 'Нормы, сложения', value: L * (2 * arch.norm.flopsFixed + 2 * arch.add.flopsFixed) + arch.finalNorm.flopsFixed },
    { label: 'Эмбеддинги', value: 0, note: 'чтение строки, без умножений' },
  ]
  const cross = attentionCrossover(arch)
  // длина промпта из симулятора после шаблона Qwen3 (с описанием инструмента, если tools включены), как в hero
  const promptTokens = useReaderPromptTokenCount()
  const prefill = promptTokens ? prefillFlopsBreakdown(m, promptTokens).total : 0
  // следующий токен после промпта смотрит на весь промпт и на себя
  const nextDecode = promptTokens ? modelFlops(arch, promptTokens + 1) : 0

  const stages = [
    { title: 'ID токенов', sub: 'N целых чисел' },
    { title: 'Эмбеддинги', sub: `${fmt.int(m.vocab)} × ${fmt.int(d)}` },
    { title: `Блок × ${L}`, sub: arch.moe ? 'attention + MoE' : 'attention + MLP', accent: true },
    { title: 'RMSNorm', sub: 'финальный' },
    { title: 'LM head', sub: `${fmt.int(d)} × ${fmt.int(m.vocab)}` },
    { title: 'Логиты', sub: `${fmt.int(m.vocab)} чисел` },
  ]

  return (
    <Figure
      id="transformer-figure-model"
      title={`Вся модель ${m.name}: где живут параметры и куда уходят FLOPs`}
      caption="Попробуйте: сравните плотную модель и MoE, затем увеличьте контекст до десятков тысяч токенов и посмотрите, как растёт доля attention. FLOPs посчитаны для одного токена decode: 2 FLOP на каждый параметр матрицы плюс attention по контексту."
    >
      <div className="grid min-w-0 gap-6">
        <div className="grid gap-x-6 gap-y-4 @xl:grid-cols-2">
          <ArchControls modelId={modelId} onModel={onModel} ctx={ctx} onCtx={onCtx} maxCtx={m.maxContext} />
        </div>
        <ol className="grid gap-1 @2xl:flex @2xl:items-stretch @2xl:gap-1.5" aria-label="Путь токена через модель">
          {stages.map((s, i) => (
            <Fragment key={s.title}>
              <li
                className={cn(
                  'flex items-baseline justify-between gap-3 rounded-control px-3 py-2 @2xl:grid @2xl:min-w-[96px] @2xl:flex-1 @2xl:content-center @2xl:justify-normal @2xl:gap-0 @2xl:text-center',
                  s.accent ? 'bg-surface-3 shadow-[inset_0_0_0_1.5px_var(--line-strong)]' : 'bg-surface-2 shadow-[inset_0_0_0_1px_var(--line)]',
                )}
              >
                <span className="text-[14px] leading-snug font-semibold text-ink">{s.title}</span>
                <span className="text-xs leading-snug text-ink-2 tabular-nums">{s.sub}</span>
              </li>
              {i < stages.length - 1 && (
                <li aria-hidden className="flex items-center justify-center text-ink-3">
                  <ArrowDownIcon size={14} className="@2xl:hidden" />
                  <ArrowRightIcon size={16} className="hidden @2xl:block" />
                </li>
              )}
            </Fragment>
          ))}
        </ol>

        <div ref={chartsRef} className="grid min-w-0 gap-6 @3xl:grid-cols-2">
          <div className="grid min-w-0 content-start gap-2">
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-sm font-medium text-ink">Параметры модели: всё лежит в памяти</span>
              <span className="text-xs text-ink-3 tabular-nums">всего {fmt.compact(total)}</span>
            </div>
            <BarChart
              data={paramRows.map((r) => ({ ...r, id: r.label, note: compact ? `${fmt.compact(r.value)} параметров` : undefined }))}
              valueFormat={(v) => (v === 0 ? '0' : compact ? fmt.pct(v / total) : `${fmt.compact(v)} · ${fmt.pct(v / total)}`)}
              color="var(--ink-3)"
              aria-label="Параметры по частям модели"
            />
          </div>
          <div className="grid min-w-0 content-start gap-2">
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-sm font-medium text-ink">FLOPs на один токен</span>
              <span className="text-xs text-ink-3 tabular-nums">
                всего {fmt.flops(flopsTotal)} при контексте {fmt.int(ctx)}
              </span>
            </div>
            <BarChart
              data={flopRows.map((r) => ({ ...r, id: r.label, note: r.note ?? (compact ? fmt.flops(r.value) : undefined) }))}
              valueFormat={(v) => (v === 0 ? '0' : compact ? fmt.pct(v / flopsTotal) : `${fmt.flops(v)} · ${fmt.pct(v / flopsTotal)}`)}
              color="var(--ink-3)"
              aria-label="FLOPs на токен по частям модели"
            />
          </div>
        </div>
        <p className="max-w-[72ch] text-sm leading-relaxed text-ink-2">
          Таблица эмбеддингов занимает {fmt.pct(arch.embed.params / total)} параметров, но FLOPs не требует: это чтение строки. LM head того же
          размера, но это настоящее умножение: {fmt.flops(arch.head.flopsFixed)} на токен. Attention по контексту растёт с длиной: у {m.name}{' '}
          после ≈ {fmt.compact(cross)} токенов он требует больше FLOPs, чем все матрицы весов вместе.
        </p>
        {promptTokens !== null && promptTokens > 0 && (
          <p className="max-w-[72ch] rounded-control bg-surface-2 px-3 py-2 text-sm leading-relaxed text-ink-2">
            <span className="font-medium text-ink">Ваш запрос из симулятора:</span> {fmt.tokens(promptTokens)} после шаблона Qwen3. Prefill для
            него ≈ {fmt.flops(prefill)}: все токены промпта за один проход. Каждый следующий шаг decode ≈ {fmt.flops(nextDecode)} ради одного
            нового токена.
          </p>
        )}
      </div>
    </Figure>
  )
}
