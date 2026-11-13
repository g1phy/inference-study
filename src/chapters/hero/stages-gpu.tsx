/**
 * Инспектор симулятора, стадии на GPU: prefill, передача KV (NIXL), decode.
 */
import { Fragment, memo, type ReactNode } from 'react'
import { motion, useReducedMotion } from 'motion/react'
import { ArrowRightIcon } from '@phosphor-icons/react'
import { Button, ChapterLink, CodeBlock, cssVar, fmt, linkClass, Term, tint, TokenChip, tokenSegments, cn } from '@/ui'
import { navigateTo } from '@/state/nav'
import { resolveSimGpu, resolveSimModel, type DecodeStepEx, type FlowRunEx } from '@/engine/flow'
import { moeExpertsTouched } from '@/engine/perf'
import { isLongDoc, LONG_DOC_PROMPT_TOKENS } from '@/engine/scenarios'
import type { FlowSettings, Token } from '@/engine/types'
import { ChannelTag, GroupLabel, KeyVals, SplitBar, Status } from './bits'
import { prettyJson, tokensGen, visibleWs } from './model'

/* ============================================================================
 * Общее: две полосы «чтение памяти / вычисления»
 * ========================================================================== */

/**
 * Сравнение двух времён шага: чтение памяти (цвет decode, memory-bound) и вычисления (цвет prefill, compute-bound).
 * Полосы без серой дорожки, подписи цветом текста; более длинная помечена словом «упор».
 */
function BoundBars({ memoryMs, computeMs, memoryLabel, computeLabel }: { memoryMs: number; computeMs: number; memoryLabel: ReactNode; computeLabel: ReactNode }) {
  const max = Math.max(memoryMs, computeMs, 1e-9)
  const bound = memoryMs >= computeMs ? 'memory' : 'compute'
  const rows = [
    { key: 'memory', label: memoryLabel, ms: memoryMs, color: cssVar('decode') },
    { key: 'compute', label: computeLabel, ms: computeMs, color: cssVar('prefill') },
  ] as const
  return (
    <div
      role="img"
      aria-label={`Чтение памяти ${fmt.ms(memoryMs)}, вычисления ${fmt.ms(computeMs)}: упор в ${bound === 'memory' ? 'память' : 'вычисления'}`}
      className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-3 gap-y-1.5 text-[12.5px] leading-5"
    >
      {rows.map((r) => (
        <Fragment key={r.key}>
          <span className="text-ink-2">{r.label}</span>
          <span className="flex min-w-0 items-center gap-2">
            <span className="min-w-0 flex-1">
              <span className="block h-2.5 rounded-full" style={{ width: `${Math.max(1.5, (r.ms / max) * 100)}%`, minWidth: 2, background: r.color }} />
            </span>
            <span className="w-[92px] shrink-0 text-ink tabular-nums">
              {fmt.ms(r.ms)}
              {bound === r.key && <span className="ml-1 font-semibold">упор</span>}
            </span>
          </span>
        </Fragment>
      ))}
    </div>
  )
}

/* ============================================================================
 * prefill
 * ========================================================================== */

function LayerCells({ layers, state, durationMs, runKey }: { layers: number; state: 'empty' | 'filling' | 'full'; durationMs: number; runKey: string }) {
  const reduce = useReducedMotion() ?? false
  const per = layers > 0 ? durationMs / layers / 1000 : 0
  return (
    <div role="img" aria-label={`${layers} слоёв модели: ${state === 'full' ? 'все пройдены' : state === 'filling' ? 'промпт проходит слой за слоем' : 'ещё не начато'}`} className="flex flex-wrap gap-[3px]">
      {Array.from({ length: layers }, (_, i) => (
        <motion.span
          key={`${runKey}-${state}-${i}`}
          aria-hidden
          className="h-4 w-2.5 rounded-[2px]"
          style={{ background: cssVar('prefill') }}
          initial={{ opacity: state === 'full' ? 1 : 0.16 }}
          animate={{ opacity: state === 'empty' ? 0.16 : 1 }}
          transition={state === 'filling' && !reduce ? { delay: i * per, duration: 0.12 } : { duration: 0 }}
        />
      ))}
    </div>
  )
}

export function PrefillStage({
  run,
  phase,
  durationMs,
}: {
  run: FlowRunEx
  /** До prefill, идёт сейчас (плеер играет), пройден. */
  phase: 'pending' | 'running' | 'done'
  durationMs: number
}) {
  const p = run.prefill
  const layers = p.layers ?? 0
  const disagg = run.settings.deployment === 'disagg'
  const perTokenKv = p.totalTokens > 0 ? (p.kvTotalBytes ?? 0) / p.totalTokens : 0
  const gpuName = resolveSimGpu(run.gpuId ?? '').gpu.name
  const model = resolveSimModel(run.modelId ?? '').model
  const first = run.decode[0]
  const memoryMs = p.memoryMs ?? 0
  const computeMs = p.computeMs ?? 0
  // порог для всего промпта без кэша: у холодного и тёплого прогона одно число
  const threshold = run.prefillThreshold ?? 0
  const longPrompt = threshold > 0 && p.totalTokens >= threshold
  const experts = model.experts
  // вкладка «Длинный документ» сменит упор, только если её промпт длиннее порога (у MoE порог выше)
  const onLongDoc = isLongDoc(run.settings)
  const docFlips = threshold > 0 && threshold < LONG_DOC_PROMPT_TOKENS
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-5">
      {phase === 'pending' && (
        <p className="rounded-control bg-surface-2 px-3 py-2 text-[13px] leading-snug text-ink-2 shadow-[inset_0_0_0_1px_var(--line)]">
          Плеер ещё не дошёл до prefill: слои заполнятся, когда запрос доберётся до GPU.
        </p>
      )}
      <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
        <GroupLabel aside={`≈ ${fmt.ms(p.layerMs ?? 0)} на слой`}>
          {fmt.int(layers)} {fmt.plural(layers, ['слой', 'слоя', 'слоёв'])}: весь промпт проходит каждый слой за один проход
        </GroupLabel>
        <LayerCells layers={layers} state={phase === 'pending' ? 'empty' : phase === 'running' ? 'filling' : 'full'} durationMs={durationMs * 0.85} runKey={`${run.seed}-${p.totalTokens}`} />
        <p className="text-[12.5px] leading-snug text-ink-2">
          Каждый слой состоит из <Term t="Attention">attention</Term> и <Term t="FFN / MLP">MLP</Term>. В attention токены обмениваются информацией:
          каждый строит три вектора, <Term t="Q / K / V">Q, K и V</Term> (что ищу, по чему меня найти, что отдам). Q сравнивается с K всех предыдущих
          токенов, <Term t="Softmax">softmax</Term> превращает оценки в веса, и токен забирает смесь их V. K и V остаются в KV cache.{' '}
          <button type="button" className={linkClass} onClick={() => navigateTo('attention-figure-heads')}>
            Attention на вашем промпте
          </button>
        </p>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
        <GroupLabel aside={fmt.tokens(p.totalTokens)}>Токены промпта</GroupLabel>
        <SplitBar
          label={`Из кэша ${p.cachedTokens}, посчитано ${p.computedTokens}`}
          parts={[
            { value: p.cachedTokens, color: cssVar('prefill'), hatched: true, label: <>из prefix cache: {fmt.tokens(p.cachedTokens)}</> },
            { value: p.computedTokens, color: cssVar('prefill'), label: <>посчитано заново: {fmt.tokens(p.computedTokens)}</> },
          ]}
        />
        {p.cachedTokens === 0 ? (
          <p className="text-[12.5px] leading-snug text-ink-3">
            Совпавших блоков нет, считается весь промпт. Включите «Тёплый» <Term t="Prefix caching">prefix cache</Term>: часть блоков уже будет лежать на
            воркере.
          </p>
        ) : (
          <p className="text-[12.5px] leading-snug text-ink-3">
            Блоки лежали прямо в памяти GPU выбранного воркера. Если бы их пришлось брать из общего кэша ноды (L3, например DaemonSet LMCache), prefill
            сначала загрузил бы их по сети или PCIe (<ChapterLink to="kv-tiers" />).
          </p>
        )}
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)] gap-4 @2xl:grid-cols-2">
        <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-2">
          <GroupLabel>Время и упор</GroupLabel>
          <KeyVals
            mono={false}
            rows={[
              ['воркер', `${p.workerName ?? p.worker}, ${run.gpus ?? 1}× ${gpuName}`],
              ['время prefill', <span className="font-medium tabular-nums">{fmt.ms(p.ms)}</span>],
            ]}
          />
          <BoundBars
            memoryMs={memoryMs}
            computeMs={computeMs}
            memoryLabel="чтение весов"
            computeLabel={<span className="tabular-nums">вычисления ({fmt.flops(p.flops ?? 0)})</span>}
          />
          <p className="text-[12.5px] leading-snug text-ink-2">
            {p.bound === 'compute' ? (
              <>
                Посчитать нужно {fmt.tokens(p.computedTokens)}: умножений так много, что время задают вычисления GPU, а не чтение весов. Такой prefill{' '}
                <Term t="Compute-bound">упирается в вычисления</Term> (<ChapterLink to="roofline" />).
              </>
            ) : p.cachedTokens > 0 ? (
              <>
                Посчитать нужно всего {fmt.int(p.computedTokens)} из {tokensGen(p.totalTokens)}: остальные взяты из prefix cache. На такую порцию GPU тратит
                меньше времени, чем на чтение весов из <Term t="HBM" />, поэтому <Term t="Memory-bound">упор в память</Term>.{' '}
                {longPrompt ? (
                  <>Без кэша этот промпт упёрся бы в вычисления: кэш сэкономил большую часть времени prefill.</>
                ) : experts ? (
                  <>
                    Без кэша этот промпт тоже упирался бы в чтение весов (порог около {fmt.tokens(threshold)}), поэтому кэш почти не сократил prefill: у
                    MoE даже такая порция задевает большинство экспертов.
                  </>
                ) : (
                  <>
                    Поэтому время prefill почти не изменилось: веса читаются целиком при любом числе токенов.
                    {!onLongDoc && <> Выигрыш от кэша виден на длинных промптах (вкладка «Длинный документ»).</>}
                  </>
                )}
              </>
            ) : (
              <>
                Ваш промпт ({fmt.tokens(p.totalTokens)}) короче порога: умножений мало, и время задаёт чтение весов из <Term t="HBM" />. Это{' '}
                <Term t="Memory-bound">упор в память</Term>. Порог для этой модели и GPU около {fmt.tokens(threshold)}
                {!onLongDoc && docFlips ? (
                  <>
                    : вставьте текст длиннее или откройте вкладку «Длинный документ», и упор сменится на вычисления (<ChapterLink to="roofline" />).
                  </>
                ) : experts ? (
                  <>
                    . У MoE он высокий: токены промпта вместе задевают почти всех экспертов, и читать приходится почти все веса, а считать только по{' '}
                    {experts.active} экспертам на токен. Даже регламент из вкладки «Длинный документ» до порога не дотягивает, упор в вычисления на нём видно
                    на Qwen3-8B и Qwen3-32B (<ChapterLink to="roofline" />).
                  </>
                ) : (
                  <>
                    : только промпт длиннее переведёт prefill в упор на вычисления (<ChapterLink to="roofline" />).
                  </>
                )}
              </>
            )}
          </p>
          {experts && (
            <p className="text-[12.5px] leading-snug text-ink-3">
              У <Term t="MoE" /> каждый токен идёт к {experts.active} из {experts.total} экспертов, но {p.cachedTokens > 0 ? 'посчитанные токены' : 'токены'}{' '}
              промпта ({fmt.int(p.computedTokens)}) вместе задевают около {fmt.int(Math.round(moeExpertsTouched(model, p.computedTokens)))} экспертов на слой. Шаг decode с batch {run.decode[1]?.batch ?? 1}{' '}
              задевает около {fmt.int(Math.round(moeExpertsTouched(model, run.decode[1]?.batch ?? 1)))} экспертов и читает только часть весов, поэтому у MoE
              prefill заметно дороже шага decode.
            </p>
          )}
        </div>
        <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-2">
          <GroupLabel>Что осталось в KV cache</GroupLabel>
          <KeyVals
            mono={false}
            rows={[
              ['записано', <span className="font-medium tabular-nums">{fmt.bytes(p.kvBytes)}</span>],
              ['весь промпт', <span className="tabular-nums">{fmt.bytes(p.kvTotalBytes ?? 0)}</span>],
              [
                'на токен',
                <span className="tabular-nums">
                  {fmt.bytes(perTokenKv)} ({fmt.bytes(perTokenKv, { mode: 'binary' })})
                </span>,
              ],
            ]}
          />
          <p className="text-[12.5px] leading-snug text-ink-2">
            Для каждого токена и каждого слоя хранятся векторы K и V. Decode будет читать их на каждом шаге вместо того, чтобы пересчитывать промпт
            (<ChapterLink to="kv-cache" />).
          </p>
        </div>
      </div>

      {first && (
        <div className="grid grid-cols-[minmax(0,1fr)] gap-2 rounded-control bg-surface-2 px-3 py-2.5 shadow-[inset_0_0_0_1px_var(--line)]">
          {disagg ? (
            <p className="text-[13px] leading-snug text-ink-2">
              Prefill-воркеру ушёл клон исходного запроса с max_tokens = 1. Он тоже выберет токен, но в связке Dynamo и vLLM этот токен дальше не идёт:
              decode-воркер, получив KV, пересчитает последний токен промпта и выберет первый токен сам.
            </p>
          ) : (
            <div className="flex flex-wrap items-center gap-2 text-[13px] leading-snug text-ink-2">
              <span>
                В конце prefill готовы <Term t="Logits">логиты</Term> последней позиции промпта, из них выбран первый токен ответа:
              </span>
              <TokenChip token={first.token} variant="generated" size="sm" />
              <span className="tabular-nums text-ink-3">{fmt.pct(first.candidates[first.chosenIndex ?? 0]?.prob ?? 0, 1)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ============================================================================
 * transfer (NIXL)
 * ========================================================================== */

export const TransferStage = memo(function TransferStage({ run, patch }: { run: FlowRunEx; patch: (p: Partial<FlowSettings>) => void }) {
  const t = run.transfer
  if (!t) {
    return (
      <div className="grid grid-cols-[minmax(0,1fr)] justify-items-start gap-3">
        <p className="text-[14px] leading-relaxed text-ink-2">
          В агрегированном режиме передачи KV нет: prefill и decode идут на одном воркере, и decode читает KV cache там же, где prefill его записал.
        </p>
        <Button size="sm" onClick={() => patch({ deployment: 'disagg' })} iconRight={ArrowRightIcon}>
          Включить дезагрегированный режим
        </Button>
      </div>
    )
  }
  const tokens = run.prefill.totalTokens
  const ratio = run.prefill.ms > 0 ? t.ms / run.prefill.ms : 0
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-5">
      <div className="grid grid-cols-[minmax(0,1fr)] gap-1.5 rounded-control bg-surface-2 px-3 py-2.5 shadow-[inset_0_0_0_1px_var(--line)]">
        <h4 className="text-[13px] leading-5 font-semibold text-ink">Зачем тогда разделять prefill и decode</h4>
        <p className="text-[13px] leading-snug text-ink-2">
          Цена <Term t="Disaggregated serving">дезагрегации</Term> здесь на виду: второй шаг роутера, передача KV и лишний шаг decode, поэтому TTFT выше.
          При низкой нагрузке агрегированный режим выгоднее. Выигрыш появляется под нагрузкой: в агрегированном режиме длинные prefill соседей встают в
          те же шаги, что и decode, и ITL скачет
          {run.neighbor && (
            <>
              {' '}
              (в этом прогоне промпт соседа на {fmt.tokens(run.neighbor.promptTokens)} растянул бы шаг decode до {fmt.ms(run.neighbor.stepMs)} при обычных{' '}
              {fmt.ms(run.neighbor.plainMs)})
            </>
          )}
          ; в дезагрегированном decode-воркеры заняты только генерацией, а пулы prefill и decode масштабируются отдельно (
          <ChapterLink to="disaggregation" />
          ).
        </p>
      </div>
      <div className="grid grid-cols-[minmax(0,1fr)] gap-5 @2xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-3">
          <div className="flex flex-wrap items-center gap-2 text-[14px] text-ink">
            <span className="rounded-control px-2 py-1 font-medium shadow-[inset_0_0_0_1px_var(--line-strong)]">{t.fromName ?? t.from}</span>
            <ArrowRightIcon size={18} aria-hidden className="text-ink-3" />
            <span className="rounded-control px-2 py-1 font-medium shadow-[inset_0_0_0_1px_var(--line-strong)]">{t.toName ?? t.to}</span>
          </div>
          <KeyVals
            mono={false}
            rows={[
              ['размер KV', <span className="font-medium tabular-nums">{fmt.bytes(t.bytes)} ({fmt.tokens(tokens)})</span>],
              ['канал', `${t.link}`],
              ['параллельно', `${t.links ?? 1} × ${fmt.bandwidth(t.GBs ?? 0)} (по карте на GPU)`],
              ['время', <span className="font-medium tabular-nums">{fmt.ms(t.ms)}</span>],
              ['из них настройка', <span className="tabular-nums">{fmt.ms(t.setupMs ?? 0)} (условно)</span>],
            ]}
          />
          <p className="text-[12.5px] leading-snug text-ink-2">
            Передача заняла {fmt.pct(ratio)} от времени prefill. Чем длиннее промпт и больше модель, тем тяжелее KV: на длинных контекстах сеть
            становится заметной частью TTFT. Без <Term t="RDMA" /> перенос откатывается на TCP и идёт в сотни раз медленнее.
          </p>
        </div>
        <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-2">
          <GroupLabel aside="значения условные">Что prefill-воркер вернул роутеру</GroupLabel>
          <CodeBlock lang="json" code={prettyJson({ kv_transfer_params: t.params })} maxHeight={240} copy={false} title="disaggregated_params (vLLM NixlConnector)" />
          <p className="text-[12.5px] leading-snug text-ink-3">По этим адресам decode-воркер сам читает блоки KV через RDMA READ.</p>
        </div>
      </div>
    </div>
  )
})

/* ============================================================================
 * decode
 * ========================================================================== */

/** Имя токена из одних пробельных символов: по глифу «↵» его не узнать. */
function wsName(text: string): string | null {
  if (!text || !/^\s+$/.test(text)) return null
  if (/^\n+$/.test(text)) return text.length === 1 ? 'перевод строки' : `${text.length} ${fmt.plural(text.length, ['перевод', 'перевода', 'переводов'])} строки`
  if (/^ +$/.test(text)) return text.length === 1 ? 'пробел' : `${text.length} ${fmt.plural(text.length, ['пробел', 'пробела', 'пробелов'])}`
  return 'пробелы'
}

/** Токен в тексте: спецтокен литералом, неполный символ байтами, пробелы и переводы строк видимыми. */
function tokenLabel(t: Token): string {
  if (t.special) return t.text
  if (t.text.includes('\uFFFD')) return fmt.hex(t.bytes)
  return visibleWs(t.text)
}

/** Кандидат: моноширинный чип с видимыми пробелами; замаскированный обведён красным пунктиром (без зачёркивания: оно прячет короткие глифы). */
function CandChip({ token, chosen, masked }: { token: Token; chosen: boolean; masked: boolean }) {
  const segs = tokenSegments(token)
  const name = token.special ? null : wsName(token.text)
  return (
    <span className="inline-flex flex-wrap items-baseline gap-x-1.5">
      <span
        className={cn(
          'inline-block rounded-chip px-[5px] font-mono text-[12.5px] leading-[1.6] whitespace-pre',
          token.special ? 'border border-dashed border-special text-ink-2' : masked ? 'border border-dashed text-ink-2 opacity-80' : 'text-ink',
        )}
        style={{
          background: token.special ? 'transparent' : chosen ? tint('decode', 22) : masked ? 'transparent' : 'var(--surface-2)',
          borderColor: masked && !token.special ? cssVar('bad') : undefined,
        }}
      >
        {segs.map((sg, i) => (
          // на подложке выбранного чипа ink-3 не дотягивает до AA (3,6:1): глифы пробелов цветом ink-2
          <span key={i} className={sg.kind === 'ws' ? (chosen ? 'text-ink-2' : 'text-ink-3') : sg.kind === 'hex' ? 'italic' : undefined}>
            {sg.value}
          </span>
        ))}
      </span>
      {/* строка выбранного кандидата на accent-soft: подпись ink-2 */}
      {name && <span className={cn('text-[11.5px] leading-4', chosen ? 'text-ink-2' : 'text-ink-3')}>{name}</span>}
    </span>
  )
}

function pct(p: number): string {
  return fmt.pct(p, p >= 0.1 ? 1 : undefined)
}

/** Вероятность: полоса без серой дорожки и число. */
function ProbBar({ value, muted }: { value: number; muted?: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-full min-w-[48px] max-w-[160px]">
        {value > 0 && (
          <div
            className="h-full rounded-full"
            style={{
              width: `${Math.min(100, value * 100)}%`,
              minWidth: 2,
              background: muted ? 'color-mix(in oklab, var(--ink-3) 55%, transparent)' : cssVar('decode'),
            }}
          />
        )}
      </div>
      <span className="w-[52px] shrink-0 text-right text-[12.5px] tabular-nums text-ink">{pct(value)}</span>
    </div>
  )
}

/** Batch шага словами: «1: только ваш запрос», «3: ваш запрос и ещё 2 соседа». */
function batchText(d: DecodeStepEx): string {
  const b = d.batch ?? 1
  if (d.phase === 'prefill') return '1: отдельный проход prefill вашего промпта'
  if (b <= 1) return '1: только ваш запрос'
  return `${fmt.int(b)}: ваш запрос и ещё ${fmt.int(b - 1)} ${fmt.plural(b - 1, ['сосед', 'соседа', 'соседей'])} по воркеру`
}

export function DecodeStage({ run, k, reached }: { run: FlowRunEx; k: number; reached: boolean }) {
  const d = run.decode[Math.max(0, Math.min(run.decode.length - 1, k))]
  const N = run.decode.length
  const total = run.prefill.totalTokens
  const masked = d.masked
  const kvTokens = d.kvTokens ?? total + d.step
  const generated = d.step
  const entropy = d.entropyBits ?? 0
  const bd = d.breakdown
  const decodeWorkerId = run.transfer?.to ?? run.prefill.worker
  const workerInfo = run.cluster?.workers.find((w) => w.id === decodeWorkerId)
  const capacityTokens = (workerInfo?.capacityBlocks ?? 0) * run.blockSize
  const seqTokens = total + N
  const fitSeqs = seqTokens > 0 ? Math.floor(capacityTokens / seqTokens) : 0
  // (G0-r2-1) на шаге k на вход идёт токен k−1, а выбранный токен k построит свои Q, K и V только на шаге k+1
  const disagg = run.settings.deployment === 'disagg'
  const last = d.step >= N - 1
  const ahead = Math.max(0, N - 1 - generated)
  const chosenText = tokenLabel(d.token)
  const prevText = d.step > 0 ? tokenLabel(run.decode[d.step - 1].token) : ''
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-5">
      {!reached && (
        <p className="rounded-control bg-surface-2 px-3 py-2 text-[13px] leading-snug text-ink-2 shadow-[inset_0_0_0_1px_var(--line)]">
          Плеер ещё не дошёл до генерации: показан первый шаг.
        </p>
      )}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex items-center gap-2">
          <span className="text-[13px] text-ink-2 tabular-nums">
            Токен {fmt.int(d.step + 1)} из {fmt.int(N)}
          </span>
          <TokenChip token={d.token} variant="generated" />
        </div>
        <ChannelTag channel={d.channel} />
        {d.phase === 'prefill' && <span className="text-[12.5px] text-ink-3">выбран в конце prefill</span>}
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)] gap-2">
        <GroupLabel aside={`temperature ${fmt.num(run.settings.temperature, 2)}; top-p и top-k не применяются; логиты условные`}>
          {masked ? 'Кандидаты: что хотела модель и что пропустила маска' : 'Кандидаты на этом шаге'}
        </GroupLabel>
        {masked && d.grammarExpecting && (
          <div className="grid grid-cols-[minmax(0,1fr)] gap-1">
            <p className="text-[13px] leading-snug text-ink-2">
              Грамматика JSON ждёт: <span className="font-medium text-ink">{d.grammarExpecting}</span>
              {(d.maskedCount ?? 0) > 0 && <>. Маска отрезала {d.maskedCount} из {d.candidates.length} показанных кандидатов.</>}
            </p>
            <p className="text-[12.5px] leading-snug text-ink-3">
              Маска здесь в строгом режиме пробелов, как <Term t="XGrammar" /> с any_whitespace=False: после «:» и «,» ровно один пробел, других пробелов
              нет. В vLLM по умолчанию между элементами JSON разрешены любые пробелы и переводы строк; строгий режим включает disable_any_whitespace.
            </p>
          </div>
        )}
        <div className="overflow-x-auto scrollbar-thin">
          <table className="w-full border-collapse text-[13px] leading-5 @lg:min-w-[440px]">
            <thead>
              <tr className="border-b border-line text-left text-[12px] text-ink-3">
                <th scope="col" className="py-1.5 pr-3 font-medium">Токен</th>
                {masked && <th scope="col" className="hidden px-2 py-1.5 font-medium @lg:table-cell">До маски</th>}
                <th scope="col" className="px-2 py-1.5 font-medium">{masked ? 'После маски' : 'Вероятность'}</th>
                <th scope="col" className="hidden py-1.5 pl-2 font-medium @lg:table-cell">
                  <span className="sr-only">Статус</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {d.candidates.map((c, i) => {
                const chosen = i === d.chosenIndex
                const status = chosen ? <Status tone="good">выбран</Status> : !c.allowed ? <Status tone="masked">маска</Status> : null
                return (
                  <tr key={`${c.token.id}-${i}`} className="border-b border-line last:border-b-0" style={chosen ? { background: 'var(--accent-soft)' } : undefined}>
                    <th scope="row" className="py-1.5 pr-3 text-left align-middle font-normal">
                      <CandChip token={c.token} chosen={chosen} masked={!c.allowed} />
                      {(status || masked) && (
                        <div className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[12px] leading-4 text-ink-3 @lg:hidden">
                          {status}
                          {masked && <span className="tabular-nums">до маски {pct(c.rawProb ?? c.prob)}</span>}
                        </div>
                      )}
                      {c.preview && <div className="mt-0.5 max-w-[240px] text-[12px] leading-4 text-ink-3">дальше: «{c.preview}»</div>}
                    </th>
                    {masked && (
                      <td className="hidden px-2 py-1.5 align-middle @lg:table-cell">
                        <ProbBar value={c.rawProb ?? c.prob} muted />
                      </td>
                    )}
                    <td className="px-2 py-1.5 align-middle">
                      <ProbBar value={c.prob} muted={!c.allowed} />
                    </td>
                    <td className="hidden py-1.5 pl-2 align-middle whitespace-nowrap @lg:table-cell">{status}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <p className="text-[12.5px] leading-snug text-ink-3">
          Вероятности посчитаны только среди показанных кандидатов, поэтому они выше, чем дал бы <Term t="Softmax">softmax</Term> по всему словарю:
          настоящий softmax идёт по всем {fmt.int(151936)} <Term t="Logits">логитам</Term>, по одному на строку выходного слоя. Логиты условные. Ответ
          заготовлен, поэтому выбран заскриптованный токен: он самый вероятный среди разрешённых.
        </p>
      </div>

      {d.note && <p className="text-[13.5px] leading-relaxed text-ink">{d.note}</p>}

      <div className="grid grid-cols-[minmax(0,1fr)] gap-4 @2xl:grid-cols-2">
        <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-2">
          <GroupLabel>{d.phase === 'prefill' ? 'Первый токен' : 'Шаг decode'}</GroupLabel>
          <KeyVals
            mono={false}
            rows={[
              [d.phase === 'prefill' ? 'время prefill' : 'время шага', <span className="font-medium tabular-nums">{fmt.ms(d.durationMs)}</span>],
              [
                <Term key="b" t="Batch">
                  batch
                </Term>,
                <span className="tabular-nums">{batchText(d)}</span>,
              ],
              [
                <Term key="e" t="Entropy">
                  энтропия
                </Term>,
                <span>
                  <span className="tabular-nums">{fmt.num(entropy, 2)} бит</span>
                  <span className="text-ink-3">: {entropy < 0.5 ? 'модель почти уверена' : entropy < 1.5 ? 'есть пара сильных вариантов' : 'выбор неочевиден'}</span>
                </span>,
              ],
            ]}
          />
          {bd ? (
            <>
              <GroupLabel className="mt-2">Время и упор</GroupLabel>
              <BoundBars memoryMs={bd.memoryMs} computeMs={bd.computeMs} memoryLabel="чтение весов и KV" computeLabel="вычисления" />
              {d.neighborPrefill ? (
                <p className="text-[12.5px] leading-snug text-ink-2">
                  В этот шаг <Term t="Chunked prefill">chunked prefill</Term> поставил {fmt.tokens(d.neighborPrefill.tokens)} промпта соседа
                  {d.neighborPrefill.tokens < d.neighborPrefill.promptTokens && <> (кусок из {fmt.tokens(d.neighborPrefill.promptTokens)}: бюджет шага {fmt.int(d.neighborPrefill.budget)})</>}
                  . Их умножения задали время шага: {fmt.ms(d.durationMs)} вместо {fmt.ms(d.neighborPrefill.plainMs)}, шаг{' '}
                  <Term t="Compute-bound">упирается в вычисления</Term>, и ваш следующий токен ждёт. Так в агрегированном режиме скачет ITL; в
                  дезагрегированном этот prefill считает prefill-воркер (<ChapterLink to="disaggregation" />). Сосед условный.
                </p>
              ) : (
                <p className="text-[12.5px] leading-snug text-ink-2">
                  За шаг воркер прочитал из памяти {fmt.bytes(bd.weightBytes)} весов и {fmt.bytes(bd.kvReadBytes)} KV cache, а посчитал совсем немного: по
                  одному новому токену на запрос. Время шага задаёт полоса памяти, поэтому decode <Term t="Memory-bound">упирается в память</Term>. Отсюда и выгода batch:
                  веса читаются один раз на всех соседей (<ChapterLink to="roofline" />).
                </p>
              )}
            </>
          ) : (
            <p className="text-[12.5px] leading-snug text-ink-3">Этот токен выбран в конце prefill: из чего сложилось его время, видно на стадии Prefill.</p>
          )}
        </div>
        <div className="grid grid-cols-[minmax(0,1fr)] content-start gap-2">
          <GroupLabel aside={fmt.bytes(d.kvBytes ?? 0)}>KV cache после шага</GroupLabel>
          <SplitBar
            label={`KV cache: ${fmt.tokens(total)} промпта и ${fmt.int(generated)} сгенерированных`}
            parts={[
              { value: total, color: cssVar('prefill'), label: <>промпт {fmt.int(total)}</> },
              { value: generated, color: cssVar('decode'), label: <>ответ {fmt.int(generated)}</> },
              // K и V последнего токена в кэш не попадают: следующего прохода нет
              { value: ahead, color: 'transparent', label: <span className="text-ink-3">впереди {fmt.int(ahead)}</span> },
            ]}
          />
          <p className="text-[12.5px] leading-snug text-ink-3">
            {d.phase === 'prefill' ? (
              <>
                Всего {fmt.tokens(kvTokens)}: это K и V промпта, их записал prefill. Токен «{chosenText}» выбран по логитам последней позиции промпта;{' '}
                {last ? (
                  <>это последний токен ответа, поэтому его K и V в кэш уже не попадут.</>
                ) : (
                  <>свои Q, K и V он построит на следующем шаге, когда пойдёт на вход модели.</>
                )}
              </>
            ) : (
              <>
                {disagg && d.step === 0 ? (
                  <>
                    Всего {fmt.tokens(kvTokens)}: K и V промпта decode-воркер получил через NIXL. На вход этого шага пошёл последний токен промпта:
                    decode-воркер пересчитал его, его Q сравнился с K всех {tokensGen(kvTokens)}, и по логитам выбран первый токен ответа «{chosenText}».
                  </>
                ) : (
                  <>
                    Всего {fmt.tokens(kvTokens)}. На вход шага пошёл предыдущий токен ответа «{prevText}»: он построил свои Q, K и V, его K и V дописались в
                    кэш, а Q сравнился с K всех {tokensGen(kvTokens)} (одна новая строка матрицы внимания). По получившимся логитам выбран
                    токен «{chosenText}».
                  </>
                )}{' '}
                {last ? (
                  <>Это последний шаг: следующего прохода нет, поэтому K и V токена «{chosenText}» в кэш уже не попадут.</>
                ) : (
                  <>Его K и V появятся в кэше на следующем шаге.</>
                )}
              </>
            )}
          </p>
          {capacityTokens > 0 && (
            <p className="text-[12.5px] leading-snug text-ink-3">
              Под KV cache на этом воркере помещается около {tokensGen(capacityTokens)}
              {fitSeqs > 0 && <>: это {fmt.int(fitSeqs)} таких запросов одновременно</>}. Длинные контексты и много пользователей упираются именно в эту
              память (<ChapterLink to="kv-cache" />).
            </p>
          )}
        </div>
      </div>
      <span className="sr-only">Выбран токен {visibleWs(d.token.text)}</span>
    </div>
  )
}
