import { useDeferredValue, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { kvBudgetBytes, kvBytesPerToken, loadVsRecompute } from '@/engine/perf'
import { useSettings } from '@/state/settings'
import { logDomain } from '@/ui/chart-utils'
import { Figure, LineChart, Segmented, Select, Slider, Stat, Toggle, cn, cssVar, fmt, seqColor, seqInk } from '@/ui'
import { BackToSim, useFollowSim } from '../kv-cache/followSim'
import { gpuOr, logGrid, minGpus, modelOptions, modelOr, pickModelId, shortGpuName } from '../kv-cache/lib'
import { NET_CHANNELS, TIERS, findCrossover, gpuHostLink, tierBandwidth, type NetChannelId, type TierId } from './tiers'

// все модели симулятора здесь есть: фигура следует за его выбором, пока читатель не выберет свою
const TIER_MODELS = ['qwen3-8b', 'qwen3-32b', 'qwen3-30b-a3b', 'qwen3-235b-a22b', 'llama-3.1-8b', 'llama-3.1-70b'] as const
const CHART_H = 240
const SHADE: Record<TierId, number> = { g1: 1, g2: 5 / 7, g3: 3 / 7, g4: 1 / 7 }
const WIDTH: Record<TierId, string> = { g1: '40%', g2: '60%', g3: '80%', g4: '100%' }
const SHORT: Record<TierId, string> = { g1: 'HBM GPU', g2: 'RAM хоста', g3: 'локальный NVMe', g4: 'сеть, общее хранилище' }

export function TierFigure() {
  const [settings] = useSettings()
  const simModelId = pickModelId(settings.modelId, TIER_MODELS)
  const modelSel = useFollowSim(simModelId)
  const modelId = modelSel.id
  const [tier, setTier] = useState<TierId>('g2')
  const [channel, setChannel] = useState<NetChannelId>('rdma')
  const [lanes8, setLanes8] = useState(false)
  const [bwOverride, setBwOverride] = useState<number | null>(null)
  const [n, setN] = useState(32768)
  const model = modelOr(modelId)
  const gpu = gpuOr(settings.gpuId)
  const gpus = useMemo(() => minGpus(model, gpu), [model, gpu])
  // PCIe одного GPU: A100 на Gen4 (~25 ГБ/с), Hopper и Blackwell на Gen5 (~50 ГБ/с)
  const host = gpuHostLink(gpu)
  // «все 8 каналов» имеют смысл для дисков и сетевых карт; RAM и так считается по линку на каждый GPU реплики,
  // а у S3 полоса задана целиком
  const lanesApply = tier === 'g3' || (tier === 'g4' && channel !== 's3')
  const lanes = tier === 'g2' ? gpus : lanesApply && lanes8 ? 8 : 1
  const def = tierBandwidth(tier, lanes, channel, { hostGBs: host.practicalGBs, gpus })
  const GBs = bwOverride ?? def.GBs
  const latencyMs = def.latencyMs
  const dGBs = useDeferredValue(GBs)
  const perToken = kvBytesPerToken(model)
  const isL1 = tier === 'g1'

  const reset = () => setBwOverride(null)
  const pick = (t: TierId) => {
    setTier(t)
    reset()
  }
  // radiogroup по правилам: Tab попадает только в выбранный уровень, стрелки переключают уровни
  const radioRefs = useRef<Array<HTMLButtonElement | null>>([])
  const onRadioKey = (e: KeyboardEvent<HTMLButtonElement>) => {
    const ids = TIERS.map((t) => t.id)
    const i = ids.indexOf(tier)
    let ni = -1
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') ni = (i + 1) % ids.length
    else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') ni = (i - 1 + ids.length) % ids.length
    else if (e.key === 'Home') ni = 0
    else if (e.key === 'End') ni = ids.length - 1
    if (ni < 0) return
    e.preventDefault()
    pick(ids[ni])
    radioRefs.current[ni]?.focus()
  }

  const series = useMemo(() => {
    const xs = logGrid(256, 131072, 40)
    const rec: Array<[number, number]> = []
    const load: Array<[number, number]> = []
    for (const x of xs) {
      const r = loadVsRecompute(model, gpu, x, dGBs, { gpus, latencyMs })
      rec.push([x, r.recomputeMs])
      load.push([x, r.loadMs])
    }
    const out = [{ id: 'rec', label: 'Пересчитать prefill', color: 'prefill', points: rec }]
    if (!isL1) out.push({ id: 'load', label: `Загрузить: ${SHORT[tier]}`, color: 'var(--ink-2)', points: load })
    return out
  }, [model, gpu, gpus, dGBs, latencyMs, isL1, tier])

  const cross = useMemo(() => (isL1 ? null : findCrossover(model, gpu, dGBs, latencyMs, gpus)), [model, gpu, dGBs, latencyMs, gpus, isL1])
  const at = loadVsRecompute(model, gpu, n, GBs, { gpus, latencyMs })
  const loadMs = isL1 ? 0 : at.loadMs
  const faster = loadMs < at.recomputeMs
  const ratio = faster ? at.recomputeMs / Math.max(loadMs, 1e-6) : loadMs / at.recomputeMs
  const gpuLabel = `${gpus > 1 ? `${gpus} × ` : ''}${shortGpuName(gpu)}`
  // У паритета точки пересчёта и загрузки почти совпадают по высоте, и две подписи над ними ложатся одна на другую.
  // Тогда подпись нижней точки ставим под ней (домен оси как в LineChart: logDomain по всем значениям).
  const lowerIsLoad = loadMs <= at.recomputeMs
  const labelsClash = useMemo(() => {
    if (isL1 || !(loadMs > 0) || !(at.recomputeMs > 0)) return false
    const ys = [...series.flatMap((s) => s.points.map((p) => p[1])), at.recomputeMs, loadMs].filter((v) => Number.isFinite(v) && v > 0)
    const [lo, hi] = logDomain(Math.min(...ys), Math.max(...ys))
    return (CHART_H * Math.abs(Math.log(at.recomputeMs / loadMs))) / Math.log(hi / lo) < 16
  }, [series, at.recomputeMs, loadMs, isL1])

  // ёмкость уровней на ноде из 8 GPU: HBM под KV считаем по выбранной GPU, остальное по DGX H100 (KV_TIERS)
  const replicas = Math.max(1, Math.floor(8 / gpus))
  const hbmKv = Math.max(0, kvBudgetBytes(model, gpu, { gpus, memUtil: 0.92 })) * replicas
  const capacity = (t: TierId) => (t === 'g1' ? hbmKv : TIERS.find((x) => x.id === t)!.nodeCapacityBytes)
  const bwText = (t: TierId) => {
    if (t === 'g1') return fmt.bandwidth(gpu.bwTBs * 1000)
    if (t === 'g2') return `≈ ${fmt.bandwidth(host.practicalGBs)} на GPU (${fmt.num(host.GBs)} в теории)`
    const b = tierBandwidth(t, 1, t === 'g4' ? channel : 'rdma')
    return `${fmt.bandwidth(b.GBs)}${t === 'g3' ? ' на диск' : t === 'g4' && channel !== 's3' ? ' на карту' : ''}`
  }

  const crossText = isL1
    ? 'KV уже в HBM: грузить нечего, пересчитывается только новый хвост.'
    : cross?.tokens
      ? `Паритет около ${fmt.compact(cross.tokens)} токенов: короче этого выгоднее ${cross.shortWinner === 'load' ? 'загрузить' : 'пересчитать'}, длиннее ${cross.shortWinner === 'load' ? 'пересчитать' : 'загрузить'}.`
      : cross?.shortWinner === 'load'
        ? 'На всех длинах от 256 до 128 тысяч токенов загрузить быстрее, чем пересчитать.'
        : 'На всех длинах до 128 тысяч токенов пересчитать быстрее: канал слишком медленный.'

  return (
    <Figure
      id="kv-tiers-figure-pyramid"
      title="Лестница памяти: загрузить KV или пересчитать prefill"
      width="wide"
      controls={
        <>
          <Select
            label="Модель"
            value={modelId}
            onChange={modelSel.set}
            options={modelOptions(TIER_MODELS)}
            hint={
              modelSel.overridden ? (
                <>
                  на {gpuLabel} (GPU из симулятора)
                  <br />
                  <BackToSim what="модель" name={modelOr(simModelId).name} onClick={modelSel.reset} />
                </>
              ) : (
                `на ${gpuLabel}, модель и GPU из симулятора`
              )
            }
          />
          <Slider
            label="Длина префикса"
            value={n}
            min={256}
            max={131072}
            scale="log"
            onChange={(v) => setN(Math.round(v))}
            format={(v) => fmt.tokens(Math.round(v))}
          />
          <Slider
            label="Полоса уровня"
            value={GBs}
            min={0.3}
            max={3350}
            step={0.01}
            scale="log"
            disabled={isL1}
            onChange={(v) => setBwOverride(v)}
            format={(v) => fmt.bandwidth(v)}
            hint={
              bwOverride === null ? (
                'по умолчанию для уровня'
              ) : (
                <button type="button" className="text-accent underline-offset-2 hover:underline" onClick={reset}>
                  вернуть {fmt.bandwidth(def.GBs)}
                </button>
              )
            }
          />
          <Toggle
            label="Все 8 каналов ноды"
            checked={lanes8 && lanesApply}
            disabled={!lanesApply}
            onChange={(v) => {
              setLanes8(v)
              reset()
            }}
            hint={`8 дисков в RAID или 8 сетевых карт. Потолок задаёт PCIe: в каждый GPU реплики его часть KV входит не быстрее ≈\u00a0${fmt.bandwidth(host.practicalGBs)}`}
          />
        </>
      }
      caption="Попробуйте: выберите сетевой уровень и канал TCP (25 Гбит/с). Появится точка паритета: на коротком контексте пересчитать быстрее, на длинном выгоднее загрузить. Пересчёт растёт быстрее длины, потому что attention квадратичен, а объём KV растёт линейно."
    >
      <div className="grid min-w-0 gap-6 @3xl:grid-cols-[minmax(0,300px)_minmax(0,1fr)]">
        <div className="grid min-w-0 content-start gap-3">
          <div role="radiogroup" aria-label="Уровень памяти" className="grid gap-2.5">
            {TIERS.map((t, ti) => {
              const sel = tier === t.id
              const shade = SHADE[t.id]
              const cap = capacity(t.id)
              return (
                <button
                  key={t.id}
                  ref={(el) => {
                    radioRefs.current[ti] = el
                  }}
                  type="button"
                  role="radio"
                  aria-checked={sel}
                  tabIndex={sel ? 0 : -1}
                  onClick={() => pick(t.id)}
                  onKeyDown={onRadioKey}
                  className="group grid justify-items-center gap-1 rounded-control px-1 py-1 focus-visible:outline-2 focus-visible:outline-accent"
                >
                  <span
                    className={cn('flex h-9 items-center justify-center rounded-[6px] px-2 text-[13px] font-semibold whitespace-nowrap transition-shadow', sel && 'ring-2 ring-accent ring-offset-2 ring-offset-surface')}
                    style={{ width: WIDTH[t.id], background: seqColor(shade), color: seqInk(shade) }}
                  >
                    {t.level.replace(' / ', ' · ')}
                  </span>
                  <span className="text-[13px] leading-tight text-ink">{SHORT[t.id]}</span>
                  <span className="text-xs leading-tight text-ink-2 tabular-nums">
                    {t.id === 'g1' ? `${fmt.bytes(cap)} под KV` : t.id === 'g4' ? 'десятки ТБ и больше' : fmt.bytes(cap)} · {bwText(t.id)}
                  </span>
                  <span className="text-xs leading-tight text-ink-3">
                    {t.id === 'g4' ? 'общее для всего кластера' : `≈ ${fmt.compact(cap / perToken)} токенов ${model.name}`}
                  </span>
                </button>
              )
            })}
          </div>
          {tier === 'g4' && (
            <Segmented
              label="Канал до хранилища"
              size="sm"
              value={channel}
              onChange={(v) => {
                setChannel(v)
                reset()
              }}
              options={NET_CHANNELS.map((c) => ({ value: c.id, label: c.label }))}
            />
          )}
          {tier === 'g4' && <p className="text-xs leading-snug text-ink-2">{NET_CHANNELS.find((c) => c.id === channel)?.note}</p>}
          <p className="text-xs leading-snug text-ink-3">
            Ширина ступеней условная, ёмкости подписаны. Ёмкости для ноды класса DGX H100 из 8 GPU: 2 ТБ RAM, 8 дисков по 3,84 ТБ.
            Скорость диска взята для NVMe Gen5, до 14 ГБ/с; в первых DGX H100 стояли диски Gen4, вдвое медленнее. HBM под KV посчитана для
            выбранной модели и GPU.
          </p>
        </div>

        <div className="grid min-w-0 content-start gap-4">
          <LineChart
            xLabel="Длина префикса, токенов"
            yLabel="Время, мс"
            xScale="log"
            yScale="log"
            height={CHART_H}
            series={series}
            directLabels={false}
            annotations={[
              ...(cross?.tokens ? [{ kind: 'x' as const, x: cross.tokens, label: `паритет ≈ ${fmt.compact(cross.tokens)}` }] : []),
              {
                kind: 'point' as const,
                x: n,
                y: at.recomputeMs,
                label: fmt.ms(at.recomputeMs),
                color: 'prefill',
                labelPosition: labelsClash && !lowerIsLoad ? ('below' as const) : ('above' as const),
              },
              ...(isL1
                ? []
                : [
                    {
                      kind: 'point' as const,
                      x: n,
                      y: loadMs,
                      label: fmt.ms(loadMs),
                      color: 'var(--ink-2)',
                      labelPosition: labelsClash && lowerIsLoad ? ('below' as const) : ('above' as const),
                    },
                  ]),
            ]}
            xFormat={(v) => fmt.compact(v)}
            yFormat={(v) => fmt.ms(v)}
            aria-label="Время загрузки KV и время пересчёта prefill в зависимости от длины префикса"
          />
          <p className="text-sm text-ink-2" role="status">
            {crossText}
          </p>
          <div className="grid grid-cols-2 gap-x-6 gap-y-5 @2xl:grid-cols-4">
            <Stat label="KV префикса" value={fmt.bytes(at.kvBytes)} hint={`${fmt.tokens(n)}, BF16`} />
            <Stat
              label={isL1 ? 'Загрузка' : `Загрузка, ${fmt.bandwidth(GBs)}`}
              value={isL1 ? '0 мс' : fmt.ms(loadMs)}
              hint={
                latencyMs
                  ? `с задержкой доступа ${fmt.ms(latencyMs)}`
                  : isL1
                    ? 'уже на месте'
                    : bwOverride === null && def.capped
                      ? 'упирается в PCIe GPU'
                      : 'объём / полоса'
              }
            />
            <Stat label="Пересчёт prefill" value={fmt.ms(at.recomputeMs)} swatch={cssVar('prefill')} hint={`MFU 50 %, ${gpuLabel}`} />
            <Stat
              label={isL1 ? 'Загрузка не нужна' : faster ? 'Быстрее загрузить' : 'Быстрее пересчитать'}
              value={isL1 ? 'уже в HBM' : fmt.multiplier(ratio, ratio < 10 ? 1 : 0)}
              hint={isL1 ? undefined : `загрузка выгодна от ${fmt.bandwidth(at.breakEvenGBs)}`}
            />
          </div>
        </div>
      </div>
    </Figure>
  )
}
