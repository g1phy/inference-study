import { useMemo } from 'react'
import { LayoutGroup, motion } from 'motion/react'
import { CheckCircleIcon, MagnifyingGlassIcon, XCircleIcon } from '@phosphor-icons/react'
import { getInterconnect } from '@/engine/data/interconnects'
import { kvBytesForSeq, prefillMs, transferMs } from '@/engine/perf'
import { useSettings } from '@/state/settings'
import { Figure, Player, Stat, cn, cssVar, fmt, seqColor, tint, useElementSize, usePlayer } from '@/ui'
import { gpuOr, minGpus, modelOr, shortGpuName } from '../kv-cache/lib'
import { gpuHostLink } from './tiers'

type Tier = 'L1' | 'L2' | 'L3'
type BlockState = 'new' | 'cached' | 'moved' | 'hit' | 'loaded' | 'other'

interface Placed {
  id: string
  tier: Tier
  slot: number
  state: BlockState
  /** Блок переехал на этом шаге: вниз (вытеснение) или вверх (подгрузка). */
  dir?: 'down' | 'up'
}

interface LStep {
  label: string
  text: string
  blocks: Placed[]
  lookup?: Partial<Record<Tier, 'hit' | 'miss'>>
}

/** Каждый нижний уровень больше верхнего, как и должно быть в настоящей иерархии. */
const CAPS: Record<Tier, number> = { L1: 8, L2: 12, L3: 16 }
const TIER_INFO: Record<Tier, { name: string; note: string; shade: number }> = {
  L1: { name: 'L1 · HBM GPU', note: 'быстро, мало', shade: 0.95 },
  L2: { name: 'L2 · RAM хоста', note: 'через PCIe', shade: 0.7 },
  L3: { name: 'L3 · NVMe / сеть', note: 'много, медленнее', shade: 0.47 },
}

const A = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
const BC = ['B0', 'B1', 'B2', 'B3', 'C0', 'C1', 'C2', 'C3']
const X = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

const put = (ids: string[], tier: Tier, state: BlockState, from = 0, dir?: Placed['dir']): Placed[] =>
  ids.map((id, i) => ({ id, tier, slot: from + i, state, dir }))

const STEPS: LStep[] = [
  {
    label: 'Запрос A: prefill считает префикс',
    text: 'Запрос A с длинным системным промптом. Prefill считает 6 блоков KV, они ложатся в HBM (L1).',
    blocks: put(A, 'L1', 'new'),
  },
  {
    label: 'Ответ готов, блоки A остаются как кэш',
    text: 'Запрос закончился, но его блоки не стираются: они лежат в HBM как кэш и пригодятся следующему запросу с тем же началом.',
    blocks: put(A, 'L1', 'cached'),
  },
  {
    label: 'HBM нужна другим: A уезжает в RAM',
    text: 'Приходят запросы B и C, HBM заполнена. Давно не использованные блоки A вытесняются, но не стираются, а копируются в RAM хоста (L2) по PCIe. Здесь показана запись при вытеснении (write-back); в SGLang HiCache и vLLM по умолчанию копия уходит в RAM сразу после prefill (write-through), и вытеснение из HBM уже ничего не копирует.',
    blocks: [...put(BC, 'L1', 'new'), ...put(A, 'L2', 'moved', 0, 'down')],
  },
  {
    label: 'RAM заполняется: хвост A уходит на NVMe',
    text: 'Проходит время, в RAM приезжают блоки других запросов. Хвостовые блоки A уходят ниже, на NVMe (L3): их префикс самый длинный, и совпадает он реже всего.',
    blocks: [
      ...put(BC, 'L1', 'cached'),
      ...put(A.slice(0, 3), 'L2', 'cached'),
      ...put(X, 'L2', 'other', 3),
      ...put(A.slice(3), 'L3', 'moved', 0, 'down'),
    ],
  },
  {
    label: 'Запрос A′ с тем же префиксом: поиск',
    text: 'Через час приходит A′ с тем же системным промптом и новым вопросом. Движок хэширует блоки и ищет их сверху вниз: в HBM нет, три блока в RAM, три на NVMe.',
    blocks: [...put(BC, 'L1', 'cached'), ...put(A.slice(0, 3), 'L2', 'hit'), ...put(X, 'L2', 'other', 3), ...put(A.slice(3), 'L3', 'hit')],
    lookup: { L1: 'miss', L2: 'hit', L3: 'hit' },
  },
  {
    label: 'Подгрузка в HBM, слой за слоем',
    text: 'Блоки поднимаются в HBM: с NVMe через RAM, из RAM по PCIe. Грузится слой за слоем, поэтому GPU может считать слой N, пока едет слой N+1. Место им освобождают блоки B и C, и они тоже не пропадают, а уезжают ниже: сколько поместится в RAM, остальные на NVMe.',
    blocks: [
      ...put(A, 'L1', 'loaded', 0, 'up'),
      ...put(BC.slice(0, 3), 'L2', 'moved', 0, 'down'),
      ...put(X, 'L2', 'other', 3),
      ...put(BC.slice(3), 'L3', 'moved', 0, 'down'),
    ],
  },
  {
    label: 'Prefill только для нового хвоста',
    text: 'Префикс уже в HBM, prefill считает только новый вопрос, и первый токен готов. Сколько это сэкономило, видно по числам ниже.',
    blocks: [
      ...put(A, 'L1', 'loaded'),
      { id: 'A6', tier: 'L1', slot: 6, state: 'new' },
      ...put(BC.slice(0, 3), 'L2', 'cached'),
      ...put(X, 'L2', 'other', 3),
      ...put(BC.slice(3), 'L3', 'cached'),
    ],
  },
]

/** Широкая раскладка: подпись слева, слоты в одну строку. */
const WIDE = { slot: 44, gap: 6 }
/** Узкая (телефон): подпись над дорожкой, слоты 30 px переносятся по строкам. */
const NARROW = { slot: 30, gap: 4 }
const NARROW_BELOW = 480

function blockStyle(b: Placed): { background: string; border: string; text: string; dashed: boolean } {
  const isA = b.id.startsWith('A')
  // чужие блоки приглушены фоном и рамкой, а текст ink-2: ink-3 на surface-3 не дотягивает до AA
  if (b.state === 'other') return { background: 'var(--surface-3)', border: 'var(--line)', text: 'var(--ink-2)', dashed: false }
  if (b.state === 'new') return { background: tint('prefill', isA ? 30 : 16), border: cssVar('prefill'), text: 'var(--ink)', dashed: false }
  if (b.state === 'hit') return { background: tint('good', 22), border: cssVar('good'), text: 'var(--ink)', dashed: false }
  if (b.state === 'loaded') return { background: tint('good', 14), border: cssVar('good'), text: 'var(--ink)', dashed: false }
  // переехал вниз на этом шаге: пунктир акцентом, чтобы и без анимации было видно, что изменилось
  if (b.state === 'moved') return { background: 'var(--surface)', border: 'var(--accent)', text: 'var(--ink)', dashed: true }
  return { background: isA ? 'var(--surface)' : 'var(--surface-2)', border: isA ? 'var(--line-strong)' : 'var(--line)', text: isA ? 'var(--ink)' : 'var(--ink-3)', dashed: false }
}

export function LifecycleFigure() {
  const [settings] = useSettings()
  const p = usePlayer(STEPS.length, { intervalMs: 2600 })
  const step = STEPS[p.step]
  const model = modelOr(settings.modelId)
  const gpu = gpuOr(settings.gpuId)
  const gpus = useMemo(() => minGpus(model, gpu), [model, gpu])
  const [boxRef, box] = useElementSize<HTMLDivElement>()
  const narrow = box.width > 0 && box.width < NARROW_BELOW
  const geo = narrow ? NARROW : WIDE
  const pitch = geo.slot + geo.gap
  // в узкой раскладке слоты переносятся: сколько помещается в строку дорожки
  const cols = narrow ? Math.max(4, Math.floor((box.width - geo.gap) / pitch)) : Infinity
  const pos = (slot: number) =>
    narrow ? { left: geo.gap + (slot % cols) * pitch, top: geo.gap + Math.floor(slot / cols) * pitch } : { left: geo.gap + slot * pitch, top: geo.gap }
  const trackSize = (t: Tier) => {
    const rows = narrow ? Math.ceil(CAPS[t] / cols) : 1
    return { width: narrow ? '100%' : CAPS[t] * pitch + geo.gap, height: rows * pitch + geo.gap }
  }

  // упрощённо: 1 блок на схеме = 1024 токена; новый вопрос 300 токенов
  const nums = useMemo(() => {
    const per = 1024
    const prefix = 6 * per
    const tail = 300
    const full = prefillMs(model, gpu, prefix + tail, { gpus })
    const tailOnly = prefillMs(model, gpu, prefix + tail, { gpus, cachedTokens: prefix })
    const half = kvBytesForSeq(model, 3 * per)
    // в HBM всё входит по PCIe самих GPU: у A100 Gen4 (~25 ГБ/с), у Hopper и Blackwell Gen5 (~50 ГБ/с), по линку на GPU
    const hostGBs = gpuHostLink(gpu).practicalGBs
    const nvme = getInterconnect('nvme5')
    const fromL2 = transferMs(half, hostGBs * gpus)
    const fromL3 = transferMs(half, nvme.GBs) + transferMs(half, hostGBs * gpus)
    const load = Math.max(fromL2, fromL3)
    const ttft = Math.max(load, tailOnly)
    return { per, prefix, tail, full, tailOnly, fromL2, fromL3, ttft, kv: kvBytesForSeq(model, prefix), hostGBs, nvmeGBs: nvme.GBs }
  }, [model, gpu, gpus])

  const tiers: Tier[] = ['L1', 'L2', 'L3']
  const gpuLabel = `${gpus > 1 ? `${gpus} × ` : ''}${shortGpuName(gpu)}`
  const last = p.step === STEPS.length - 1

  return (
    <Figure
      id="kv-tiers-figure-lifecycle"
      title="Жизнь блоков KV на разных уровнях"
      width="wide"
      controls={<Player {...p.bind} labels={STEPS.map((s) => s.label)} aria-label="Шаги жизни блоков" />}
      caption="Попробуйте: пройдите шаги по одному и следите за блоками A. Они не исчезают при вытеснении, а уезжают ниже, а потом возвращаются наверх, и запрос A′ пропускает почти весь prefill."
    >
      <div ref={boxRef} className="grid min-w-0 gap-4">
        <p className="min-h-[5.6em] max-w-[72ch] text-[15px] leading-snug text-ink">{step.text}</p>
        <LayoutGroup>
          <div className={cn('grid min-w-0 gap-2.5 pb-1', !narrow && 'overflow-x-auto')}>
            {tiers.map((t) => {
              const info = TIER_INFO[t]
              const look = step.lookup?.[t]
              const inTier = step.blocks.filter((b) => b.tier === t)
              return (
                <div key={t} className={cn('grid items-center gap-x-3 gap-y-1.5', narrow ? 'grid-cols-1' : 'min-w-max grid-cols-[150px_auto]')}>
                  <div className={cn('gap-x-2 gap-y-0.5', narrow ? 'flex flex-wrap items-baseline' : 'grid')}>
                    <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-ink">
                      <span aria-hidden className="inline-block h-3 w-3 rounded-[3px]" style={{ background: seqColor(info.shade) }} />
                      {info.name}
                    </span>
                    <span className="text-xs text-ink-3">{info.note}</span>
                    {look && (
                      <span className="inline-flex items-center gap-1 text-xs font-medium text-ink">
                        {look === 'hit' ? (
                          <CheckCircleIcon size={16} aria-hidden style={{ color: cssVar('good') }} />
                        ) : (
                          <XCircleIcon size={16} aria-hidden style={{ color: cssVar('bad') }} />
                        )}
                        {look === 'hit' ? 'нашлось' : 'нет'}
                      </span>
                    )}
                  </div>
                  <div
                    className="relative rounded-control border border-line bg-surface-2"
                    style={trackSize(t)}
                    role="list"
                    aria-label={`${info.name}: ${inTier.map((b) => b.id).join(', ') || 'пусто'}`}
                  >
                    {Array.from({ length: CAPS[t] }, (_, i) => (
                      <span
                        key={i}
                        aria-hidden
                        className="absolute rounded-[6px] border border-dashed border-line-strong"
                        style={{ ...pos(i), width: geo.slot, height: geo.slot }}
                      />
                    ))}
                    {inTier.map((b) => {
                      const st = blockStyle(b)
                      return (
                        <motion.span
                          key={b.id}
                          layoutId={b.id}
                          role="listitem"
                          aria-label={`${b.id}${b.dir === 'down' ? ', переехал вниз' : b.dir === 'up' ? ', поднят наверх' : ''}`}
                          transition={{ type: 'spring', stiffness: 170, damping: 24 }}
                          className={cn(
                            'absolute flex items-center justify-center rounded-[6px] border-2 font-mono font-semibold',
                            narrow ? 'text-[10.5px]' : 'text-[12px]',
                            st.dashed && 'border-dashed',
                          )}
                          style={{
                            ...pos(b.slot),
                            width: geo.slot,
                            height: geo.slot,
                            background: st.background,
                            borderColor: st.border,
                            color: st.text,
                          }}
                        >
                          {b.id === 'A6' ? 'нов.' : b.id}
                          {b.dir && (
                            <span
                              aria-hidden
                              className="absolute -top-1.5 -right-1.5 flex h-4 w-4 items-center justify-center rounded-full border border-accent bg-surface font-sans text-[11px] leading-none font-bold text-accent-hover"
                            >
                              {b.dir === 'down' ? '↓' : '↑'}
                            </span>
                          )}
                        </motion.span>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </LayoutGroup>
        <div className="flex flex-wrap gap-x-5 gap-y-1.5 text-[13px] text-ink-2">
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="inline-block h-3 w-3 rounded-[3px] border-2" style={{ borderColor: cssVar('prefill'), background: tint('prefill', 30) }} />
            посчитано в prefill
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="inline-block h-3 w-3 rounded-[3px] border-2" style={{ borderColor: cssVar('good'), background: tint('good', 22) }} />
            найдено в кэше
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="inline-block h-3 w-3 rounded-[3px] border-2 border-line-strong bg-surface" />
            лежит как кэш
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span
              aria-hidden
              className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-accent bg-surface text-[11px] leading-none font-bold text-accent-hover"
            >
              ↓
            </span>
            переехал на этом шаге
          </span>
          <span className="inline-flex items-center gap-1.5">
            <MagnifyingGlassIcon size={16} aria-hidden />
            поиск идёт по хэшам блоков
          </span>
        </div>
        {/* Итог виден на всех шагах: высота фигуры не меняется, и панель плеера не прыгает */}
        <div className="grid gap-3 border-t border-line pt-4">
          <p className="text-sm text-ink-2">
            <span className={cn('font-medium', last ? 'text-ink' : 'text-ink-2')}>Итог для запроса A′ (шаги 5–7).</span> Упрощённо: блок на
            схеме = {fmt.tokens(nums.per)}, префикс {fmt.tokens(nums.prefix)}, новый вопрос {fmt.tokens(nums.tail)}. {model.name} на {gpuLabel};
            RAM по PCIe около {fmt.bandwidth(nums.hostGBs)} на GPU, один NVMe Gen5 {fmt.bandwidth(nums.nvmeGBs)}.
          </p>
          <div className="grid grid-cols-2 gap-x-6 gap-y-5 @3xl:grid-cols-4">
            <Stat label="Полный prefill" value={fmt.ms(nums.full)} swatch={cssVar('prefill')} />
            <Stat label="Загрузка KV" value={fmt.ms(Math.max(nums.fromL2, nums.fromL3))} hint={`${fmt.bytes(nums.kv)} из RAM и NVMe`} />
            <Stat label="Prefill хвоста" value={fmt.ms(nums.tailOnly)} />
            <Stat label="TTFT с иерархией" value={fmt.ms(nums.ttft)} swatch={cssVar('good')} hint="загрузка идёт параллельно со счётом" />
          </div>
        </div>
      </div>
    </Figure>
  )
}
