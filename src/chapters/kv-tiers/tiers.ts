/**
 * Уровни KV cache для главы 14: лестница из движка (KV_TIERS) плюс каналы до сетевого уровня
 * и поиск точки паритета «загрузить KV против пересчитать prefill» (perf.loadVsRecompute).
 */
import type { ModelSpecExt } from '@/engine/data/models'
import type { GpuSpecExt } from '@/engine/data/gpus'
import { KV_TIERS, getInterconnect, type KvTier } from '@/engine/data/interconnects'
import { loadVsRecompute } from '@/engine/perf'

export type TierId = KvTier['id']

/**
 * Каналы до сетевого уровня (G4). Объектное хранилище: ~1 ГБ/с (LMCache с S3 Express) и ~100 мс на доступ
 * (Dynamo KVBM, G4 S3/MinIO); notes-kvcache-tiers.md §1.3, §2.2.
 */
export const NET_CHANNELS = [
  { id: 'rdma', label: 'RDMA', GBs: getInterconnect('ib-ndr').GBs, latencyMs: 0, note: 'распределённый пул по RDMA: одна карта 400 Гбит/с' },
  { id: 'tcp', label: 'TCP', GBs: getInterconnect('tcp-25g').GBs, latencyMs: 0, note: 'обычная сеть без RDMA: 25 Гбит/с' },
  {
    id: 's3',
    label: 'S3',
    GBs: 1,
    latencyMs: 100,
    note: 'объектное хранилище вроде S3: около 1 ГБ/с (как у S3 Express) и порядка 100 мс на доступ (оценка Dynamo KVBM; S3 Express One Zone отвечает за единицы мс)',
  },
] as const

export type NetChannelId = (typeof NET_CHANNELS)[number]['id']

export const TIERS = KV_TIERS

export function tierById(id: TierId): KvTier {
  return KV_TIERS.find((t) => t.id === id) ?? KV_TIERS[1]
}

export interface HostLink {
  id: 'pcie4' | 'pcie5'
  /** Практическая полоса в одну сторону, ГБ/с (~25 у Gen4, ~50 у Gen5). */
  practicalGBs: number
  /** Теоретическая полоса в одну сторону, ГБ/с (31,5 и 63). */
  GBs: number
}

/**
 * Хост-линк одного GPU: по нему KV из RAM хоста, с NVMe и из сети (GPUDirect RDMA) попадает в HBM. Каждый GPU
 * (TP-ранг) принимает свою часть KV по своему PCIe x16, поэтому реплика из N GPU получает не больше N × этот линк.
 * A100 и L40S стоят на PCIe Gen4. Hopper и Blackwell на Gen5: у B200 и B300 сам чип умеет Gen6, но процессоры
 * хоста в DGX B200 / B300 дают Gen5, поэтому строку «Gen6» из gpus.ts здесь не берём. NVLink-C2C (Grace) не моделируем.
 */
export function gpuHostLink(gpu: { hostLink?: string }): HostLink {
  const id = /Gen4/i.test(gpu.hostLink ?? '') ? 'pcie4' : 'pcie5'
  const link = getInterconnect(id)
  return { id, practicalGBs: link.practicalGBs ?? link.GBs, GBs: link.GBs }
}

/**
 * Полоса уровня по умолчанию, ГБ/с, для `lanes` параллельных каналов (PCIe каждого GPU, диски в RAID, сетевые карты).
 * Для PCIe берём практическую полосу (~50 ГБ/с у Gen5), а не теоретические 63: так же считают остальные фигуры главы.
 * С `opts.gpus` всё, что ниже HBM, упирается в хост-линки самих GPU: не больше gpus × opts.hostGBs
 * (сколько дисков или сетевых карт ни ставь, в одну GPU KV входит только по её PCIe). `capped` = упёрлось в этот потолок.
 */
export function tierBandwidth(
  id: TierId,
  lanes: number,
  channel: NetChannelId = 'rdma',
  opts: { hostGBs?: number; gpus?: number } = {},
): { GBs: number; latencyMs: number; capped: boolean } {
  if (id === 'g1') return { GBs: getInterconnect(tierById(id).linkId).GBs, latencyMs: 0, capped: false }
  const pcie = getInterconnect('pcie5')
  const hostGBs = opts.hostGBs ?? pcie.practicalGBs ?? pcie.GBs
  let raw: number
  let latencyMs = 0
  if (id === 'g4') {
    const ch = NET_CHANNELS.find((c) => c.id === channel) ?? NET_CHANNELS[0]
    raw = ch.id === 's3' ? ch.GBs : ch.GBs * lanes
    latencyMs = ch.latencyMs
  } else if (id === 'g2') {
    raw = hostGBs * lanes
  } else {
    const link = getInterconnect(tierById(id).linkId)
    raw = (link.practicalGBs ?? link.GBs) * lanes
  }
  const cap = opts.gpus ? opts.gpus * hostGBs : Infinity
  return { GBs: Math.min(raw, cap), latencyMs, capped: raw > cap }
}

export interface Crossover {
  /** Длина контекста, при которой загрузка и пересчёт равны (null, если в диапазоне нет пересечения). */
  tokens: number | null
  /** Кто выигрывает на коротком контексте. */
  shortWinner: 'load' | 'recompute'
}

/**
 * Ищем длину, где loadMs = recomputeMs, на логарифмической сетке lo..hi, потом уточняем делением пополам.
 * Пересчёт растёт быстрее длины (attention квадратичен), загрузка линейно, поэтому знак разности меняется не больше раза.
 */
export function findCrossover(
  model: ModelSpecExt,
  gpu: GpuSpecExt,
  GBs: number,
  latencyMs: number,
  gpus: number,
  lo = 256,
  hi = 131072,
): Crossover {
  const diff = (n: number) => {
    const r = loadVsRecompute(model, gpu, n, GBs, { gpus, latencyMs })
    return r.loadMs - r.recomputeMs
  }
  const dLo = diff(lo)
  const shortWinner = dLo < 0 ? 'load' : 'recompute'
  const steps = 48
  let prevN = lo
  let prevD = dLo
  for (let i = 1; i <= steps; i++) {
    const n = Math.round(lo * Math.pow(hi / lo, i / steps))
    const d = diff(n)
    if (Math.sign(d) !== Math.sign(prevD) && d !== 0) {
      let a = prevN
      let b = n
      for (let k = 0; k < 30 && b - a > 1; k++) {
        const mid = Math.round((a + b) / 2)
        if (Math.sign(diff(mid)) === Math.sign(prevD)) a = mid
        else b = mid
      }
      return { tokens: b, shortWinner }
    }
    prevN = n
    prevD = d
  }
  return { tokens: null, shortWinner }
}
