/**
 * Каналы передачи данных: интерконнекты GPU, сеть, хранилища, память хоста.
 * Перенесено из research/numbers.json (ключ `interconnects`) и research/notes-numbers.md §2.3,
 * уровни KV cache: research/notes-kvcache-tiers.md §2.2–2.3.
 *
 * Соглашение: GBs = ГБ/с (10^9 байт/с) в ОДНУ сторону. NVIDIA пишет NVLink и PCIe как сумму
 * двух направлений (900 ГБ/с у H100 = 450 в каждую сторону); сеть меряют в Гбит/с (400 Gb/s = 50 ГБ/с).
 */
import type { InterconnectSpec } from '../types'

/** InterconnectSpec + необязательные поля. */
export interface InterconnectSpecExt extends InterconnectSpec {
  /** Сумма двух направлений (как пишет NVIDIA), если применимо. */
  GBsBidirectional?: number
  /** Реально достижимая полоса в одну сторону, если заметно ниже номинала. */
  practicalGBs?: number
  kind: 'gpu-link' | 'host-link' | 'network' | 'storage' | 'memory'
  aliases?: string[]
}

/** Все каналы: от быстрых (NVLink) к медленным (TCP, NVMe), память для масштаба в конце. */
export const INTERCONNECTS: InterconnectSpecExt[] = [
  {
    id: 'nvlink5',
    name: 'NVLink 5 (B200 / GB200 / B300)',
    kind: 'gpu-link',
    GBs: 900,
    GBsBidirectional: 1800, // книга с. 84: 1,800 GB/s
    notes: 'GPU↔GPU внутри ноды (8 GPU в HGX) или стойки NVL72; 18 линков.',
    source: 'https://www.nvidia.com/en-us/data-center/nvlink/ ; книга с. 84 ; research/numbers.json interconnects[nvlink5]',
  },
  {
    id: 'nvlink4',
    name: 'NVLink 4 (H100 / H200)',
    kind: 'gpu-link',
    GBs: 450,
    GBsBidirectional: 900, // книга с. 84: 900 GB/s на Hopper
    notes: 'GPU↔GPU внутри 8-GPU ноды через NVSwitch; 18 линков.',
    source: 'https://www.nvidia.com/en-us/data-center/nvlink/ ; книга с. 84 ; research/numbers.json interconnects[nvlink4]',
  },
  {
    id: 'nvlink3',
    name: 'NVLink 3 (A100)',
    kind: 'gpu-link',
    GBs: 300,
    GBsBidirectional: 600, // 12 links × 50 GB/s
    notes: '12 линков по 50 ГБ/с в обе стороны.',
    source: 'https://www.nvidia.com/en-us/data-center/a100/ ; research/numbers.json interconnects[nvlink3]',
  },
  {
    id: 'nvlink-c2c',
    name: 'NVLink-C2C (Grace ↔ GPU)',
    kind: 'host-link',
    GBs: 450,
    GBsBidirectional: 900, // книга с. 82
    notes: 'Когерентная связь CPU–GPU в GH200 / GB200: в разы быстрее PCIe.',
    source: 'книга с. 82 ; NVIDIA Blackwell Ultra blog ; research/numbers.json interconnects[nvlink-c2c]',
  },
  {
    id: 'pcie6',
    name: 'PCIe Gen6 x16',
    kind: 'host-link',
    GBs: 121,
    GBsBidirectional: 242, // NVIDIA пишет «256 GB/s bidir» (raw)
    notes: '64 GT/s PAM4, FLIT-кодирование; хост-линк Blackwell.',
    source: 'https://en.wikipedia.org/wiki/PCI_Express ; NVIDIA Blackwell Ultra blog ; research/numbers.json interconnects[pcie6-x16]',
  },
  {
    id: 'pcie5',
    aliases: ['pcie5-x16'],
    name: 'PCIe Gen5 x16',
    kind: 'host-link',
    GBs: 63.0, // 32 GT/s/lane, 128b/130b: 63.015 GB/s в одну сторону
    GBsBidirectional: 126, // NVIDIA округляет до «128 GB/s»
    practicalGBs: 50, // измерено ~50 ГБ/с в одну сторону (notes-kvcache-tiers.md §2.2)
    notes: 'Хост-линк H100 / H200: по нему GPU читает RAM хоста (уровень L2 / G2).',
    source: 'https://en.wikipedia.org/wiki/PCI_Express ; H100 datasheet ; research/numbers.json interconnects[pcie5-x16]',
  },
  {
    id: 'pcie4',
    aliases: ['pcie4-x16'],
    name: 'PCIe Gen4 x16',
    kind: 'host-link',
    GBs: 31.5, // 16 GT/s/lane, 128b/130b: 31.508 GB/s
    GBsBidirectional: 63, // NVIDIA пишет «64 GB/s»
    practicalGBs: 25, // реально ~25 ГБ/с в одну сторону
    notes: 'Хост-линк A100 / L40S.',
    source: 'https://en.wikipedia.org/wiki/PCI_Express ; A100/L40S datasheets ; research/numbers.json interconnects[pcie4-x16]',
  },
  {
    id: 'ib-xdr',
    name: 'InfiniBand XDR 800 Гбит/с (ConnectX-8)',
    kind: 'network',
    GBs: 100,
    GBsBidirectional: 200, // 4 lanes × 200 Gb/s
    notes: 'Одна сетевая карта; в DGX B300 их 8.',
    source: 'https://en.wikipedia.org/wiki/InfiniBand ; https://www.nvidia.com/en-us/data-center/dgx-b300/ ; research/numbers.json interconnects[ib-xdr]',
  },
  {
    id: 'ib-ndr',
    name: 'InfiniBand NDR 400 Гбит/с (ConnectX-7)',
    kind: 'network',
    GBs: 50,
    GBsBidirectional: 100, // 4 lanes × 100 Gb/s
    notes: 'Одна сетевая карта (400 Гбит/с = 50 ГБ/с); в DGX H100 / B200 их 8 на ноду.',
    source: 'https://en.wikipedia.org/wiki/InfiniBand ; DGX H100 user guide ; research/numbers.json interconnects[ib-ndr]',
  },
  {
    id: 'ib-ndr-x8',
    name: '8 × InfiniBand NDR (вся нода DGX H100)',
    kind: 'network',
    GBs: 400, // 8 NIC × 400 Gb/s = 400 GB/s на ноду в одну сторону
    notes: 'Если KV передаётся всеми восемью картами параллельно (например, каждый TP-ранг своей).',
    source: 'https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html ; research/notes-numbers.md §3.16',
  },
  {
    id: 'roce-400',
    aliases: ['roce-400g'],
    name: 'RoCE v2 / Ethernet 400GbE',
    kind: 'network',
    GBs: 50,
    GBsBidirectional: 100,
    notes: 'RDMA поверх Ethernet, та же скорость линии, что NDR. Утверждение книги «Ethernet максимум 100 Гбит/с» устарело.',
    source: 'NVIDIA ConnectX-7/-8 datasheets ; DGX H100 user guide (400GbE) ; research/numbers.json interconnects[roce-400g]',
  },
  {
    id: 'tcp-25g',
    name: 'TCP 25 Гбит/с (без RDMA)',
    kind: 'network',
    GBs: 3.125, // 25 Gb/s ≈ 3.1 GB/s (notes-dynamo.md §3.8)
    notes: 'Без RDMA перенос KV деградирует в ~200–500 раз (пример из документации Dynamo).',
    source: 'research/notes-dynamo.md §3.8 и §1.14 [S-INST]',
  },
  {
    id: 'nvme5',
    aliases: ['nvme-gen5'],
    name: 'NVMe SSD PCIe Gen5 x4 (последовательное чтение)',
    kind: 'storage',
    GBs: 14.0, // Micron 9550: 14,000 MB/s; потолок Gen5 x4 ≈ 14.3 GB/s
    notes: 'Один диск; уровень L3 / G3 (локальный SSD). В DGX H100 / B200 8 таких слотов.',
    source: 'https://www.micron.com/products/storage/ssd/data-center-ssd/9550-ssd ; research/numbers.json interconnects[nvme-gen5]',
  },
  {
    id: 'nvme5-x8',
    name: '8 × NVMe Gen5 (RAID0)',
    kind: 'storage',
    GBs: 112, // 8 × 14 GB/s (notes-numbers.md §3.16–3.17)
    notes: 'Все восемь дисков ноды параллельно.',
    source: 'research/notes-numbers.md §3.16–3.17',
  },
  {
    id: 'nvme4',
    aliases: ['nvme-gen4'],
    name: 'NVMe SSD PCIe Gen4 x4 (последовательное чтение)',
    kind: 'storage',
    GBs: 7.0, // Samsung 990 PRO 7,450 MB/s; DC-диски Gen4 ~6.5–7 GB/s
    notes: 'Один диск; книга даёт для локального SSD 5–10 ГБ/с (с. 139).',
    source:
      'https://download.semiconductor.samsung.com/resources/data-sheet/Samsung_NVMe_SSD_990_PRO_Datasheet_Rev.1.0.pdf ; книга с. 139 ; research/numbers.json interconnects[nvme-gen4]',
  },
  {
    id: 'host-dram',
    aliases: ['ddr5-spr-8480'],
    name: 'DRAM хоста: DDR5-4800 × 8 каналов (DGX H100, 1 сокет)',
    kind: 'memory',
    GBs: 307.2, // 8 × 38.4 GB/s (теоретический пик)
    notes:
      'Сама память хоста быстрая, но GPU читает её через PCIe Gen5 (63 ГБ/с теор., ~50 ГБ/с на практике), поэтому для KV на уровне L2 считайте по pcie5.',
    source: 'https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html ; research/numbers.json interconnects[ddr5-spr-8480]',
  },
  {
    id: 'hbm3-h100',
    name: 'HBM3 (H100 SXM), для масштаба',
    kind: 'memory',
    GBs: 3350,
    notes: 'Не интерконнект, а память GPU: самый быстрый уровень (L1 / G1).',
    source: 'https://www.nvidia.com/en-us/data-center/h100/ ; research/numbers.json interconnects[hbm3-h100]',
  },
  {
    id: 'hbm3e-b200',
    name: 'HBM3e (B200 / B300), для масштаба',
    kind: 'memory',
    GBs: 8000,
    notes: 'Память GPU Blackwell, 8 ТБ/с.',
    source: 'https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ ; research/numbers.json interconnects[hbm3e-b200]',
  },
]

const LINK_INDEX: Map<string, InterconnectSpecExt> = (() => {
  const m = new Map<string, InterconnectSpecExt>()
  for (const l of INTERCONNECTS) {
    m.set(l.id, l)
    for (const a of l.aliases ?? []) m.set(a, l)
  }
  return m
})()

/** Канал по id (или по алиасу из research/numbers.json); бросает ошибку, если такого нет. */
export function getInterconnect(id: string): InterconnectSpecExt {
  const l = LINK_INDEX.get(id)
  if (!l) throw new Error(`getInterconnect: неизвестный канал «${id}»`)
  return l
}

/** То же, что getInterconnect, но возвращает undefined вместо ошибки. */
export function findInterconnect(id: string): InterconnectSpecExt | undefined {
  return LINK_INDEX.get(id)
}

/** Каналы для калькулятора передачи KV при дезагрегации (SPEC §6.16, фигура 3). */
export const KV_TRANSFER_LINK_IDS: readonly string[] = ['nvlink5', 'nvlink4', 'ib-ndr-x8', 'ib-xdr', 'ib-ndr', 'roce-400', 'pcie5', 'tcp-25g']

/** Уровень иерархии KV cache для главы 14. */
export interface KvTier {
  id: 'g1' | 'g2' | 'g3' | 'g4'
  /** Имя в терминах SGLang HiCache / сайта (L1–L3) и книги / KVBM (G1–G4). */
  level: string
  name: string
  /** Канал, которым данные этого уровня попадают в GPU (по нему считаем время загрузки). */
  linkId: string
  /** Ёмкость уровня на ноде класса DGX H100, байты. */
  nodeCapacityBytes: number
  capacityNote: string
  bandwidthNote: string
  latencyNote: string
  source: string
}

/**
 * Лестница памяти для KV cache: G1 GPU → G2 RAM хоста → G3 локальный SSD → G4 сеть.
 * Книга с. 139 (G1–G4); ёмкости DGX H100: notes-kvcache-tiers.md §2.3; полосы: там же §2.2.
 */
export const KV_TIERS: KvTier[] = [
  {
    id: 'g1',
    level: 'L1 / G1',
    name: 'HBM GPU',
    linkId: 'hbm3-h100',
    nodeCapacityBytes: 8 * 80e9, // 8 × 80 GB
    capacityNote: 'десятки–сотни ГБ (8 × 80 ГБ на ноде DGX H100)',
    bandwidthNote: 'ТБ/с (H100: 3,35 ТБ/с)',
    latencyNote: '~нс',
    source: 'книга с. 79, 139 ; research/notes-kvcache-tiers.md §2.2–2.3',
  },
  {
    id: 'g2',
    level: 'L2 / G2',
    name: 'RAM хоста',
    linkId: 'pcie5',
    nodeCapacityBytes: 2e12, // DGX H100: 2 TB RAM
    capacityNote: 'сотни ГБ – ТБ (2 ТБ на ноде DGX H100)',
    bandwidthNote: 'десятки–сотни ГБ/с; к одному GPU по PCIe Gen5: 63 ГБ/с теор., ~50 на практике',
    latencyNote: '~мкс',
    source: 'книга с. 139 ; research/notes-kvcache-tiers.md §2.2 (G2 через PCIe Gen5) и §2.3',
  },
  {
    id: 'g3',
    level: 'L3 / G3',
    name: 'Локальный NVMe SSD',
    linkId: 'nvme5',
    nodeCapacityBytes: 30.72e12, // 8 × 3.84 TB U.2
    capacityNote: 'ТБ (8 × 3,84 ТБ = 30,72 ТБ на ноде DGX H100)',
    bandwidthNote: '5–10 ГБ/с по книге; один диск Gen5 до 14 ГБ/с',
    latencyNote: '~60 мкс у устройства, ~мс по KVBM',
    source: 'книга с. 139 ; research/notes-kvcache-tiers.md §2.2–2.3',
  },
  {
    id: 'g4',
    level: 'L3 / G4',
    name: 'Сетевое хранилище',
    linkId: 'ib-ndr',
    nodeCapacityBytes: 100e12, // «десятки ТБ» и больше: условная оценка для визуализации
    capacityNote: 'десятки ТБ и больше (общий пул кластера)',
    bandwidthNote: 'ГБ/с; один NIC 400 Гбит/с = 50 ГБ/с',
    latencyNote: '~мкс сеть + хранилище',
    source: 'книга с. 139 ; research/notes-kvcache-tiers.md §2.2 (L3 удалённый пул по RDMA)',
  },
]
