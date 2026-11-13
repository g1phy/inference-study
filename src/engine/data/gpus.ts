/**
 * GPU для калькуляторов и симуляций. Перенесено из research/numbers.json (ключ `gpus`,
 * генератор research/scripts/numbers_calc.py, данные на 2026-11-12) и
 * research/notes-numbers.md §2.1–2.2.
 *
 * Соглашения (notes-numbers.md, «Соглашения»):
 * - tflops только DENSE (без 2:4 sparsity): в datasheet NVIDIA число со звёздочкой «with sparsity» = 2× dense;
 * - bwTBs: десятичные ТБ/с (10^12 байт/с);
 * - memGB: ёмкость «по паспорту»; usableMiB: memory.total из nvidia-smi (по нему считаем бюджет KV);
 * - nvlinkGBs: сумма двух направлений на GPU, как в datasheet (в одну сторону = половина).
 */
import type { GpuSpec } from '../types'

/** GpuSpec + необязательные поля, которых нет в общем контракте. */
export interface GpuSpecExt extends GpuSpec {
  /** Другие id этого GPU (например, id из research/numbers.json). */
  aliases?: string[]
  arch?: string
  memType?: string
  /** memory.total из nvidia-smi в МиБ: реальная ёмкость, от неё считается бюджет KV. */
  usableMiB?: number
  tdpW?: number
  nvlinkGen?: string
  hostLink?: string
  /** Краткая заметка по-русски: чем GPU примечателен для инференса. */
  notes?: string
  /** Основной для селекторов на сайте (H100 / H200 / B200 и т.п.). */
  featured?: boolean
}

/** Все GPU: сначала основные для инференса (A100…L40S), затем потребительские и AMD для сравнения. */
export const GPUS: GpuSpecExt[] = [
  {
    id: 'a100-80g',
    aliases: ['a100-sxm-80gb', 'a100'],
    name: 'A100 80GB SXM',
    arch: 'Ampere',
    memGB: 80,
    usableMiB: 81920, // nvidia-smi 81920 MiB = 80.0 GiB: github.com/jax-ml/jax/issues/15962
    memType: 'HBM2e',
    bwTBs: 2.039, // datasheet A100 80GB SXM: 2,039 GB/s
    tflops: { bf16: 312 }, // 312 | 624* (*sparsity); FP8 tensor cores нет (появились в Hopper)
    nvlinkGBs: 600, // NVLink 3, 12 links
    nvlinkGen: 'NVLink 3 (12 links)',
    hostLink: 'PCIe Gen4 x16 (64 ГБ/с в обе стороны)',
    tdpW: 400,
    year: 2020,
    notes: 'Предыдущее поколение: нет FP8, полоса памяти 2 ТБ/с.',
    source:
      'https://www.nvidia.com/en-us/data-center/a100/ (312 | 624* TFLOPS BF16, *with sparsity); research/numbers.json gpus[a100-sxm-80gb]',
  },
  {
    id: 'h100-sxm',
    aliases: ['h100'],
    name: 'H100 SXM',
    arch: 'Hopper',
    memGB: 80,
    usableMiB: 81559, // nvidia-smi 81559 MiB = 79.65 GiB (docs.nvidia.com/dgx-basepod)
    memType: 'HBM3',
    bwTBs: 3.35, // datasheet; книга с. 62, 79
    tflops: { bf16: 989, fp8: 1979 }, // datasheet 1,979* / 3,958* with sparsity → dense /2; книга с. 62: 989 / 3.35 ≈ 295
    nvlinkGBs: 900, // NVLink 4, 18 links (450 ГБ/с в одну сторону)
    nvlinkGen: 'NVLink 4 (18 links)',
    hostLink: 'PCIe Gen5 x16 (128 ГБ/с в обе стороны)',
    tdpW: 700,
    year: 2022,
    featured: true,
    notes: 'Рабочая лошадка инференса 2023–2026: 80 ГБ HBM3, 3,35 ТБ/с, FP8.',
    source:
      'https://www.nvidia.com/en-us/data-center/h100/ ; книга Inference Engineering с. 62, 79; research/numbers.json gpus[h100-sxm]',
  },
  {
    id: 'h200',
    aliases: ['h200-sxm'],
    name: 'H200 SXM',
    arch: 'Hopper',
    memGB: 141,
    usableMiB: 143771, // nvidia-smi 143771 MiB = 140.4 GiB (github.com/NVIDIA/cuda-samples/issues/311)
    memType: 'HBM3e',
    bwTBs: 4.8, // datasheet; книга с. 79
    tflops: { bf16: 989, fp8: 1979 }, // тот же кристалл, что H100
    nvlinkGBs: 900,
    nvlinkGen: 'NVLink 4 (18 links)',
    hostLink: 'PCIe Gen5 x16 (128 ГБ/с в обе стороны)',
    tdpW: 700,
    year: 2024,
    featured: true,
    notes: 'Вычисления как у H100, но больше и быстрее память: decode быстрее, prefill такой же.',
    source: 'https://www.nvidia.com/en-us/data-center/h200/ ; книга с. 79; research/numbers.json gpus[h200-sxm]',
  },
  {
    id: 'b200',
    aliases: ['b200-hgx'],
    name: 'B200 (HGX)',
    arch: 'Blackwell',
    memGB: 180, // физически 192 GB HBM3e, NVIDIA пишет 180 GB
    usableMiB: 183359, // nvidia-smi 183359 MiB = 179.06 GiB (forums.developer.nvidia.com/t/364164)
    memType: 'HBM3e',
    bwTBs: 8.0,
    tflops: { bf16: 2250, fp8: 4500, fp4: 9000 }, // HGX B200 (8 GPU): FP4 144|72 PF, FP8 72 PF*, BF16 36 PF* → /8 и /2
    nvlinkGBs: 1800, // NVLink 5
    nvlinkGen: 'NVLink 5 (18 links)',
    hostLink: 'PCIe Gen6 x16 (256 ГБ/с в обе стороны)',
    tdpW: 1000,
    year: 2025,
    featured: true,
    notes: 'Blackwell: 8 ТБ/с, FP4 (NVFP4) в тензорных ядрах.',
    source:
      'https://www.nvidia.com/en-us/data-center/hgx/ ; https://www.nvidia.com/en-us/data-center/dgx-b200/ ; research/numbers.json gpus[b200-hgx]',
  },
  {
    id: 'gb200',
    name: 'GB200 (1 GPU в NVL72)',
    arch: 'Blackwell',
    memGB: 186,
    usableMiB: 189471, // nvidia-smi 189471 MiB = 185.0 GiB
    memType: 'HBM3e',
    bwTBs: 8.0,
    tflops: { bf16: 2500, fp8: 5000, fp4: 10000 }, // superchip (1 Grace + 2 GPU): FP4 40|20 PF, FP8 20 PF*, BF16 10 PF* → /2
    nvlinkGBs: 1800,
    nvlinkGen: 'NVLink 5, домен 72 GPU',
    hostLink: 'NVLink-C2C к Grace (900 ГБ/с в обе стороны)',
    tdpW: 1200,
    year: 2025,
    notes: 'Тот же кристалл B200 при большем TDP; домен NVLink на 72 GPU.',
    source:
      'https://www.nvidia.com/en-us/data-center/gb200-nvl72/ ; https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ ; research/numbers.json gpus[gb200]',
  },
  {
    id: 'b300',
    aliases: ['b300-hgx'],
    name: 'B300 (HGX, Blackwell Ultra)',
    arch: 'Blackwell Ultra',
    memGB: 288,
    usableMiB: 275040, // nvidia-smi «B300 SXM6 AC» 275040 MiB = 268.6 GiB (Oracle OCI B300 quickstart)
    memType: 'HBM3e 12-high',
    bwTBs: 8.0,
    tflops: { bf16: 2250, fp8: 4500, fp4: 13500 }, // HGX B300: FP4 144|108 PF, FP8 72 PF*, BF16 36 PF* → /8 (и /2)
    nvlinkGBs: 1800,
    nvlinkGen: 'NVLink 5 (18 links)',
    hostLink: 'PCIe Gen6 x16 (256 ГБ/с в обе стороны)',
    tdpW: 1100,
    year: 2025,
    notes: 'Больше памяти (288 ГБ) и FP4; BF16/FP8 как у B200.',
    source:
      'https://www.nvidia.com/en-us/data-center/hgx/ ; https://docs.oracle.com/en-us/iaas/Content/Compute/gpu-quick-start/nvidia/B300/README-B300.htm ; research/numbers.json gpus[b300-hgx]',
  },
  {
    id: 'l40s',
    name: 'L40S 48GB',
    arch: 'Ada Lovelace',
    memGB: 48,
    usableMiB: 46068, // nvidia-smi 46068 MiB (ECC on)
    memType: 'GDDR6 ECC',
    bwTBs: 0.864,
    tflops: { bf16: 362, fp8: 733 }, // datasheet: BF16 362.05 | 733*, FP8 733 | 1,466*
    hostLink: 'PCIe Gen4 x16 (64 ГБ/с в обе стороны)',
    nvlinkGen: 'нет NVLink',
    tdpW: 350,
    year: 2023,
    notes: 'PCIe-карта без NVLink и с медленной GDDR6: дешевле, но decode заметно медленнее.',
    source: 'https://www.nvidia.com/en-us/data-center/l40s/ ; research/numbers.json gpus[l40s]',
  },
  {
    id: 'l4',
    name: 'L4 24GB',
    arch: 'Ada Lovelace',
    memGB: 24,
    usableMiB: 23034, // nvidia-smi 23034 MiB
    memType: 'GDDR6',
    bwTBs: 0.3, // книга с. 80: 300 GB/s
    tflops: { bf16: 121, fp8: 242.5 }, // datasheet BF16 242*, FP8 485* (*sparsity)
    hostLink: 'PCIe Gen4 x16',
    nvlinkGen: 'нет NVLink',
    tdpW: 72,
    year: 2023,
    notes: 'Маленькая энергоэффективная карта: 300 ГБ/с, для небольших моделей.',
    source: 'https://www.nvidia.com/en-us/data-center/l4/ ; книга с. 80; research/numbers.json gpus[l4]',
  },
  {
    id: 'h100-nvl',
    name: 'H100 NVL 94GB (PCIe)',
    arch: 'Hopper',
    memGB: 94,
    memType: 'HBM3',
    bwTBs: 3.9,
    tflops: { bf16: 835.5, fp8: 1670.5 }, // 1,671* / 3,341* with sparsity
    nvlinkGBs: 600, // NVLink bridge (2-way)
    nvlinkGen: 'NVLink bridge (2-way)',
    hostLink: 'PCIe Gen5 x16',
    tdpW: 400,
    year: 2023,
    notes: 'PCIe-вариант H100 с 94 ГБ памяти.',
    source: 'https://www.nvidia.com/en-us/data-center/h100/ ; research/numbers.json gpus[h100-nvl]',
  },
  {
    id: 'gb300',
    name: 'GB300 (1 GPU в NVL72)',
    arch: 'Blackwell Ultra',
    memGB: 288,
    memType: 'HBM3e 12-high',
    bwTBs: 8.0,
    tflops: { bf16: 2500, fp8: 5000, fp4: 15000 }, // GB300 NVL72: FP4 1440|1080 PF, FP8 720 PF*, BF16 360 PF* → /72
    nvlinkGBs: 1800,
    nvlinkGen: 'NVLink 5, домен 72 GPU',
    hostLink: 'NVLink-C2C к Grace (900 ГБ/с в обе стороны)',
    tdpW: 1400,
    year: 2025,
    notes: 'Blackwell Ultra в стойке NVL72; usable-память по nvidia-smi не проверена.',
    source:
      'https://www.nvidia.com/en-us/data-center/gb300-nvl72/ ; https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/',
  },
  {
    id: 'rtx-4090',
    name: 'GeForce RTX 4090 24GB',
    arch: 'Ada Lovelace',
    memGB: 24,
    memType: 'GDDR6X',
    bwTBs: 1.008,
    tflops: { bf16: 165.2, fp8: 330.3 }, // с FP32-accumulate (так считают фреймворки), dense
    hostLink: 'PCIe Gen4 x16',
    nvlinkGen: 'нет NVLink',
    tdpW: 450,
    year: 2022,
    notes: 'Потребительская карта для локального инференса.',
    source: 'NVIDIA Ada GPU Architecture whitepaper, Appendix A; research/numbers.json gpus[rtx-4090]',
  },
  {
    id: 'rtx-5090',
    name: 'GeForce RTX 5090 32GB',
    arch: 'Blackwell (GB202)',
    memGB: 32,
    memType: 'GDDR7',
    bwTBs: 1.792, // книга с. 90
    tflops: { bf16: 209.5, fp8: 419, fp4: 1676 }, // whitepaper Table 3, FP32-accumulate, dense
    hostLink: 'PCIe Gen5 x16',
    nvlinkGen: 'нет NVLink',
    tdpW: 575,
    year: 2025,
    notes: 'Потребительский Blackwell: 32 ГБ, 1,8 ТБ/с.',
    source: 'NVIDIA RTX Blackwell GPU Architecture whitepaper, Table 3; книга с. 90; research/numbers.json gpus[rtx-5090]',
  },
  {
    id: 'mi300x',
    name: 'AMD Instinct MI300X (для сравнения)',
    arch: 'CDNA 3',
    memGB: 192,
    memType: 'HBM3',
    bwTBs: 5.3,
    tflops: { bf16: 1307.4, fp8: 2614.9 }, // dense; sparse = ×2
    nvlinkGen: 'Infinity Fabric (не NVLink)',
    hostLink: 'PCIe Gen5 x16',
    tdpW: 750,
    year: 2023,
    notes: 'Не NVIDIA: для сравнения памяти и полосы.',
    source: 'https://www.glennklockwood.com/garden/processors/mi300x ; research/numbers.json gpus[mi300x]',
  },
  {
    id: 'mi355x',
    name: 'AMD Instinct MI355X (для сравнения)',
    arch: 'CDNA 4',
    memGB: 288,
    memType: 'HBM3e',
    bwTBs: 8.0,
    tflops: { bf16: 2516.6, fp8: 5033.2, fp4: 10066.3 }, // dense
    nvlinkGen: 'Infinity Fabric (не NVLink)',
    hostLink: 'PCIe Gen5 x16',
    tdpW: 1400,
    year: 2025,
    notes: 'Не NVIDIA: для сравнения памяти и полосы.',
    source:
      'https://www.glennklockwood.com/garden/processors/mi355x ; https://inferencex.semianalysis.com/chips/mi355x ; research/numbers.json gpus[mi355x]',
  },
]

/** GPU для основных селекторов сайта (в порядке показа). */
export const FEATURED_GPU_IDS: readonly string[] = ['h100-sxm', 'h200', 'b200']

const GPU_INDEX: Map<string, GpuSpecExt> = (() => {
  const m = new Map<string, GpuSpecExt>()
  for (const g of GPUS) {
    m.set(g.id, g)
    for (const a of g.aliases ?? []) m.set(a, g)
  }
  return m
})()

/** GPU по id (или по алиасу из research/numbers.json); бросает ошибку, если такого нет. */
export function getGpu(id: string): GpuSpecExt {
  const g = GPU_INDEX.get(id)
  if (!g) throw new Error(`getGpu: неизвестный GPU «${id}»`)
  return g
}

/** То же, что getGpu, но возвращает undefined вместо ошибки. */
export function findGpu(id: string): GpuSpecExt | undefined {
  return GPU_INDEX.get(id)
}
