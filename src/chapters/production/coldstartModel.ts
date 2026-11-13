/**
 * Разложение холодного старта реплики (глава 26, фигура 2).
 *
 * Книга делит cold start на четыре части: GPU procurement, загрузка образа, загрузка весов,
 * старт движка с компиляцией (notes-foundations-production §1.17, Figure 7.9). Здесь веса
 * разбиты на «скачать до узла» и «поднять в память GPU», а старт движка на «компиляция и
 * CUDA graphs» и «прогрев», чтобы было видно, что сокращает каждый приём.
 *
 * Время скачивания весов = байты весов (engine/perf.weightBytes) / полоса источника (F15).
 * Полосы источников и длительности прочих этапов условные: книга чисел не даёт, кроме
 * «компиляция часто несколько минут» (стр. 190) и таблицы notes-foundations-production §3.10
 * (0,2-25 ГБ/с как ползунок).
 *
 * TensorRT-LLM: с версии 1.0 по умолчанию PyTorch-рантайм без отдельной сборки engine (SPEC §8.8,
 * notes-engines §1.22); на старте автотюнинг ядер и захват CUDA graphs, поэтому порядок тот же, что
 * у vLLM. Сборка engine командой trtllm-build (минуты) относится к старому TensorRT-бэкенду до 1.0;
 * в ветке 1.4 он удалён. Блог TRT-LLM №20: на DeepSeek-R1 два захвата CUDA graphs заняли по ~10 с
 * из 184 с инициализации, дольше всего грузились веса.
 */
import { weightBytes, type ModelLike } from '@/engine/perf'
import type { DType } from '@/engine/types'

export type WeightSource = 'hf' | 's3' | 'node'
export type EngineKind = 'vllm' | 'trtllm'

export interface ColdStartOptions {
  model: ModelLike
  dtype: DType
  source: WeightSource
  warmPool: boolean
  imageCached: boolean
  engine: EngineKind
  compileCache: boolean
  /** Сколько GPU в реплике: веса грузятся в них параллельно. */
  gpus?: number
}

export type ColdStartStageId = 'node' | 'image' | 'download' | 'load' | 'compile' | 'warmup'

export interface ColdStartStage {
  id: ColdStartStageId
  label: string
  seconds: number
  /** Как получилось число (по-русски, одна строка). */
  detail: string
  /** Как этот этап сокращают. */
  shorten: string
  /** Число посчитано (true) или условное (false). */
  computed: boolean
}

/** Условные полосы источников весов, ГБ/с (notes-foundations-production §3.10: 0,2-25 ГБ/с). */
export const SOURCE_GBS: Record<WeightSource, number> = { hf: 0.2, s3: 1, node: 0 }
export const SOURCE_LABEL: Record<WeightSource, string> = {
  hf: 'Hugging Face',
  s3: 'S3 в том же регионе',
  node: 'Кэш на узле',
}
/** Чтение весов с локального NVMe в HBM с разбором safetensors, ГБ/с (условно). */
export const LOAD_GBS = 5
/** Образ движка: CUDA, PyTorch, сам движок (условно; книга: «многие гигабайты»). */
export const IMAGE_GB = 10
/** Скачивание и распаковка образа, ГБ/с (условно). */
export const IMAGE_GBS = 0.4

export function coldStartStages(o: ColdStartOptions): ColdStartStage[] {
  const bytes = weightBytes(o.model, o.dtype)
  const gb = bytes / 1e9
  const gpus = Math.max(1, o.gpus ?? 1)
  // ru-RU: запятая и неразрывный пробел между тысячами («1 342»)
  const gbStr = new Intl.NumberFormat('ru-RU', { minimumFractionDigits: gb >= 100 ? 0 : 1, maximumFractionDigits: gb >= 100 ? 0 : 1 }).format(gb)
  const fmtGb = (v: number) => `${String(v).replace('.', ',')} ГБ/с`
  const nodes = Math.ceil(gpus / 8)
  const node: ColdStartStage = {
    id: 'node',
    label: nodes > 1 ? 'Узлы с GPU' : 'Узел с GPU',
    seconds: o.warmPool ? 0 : 240,
    detail: o.warmPool
      ? nodes > 1
        ? 'узлы уже в кластере (тёплый пул)'
        : 'узел уже в кластере (тёплый пул)'
      : nodes > 1
        ? `облако выделяет и загружает ${nodes} узла параллельно, условно 4 мин`
        : 'облако выделяет и загружает новый узел, условно 4 мин',
    shorten: 'тёплый пул узлов, резерв мощностей, время выдачи узла в договоре с провайдером',
    computed: false,
  }
  const image: ColdStartStage = {
    id: 'image',
    label: 'Образ контейнера',
    seconds: o.imageCached ? 2 : IMAGE_GB / IMAGE_GBS,
    detail: o.imageCached ? 'образ уже лежит на узле' : `≈ ${IMAGE_GB} ГБ по ${fmtGb(IMAGE_GBS)} с распаковкой (условно)`,
    shorten: 'маленький образ без лишнего, заранее скачанный на узлы (DaemonSet-прогрев), ленивая загрузка слоёв',
    computed: false,
  }
  const srcGBs = SOURCE_GBS[o.source]
  const download: ColdStartStage = {
    id: 'download',
    label: 'Веса до узла',
    seconds: srcGBs > 0 ? gb / srcGBs : 0,
    detail: srcGBs > 0 ? `${gbStr} ГБ весов / ${fmtGb(srcGBs)} (${SOURCE_LABEL[o.source]}, условно)` : `${gbStr} ГБ уже на локальном диске узла`,
    shorten: 'кэш весов рядом с GPU (NVMe узла, общий кэш в ЦОД), квантизация (FP8 вдвое меньше), параллельно с образом',
    computed: srcGBs > 0,
  }
  const load: ColdStartStage = {
    id: 'load',
    label: 'Веса в память GPU',
    seconds: gb / (LOAD_GBS * gpus),
    detail: `${gbStr} ГБ с диска в HBM по ${fmtGb(LOAD_GBS)}${gpus > 1 ? ` на каждый из ${gpus} GPU параллельно` : ''} (условно)`,
    shorten: 'safetensors через mmap, потоковая загрузка шардов параллельно на все GPU, GPUDirect Storage',
    computed: true,
  }
  let compileSec: number
  let compileDetail: string
  if (o.engine === 'vllm') {
    compileSec = o.compileCache ? 12 : 45
    compileDetail = o.compileCache
      ? 'torch.compile из кэша, захват CUDA graphs для набора batch (условно)'
      : 'torch.compile и захват CUDA graphs для набора batch (условно)'
  } else {
    compileSec = o.compileCache ? 20 : 60
    compileDetail = o.compileCache
      ? 'автотюнинг и JIT-ядра из кэша, захват CUDA graphs (условно)'
      : 'PyTorch-рантайм TensorRT-LLM (с версии 1.0): автотюнинг ядер и захват CUDA graphs для набора batch (условно)'
  }
  const compile: ColdStartStage = {
    id: 'compile',
    label: 'Компиляция и CUDA graphs',
    seconds: compileSec,
    detail: compileDetail,
    shorten: 'кэш компиляции (torch.compile, JIT-ядра, результаты автотюнинга; только для той же GPU, CUDA и зависимостей), меньше размеров для CUDA graphs',
    computed: false,
  }
  const warmup: ColdStartStage = {
    id: 'warmup',
    label: 'Прогрев',
    seconds: 10,
    detail: 'профилирующий проход для размера KV cache, тестовые запросы, readiness-проба (условно)',
    shorten: 'короткий прогрев, readiness только после него, чтобы балансировщик не слал запросы раньше',
    computed: false,
  }
  return [node, image, download, load, compile, warmup]
}

export function coldStartTotal(stages: readonly ColdStartStage[]): number {
  return stages.reduce((s, x) => s + x.seconds, 0)
}
