import { useMemo, useState, type ReactNode } from 'react'
import { getGpu } from '@/engine/data/gpus'
import { getInterconnect } from '@/engine/data/interconnects'
import { getModel } from '@/engine/data/models'
import { decodeLowerBoundMs, kvBytesPerToken, maxConcurrentSeqs, ridgePoint, transferMs, weightBytes } from '@/engine/perf'
import { vllmDefaultLimits } from '@/engine/sched'
import { Figure, ScrollX, Tabs, fmt } from '@/ui'
import { VLLM_MEM_UTIL } from '@/chapters/prefill-decode/shared/hw-core'

type TabId = 'gpu' | 'models' | 'links' | 'defaults'

const GPU_ROWS = ['a100-80g', 'h100-sxm', 'h200', 'b200', 'gb200', 'b300', 'l40s']
const MODEL_ROWS = ['qwen3-8b', 'llama-3.1-8b', 'qwen3-30b-a3b', 'qwen3-32b', 'llama-3.1-70b', 'gpt-oss-120b', 'deepseek-v3']
const LINK_ROWS = ['nvlink5', 'nvlink4', 'pcie5', 'ib-xdr', 'ib-ndr', 'roce-400', 'nvme5', 'tcp-25g']

export function NumbersFigure() {
  const [tab, setTab] = useState<TabId>('gpu')
  return (
    <Figure
      id="cheatsheet-figure-numbers"
      title="Ключевые числа"
      width="wide"
      caption="Попробуйте: переключите вкладки. GPU, модели и каналы посчитаны из данных движка сайта (даташиты NVIDIA и config.json моделей, dense TFLOPS, полоса в одну сторону). Дефолты движков меняются от версии к версии: здесь значения на ноябрь 2026."
    >
      <Tabs<TabId>
        tabs={[
          { id: 'gpu', label: 'GPU' },
          { id: 'models', label: 'Модели' },
          { id: 'links', label: 'Каналы' },
          { id: 'defaults', label: 'Дефолты' },
        ]}
        value={tab}
        onChange={setTab}
        aria-label="Таблицы ключевых чисел"
      >
        {(active) => (
          <div className="pt-4">
            {active === 'gpu' && <GpuTable />}
            {active === 'models' && <ModelTable />}
            {active === 'links' && <LinkTable />}
            {active === 'defaults' && <DefaultsTable />}
          </div>
        )}
      </Tabs>
    </Figure>
  )
}

function Table({ head, rows, minWidth = 640, caption }: { head: ReactNode[]; rows: ReactNode[][]; minWidth?: number; caption: string }) {
  return (
    <ScrollX label={caption} scrollerClassName="rounded-control border border-line">
      <table className="w-full text-left text-[14px]" style={{ minWidth }}>
        <caption className="sr-only">{caption}</caption>
        <thead className="bg-surface-2 text-[13px] text-ink-2">
          <tr>
            {head.map((h, i) => (
              <th key={i} scope="col" className={i === 0 ? 'px-3 py-2 font-medium' : 'px-3 py-2 text-right font-medium'}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-line">
              {r.map((c, j) =>
                j === 0 ? (
                  <th key={j} scope="row" className="px-3 py-2 font-medium text-ink">
                    {c}
                  </th>
                ) : (
                  <td key={j} className="px-3 py-2 text-right whitespace-nowrap text-ink-2 tabular-nums">
                    {c}
                  </td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollX>
  )
}

function GpuTable() {
  const rows = useMemo(
    () =>
      GPU_ROWS.map((id) => {
        const g = getGpu(id)
        return [
          g.name,
          `${fmt.int(g.memGB)} ГБ`,
          fmt.bandwidth(g.bwTBs * 1000),
          fmt.int(g.tflops.bf16),
          g.tflops.fp8 ? fmt.int(g.tflops.fp8) : 'нет',
          g.tflops.fp4 ? fmt.int(g.tflops.fp4) : 'нет',
          fmt.int(ridgePoint(g, 'bf16')),
        ]
      }),
    [],
  )
  return (
    <div className="grid gap-2">
      <Table caption="GPU" head={['GPU', 'Память', 'Полоса HBM', 'BF16, TFLOPS', 'FP8', 'FP4', 'Ridge BF16, FLOP/байт']} rows={rows} minWidth={720} />
      <p className="text-[13px] text-ink-3">
        H200 при тех же FLOPS, что у H100, быстрее только в decode (полоса). «нет» значит, что тензорных ядер этого формата нет: веса можно хранить, но
        считать придётся в BF16.
      </p>
    </div>
  )
}

function ModelTable() {
  const h100 = getGpu('h100-sxm')
  const rows = useMemo(
    () =>
      MODEL_ROWS.map((id) => {
        const m = getModel(id)
        // у gpt-oss нативный чекпоинт смешанный (эксперты MXFP4): считаем по нему
        const dtype = m.native ? m.native.dtype : 'bf16'
        const w = weightBytes(m, dtype)
        const lb = decodeLowerBoundMs(m, h100, dtype)
        const seqs = maxConcurrentSeqs(m, h100, { contextLen: 8192, dtype, memUtil: VLLM_MEM_UTIL })
        return [
          m.native ? `${m.name}, MXFP4` : m.name,
          m.experts ? `${fmt.num(m.paramsB, 0)} / ${fmt.num(m.activeParamsB, 1)} млрд` : `${fmt.num(m.paramsB, 1)} млрд`,
          `${m.layers} · ${m.attention === 'mla' ? 'MLA' : `${m.heads}/${m.kvHeads}`}`,
          fmt.bytes(kvBytesPerToken(m, 'bf16'), { mode: 'binary' }),
          fmt.bytes(w),
          w > h100.memGB * 1e9 ? 'не влезает' : `${fmt.ms(lb)} (${fmt.int(1000 / lb)} ток/с)`,
          seqs > 0 ? fmt.int(seqs) : 'не влезает',
        ]
      }),
    [h100],
  )
  return (
    <div className="grid gap-2">
      <Table
        caption="Модели"
        head={['Модель', 'Параметры (всего / активные)', 'Слоёв · головы Q/KV', 'KV на токен', 'Веса', 'Шаг decode на H100, минимум', '8k-запросов на H100']}
        rows={rows}
        minWidth={860}
      />
      <p className="text-[13px] text-ink-3">
        Веса в BF16, у gpt-oss в нативном формате (эксперты MXFP4). Шаг decode: веса, которые читает один шаг, делённые на 3,35 ТБ/с (batch 1, 100 %
        полосы); у MoE читаются только активные эксперты. Запросы на 8 192 токена: сколько последовательностей помещается в KV на одном H100 при
        gpu_memory_utilization {fmt.num(VLLM_MEM_UTIL, 2)}, как в vLLM по умолчанию.
      </p>
    </div>
  )
}

function LinkTable() {
  const qwen = getModel('qwen3-8b')
  const kv8k = kvBytesPerToken(qwen, 'bf16') * 8192
  const rows = useMemo(
    () =>
      LINK_ROWS.map((id) => {
        const l = getInterconnect(id)
        return [l.name, `${fmt.num(l.GBs, l.GBs < 10 ? 1 : 0)} ГБ/с`, fmt.ms(transferMs(1e9, l.GBs)), fmt.ms(transferMs(kv8k, l.GBs))]
      }),
    [kv8k],
  )
  return (
    <div className="grid gap-2">
      <Table caption="Каналы" head={['Канал', 'Полоса в одну сторону', '1 ГБ', `KV 8k токенов Qwen3-8B (${fmt.bytes(kv8k)})`]} rows={rows} minWidth={640} />
      <p className="text-[13px] text-ink-3">
        Производители пишут NVLink и PCIe как сумму двух направлений, а сеть в гигабитах: «400G» это 50 ГБ/с. Для времени передачи берите одну сторону.
      </p>
    </div>
  )
}

function DefaultsTable() {
  const lim = vllmDefaultLimits(getGpu('h100-sxm'))
  const rows: ReactNode[][] = [
    ['vLLM: размер блока KV', '16 токенов', 'во всех релизах; в main переключили на 64, ещё не выпущено'],
    ['vLLM: gpu_memory_utilization', fmt.num(VLLM_MEM_UTIL, 2), 'было 0,9 до v0.20; доля всей памяти GPU на процесс'],
    ['vLLM: max_num_batched_tokens', fmt.int(lim.tokenBudget), 'на H100; бюджет токенов на шаг для chunked prefill'],
    ['vLLM: max_num_seqs', fmt.int(lim.maxSeqs), 'на H100; сколько последовательностей в batch'],
    ['vLLM: chunked prefill, prefix caching', 'включены', 'по умолчанию в V1'],
    ['vLLM: preemption', 'recompute', 'swap убран в v0.18: вытесненный запрос пересчитывает prefill'],
    ['SGLang: --page-size', '1', 'radix-дерево хранит KV с точностью до токена'],
    ['Dynamo: режим роутера', 'round-robin', 'KV-aware (kv) нужно включить, ему нужны KV events от воркеров'],
    ['Dynamo: хэши блоков', 'XXH3_64, seed 1337', 'цепочкой; неполный хвост не хэшируется'],
    ['Qwen3: словарь / ширина логитов', `${fmt.int(151669)} / ${fmt.int(151936)}`, 'маска structured output имеет ширину логитов'],
  ]
  return (
    <ScrollX label="Дефолты движков" scrollerClassName="rounded-control border border-line">
      <table className="w-full min-w-[640px] text-left text-[14px]">
        <caption className="sr-only">Дефолты движков</caption>
        <thead className="bg-surface-2 text-[13px] text-ink-2">
          <tr>
            <th scope="col" className="px-3 py-2 font-medium">
              Параметр
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Значение
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Комментарий
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-line align-top">
              <th scope="row" className="px-3 py-2 font-medium text-ink">
                {r[0]}
              </th>
              <td className="px-3 py-2 whitespace-nowrap text-ink">{r[1]}</td>
              <td className="px-3 py-2 text-ink-2">{r[2]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollX>
  )
}
