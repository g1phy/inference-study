import { useMemo } from 'react'
import { getModel } from '@/engine/data/models'
import { kvBytesForSeq, kvBytesPerToken } from '@/engine/perf'
import { BarChart, LineChart, fmt } from '@/ui'
import { kvPerTokenWithHeads, logGrid } from './lib'

/** KV на токен: одна архитектура с разным числом KV-голов и реальные модели (всё через perf.ts). */
export function KvPerTokenBars() {
  const data = useMemo(() => {
    const q = getModel('qwen3-8b')
    const rows = [
      {
        label: 'Qwen3-8B, если бы MHA',
        value: kvPerTokenWithHeads(q, q.heads),
        note: '32 KV-головы: гипотетическая версия той же модели',
        muted: true,
      },
      { label: 'Qwen3-8B как есть, GQA', value: kvBytesPerToken(q), note: '32 головы Query, 8 KV-голов' },
      {
        label: 'Qwen3-8B, если бы MQA',
        value: kvPerTokenWithHeads(q, 1),
        note: '1 KV-голова: гипотетическая версия той же модели',
        muted: true,
      },
      { label: 'Llama-3.1-70B, GQA', value: kvBytesPerToken(getModel('llama-3.1-70b')), note: '80 слоёв, 8 KV-голов из 64' },
      { label: 'Qwen3-235B-A22B, GQA', value: kvBytesPerToken(getModel('qwen3-235b-a22b')), note: '94 слоя, 4 KV-головы из 64' },
      { label: 'DeepSeek-V3, MLA', value: kvBytesPerToken(getModel('deepseek-v3')), note: '671 млрд параметров, 61 слой, один сжатый вектор на слой' },
    ]
    return rows.map((r) => ({ ...r, id: r.label }))
  }, [])
  return (
    <BarChart
      data={data}
      color="var(--ink-2)"
      valueFormat={(v) => fmt.bytes(v, { mode: 'binary' })}
      aria-label="Байт KV на токен у разных схем attention, BF16"
    />
  )
}

/** gpt-oss-120b: половина слоёв со скользящим окном 128 токенов. KV последовательности против «наивного» подсчёта. */
export function SlidingWindowChart() {
  const series = useMemo(() => {
    const m = getModel('gpt-oss-120b')
    const xs = logGrid(1024, 131072, 30)
    return [
      { id: 'swa', label: 'Окно учтено (гибридный кэш)', color: 'var(--ink)', points: xs.map((x) => [x, kvBytesForSeq(m, x) / 1e9] as [number, number]) },
      {
        id: 'naive',
        label: 'Все слои как полные',
        color: 'var(--ink-3)',
        dashed: true,
        points: xs.map((x) => [x, kvBytesForSeq(m, x, 'bf16', { swaAware: false }) / 1e9] as [number, number]),
      },
    ]
  }, [])
  return (
    <LineChart
      xLabel="Длина последовательности, токенов"
      yLabel="KV одной последовательности, ГБ"
      xScale="log"
      height={200}
      series={series}
      xFormat={(v) => fmt.compact(v)}
      yFormat={(v) => fmt.num(v, Number.isInteger(v) ? 0 : 1)}
      aria-label="KV gpt-oss-120b с учётом скользящего окна и без"
    />
  )
}
