import { useState } from 'react'
import { cn } from '@/ui'

const COLS = ['HBM GPU', 'RAM хоста', 'Локальный NVMe', 'Сеть, общее хранилище'] as const

const ROWS: Array<{ system: string; cells: [string, string, string, string] }> = [
  { system: 'Книга и Dynamo KVBM', cells: ['G1', 'G2 (pinned RAM)', 'G3', 'G4: сетевой SSD, S3'] },
  { system: 'SGLang HiCache', cells: ['L1', 'L2, своя у каждого инстанса', 'L3: бэкенд file', 'L3: mooncake, hf3fs (3FS), nixl, aibrix'] },
  { system: 'LMCache MP', cells: ['KV движка', 'L1: RAM сервера', 'L2: адаптер fs', 'L2: mooncake_store, Redis, S3'] },
  { system: 'vLLM OffloadingConnector', cells: ['GPU', 'CPU, основной уровень', 'вторичный: fs', 'вторичные: fs на общем томе, obj (S3), p2p'] },
  { system: 'Mooncake Store', cells: ['не использует', 'DRAM клиентов, общий пул', 'SSD клиентов', 'весь пул общий на кластер'] },
  { system: 'NVIDIA CMX', cells: ['G1', 'G2', 'G3', 'G3.5: флеш на группу стоек; G4'] },
]

/**
 * «Розеттский камень» названий уровней: наведите на столбец (или нажмите на его заголовок), чтобы подсветить его во всех системах.
 * На узком экране таблица прокручивается вбок, а столбец с названием системы остаётся на месте.
 */
export function TermsTable() {
  const [col, setCol] = useState<number | null>(null)
  return (
    <div className="overflow-x-auto rounded-panel border border-line">
      <table className="w-full min-w-[640px] border-collapse text-left text-sm" onPointerLeave={() => setCol(null)}>
        <caption className="sr-only">Как разные системы называют одни и те же уровни памяти</caption>
        <thead>
          <tr className="bg-surface-2">
            <th scope="col" className="sticky left-0 w-[9.5rem] bg-surface-2 px-3 py-2 font-medium text-ink-2">
              Система
            </th>
            {COLS.map((c, i) => (
              <th
                key={c}
                scope="col"
                tabIndex={0}
                onPointerEnter={() => setCol(i)}
                onFocus={() => setCol(i)}
                onBlur={() => setCol(null)}
                className={cn('px-3 py-2 font-medium text-ink transition-colors focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent', col === i && 'bg-accent-soft')}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ROWS.map((r) => (
            <tr key={r.system} className="border-t border-line">
              <th scope="row" className="sticky left-0 bg-surface px-3 py-2 align-top font-medium text-ink">
                {r.system}
              </th>
              {r.cells.map((c, i) => (
                <td key={i} onPointerEnter={() => setCol(i)} className={cn('px-3 py-2 align-top text-ink-2 transition-colors', col === i && 'bg-accent-soft text-ink')}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
