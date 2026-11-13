import { Fragment, type ReactNode } from 'react'
import { ChapterLink, Figure, cn } from '@/ui'

/**
 * Кто есть кто: три движка и слой над ними, рядом (notes-engines.md §1.0 и §7.1, notes-dynamo.md §1.9-1.12).
 * Широкая фигура показывает таблицу; на узкой те же строки идут по аспектам, чтобы сравнение не терялось.
 */
const SYSTEMS: Array<{ name: string; short: string; layer?: boolean }> = [
  { name: 'vLLM', short: 'vLLM' },
  { name: 'SGLang', short: 'SGLang' },
  { name: 'TensorRT-LLM', short: 'TRT-LLM' },
  { name: 'NVIDIA Dynamo', short: 'Dynamo', layer: true },
]

const ROWS: Array<{ aspect: string; cells: [ReactNode, ReactNode, ReactNode, ReactNode] }> = [
  {
    aspect: 'Что это',
    cells: [
      'движок: одна реплика модели на своих GPU',
      'движок: одна реплика модели на своих GPU',
      'движок NVIDIA: одна реплика на своих GPU',
      'слой над движками: много реплик на многих машинах',
    ],
  },
  {
    aspect: 'Главная идея',
    cells: [
      'PagedAttention и бюджет токенов на шаг',
      'RadixAttention: дерево префиксов с точностью до токена',
      'рукописные ядра под Hopper и Blackwell, FP8 и NVFP4',
      'KV-aware роутер, дезагрегация через NIXL, Planner',
    ],
  },
  {
    aspect: 'Шаблон, токены, детокенизация, парсеры',
    cells: [
      'процесс API server',
      'TokenizerManager и DetokenizerManager',
      'слой LLM API',
      'frontend на Rust, один и тот же для всех трёх движков',
    ],
  },
  {
    aspect: 'KV на GPU',
    cells: ['блоки по 16 токенов', 'страницы по 1 токену', 'блоки по 32 токена', 'держит движок воркера'],
  },
  {
    aspect: 'Роутер между репликами',
    cells: [
      'отдельно: vLLM Router, llm-d',
      'SGLang Model Gateway',
      'через Dynamo или Triton',
      <>
        встроен: по умолчанию round-robin, KV-aware флагом <code>--router-mode kv</code>
      </>,
    ],
  },
  {
    aspect: 'Дезагрегация',
    cells: [
      'коннекторы NIXL, Mooncake, LMCache',
      'Mooncake или NIXL',
      'context- и generation-инстансы, NIXL',
      'PrefillRouter выбирает prefill-воркер первым, KV едет через NIXL',
    ],
  },
  {
    aspect: 'Когда брать',
    cells: [
      'широкая поддержка моделей и железа, быстрый старт',
      'много общих префиксов, большие MoE',
      'максимум скорости на Hopper и Blackwell',
      'много реплик, длинные общие префиксы, жёсткие цели по TTFT и ITL',
    ],
  },
]

export function WhoIsWhoFigure() {
  return (
    <Figure
      id="engines-figure-who"
      title="Кто есть кто: три движка и слой над ними"
      width="wide"
      caption={
        <>
          Состояние на ноябрь 2026. Движки считают модель на своих GPU, Dynamo распределяет работу между ними: подробно в главах <ChapterLink to="dynamo" /> и{' '}
          <ChapterLink to="routing" />.
        </>
      }
    >
      {/* широкая фигура: таблица, столбец Dynamo выделен подложкой как другой слой */}
      <div className="hidden min-w-0 @3xl:block">
        <table className="w-full table-fixed border-collapse text-[14px] leading-snug [&_code]:text-[12.5px] [&_code]:whitespace-nowrap">
          <colgroup>
            <col className="w-[16%]" />
            <col />
            <col />
            <col />
            <col className="w-[24%]" />
          </colgroup>
          <thead>
            <tr className="text-left text-[12.5px] text-ink-3">
              <th className="pb-1" />
              <th colSpan={3} className="px-3 pb-1 font-normal">
                движки: одна реплика модели
              </th>
              <th className="rounded-t-[8px] bg-surface-2 px-3 pt-2 pb-1 font-normal">слой над движками</th>
            </tr>
            <tr className="border-b border-line-strong text-left">
              <th scope="col" className="py-2 pr-3">
                <span className="sr-only">Что сравниваем</span>
              </th>
              {SYSTEMS.map((s) => (
                <th key={s.name} scope="col" className={cn('px-3 py-2 text-[15px] font-semibold text-ink', s.layer && 'bg-surface-2')}>
                  {s.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r.aspect} className="border-b border-line align-top">
                <th scope="row" className="py-2.5 pr-3 text-left text-[13px] font-normal text-ink-2">
                  {r.aspect}
                </th>
                {r.cells.map((c, i) => (
                  <td key={i} className={cn('px-3 py-2.5 text-ink', SYSTEMS[i].layer && 'bg-surface-2')}>
                    {c}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* узкая фигура: по аспектам, четыре строки в каждом */}
      <div className="grid min-w-0 gap-5 @3xl:hidden">
        {ROWS.map((r) => (
          <div key={r.aspect} className="grid min-w-0 gap-1.5">
            <div className="text-[14px] font-semibold text-ink">{r.aspect}</div>
            <dl className="grid grid-cols-[4.75rem_minmax(0,1fr)] gap-x-3 gap-y-1 text-[14px] leading-snug [&_code]:whitespace-nowrap">
              {SYSTEMS.map((s, i) => (
                <Fragment key={s.name}>
                  <dt className={cn(s.layer ? 'font-medium text-ink' : 'text-ink-3')}>{s.short}</dt>
                  <dd className="min-w-0 text-ink">{r.cells[i]}</dd>
                </Fragment>
              ))}
            </dl>
          </div>
        ))}
      </div>
    </Figure>
  )
}
