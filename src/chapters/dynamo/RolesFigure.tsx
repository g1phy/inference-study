import { Fragment, type ReactNode } from 'react'
import { Figure } from '@/ui'

/**
 * Кто что делает: одни и те же этапы у vLLM самого по себе и под Dynamo (notes-engines.md §1.0,
 * notes-dynamo.md §1.10-1.12). Широкая фигура показывает таблицу, узкая те же строки карточками.
 */
const ROWS: Array<{ stage: string; alone: ReactNode; dynamo: ReactNode }> = [
  { stage: 'Принять HTTP-запрос', alone: 'процесс API server', dynamo: 'frontend на Rust, один на много воркеров' },
  { stage: 'Шаблон и токенизация', alone: 'API server', dynamo: 'preprocessor во frontend' },
  {
    stage: 'Выбрать реплику',
    alone: 'реплика одна, снаружи обычный балансировщик',
    dynamo: (
      <>
        роутер во frontend (KV-aware с <code>--router-mode kv</code>)
      </>
    ),
  },
  { stage: 'Когда и что считать: scheduler, KV cache, batching', alone: 'EngineCore', dynamo: 'тот же EngineCore внутри воркера' },
  { stage: 'Forward, маска structured output, сэмплинг', alone: 'GPU-воркеры', dynamo: 'то же самое, в воркере' },
  { stage: 'Детокенизация и стоп-строки', alone: 'API server', dynamo: 'postprocessor во frontend' },
  {
    stage: 'Парсеры reasoning и tool calls',
    alone: (
      <>
        API server (<code>--reasoning-parser</code>)
      </>
    ),
    dynamo: (
      <>
        postprocessor (<code>--dyn-reasoning-parser</code> на воркере)
      </>
    ),
  },
  { stage: 'Перенести KV на другую машину', alone: 'коннектор NixlConnector и свой прокси', dynamo: 'NIXL, порядок задаёт роутер' },
  { stage: 'Сколько реплик держать', alone: 'решают вне движка', dynamo: 'Planner' },
]

export function RolesFigure() {
  return (
    <Figure
      id="dynamo-figure-roles"
      title="Кто что делает: vLLM сам по себе и под Dynamo"
      width="wide"
      caption="Под Dynamo движок работает в режиме «токены на входе, токены на выходе»: его собственный HTTP-сервер, шаблон и парсеры не используются."
    >
      <div className="hidden min-w-0 @xl:block">
        <table className="w-full table-fixed border-collapse text-[14px] leading-snug [&_code]:text-[12.5px] [&_code]:whitespace-nowrap">
          <colgroup>
            <col className="w-[36%]" />
            <col className="w-[30%]" />
            <col />
          </colgroup>
          <thead>
            <tr className="border-b border-line-strong text-left">
              <th scope="col" className="py-2 pr-3 text-[13px] font-normal text-ink-3">
                Этап
              </th>
              <th scope="col" className="px-3 py-2 text-[15px] font-semibold text-ink">
                vLLM сам по себе
              </th>
              <th scope="col" className="bg-surface-2 px-3 py-2 text-[15px] font-semibold text-ink">
                Под Dynamo
              </th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r.stage} className="border-b border-line align-top">
                <th scope="row" className="py-2.5 pr-3 text-left text-[14px] font-normal text-ink-2">
                  {r.stage}
                </th>
                <td className="px-3 py-2.5 text-ink">{r.alone}</td>
                <td className="bg-surface-2 px-3 py-2.5 text-ink">{r.dynamo}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid min-w-0 gap-4 @xl:hidden">
        {ROWS.map((r) => (
          <div key={r.stage} className="grid min-w-0 gap-1">
            <div className="text-[14px] font-semibold text-ink">{r.stage}</div>
            <dl className="grid grid-cols-[6.5rem_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-[14px] leading-snug [&_code]:text-[12.5px]">
              <Fragment>
                <dt className="text-ink-3">vLLM сам</dt>
                <dd className="min-w-0 text-ink">{r.alone}</dd>
                <dt className="font-medium text-ink-2">под Dynamo</dt>
                <dd className="min-w-0 text-ink">{r.dynamo}</dd>
              </Fragment>
            </dl>
          </div>
        ))}
      </div>
    </Figure>
  )
}
