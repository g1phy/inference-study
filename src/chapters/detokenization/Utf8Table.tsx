import { Figure, cn } from '@/ui'
import { byteRole, hexByte } from './utf8'

const SAMPLES = [
  { ch: 'K', what: 'латиница, цифры, пробел' },
  { ch: 'щ', what: 'кириллица' },
  { ch: '推', what: 'иероглифы, многие символы' },
  { ch: '🚀', what: 'эмодзи' },
] as const

const encoder = new TextEncoder()

/** Двоичная запись байта с выделенным префиксом (0, 110, 1110, 11110, 10). */
function Bits({ b }: { b: number }) {
  const bits = b.toString(2).padStart(8, '0')
  const role = byteRole(b)
  const prefixLen = role === 'ascii' ? 1 : role === 'cont' ? 2 : bits.indexOf('0') + 1
  return (
    <span className="font-mono text-[12px] tracking-[0.04em] text-ink-3">
      <span className="font-semibold text-ink">{bits.slice(0, prefixLen)}</span>
      {bits.slice(prefixLen)}
    </span>
  )
}

/** Статичная таблица: сколько байтов занимает символ в UTF-8 и как ведущий байт сообщает длину. */
export function Utf8Table() {
  return (
    <Figure
      id="detokenization-figure-utf8"
      title={
        <>
          Сколько байтов занимает символ в <span className="whitespace-nowrap">UTF-8</span>
        </>
      }
      width="narrow"
      caption="Ведущий байт сообщает длину символа: 110… значит 2 байта, 1110… значит 3, 11110… значит 4. Байты продолжения всегда начинаются с 10. Пока не пришли все, символ не напечатать."
    >
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left text-[14px]">
          <thead>
            <tr className="text-[13px] text-ink-3">
              <th className="py-1.5 pr-3 font-medium">Символ</th>
              <th className="py-1.5 pr-3 font-medium">Байты</th>
              <th className="py-1.5 font-medium">Первый байт в битах</th>
            </tr>
          </thead>
          <tbody>
            {SAMPLES.map((s) => {
              const bytes = [...encoder.encode(s.ch)]
              return (
                <tr key={s.ch} className="border-t border-line align-top">
                  <td className="py-2 pr-3">
                    <div className="grid gap-1">
                      <span className="font-mono text-[18px] leading-none text-ink">{s.ch}</span>
                      <span className="text-[13px] leading-snug text-ink-3">{s.what}</span>
                    </div>
                  </td>
                  <td className="py-2 pr-3">
                    <div className="flex flex-wrap gap-1">
                      {bytes.map((b, i) => (
                        <span
                          key={i}
                          className={cn(
                            'rounded-[4px] px-1 font-mono text-[12.5px] tabular-nums',
                            i === 0 ? 'bg-surface-3 text-ink' : 'text-ink-2 shadow-[inset_0_0_0_1px_var(--line-strong)]',
                          )}
                        >
                          {hexByte(b)}
                        </span>
                      ))}
                    </div>
                    <div className="mt-1 text-xs text-ink-3">{bytes.length === 1 ? '1 байт' : `${bytes.length} байта`}</div>
                  </td>
                  <td className="py-2">
                    <Bits b={bytes[0]} />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Figure>
  )
}
