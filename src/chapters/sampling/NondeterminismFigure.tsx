import { useMemo, useState } from 'react'
import { Figure, Slider, fmt } from '@/ui'
import { cn } from '@/ui/cn'
import { hashString } from '@/engine/rng'

/**
 * Иллюстрация недетерминизма при T = 0: от размера batch зависит порядок суммирования в ядрах,
 * логиты отличаются в последних знаках, и при почти равных кандидатах argmax переключается.
 * Числа условные: базовые логиты и «шум» подобраны для наглядности.
 */
const A = { text: ' поэтому', z: 8.1374 }
const B = { text: ' так', z: 8.1371 }
const NOISE = 4e-4
const MAX_BATCH = 64

function noisy(batch: number, salt: string) {
  const h = hashString(`${salt}:${batch}`) / 4294967296
  return (h * 2 - 1) * NOISE
}

function logitsFor(batch: number) {
  return { a: A.z + noisy(batch, 'a'), b: B.z + noisy(batch, 'b') }
}

export function NondeterminismFigure() {
  const [batch, setBatch] = useState(8)
  const all = useMemo(() => Array.from({ length: MAX_BATCH }, (_, i) => logitsFor(i + 1)), [])
  const cur = all[batch - 1]
  const winsA = cur.a >= cur.b
  const countA = all.filter((x) => x.a >= x.b).length

  return (
    <Figure
      id="sampling-figure-nondeterminism"
      title="T = 0, а ответы разные: логиты зависят от размера batch"
      width="narrow"
      controls={<Slider label="Сколько запросов в batch" value={batch} min={1} max={MAX_BATCH} onChange={setBatch} format={(v) => fmt.int(v)} />}
      caption="Попробуйте: двигайте размер batch. Ваш запрос тот же, меняется только число запросов рядом с ним, а с ним порядок суммирования в ядрах GPU. Числа условные."
    >
      <div className="grid gap-4">
        <table className="w-full text-left text-[15px] tabular-nums">
          <thead className="text-sm text-ink-3">
            <tr>
              <th className="py-1 font-medium">Кандидат</th>
              <th className="py-1 text-right font-medium">Логит</th>
              <th className="py-1 text-right font-medium">Greedy</th>
            </tr>
          </thead>
          <tbody>
            {[
              { c: A, z: cur.a, win: winsA },
              { c: B, z: cur.b, win: !winsA },
            ].map((r) => (
              <tr key={r.c.text} className={cn('border-t border-line', r.win && 'bg-accent-soft')}>
                <td className="py-1.5 pl-1 font-mono">{r.c.text.replace(/^ /, '·')}</td>
                {/* 6 знаков: при 4–5 знаках почти равные логиты иногда выглядели одинаковыми, хотя один «выбран».
                    Разрыв между ними здесь не меньше 1,8·10⁻⁶, а шаг float32 у чисел около 8 примерно 10⁻⁶ */}
                <td className="py-1.5 text-right">{fmt.num(r.z, 6)}</td>
                <td className="py-1.5 pr-1 text-right">{r.win ? 'выбран' : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="grid gap-1.5">
          <div className="text-sm text-ink-2">
            Победитель при каждом размере batch от 1 до {MAX_BATCH}: «поэтому» в {countA} {fmt.plural(countA, ['случае', 'случаях', 'случаях'])}, «так» в{' '}
            {MAX_BATCH - countA}
          </div>
          <div className="grid w-full max-w-[420px] grid-cols-[repeat(16,minmax(0,1fr))] gap-[3px]" role="group" aria-label="Какой кандидат выбран при каждом размере batch">
            {all.map((x, i) => {
              const a = x.a >= x.b
              return (
                <button
                  key={i}
                  type="button"
                  tabIndex={-1}
                  onClick={() => setBatch(i + 1)}
                  aria-label={`batch ${i + 1}: ${a ? 'поэтому' : 'так'}`}
                  className={cn(
                    'aspect-square rounded-[3px] transition-shadow',
                    a ? 'bg-decode' : 'bg-surface shadow-[inset_0_0_0_1.5px_var(--c-decode)]',
                    i + 1 === batch && 'outline-2 outline-offset-1 outline-accent',
                  )}
                />
              )
            })}
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-3">
            <span className="flex items-center gap-1.5">
              <span aria-hidden className="h-3 w-3 rounded-[2px] bg-decode" />
              «поэтому»
            </span>
            <span className="flex items-center gap-1.5">
              <span aria-hidden className="h-3 w-3 rounded-[2px] shadow-[inset_0_0_0_1.5px_var(--c-decode)]" />
              «так»
            </span>
          </div>
        </div>
      </div>
    </Figure>
  )
}
