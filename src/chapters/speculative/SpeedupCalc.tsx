import { useMemo, useState } from 'react'
import { BarChart, Figure, Formula, LineChart, Slider, Stat, fmt, useElementSize, type LineAnnotation } from '@/ui'
import { acceptedDistribution, acceptedFraction, expectedSpeedup, expectedTokens, optimalK } from './spec'

const tex = (x: number, d = 2) => fmt.num(x, d).replace(',', '{,}').replace(/\s/g, '\\,')
/** «2,80×»: ускорение одной записью в формуле, плитке и подсказке. */
const times = (x: number) => `${fmt.num(x, 2)}×`

export function SpeedupCalc() {
  const [alpha, setAlpha] = useState(0.8)
  const [k, setK] = useState(4)
  const [c, setC] = useState(0.05)
  const [ref, { width }] = useElementSize<HTMLDivElement>()
  // на узкой фигуре формулы в две строки (общий вид, затем подстановка), столбики тоньше
  const narrow = width > 0 && width < 520

  const E = expectedTokens(alpha, k)
  const S = expectedSpeedup(alpha, k, c)
  const best = useMemo(() => optimalK(alpha, c, 12), [alpha, c])
  const series = useMemo(() => {
    const ks = Array.from({ length: 12 }, (_, i) => i + 1)
    return [
      { id: 'cost', label: `черновик стоит ${fmt.num(c, 2)} шага`, color: 'decode', points: ks.map((kk) => [kk, expectedSpeedup(alpha, kk, c)] as [number, number]), markers: true },
      { id: 'free', label: 'бесплатный черновик', color: 'ink-3', points: ks.map((kk) => [kk, expectedSpeedup(alpha, kk, 0)] as [number, number]) },
    ]
  }, [alpha, c])
  const dist = useMemo(
    () =>
      acceptedDistribution(alpha, k).map((p, j) => ({
        id: String(j),
        label: j === k ? 'все' : String(j),
        value: p,
        color: 'decode',
      })),
    [alpha, k],
  )
  const annotations: LineAnnotation[] = [
    ...(k === best.k ? [] : [{ kind: 'x' as const, x: best.k, label: `оптимум k = ${best.k}` }]),
    { kind: 'y', y: 1, label: 'без спекуляции', labelAlign: 'start' },
    { kind: 'point', x: k, y: S, label: k === best.k ? `ваш k = ${k}, оптимум` : `ваш k = ${k}` },
  ]

  const eTex = narrow
    ? `\\begin{aligned}E&=\\frac{1-\\alpha^{k+1}}{1-\\alpha}\\\\&=\\frac{1-${tex(alpha)}^{${k + 1}}}{1-${tex(alpha)}}\\approx ${tex(E)}\\end{aligned}`
    : `E=\\frac{1-\\alpha^{k+1}}{1-\\alpha}=\\frac{1-${tex(alpha)}^{${k + 1}}}{1-${tex(alpha)}}\\approx ${tex(E)}`
  const sTex = narrow
    ? `\\begin{aligned}S&=\\frac{E}{k\\cdot c+1}\\\\&=\\frac{${tex(E)}}{${k}\\cdot ${tex(c)}+1}\\approx ${tex(S)}\\times\\end{aligned}`
    : `S=\\frac{E}{k\\cdot c+1}=\\frac{${tex(E)}}{${k}\\cdot ${tex(c)}+1}\\approx ${tex(S)}\\times`

  return (
    <Figure
      id="speculative-figure-speedup"
      title="Сколько даёт спекуляция в среднем"
      controls={
        <>
          <Slider label="Acceptance rate α" value={alpha} min={0.3} max={0.95} step={0.05} onChange={setAlpha} format={(x) => fmt.num(x, 2)} />
          <Slider label="Длина черновика k" value={k} min={1} max={12} step={1} onChange={setK} format={(x) => fmt.int(x)} />
          <Slider label="Цена черновика c" value={c} min={0} max={0.3} step={0.01} onChange={setC} format={(x) => fmt.num(x, 2)} hint="время шага черновика в долях шага target" />
        </>
      }
      caption="Попробуйте: при α = 0,8 опустите цену черновика до 0,02 (лучшая длина k = 11), затем поднимите до 0,2 (лучшая k = 4): дорогой черновик выгодно делать коротким."
      source="Формулы Leviathan и др. (2023) в предположении, что каждый черновой токен принимается независимо с вероятностью α."
    >
      <div ref={ref} className="grid min-w-0 gap-6">
        <div className="grid min-w-0 gap-3">
          <Formula block tex={eTex} label="E: сколько токенов в среднем даёт один проход target (acceptance length)" />
          <Formula block tex={sTex} label="S: во сколько раз больше токенов в секунду на пользователя; знаменатель это время раунда в шагах target" />
        </div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 @lg:grid-cols-4">
          <Stat label="Токенов за проход" value={fmt.num(E, 2)} hint="acceptance length" />
          <Stat label="Ускорение decode" value={times(S)} hint="ток/с на пользователя" />
          <Stat label="Принято черновиков" value={fmt.pct(acceptedFraction(alpha, k))} hint={`от предложенных, хотя α = ${fmt.num(alpha, 2)}`} />
          <Stat label="Лучшая длина k" value={fmt.int(best.k)} hint={`даёт ${times(best.speedup)} при этой цене`} />
        </div>
        <div className="grid min-w-0 gap-6 @3xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
          <div className="min-w-0">
            <div className="mb-2 text-sm font-medium text-ink-2">Ускорение от длины черновика</div>
            <LineChart
              series={series}
              xLabel="Длина черновика k"
              yLabel="Ускорение, ×"
              height={220}
              xDomain={[1, 12]}
              xTicks={[1, 2, 4, 6, 8, 10, 12]}
              xFormat={(v) => fmt.int(v)}
              yFormat={(v) => fmt.num(v, 1)}
              directLabels={false}
              annotations={annotations}
              aria-label="Ожидаемое ускорение в зависимости от длины черновика"
            />
          </div>
          <div className="min-w-0">
            <div className="mb-2 text-sm font-medium text-ink-2">Сколько черновиков принято за проход</div>
            <BarChart
              data={dist}
              horizontal={false}
              height={220}
              barSize={narrow ? 14 : undefined}
              showValues={k <= 6}
              valueFormat={(v) => fmt.pct(v)}
              color="decode"
              aria-label="Распределение числа принятых черновиков"
            />
            {k > 6 && <p className="mt-1 text-[12.5px] text-ink-3">Доли видны при наведении на столбик.</p>}
          </div>
        </div>
      </div>
    </Figure>
  )
}
