import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { ArrowRightIcon } from '@phosphor-icons/react'
import { ChapterLink, CodeBlock, Figure, Segmented, Slider, Toggle, cn, cssVar, fmt } from '@/ui'
import { COMP_HINT, COMP_LABEL, GPUS_PER_NODE, buildYaml, layout, type Comp, type NodeView, type Pod } from './k8sModel'

/** Фигура 3 главы 19: DynamoGraphDeployment, оператор и что работает на каждой ноде. */
export function K8sFigure() {
  const [prefill, setPrefill] = useState(2)
  const [decode, setDecode] = useState(3)
  // TP 1 типичен для модели на 8 млрд параметров; 2 и 4 показывают, как воркеры занимают вторую ноду
  const [gpus, setGpus] = useState<'1' | '2' | '4'>('1')
  const [planner, setPlanner] = useState(true)
  const [cacheDs, setCacheDs] = useState(false)
  const [focus, setFocus] = useState<Comp | null>(null)
  const g = Number(gpus)
  const yaml = useMemo(() => buildYaml(prefill, decode, g, planner), [prefill, decode, g, planner])
  const nodes = useMemo(() => layout(prefill, decode, g, planner, cacheDs), [prefill, decode, g, planner, cacheDs])
  const totalGpus = (prefill + decode) * g
  const gpuNodes = nodes.filter((n) => n.gpu).length

  const comps: Comp[] = ['operator', 'frontend', 'prefill', 'decode', ...(planner ? (['planner'] as Comp[]) : []), 'daemon']

  // YAML длинный: при выборе компонента прокручиваем блок к его первой строке, иначе подсветка уходит за край
  const yamlBox = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!focus) return
    const pre = yamlBox.current?.querySelector('pre')
    const first = yaml.lines[focus][0]
    const line = first ? pre?.querySelector(`code > span:nth-child(${first})`) : null
    if (!pre || !line) return
    const delta = line.getBoundingClientRect().top - pre.getBoundingClientRect().top
    pre.scrollTo({ top: Math.max(0, pre.scrollTop + delta - 24), behavior: 'smooth' })
  }, [focus, yaml])

  return (
    <Figure
      id="dynamo-figure-k8s"
      title="Dynamo в Kubernetes: один YAML, оператор и поды на нодах"
      width="full"
      controls={
        <>
          <Slider label="Реплик prefill" value={prefill} min={1} max={4} onChange={setPrefill} />
          <Slider label="Реплик decode" value={decode} min={1} max={6} onChange={setDecode} />
          <Segmented
            label="GPU на воркер (TP)"
            size="sm"
            value={gpus}
            onChange={setGpus}
            options={[
              { value: '1', label: '1' },
              { value: '2', label: '2' },
              { value: '4', label: '4' },
            ]}
          />
          <div className="grid content-start gap-3">
            <Toggle label="Planner" checked={planner} onChange={setPlanner} />
            <Toggle label="Кэш KV на каждой ноде" checked={cacheDs} onChange={setCacheDs} hint="общий кэш ноды, DaemonSet LMCache" />
          </div>
        </>
      }
      caption="Попробуйте: добавьте реплики decode и поставьте 2 GPU на воркер: оператор создаст новые поды, планировщик Kubernetes поставит их на ноды со свободными GPU, и понадобится вторая нода. Нажмите на компонент в списке, чтобы подсветить его строки в YAML и его поды. Включите «Кэш KV на каждой ноде»: на каждой GPU-ноде появится ещё один DaemonSet-под, общий кэш KV этой ноды (LMCache). Он принадлежит ноде, а не модели, и переживает перезапуск воркеров. Раскладка упрощённая: настоящий планировщик Kubernetes учитывает ещё много условий."
    >
      <div className="grid min-w-0 gap-5">
        <div role="group" aria-label="Компоненты" className="flex flex-wrap gap-2">
          {comps.map((c) => (
            <button
              key={c}
              type="button"
              aria-pressed={focus === c}
              onClick={() => setFocus(focus === c ? null : c)}
              className={cn(
                'inline-flex h-8 items-center gap-1.5 rounded-control border px-3 text-[13px] transition-colors',
                focus === c ? 'border-accent bg-accent-soft text-ink' : 'border-line bg-surface text-ink-2 hover:text-ink',
              )}
            >
              <CompMark comp={c} />
              {COMP_LABEL[c]}
            </button>
          ))}
        </div>
        <p className="min-h-[2.75em] text-[14px] leading-snug text-ink-2">
          {focus ? (
            <>
              <span className="font-medium text-ink">{COMP_LABEL[focus]}:</span> {COMP_HINT[focus]}.
            </>
          ) : (
            <>
              Оператор читает один объект DynamoGraphDeployment и превращает каждый компонент в поды и сервисы. Сейчас {fmt.int(totalGpus)} GPU на{' '}
              {fmt.count(gpuNodes, ['GPU-ноде', 'GPU-нодах', 'GPU-нодах'])} по {GPUS_PER_NODE}.
              {planner && totalGpus > 8 && <> Planner с бюджетом по умолчанию (<code>max_gpu_budget: 8</code>) столько GPU сам не поднял бы.</>}
            </>
          )}
        </p>
        <div className="grid min-w-0 gap-5 @4xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.25fr)]">
          <div ref={yamlBox} className="min-w-0 self-start">
            <CodeBlock lang="yaml" title="dgd.yaml (упрощённо)" code={yaml.code} highlightLines={focus ? yaml.lines[focus] : undefined} maxHeight={520} wrap />
          </div>
          <div className="grid min-w-0 content-start gap-3">
            {planner && (
              <div className="flex flex-wrap items-center gap-2 rounded-control bg-surface-2 px-3 py-2 text-[13px] text-ink-2">
                <CompMark comp="planner" />
                Planner
                <ArrowRightIcon size={14} aria-hidden className="text-ink-3" />
                <code className="font-mono text-[12.5px] text-ink">spec.components[decode].replicas</code>
                <ArrowRightIcon size={14} aria-hidden className="text-ink-3" />
                оператор создаёт или удаляет поды
              </div>
            )}
            <div className="grid min-w-0 gap-3 @2xl:grid-cols-2">
              {nodes.map((n) => (
                <NodeCard key={n.name} node={n} focus={focus} />
              ))}
            </div>
          </div>
        </div>
        <p className="text-[13px] leading-snug text-ink-3">
          Общий кэш KV ноды как DaemonSet подробно разобран в главе <ChapterLink to="kv-tiers" />. Многонодовым воркерам (одна реплика движка на нескольких
          нодах) нужно, чтобы все куски реплики стартовали вместе: Grove собирает их поды в группу, а планировщик KAI ставит группу целиком (другой
          вариант: LeaderWorkerSet с Volcano).
        </p>
      </div>
    </Figure>
  )
}

function CompMark({ comp }: { comp: Comp }) {
  if (comp === 'prefill' || comp === 'decode') return <span aria-hidden className="h-2.5 w-2.5 rounded-[3px]" style={{ background: cssVar(comp) }} />
  if (comp === 'daemon') return <span aria-hidden className="h-2.5 w-2.5 rounded-[3px] border border-dashed border-ink-3" />
  return <span aria-hidden className="h-2.5 w-2.5 rounded-[3px] border border-ink-3" />
}

function NodeCard({ node, focus }: { node: NodeView; focus: Comp | null }) {
  const dim = (c: Comp) => focus !== null && focus !== c
  return (
    <div className="grid min-w-0 content-start gap-2.5 rounded-panel border border-line bg-surface p-3">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[14px] font-semibold text-ink">{node.name}</span>
        <span className="text-[12px] text-ink-3">{node.gpu ? `${GPUS_PER_NODE} GPU` : 'без GPU'}</span>
      </div>
      {node.gpu && (
        <div className="flex gap-1" role="img" aria-label={`занято GPU: ${node.slots.filter((s) => s !== null).length} из ${GPUS_PER_NODE}`}>
          {node.slots.map((s, i) => {
            const pod = s === null ? null : node.pods[s]
            return (
              <span
                key={i}
                title={pod ? pod.name : 'свободный GPU'}
                className={cn('h-5 flex-1 rounded-[4px] transition-opacity', pod ? '' : 'border border-dashed border-line-strong')}
                style={{ background: pod ? cssVar(pod.comp as never) : undefined, opacity: pod && dim(pod.comp) ? 0.25 : 1 }}
              />
            )
          })}
        </div>
      )}
      <ul className="flex flex-wrap gap-1.5">
        {node.pods.map((p) => (
          <PodChip key={p.name} pod={p} dim={dim(p.comp)} />
        ))}
      </ul>
    </div>
  )
}

function PodChip({ pod, dim }: { pod: Pod; dim: boolean }): ReactNode {
  return (
    <li
      className={cn(
        'inline-flex items-center gap-1.5 rounded-chip px-2 py-1 font-mono text-[12px] text-ink transition-opacity',
        pod.comp === 'daemon' ? 'border border-dashed border-line-strong' : 'bg-surface-2',
        dim && 'opacity-35',
      )}
    >
      <CompMark comp={pod.comp} />
      {pod.name}
      {pod.gpus > 0 && <span className="text-ink-3">{pod.gpus} GPU</span>}
    </li>
  )
}
