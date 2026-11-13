import { useMemo, useState, type ReactNode } from 'react'
import { CodeBlock, Figure, Segmented, Tabs, cn } from '@/ui'
import type { ModelFamily } from '@/engine/types'
import { FAMILY_CASES, FAMILY_ORDER, familyFlags, familySpecials, parseFamilyOutput, splitSpecial, type OutputKind, type ParsedOutput } from './scenarios'

const TAB_LABELS: Record<ModelFamily, string> = {
  qwen3: 'Qwen3',
  'deepseek-r1': 'DeepSeek-R1',
  'gpt-oss': 'gpt-oss',
  llama3: 'Llama 3.1',
}

export function FamiliesFigure() {
  const [family, setFamily] = useState<ModelFamily>('qwen3')
  const [kind, setKind] = useState<OutputKind>('answer')

  return (
    <Figure
      id="parsers-figure-families"
      title="Четыре семейства: один и тот же смысл, разная разметка"
      width="wide"
      caption="Попробуйте: откройте DeepSeek-R1, вариант «Ответ», и найдите в выводе открывающий <think>. Его нет: тег стоит в промпте, поэтому парсер начинает в состоянии рассуждения. Потом включите «Вызов инструмента» и сравните, как вызов выглядит у gpt-oss и у Llama 3.1. Ответы условные."
    >
      <div className="grid grid-cols-[minmax(0,1fr)] gap-4">
        <Tabs<ModelFamily>
          className="min-w-0"
          aria-label="Семейство моделей"
          value={family}
          onChange={setFamily}
          wrap
          tabs={FAMILY_ORDER.map((f) => ({ id: f, label: TAB_LABELS[f] }))}
        >
          {(active) => <FamilyPanel family={active} kind={kind} onKind={setKind} />}
        </Tabs>
      </div>
    </Figure>
  )
}

function FamilyPanel({ family, kind, onKind }: { family: ModelFamily; kind: OutputKind; onKind: (k: OutputKind) => void }) {
  const c = FAMILY_CASES[family]
  const raw = c.raw[kind]
  const parsed = useMemo(() => parseFamilyOutput(family, raw), [family, raw])
  // у DeepSeek вызовы умеет только R1-0528: для варианта с вызовом своя модель и флаги, для «Ответа» только парсер рассуждений
  const override = kind === 'tool' ? c.toolFlags : c.answerFlags
  const flags = override ? { ...familyFlags(family), ...override } : familyFlags(family)
  const parts = useMemo(() => splitSpecial(raw, [...familySpecials(family), '<think>', '</think>']), [family, raw])
  const bash = [
    flags.vllm ? `# vLLM\nvllm serve ${flags.model} \\\n  ${flags.vllm.split(' --').join(' \\\n  --')}` : null,
    flags.dynamo ? `# Dynamo: флаги воркера\n${flags.dynamo.split(' --').join(' \\\n  --')}` : null,
  ]
    .filter(Boolean)
    .join('\n\n')

  return (
    <div className="grid gap-4 pt-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="max-w-[72ch] min-w-0 text-[15px] leading-relaxed [overflow-wrap:anywhere] text-ink-2">{c.how}</p>
        <Segmented<OutputKind>
          size="sm"
          aria-label="Тип ответа"
          value={kind}
          onChange={onKind}
          wrap
          options={[
            { value: 'answer', label: 'Ответ' },
            { value: 'tool', label: c.toolLabel ?? 'Вызов инструмента' },
          ]}
        />
      </div>

      <div className="grid gap-4 @2xl:grid-cols-2">
        <div className="grid min-w-0 content-start gap-2">
          <div className="text-sm font-medium text-ink-2">Что печатает модель</div>
          <div className="rounded-control bg-surface-2 px-3 py-2.5 font-mono text-[13px] leading-relaxed [overflow-wrap:anywhere] whitespace-pre-wrap text-ink">
            {parts.map((part, i) =>
              part.special ? (
                <span key={i} className="mx-px rounded-[4px] px-0.5 whitespace-nowrap text-ink-2 outline-1 -outline-offset-1 outline-dashed outline-[var(--c-special)]">
                  {part.text}
                </span>
              ) : (
                <span key={i}>{part.text.replace(/\n/g, '↵\n')}</span>
              ),
            )}
          </div>
          <p className="text-[12.5px] text-ink-3">Пунктиром обведены служебные токены: каждый занимает ровно один токен словаря.</p>
        </div>
        <Parsed parsed={parsed} />
      </div>

      <CodeBlock lang="bash" title="флаги парсеров" code={bash} />
    </div>
  )
}

function Field({ name, swatch, children, reasoning, empty }: { name: string; swatch: ReactNode; children: ReactNode; reasoning?: boolean; empty: boolean }) {
  return (
    <div className="grid gap-1">
      <div className="flex items-center gap-2 font-mono text-[12.5px] text-ink-2">
        {swatch}
        {name}
      </div>
      <div
        className={cn(
          'min-h-8 rounded-control bg-surface px-3 py-1.5 text-[14px] leading-relaxed break-words whitespace-pre-wrap shadow-[inset_0_0_0_1px_var(--line)]',
          empty ? 'text-ink-3' : 'text-ink',
          reasoning && !empty && 'border-l-2 border-dashed border-[var(--c-reasoning)] text-ink-2 italic',
        )}
      >
        {empty ? 'null' : children}
      </div>
    </div>
  )
}

function Parsed({ parsed }: { parsed: ParsedOutput }) {
  const content = parsed.content.replace(/^\n+/, '')
  return (
    <div className="grid min-w-0 content-start gap-2.5">
      <div className="text-sm font-medium text-ink-2">Что получает клиент</div>
      <Field name="reasoning_content" swatch={<span aria-hidden className="h-2.5 w-2.5 rounded-[3px] border-[1.5px] border-dashed border-[var(--c-reasoning)]" />} reasoning empty={!parsed.reasoning}>
        {parsed.reasoning.trim()}
      </Field>
      <Field name="content" swatch={<span aria-hidden className="h-2.5 w-2.5 rounded-[3px] bg-decode" />} empty={!content}>
        {content}
      </Field>
      <Field name="tool_calls" swatch={<span aria-hidden className="h-2.5 w-2.5 rounded-[3px] bg-tool" />} empty={parsed.toolCalls.length === 0}>
        <span className="font-mono text-[13px]">
          {parsed.toolCalls.map((t, i) => (
            <span key={i} className="block">
              <span className="text-ink-3">name: </span>
              {t.name}
              {'\n'}
              <span className="text-ink-3">arguments (строка): </span>
              {t.arguments}
            </span>
          ))}
        </span>
      </Field>
    </div>
  )
}
