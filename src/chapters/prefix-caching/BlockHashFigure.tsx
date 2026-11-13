import { useMemo, useState } from 'react'
import { CheckCircleIcon, LinkBreakIcon, MinusCircleIcon, XCircleIcon } from '@phosphor-icons/react'
import type { Token } from '@/engine/types'
import { blockHashes, commonPrefixBlocks, type BlockInfoExt } from '@/engine/kv'
import { effectiveCachedTokens, prefillFlops } from '@/engine/perf'
import { useTokenizer } from '@/engine/tokenizer/useTokenizer'
import { useSettings } from '@/state/settings'
import { Button, Figure, FigureLive, Segmented, Stat, TextArea, TokenStrip, TokenStripSkeleton, cn, cssVar, fmt, tint } from '@/ui'
import { modelOr } from '../kv-cache/lib'
import { renderQwen3Prompt } from '../kv-cache/usePromptTokens'

/** Длинный системный промпт для демонстрации (условный пример агента поддержки). */
export const LONG_SYSTEM =
  'Ты ассистент службы поддержки интернет-магазина электроники «Вольт». Отвечай вежливо и по делу, на русском языке. ' +
  'Если вопрос касается заказа, попроси номер заказа и не придумывай статус доставки. Не обсуждай политику и не давай медицинских советов. ' +
  'Цены и наличие проверяй только через инструмент get_stock. Если пользователь недоволен, извинись и предложи связаться с оператором. ' +
  'Ответ держи в пределах пяти предложений.'

const OTHER_QUESTION = 'А сколько памяти он занимает, если контекст 32 тысячи токенов?'
/**
 * Время в начале системного промпта: в кэше запрос, отправленный секундой раньше. Qwen3 режет числа по одной
 * цифре, поэтому длина в токенах та же и все следующие блоки содержат ровно те же токены, а хэши у них другие.
 */
const TIME_CACHED = 'Сейчас 14:03:27.\n'
const TIME_NEW = 'Сейчас 14:03:28.\n'

type PresetId = 'same' | 'question' | 'date' | 'edit' | 'custom'

const PRESET_LABELS: Array<{ id: Exclude<PresetId, 'custom'>; label: string }> = [
  { id: 'same', label: 'Тот же запрос' },
  { id: 'question', label: 'Другой вопрос' },
  { id: 'edit', label: 'Фраза в конце системного промпта' },
  { id: 'date', label: 'Время в начале системного промпта' },
]

function editEnd(system: string): string {
  return system.includes('пяти предложений') ? system.replace('пяти предложений', 'трёх предложений') : `${system.trimEnd()} Отвечай коротко.`
}

interface BlockRow {
  block: BlockInfoExt
  cached?: BlockInfoExt
  tokens: Token[]
  status: 'hit' | 'miss' | 'tail'
  /** Индекс первого отличающегося токена внутри блока (если он здесь). */
  diffAt: number | null
  /** Промах, хотя токены блока те же, что в кэше: изменился родитель, а с ним и хэш. */
  sameTokens: boolean
}

/** Длинные серии одинаковых блоков сворачиваем: первые два и последние два видны, середина одной строкой. */
type ListItem = { kind: 'row'; row: BlockRow } | { kind: 'gap'; from: number; to: number; status: BlockRow['status'] }

function compactRows(rows: BlockRow[], expanded: boolean): ListItem[] {
  if (expanded) return rows.map((row) => ({ kind: 'row', row }))
  const out: ListItem[] = []
  let i = 0
  while (i < rows.length) {
    let j = i
    while (j + 1 < rows.length && rows[j + 1].status === rows[i].status && rows[j + 1].sameTokens === rows[i].sameTokens) j++
    if (j - i + 1 > 6) {
      out.push({ kind: 'row', row: rows[i] }, { kind: 'row', row: rows[i + 1] })
      out.push({ kind: 'gap', from: rows[i + 2].block.index, to: rows[j - 2].block.index, status: rows[i].status })
      out.push({ kind: 'row', row: rows[j - 1] }, { kind: 'row', row: rows[j] })
    } else for (let k = i; k <= j; k++) out.push({ kind: 'row', row: rows[k] })
    i = j + 1
  }
  return out
}

const STATUS_WORD: Record<BlockRow['status'], string> = { hit: 'из кэша', miss: 'промах', tail: 'хвост' }

function StatusChip({ status }: { status: BlockRow['status'] }) {
  if (status === 'tail')
    return (
      <span className="inline-flex items-center gap-1 text-xs text-ink-2">
        <MinusCircleIcon size={16} aria-hidden className="text-ink-3" />
        хвост, не хэшируется
      </span>
    )
  const hit = status === 'hit'
  const Icon = hit ? CheckCircleIcon : XCircleIcon
  return (
    <span className="inline-flex items-center gap-1 rounded-chip px-1.5 py-0.5 text-xs font-medium text-ink" style={{ background: tint(hit ? 'good' : 'bad', 14) }}>
      <Icon size={16} aria-hidden style={{ color: cssVar(hit ? 'good' : 'bad') }} />
      {hit ? 'из кэша' : 'промах'}
    </span>
  )
}

export function BlockHashFigure() {
  const [settings] = useSettings()
  const { tokenizer, status } = useTokenizer()
  const [sysMode, setSysMode] = useState<'long' | 'yours'>('long')
  const baseSystem = sysMode === 'long' ? LONG_SYSTEM : settings.systemText
  const [preset, setPreset] = useState<PresetId>('same')
  // Что лежит в кэше: тот же системный промпт, а для пресета «время» ещё и строка со временем секундой раньше
  const [cachedPrefix, setCachedPrefix] = useState('')
  const cachedSystem = cachedPrefix + baseSystem
  const [expanded, setExpanded] = useState(false)
  const [customSys, setCustomSys] = useState<string | null>(null)
  const [customUser, setCustomUser] = useState<string | null>(null)
  const [blockSize, setBlockSize] = useState<16 | 64>(16)

  const presetSys = preset === 'date' ? TIME_NEW + baseSystem : preset === 'edit' ? editEnd(baseSystem) : baseSystem
  const presetUser = preset === 'question' ? OTHER_QUESTION : settings.userText
  const newSystem = preset === 'custom' ? (customSys ?? presetSys) : presetSys
  const newUser = preset === 'custom' ? (customUser ?? presetUser) : presetUser

  const choose = (id: Exclude<PresetId, 'custom'>) => {
    setPreset(id)
    setCachedPrefix(id === 'date' ? TIME_CACHED : '')
    setCustomSys(null)
    setCustomUser(null)
  }

  const data = useMemo(() => {
    if (!tokenizer) return null
    const cachedTok = tokenizer.encode(renderQwen3Prompt(cachedSystem, settings.userText, settings.thinking))
    const newTok = tokenizer.encode(renderQwen3Prompt(newSystem, newUser, settings.thinking))
    const cb = blockHashes(cachedTok.map((t) => t.id), blockSize)
    const nb = blockHashes(newTok.map((t) => t.id), blockSize)
    const hit = commonPrefixBlocks(cb, nb)
    let firstDiff = 0
    while (firstDiff < cachedTok.length && firstDiff < newTok.length && cachedTok[firstDiff].id === newTok[firstDiff].id) firstDiff++
    const rows: BlockRow[] = nb.map((b) => {
      const tokens = newTok.slice(b.start, b.start + b.tokenIds.length)
      const inBlock = firstDiff >= b.start && firstDiff < b.start + b.tokenIds.length && firstDiff < newTok.length
      const status: BlockRow['status'] = !b.full ? 'tail' : b.index < hit ? 'hit' : 'miss'
      const old = cb[b.index]
      return {
        block: b,
        cached: old,
        tokens,
        status,
        diffAt: inBlock ? firstDiff - b.start : null,
        sameTokens: status === 'miss' && Boolean(old?.full) && old.tokenIds.length === b.tokenIds.length && old.tokenIds.every((id, k) => id === b.tokenIds[k]),
      }
    })
    return { cachedTok, newTok, rows, hit, firstDiff, fullBlocks: nb.filter((b) => b.full).length }
  }, [tokenizer, cachedSystem, settings.userText, settings.thinking, newSystem, newUser, blockSize])

  const model = modelOr(settings.modelId)
  const total = data?.newTok.length ?? 0
  const hitTokens = data ? Math.min(total, data.hit * blockSize) : 0
  const computed = total - effectiveCachedTokens(total, hitTokens)
  const flopsFull = data ? prefillFlops(model, total) : 0
  const flopsHit = data ? prefillFlops(model, total, { cachedTokens: hitTokens }) : 0
  const saved = flopsFull > 0 ? 1 - flopsHit / flopsFull : 0
  const diffToken = data && data.firstDiff < data.newTok.length ? data.newTok[data.firstDiff] : null
  const identical = data ? data.firstDiff >= data.newTok.length && data.newTok.length === data.cachedTok.length : false
  const prefixOfCached = data ? !identical && data.firstDiff >= data.newTok.length : false
  const diffInTail = data ? data.hit === data.fullBlocks && data.firstDiff >= data.fullBlocks * blockSize : false
  const sameCount = data ? data.rows.filter((r) => r.sameTokens).length : 0
  const items = data ? compactRows(data.rows, expanded) : []
  const collapsed = items.some((it) => it.kind === 'gap')
  const canCollapse = data ? compactRows(data.rows, false).some((it) => it.kind === 'gap') : false

  return (
    <Figure
      id="prefix-caching-figure-hashes"
      title="Хэши блоков на настоящих токенах вашего запроса"
      width="wide"
      actions={
        <Segmented
          size="sm"
          aria-label="Размер блока"
          value={blockSize}
          onChange={setBlockSize}
          options={[
            { value: 16, label: 'Блок 16' },
            { value: 64, label: 'Блок 64' },
          ]}
        />
      }
      caption="Попробуйте: нажмите «Время в начале системного промпта». В кэше и в новом запросе отличается одна цифра секунд, остальные токены те же, но хэш первого блока стал другим, а за ним вся цепочка: дальше токены в блоках совпадают с кэшем, а хэши нет. Потом нажмите «Другой вопрос»: системный промпт снова берётся из кэша."
    >
      <div className="grid min-w-0 gap-5">
        <div className="grid min-w-0 gap-3">
          <p className="max-w-[68ch] text-sm text-ink-2">
            В кэше уже лежит запрос: системный промпт и ваш вопрос из симулятора, в шаблоне Qwen3
            {cachedPrefix ? <>, а в начале системного промпта время «{cachedPrefix.trim()}»</> : null}. Ниже второй запрос: меняйте его и
            смотрите, какие блоки найдутся в кэше.
          </p>
          <Segmented
            label="Системный промпт в кэше"
            size="sm"
            wrap
            value={sysMode}
            onChange={(v) => {
              setSysMode(v)
              choose(preset === 'custom' ? 'same' : preset)
            }}
            options={[
              { value: 'long', label: 'Агент поддержки (длинный)' },
              { value: 'yours', label: 'Ваш из симулятора' },
            ]}
          />
          <div className="flex flex-wrap gap-2" role="group" aria-label="Что изменить во втором запросе">
            {PRESET_LABELS.map((p) => (
              <Button key={p.id} size="sm" variant={preset === p.id ? 'primary' : 'secondary'} aria-pressed={preset === p.id} onClick={() => choose(p.id)}>
                {p.label}
              </Button>
            ))}
          </div>
          <div className="grid min-w-0 items-start gap-3 @2xl:grid-cols-2">
            <TextArea
              label="Второй запрос: системный промпт"
              value={newSystem}
              rows={3}
              autoGrow
              onChange={(v) => {
                setPreset('custom')
                setCustomSys(v)
                if (customUser === null) setCustomUser(newUser)
              }}
            />
            <TextArea
              label="Второй запрос: вопрос"
              value={newUser}
              rows={3}
              autoGrow
              onChange={(v) => {
                setPreset('custom')
                setCustomUser(v)
                if (customSys === null) setCustomSys(newSystem)
              }}
            />
          </div>
        </div>

        {!data ? (
          status === 'error' ? (
            <p className="text-sm text-ink-3">Токенизатор не загрузился, хэши посчитать не из чего.</p>
          ) : (
            <TokenStripSkeleton count={40} />
          )
        ) : (
          <>
            <div className="grid grid-cols-2 gap-x-6 gap-y-5 @3xl:grid-cols-4">
              <Stat label="Совпало блоков" value={`${data.hit} из ${data.fullBlocks}`} hint={`полных блоков по ${blockSize}`} />
              <Stat label="Из кэша" value={fmt.int(hitTokens)} unit={fmt.plural(hitTokens, ['токен', 'токена', 'токенов'])} hint={`из ${fmt.int(total)} ${fmt.plural(total, ['токена', 'токенов', 'токенов'])}`} swatch={cssVar('good')} />
              <Stat label="Посчитать в prefill" value={fmt.int(computed)} unit={fmt.plural(computed, ['токен', 'токена', 'токенов'])} swatch={cssVar('prefill')} />
              <Stat label="Экономия prefill" value={fmt.pct(saved)} hint={`по FLOP, ${model.name}`} />
            </div>
            <ChainStrip rows={data.rows} />
            <FigureLive as="p" className="text-sm text-ink-2" role="status">
              {data.fullBlocks === 0 ? (
                <>
                  Полных блоков нет: запрос короче одного блока ({fmt.tokens(blockSize)}), его целиком считает prefill.
                </>
              ) : identical ? (
                <>Запросы совпадают целиком. Из кэша берутся все полные блоки; хвост короче блока не хэшируется и считается заново.</>
              ) : prefixOfCached ? (
                <>
                  Второй запрос целиком совпадает с началом закэшированного: все его полные блоки взяты из кэша, хвост короче блока
                  считается заново.
                </>
              ) : diffInTail ? (
                <>
                  Расхождение на позиции {data.firstDiff} попало в хвост, который короче блока и всё равно не хэшируется: все полные блоки
                  взяты из кэша.
                </>
              ) : (
                <>
                  Первое расхождение на позиции {data.firstDiff}
                  {diffToken ? (
                    <>
                      {' '}
                      (токен <span className="rounded-chip bg-surface-2 px-1 font-mono text-ink">{diffToken.text.replace(/ /g, '·').replace(/\n/g, '↵')}</span>)
                    </>
                  ) : null}
                  .{' '}
                  {data.hit > 0
                    ? `Блоки до него совпали, а начиная с блока ${data.hit} у всех хэши другие, потому что каждый хэш включает хэш предыдущего.`
                    : 'Оно попало уже в первый блок, поэтому у всех блоков хэши другие: каждый хэш включает хэш предыдущего.'}
                  {sameCount > 0
                    ? ` У ${fmt.int(sameCount)} ${fmt.plural(sameCount, ['блока', 'блоков', 'блоков'])} из промахов токены ровно те же, что в кэше: другой у них только хэш родителя.`
                    : ''}
                </>
              )}
            </FigureLive>
            <ol className="grid grid-cols-[repeat(auto-fill,minmax(230px,1fr))] gap-2.5" aria-label="Блоки второго запроса">
              {items.map((it) =>
                it.kind === 'gap' ? (
                  <li
                    key={`gap-${it.from}`}
                    className="col-span-full flex flex-wrap items-center justify-between gap-2 rounded-control border border-dashed border-line-strong px-3 py-2 text-sm text-ink-2"
                  >
                    <span>
                      блоки {it.from}–{it.to}: {STATUS_WORD[it.status]} ({fmt.int(it.to - it.from + 1)}), свёрнуты
                    </span>
                    <Button size="sm" variant="ghost" onClick={() => setExpanded(true)}>
                      Показать
                    </Button>
                  </li>
                ) : (
                  <BlockCard key={it.row.block.index} r={it.row} blockSize={blockSize} />
                ),
              )}
            </ol>
            {expanded && !collapsed && canCollapse && (
              <div>
                <Button size="sm" variant="ghost" onClick={() => setExpanded(false)}>
                  Свернуть длинные серии блоков
                </Button>
              </div>
            )}
          </>
        )}
      </div>
    </Figure>
  )
}

/** Полоска-конспект цепочки: по квадрату на блок, рамкой выделен блок с первым расхождением. */
function ChainStrip({ rows }: { rows: BlockRow[] }) {
  const hit = rows.filter((r) => r.status === 'hit').length
  const miss = rows.filter((r) => r.status === 'miss').length
  const tail = rows.some((r) => r.status === 'tail')
  return (
    <div className="grid gap-1.5">
      <div
        role="img"
        aria-label={`Цепочка блоков: ${hit} из кэша, ${miss} промахов${tail ? ', хвост без хэша' : ''}`}
        className="flex flex-wrap gap-1"
      >
        {rows.map((r) => (
          <span
            key={r.block.index}
            className={cn('h-3.5 w-3.5 rounded-[3px]', r.status === 'tail' && 'border border-dashed border-line-strong')}
            style={{
              background: r.status === 'hit' ? cssVar('good') : r.status === 'miss' ? tint('bad', 45) : undefined,
              boxShadow: r.diffAt !== null ? '0 0 0 2px var(--surface), 0 0 0 3.5px var(--accent)' : undefined,
            }}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
        <span className="inline-flex items-center gap-1">
          <CheckCircleIcon size={16} aria-hidden style={{ color: cssVar('good') }} />
          из кэша: {fmt.int(hit)}
        </span>
        <span className="inline-flex items-center gap-1">
          <XCircleIcon size={16} aria-hidden style={{ color: cssVar('bad') }} />
          промах: {fmt.int(miss)}
        </span>
        {tail && (
          <span className="inline-flex items-center gap-1">
            <MinusCircleIcon size={16} aria-hidden className="text-ink-3" />
            хвост без хэша
          </span>
        )}
        {rows.some((r) => r.diffAt !== null) && <span className="text-ink-3">рамка: блок с первым расхождением</span>}
      </div>
    </div>
  )
}

function BlockCard({ r, blockSize }: { r: BlockRow; blockSize: number }) {
  return (
    <li
      className={cn('grid min-w-0 content-start gap-2 rounded-control border p-2.5', r.status === 'tail' ? 'border-dashed border-line' : 'border-line')}
      style={r.status === 'hit' ? { background: tint('good', 7) } : r.status === 'miss' ? { background: tint('bad', 6) } : undefined}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs text-ink-2">
          блок {r.block.index} · <span className="font-mono text-[12.5px] font-semibold text-ink">{r.block.full ? r.block.hash : '········'}</span>
        </span>
        <StatusChip status={r.status} />
      </div>
      <TokenStrip tokens={r.tokens} size="sm" highlightIndex={r.diffAt} />
      <div className="grid font-mono text-[11px] leading-snug text-ink-3">
        {r.block.full ? (
          <>
            <span>
              H({r.block.parentHash ? r.block.parentHash.slice(0, 8) : 'начало'}, {r.block.tokenIds.length} ток.)
            </span>
            <span>{r.cached?.full ? `в кэше: ${r.cached.hash}` : 'в кэше: такого блока нет'}</span>
          </>
        ) : (
          <span>
            {r.block.tokenIds.length} из {blockSize} токенов
          </span>
        )}
      </div>
      {r.sameTokens && (
        <div className="flex items-start gap-1.5 text-xs leading-snug text-ink">
          <LinkBreakIcon size={16} aria-hidden className="mt-px shrink-0 text-ink-2" />
          токены те же, что в кэше, а хэш другой: изменился хэш родителя
        </div>
      )}
    </li>
  )
}
