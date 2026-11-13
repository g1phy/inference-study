/**
 * Промпты для глав 19-20: настоящий шаблон Qwen3 + настоящий токенизатор → блоки по 16 и их хэши.
 * Чистые функции (без React), их используют фигуры роутинга и жизненного цикла Dynamo.
 */
import type { ChatMessage, FlowSettings, Token, Tokenizer, ToolDef } from '@/engine/types'
import { renderTemplate, type RenderedTemplateEx } from '@/engine/templates'
import { blockHashes, DEFAULT_BLOCK_SIZE, type BlockInfoExt } from '@/engine/kv'
import { WEATHER_TOOL } from '@/engine/scenarios'

/**
 * Инструмент из сценария «Вызов инструмента» симулятора (SPEC §6.0): тот же объект, что в engine/scenarios,
 * чтобы промпт с tools в главах 18-20 давал те же токены, что в симуляторе.
 */
export { WEATHER_TOOL }

/**
 * Инструменты условного агента поддержки: их описание шаблон кладёт в системную часть промпта,
 * поэтому у всех пользователей агента одинаковое длинное начало (общий префикс).
 */
export const AGENT_TOOLS: ToolDef[] = [
  WEATHER_TOOL,
  {
    type: 'function',
    function: {
      name: 'search_docs',
      description: 'Поиск по внутренней документации платформы инференса: возвращает до пяти фрагментов со ссылками',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Поисковый запрос на естественном языке' },
          top_k: { type: 'integer', description: 'Сколько фрагментов вернуть, от 1 до 5' },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'create_ticket',
      description: 'Создать тикет для дежурного инженера, если проблему нельзя решить ответом',
      parameters: {
        type: 'object',
        properties: {
          title: { type: 'string', description: 'Короткий заголовок' },
          severity: { type: 'string', description: 'Срочность', enum: ['low', 'medium', 'high'] },
          details: { type: 'string', description: 'Что случилось и как воспроизвести' },
        },
        required: ['title', 'severity'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_metrics',
      description: 'Метрики деплоя за последние минуты: TTFT, ITL, длина очереди и доля попаданий в prefix cache',
      parameters: {
        type: 'object',
        properties: {
          deployment: { type: 'string', description: 'Имя деплоя' },
          window_minutes: { type: 'integer', description: 'Окно в минутах' },
        },
        required: ['deployment'],
      },
    },
  },
]

/** Вопросы «других пользователей» того же приложения (для кнопки «Отправить ещё запрос»). */
export const OTHER_QUESTIONS: readonly string[] = [
  'Почему у нас вырос TTFT после релиза?',
  'Найди в документации, как включить prefix caching в vLLM.',
  'Какая погода сейчас в Казани?',
  'Сколько памяти займёт KV cache Qwen3-8B на 32 тысячи токенов?',
  'Создай тикет: стриминг обрывается на длинных ответах.',
  'Чем decode отличается от prefill?',
  'Покажи метрики деплоя chat-prod за последние 15 минут.',
  'Что значит finish_reason length?',
  'Зачем нужен continuous batching?',
  'Какой block size стоит у vLLM по умолчанию?',
]

export interface PromptParts {
  systemText: string
  userText: string
  tools?: ToolDef[]
  thinking?: boolean
}

export interface BuiltPrompt {
  template: RenderedTemplateEx
  tokens: Token[]
  ids: number[]
  blocks: BlockInfoExt[]
  /** Полных блоков (хэшируются). */
  fullBlocks: number
  /** Токенов в неполном хвосте (не хэшируется). */
  tailTokens: number
}

/** Сообщения в порядке шаблона: system (если не пустой), затем user. */
export function messagesOf(p: PromptParts): ChatMessage[] {
  const out: ChatMessage[] = []
  if (p.systemText.trim()) out.push({ role: 'system', content: p.systemText })
  out.push({ role: 'user', content: p.userText.trim() ? p.userText : '…' })
  return out
}

/** Шаблон Qwen3 → токены → блоки по B с цепочкой хэшей (хвост короче B не хэшируется). */
export function buildPrompt(p: PromptParts, tokenizer: Tokenizer, blockSize = DEFAULT_BLOCK_SIZE): BuiltPrompt {
  const template = renderTemplate({
    family: 'qwen3',
    messages: messagesOf(p),
    tools: p.tools && p.tools.length ? p.tools : undefined,
    enableThinking: p.thinking ?? true,
    addGenerationPrompt: true,
  })
  const tokens = tokenizer.encode(template.text)
  const ids = tokens.map((t) => t.id)
  const blocks = blockHashes(ids, blockSize)
  const fullBlocks = blocks.filter((b) => b.full).length
  return { template, tokens, ids, blocks, fullBlocks, tailTokens: ids.length - fullBlocks * blockSize }
}

/** Части промпта читателя из глобальных настроек (сообщение, системный промпт, thinking, tools). */
export function readerParts(s: FlowSettings): PromptParts {
  return { systemText: s.systemText, userText: s.userText, thinking: s.thinking, tools: s.tools ? [WEATHER_TOOL] : undefined }
}
