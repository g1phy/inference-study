/**
 * Тексты и иконки стадий симулятора: подписи узлов схемы, где стадия работает, фраза-помощник для новичка,
 * главы «Подробнее».
 */
import type { ReactNode } from 'react'
import type { Icon } from '@phosphor-icons/react'
import {
  ArrowsLeftRightIcon,
  ArrowsSplitIcon,
  DesktopIcon,
  GlobeSimpleIcon,
  GraphicsCardIcon,
  TextAaIcon,
  TreeStructureIcon,
} from '@phosphor-icons/react'
import { Term } from '@/ui'
import { STAGE_INFO } from '@/engine/flow'
import type { StageId } from '@/engine/types'
import { isChapterId, type ChapterId } from '@/content/chapters'

export interface StageCopy {
  /** Подпись узла на схеме. */
  title: string
  /** Что делает узел, 2-3 слова. */
  sub: string
  /** Где стадия работает: на CPU во frontend, на GPU-воркере, в сети (в связке Dynamo с воркерами vLLM). */
  where: string
  icon: Icon
  /** Одна-две фразы простыми словами для инспектора. */
  help: ReactNode
}

const FRONTEND_CPU = 'CPU, процесс frontend'

export const STAGES: Readonly<Record<StageId, StageCopy>> = {
  client: {
    title: 'Клиент',
    sub: 'шлёт JSON',
    where: 'ваше приложение',
    icon: DesktopIcon,
    help: (
      <>
        Ваше приложение отправляет на сервер обычный HTTP-запрос: JSON с сообщениями и параметрами генерации. Флаг stream: true просит отдавать ответ по
        кусочкам.
      </>
    ),
  },
  frontend: {
    title: 'Frontend',
    sub: 'приём HTTP',
    where: FRONTEND_CPU,
    icon: GlobeSimpleIcon,
    help: (
      <>
        <Term t="Dynamo Frontend">Frontend</Term> работает как приёмная сервера: проверяет запрос и раскладывает его поля на правила генерации (как
        выбирать токены и когда остановиться). GPU здесь ещё не участвует.
      </>
    ),
  },
  preprocess: {
    title: 'Preprocessor',
    sub: 'шаблон и токены',
    where: FRONTEND_CPU,
    icon: TextAaIcon,
    help: (
      <>
        Модель не читает JSON. <Term t="Preprocessor" />, часть frontend на CPU, склеивает сообщения по <Term t="chat template">шаблону чата</Term> в
        один текст со служебными маркерами, а <Term t="tokenizer">токенизатор</Term> режет этот текст на токены. Дальше по сети и на GPU едут только их
        номера.
      </>
    ),
  },
  route: {
    title: 'Router',
    sub: 'выбор воркера',
    where: FRONTEND_CPU,
    icon: ArrowsSplitIcon,
    help: (
      <>
        Копий модели несколько. <Term t="KV-aware routing">KV-aware роутер</Term> ищет ту, у которой в <Term t="KV cache" /> уже лежит начало вашего
        промпта (его не придётся считать заново), и учитывает, насколько она занята. В Dynamo этот режим включают флагом --router-mode kv, по умолчанию
        запросы раздаются по кругу (<Term t="Round-robin">round-robin</Term>).
      </>
    ),
  },
  prefill: {
    title: 'Prefill',
    sub: 'весь промпт',
    where: 'GPU-воркер',
    icon: GraphicsCardIcon,
    help: (
      <>
        GPU прогоняет весь промпт через все слои модели за один проход. По пути он сохраняет ключи и значения (<Term t="Q / K / V">K и V</Term>) каждого
        токена в KV cache, а в конце получает вероятности первого токена ответа.
      </>
    ),
  },
  transfer: {
    title: 'NIXL',
    sub: 'KV по сети',
    where: 'сеть между GPU',
    icon: ArrowsLeftRightIcon,
    help: (
      <>
        Prefill и decode идут на разных GPU, поэтому готовый KV cache нужно перевезти. <Term t="NIXL" /> копирует его напрямую из памяти одного GPU в
        память другого по <Term t="RDMA" />, почти без участия CPU.
      </>
    ),
  },
  decode: {
    title: 'Decode',
    sub: 'токен за шаг',
    where: 'GPU-воркер',
    icon: GraphicsCardIcon,
    help: (
      <>
        Ответ рождается по одному токену за шаг. На каждом шаге GPU читает из памяти веса модели (у <Term t="MoE" /> не все: общие слои и только тех
        экспертов, к которым попали токены шага) и весь KV cache, считает вероятности следующего токена, и сэмплер выбирает один. KV cache растёт на
        токен.
      </>
    ),
  },
  postprocess: {
    title: 'Postprocessor',
    sub: 'текст и парсеры',
    where: FRONTEND_CPU,
    icon: TreeStructureIcon,
    help: (
      <>
        Воркер вернул во frontend только номера токенов. Там, на CPU, они снова становятся текстом: <Term t="Detokenization">детокенизатор</Term>{' '}
        придерживает неполные символы, а парсеры раскладывают текст по полям: рассуждение, ответ, вызов инструмента.
      </>
    ),
  },
  stream: {
    title: 'Клиент (SSE)',
    sub: 'читает чанки',
    where: 'сеть до клиента',
    icon: DesktopIcon,
    help: (
      <>
        Клиент получает ответ по кусочкам в формате <Term t="SSE" />, не дожидаясь конца генерации. В конце приходят finish_reason, usage и строка
        data: [DONE].
      </>
    ),
  },
}

/** Главы для ссылки «Подробнее» (из STAGE_INFO движка). */
export function stageChapters(stage: StageId): ChapterId[] {
  return STAGE_INFO[stage].chapters.filter(isChapterId)
}

/** Подписи каналов ответа. */
export const CHANNEL_LABEL = {
  reasoning: 'reasoning',
  content: 'content',
  tool_call: 'tool_calls',
} as const
