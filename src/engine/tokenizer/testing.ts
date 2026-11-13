/**
 * Только для тестов (vitest, environment: 'node'): чтение файлов проекта и настоящий
 * словарь Qwen3 из public/tokenizer/qwen3.json. В браузерный бандл не попадает,
 * потому что его импортируют только *.test.ts.
 */
// @ts-ignore -- в проекте нет @types/node (tsconfig types: vite/client), а в vitest модуль есть
import { readFileSync } from 'node:fs'
import { parseBpeAsset } from './asset'
import { ByteLevelBpeTokenizer } from './bpe'

/** Читает файл по пути относительно корня проекта. */
export function readProjectFile(relPath: string): string {
  return readFileSync(new URL('../../../' + relPath, import.meta.url), 'utf8') as string
}

let qwen3: ByteLevelBpeTokenizer | null = null

/** Настоящий токенизатор Qwen3 из public/tokenizer/qwen3.json (собирается один раз на процесс). */
export function loadQwen3ForTests(): ByteLevelBpeTokenizer {
  if (!qwen3) qwen3 = new ByteLevelBpeTokenizer(parseBpeAsset(JSON.parse(readProjectFile('public/tokenizer/qwen3.json'))))
  return qwen3
}
