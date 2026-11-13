# Third-party notices

## Qwen3 tokenizer and chat template

- `public/tokenizer/qwen3.json` is the Qwen3 byte-level BPE vocabulary and merges, converted from the `tokenizer.json`
  published with the Qwen3 models (https://huggingface.co/Qwen).
- `src/engine/templates.ts` re-implements the Qwen3 chat template, and `src/chapters/template/TemplateDeeper.tsx` shows a
  short excerpt of it from `tokenizer_config.json`.
- The test fixtures in `src/engine/tokenizer/fixtures/` and `src/engine/*.fixture.json` hold reference outputs of
  Hugging Face `tokenizers`, `transformers` and XGrammar on these files.

Copyright Alibaba Cloud (Qwen team). Licensed under the Apache License, Version 2.0; a copy is in
[`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) and at https://www.apache.org/licenses/LICENSE-2.0.

## npm dependencies

Bundled into the built site under their own licenses (see each package in `node_modules` after `npm ci`):

| Package | License |
|---|---|
| react, react-dom | MIT |
| motion | MIT |
| katex | MIT |
| @phosphor-icons/react | MIT |
| gpt-tokenizer (the o200k vocabulary used for the comparison in the tokenization chapter) | MIT |
| tailwindcss | MIT |
| @fontsource-variable/onest, geist, geist-mono, jetbrains-mono | SIL Open Font License 1.1 |
