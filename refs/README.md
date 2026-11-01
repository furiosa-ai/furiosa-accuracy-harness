# Published scores to verify

Every number publicly reported for the RNGD model zoo on the five headline benchmarks, collected 2026-11-02 from
model cards, vendor tech reports and blogs, Artificial Analysis (AA), MathArena and the Open LLM Leaderboard v2 (OLL).
Each number was read from a fetched page; nothing is from memory.

- `published_scores.jsonl`: 294 rows (240 with a value). Fields: model, hf_repo, bench, value, variant, mode,
  checkpoint_precision, shots_sampling, source_type, url, evidence, comparable, note.
- `comparable: true` means the source's setting matches this harness: same benchmark and mode, 0-shot generative, and
  for LiveCodeBench the AA window 2024-07..2025-01. Rows with `false` are kept for context:
  - IFBench instead of IFEval
  - LCB v5/v6 windows
  - AIME 2026
  - 5-shot or loglikelihood MMLU-Pro/GPQA
  - IFEval averaged over strict and loose

## Matrix: the best directly comparable number per cell

Notation: `vendor / AA`. Cells marked `?` have a published number whose exact metric is not stated (see notes).
`—` means nothing comparable is published.

| Model (mode) | MMLU-Pro | GPQA Diamond | IFEval (prompt-strict) | LiveCodeBench (AA window) | AIME 2025 |
|---|---|---|---|---|---|
| **contrib ports** | | | | | |
| Nemotron-3.5-Lightning NVFP4 (reasoning) | 81.62 | 75.57 (8 samples) | — | — | — |
| Nemotron-3.5-Bolt-E4B | — | — | — | — | — |
| Sarvam-30B | 80.0 | 66.5 / 63.3 | — | — | 88.3 or 80.0 (conflict) |
| Gemma-4-E4B (thinking) | 69.4 (label conflict) | 58.6 / 57.6 | 90.0? (Liquid AI) | — | — |
| Nanbeige4.3-33B (thinking) | — | 89.5 (BF16) | — | — | — |
| flint | — | — | 95.1? | — | — |
| LFM3-1.7B (reasoning) | — | 63.4 | 96.7? | — | — |
| phi-2 (base) | — | — | — | — | — |
| **SDK 2026.4** | | | | | |
| gpt-oss-120b (high) | — | 80.1 / 78.2 | — | AA 87.8 | 92.5 / 93.4 / MathArena 90.0 |
| gpt-oss-120b (low) | — | 67.1 / 67.2 | — | AA 70.7 | 50.4 / 66.7 |
| gpt-oss-20b (high) | — | 71.5 / 68.8 | — | AA 77.7 | 91.7 / 89.3 / MathArena 89.2 |
| gpt-oss-20b (low) | — | 56.8 / 61.1 | — | AA 65.2 | 37.1 / 62.3 |
| Qwen3-32B (thinking) | 79.1 | 68.4 / 66.8 | 85.0 | AA 54.6 | 72.9 / 73.0 |
| Qwen3-32B (non-thinking) | 71.9 | 54.6 / 53.5 | 83.2 | AA 28.8 | 20.2 / 19.7 |
| Qwen3-8B (thinking) | 74.6 | 62.0 / 58.9 | 85.0 | AA 40.6 | 67.3 / 19.0 |
| Qwen3-8B (non-thinking) | 63.4 | 39.3 / 45.2 | 83.0 | AA 20.2 | 20.9 / 24.3 |
| Qwen3-4B (thinking) | 70.4 | 55.9 / 52.2 | 81.9 | AA 46.5 | 65.6 / 22.3 |
| Qwen3-4B (non-thinking) | 58.0 | 41.7 / 39.8 | 81.2 | AA 23.3 | 19.1 |
| Qwen3-30B-A3B-Instruct-2507 | 78.4 | 70.4 / 65.9 | 84.7? | AA 51.5 | 61.3 / 66.3 |
| Qwen3-VL-32B-Instruct | 78.6 | 68.9 / 67.1 | 84.7? | AA 51.4 | 66.2 / 68.3 |
| Qwen3-VL-4B-Instruct | 67.1 | 55.9 / 37.1 | 82.3? | AA 29.0 | 46.6 / 37.0 |
| Qwen3-VL-2B-Instruct | 49.0 | 42.0 | 68.2? | — | 22.2 |
| Gemma-4-31B (thinking) | 85.2 (label conflict) | 84.3 / 85.7 | — | — | — |
| Gemma-4-31B (non-thinking) | — | AA 76.3 | — | — | — |
| EXAONE-4.5-33B (reasoning) | 83.3 | 80.5 / 79.4 | 89.6? | — | 92.9 |
| Mistral-Nemo-12B | — | — | OLL 58.8 | — | — |
| Llama-3.3-70B | — | 50.5 / 49.8 | OLL 88.0 | AA 28.8 | AA 7.7 |
| Llama-3.1-8B | — | 31.8 / 25.9 | — | AA 11.6 | AA 4.3 |

### Notes on the matrix

- **Comparable but not identical settings.** Vendor AIME numbers average more samples than our 4: Qwen3 uses 64 and
  AA uses 10. Vendor GPQA numbers often average several samples: Qwen3 uses 10 and Nemotron 8. Use `--repeats` to
  match.
- **IFEval `?` cells.** The metric is not stated. Most vendors mean prompt-strict, but compare against all four of the
  harness's IFEval numbers.
- **LiveCodeBench.** Vendors report only v5 (2410-2502) or v6 (2502-2505). Those windows need release file test6,
  which the harness does not load yet, so only AA's numbers are comparable.
- **Gemma-4 "MMLU Pro".** The Hugging Face card labels it MMLU Pro, while deepmind.google labels the same number
  MMMLU. Measuring it settles which label is right.
- **Sarvam-30B AIME 2025.** The card says 88.3, but the blog table says 80.0. Note that 88.3 is also the 105B's score.
- **Nemotron-3.5-Lightning GPQA.** NGC lists 76.89 for the BF16 checkpoint, while the HF card lists 75.44.
- **Llama-3.1-8B GPQA.** The card's 30.4 is the GPQA main set, not Diamond. The Diamond CoT number (31.8) appears in
  the Llama-3.3 card's comparison column.
- **Meta's IFEval.** Meta reports 92.1 for Llama-3.3-70B and 80.4 for Llama-3.1-8B, but these are the mean of
  prompt/instruction × strict/loose. That is not comparable, so the OLL prompt-strict number is used instead.
- **Large source disagreements.** These are where a measurement is most informative:
  - Qwen3-8B and 4B thinking AIME 2025: vendor 67.3 vs AA 19.0, and vendor 65.6 vs AA 22.3.
  - Qwen3-VL-4B GPQA: vendor 55.9 vs AA 37.1.
  - gpt-oss low-effort AIME: OpenAI 50.4 vs AA 66.7.

## Verification runs

Use the standard suite in every run, with each mode in its own `--tag`. Pass `--model` the served name.

These are the 95 % sampling half-widths at the harness's default size. A result outside the band, against every
source, points at the port, not at noise.

| Bench | Default size | 95 % half-width |
|---|---|---|
| MMLU-Pro | 2,000 | ±2 |
| GPQA Diamond | 198 × 1 | ±7 |
| GPQA Diamond | 198 × 8 | ±4 to ±7 (repeats of the same question are correlated) |
| IFEval | 541 | ±4 |
| LiveCodeBench | 260 | ±6 |
| AIME 2025 | 30 × 4 | ±10 to ±18 (only 30 problems) |

| Run | Flags | Cells to check |
|---|---|---|
| Nemotron-3.5-Lightning | `--temperature 1.0 --param top_p=0.95 --repeats gpqa_diamond=8` | MMLU-Pro 81.62 (**reproduced: 81.6** on RNGD, 2026-11-02), GPQA 75.57 |
| Nanbeige4.3-33B | `--temperature 0.6 --param top_p=0.95 --bench gpqa_diamond` | GPQA 89.5 |
| Sarvam-30B | (card sampling) | MMLU-Pro 80.0, GPQA 66.5/63.3, AIME 88.3 vs 80.0 |
| Gemma-4-E4B, Gemma-4-31B | thinking on (template switch to confirm) | MMLU-Pro 69.4 / 85.2 (label check), GPQA |
| LFM3-1.7B | default | GPQA 63.4, IFEval 96.7 |
| flint | `--temperature 1.0 --param top_p=0.95 top_k=20` | IFEval 95.1 |
| gpt-oss-120b / 20b | default (high) | GPQA, LCB, AIME (see matrix) |
| gpt-oss-120b / 20b, low | `--reasoning-effort low --tag std-low` | GPQA, LCB, AIME |
| Qwen3-32B / 8B / 4B, thinking | default | all five |
| Qwen3-32B / 8B / 4B, non-thinking | `--no-think --temperature 0.7 --param top_p=0.8 --tag std-nothink` | all five |
| Qwen3-30B-A3B-Instruct-2507 | `--temperature 0.7 --param top_p=0.8` (card; the Qwen3 profile would apply thinking sampling) | all five |
| Qwen3-VL-32B / 4B / 2B | default (Qwen3-VL profile) | all five except 2B LCB |
| EXAONE-4.5-33B | default (reasoning on) | MMLU-Pro, GPQA, IFEval, AIME |
| Llama-3.3-70B, Llama-3.1-8B | default | GPQA, IFEval (3.3), LCB, AIME |
| Mistral-Nemo-12B | default | IFEval 58.8 |

Nothing comparable is published for Nemotron-3.5-Bolt-E4B or phi-2. For those two, the harness numbers are the
reference.
