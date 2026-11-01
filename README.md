# furiosa-accuracy-harness (`fah`)

How smart is each model as served on RNGD? Five benchmarks, asked through any OpenAI-compatible
`/v1/chat/completions` endpoint (the public RNGD endpoint, a local `furiosa-llm serve`, or vLLM for a GPU reference),
scored locally. The **Furiosa Accuracy Index** is the plain mean of the five headline scores (0–100).

| Bench | Data | Items | Prompt | Headline score |
|---|---|---|---|---|
| `mmlu_pro` | TIGER-Lab/MMLU-Pro test | 2,000 stratified of 12,032 (14 subjects, ≤10 options; `--full` for all) | zero-shot CoT, `Answer: X` | accuracy |
| `gpqa_diamond` | Idavidrein/gpqa diamond (gated) | 198 (4 options, order fixed per question) | zero-shot CoT, `Answer: X` | accuracy |
| `ifeval` | google/IFEval | 541 | the prompt as is | prompt-level strict (loose and instruction-level also saved) |
| `livecodebench` | livecodebench/code_generation_lite, contests 2024-07-01..2025-01-01 | 260 | LiveCodeBench repo prompts | pass@1 on public + private tests |
| `aime_2025` | MathArena/aime_2025 (I + II) | 30 × 4 samples | `\boxed{}` integer | accuracy over all samples |

The prompt style follows Artificial Analysis / simple-evals, so numbers sit next to theirs, but they are **not** the
AA Intelligence Index (a different, larger mix). Base (non-chat) models skip IFEval and LiveCodeBench.

## Setup

```bash
git clone https://github.com/furiosa-ai/furiosa-accuracy-harness.git && cd furiosa-accuracy-harness
cp .env.example .env          # put your key in FAH_API_KEY; FAH_BASE_URL defaults to the public RNGD endpoint. .env is gitignored
uv sync                       # or: python -m venv .venv && .venv/bin/pip install -e .
uv run fah models             # the key works and the endpoint answers
uv run fah run --model furiosa-ai/gpt-oss-120b --limit 5 --tag smoke
```

Inside a Dropbox folder, keep the venv outside it (Dropbox evicts venv files): `UV_PROJECT_ENVIRONMENT=~/.venvs/fah uv sync`.
Needs Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/) (or pip).

GPQA is gated: accept its terms once at https://huggingface.co/datasets/Idavidrein/gpqa with the account whose token is in
`~/.cache/huggingface/token`. LiveCodeBench downloads ~1.8 GB once (Hugging Face cache) and caches the filtered problems
in `~/.cache/fah/`.

## Run

```bash
F="uv run fah"
$F models                                                    # what the endpoint serves
$F run --model furiosa-ai/gpt-oss-120b --limit 5 --tag smoke # quick check: 5 items per bench
$F run --model furiosa-ai/gpt-oss-120b                       # standard suite -> results/furiosa-ai__gpt-oss-120b/std/
$F run --model furiosa-ai/gpt-oss-120b --full                # MMLU-Pro on all 12,032 too -> .../full/
$F report                                                    # table + results/leaderboard.csv
```

- **Resumable:** every reply is appended to `<bench>.responses.jsonl` as it arrives. Re-running the same command asks only
  what is missing; failed requests are asked again.
- **Re-score without asking again** (e.g. after an extractor fix): `$F score --run results/<model>/<tag>`.
- **Request settings** come from `fah/profiles.py` (the sampling each model card recommends; gpt-oss at
  `reasoning_effort=high`, Qwen3/EXAONE thinking on). Override them with `--temperature`, `--max-tokens`,
  `--reasoning-effort`, `--no-think` or `--param k=v`. A run folder refuses a second set of params unless you pass `--force`;
  use a new `--tag` for a variant (e.g. `--no-think --tag nothink`). `max_tokens` is capped at the served context − 4K.
- **Load:** `--concurrency 16` by default. The public endpoint is shared, so keep it modest.

Cost: reasoning models write 2–20K tokens per item. On gpt-oss-120b at high effort, one request runs at ~65 tok/s, so a
full MMLU-Pro (12K items) takes more than a day, so the standard suite asks a stratified 2,000 (seed 0, ~±1 pt).

## All models of the public endpoint

```bash
bash scripts/run-endpoint-models.sh      # gpt-oss-120b, Qwen3-32B, Qwen3-VL-32B, Gemma-4-31B in parallel tmux sessions
```

The public endpoint's gateway must allow long requests: a reasoning reply of 15–30K tokens runs past 5 minutes, and a gateway
that cuts requests at 300 s turns those items into errors (scored wrong). The client streams, retries transient failures,
and stops retrying an item that was cut twice after a long generation; `summary.json` counts such errors per bench.

## Output

`results/<model>/<tag>/`: `meta.json` (endpoint, params, limits, repeats, seed), `<bench>.responses.jsonl` (raw replies
with reasoning, usage, latency), `<bench>.scored.jsonl` (extracted answer, correctness), `summary.json` (per-bench
aggregate, truncation and error counts, mean completion tokens, `index`, `index_complete`). An index is marked
complete when all five benches ran the standard suite (or the full sets) with seed 0.

## Published scores to check against

`refs/README.md` lists every publicly reported score for the RNGD model zoo on the five benchmarks (model cards, tech
reports, Artificial Analysis, MathArena, Open LLM Leaderboard), marks which ones match this harness's settings, and gives
the run flags per model and mode. Raw rows with source URLs: `refs/published_scores.jsonl`.

## Safety of the code benchmark

LiveCodeBench runs model-written Python. Each test runs as `python -I` in a fresh temporary folder, with CPU-time and
file-size limits and a 6 s timeout, and without the API key in its environment. On macOS it also runs under
`sandbox-exec`, which denies all network access and any file write outside that folder. This contains honest mistakes;
it is not a hardened jail. Never run it as root. LiveCodeBench's private tests are pickled; they are loaded with an
unpickler that refuses every global.

## Files

`fah/client.py` (async client, retries), `fah/profiles.py`, `fah/benches/*.py` (one file per benchmark),
`fah/extract.py` (answer parsing), `fah/sandbox.py`, `fah/runner.py` (ask / score / summary), `fah/cli.py`,
`fah/vendor/ifeval/` (Google's IFEval checker, Apache-2.0, imports made package-relative).
