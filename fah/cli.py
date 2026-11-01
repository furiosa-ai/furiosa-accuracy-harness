"""fah run | score | report | models  — see README.md."""
import argparse
import asyncio
import glob
import json
import os
import sys
import time

import httpx

from .benches import registry
from .client import Client, load_dotenv
from .profiles import profile_for
from .runner import HEADLINE, STANDARD_LIMITS, ask, score, slug, summarise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))


def _bench_list(s):
    reg = registry()
    names = HEADLINE if s in ("all", "") else [x.strip() for x in s.split(",")]
    bad = [n for n in names if n not in reg]
    if bad:
        raise SystemExit(f"unknown bench {bad}; choose from {list(reg)}")
    return [reg[n] for n in names]


def _kv(pairs):
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _served_ctx(base_url, api_key, model):
    try:
        r = httpx.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        for m in r.json().get("data", []):
            if m["id"] == model:
                return m.get("max_model_len")
    except Exception:
        pass
    return None


def _params(a):
    prof = profile_for(a.model)
    p = {k: v for k, v in prof.items() if k != "chat"}
    if a.temperature is not None:
        p["temperature"] = a.temperature
    if a.max_tokens is not None:
        p["max_tokens"] = a.max_tokens
    if a.reasoning_effort:
        p["reasoning_effort"] = None if a.reasoning_effort == "none" else a.reasoning_effort
    if a.no_think:
        p["chat_template_kwargs"] = {**p.get("chat_template_kwargs", {}), "enable_thinking": False}
    p.update(_kv(a.param))
    return {k: v for k, v in p.items() if v is not None}, prof.get("chat", True)


def _outdir(a):
    tag = a.tag or ("partial" if (a.limit or a.limits) else "full" if a.full else "std")
    d = os.path.join(a.results, slug(a.model), tag)
    os.makedirs(d, exist_ok=True)
    return d, tag


def _repeats(a, benches):
    over = {k: int(v) for k, v in _kv(a.repeats).items()}
    return {b.name: over.get(b.name, b.repeats) for b in benches}


def _set_window(benches, window):
    if window:
        start, end = window.split(":")
        for b in benches:
            if b.name == "livecodebench":
                b.start, b.end = start, end


async def _run(a):
    benches = _bench_list(a.bench)
    _set_window(benches, a.lcb_window)
    params, is_chat = _params(a)
    client = Client(a.model, base_url=a.base_url, concurrency=a.concurrency, params=params, stream=not a.no_stream)
    ctx = _served_ctx(client.base_url, client.api_key, a.model)
    if ctx and params.get("max_tokens", 0) > ctx - 4096:
        params["max_tokens"] = client.params["max_tokens"] = max(1024, ctx - 4096)
    outdir, tag = _outdir(a)
    reps = _repeats(a, benches)
    base = {} if (a.full or a.limit) else STANDARD_LIMITS
    lim = {b.name: _kv(a.limits).get(b.name, a.limit or base.get(b.name)) for b in benches}
    meta = dict(model=a.model, tag=tag, base_url=client.base_url, served_max_model_len=ctx, params=params,
                limit=a.limit, limits=lim, seed=a.seed, repeats=reps, lcb_window=a.lcb_window or "2024-07-01:2025-01-01",
                started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    mpath = os.path.join(outdir, "meta.json")
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        if old.get("params") != params and not a.force:
            raise SystemExit(f"{mpath} was asked with different params {old.get('params')}; use another --tag or --force")
        meta["repeats"] = {**old.get("repeats", {}), **reps}
        meta["limits"] = {**old.get("limits", {}), **lim}
        meta["started"] = old.get("started", meta["started"])
    json.dump(meta, open(mpath, "w"), indent=1)
    print(f"model {a.model}  ctx {ctx}  params {params}\n-> {outdir}", flush=True)
    results = {}
    try:
        for b in benches:
            if b.chat_only and not is_chat:
                print(f"[{b.name}] skipped: base model (not instruction-tuned)")
                results[b.name] = dict(score=None, n=0, skipped="base model")
                continue
            items = b.items(lim[b.name], a.seed)
            await ask(client, b, items, reps[b.name], outdir)
            results[b.name] = await score(b, items, outdir)
            print(f"[{b.name}] score {results[b.name]['score']:.2f}  "
                  f"(n {results[b.name]['n']}, no answer {results[b.name].get('no_answer')}, "
                  f"truncated {results[b.name]['truncated']}, errors {results[b.name]['errors']})", flush=True)
    finally:
        await client.close()
    s = summarise(outdir, results, meta)
    print(json.dumps(dict(index=s["index"], complete=s["index_complete"], headline=s["headline"]), indent=1))


async def _rescore(a):
    meta = json.load(open(os.path.join(a.run, "meta.json")))
    benches = _bench_list(a.bench)
    _set_window(benches, meta.get("lcb_window"))
    results = {}
    for b in benches:
        if os.path.exists(os.path.join(a.run, f"{b.name}.responses.jsonl")):
            results[b.name] = await score(b, b.items(meta.get("limits", {}).get(b.name, meta["limit"]), meta["seed"]), a.run)
            print(f"[{b.name}] {results[b.name]['score']:.2f}")
    s = summarise(a.run, results, meta)
    print(json.dumps(dict(index=s["index"], headline=s["headline"]), indent=1))


def _report(a):
    rows = []
    for p in sorted(glob.glob(os.path.join(a.results, "*", "*", "summary.json"))):
        s = json.load(open(p))
        rows.append(s)
    cols = HEADLINE
    hdr = ["model", "tag", "index"] + cols
    lines = [",".join(hdr)]
    print(f"{'model':42} {'tag':10} {'index':>6} " + " ".join(f"{c[:12]:>12}" for c in cols))
    for s in rows:
        v = [s["headline"].get(c) for c in cols]
        idx = f"{s['index']:.1f}" + ("" if s.get("index_complete") else "*") if s["index"] is not None else "-"
        print(f"{s['model']:42} {s['tag']:10} {idx:>6} " + " ".join(f"{x:12.1f}" if x is not None else f"{'-':>12}" for x in v))
        lines.append(",".join([s["model"], s["tag"], str(s["index"])] + ["" if x is None else f"{x:.2f}" for x in v]))
    out = os.path.join(a.results, "leaderboard.csv")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"* = partial (limit, missing bench or unfinished)\n-> {out}")


def _models(a):
    key = os.environ.get("FAH_API_KEY")
    base = (a.base_url or os.environ.get("FAH_BASE_URL") or "https://endpoint.access.furiosa.dev/v1").rstrip("/")
    r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    for m in r.json().get("data", []):
        print(f"{m['id']:50} max_model_len {m.get('max_model_len')}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fah", description="Furiosa accuracy harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ask, score and summarise")
    r.add_argument("--model", required=True)
    r.add_argument("--bench", default="all", help="all or a comma list of " + ",".join(HEADLINE))
    r.add_argument("--limit", type=int, help="items per bench (stratified, deterministic); omit for the full set")
    r.add_argument("--full", action="store_true", help="every item (MMLU-Pro all 12,032) instead of the standard suite")
    r.add_argument("--limits", nargs="*", help="per-bench item limits bench=N (override --limit), e.g. mmlu_pro=2000")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--repeats", nargs="*", help="bench=k, e.g. aime_2025=8")
    r.add_argument("--tag", help="result folder name (default std, full with --full, partial with --limit/--limits)")
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--base-url")
    r.add_argument("--temperature", type=float)
    r.add_argument("--max-tokens", type=int)
    r.add_argument("--reasoning-effort", help="low | medium | high | none (gpt-oss)")
    r.add_argument("--no-think", action="store_true", help="chat_template_kwargs.enable_thinking=false (Qwen3, EXAONE)")
    r.add_argument("--param", nargs="*", help="extra request fields k=v (JSON values)")
    r.add_argument("--lcb-window", help="LiveCodeBench contest dates START:END (default 2024-07-01:2025-01-01)")
    r.add_argument("--no-stream", action="store_true", help="plain (non-streamed) requests")
    r.add_argument("--force", action="store_true", help="reuse a result folder even if the request params changed")
    r.add_argument("--results", default=os.path.join(ROOT, "results"))
    s = sub.add_parser("score", help="re-score saved responses (e.g. after an extractor fix)")
    s.add_argument("--run", required=True, help="results/<model>/<tag>")
    s.add_argument("--bench", default="all")
    p = sub.add_parser("report", help="table + results/leaderboard.csv over every run")
    p.add_argument("--results", default=os.path.join(ROOT, "results"))
    m = sub.add_parser("models", help="list the endpoint's models")
    m.add_argument("--base-url")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        asyncio.run(_run(a))
    elif a.cmd == "score":
        asyncio.run(_rescore(a))
    elif a.cmd == "report":
        _report(a)
    else:
        _models(a)


if __name__ == "__main__":
    sys.exit(main())
