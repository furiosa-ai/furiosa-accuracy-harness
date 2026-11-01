"""Ask every (item, repeat), save raw responses as they arrive (resumable), then score and summarise.

results/<model-slug>/<tag>/
  meta.json                 model, endpoint, request params, benches, limits, seed
  <bench>.responses.jsonl   one line per (id, rep): the raw reply (content, reasoning, usage, latency) or an error
  <bench>.scored.jsonl      one line per (id, rep): extracted answer, correctness, bench-specific detail
  summary.json              per-bench aggregate + the Furiosa Accuracy Index (mean of the five headline scores)
"""
import asyncio
import inspect
import json
import os
import time

HEADLINE = ["mmlu_pro", "gpqa_diamond", "ifeval", "livecodebench", "aime_2025"]
# The standard suite: MMLU-Pro on a stratified 2,000 of its 12,032 questions (seed 0, ~±1 pt), everything else full.
STANDARD_LIMITS = {"mmlu_pro": 2000}


def slug(model):
    return model.replace("/", "__")


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass          # a line cut by a crash; it is asked again
    return out


async def ask(client, bench, items, repeats, outdir, log_every=25):
    path = os.path.join(outdir, f"{bench.name}.responses.jsonl")
    done = {(r["id"], r["rep"]) for r in _read_jsonl(path) if not r.get("error")}
    todo = [(it, k) for it in items for k in range(repeats) if (it["id"], k) not in done]
    print(f"[{bench.name}] {len(items)} items x {repeats} = {len(items) * repeats}; {len(done)} saved, {len(todo)} to ask",
          flush=True)
    if not todo:
        return
    f = open(path, "a")
    t0, n, toks, errs = time.time(), 0, 0, 0

    async def one(it, k):
        nonlocal n, toks, errs
        resp = await client.chat(it["messages"])
        f.write(json.dumps(dict(id=it["id"], rep=k, **resp)) + "\n")
        f.flush()
        n += 1
        toks += (resp.get("usage") or {}).get("completion_tokens", 0)
        errs += bool(resp.get("error"))
        if n % log_every == 0 or n == len(todo):
            dt = time.time() - t0
            print(f"[{bench.name}] {n}/{len(todo)}  {dt / 60:.1f} min  {toks / max(dt, 1e-9):.0f} out tok/s  errors {errs}",
                  flush=True)

    try:
        await asyncio.gather(*(one(it, k) for it, k in todo))
    finally:
        f.close()


async def score(bench, items, outdir):
    by_id = {it["id"]: it for it in items}
    # last successful reply per (id, rep); errors count as wrong (no answer) so a flaky endpoint cannot inflate a score
    resp = {}
    for r in _read_jsonl(os.path.join(outdir, f"{bench.name}.responses.jsonl")):
        if r["id"] in by_id and (not r.get("error") or (r["id"], r["rep"]) not in resp):
            resp[(r["id"], r["rep"])] = r
    rows = []

    async def one(key, r):
        it = by_id[key[0]]
        if r.get("error"):
            s = dict(pred=None, correct=0.0, error=r["error"][:200])
        else:
            s = bench.score(it, r)
            if inspect.isawaitable(s):
                s = await s
        s.update(id=key[0], rep=key[1], finish_reason=r.get("finish_reason"),
                 completion_tokens=(r.get("usage") or {}).get("completion_tokens"))
        rows.append(s)

    await asyncio.gather(*(one(k, r) for k, r in sorted(resp.items())))
    rows.sort(key=lambda s: (str(s["id"]), s["rep"]))
    with open(os.path.join(outdir, f"{bench.name}.scored.jsonl"), "w") as f:
        for s in rows:
            f.write(json.dumps(s) + "\n")
    agg = bench.aggregate(rows) if rows else dict(score=None, n=0)
    toks = [s["completion_tokens"] for s in rows if s.get("completion_tokens")]
    agg.update(items=len(items), answered=len(rows), errors=sum(1 for s in rows if s.get("error")),
               truncated=sum(1 for s in rows if s.get("finish_reason") == "length"),
               mean_completion_tokens=round(sum(toks) / len(toks)) if toks else None)
    return agg


def summarise(outdir, results, meta):
    path = os.path.join(outdir, "summary.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    benches = {**prev.get("benches", {}), **results}
    head = {b: benches[b]["score"] for b in HEADLINE if b in benches and benches[b].get("score") is not None}
    s = dict(model=meta["model"], tag=meta["tag"], updated=time.strftime("%Y-%m-%dT%H:%M:%S"),
             index=round(sum(head.values()) / len(head), 2) if head else None,
             index_complete=len(head) == len(HEADLINE) and all(benches[b].get("n") == benches[b].get("items") *
                                                               meta["repeats"].get(b, 1) for b in head)
                                and all((meta.get("limits") or {}).get(b) in (None, STANDARD_LIMITS.get(b))
                                        for b in head) and meta.get("seed", 0) == 0,
             headline={b: round(v, 2) for b, v in head.items()}, benches=benches)
    json.dump(s, open(path, "w"), indent=1)
    return s
