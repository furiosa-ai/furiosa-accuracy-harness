"""Benchmark interface.

A bench yields items {id, messages, ...gold fields}; score(item, resp) -> {"correct": 0..1, ...}; aggregate(rows) -> summary
with "score" in percent. Items are fixed by (limit, seed) so a resumed or repeated run asks the same questions.
"""
import hashlib
import random

from huggingface_hub import hf_hub_download


def hf_file(repo, filename, revision=None):
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", revision=revision)


def subsample(items, limit, seed, key=None):
    """Deterministic subsample; stratified by key (round-robin over strata) so small limits keep the category mix."""
    if not limit or limit >= len(items):
        return items
    rng = random.Random(seed)
    if key is None:
        return sorted(rng.sample(items, limit), key=lambda x: str(x["id"]))
    strata = {}
    for it in items:
        strata.setdefault(key(it), []).append(it)
    for v in strata.values():
        rng.shuffle(v)
    out, i = [], 0
    keys = sorted(strata)
    while len(out) < limit:
        for k in keys:
            if i < len(strata[k]) and len(out) < limit:
                out.append(strata[k][i])
        i += 1
    return out


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class Bench:
    name = ""
    chat_only = False      # needs an instruction-tuned chat model (skipped for base models)
    repeats = 1            # default samples per item (AIME uses several: 30 questions are noisy)

    def items(self, limit=None, seed=0):
        raise NotImplementedError

    def score(self, item, resp):
        raise NotImplementedError

    def aggregate(self, rows):
        n = len(rows)
        acc = sum(r["correct"] for r in rows) / n if n else 0.0
        return dict(score=100.0 * acc, n=n, no_answer=sum(1 for r in rows if r.get("pred") is None))


def registry():
    from .aime import AIME2025
    from .gpqa import GPQADiamond
    from .ifeval import IFEval
    from .livecodebench import LiveCodeBench
    from .mmlu_pro import MMLUPro
    return {b.name: b for b in (MMLUPro(), GPQADiamond(), IFEval(), LiveCodeBench(), AIME2025())}
