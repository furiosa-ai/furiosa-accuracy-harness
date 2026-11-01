"""MMLU-Pro (TIGER-Lab/MMLU-Pro test, 12,032 questions, up to 10 options, 14 subjects). Zero-shot CoT."""
import pyarrow.parquet as pq

from . import Bench, hf_file, subsample
from .mc import mc_messages, mc_score


class MMLUPro(Bench):
    name = "mmlu_pro"

    def items(self, limit=None, seed=0):
        t = pq.read_table(hf_file("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet")).to_pylist()
        out = []
        for r in t:
            opts = list(r["options"])
            out.append(dict(id=str(r["question_id"]), category=r["category"], messages=mc_messages(r["question"], opts),
                            gold=r["answer"], n_options=len(opts)))
        return subsample(out, limit, seed, key=lambda x: x["category"])

    def score(self, item, resp):
        return {**mc_score(item, resp), "category": item["category"]}

    def aggregate(self, rows):
        s = super().aggregate(rows)
        cats = {}
        for r in rows:
            cats.setdefault(r["category"], []).append(r["correct"])
        s["by_category"] = {k: round(100.0 * sum(v) / len(v), 2) for k, v in sorted(cats.items())}
        return s
