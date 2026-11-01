"""GPQA Diamond (Idavidrein/gpqa, 198 questions, 4 options). Zero-shot CoT; options shuffled per question with a fixed
seed so every model sees the same order. The dataset is gated: accept its terms on huggingface.co once with the account
whose token is in ~/.cache/huggingface/token."""
import csv
import random

from . import Bench, hf_file, stable_seed, subsample
from .mc import LETTERS, mc_messages, mc_score


class GPQADiamond(Bench):
    name = "gpqa_diamond"

    def items(self, limit=None, seed=0):
        path = hf_file("Idavidrein/gpqa", "gpqa_diamond.csv")
        out = []
        for r in csv.DictReader(open(path, encoding="utf-8")):
            correct = r["Correct Answer"].strip()
            opts = [correct, r["Incorrect Answer 1"].strip(), r["Incorrect Answer 2"].strip(), r["Incorrect Answer 3"].strip()]
            qid = r["Record ID"]
            random.Random(stable_seed("gpqa", qid)).shuffle(opts)
            out.append(dict(id=qid, domain=r.get("High-level domain", ""), messages=mc_messages(r["Question"], opts),
                            gold=LETTERS[opts.index(correct)], n_options=4))
        return subsample(out, limit, seed)

    def score(self, item, resp):
        return mc_score(item, resp)
