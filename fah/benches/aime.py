"""AIME 2025 (MathArena/aime_2025: AIME I + II, 30 problems, integer answers 0-999). Several samples per problem
(default 4) because 30 questions alone swing the score by ~3 points per question."""
import pyarrow.parquet as pq

from . import Bench, hf_file, subsample
from ..extract import integer_answer

PROMPT = ("Solve the following math problem. Put your final answer within \\boxed{{}}. "
          "The answer is an integer between 0 and 999 inclusive.\n\n{problem}")


class AIME2025(Bench):
    name = "aime_2025"
    repeats = 4

    def items(self, limit=None, seed=0):
        t = pq.read_table(hf_file("MathArena/aime_2025", "data/train-00000-of-00001.parquet")).to_pylist()
        out = [dict(id=str(r["problem_idx"]), messages=[{"role": "user", "content": PROMPT.format(problem=r["problem"])}],
                    gold=int(r["answer"])) for r in t]
        return subsample(out, limit, seed)

    def score(self, item, resp):
        pred = integer_answer(resp.get("content") or "")
        return dict(pred=pred, gold=item["gold"], correct=float(pred == item["gold"]))
