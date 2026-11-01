"""IFEval (google/IFEval, 541 prompts with verifiable instructions). Scored with Google's own checker
(fah/vendor/ifeval, Apache-2.0), strict and loose, prompt-level and instruction-level, exactly as evaluation_lib.py.
Headline score = prompt-level strict accuracy. Only the final answer is checked, never the reasoning."""
import json

from . import Bench, hf_file, subsample
from ..extract import strip_think


def _ensure_nltk():
    import nltk
    for res, path in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)


def _follows(instruction_ids, kwargs_list, prompt, response):
    from ..vendor.ifeval import instructions_registry as reg
    out = []
    for iid, kw in zip(instruction_ids, kwargs_list):
        ins = reg.INSTRUCTION_DICT[iid](iid)
        ins.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
        args = ins.get_instruction_args()
        if args and "prompt" in args:
            ins.build_description(prompt=prompt)
        out.append(bool(response.strip()) and bool(ins.check_following(response)))
    return out


def _loose_variants(r):
    lines = r.split("\n")
    rs = [r, "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    return rs + [x.replace("*", "") for x in rs]


class IFEval(Bench):
    name = "ifeval"
    chat_only = True

    def items(self, limit=None, seed=0):
        out = []
        for line in open(hf_file("google/IFEval", "ifeval_input_data.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            out.append(dict(id=str(r["key"]), messages=[{"role": "user", "content": r["prompt"]}], prompt=r["prompt"],
                            instruction_id_list=r["instruction_id_list"], kwargs=r["kwargs"]))
        return subsample(out, limit, seed)

    def score(self, item, resp):
        _ensure_nltk()
        text = strip_think(resp.get("content") or "")
        ids, kws, p = item["instruction_id_list"], item["kwargs"], item["prompt"]
        strict = _follows(ids, kws, p, text)
        loose = [any(v) for v in zip(*[_follows(ids, kws, p, x) for x in _loose_variants(text)])]
        return dict(pred=text[:80] if text.strip() else None, correct=float(all(strict)),
                    strict=strict, loose=loose)

    def aggregate(self, rows):
        n = len(rows)
        inst_s = [x for r in rows for x in r["strict"]]
        inst_l = [x for r in rows for x in r["loose"]]
        pct = lambda xs: round(100.0 * sum(xs) / len(xs), 2) if xs else 0.0
        return dict(score=pct([all(r["strict"]) for r in rows]), n=n,
                    no_answer=sum(1 for r in rows if r.get("pred") is None),
                    prompt_strict=pct([all(r["strict"]) for r in rows]), inst_strict=pct(inst_s),
                    prompt_loose=pct([all(r["loose"]) for r in rows]), inst_loose=pct(inst_l))
