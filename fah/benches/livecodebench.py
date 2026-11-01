"""LiveCodeBench code generation (livecodebench/code_generation_lite), pass@1 over public + private tests.

Default window 2024-07-01 .. 2025-01-01 (the 315-problem set Artificial Analysis reports), read from release files
test4.jsonl + test5.jsonl (~1.8 GB download once into the Hugging Face cache; the filtered problems are cached in
~/.cache/fah/). Prompts follow the LiveCodeBench repo (stdin programs vs LeetCode-style starter code).

Private tests ship as base64(zlib(pickle(json str))). They are unpickled with a loader that refuses every global, so only
plain str/list/dict payloads can load: a tampered file cannot run code through pickle.
"""
import asyncio
import base64
import io
import json
import os
import pickle
import zlib
from decimal import Decimal, InvalidOperation

from . import Bench, hf_file, subsample
from ..extract import last_code_block
from ..sandbox import run_python

SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will generate a "
          "correct Python program that matches the specification and passes all tests.")
FMT_STDIN = ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test "
             "on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program "
             "runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n")
FMT_STARTER = ("### Format: You will use the following starter code to write the solution to the problem and enclose your "
               "code within delimiters.\n```python\n{starter}\n```\n\n")
ANSWER = "### Answer: (use the provided format with backticks)\n\n"

PRELUDE = ("import sys, os, math, re, string, random, bisect, heapq, functools, itertools, collections, operator, copy\n"
           "from typing import *\nfrom collections import *\nfrom itertools import *\nfrom functools import *\n"
           "from heapq import *\nfrom bisect import *\nfrom math import *\n"
           "from builtins import *  # last, as LiveCodeBench does: math's star import must not shadow pow/sum/etc.\n"
           "sys.setrecursionlimit(1 << 20)\n")

CACHE = os.path.expanduser("~/.cache/fah")
FILES = ("test4.jsonl", "test5.jsonl")
TEST_TIMEOUT_S = 6.0
EXEC_PARALLEL = max(2, (os.cpu_count() or 4) // 2)


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def _private_tests(s):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        raw = _NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(s.encode())))).load()
        return json.loads(raw)


def _load(start, end):
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"lcb_{start}_{end}.jsonl")
    if not os.path.exists(cache):
        rows = []
        for fn in FILES:
            for line in open(hf_file("livecodebench/code_generation_lite", fn), encoding="utf-8"):
                r = json.loads(line)
                if not (start <= r["contest_date"][:10] < end):
                    continue
                tests = json.loads(r["public_test_cases"]) + _private_tests(r["private_test_cases"])
                meta = json.loads(r.get("metadata") or "{}")
                rows.append(dict(id=r["question_id"], title=r["question_title"], platform=r["platform"],
                                 difficulty=r["difficulty"], contest_date=r["contest_date"][:10],
                                 content=r["question_content"], starter=r.get("starter_code") or "",
                                 func_name=meta.get("func_name"), tests=tests))
        with open(cache + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(cache + ".tmp", cache)
    return [json.loads(l) for l in open(cache)]


def _messages(p):
    user = f"### Question:\n{p['content']}\n\n"
    user += FMT_STARTER.format(starter=p["starter"]) if p["starter"].strip() else FMT_STDIN
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user + ANSWER}]


def _same_stdout(pred, gold):
    p = [l.strip() for l in pred.strip().split("\n")]
    g = [l.strip() for l in gold.strip().split("\n")]
    if len(p) != len(g):
        return False
    for a, b in zip(p, g):
        if a == b:
            continue
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            return False
        for x, y in zip(ta, tb):
            if x == y:
                continue
            try:
                if Decimal(x) != Decimal(y):
                    return False
            except InvalidOperation:
                return False
    return True


def _norm(v):
    return json.loads(json.dumps(v))      # tuples -> lists, as the reference grader compares


def _functional_program(code, func_name):
    return (PRELUDE + code + "\n\nimport json as _j, sys as _s\n"
            "_a = [_j.loads(x) for x in _s.stdin.read().strip().split('\\n') if x.strip()]\n"
            f"_r = Solution().{func_name}(*_a)\nprint(_j.dumps(_r))\n")


class LiveCodeBench(Bench):
    name = "livecodebench"
    chat_only = True

    def __init__(self, start="2024-07-01", end="2025-01-01"):
        self.start, self.end = start, end
        self.exec_sem = None

    def items(self, limit=None, seed=0):
        out = []
        for p in _load(self.start, self.end):
            out.append(dict(id=p["id"], messages=_messages(p), difficulty=p["difficulty"], platform=p["platform"],
                            func_name=p["func_name"], functional=bool(p["starter"].strip()), tests=p["tests"]))
        return subsample(out, limit, seed, key=lambda x: x["difficulty"])

    async def score(self, item, resp):
        if self.exec_sem is None:
            self.exec_sem = asyncio.Semaphore(EXEC_PARALLEL)
        base = dict(difficulty=item["difficulty"])
        code = last_code_block(resp.get("content") or "")
        if not code:
            return dict(base, pred=None, correct=0.0, passed=0, total=len(item["tests"]), fail="no code block")
        prog = _functional_program(code, item["func_name"]) if item["functional"] else PRELUDE + code
        passed = 0
        async with self.exec_sem:
            for t in item["tests"]:
                status, out, err = await run_python(prog, t["input"], TEST_TIMEOUT_S)
                if status != "ok":
                    return dict(base, pred="code", correct=0.0, passed=passed, total=len(item["tests"]),
                                fail=status + (": " + err.strip().splitlines()[-1] if err.strip() else ""))
                if item["functional"]:
                    try:
                        ok = _norm(json.loads(out.strip().splitlines()[-1])) == _norm(json.loads(t["output"]))
                    except (json.JSONDecodeError, IndexError):
                        ok = False
                else:
                    ok = _same_stdout(out, t["output"])
                if not ok:
                    return dict(base, pred="code", correct=0.0, passed=passed, total=len(item["tests"]), fail="wrong answer")
                passed += 1
        return dict(base, pred="code", correct=1.0, passed=passed, total=len(item["tests"]))

    def aggregate(self, rows):
        s = super().aggregate(rows)
        d = {}
        for r in rows:
            d.setdefault(r["difficulty"], []).append(r["correct"])
        s["by_difficulty"] = {k: round(100.0 * sum(v) / len(v), 2) for k, v in sorted(d.items())}
        return s
