"""Shared zero-shot chain-of-thought multiple-choice prompt (the Artificial Analysis / simple-evals style)."""
from ..extract import choice_letter

LETTERS = "ABCDEFGHIJ"

TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be in the following "
    "format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.\n\n"
    "{question}\n\n{options}"
)


def mc_messages(question, options):
    letters = LETTERS[: len(options)]
    opts = "\n".join(f"{l}) {o}" for l, o in zip(letters, options))
    return [{"role": "user", "content": TEMPLATE.format(letters=", ".join(letters), question=question.strip(), options=opts)}]


def mc_score(item, resp):
    letters = LETTERS[: item["n_options"]]
    pred = choice_letter(resp.get("content") or "", letters)
    return dict(pred=pred, gold=item["gold"], correct=float(pred == item["gold"]))
