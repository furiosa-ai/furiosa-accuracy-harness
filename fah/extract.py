"""Answer extraction from free-form model output."""
import re

# 'Answer: X' with markdown / LaTeX wrapping: **D**, (D), $D$, $\boxed{D}$, \text{D}, \textbf{D}
_ANSWER = re.compile(r"(?i)answer\s*(?:is)?\s*[:：]\s*(?:[\*\s\$\(]|\\\(|\\boxed\{|\\text(?:bf)?\{|\\mathrm\{)*([A-J])(?![A-Za-z])")
_BOXED_LETTER = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-J])\)?\s*\}?\s*\}")


def strip_think(text):
    """Drop a leaked <think>...</think> block (servers that do not split reasoning out)."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


def choice_letter(text, letters):
    """Last 'Answer: X' (X in letters); then a boxed letter; then a lone letter on the last non-empty line."""
    text = strip_think(text)
    for pat in (_ANSWER, _BOXED_LETTER):
        hits = [m.group(1).upper() for m in pat.finditer(text)]
        hits = [h for h in hits if h in letters]
        if hits:
            return hits[-1]
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        m = re.fullmatch(r"[\*\(\s]*([A-J])[\)\.\*\s]*", lines[-1])
        if m and m.group(1) in letters:
            return m.group(1)
    return None


def last_boxed(text):
    """Content of the last \\boxed{...} (balanced braces), or None."""
    text = strip_think(text)
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def integer_answer(text):
    """AIME answer: the last boxed value as an integer; falls back to 'answer is N' / last integer on the last line."""
    b = last_boxed(text)
    cands = []
    if b is not None:
        cands.append(b)
    t = strip_think(text)
    m = list(re.finditer(r"(?i)answer\s*(?:is)?\s*[:：]?\s*\$?\s*(\d{1,4})", t))
    if m:
        cands.append(m[-1].group(1))
    for c in cands:
        c = re.sub(r"\\(?:text|mathrm|textbf)\{([^}]*)\}", r"\1", c)
        c = c.replace(",", "").replace("\\!", "").replace("$", "").strip()
        n = re.fullmatch(r"\s*0*(\d+)\s*(?:\^\\circ|°)?\s*", c)
        if n:
            return int(n.group(1))
    return None


def last_code_block(text):
    """Last ```python fenced block (else the last fenced block of any language)."""
    text = strip_think(text)
    blocks = re.findall(r"```(\w*)[^\n]*\n(.*?)```", text, flags=re.S)
    if not blocks:
        return None
    py = [b for lang, b in blocks if lang.lower() in ("python", "python3", "py")]
    return (py or [b for _, b in blocks])[-1]
