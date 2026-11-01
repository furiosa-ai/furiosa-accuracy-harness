"""Per-model request settings: the sampling each model card recommends, plus reasoning switches.

Matched by substring on the served model name (first match wins). Every run records the settings it used in meta.json.
Override any field from the CLI (--temperature, --reasoning-effort, --no-think, --max-tokens, --param k=v).

chat: False marks a base (non-chat) model; the chat-format benchmarks are skipped for it (reported N/A).
"""

PROFILES = [
    # gpt-oss: harmony reasoning; Artificial Analysis headlines the "high" effort. temperature 1.0 per the model card.
    ("gpt-oss", dict(temperature=1.0, top_p=1.0, reasoning_effort="high", max_tokens=32768)),
    # Qwen3 hybrid thinking models: thinking on by default; card: T 0.6, top_p 0.95, top_k 20.
    ("Qwen3-VL", dict(temperature=0.7, top_p=0.8, top_k=20, max_tokens=16384)),   # Instruct (non-thinking) VL
    ("Qwen3", dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)),
    # EXAONE 4.x: reasoning mode via enable_thinking; card: T 0.6, top_p 0.95.
    ("EXAONE", dict(temperature=0.6, top_p=0.95, max_tokens=32768,
                    chat_template_kwargs={"enable_thinking": True})),
    # Gemma 4: card T 1.0, top_p 0.95, top_k 64.
    ("gemma", dict(temperature=1.0, top_p=0.95, top_k=64, max_tokens=16384)),
    ("phi-2", dict(temperature=0.0, max_tokens=2048, chat=False)),
]

DEFAULT = dict(temperature=0.0, max_tokens=16384)


def profile_for(model):
    for key, p in PROFILES:
        if key.lower() in model.lower():
            return {**DEFAULT, **p}
    return dict(DEFAULT)
