"""Deterministic offline mock judge for judge-gate self-tests.

NOT a model. score = 5 + jitter(transcript) + label_bias (only when the label
presented equals favored_label). jitter = sha256(transcript)[:8] mod 3, which is
label-independent by construction, so:

    label_bias = 0.0  -> every paired diff is exactly 0  -> gate must PASS
    label_bias = 2.0  -> every paired diff is exactly 2  -> gate must FAIL

Emits "SCORE: <int>" exactly in the format the judge prompt demands. Every call
is logged (with usage marked estimated) so receipts stay auditable in mock mode.
"""
import hashlib
import re
import time

from .client import JudgeResult


def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class MockJudgeClient:
    def __init__(self, label_bias: float = 0.0, favored_label: str = "Author A",
                 budget: int = 10 ** 6):
        self.label_bias = float(label_bias)
        self.favored_label = favored_label
        self.budget = int(budget)
        self.mock = True
        self._calls: list = []

    @property
    def call_log(self) -> list:
        return list(self._calls)

    def judge(self, model, messages, *, max_tokens=4096, temperature=0.0) -> JudgeResult:
        t0 = time.time()
        text = "\n".join(m.get("content", "") for m in messages)
        m = re.search(r'"""\n(.*?)\n"""', text, re.S)
        transcript = m.group(1) if m else text
        lm = re.search(r"Author label: (.+)", text)
        label = lm.group(1).strip() if lm else ""
        base = int(hashlib.sha256(transcript.encode("utf-8")).hexdigest()[:8], 16) % 3
        score = 5 + base + (self.label_bias if label == self.favored_label else 0.0)
        latency = round(time.time() - t0, 6)
        usage = {"prompt_tokens": max(1, len(text) // 4), "completion_tokens": 2,
                 "estimated": True}
        self._calls.append({
            "n": len(self._calls) + 1, "kind": "judge", "model": model,
            "status": 200, "latency": latency, "usage": usage,
            "finish_reason": "stop", "max_tokens": max_tokens, "attempts": 1,
            "error": None, "notes": [], "mock": True, "ts": _iso(),
        })
        return JudgeResult(ok=True, content=f"SCORE: {score:g}", model=model,
                           finish_reason="stop", usage=usage, latency=latency,
                           attempts=1, max_tokens_sent=max_tokens, error=None)
