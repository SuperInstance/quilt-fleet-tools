"""Judge backends: a JudgeClient Protocol plus the DeepInfra OpenAI-compatible client.

Conventions (mirroring scripts/quilt-cell-lab/client.py):
- HARD model-id ALLOWLIST, enforced before any HTTP is attempted.
- <= 90 judge calls per invocation; EVERY call (including transport retries and
  the one length-retry) is logged with latency + usage.
- max_tokens default 4096; ONE retry at 1.5x max_tokens when finish_reason ==
  "length" (hard cap 8192). The retry call is a real call: budgeted and logged.
- Key from env DEEPINFRA_API_KEY (or api_key= argument). Never printed, never
  written to receipts.

The Protocol exists so all sealing/gating logic (experiment.py, seal.py, the
tests) runs fully offline against any object with the same two members.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import requests

from .allowlist import assert_allowed

DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
BUDGET = 90
DEFAULT_MAX_TOKENS = 4096
LENGTH_RETRY_CAP = 8192
RETRY_STATUSES = (429, 500, 502, 503, 504)


class BudgetExhausted(RuntimeError):
    pass


def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class JudgeResult:
    ok: bool
    content: str
    model: str
    finish_reason: str | None
    usage: dict
    latency: float
    attempts: int
    max_tokens_sent: int
    error: str | None = None
    notes: list = field(default_factory=list)


@runtime_checkable
class JudgeClient(Protocol):
    """Any judge backend: judge() per call; call_log with one entry per call."""

    def judge(self, model: str, messages: list, *,
              max_tokens: int = DEFAULT_MAX_TOKENS,
              temperature: float = 0.0) -> JudgeResult: ...

    @property
    def call_log(self) -> list: ...


class DeepInfraJudgeClient:
    """DeepInfra chat-completions judge client (OpenAI-compatible endpoint)."""

    def __init__(self, api_key: str | None = None, url: str = DEEPINFRA_URL,
                 budget: int = BUDGET, session=None, timeout=(15, 240)):
        self.api_key = api_key if api_key is not None else os.environ.get("DEEPINFRA_API_KEY")
        self.url = url
        self.budget = int(budget)
        self._session = session  # injectable for offline tests
        self.timeout = timeout
        self._calls: list = []
        self._n = 0

    @property
    def call_log(self) -> list:
        return list(self._calls)

    def _log(self, **kw) -> dict:
        self._n += 1
        kw["n"] = self._n
        kw.setdefault("kind", "judge")
        kw.setdefault("ts", _iso())
        self._calls.append(kw)
        return kw

    def _post(self, payload: dict):
        if not self.api_key:
            raise RuntimeError("DEEPINFRA_API_KEY is not set; pass api_key= "
                               "or export the env var")
        s = self._session
        if s is None:
            s = requests.Session()
            s.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            })
            self._session = s
        t0 = time.time()
        try:
            r = s.post(self.url, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            return None, round(time.time() - t0, 3), f"net-error {type(e).__name__}: {e}"
        latency = round(time.time() - t0, 3)
        if r.status_code == 200:
            try:
                return json.loads(r.text), latency, None
            except json.JSONDecodeError as e:
                return None, latency, f"bad-json: {e}"
        return None, latency, f"http {r.status_code}: {(r.text or '')[:300]}"

    def judge(self, model: str, messages: list, *,
              max_tokens: int = DEFAULT_MAX_TOKENS,
              temperature: float = 0.0) -> JudgeResult:
        assert_allowed(model)
        if self._n >= self.budget:
            raise BudgetExhausted(
                f"budget cap {self.budget} reached before call (used={self._n})")
        mt = int(max_tokens)
        length_retried = False
        notes: list = []
        t_start = time.time()
        attempts = 0
        while True:
            attempts += 1
            data, latency, err = self._post({
                "model": model, "messages": messages,
                "max_tokens": mt, "temperature": temperature})
            if data is not None:
                usage = data.get("usage") or {}
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                content = msg.get("content") or ""
                fr = choice.get("finish_reason")
                call_notes: list = []
                if not content.strip() and msg.get("reasoning_content"):
                    content = msg["reasoning_content"]
                    call_notes.append("used_reasoning_fallback")
                self._log(model=model, status=200, latency=latency, usage=usage,
                          finish_reason=fr, max_tokens=mt, attempts=attempts,
                          error=None, notes=call_notes)
                if fr == "length" and not length_retried and mt < LENGTH_RETRY_CAP:
                    if self._n >= self.budget:
                        notes.append("length_retry_skipped_budget_exhausted")
                    else:
                        new_mt = int(mt * 1.5)
                        if new_mt <= mt:
                            new_mt = mt + 1
                        new_mt = min(new_mt, LENGTH_RETRY_CAP)
                        if new_mt > mt:
                            length_retried = True
                            notes.append(f"length_retry_max_tokens={new_mt}")
                            mt = new_mt
                            continue
                return JudgeResult(ok=True, content=content, model=model,
                                   finish_reason=fr, usage=usage,
                                   latency=round(time.time() - t_start, 3),
                                   attempts=attempts, max_tokens_sent=mt,
                                   notes=notes + call_notes)
            is_http_err = bool(err) and err.startswith("http ")
            retryable = (is_http_err and any(f"http {c}:" in err for c in RETRY_STATUSES)) \
                or not is_http_err
            self._log(model=model, status=err.split(":")[0] if is_http_err else "error",
                      latency=latency, usage={}, finish_reason=None, max_tokens=mt,
                      attempts=attempts, error=(err or "")[:300], notes=[])
            if retryable and attempts < 3:
                if self._n >= self.budget:
                    notes.append("transport_retry_skipped_budget_exhausted")
                    return JudgeResult(ok=False, content="", model=model,
                                       finish_reason=None, usage={},
                                       latency=round(time.time() - t_start, 3),
                                       attempts=attempts, max_tokens_sent=mt,
                                       error=err, notes=notes)
                time.sleep(min(2 * attempts, 8))
                continue
            return JudgeResult(ok=False, content="", model=model, finish_reason=None,
                               usage={}, latency=round(time.time() - t_start, 3),
                               attempts=attempts, max_tokens_sent=mt, error=err,
                               notes=notes)
