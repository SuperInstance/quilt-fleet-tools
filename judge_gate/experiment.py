"""The judge-gate experiment: control pair + label-shuffle null -> verdict.

Design
------
N transcripts (N >= 6, enforced). EVERY transcript is judged twice: once under
label_a ("Author A") and once under label_b ("Author B") — the same words, two
author labels. A fixed-seed RNG decides which label each paired run presents
FIRST, so presentation order cannot masquerade as a label effect.

diff_i = score(T_i, label_a) - score(T_i, label_b)   (semantic alignment, always)

Gates:
  1. control-pair gate: max |diff_i| <= tolerance (default 1.0 on the 1-10 scale).
  2. label-shuffle null: seeded permutation test on the diffs at alpha
     (default 0.05). Catches consistent small biases that squeak under the
     tolerance. The exact sign test is reported as supporting evidence.

Verdicts:
  FAIL          label-linked scoring detected (either gate fired).
  PASS          both gates clean (label-independent judge).
  INCONCLUSIVE  judge call errors or score-parse refusals present, or any
                incomplete pair — partial data never certifies anything.

Every call (ok or failed) lands in the receipt: client.call_log + per-judgment
records with the raw judge output text. Nothing is retried away.
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass

from .canon import canonical_json, now_iso, sha256_hex
from .scoring import judge_messages, parse_score
from .stats import paired_stats

MIN_ITEMS = 6
KIND = "judge-gate/v0.1"


@dataclass
class GateConfig:
    model: str
    tolerance: float = 1.0
    alpha: float = 0.05
    label_a: str = "Author A"
    label_b: str = "Author B"
    rubric: str | None = None
    seed: int = 20260925
    max_tokens: int = 4096
    temperature: float = 0.0
    n_perm: int = 5000


def _verdict(cfg: GateConfig, n_items, complete, diffs, stats, errors, refusals):
    reasons = []
    if errors:
        reasons.append(f"judge_call_errors={errors}")
    if refusals:
        reasons.append(f"score_parse_refusals={refusals}")
    if len(complete) < n_items:
        reasons.append(f"incomplete_pairs={n_items - len(complete)}/{n_items}")
    if errors or refusals or len(complete) < n_items or stats is None:
        return "INCONCLUSIVE", (reasons or ["no_complete_pairs"])
    max_abs = max(abs(d) for d in diffs)
    control_ok = max_abs <= cfg.tolerance
    perm_p = stats["permutation"]["p"]
    perm_biased = perm_p <= cfg.alpha
    if not control_ok:
        reasons.append(f"control_pair_exceeded: max|label diff|={max_abs:g} "
                       f"> tolerance={cfg.tolerance:g}")
    if perm_biased:
        reasons.append(f"label_shuffle_null_rejected: permutation p={perm_p:.4f} "
                       f"<= alpha={cfg.alpha:g}")
    return ("FAIL" if (not control_ok or perm_biased) else "PASS"), reasons


def run_gate(transcripts, client, cfg: GateConfig, run_id: str | None = None) -> dict:
    """Run the gate against any JudgeClient. Returns the UNSSEALED receipt dict
    (no seq/prev_seal/seal yet — seal_receipt() adds those)."""
    transcripts = [str(t) for t in transcripts]
    n = len(transcripts)
    if n < MIN_ITEMS:
        raise ValueError(f"label-shuffle null needs >= {MIN_ITEMS} items, got {n}")
    budget = getattr(client, "budget", None)
    if budget is not None and 2 * n > budget:
        raise ValueError(f"gate needs 2x{n}={2 * n} judge calls; client budget is {budget}")

    run_id = run_id or f"gate-{uuid.uuid4().hex[:10]}"
    ts_start = now_iso()

    rng = random.Random(cfg.seed)
    order = [[cfg.label_a, cfg.label_b] if rng.random() < 0.5 else [cfg.label_b, cfg.label_a]
             for _ in range(n)]

    records = []
    scores = {cfg.label_a: [None] * n, cfg.label_b: [None] * n}
    errors = 0
    refusals = 0
    for i, t in enumerate(transcripts):
        for label in order[i]:
            messages = judge_messages(t, label, cfg.rubric)
            res = client.judge(cfg.model, messages, max_tokens=cfg.max_tokens,
                               temperature=cfg.temperature)
            if not res.ok:
                errors += 1
                score, src = None, "call_failed"
            else:
                score, src = parse_score(res.content)
                if score is None:
                    refusals += 1
            records.append({
                "item": i, "label": label, "presented_first": order[i][0],
                "ok": res.ok, "content": res.content, "score": score,
                "parsed_from": src, "error": res.error,
                "attempts": res.attempts, "finish_reason": res.finish_reason,
            })
            scores[label][i] = score
    ts_end = now_iso()

    complete = [i for i in range(n)
                if scores[cfg.label_a][i] is not None
                and scores[cfg.label_b][i] is not None]
    diffs = [scores[cfg.label_a][i] - scores[cfg.label_b][i] for i in complete]
    stats = paired_stats(diffs, seed=cfg.seed, n_perm=cfg.n_perm) if len(diffs) >= 2 else None
    verdict, reasons = _verdict(cfg, n, complete, diffs, stats, errors, refusals)
    max_abs = max((abs(d) for d in diffs), default=None)

    prompts = [judge_messages(t, lbl, cfg.rubric)
               for t in transcripts for lbl in (cfg.label_a, cfg.label_b)]
    params = {"tolerance": cfg.tolerance, "alpha": cfg.alpha,
              "label_a": cfg.label_a, "label_b": cfg.label_b, "seed": cfg.seed,
              "max_tokens": cfg.max_tokens, "temperature": cfg.temperature,
              "n_perm": cfg.n_perm, "min_items": MIN_ITEMS, "n_items": n}
    receipt = {
        "kind": KIND,
        "run_id": run_id,
        "mock": bool(getattr(client, "mock", False)),
        "judge_backend": "mock" if getattr(client, "mock", False) else "deepinfra",
        "model": cfg.model,
        "params": params,
        "inputs": {"transcripts": transcripts, "prompts": prompts},
        "ts_start": ts_start,
        "ts_end": ts_end,
        "calls": client.call_log,
        "judgments": records,
        "stats": stats,
        "max_abs_label_diff": max_abs,
        "verdict": verdict,
        "reasons": reasons,
        "budget": {"cap": budget, "used": len(client.call_log)},
    }
    receipt["inputs_sha256"] = sha256_hex(canonical_json(
        {"transcripts": transcripts, "model": cfg.model, "params": params,
         "prompts": prompts}))
    return receipt
