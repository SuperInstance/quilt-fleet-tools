"""judge-gate: judge-instrument verification for LLM-as-judge pipelines.

Task 69-b implementation of the judge-gate sketch from study/DEEP-STUDY-68b.md.

What it does (see README for the honest scope):
  - control pair: the same transcript judged under two author labels; the score
    diff must stay within tolerance (default 1.0 on the 1-10 scale).
  - label-shuffle null: >= 6 items, fixed-seed label swaps across paired runs;
    paired mean signed diff + sign test + seeded permutation test.
  - verdict PASS / FAIL / INCONCLUSIVE + effect size (Cohen's d on paired diffs).
  - sealed receipt per run: sha256 of all inputs (transcripts/prompts/model/
    params), timestamps, raw judge outputs + usage, verdict, prev_seal hash
    chaining to the prior receipt in the output dir (append-only chain).
"""
from .allowlist import ALLOWLIST, ModelNotAllowlisted, assert_allowed
from .client import (BudgetExhausted, DeepInfraJudgeClient, JudgeClient,
                     JudgeResult)
from .experiment import GateConfig, run_gate
from .mock import MockJudgeClient
from .scoring import judge_messages, parse_score
from .seal import seal_receipt, verify_receipts
from .stats import paired_stats

__all__ = [
    "ALLOWLIST", "ModelNotAllowlisted", "assert_allowed",
    "BudgetExhausted", "DeepInfraJudgeClient", "JudgeClient", "JudgeResult",
    "GateConfig", "run_gate", "MockJudgeClient", "judge_messages",
    "parse_score", "seal_receipt", "verify_receipts", "paired_stats",
]
