"""judge-gate CLI.

    python3 -m judge_gate run --transcripts T.json --model MODEL --out-dir DIR
        [--tolerance 1.0] [--alpha 0.05] [--seed N] [--label-a A] [--label-b B]
        [--rubric TEXT] [--max-tokens 4096] [--temperature 0.0] [--run-id ID]
        [--mock independent|biased]

    python3 -m judge_gate verify DIR        # offline receipt-chain verification

--model must be on the hard allowlist (judge_gate/allowlist.py). --mock swaps in
the deterministic offline mock judge (a harness self-test, marked "mock": true
in the receipt; NOT a model certification).

Exit codes: run -> 0 PASS, 1 FAIL, 2 INCONCLUSIVE; verify -> 0 PASS, 1 FAIL.
"""
import argparse
import json
import os
import sys

from .client import DeepInfraJudgeClient
from .experiment import GateConfig, run_gate
from .mock import MockJudgeClient
from .seal import seal_receipt, verify_receipts


def _load_transcripts(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("transcripts")
    if not isinstance(data, list) or not data or not all(isinstance(t, str) for t in data):
        raise SystemExit("transcripts file must be a JSON array of strings "
                         'or {"transcripts": [...]}')
    return data


def cmd_run(a) -> int:
    transcripts = _load_transcripts(a.transcripts)
    if a.mock:
        client = MockJudgeClient(label_bias=2.0 if a.mock == "biased" else 0.0,
                                 favored_label=a.label_a)
        model = f"mock/{a.mock}"
    else:
        model = a.model
        client = DeepInfraJudgeClient()
    cfg = GateConfig(model=model, tolerance=a.tolerance, alpha=a.alpha,
                     label_a=a.label_a, label_b=a.label_b, rubric=a.rubric,
                     seed=a.seed, max_tokens=a.max_tokens,
                     temperature=a.temperature)
    receipt = run_gate(transcripts, client, cfg, run_id=a.run_id)
    sealed = seal_receipt(receipt, a.out_dir)
    print(json.dumps({
        "verdict": sealed["verdict"],
        "reasons": sealed["reasons"],
        "run_id": sealed["run_id"],
        "mock": sealed["mock"],
        "model": sealed["model"],
        "max_abs_label_diff": sealed["max_abs_label_diff"],
        "permutation_p": (sealed["stats"] or {}).get("permutation", {}).get("p"),
        "calls": len(sealed["calls"]),
        "receipt": os.path.join(a.out_dir, f"{sealed['run_id']}.receipt.json"),
    }, indent=2))
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[sealed["verdict"]]


def cmd_verify(a) -> int:
    rep = verify_receipts(a.dir)
    print(json.dumps(rep, indent=2))
    return 0 if rep["ok"] else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="judge-gate",
        description="judge-instrument verification: author-label bias gate + "
                    "sealed receipt chain for LLM-as-judge")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the gate; seals a receipt into --out-dir")
    r.add_argument("--transcripts", required=True,
                   help="JSON file: array of transcripts (>= 6)")
    r.add_argument("--model", default="XiaomiMiMo/MiMo-V2.6-Flash",
                   help="DeepInfra model id (allowlist-enforced)")
    r.add_argument("--out-dir", default="judge-gate-receipts",
                   help="receipt/chain directory")
    r.add_argument("--tolerance", type=float, default=1.0,
                   help="max |score diff| across a label pair (default 1.0)")
    r.add_argument("--alpha", type=float, default=0.05,
                   help="permutation-test significance level (default 0.05)")
    r.add_argument("--seed", type=int, default=20260925,
                   help="fixed seed for presentation order + permutation test")
    r.add_argument("--label-a", default="Author A")
    r.add_argument("--label-b", default="Author B")
    r.add_argument("--rubric", default=None, help="extra rubric text for the judge")
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--run-id", default=None)
    r.add_argument("--mock", choices=["independent", "biased"], default=None,
                   help="offline mock judge (harness self-test, no network)")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("verify", help="verify the receipt chain in a directory (offline)")
    v.add_argument("dir")
    v.set_defaults(func=cmd_verify)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
