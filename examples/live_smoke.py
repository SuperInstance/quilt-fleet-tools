"""One-call live smoke for the judge backend (the ONLY judge-gate script that
spends a call). Requires DEEPINFRA_API_KEY in the environment.

    python3 examples/live_smoke.py [model_id]     # default: XiaomiMiMo/MiMo-V2.6-Flash

Costs exactly 1 chat call (2 if the judge hits max_tokens once and the client
exercises its single length-retry). Never prints the key. Exit 0 = parseable
score received; 1 = transport/HTTP failure; 2 = answer arrived but no parseable
score (a refusal — booked, not retried away).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from judge_gate.client import DeepInfraJudgeClient          # noqa: E402
from judge_gate.scoring import judge_messages, parse_score  # noqa: E402

TRANSCRIPT = ("User: install the dep and run the suite. Agent: ran pip install, "
              "then pytest: 3/3 passed. Reported done with the command outputs.")


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "XiaomiMiMo/MiMo-V2.6-Flash"
    client = DeepInfraJudgeClient()
    res = client.judge(model, judge_messages(TRANSCRIPT, "Author A"))
    if not res.ok:
        print(json.dumps({"ok": False, "error": res.error,
                          "attempts": res.attempts, "latency": res.latency}))
        return 1
    score, src = parse_score(res.content)
    print(json.dumps({"ok": True, "model": model, "score": score,
                      "parsed_from": src, "finish_reason": res.finish_reason,
                      "attempts": res.attempts, "usage": res.usage,
                      "latency": res.latency,
                      "call_log_len": len(client.call_log)}, indent=2))
    return 0 if score is not None else 2


if __name__ == "__main__":
    sys.exit(main())
