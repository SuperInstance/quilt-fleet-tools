"""fleet_tools.judge_gate — judge-instrument verification for LLM-as-judge.

Verifies that a judge panel is USABLE AS AN INSTRUMENT before its scores book
as evidence, via two pre-registered checks:

  1. STABILITY (control pair): the same block judged twice under identical
     labels. Rule: max per-item |delta| <= tol (default 1.0).
  2. LABEL BIAS (label-shuffle null): each item also judged under a swapped
     label; if scores track the LABEL rather than the artifact, the judge is
     biased. Rules: mean |swapped - true| <= max_abs (default 1.0) and
     |mean signed| <= max_signed (default 0.5).

Domain-agnostic: items carry an artifact `text` and two labels. Every run
writes a SEALED RECEIPT (sha256 of all inputs + outputs + prev-receipt hash
chain within the output directory). Fully offline except DeepInfraJudgeClient.
"""
import hashlib
import json
import os
import re
import statistics
import time
from typing import Dict, List, Optional, Protocol

ALLOWLIST = {
    "XiaomiMiMo/MiMo-V2.6-Flash",
    "ByteDance/Seed-2.0-mini",
    "inclusionAI/Ling-3.0-flash",
    "meta-models/Muse-Glimmer-30B",
    "nvidia/Nemotron-3-Nano-30B-A3B",
    "thinkingmachines/Inkling-Small",
    "tencent/Hy3",
}

DEFAULT_RUBRIC = [
    "Score each candidate below on an integer 1-10 scale.",
    "Rubric: spec adherence 40%, correctness reasoning 30%, clarity 30%.",
    "Judge honestly and independently; do not assume any candidate is correct.",
    "Output STRICT JSON — a list with exactly one object per candidate id, in this form: ",
    '[{"id": "<id>", "score": <1-10>, "reason": "<one line>"}]',
    "No other text before or after the JSON.",
]


def assert_allowed(model: str) -> str:
    if model not in ALLOWLIST:
        raise ValueError(f"model {model!r} not in allowlist")
    return model


class JudgeClient(Protocol):
    def score(self, judge_model: str, prompt: str) -> str:
        """Return the raw judge reply text for one prompt."""
        ...


class DeepInfraJudgeClient:
    """OpenAI-compatible client for the allowlisted DeepInfra models."""

    MAX_TOKENS = 4096
    URL = "https://api.deepinfra.com/v1/openai/chat/completions"

    def __init__(self, api_key: Optional[str] = None, call_cap: int = 90):
        import requests
        self.api_key = api_key or os.environ.get("DEEPINFRA_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("DEEPINFRA_API_KEY not set")
        self.s = requests.Session()
        self.call_cap = call_cap
        self.calls: List[dict] = []

    def score(self, judge_model: str, prompt: str) -> str:
        assert_allowed(judge_model)
        if len(self.calls) >= self.call_cap:
            raise RuntimeError("call cap reached")
        r = self.s.post(self.URL, timeout=(15, 240), headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"},
            json={"model": judge_model, "max_tokens": self.MAX_TOKENS,
                  "temperature": 0.2,
                  "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        data = r.json()
        self.calls.append({"judge": judge_model,
                           "usage": data.get("usage", {}),
                           "ts": time.time()})
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


def parse_scores(text: str, expected_ids: List[str]) -> Dict[str, int]:
    scores = {}
    if not text:
        return scores
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            for item in data if isinstance(data, list) else []:
                if isinstance(item, dict) and "id" in item:
                    try:
                        sc = int(str(item.get("score")).strip())
                    except (TypeError, ValueError):
                        continue
                    sid = str(item["id"]).strip()
                    if sid in expected_ids:
                        scores[sid] = max(1, min(10, sc))
        except json.JSONDecodeError:
            pass
    for sid in expected_ids:
        if sid in scores:
            continue
        m2 = re.search(re.escape(sid) + r"\D{0,120}?score\D{0,12}(\d+)",
                       text, re.S | re.I)
        if m2:
            scores[sid] = max(1, min(10, int(m2.group(1))))
    return scores


def build_prompt(items: List[dict], label_map: Dict[str, str],
                 rubric: List[str] = None) -> (str, List[str]):
    """items: [{'id', 'text'}]; label_map: item id -> displayed label."""
    rubric = rubric or DEFAULT_RUBRIC
    parts = list(rubric)
    ids = []
    for it in items:
        cid = f"{label_map[it['id']]}/{it['id']}"
        ids.append(cid)
        parts.append(f"\n--- candidate id: {cid}")
        parts.append(str(it["text"]))
    parts.append("\nCandidates to score: " + ", ".join(ids))
    return "\n".join(parts), ids


def _judge_block(client: JudgeClient, judge: str, items, label_map) -> Dict[str, int]:
    """Judge one block; returns scores keyed by BARE item id (the displayed
    candidate id is f'<label>/<item id>' — stripped here so comparisons are
    label-independent)."""
    prompt, ids = build_prompt(items, label_map)
    text = client.score(judge, prompt)
    parsed = parse_scores(text, ids)
    return {cid.rsplit("/", 1)[1]: sc for cid, sc in parsed.items()}


def control_pair(client: JudgeClient, judge: str, items, tol: float = 1.0):
    """Same block, identical labels, twice."""
    lm = {it["id"]: it.get("label_true", it["id"]) for it in items}
    s1 = _judge_block(client, judge, items, lm)
    s2 = _judge_block(client, judge, items, lm)
    common = [i for i in lm if i in s1 and i in s2]
    deltas = [abs(s1[i] - s2[i]) for i in common]
    mx = max(deltas) if deltas else None
    return {"max_abs_delta": mx, "n_compared": len(common), "s1": s1, "s2": s2,
            "rule": "PASS" if (mx is not None and mx <= tol) else "FAIL",
            "tol": tol}


def label_shuffle(client: JudgeClient, judge: str, items,
                  true_scores: Dict[str, int] = None,
                  max_abs: float = 1.0, max_signed: float = 0.5):
    """Score items under swapped labels; compare to true-label scores.

    true_scores may be supplied (e.g. from a prior control run); otherwise a
    fresh true-label block is judged first within this call.
    """
    lm_true = {it["id"]: it.get("label_true", it["id"]) for it in items}
    lm_swap = {it["id"]: it.get("label_swapped", it["id"]) for it in items}
    if true_scores is None:
        true_scores = _judge_block(client, judge, items, lm_true)
    swapped = _judge_block(client, judge, items, lm_swap)
    pairs = []
    for it in items:
        iid = it["id"]
        if iid in true_scores and iid in swapped:
            sw = swapped[iid]
            pairs.append({"id": iid, "score_true": true_scores[iid],
                          "score_swapped": sw, "signed_diff": sw - true_scores[iid]})
    absd = [abs(p["signed_diff"]) for p in pairs]
    sig = [p["signed_diff"] for p in pairs]
    mean_abs = round(statistics.mean(absd), 3) if absd else None
    mean_sig = round(statistics.mean(sig), 3) if sig else None
    ok = (mean_abs is not None and mean_abs <= max_abs and
          abs(mean_sig) <= max_signed)
    return {"pairs": pairs, "mean_abs_diff": mean_abs,
            "mean_signed_diff": mean_sig, "rule": "PASS" if ok else "FAIL",
            "max_abs": max_abs, "max_signed": max_signed}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def seal_receipt(out_dir: str, receipt: dict) -> str:
    """Write receipt with a prev-receipt hash chain within out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    prev = None
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith("gate_receipt_") and fn.endswith(".json"):
            prev = _sha256_file(os.path.join(out_dir, fn))
    receipt = dict(receipt)
    receipt["prev_receipt_sha256"] = prev
    receipt["sealed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    # distinct name per receipt: same-ms reseals must not overwrite the
    # previous link (the wave-71 lesson — see worklog 71-e)
    ms = int(time.time() * 1000)
    path = os.path.join(out_dir, f"gate_receipt_{ms}.json")
    n = 0
    while os.path.exists(path):
        n += 1
        path = os.path.join(out_dir, f"gate_receipt_{ms}_{n}.json")
    with open(path, "w") as f:
        json.dump(receipt, f, indent=1, sort_keys=True)
    return path


def run_gate(client: JudgeClient, judges: List[str], items: List[dict],
             out_dir: str, item_inputs_sha256: Optional[str] = None) -> dict:
    """Full instrument gate: control pair + label shuffle per judge -> receipt."""
    receipt = {"judges": judges, "n_items": len(items), "results": {},
               "item_inputs_sha256": item_inputs_sha256}
    gates = []
    for j in judges:
        cp = control_pair(client, j, items)
        ls = label_shuffle(client, j, items, true_scores=cp["s1"])
        receipt["results"][j] = {
            "stability": {k: v for k, v in cp.items() if k not in ("s1", "s2", "pairs")},
            "label_shuffle": {k: v for k, v in ls.items() if k != "pairs"}}
        gates.append(cp["rule"] == "PASS" and ls["rule"] == "PASS")
    receipt["GATE"] = "PASS" if gates and all(gates) else "FAIL"
    receipt["usage"] = getattr(client, "calls", [])
    path = seal_receipt(out_dir, receipt)
    receipt["receipt_path"] = path
    return receipt
