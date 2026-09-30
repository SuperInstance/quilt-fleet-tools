"""judge-gate tests with MOCK judges (no network): biased mock must FAIL the
label-shuffle rule, label-independent mock must PASS; allowlist enforced;
receipt sealing + chain."""
import json
import os

import pytest

from fleet_tools import judge_gate


ITEMS = [{"id": f"item{i}", "text": f"def f{i}(): return {i}",
          "label_true": "alpha", "label_swapped": "beta"} for i in range(6)]


class IndependentMock:
    """Scores depend only on the item id — a valid instrument."""

    def __init__(self):
        self.calls = 0
        self.base = {it["id"]: 5 + (i % 3) for i, it in enumerate(ITEMS)}

    def score(self, judge_model, prompt):
        self.calls += 1
        judge_gate.assert_allowed(judge_model)
        ids = [w.strip().rstrip(",") for w in
               prompt.split("Candidates to score:")[1].split()]
        return json.dumps([{"id": cid, "score": self.base[cid.split("/", 1)[1]],
                            "reason": "x"} for cid in ids])


class BiasedMock(IndependentMock):
    """Scores jump +4 whenever the displayed label is 'beta' — label-biased."""

    def score(self, judge_model, prompt):
        text = super().score(judge_model, prompt)
        data = json.loads(text)
        for d in data:
            if d["id"].startswith("beta/"):
                d["score"] = min(10, d["score"] + 4)
        return json.dumps(data)


def test_allowlist_rejects_unknown_model():
    with pytest.raises(ValueError):
        judge_gate.assert_allowed("openai/gpt-4")


def test_independent_mock_passes_gate():
    client = IndependentMock()
    receipt = judge_gate.run_gate(client, ["tencent/Hy3"], ITEMS,
                                  out_dir="gate_receipts_test_ind",
                                  item_inputs_sha256="deadbeef")
    assert receipt["GATE"] == "PASS"
    for r in receipt["results"].values():
        assert r["stability"]["rule"] == "PASS"
        assert r["label_shuffle"]["rule"] == "PASS"


def test_biased_mock_fails_label_shuffle():
    client = BiasedMock()
    receipt = judge_gate.run_gate(client, ["tencent/Hy3"], ITEMS,
                                  out_dir="gate_receipts_test_biased")
    assert receipt["GATE"] == "FAIL"
    for r in receipt["results"].values():
        assert r["label_shuffle"]["rule"] == "FAIL"


def test_receipt_chain_links(tmp_path):
    out = str(tmp_path / "receipts")
    judge_gate.run_gate(IndependentMock(), ["tencent/Hy3"],
                        [{**it, "text": it["text"] + " #1"} for it in ITEMS],
                        out_dir=out)
    judge_gate.run_gate(IndependentMock(), ["tencent/Hy3"],
                        [{**it, "text": it["text"] + " #2"} for it in ITEMS],
                        out_dir=out)
    files = sorted(f for f in os.listdir(out) if f.startswith("gate_receipt_"))
    assert len(files) == 2
    r1 = json.load(open(os.path.join(out, files[0])))
    r2 = json.load(open(os.path.join(out, files[1])))
    assert r1["prev_receipt_sha256"] is None
    assert r2["prev_receipt_sha256"] == judge_gate._sha256_file(
        os.path.join(out, files[0]))


def test_parse_scores_fallback():
    text = 'blah Seed-2.0-mini/item1 ... score: 7 blah'
    assert judge_gate.parse_scores(text, ["item1"]) == {"item1": 7}
