import json

import pytest

from judge_gate import (ALLOWLIST, BudgetExhausted, DeepInfraJudgeClient,
                        GateConfig, MockJudgeClient, ModelNotAllowlisted,
                        assert_allowed, judge_messages, paired_stats,
                        parse_score, run_gate, seal_receipt, verify_receipts)
from judge_gate.client import DEFAULT_MAX_TOKENS, LENGTH_RETRY_CAP

ALLOWED = "XiaomiMiMo/MiMo-V2.6-Flash"
ITEMS = [f"Agent transcript {i}: user asked to run the suite; agent ran it; "
         f"{i + 3} of 10 checks green; reported the misses honestly." for i in range(6)]


def cfg(**kw):
    base = dict(model="mock/independent", tolerance=1.0, alpha=0.05)
    base.update(kw)
    return GateConfig(**base)


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = body


def api_body(content="SCORE: 7", finish_reason="stop", usage=None):
    return json.dumps({
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 12, "completion_tokens": 3,
                           "total_tokens": 15},
    })


class FakeSession:
    """Offline stand-in for requests.Session; replays queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if not self.responses:
            raise AssertionError("FakeSession out of scripted responses")
        return self.responses.pop(0)


# ---------------- allowlist ----------------

def test_allowlist_is_exactly_the_seven_ids():
    assert ALLOWLIST == frozenset({
        "XiaomiMiMo/MiMo-V2.6-Flash",
        "ByteDance/Seed-2.0-mini",
        "inclusionAI/Ling-3.0-flash",
        "meta-models/Muse-Glimmer-30B",
        "nvidia/Nemotron-3-Nano-30B-A3B",
        "thinkingmachines/Inkling-Small",
        "tencent/Hy3",
    })


def test_allowlist_rejection_before_any_http():
    s = FakeSession([])
    c = DeepInfraJudgeClient(api_key="offline-test-key", session=s)
    with pytest.raises(ModelNotAllowlisted):
        c.judge("openai/gpt-4o", [{"role": "user", "content": "hi"}])
    assert s.calls == []  # rejected before any HTTP attempt
    for m in ALLOWLIST:
        assert_allowed(m)  # all seven pass the gate


# ---------------- DeepInfra client (offline, injected transport) ----------------

def test_budget_cap_enforced_and_every_call_logged():
    s = FakeSession([FakeResponse(200, api_body()) for _ in range(4)])
    c = DeepInfraJudgeClient(api_key="k", session=s, budget=2)
    msgs = judge_messages("t", "Author A")
    assert c.judge(ALLOWED, msgs).ok
    assert c.judge(ALLOWED, msgs).ok
    with pytest.raises(BudgetExhausted):
        c.judge(ALLOWED, msgs)
    assert len(c.call_log) == 2


def test_every_call_logged_with_latency_and_usage():
    s = FakeSession([FakeResponse(200, api_body())])
    c = DeepInfraJudgeClient(api_key="k", session=s)
    c.judge(ALLOWED, judge_messages("t", "Author A"))
    e = c.call_log[0]
    assert e["status"] == 200
    assert "latency" in e and e["usage"]["prompt_tokens"] == 12
    assert e["finish_reason"] == "stop" and "ts" in e


def test_length_retry_at_1_5x():
    s = FakeSession([FakeResponse(200, api_body(finish_reason="length")),
                     FakeResponse(200, api_body())])
    c = DeepInfraJudgeClient(api_key="k", session=s)
    r = c.judge(ALLOWED, judge_messages("t", "Author A"))
    assert r.ok and len(s.calls) == 2 and len(c.call_log) == 2
    assert s.calls[0]["json"]["max_tokens"] == DEFAULT_MAX_TOKENS == 4096
    assert s.calls[1]["json"]["max_tokens"] == int(DEFAULT_MAX_TOKENS * 1.5) == 6144
    assert c.call_log[0]["finish_reason"] == "length"


def test_length_retry_caps_at_8192():
    s = FakeSession([FakeResponse(200, api_body(finish_reason="length")),
                     FakeResponse(200, api_body())])
    c = DeepInfraJudgeClient(api_key="k", session=s)
    r = c.judge(ALLOWED, judge_messages("t", "Author A"), max_tokens=6000)
    assert r.ok
    assert s.calls[1]["json"]["max_tokens"] == LENGTH_RETRY_CAP == 8192


def test_no_second_length_retry():
    s = FakeSession([FakeResponse(200, api_body(finish_reason="length")),
                     FakeResponse(200, api_body(finish_reason="length")),
                     FakeResponse(200, api_body())])
    c = DeepInfraJudgeClient(api_key="k", session=s)
    r = c.judge(ALLOWED, judge_messages("t", "Author A"))
    assert r.ok and len(s.calls) == 2  # exactly ONE length retry, even if it truncates again
    assert r.finish_reason == "length"


def test_transport_errors_retried_then_honest_failure():
    s = FakeSession([FakeResponse(503, "boom"), FakeResponse(503, "boom"),
                     FakeResponse(503, "boom")])
    c = DeepInfraJudgeClient(api_key="k", session=s)
    r = c.judge(ALLOWED, judge_messages("t", "Author A"))
    assert not r.ok and "503" in r.error
    assert len(c.call_log) == 3  # every attempt logged, none retried away


def test_budget_guards_length_retry_too():
    s = FakeSession([FakeResponse(200, api_body(finish_reason="length"))])
    c = DeepInfraJudgeClient(api_key="k", session=s, budget=1)
    r = c.judge(ALLOWED, judge_messages("t", "Author A"))
    assert r.ok and len(s.calls) == 1
    assert any("length_retry_skipped_budget_exhausted" in n for n in r.notes)


# ---------------- scoring ----------------

def test_parse_score_forms():
    assert parse_score("SCORE: 7") == (7.0, "score_line")
    assert parse_score("score=8") == (8.0, "score_line")
    assert parse_score("The score: 6\nsome notes") == (6.0, "score_line")
    assert parse_score('{"score": 9}') == (9.0, "json")
    assert parse_score("no numbers here")[0] is None
    assert parse_score("SCORE: 42")[1] == "out_of_range"
    assert parse_score("SCORE: 0")[1] == "out_of_range"
    assert parse_score("   ")[1] == "empty"


# ---------------- stats ----------------

def test_paired_stats_biased_signal():
    st = paired_stats([2.0] * 6, seed=1)
    assert st["mean_signed_diff"] == 2.0
    # floor for n=6 two-sided sign-flip permutation is ~2/64 ~ 0.031, not 0.001
    assert st["permutation"]["p"] <= 0.05
    assert st["sign_test"]["p"] <= 0.05
    assert st["effect_note"] == "degenerate_zero_variance"


def test_paired_stats_null_is_null():
    st = paired_stats([0.0] * 6, seed=1)
    assert st["mean_signed_diff"] == 0.0
    assert st["permutation"]["p"] == 1.0
    assert st["effect_size_cohens_d"] == 0.0
    assert st["sign_test"]["p"] == 1.0


def test_paired_stats_deterministic_under_seed():
    d = [1.0, -0.5, 2.0, 0.5, -1.0, 0.25]
    assert paired_stats(d, seed=7) == paired_stats(d, seed=7)


# ---------------- gate end-to-end (mock judges, offline) ----------------

def test_independent_mock_passes():
    receipt = run_gate(ITEMS, MockJudgeClient(label_bias=0.0), cfg())
    assert receipt["verdict"] == "PASS"
    assert receipt["max_abs_label_diff"] == 0.0
    assert receipt["stats"]["mean_signed_diff"] == 0.0
    assert receipt["stats"]["permutation"]["p"] == 1.0
    assert receipt["mock"] is True and receipt["judge_backend"] == "mock"
    assert len(receipt["calls"]) == 12  # 6 items x 2 labels, every call logged
    assert len(receipt["judgments"]) == 12


def test_biased_mock_fails():
    receipt = run_gate(ITEMS, MockJudgeClient(label_bias=2.0, favored_label="Author A"), cfg())
    assert receipt["verdict"] == "FAIL"
    assert receipt["max_abs_label_diff"] == 2.0
    assert any("control_pair_exceeded" in r for r in receipt["reasons"])
    assert receipt["stats"]["permutation"]["p"] <= 0.05


def test_consistent_small_bias_caught_by_permutation_gate():
    # bias squeaks under the tolerance gate but is perfectly consistent
    receipt = run_gate(ITEMS, MockJudgeClient(label_bias=0.5, favored_label="Author B"),
                       cfg(tolerance=5.0))
    assert receipt["verdict"] == "FAIL"
    assert any("label_shuffle_null_rejected" in r for r in receipt["reasons"])


def test_verdict_invariant_to_seed():
    for seed in (1, 7, 20260925):
        assert run_gate(ITEMS, MockJudgeClient(), cfg(seed=seed))["verdict"] == "PASS"
        assert run_gate(ITEMS, MockJudgeClient(label_bias=2.0),
                        cfg(seed=seed))["verdict"] == "FAIL"


def test_fewer_than_six_items_rejected():
    with pytest.raises(ValueError):
        run_gate(ITEMS[:5], MockJudgeClient(), cfg())


def test_budget_guard_counts_both_labels():
    with pytest.raises(ValueError):
        run_gate(ITEMS, MockJudgeClient(budget=10), cfg())  # needs 2x6=12


def test_inputs_sha256_covers_model_and_params():
    a = run_gate(ITEMS, MockJudgeClient(), cfg(), run_id="h1")
    b = run_gate(ITEMS, MockJudgeClient(), cfg(model="mock/other"), run_id="h2")
    assert a["inputs_sha256"] != b["inputs_sha256"]


# ---------------- receipt sealing + chain ----------------

def test_receipt_seal_and_prev_seal_chain(tmp_path):
    s1 = seal_receipt(run_gate(ITEMS, MockJudgeClient(), cfg(), run_id="g1"), str(tmp_path))
    assert s1["seq"] == 1 and s1["prev_seal"] == "0" * 64
    s2 = seal_receipt(run_gate(ITEMS, MockJudgeClient(label_bias=2.0), cfg(),
                               run_id="g2"), str(tmp_path))
    assert s2["seq"] == 2 and s2["prev_seal"] == s1["seal"]
    rep = verify_receipts(str(tmp_path))
    assert rep["ok"] and rep["n"] == 2
    assert all(l["receipt_file"] == "match" for l in rep["links"])


def test_receipt_file_tamper_detected(tmp_path):
    # seal a FAIL receipt (biased mock), then forge the verdict to PASS
    seal_receipt(run_gate(ITEMS, MockJudgeClient(label_bias=2.0), cfg(),
                          run_id="g1"), str(tmp_path))
    p = tmp_path / "g1.receipt.json"
    d = json.loads(p.read_text())
    assert d["verdict"] == "FAIL"
    d["verdict"] = "PASS"  # forger upgrades the verdict
    p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    rep = verify_receipts(str(tmp_path))
    assert not rep["ok"]
    assert any("MISMATCH" in pr for l in rep["links"] for pr in l["problems"])


def test_chain_line_tamper_detected(tmp_path):
    # seal a FAIL receipt (biased mock), then forge the chain line's verdict to PASS
    seal_receipt(run_gate(ITEMS, MockJudgeClient(label_bias=2.0), cfg(),
                          run_id="g1"), str(tmp_path))
    cp = tmp_path / "judge-gate.chain.jsonl"
    e = json.loads(cp.read_text())
    assert e["verdict"] == "FAIL"
    e["verdict"] = "PASS"
    cp.write_text(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n")
    assert not verify_receipts(str(tmp_path))["ok"]


def test_verify_from_receipt_files_alone(tmp_path):
    seal_receipt(run_gate(ITEMS, MockJudgeClient(), cfg(), run_id="g1"), str(tmp_path))
    seal_receipt(run_gate(ITEMS, MockJudgeClient(), cfg(), run_id="g2"), str(tmp_path))
    (tmp_path / "judge-gate.chain.jsonl").unlink()
    rep = verify_receipts(str(tmp_path))
    assert rep["ok"] and rep["n"] == 2


def test_double_seal_refused(tmp_path):
    r = run_gate(ITEMS, MockJudgeClient(), cfg(), run_id="g1")
    seal_receipt(r, str(tmp_path))
    with pytest.raises(ValueError):
        seal_receipt(r, str(tmp_path))  # same slot re-sealed -> chain grew -> refuse
