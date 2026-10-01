"""rehydrate tests: decay-model fit sanity, decision boundaries, monotonicity,
CLI smoke. Offline and deterministic.

The PAPERMILL_CURVE fixture is real measured data (w72-decay-curve.json,
scripts/w72_decay_curve.mjs on the papermill corpus; the t=0.4950/117-paths
point is the sealed w71 PROBE-2 receipt, reproduced exactly).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fleet_tools import rehydrate
from fleet_tools.rehydrate import DecayModel, RehydrationScheduler

PAPERMILL_CURVE = [
    (0.0, 0.0),
    (0.247525, 0.209677),   # 52/248 divergent paths (25 of 101 commits dropped)
    (0.49505, 0.471774),    # 117/248 — sealed w71 PROBE-2 point, reproduced
    (0.742574, 0.786290),   # 195/248
    (1.0, 1.0),
]


def _logistic_true(t, a=6.0, t0=0.45):
    return rehydrate._logistic(a, t0, t)


# ---------- curve fit sanity ----------

def test_logistic_fit_recovers_synthetic_curve():
    pts = [(t / 10, _logistic_true(t / 10)) for t in range(11)]
    m = DecayModel.fit(pts, kind="logistic")
    assert m.r2 > 0.99
    # recovered curve tracks the generator within 0.01 everywhere sampled
    for t in (0.05, 0.3, 0.45, 0.6, 0.9):
        assert abs(m.divergence(t) - _logistic_true(t)) < 0.01


def test_fit_is_deterministic():
    a = DecayModel.fit(PAPERMILL_CURVE, kind="logistic")
    b = DecayModel.fit(PAPERMILL_CURVE, kind="logistic")
    assert a.params == b.params and a.r2 == b.r2


def test_piecewise_fit_exact_on_knots():
    m = DecayModel.fit(PAPERMILL_CURVE, kind="piecewise")
    for t, d in PAPERMILL_CURVE:
        assert m.divergence(t) == pytest.approx(d, abs=1e-12)
    # between knots: exact linear interpolation of the bracketing knots
    (t0, d0), (t1, d1) = PAPERMILL_CURVE[2], PAPERMILL_CURVE[3]
    t_mid = 0.6
    w = (t_mid - t0) / (t1 - t0)
    assert m.divergence(t_mid) == pytest.approx(d0 + w * (d1 - d0), abs=1e-12)
    # outside the knots: constant hold, still in [0,1]
    assert m.divergence(-5.0) == 0.0 and m.divergence(5.0) == 1.0


def test_fit_rejects_out_of_range_and_thin_curves():
    with pytest.raises(ValueError):
        DecayModel.fit([(0.0, 0.0), (1.5, 0.5)])
    with pytest.raises(ValueError):
        DecayModel.fit([(0.0, 0.0)])


def test_load_curve_w72_schema_and_bare_list(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"stream_commits": 101, "head_paths": 248,
                             "curve": [{"t": 0.0, "divergence_fraction": 0.0},
                                       {"t": 0.5, "divergence_fraction": 0.4}]}))
    pts, meta = rehydrate.load_curve(str(p))
    assert meta["stream_commits"] == 101 and len(pts) == 2
    bare = tmp_path / "b.json"
    bare.write_text(json.dumps([[0.0, 0.0], [1.0, 1.0]]))
    pts2, meta2 = rehydrate.load_curve(str(bare))
    assert pts2 == [[0.0, 0.0], [1.0, 1.0]] and meta2 == {}


# ---------- decision boundary behavior ----------

def _linear_model():
    # piecewise knots (0,0)-(1,1): d(t) == t exactly
    return DecayModel.fit([(0.0, 0.0), (1.0, 1.0)], kind="piecewise")


def test_decision_boundary_below_threshold_catches_up():
    s = RehydrationScheduler(_linear_model(), threshold=0.02)
    # n=100, p=99 -> t=0.01 -> d=0.01 <= 0.02; catch-up 90 diffs at 1 = 90 < 300
    r = s.decide(prefix_receipts_known=99, replica_depth=10, stream_length=100,
                 cost_full_rehydrate=300, cost_catchup_per_diff=1)
    assert r["action"] == "catchup"
    assert r["expected_divergence"] == pytest.approx(0.01, abs=1e-12)
    assert r["catchup_cost"] == 90


def test_decision_boundary_above_threshold_rehydrates():
    s = RehydrationScheduler(_linear_model(), threshold=0.02)
    # n=100, p=97 -> t=0.03 -> d=0.03 > 0.02 -> rehydrate despite cheap catch-up
    r = s.decide(prefix_receipts_known=97, replica_depth=10, stream_length=100,
                 cost_full_rehydrate=300, cost_catchup_per_diff=1)
    assert r["action"] == "rehydrate"
    assert r["expected_divergence"] == pytest.approx(0.03, abs=1e-12)
    assert "exceeds threshold" in r["rationale"]


def test_threshold_exact_boundary_catches_up():
    s = RehydrationScheduler(_linear_model(), threshold=0.02)
    # t == threshold exactly: policy is strictly '>' so this stays in deadband
    r = s.decide(prefix_receipts_known=98, replica_depth=0, stream_length=100,
                 cost_full_rehydrate=300, cost_catchup_per_diff=1)
    assert r["action"] == "catchup"


def test_cost_can_force_rehydrate_inside_deadband():
    s = RehydrationScheduler(_linear_model(), threshold=0.02)
    # divergence fine but rehydrate cheaper: 50 <= catchup 91
    r = s.decide(prefix_receipts_known=100, replica_depth=10, stream_length=101,
                 cost_full_rehydrate=50, cost_catchup_per_diff=1)
    assert r["action"] == "rehydrate"
    assert "cheaper" in r["rationale"] or "exactly convergent" in r["rationale"]


def test_intact_stream_has_zero_divergence_and_fully_decayed_rehydrates():
    s = RehydrationScheduler(DecayModel.fit(PAPERMILL_CURVE), threshold=0.02)
    intact = s.decide(101, 30, 101, 300, 1)
    # anchored fit: intact stream is EXACTLY convergent (w71 P2 proof)
    assert intact["expected_divergence"] == pytest.approx(0.0, abs=1e-12)
    assert intact["action"] == "catchup"
    dead = s.decide(0, 30, 101, 300, 1)
    assert dead["decay_fraction"] == 1.0
    assert dead["expected_divergence"] == pytest.approx(1.0, abs=1e-12)
    assert dead["action"] == "rehydrate"


def test_decide_validates_inputs():
    s = RehydrationScheduler(_linear_model())
    with pytest.raises(ValueError):
        s.decide(-1, 0, 100, 300, 1)
    with pytest.raises(ValueError):
        s.decide(100, 0, 100, -5, 1)
    # n = 0 degenerate stream: no decay, zero-cost catch-up
    r = s.decide(0, 0, 0, 300, 1)
    assert r["action"] == "catchup" and r["expected_divergence"] == 0.0


# ---------- monotonicity ----------

def test_deeper_truncation_never_decreases_divergence():
    for kind in ("logistic", "piecewise"):
        m = DecayModel.fit(PAPERMILL_CURVE, kind=kind)
        assert m.is_monotone()
        prev = -1.0
        for i in range(0, 101):
            d = m.divergence(i / 100)
            assert d >= prev - 1e-12
            prev = d


def test_scheduler_divergence_monotone_in_decay():
    s = RehydrationScheduler(DecayModel.fit(PAPERMILL_CURVE))
    prev = -1.0
    for p in range(101, -1, -1):        # p decreasing -> decay increasing
        r = s.decide(p, 0, 101, 300, 1)
        assert r["expected_divergence"] >= prev - 1e-12
        prev = r["expected_divergence"]


# ---------- CLI smoke ----------

def test_cli_rehydrate_plan_smoke(tmp_path):
    curve = tmp_path / "curve.json"
    curve.write_text(json.dumps({
        "stream_commits": 101, "head_paths": 248,
        "curve": [{"t": t, "divergence_fraction": d} for t, d in PAPERMILL_CURVE]}))
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    r = subprocess.run(
        [sys.executable, "-m", "fleet_tools", "rehydrate-plan",
         "--curve", str(curve), "--stream-length", "101",
         "--prefix-known", "51", "--replica-depth", "30",
         "--cost-full", "300", "--cost-per-diff", "1"],
        cwd=str(root), env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["action"] in ("catchup", "rehydrate")
    assert 0.0 <= out["expected_divergence"] <= 1.0
    assert out["model"]["kind"] == "logistic"
    assert out["model"]["r2"] > 0.9
    # degraded-prefix scenario on a write-once corpus must refuse catch-up
    assert out["action"] == "rehydrate"
