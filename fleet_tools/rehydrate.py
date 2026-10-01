"""fleet_tools.rehydrate — decay-curve model + rehydration scheduler.

Wave-72 seed from PROBE-2 (w71 papermill-decay-rehydrate): a receipt-chained
diff stream that has DECAYED (missing/unknown prefix receipts) folds into a
state that diverges from the sealed HEAD by a measurable fraction of paths.
The full-stream fold is exact (248==248 byte-identical), so divergence is
attributable to the missing receipts alone — rehydration is well-defined.

This module:
  - DecayModel: fits divergence fraction d(t) vs truncation depth t (the
    fraction of the stream whose receipts are missing/unknown) from a
    measured curve. Two fits, both ANCHORED at the two points the wave-71
    probe proved exactly — d(0)=0 (intact stream folds byte-identical to
    HEAD) and d(1)=1 (empty stream shares no paths with HEAD):
      "logistic":  normalized logistic  d(t) = (s(a(t-t0)) - s(-a*t0)) /
                   (s(a(1-t0)) - s(-a*t0)),  s = sigmoid; deterministic
                   coarse-to-fine grid search over (a, t0). As a -> 0 the
                   family degrades gracefully to the linear d(t) = t.
      "piecewise": linear interpolation through the measured knots; exact
                   on knots by construction.
  - RehydrationScheduler: decides 'catchup' vs 'rehydrate' for a replica
    checkpointed at depth k of an n-diff stream where only p prefix
    receipts are known/verifiable:
        t = (n - p) / n                (decay fraction)
        expected_divergence = d(t)     (from the fitted model)
        policy: expected_divergence > threshold  -> rehydrate (default 0.02)
                else cost_full_rehydrate <= catchup_cost -> rehydrate
                else catchup
    catchup_cost = (n - k) * cost_catchup_per_diff.
  Deterministic and fully offline.
"""
import math

DEFAULT_THRESHOLD = 0.02


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logistic(a, t0, t):
    """Sigmoid anchored to pass exactly through d(0)=0 and d(1)=1.

    d(t) = (s(a(t-t0)) - s(-a*t0)) / (s(a(1-t0)) - s(-a*t0)); monotone for
    a > 0 and -> linear d(t)=t as a -> 0.
    """
    denom = _sigmoid(a * (1.0 - t0)) - _sigmoid(-a * t0)
    if denom <= 1e-12:
        return t
    return (_sigmoid(a * (t - t0)) - _sigmoid(-a * t0)) / denom


def _sse_logistic(params, pts):
    a, t0 = params
    return sum((_logistic(a, t0, t) - d) ** 2 for t, d in pts)


class DecayModel:
    """Divergence fraction d(t) for truncation depth t in [0, 1]."""

    def __init__(self, kind, points, params=None, r2=None):
        self.kind = kind
        self.points = sorted(points)          # [(t, d)] measured knots
        self.params = params or {}
        self.r2 = r2

    # ---------- fitting ----------
    @classmethod
    def fit(cls, points, kind="logistic"):
        pts = cls._clean(points)
        if kind == "logistic":
            params, r2 = cls._fit_logistic(pts)
            return cls(kind, pts, params, r2)
        if kind == "piecewise":
            return cls(kind, pts, {}, cls._r2(pts, lambda t: cls._interp(pts, t)))
        raise ValueError(f"unknown fit kind: {kind!r}")

    @staticmethod
    def _clean(points):
        pts = []
        for p in points:
            t, d = (p["t"], p.get("divergence_fraction",
                    p.get("divergence", p.get("d")))) if isinstance(p, dict) else p
            t, d = float(t), float(d)
            if not (0.0 <= t <= 1.0) or not (0.0 <= d <= 1.0):
                raise ValueError(f"curve point out of range: t={t} d={d}")
            pts.append([t, d])
        if len(pts) < 2:
            raise ValueError("need at least 2 curve points to fit")
        pts.sort(key=lambda x: x[0])
        merged = []                            # average duplicate depths
        for t, d in pts:
            if merged and abs(merged[-1][0] - t) < 1e-12:
                merged[-1][1] = (merged[-1][1] + d) / 2.0
            else:
                merged.append([t, d])
        return [(t, d) for t, d in merged]

    @staticmethod
    def _r2(pts, f):
        n = len(pts)
        mean = sum(d for _, d in pts) / n
        tss = sum((d - mean) ** 2 for _, d in pts)
        sse = sum((f(t) - d) ** 2 for t, d in pts)
        if tss <= 1e-18:
            return 1.0 if sse <= 1e-18 else 0.0
        return 1.0 - sse / tss

    @staticmethod
    def _fit_logistic(pts):
        """Deterministic coarse grid + coordinate refinement (no deps)."""
        best, bsse = (4.0, 0.5), float("inf")
        for a in [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]:
            for t0 in [-0.25 + 0.05 * i for i in range(31)]:
                sse = _sse_logistic((a, t0), pts)
                if sse < bsse:
                    best, bsse = (a, t0), sse
        step = [0.5, 0.02]
        for _ in range(8):
            improved = False
            for i in range(2):
                for sign in (1, -1):
                    cand = list(best)
                    cand[i] += sign * step[i]
                    cand[0] = min(max(cand[0], 0.05), 200.0)
                    cand[1] = min(max(cand[1], -2.0), 3.0)
                    sse = _sse_logistic(tuple(cand), pts)
                    if sse < bsse:
                        best, bsse, improved = tuple(cand), sse, True
            if not improved:
                step = [s / 2 for s in step]
        a, t0 = best
        model = lambda t: _logistic(a, t0, t)
        return {"a": a, "t0": t0}, DecayModel._r2(pts, model)

    @staticmethod
    def _interp(pts, t):
        """Piecewise-linear through knots; constant outside the range."""
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        for (t0, d0), (t1, d1) in zip(pts, pts[1:]):
            if t0 <= t <= t1:
                if t1 <= t0:
                    return d1
                w = (t - t0) / (t1 - t0)
                return d0 + w * (d1 - d0)
        return pts[-1][1]

    # ---------- evaluation ----------
    def divergence(self, t):
        """Expected divergence fraction at truncation depth t (clamped [0,1])."""
        t = min(max(float(t), 0.0), 1.0)
        if self.kind == "logistic":
            d = _logistic(self.params["a"], self.params["t0"], t)
        elif self.kind == "piecewise":
            d = self._interp(self.points, t)
        else:
            raise ValueError(f"unknown model kind: {self.kind!r}")
        return min(max(d, 0.0), 1.0)

    def is_monotone(self, steps=64):
        """Deeper truncation never decreases divergence (dense-grid check)."""
        prev = self.divergence(0.0)
        for i in range(1, steps + 1):
            cur = self.divergence(i / steps)
            if cur < prev - 1e-12:
                return False
            prev = cur
        return True


class RehydrationScheduler:
    """Decide 'catchup' vs 'rehydrate' for a decayed diff stream.

    Scenario semantics (documented in studies/REHYDRATION-72.md):
      stream_length n        — canonical stream depth claimed by the seal.
      prefix_receipts_known p— prefix receipts the replica can verify
                               (its own checkpoint plus any surviving
                               upstream chain). Missing/unknown = n - p.
      replica_depth k        — diffs the replica has already applied.
    The decay fraction is t = (n - p) / n; a fully intact stream (p = n)
    gives t = 0 and the P2-proven exact convergence d = 0.
    """

    def __init__(self, model, threshold=DEFAULT_THRESHOLD):
        self.model = model
        self.threshold = float(threshold)

    def decide(self, prefix_receipts_known, replica_depth, stream_length,
               cost_full_rehydrate, cost_catchup_per_diff):
        p, k, n = int(prefix_receipts_known), int(replica_depth), int(stream_length)
        c_full, c_diff = float(cost_full_rehydrate), float(cost_catchup_per_diff)
        if min(p, k, n) < 0 or c_full < 0 or c_diff < 0:
            raise ValueError("depths, stream length and costs must be >= 0")
        missing = max(0, n - p)
        t = missing / n if n > 0 else 0.0
        expected = self.model.divergence(t)
        catchup_diffs = max(0, n - k)
        catchup_cost = catchup_diffs * c_diff
        if expected > self.threshold:
            action = "rehydrate"
            rationale = (f"expected divergence {expected:.4f} at decay t={t:.4f} "
                         f"exceeds threshold {self.threshold:g}; incremental catch-up "
                         f"would land ~{expected:.0%} divergent — "
                         f"rebuild from seal instead")
        elif c_full <= catchup_cost:
            action = "rehydrate"
            rationale = (f"divergence within deadband (d={expected:.4f} <= "
                         f"{self.threshold:g}) but full rehydration "
                         f"({c_full:g}) <= catch-up ({catchup_cost:g} for "
                         f"{catchup_diffs} diffs) and is exactly convergent")
        else:
            action = "catchup"
            rationale = (f"expected divergence {expected:.4f} within threshold "
                         f"{self.threshold:g} and catch-up cheaper "
                         f"({catchup_cost:g} for {catchup_diffs} diffs vs "
                         f"{c_full:g} for rehydration)")
        return {
            "action": action,
            "expected_divergence": expected,
            "rationale": rationale,
            "decay_fraction": t,
            "missing_receipts": missing,
            "replica_depth": k,
            "stream_length": n,
            "prefix_receipts_known": p,
            "catchup_diffs": catchup_diffs,
            "catchup_cost": catchup_cost,
            "cost_full_rehydrate": c_full,
            "threshold": self.threshold,
            "model_kind": self.model.kind,
        }


def load_curve(path):
    """Load a decay-curve JSON -> (points, meta).

    Accepts the w72-decay-curve.json schema ({"curve": [...]} with per-point
    "t"/"divergence_fraction") or a bare list of {"t","d"} / [t, d] pairs.
    """
    import json
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        pts = data.get("curve") or data.get("points")
        meta = {k: data[k] for k in ("corpus", "stream_commits", "head_paths")
                if k in data}
        if pts is None:
            raise ValueError("curve JSON has no 'curve'/'points' list")
    else:
        pts, meta = data, {}
    return pts, meta
