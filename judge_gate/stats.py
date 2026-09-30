"""Paired label-shuffle statistics for judge-gate. Stdlib only, fully deterministic
given the seed (random.Random, not global state)."""

import math
import random


def _binom_two_sided(k: int, n: int) -> float:
    """Two-sided exact binomial p under p0=0.5, doubled-one-tail convention."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    num = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * num / (2.0 ** n))


def paired_stats(diffs, seed: int, n_perm: int = 5000) -> dict:
    """Stats over paired signed diffs (score under label A minus under label B).

    - mean signed diff: systematic label effect.
    - sign test: exact two-sided binomial on non-zero diffs (zeros are ties).
    - permutation test: seeded sign-flip permutation of the diffs (equivalent to
      permuting labels within pairs); two-sided p for |mean|.
    - effect size: Cohen's d = mean / population sd; None with a note when the
      diffs have zero variance but non-zero mean (uniform bias).
    """
    diffs = [float(d) for d in diffs]
    n = len(diffs)
    if n == 0:
        raise ValueError("paired_stats needs at least one diff")
    mean = sum(diffs) / n
    mean_abs = sum(abs(d) for d in diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / n)

    nonzero = [d for d in diffs if d != 0.0]
    n_pos = sum(1 for d in nonzero if d > 0)
    n_neg = len(nonzero) - n_pos
    sign_p = _binom_two_sided(min(n_pos, n_neg), len(nonzero)) if nonzero else 1.0

    rng = random.Random(seed)
    obs = abs(mean)
    count = 0
    for _ in range(n_perm):
        m = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(m) >= obs - 1e-12:
            count += 1
    perm_p = (1 + count) / (1 + n_perm)

    if sd > 1e-12:
        d_cohen, note = mean / sd, None
    elif abs(mean) <= 1e-12:
        d_cohen, note = 0.0, None
    else:
        d_cohen, note = None, "degenerate_zero_variance"

    return {
        "n": n,
        "mean_signed_diff": mean,
        "mean_abs_diff": mean_abs,
        "sd": sd,
        "effect_size_cohens_d": d_cohen,
        "effect_note": note,
        "sign_test": {"n_nonzero": len(nonzero), "n_pos": n_pos,
                      "n_neg": n_neg, "p": sign_p},
        "permutation": {"n_perm": n_perm, "seed": seed, "p": perm_p},
    }
