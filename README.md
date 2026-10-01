# quilt-fleet-tools

Sealed instrument gates for the SuperInstance quilt fleet:

- **judge_gate** — verifies an LLM judge panel is usable as an instrument
  BEFORE its scores book as evidence:
  - *control pair* (stability): same block, identical labels, twice;
    max per-item |delta| must be within tolerance.
  - *label-shuffle null* (bias): the same artifacts judged under swapped
    labels; scores must not track the label (mean |diff| and |mean signed|
    thresholds). A biased mock judge FAILS; an independent mock PASSES —
    both are pinned in the test suite.
  - Every run writes a **sealed receipt** with sha256 of inputs/outputs and
    a prev-receipt hash chain inside the receipts directory.
  - Judge backend: DeepInfra OpenAI-compatible API with a HARD allowlist of
    the 7 fleet models; call-capped ledger; every call logged with usage.

- **bench_seal** — sealed, hash-chained benchmark receipts, fully offline:
  `seal` fingerprints a results file (sha256 + env) and appends an entry to
  an append-only chain where each entry hashes the previous one; `verify`
  recomputes every link and detects tampering, reordering, and truncation.

- **rehydrate** — decay-curve model + rehydration scheduler (wave-72 seed):
  fits divergence fraction d(t) vs truncation depth t from a measured curve
  (anchored logistic or piecewise-linear; both pinned at the PROBE-2-proven
  d(0)=0 / d(1)=1), then decides `catchup` vs `rehydrate` for a replica
  checkpointed at depth k of an n-diff stream where only p prefix receipts
  are verifiable. Deterministic, offline, no network.

## Honest scope

- judge_gate verifies the INSTRUMENT (stability + label independence). It
  does not certify judge competence — a consistently mediocre judge passes.
- bench_seal proves a file's bytes at seal time and chain integrity; it does
  not prove the benchmark was run honestly.
- The gate thresholds (tol 1.0, max_abs 1.0, max_signed 0.5) are defaults,
  not laws — pre-register yours before a study.

## Usage

```sh
python -m fleet_tools bench-seal seal outputs/bench.json --dir seals/
python -m fleet_tools bench-seal verify --dir seals/
python -m fleet_tools judge-gate --items items.json --judges tencent/Hy3,inclusionAI/Ling-3.0-flash --out gate_receipts/
python -m fleet_tools rehydrate-plan --curve outputs/w72-decay-curve.json \
  --stream-length 101 --prefix-known 51 --replica-depth 30 \
  --cost-full 300 --cost-per-diff 1 --threshold 0.02
```

`items.json` entries: `{"id", "text", "label_true", "label_swapped"}`.

## Tests

```sh
python -m pytest tests/ -q   # offline; mock judges; no network
```
