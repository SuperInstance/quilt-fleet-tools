# quilt-fleet-tools

Two sealed-instrument tools for model-evaluation pipelines, grown from the
bench-seal / judge-gate sketches in `study/DEEP-STUDY-68b.md` (SuperInstance
quilt project, task 69-b):

- **judge-gate** — judge-instrument verification for LLM-as-judge: does the judge
  actually score the *content*, or the *author label*?
- **bench-seal** — tamper-evident benchmark receipt sealing with offline
  whole-chain verification.

Python 3.12, pytest, MIT. One runtime dependency (`requests`). The judge-gate
CLI talks to DeepInfra only when you ask it to; every test and every bench-seal
operation is fully offline.

```bash
python3 -m pytest -q                        # offline; no network, no keys
```

---

## judge-gate

**Question it answers:** if I hand the same transcript to the judge under two
different author labels, do the scores move? A judge that favors a label is not
an instrument; it is a party.

**Design** (N >= 6 transcripts, enforced):

1. Every transcript is judged twice — once as `Author A`, once as `Author B`:
   same words, two labels. A fixed-seed RNG decides which label each paired run
   presents first, so presentation order cannot masquerade as a label effect.
2. **Control-pair gate:** `|score(A) - score(B)| <= tolerance` for every pair
   (default tolerance 1.0 on the 1-10 scale).
3. **Label-shuffle null:** over the paired signed diffs, a seeded sign-flip
   permutation test (default 5000 perms) at alpha (default 0.05), plus an exact
   sign test reported as supporting evidence. This catches a *consistent* bias
   that squeaks under the tolerance (e.g. +0.5 every time).
4. **Verdict:** `PASS` (both gates clean), `FAIL` (label-linked scoring
   detected), or `INCONCLUSIVE` (judge call errors or unparseable-score
   refusals — booked, never retried away). Effect size = Cohen's d on the diffs.

**Judge backend:** DeepInfra OpenAI-compatible
(`POST https://api.deepinfra.com/v1/openai/chat/completions`, Bearer from env
`DEEPINFRA_API_KEY`). Hard allowlist of exactly seven model ids
(`judge_gate/allowlist.py`); anything else is rejected before any HTTP. Max 90
calls per invocation; every call logged with latency + usage; `max_tokens` 4096
default with ONE retry at 1.5x on `finish_reason=length` (cap 8192). The
`JudgeClient` protocol (`judge()` + `call_log`) is implemented by
`DeepInfraJudgeClient` and by a deterministic offline `MockJudgeClient`, so all
sealing and gating logic is testable with zero network.

**Sealed receipt per run** (`<run_id>.receipt.json` + one canonical line in
`judge-gate.chain.jsonl`):

- `inputs_sha256` — sha256 over canonical JSON of transcripts + prompts + model + params
- `ts_start` / `ts_end`, every raw judge output + usage (`calls`, `judgments`)
- `stats`, `verdict`, `reasons`, `max_abs_label_diff`
- `seal` — sha256 of the canonical receipt; `prev_seal` — the previous
  receipt's seal in that out-dir (genesis = 64 zeros); `seq` — chain position.
  Append-only: re-sealing an existing run_id with different content is refused.

```bash
# live (12 judge calls for 6 transcripts; needs DEEPINFRA_API_KEY in env)
python3 -m judge_gate run --transcripts examples/transcripts.json \
    --model thinkingmachines/Inkling-Small --out-dir receipts/

# offline harness self-test (marked "mock": true in the receipt)
python3 -m judge_gate run --transcripts examples/transcripts.json \
    --mock biased --out-dir receipts-mock/        # expect verdict FAIL (exit 1)
python3 -m judge_gate run --transcripts examples/transcripts.json \
    --mock independent --out-dir receipts-mock/   # expect verdict PASS (exit 0)

# offline chain verification
python3 -m judge_gate verify receipts/

# one-call live smoke (1 chat call)
python3 examples/live_smoke.py [model_id]
```

Exit codes: run -> 0 PASS / 1 FAIL / 2 INCONCLUSIVE; verify -> 0 / 1.

## bench-seal

**Question it answers:** is the file I am quoting right now byte-for-byte the
file the run produced?

```bash
python3 -m bench_seal seal results/bench.json --out-dir results/
# -> results/bench.receipt.json + append-only results/bench-seal.chain.jsonl

python3 -m bench_seal verify results/          # or the chain file path directly
```

`seal` hashes the input bytes AND the canonical JSON form, fingerprints the
environment (python version, platform, cpu count, UTC timestamp), refuses
non-finite numbers (NaN/Inf) and non-JSON input, and refuses re-sealing an
existing `<name>.receipt.json` with different content. `verify` recomputes
every link offline: a tampered field breaks its `entry_hash`; a spliced,
dropped, or reordered row breaks `seq`/`prev`; an edited receipt file breaks
the chain-vs-file cross-check. PASS/FAIL is reported per link. No network, no
credentials, no dependencies.

## Receipt / chain format

Both tools use the same idiom: canonical JSON (sorted keys, tight separators,
ASCII-only, NaN/Inf refused), sha256 chained receipts, genesis = `"0"*64`.
The implementations are deliberately independent (two small `canonical_json`
functions, one per package) — a bug in one cannot silently validate the other.

## What this does NOT do (read before trusting a green receipt)

- judge-gate does **not** prove a judge is unbiased. It detects the planted
  author-label bias class (label attribution + presentation order). A judge can
  still be biased by writing style, length, topic, or model family — out of
  scope here.
- The permutation test at alpha=0.05 will flag a *fair* judge about 5% of the
  time (plus tolerance-gate risk). alpha/tolerance are policy knobs, not truth.
- Mock receipts are marked `"mock": true`. They are harness self-tests, not
  model certifications; a mock PASS says the harness works, nothing about any
  real judge.
- bench-seal proves **bytes, not semantics**: a green verify says the file you
  hold is the file that was sealed — it does not say the benchmark was any good.
- The 90-call budget and the 7-model allowlist are this repo's policy, not a
  property of DeepInfra.

## When it fails — what to do

- judge-gate `FAIL`: the judge moves scores with the author label. Do not use
  it for scored comparison without remediation (strip/blind labels, rotate
  judges, re-run the gate). The receipt's `judgments` + `reasons` say which
  gate fired and by how much.
- judge-gate `INCONCLUSIVE`: some judge calls failed or answered off-format.
  Read `calls[].error` / `judgments[].parsed_from`; fix transport or prompt
  format; re-run. Do not average over partial pairs.
- `verify` FAIL (either tool): the chain is tampered or reordered. Do not trust
  it; compare against an external copy (the chain head recorded outside the
  directory is the reference). `verify` tells you the first broken link.

## Tests & CI

- `tests/` — pytest, zero network: chain append/verify/tamper/reorder for both
  tools, allowlist rejection before HTTP, budget cap, length-retry at 1.5x +
  8192 cap via an injected fake transport, label-shuffle stats with mock judges
  (biased -> FAIL, independent -> PASS).
- `.github/workflows/ci.yml` — pytest job + a key-scan **negative control**
  job (the scanner's own regex literal is the only permitted self-match) + an
  offline self-play smoke that asserts the biased mock gate exits 1.

## Provenance

Grown in the SuperInstance quilt engineering marathon (task 69-b) from the
`bench-seal` / `judge-gate` contracts sketched in `study/DEEP-STUDY-68b.md`
(a defect-census deep-study of SuperInstance/quilt-tools). The receipt idiom
follows the fleet WitnessLog laws: ASCII-only canon, hash-chained rows,
refusals booked as receipts, `main` stays green.

MIT — see [LICENSE](LICENSE).
