"""fleet_tools — sealed instrument gates for LLM-judge pipelines and benchmarks.

judge_gate: verify a judge panel is usable as an instrument (control-pair
stability + label-shuffle null) before its scores book as evidence.
bench_seal: sealed, hash-chained benchmark receipts (append-only, offline).
promise_census: promise->implementation linkage census for any repo
(generalized from wave-71 PROBE-1 synesis-promise-seal).
"""
from . import bench_seal, judge_gate, promise_census  # noqa: F401

__version__ = "0.1.0"
