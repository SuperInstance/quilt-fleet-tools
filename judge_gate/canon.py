"""Canonical-JSON helpers for judge-gate (ASCII canon law, 68-b D9/D10)."""
import hashlib
import json
import time


def canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, tight separators, ASCII-only, NaN/Inf refused."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
