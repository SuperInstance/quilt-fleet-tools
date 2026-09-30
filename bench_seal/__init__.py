"""bench-seal: tamper-evident sealing and offline verification for benchmark runs.

Task 69-b implementation of the bench-seal sketch from study/DEEP-STUDY-68b.md.

Chain mechanics (sha256 over canonical JSON, sorted keys, ASCII-only):
  entry = {seq, prev, kind, name, input_file, input_sha256, canonical_sha256,
           input_size, env, entry_hash}
  entry_hash = sha256(canonical_json(entry minus entry_hash))
  genesis prev = 64 zeros; every later prev = previous entry's entry_hash.
  verify_chain() recomputes every link offline:
    - tampered field          -> entry_hash mismatch
    - reordered / spliced row -> seq or prev mismatch
    - edited receipt file     -> receipt_file cross-check MISMATCH

Deliberately self-contained: it does NOT import judge_gate. Two independent
canonical-JSON implementations sharing a spec is a feature — a bug in one does
not silently validate the other.

Non-negotiable input fence (68-b defect D6 lesson): non-finite numbers (NaN /
Inf) are REFUSED at seal time, not silently canonicalized.
"""
import hashlib
import json
import os
import platform
import time

KIND = "bench-seal/v0.1"
ZERO = "0" * 64
CHAIN_FILE = "bench-seal.chain.jsonl"


def canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, tight separators, ASCII-only, NaN/Inf refused."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def entry_hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "entry_hash"}
    return sha256_hex(canonical_json(body))


def env_fingerprint() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() if os.cpu_count() is not None else -1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timestamp_unix": int(time.time()),
    }


def read_chain(path: str) -> list:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _chain_path_for_dir(out_dir: str) -> str:
    return os.path.join(out_dir, CHAIN_FILE)


def _head_and_seq(chain_path: str):
    """-> (next_seq, prev_entry_hash) from the existing chain (genesis if absent/empty)."""
    if not os.path.exists(chain_path):
        return 1, ZERO
    entries = read_chain(chain_path)
    if not entries:
        return 1, ZERO
    return entries[-1]["seq"] + 1, entries[-1]["entry_hash"]


def seal_benchmark(input_path: str, out_dir: str | None = None,
                   name: str | None = None) -> dict:
    """Seal a benchmark JSON file: sha256 + env fingerprint -> receipt + chain append.

    Returns the sealed entry. Refuses:
      - input that is not valid JSON, is not an object/array, or contains
        non-finite numbers (NaN/Inf)
      - re-sealing a <name>.receipt.json that already exists with different
        content (append-only discipline: use a new name or a new out-dir)
    """
    p = os.path.abspath(input_path)
    if not os.path.isfile(p):
        raise ValueError(f"input not found: {p}")
    name = name or os.path.splitext(os.path.basename(p))[0]
    out_dir = os.path.abspath(out_dir or os.path.dirname(p))
    os.makedirs(out_dir, exist_ok=True)

    with open(p, "rb") as f:
        raw = f.read()
    input_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"input is not valid JSON: {e}") from e
    if not isinstance(parsed, (dict, list)):
        raise ValueError("benchmark JSON must be an object or an array")
    try:
        canonical = canonical_json(parsed)
    except ValueError as e:
        raise ValueError(f"benchmark JSON rejected (non-finite numbers are not "
                         f"sealable): {e}") from e

    chain_path = _chain_path_for_dir(out_dir)
    seq, prev = _head_and_seq(chain_path)
    entry = {
        "kind": KIND,
        "seq": seq,
        "prev": prev,
        "name": name,
        "input_file": os.path.basename(p),
        "input_sha256": input_sha256,
        "canonical_sha256": sha256_hex(canonical),
        "input_size": len(raw),
        "env": env_fingerprint(),
    }
    entry["entry_hash"] = entry_hash(entry)

    receipt_path = os.path.join(out_dir, f"{name}.receipt.json")
    if os.path.exists(receipt_path):
        with open(receipt_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("entry_hash") != entry["entry_hash"]:
            raise ValueError(f"{receipt_path} already sealed with different "
                             f"content; append-only: use a new name or out-dir")
    else:
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, sort_keys=True)
            f.write("\n")
    with open(chain_path, "a", encoding="utf-8") as f:
        f.write(canonical_json(entry) + "\n")
    return entry


def _resolve_chain_path(path: str) -> str:
    if os.path.isdir(path):
        for cand in (CHAIN_FILE, "chain.jsonl"):
            cp = os.path.join(path, cand)
            if os.path.exists(cp):
                return cp
        raise FileNotFoundError(f"no chain file in {path} (expected {CHAIN_FILE})")
    return path


def _receipt_file_state(base_dir: str, name, head_hash) -> tuple:
    """Cross-check <name>.receipt.json on disk against the chain entry hash."""
    if not name:
        return "missing"
    rf = os.path.join(base_dir, f"{name}.receipt.json")
    if not os.path.exists(rf):
        return "missing"
    try:
        with open(rf, encoding="utf-8") as f:
            disk = json.load(f)
        if entry_hash(disk) == disk.get("entry_hash") == head_hash:
            return "match"
        return "MISMATCH"
    except Exception:
        return "UNREADABLE"


def verify_chain(path: str) -> dict:
    """Offline full-chain verification. PASS/FAIL per link; detects tamper and reorder."""
    chain_path = _resolve_chain_path(path)
    entries = read_chain(chain_path)
    base_dir = os.path.dirname(os.path.abspath(chain_path))
    prev_expected = ZERO
    links = []
    ok = True
    for i, e in enumerate(entries, start=1):
        problems = []
        if e.get("seq") != i:
            problems.append(f"seq expected {i}, got {e.get('seq')} "
                            f"(row spliced, dropped, or reordered)")
        if e.get("prev") != prev_expected:
            problems.append("prev does not match previous entry_hash "
                            "(tamper, reorder, or gap)")
        h = e.get("entry_hash")
        if not h or entry_hash(e) != h:
            problems.append("entry_hash mismatch (content tampered)")
        rstate = _receipt_file_state(base_dir, e.get("name"), h)
        if rstate in ("MISMATCH", "UNREADABLE"):
            problems.append(f"receipt file {e.get('name')}.receipt.json is "
                            f"{rstate} vs chain entry")
        links.append({"seq": e.get("seq", i), "name": e.get("name"),
                      "ok": not problems, "problems": problems,
                      "receipt_file": rstate})
        if problems:
            ok = False
        prev_expected = h  # chain forward from what is written, so each link judged independently
    return {"ok": ok, "chain": chain_path, "n": len(entries), "links": links}
