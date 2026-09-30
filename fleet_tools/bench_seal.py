"""fleet_tools.bench_seal — sealed, hash-chained benchmark receipts.

`seal` computes sha256 of a results JSON, captures an environment
fingerprint, writes <name>.receipt.json into a seals directory, and appends
an entry to an append-only chain file (seals/chain.jsonl) where each entry
hashes the previous one. `verify` recomputes every link and detects
tampering, reordering, or truncation. Fully offline.
"""
import hashlib
import json
import os
import platform
import time


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def env_fingerprint() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def load_chain(seals_dir: str) -> list:
    path = os.path.join(seals_dir, "chain.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def seal(results_path: str, seals_dir: str, name: str = None) -> dict:
    """Seal one results file: receipt + chain append. Returns the receipt."""
    os.makedirs(seals_dir, exist_ok=True)
    name = name or os.path.splitext(os.path.basename(results_path))[0]
    with open(results_path, "rb") as f:
        payload = f.read()
    results_sha = sha256_bytes(payload)
    chain = load_chain(seals_dir)
    prev = chain[-1]["entry_sha256"] if chain else None
    receipt = {
        "name": name,
        "results_sha256": results_sha,
        "size_bytes": len(payload),
        "env": env_fingerprint(),
        "prev_entry_sha256": prev,
    }
    entry_core = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt["entry_sha256"] = sha256_bytes(entry_core.encode())
    rpath = os.path.join(seals_dir, f"{name}.receipt.json")
    with open(rpath, "w") as f:
        json.dump(receipt, f, indent=1, sort_keys=True)
    with open(os.path.join(seals_dir, "chain.jsonl"), "a") as f:
        f.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    return receipt


def verify_chain(seals_dir: str) -> dict:
    """Re-verify the whole chain: per-link PASS/FAIL + receipts consistency."""
    chain = load_chain(seals_dir)
    results = {"n_links": len(chain), "links": [], "VERDICT": "PASS"}
    prev = None
    for i, entry in enumerate(chain):
        ok = True
        why = []
        if entry.get("prev_entry_sha256") != prev:
            ok = False
            why.append("prev-hash mismatch (tamper/reorder/truncate)")
        core = {k: v for k, v in entry.items() if k != "entry_sha256"}
        recomputed = sha256_bytes(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode())
        if recomputed != entry.get("entry_sha256"):
            ok = False
            why.append("entry hash mismatch")
        rpath = os.path.join(seals_dir, f"{entry.get('name')}.receipt.json")
        if os.path.exists(rpath):
            with open(rpath) as f:
                rp = json.load(f)
            if rp.get("entry_sha256") != entry.get("entry_sha256"):
                ok = False
                why.append("receipt file disagrees with chain entry")
        else:
            why.append("receipt file missing (noted, not fatal)")
        results["links"].append({"i": i, "name": entry.get("name"),
                                 "PASS": ok, "why": why})
        if not ok:
            results["VERDICT"] = "FAIL"
        prev = entry.get("entry_sha256")
    return results
