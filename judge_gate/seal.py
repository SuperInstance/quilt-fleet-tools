"""Sealed receipts for judge-gate: sha256 over canonical JSON + prev_seal chaining.

Per run: <run_id>.receipt.json in the out-dir, plus one canonical line appended
to judge-gate.chain.jsonl. Each receipt carries:
  seal      = sha256(canonical_json(receipt minus "seal"))
  prev_seal = seal of the previous receipt in the out-dir (genesis = 64 zeros)
  seq       = 1-based position in the dir's chain

Append-only discipline: sealing a run_id that already exists on disk with a
different seal is REFUSED (pick a new run_id). verify_receipts() re-checks the
whole chain offline: seal recompute, prev_seal linkage, seq order, and a
cross-check of each on-disk receipt file against its chain line.
"""
import glob
import json
import os

from .canon import canonical_json, read_jsonl, sha256_hex

GENESIS = "0" * 64
CHAIN_FILE = "judge-gate.chain.jsonl"


def receipt_seal(receipt: dict) -> str:
    body = {k: v for k, v in receipt.items() if k != "seal"}
    return sha256_hex(canonical_json(body))


def seal_receipt(receipt: dict, out_dir: str) -> dict:
    """Assign seq + prev_seal + seal; write the receipt file and append the chain."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    chain_path = os.path.join(out_dir, CHAIN_FILE)
    entries = read_jsonl(chain_path) if os.path.exists(chain_path) else []
    receipt = dict(receipt)
    receipt["seq"] = entries[-1]["seq"] + 1 if entries else 1
    receipt["prev_seal"] = entries[-1]["seal"] if entries else GENESIS
    receipt["seal"] = receipt_seal(receipt)

    path = os.path.join(out_dir, f"{receipt['run_id']}.receipt.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("seal") != receipt["seal"]:
            raise ValueError(f"{path} already sealed with different content; "
                             f"append-only: pick a new run_id")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, sort_keys=True)
            f.write("\n")
    with open(chain_path, "a", encoding="utf-8") as f:
        f.write(canonical_json(receipt) + "\n")
    return receipt


def _receipt_file_state(base_dir: str, run_id, head_seal) -> str:
    if not run_id:
        return "missing"
    rf = os.path.join(base_dir, f"{run_id}.receipt.json")
    if not os.path.exists(rf):
        return "missing"
    try:
        with open(rf, encoding="utf-8") as f:
            disk = json.load(f)
        if receipt_seal(disk) == disk.get("seal") == head_seal:
            return "match"
        return "MISMATCH"
    except Exception:
        return "UNREADABLE"


def verify_receipts(path: str) -> dict:
    """Offline verification of a judge-gate receipt chain (dir or chain file)."""
    if os.path.isdir(path):
        chain_path = os.path.join(path, CHAIN_FILE)
        if os.path.exists(chain_path):
            entries = read_jsonl(chain_path)
            base_dir = path
        else:
            entries = []
            for p in glob.glob(os.path.join(path, "*.receipt.json")):
                with open(p, encoding="utf-8") as f:
                    entries.append(json.load(f))
            entries.sort(key=lambda r: (r.get("seq", 0), r.get("run_id", "")))
            base_dir = path
    else:
        chain_path = os.path.abspath(path)
        entries = read_jsonl(chain_path)
        base_dir = os.path.dirname(chain_path)

    prev = GENESIS
    links = []
    ok = True
    for i, r in enumerate(entries, start=1):
        problems = []
        if r.get("seq") != i:
            problems.append(f"seq expected {i}, got {r.get('seq')} "
                            f"(reordered, spliced, or dropped)")
        if r.get("prev_seal") != prev:
            problems.append("prev_seal does not match previous receipt seal "
                            "(tamper, reorder, or gap)")
        h = r.get("seal")
        if not h or receipt_seal(r) != h:
            problems.append("seal mismatch (content tampered)")
        rstate = _receipt_file_state(base_dir, r.get("run_id"), h)
        if rstate in ("MISMATCH", "UNREADABLE"):
            problems.append(f"receipt file {r.get('run_id')}.receipt.json is "
                            f"{rstate} vs chain")
        links.append({"seq": r.get("seq", i), "run_id": r.get("run_id"),
                      "verdict": r.get("verdict"), "ok": not problems,
                      "problems": problems, "receipt_file": rstate})
        if problems:
            ok = False
        prev = h
    return {"ok": ok, "chain": chain_path if entries else None,
            "n": len(entries), "links": links}
