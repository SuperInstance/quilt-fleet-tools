"""bench-seal CLI.

    python3 -m bench_seal seal INPUT.json [--name NAME] [--out-dir DIR]
    python3 -m bench_seal verify PATH   # dir with bench-seal.chain.jsonl, or the chain file

Exit codes: seal -> 0 on success; verify -> 0 chain PASS, 1 chain FAIL.
Fully offline; no network, no credentials.
"""
import argparse
import json
import sys

from . import seal_benchmark, verify_chain


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="bench-seal",
        description="seal benchmark JSON into a tamper-evident hash chain; verify offline")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seal", help="sha256 + env fingerprint a benchmark JSON; append to chain")
    s.add_argument("input", help="path to the benchmark .json file")
    s.add_argument("--name", default=None, help="receipt name (default: input filename stem)")
    s.add_argument("--out-dir", default=None,
                   help="receipt/chain directory (default: input file's directory)")

    v = sub.add_parser("verify", help="verify the whole chain offline; PASS/FAIL per link")
    v.add_argument("path", help="dir containing bench-seal.chain.jsonl, or the chain file itself")

    a = p.parse_args(argv)
    if a.cmd == "seal":
        entry = seal_benchmark(a.input, out_dir=a.out_dir, name=a.name)
        print(json.dumps({"sealed": True, "name": entry["name"], "seq": entry["seq"],
                          "entry_hash": entry["entry_hash"],
                          "input_sha256": entry["input_sha256"]}, indent=2))
        return 0
    rep = verify_chain(a.path)
    print(json.dumps(rep, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
