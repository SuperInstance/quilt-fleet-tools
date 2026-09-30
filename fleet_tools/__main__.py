"""CLI:
  python -m fleet_tools bench-seal seal RESULTS.json --dir seals/ [--name tag]
  python -m fleet_tools bench-seal verify --dir seals/
  python -m fleet_tools judge-gate --items items.json --judges M1,M2 --out receipts/
    items.json: [{"id": "...", "text": "...", "label_true": "A", "label_swapped": "B"}]
"""
import argparse
import json
import sys

from . import bench_seal, judge_gate


def main():
    ap = argparse.ArgumentParser(prog="fleet_tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    seal_p = sub.add_parser("bench-seal")
    seal_sub = seal_p.add_subparsers(dest="op", required=True)
    s1 = seal_sub.add_parser("seal")
    s1.add_argument("results")
    s1.add_argument("--dir", default="seals")
    s1.add_argument("--name")
    s2 = seal_sub.add_parser("verify")
    s2.add_argument("--dir", default="seals")

    jg = sub.add_parser("judge-gate")
    jg.add_argument("--items", required=True)
    jg.add_argument("--judges", required=True,
                    help="comma-separated allowlisted judge model ids")
    jg.add_argument("--out", default="gate_receipts")

    args = ap.parse_args()
    if args.cmd == "bench-seal":
        if args.op == "seal":
            r = bench_seal.seal(args.results, args.dir, args.name)
            print(json.dumps(r, indent=1, sort_keys=True))
        else:
            v = bench_seal.verify_chain(args.dir)
            print(json.dumps(v, indent=1))
            sys.exit(0 if v["VERDICT"] == "PASS" else 1)
    elif args.cmd == "judge-gate":
        with open(args.items) as f:
            items = json.load(f)
        judges = [j.strip() for j in args.judges.split(",") if j.strip()]
        client = judge_gate.DeepInfraJudgeClient()
        receipt = judge_gate.run_gate(client, judges, items, args.out)
        out = {k: v for k, v in receipt.items() if k != "usage"}
        print(json.dumps(out, indent=1, sort_keys=True))
        sys.exit(0 if receipt["GATE"] == "PASS" else 1)


if __name__ == "__main__":
    main()
