"""CLI:
  python -m fleet_tools bench-seal seal RESULTS.json --dir seals/ [--name tag]
  python -m fleet_tools bench-seal verify --dir seals/
  python -m fleet_tools judge-gate --items items.json --judges M1,M2 --out receipts/
    items.json: [{"id": "...", "text": "...", "label_true": "A", "label_swapped": "B"}]
  python -m fleet_tools promise-census REPO [REPO...] [--out FILE]
    [--tag-crosscheck REGEX]   # e.g. '<promise>[A-Z0-9_]+</promise>'
  python -m fleet_tools rehydrate-plan --curve CURVE.json [--stream-length N]
    [--prefix-known P] [--replica-depth K] [--cost-full X] [--cost-per-diff Y]
    [--threshold 0.02] [--fit logistic|piecewise]   # prints decision JSON
"""
import argparse
import json
import sys

from . import bench_seal, judge_gate, rehydrate, promise_census


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

    pc = sub.add_parser("promise-census",
                        help="promise->implementation linkage census per repo")
    pc.add_argument("repos", nargs="+")
    pc.add_argument("--out", help="write combined JSON here (also prints)")
    pc.add_argument("--tag-crosscheck", default=None,
                    help="optional tag regex cross-check (e.g. wave-71 "
                         "'<promise>[A-Z0-9_]+</promise>')")

    rp = sub.add_parser("rehydrate-plan",
                        help="catchup-vs-rehydrate decision for a decayed stream")
    rp.add_argument("--curve", required=True,
                    help="decay curve JSON (w72-decay-curve.json or bare point list)")
    rp.add_argument("--stream-length", type=int, default=None,
                    help="canonical stream depth n (default: curve's stream_commits)")
    rp.add_argument("--prefix-known", type=int, default=None,
                    help="verifiable prefix receipts p (default: n, i.e. intact)")
    rp.add_argument("--replica-depth", type=int, default=0,
                    help="replica checkpoint depth k")
    rp.add_argument("--cost-full", type=float, default=300.0,
                    help="cost of one full rehydration from seal (policy knob)")
    rp.add_argument("--cost-per-diff", type=float, default=1.0,
                    help="cost per incrementally applied diff (policy knob)")
    rp.add_argument("--threshold", type=float, default=rehydrate.DEFAULT_THRESHOLD,
                    help="max acceptable expected divergence fraction (default 0.02)")
    rp.add_argument("--fit", choices=("logistic", "piecewise"), default="logistic")

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
    elif args.cmd == "promise-census":
        results = promise_census.census_repos(
            args.repos, out_path=args.out, cross_check_tag=args.tag_crosscheck)
        print(json.dumps(results, indent=1, sort_keys=True))
    elif args.cmd == "rehydrate-plan":
        points, meta = rehydrate.load_curve(args.curve)
        model = rehydrate.DecayModel.fit(points, kind=args.fit)
        n = args.stream_length if args.stream_length is not None else meta.get("stream_commits")
        if n is None:
            ap.error("--stream-length required (curve JSON has no stream_commits)")
        p = args.prefix_known if args.prefix_known is not None else n
        sched = rehydrate.RehydrationScheduler(model, threshold=args.threshold)
        decision = sched.decide(p, args.replica_depth, n,
                                args.cost_full, args.cost_per_diff)
        decision["model"] = {"kind": model.kind, "params": model.params,
                             "r2": model.r2, "points": len(model.points)}
        print(json.dumps(decision, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
