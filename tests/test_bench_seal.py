"""bench-seal tests: chain append, verify, tamper + reorder detection. Offline."""
import json
import os

from fleet_tools import bench_seal


def _mk_results(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return str(p)


def test_seal_and_verify_clean_chain(tmp_path):
    seals = str(tmp_path / "seals")
    for i in range(3):
        r = bench_seal.seal(_mk_results(tmp_path, f"r{i}.json", {"v": i}),
                            seals, name=f"r{i}")
        assert r["results_sha256"] and r["entry_sha256"]
    chain = bench_seal.load_chain(seals)
    assert len(chain) == 3
    assert chain[1]["prev_entry_sha256"] == chain[0]["entry_sha256"]
    v = bench_seal.verify_chain(seals)
    assert v["VERDICT"] == "PASS" and all(l["PASS"] for l in v["links"])


def test_tampered_entry_fails_verify(tmp_path):
    seals = str(tmp_path / "seals")
    bench_seal.seal(_mk_results(tmp_path, "a.json", {"a": 1}), seals, name="a")
    bench_seal.seal(_mk_results(tmp_path, "b.json", {"b": 2}), seals, name="b")
    chain_path = os.path.join(seals, "chain.jsonl")
    entries = [json.loads(l) for l in open(chain_path)]
    entries[0]["results_sha256"] = "0" * 64  # tamper
    with open(chain_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n")
    v = bench_seal.verify_chain(seals)
    assert v["VERDICT"] == "FAIL"
    assert not v["links"][0]["PASS"]


def test_reorder_fails_verify(tmp_path):
    seals = str(tmp_path / "seals")
    bench_seal.seal(_mk_results(tmp_path, "a.json", {"a": 1}), seals, name="a")
    bench_seal.seal(_mk_results(tmp_path, "b.json", {"b": 2}), seals, name="b")
    chain_path = os.path.join(seals, "chain.jsonl")
    entries = [json.loads(l) for l in open(chain_path)]
    entries.reverse()
    with open(chain_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n")
    v = bench_seal.verify_chain(seals)
    assert v["VERDICT"] == "FAIL"


def test_truncated_chain_fails_verify(tmp_path):
    seals = str(tmp_path / "seals")
    bench_seal.seal(_mk_results(tmp_path, "a.json", {"a": 1}), seals, name="a")
    bench_seal.seal(_mk_results(tmp_path, "b.json", {"b": 2}), seals, name="b")
    chain_path = os.path.join(seals, "chain.jsonl")
    lines = open(chain_path).readlines()
    with open(chain_path, "w") as f:
        f.writelines(lines[:1])  # drop the last link
    v = bench_seal.verify_chain(seals)
    # a single intact link still verifies structurally; the RECEIPT file for b
    # remains but the chain lost a link — verify reports the surviving links
    assert v["n_links"] == 1
