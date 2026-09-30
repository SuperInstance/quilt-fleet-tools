import json

import pytest

import bench_seal
from bench_seal import ZERO, seal_benchmark, verify_chain


def _write(tmp_path, name, obj):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(obj))
    return str(p)


def test_seal_creates_receipt_and_chain(tmp_path):
    p = _write(tmp_path, "bench", {"k": [1, 2, 3]})
    r = seal_benchmark(p)
    assert (tmp_path / "bench.receipt.json").exists()
    assert (tmp_path / "bench-seal.chain.jsonl").exists()
    assert r["seq"] == 1 and r["prev"] == ZERO
    assert r["input_sha256"] == bench_seal.file_sha256(p)
    assert r["canonical_sha256"] == bench_seal.sha256_hex(bench_seal.canonical_json({"k": [1, 2, 3]}))
    rep = verify_chain(str(tmp_path))
    assert rep["ok"] and rep["n"] == 1
    assert rep["links"][0]["receipt_file"] == "match"


def test_env_fingerprint_present(tmp_path):
    r = seal_benchmark(_write(tmp_path, "e", {}))
    env = r["env"]
    assert env["python"].startswith("3.")
    assert env["cpu_count"] >= 1
    assert "timestamp" in env and "timestamp_unix" in env
    assert "platform" in env


def test_chain_appends_and_links(tmp_path):
    a = seal_benchmark(_write(tmp_path, "a", {"x": 1}))
    b = seal_benchmark(_write(tmp_path, "b", {"y": 2}))
    assert b["seq"] == 2 and b["prev"] == a["entry_hash"]
    rep = verify_chain(str(tmp_path))
    assert rep["ok"] and rep["n"] == 2


def test_tamper_detected(tmp_path):
    seal_benchmark(_write(tmp_path, "bench", {"v": 1}))
    cp = tmp_path / "bench-seal.chain.jsonl"
    e = json.loads(cp.read_text())
    e["input_sha256"] = "f" * 64  # forger swaps the input hash
    cp.write_text(json.dumps(e) + "\n")
    rep = verify_chain(str(tmp_path))
    assert not rep["ok"]
    assert any("entry_hash" in p for p in rep["links"][0]["problems"])


def test_reorder_detected(tmp_path):
    seal_benchmark(_write(tmp_path, "a", {"x": 1}))
    seal_benchmark(_write(tmp_path, "b", {"y": 2}))
    cp = tmp_path / "bench-seal.chain.jsonl"
    lines = cp.read_text().strip().splitlines()
    cp.write_text("\n".join(reversed(lines)) + "\n")
    rep = verify_chain(str(tmp_path))
    assert not rep["ok"]
    flat = " | ".join(p for l in rep["links"] for p in l["problems"])
    assert "seq" in flat and "prev" in flat


def test_receipt_file_tamper_detected(tmp_path):
    seal_benchmark(_write(tmp_path, "bench", {"v": 1}))
    rp = tmp_path / "bench.receipt.json"
    d = json.loads(rp.read_text())
    d["env"]["timestamp"] = "2030-01-01T00:00:00Z"  # forger edits the receipt copy
    rp.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    rep = verify_chain(str(tmp_path))
    assert not rep["ok"]
    assert rep["links"][0]["receipt_file"] == "MISMATCH"


def test_non_json_and_nan_refused(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json at all")
    with pytest.raises(ValueError):
        seal_benchmark(str(bad))
    nan = tmp_path / "nan.json"
    nan.write_text('{"v": NaN}')
    with pytest.raises(ValueError):
        seal_benchmark(str(nan))


def test_same_name_reseal_refused(tmp_path):
    p = _write(tmp_path, "bench", {"v": 1})
    seal_benchmark(p)
    with pytest.raises(ValueError):
        seal_benchmark(p)  # append-only: same slot, new content -> refuse


def test_cli_seal_and_verify_roundtrip(tmp_path):
    p = _write(tmp_path, "cli", {"runs": [1, 2, 3]})
    from bench_seal.__main__ import main
    assert main(["seal", p, "--out-dir", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "cli.receipt.json").exists()
    assert main(["verify", str(tmp_path / "out")]) == 0
