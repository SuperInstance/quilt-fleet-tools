"""promise-census tests: detection, unit extraction, linkage, binary skip.
Fixture repo built in tmp_path: one linked promise, one unlinked promise.
Offline, deterministic."""
import json

from fleet_tools import promise_census as pc


def _mk_repo(tmp_path):
    """Tiny repo: ROADMAP.md (promise doc) with one linked + one unlinked
    promise unit; one implementation file; one binary file."""
    root = tmp_path / "fixture-repo"
    (root / "src").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "docs" / "ROADMAP.md").write_text(
        "# Roadmap\n\n"
        "We plan to ship the planned modules below.\n\n"
        "- Add `fusion_core` module for planar sheets\n"
        "- Add `warp_drive` module for superluminal merges\n"
        "- Improve error messages everywhere\n"
    )
    (root / "src" / "fusion_core.py").write_text(
        '"""fusion_core: planar sheet fusion."""\n'
        "def fuse(cells):\n"
        "    return cells\n"
    )
    (root / "src" / "blob.bin").write_bytes(b"\x00\x01\x02binary\0")
    return root


def test_promise_doc_detection_patterns():
    assert pc.is_promise_doc("# Roadmap\n- item") == ["roadmap_header"]
    assert pc.is_promise_doc("<promise>TXRS_COMPLETE</promise>") == ["promise_tag"]
    assert "we_plan" in pc.is_promise_doc("We plan to add retries.")
    assert "will_commit" in pc.is_promise_doc("The CLI will implement retries.")
    assert "todo_implement" in pc.is_promise_doc("TODO: implement parser")
    assert pc.is_promise_doc("# Baking recipes\n2 cups flour.\nEnjoy.") == []
    # "will" alone is not a commitment: no promise verb follows
    assert pc.is_promise_doc("The sauce will thicken.") == []


def test_unit_extraction_and_linkage_fixture(tmp_path):
    root = _mk_repo(tmp_path)
    r = pc.census_repo(str(root), repo_name="fixture-repo")
    assert r["promise_docs"] == 1
    assert "docs/ROADMAP.md" in r.get("promise_doc_paths", [])
    # 3 bullet units + standalone identifiers (fusion_core, warp_drive)
    assert r["promise_units"] >= 4
    linked_units = {(e["doc"], e["unit"]) for e in r["linked_examples"]}
    # the linked promise: fusion_core appears in implementation bytes
    assert any("fusion_core" in u for _, u in linked_units)
    # the unlinked promise: warp_drive appears nowhere in code
    unlinked = {(e["doc"], e["unit"]) for e in r["unlinked_examples"]}
    assert any("warp_drive" in u for _, u in unlinked)
    assert 0.0 < r["linkage_ratio"] < 1.0
    # binary file skipped, implementation file collected
    assert r["impl_files"] == 1 and r["impl_bytes"] > 0


def test_ngram_linkage_without_identifier(tmp_path):
    root = tmp_path / "ngram-repo"
    root.mkdir()
    (root / "PLAN.md").write_text(
        "# Plan\n\n- The engine uses deterministic merge orders\n")
    (root / "engine.js").write_text("// the engine uses deterministic merge orders\n")
    r = pc.census_repo(str(root), repo_name="ngram-repo")
    assert r["linked_units"] >= 1
    assert any(e["via"].startswith("ngram:") for e in r["linked_examples"])


def test_docs_only_repo_zero_linkage(tmp_path):
    root = tmp_path / "docs-only"
    root.mkdir()
    (root / "README.md").write_text("# Roadmap\n\n- Will implement `solver_rs`\n")
    r = pc.census_repo(str(root), repo_name="docs-only")
    assert r["promise_docs"] == 1 and r["promise_units"] >= 1
    assert r["linked_units"] == 0 and r["linkage_ratio"] == 0.0
    assert r["impl_files"] == 0
    assert any("docs-only" in n or "no implementation" in n for n in r["notes"])


def test_cross_check_tag(tmp_path):
    root = tmp_path / "tag-repo"
    root.mkdir()
    (root / "A.md").write_text("<promise>MOD_COMPLETE</promise>\n- `alpha_core` done\n")
    (root / "B.md").write_text("# Roadmap\n- plain bullet\n")
    (root / "impl.py").write_text("x = 1\n")
    r = pc.census_repo(str(root), repo_name="tag-repo",
                       cross_check_tag=r"<promise>[A-Z0-9_]+</promise>")
    cc = r["cross_check"]
    assert cc["tag_files"] == 1
    assert cc["tag_files_also_promise_docs"] == 1
    assert cc["tag_linkage_ratio"] == 0.0


def test_census_repos_combined_and_deterministic(tmp_path):
    a = tmp_path / "repo-a"
    a.mkdir()
    (a / "PLAN.md").write_text("# Roadmap\n- `shared_util` module\n")
    b = tmp_path / "repo-b"
    b.mkdir()
    (b / "PLAN.md").write_text("# Roadmap\n- `shared_util` module\n")
    (b / "lib.py").write_text("import shared_util\n")
    out = tmp_path / "census" / "combined.json"
    r1 = pc.census_repos([str(a), str(b)], out_path=str(out))
    r2 = pc.census_repos([str(a), str(b)])
    assert [x["repo"] for x in r1] == ["repo-a", "repo-b"]
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    assert r1[0]["linkage_ratio"] == 0.0 and r1[1]["linkage_ratio"] > 0.0
    saved = json.loads(out.read_text())
    assert len(saved) == 2


def test_cli_promise_census(tmp_path, capsys):
    from fleet_tools.__main__ import main
    import sys
    root = _mk_repo(tmp_path)
    out = tmp_path / "o" / "census.json"
    argv = sys.argv
    sys.argv = ["fleet_tools", "promise-census", str(root),
                "--out", str(out)]
    try:
        main()
    finally:
        sys.argv = argv
    data = json.loads(capsys.readouterr().out)
    assert data[0]["promise_docs"] == 1
    assert json.loads(out.read_text())[0]["repo"] == "fixture-repo"
