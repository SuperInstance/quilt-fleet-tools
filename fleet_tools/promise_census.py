"""fleet_tools.promise_census — promise→implementation linkage census.

A standing atlas instrument generalized from wave-71 PROBE-1
(synesis-promise-seal). For any repository it:

1. finds PROMISE DOCUMENTS: markdown files containing forward-looking
   commitment patterns (promise tags, roadmap/planned/todo headers,
   "we plan to", "will implement", "not yet implemented", planned
   crate/module lists);
2. extracts PROMISE UNITS: bullet/list items plus crate/module-style
   identifiers (backticked spans, snake_case, CamelCase, *-rs crates);
3. collects IMPLEMENTATION BYTES: every code/build/config text file
   (binary files skipped via a NUL-byte probe);
4. computes LINKAGE: a unit is linked iff one of its identifiers
   (case-sensitive substring) or one of its distinctive bigrams
   (2 consecutive words, lowercased, alphanumeric-only, both >=3 chars
   and non-stopword) appears in the implementation corpus. Implementation
   bytes = code + build/config text files; markdown, prose-data (.txt
   etc.) and repo plumbing (LICENSE, .gitignore) do not count.

Output per repo (deterministic, sorted by path, no network, no LLM):
  {repo, path, promise_docs, promise_units, linked_units, linkage_ratio,
   impl_files, impl_bytes, unlinked_examples, linked_examples, notes,
   [promise_doc_paths], [cross_check]}

Linkage is LEXICAL: it certifies name/phrase overlap between a promise and
implementation text, not causal fulfillment. See LIMITATIONS in the atlas
study (name collisions, quoted code, docs-as-code repos link by design).
"""
import json
import os
import re

# --------------------------------------------------------------------------
# promise-document detection
# --------------------------------------------------------------------------
PROMISE_DOC_PATTERNS = [
    # literal promise token, wave-71's synesis marker: <promise>TXRS_COMPLETE</promise>
    ("promise_tag", re.compile(r"<promise>[A-Za-z0-9_]+</promise>", re.I)),
    # roadmap-ish headers: ## Roadmap / ## Planned crates / ## Future work / ## TODO / ## Spec
    ("roadmap_header", re.compile(
        r"(?im)^#{1,6}[^\n]*\b(roadmap|planned|plan\b|planning|upcoming|backlog"
        r"|future work|next steps|milestones?|todo|specs?|specifications?)\b")),
    # "we plan to ...", "we intend to ..."
    ("we_plan", re.compile(
        r"(?i)\b(we|i)\s+(plan|intend|aim|expect|propose|commit)\s+to\b")),
    # "will implement / will be implemented / shall provide ..."
    ("will_commit", re.compile(
        r"(?i)\b(will|shall)\s+(be\s+)?"
        r"(implement|add|support|ship|release|provide|build|deliver|land|publish"
        r"|introduce|extend|replace|migrate|include|cover|ship)\w*\b")),
    # "TODO: implement ...", "not yet implemented"
    ("todo_implement", re.compile(
        r"(?i)\b(todo\s*:?|not\s+yet\s+implemented|to\s+be\s+implemented)\b")),
    # planned/proposed crate|module|package lists
    ("planned_crate_list", re.compile(
        r"(?i)\b(planned|proposed|future|upcoming)\s+"
        r"(crates?|modules?|packages?|components?|cells?)\b")),
]

PROMISE_TAG_RE = re.compile(r"<promise>[A-Z0-9_]+</promise>")

MARKDOWN_EXTS = {".md", ".markdown", ".mdown"}

# implementation bytes = code + build/config text. NOT markdown, NOT
# prose-data (.txt/.csv/.tsv — synesis's .research_summary.txt is notes,
# not implementation), NOT repo plumbing (LICENSE/.gitignore). Mirrors
# wave-71 PROBE-1's registered impl set (rs/toml/json/ts/js/py, non-md)
# while generalizing to the fleet's polyglot repos.
IMPL_EXTS = {
    ".py", ".rs", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".hh", ".go", ".java", ".rb",
    ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".toml", ".yaml", ".yml", ".json", ".jsonl",
    ".ini", ".cfg", ".conf", ".sql", ".xml", ".html", ".htm", ".css",
    ".scss", ".less", ".vue", ".svelte", ".swift", ".kt", ".scala",
    ".lua", ".pl", ".php",
    ".mo", ".wat", ".nim", ".zig", ".jl", ".ex", ".exs",
    ".bat", ".lock", ".proto", ".graphql",
}
IMPL_PLAINNAMES = {"Makefile", "Dockerfile"}

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
             ".pytest_cache", "target", "dist", "build"}
MAX_FILE_TEXT = 1 << 20  # cap per-file implementation text (deterministic)
BINARY_PROBE = 8192

# identifiers: backticked spans, snake_case, CamelCase, dotted paths, *-rs crates
IDENT_RE = re.compile(
    r"`([^`\n]{3,})`"                                   # backticked span
    r"|\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b"             # snake_case
    r"|\b([A-Z][a-z0-9]*(?:[A-Z][a-z0-9]+)+)\b"         # CamelCase
    r"|\b([a-z][a-z0-9]*(?:-[a-z0-9]+)*-rs)\b"          # hyphen crate (-rs)
    r"|\b([a-z][a-z0-9]*(?:/[a-z0-9_.-]+)+)\b")         # dotted/slashed path
MAX_IDENTS_PER_DOC = 200

BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(\S.*)$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")

_STOP = set("""the a an and or but if then than that this these those there here
of to in on for with without from by at as is are was were be been being am
it its it's we our us you your they them their he she his her i me my mine
all any some such only also just not no nor so too very can could may might
must shall should will would do does did done has have had get gets got use
used uses using one two three first second next last new old more most other
per via via how what when where which who whom why each both any few own same
out up down over under again once about into through during before after
above below between because while until against""".split())


def is_promise_doc(text: str):
    """Return list of matched pattern names (non-empty => promise doc)."""
    hits = []
    for name, rx in PROMISE_DOC_PATTERNS:
        if rx.search(text):
            hits.append(name)
    return hits


def _looks_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(BINARY_PROBE)
    except OSError:
        return True


def collect_impl_corpus(repo_root: str):
    """All non-markdown text/code files -> {relpath: text} (deterministic)."""
    corpus = {}
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1]
            if ext in MARKDOWN_EXTS:
                continue
            if ext not in IMPL_EXTS and fn not in IMPL_PLAINNAMES:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, repo_root).replace(os.sep, "/")
            if _looks_binary(full):
                continue
            try:
                with open(full, "r", errors="replace") as f:
                    corpus[rel] = f.read(MAX_FILE_TEXT)
            except OSError:
                continue
    return corpus


def idents_in(s: str):
    """Distinct identifiers (>=3 chars) inside one promise-unit string."""
    out, seen = [], set()
    for t in IDENT_RE.findall(s):
        tok = next(x for x in t if x)
        if len(tok) >= 3 and tok.lower() not in _STOP and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def extract_promise_units(text: str):
    """(bullet_items, identifiers) from one markdown document.

    bullets: list items outside code fences, in order, deduped.
    identifiers: crate/module-style identifiers (>=3 chars), doc-level,
    deduped, capped at MAX_IDENTS_PER_DOC (used as standalone units).
    """
    bullets, b_seen = [], set()
    idents, seen_i = [], set()
    in_fence = False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = BULLET_RE.match(line)
        if m:
            u = m.group(1).strip()
            if u and u not in b_seen:
                b_seen.add(u)
                bullets.append(u)
        if len(idents) < MAX_IDENTS_PER_DOC:
            for tok in idents_in(line):
                if tok not in seen_i:
                    seen_i.add(tok)
                    idents.append(tok)
    return bullets, idents


def _bigrams(unit: str):
    words = re.findall(r"[a-z0-9]+", unit.lower())
    out = []
    for a, b in zip(words, words[1:]):
        if len(a) >= 3 and len(b) >= 3 and a not in _STOP and b not in _STOP:
            out.append(f"{a} {b}")
    return out


def unit_linkage(unit: str, idents, impl_raw: str, impl_lower: str):
    """Return match descriptor or None."""
    for tok in idents:
        if tok in impl_raw:
            return "identifier:" + tok
    for bg in _bigrams(unit):
        if bg in impl_lower:
            return "ngram:'" + bg + "'"
    return None


def _clip(s: str, n: int = 160) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def census_repo(repo_root: str, repo_name: str = None,
                cross_check_tag: str = None, max_examples: int = 20) -> dict:
    """Run the promise-linkage census on one repository directory."""
    repo_root = os.path.abspath(repo_root)
    repo_name = repo_name or os.path.basename(repo_root)
    notes = []

    # 1-2. promise documents + units
    md_paths = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1] in MARKDOWN_EXTS:
                md_paths.append(os.path.relpath(
                    os.path.join(dirpath, fn), repo_root).replace(os.sep, "/"))

    promise_docs, units = [], []   # units: (doc, text, idents)
    pattern_hits = {}
    for rel in sorted(md_paths):
        with open(os.path.join(repo_root, rel), "r", errors="replace") as f:
            text = f.read()
        hits = is_promise_doc(text)
        if not hits:
            continue
        promise_docs.append(rel)
        for h in hits:
            pattern_hits[h] = pattern_hits.get(h, 0) + 1
        bullets, idents = extract_promise_units(text)
        doc_units = [(b, idents_in(b)) for b in bullets]   # per-unit idents only
        doc_units += [(i, [i]) for i in idents]            # crate/module name units
        seen_u = set()
        for u, u_idents in doc_units:
            if (rel, u) in seen_u:
                continue
            seen_u.add((rel, u))
            units.append((rel, u, u_idents))

    # 3. implementation bytes
    corpus = collect_impl_corpus(repo_root)
    impl_raw = "\n".join(corpus[k] for k in sorted(corpus))
    impl_lower = impl_raw.lower()
    impl_bytes = sum(len(v.encode("utf-8", errors="replace")) for v in corpus.values())
    if not corpus:
        notes.append("no implementation bytes found (docs-only repo): "
                     "linkage ratio is 0.0 by construction")

    # 4. linkage
    linked = 0
    linked_examples, unlinked_examples = [], []
    for doc, unit, idents in units:
        via = unit_linkage(unit, idents, impl_raw, impl_lower) if corpus else None
        if via:
            linked += 1
            if len(linked_examples) < max_examples:
                linked_examples.append({"doc": doc, "unit": _clip(unit), "via": via})
        else:
            if len(unlinked_examples) < max_examples:
                unlinked_examples.append({"doc": doc, "unit": _clip(unit)})

    n_units = len(units)
    out = {
        "repo": repo_name,
        "path": repo_root,
        "promise_docs": len(promise_docs),
        "promise_units": n_units,
        "linked_units": linked,
        "linkage_ratio": round(linked / n_units, 4) if n_units else 0.0,
        "impl_files": len(corpus),
        "impl_bytes": impl_bytes,
        "unlinked_examples": unlinked_examples,
        "linked_examples": linked_examples,
        "pattern_hits": dict(sorted(pattern_hits.items())),
        "notes": notes,
    }
    if len(promise_docs) <= 64:
        out["promise_doc_paths"] = promise_docs

    if cross_check_tag:
        rx = re.compile(cross_check_tag)
        tag_files, tag_linked = [], 0
        for rel in sorted(md_paths):
            with open(os.path.join(repo_root, rel), "r", errors="replace") as f:
                if rx.search(f.read()):
                    tag_files.append(rel)
        tag_units = [u for u in units if u[0] in set(tag_files)]
        for doc, unit, idents in tag_units:
            if corpus and unit_linkage(unit, idents, impl_raw, impl_lower):
                tag_linked += 1
        out["cross_check"] = {
            "tag_regex": cross_check_tag,
            "tag_files": len(tag_files),
            "tag_files_also_promise_docs": sum(1 for f in tag_files
                                               if f in set(promise_docs)),
            "tag_units": len(tag_units),
            "tag_linked_units": tag_linked,
            "tag_linkage_ratio": round(tag_linked / len(tag_units), 4)
            if tag_units else 0.0,
        }
    return out


def census_repos(repo_roots, out_path: str = None, cross_check_tag: str = None):
    """Census a list of repos -> combined JSON (list, arg order)."""
    results = [census_repo(r, cross_check_tag=cross_check_tag) for r in repo_roots]
    payload = json.dumps(results, indent=1, sort_keys=True)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(payload + "\n")
    return results
