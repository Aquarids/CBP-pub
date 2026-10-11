"""Outcome-blind, revision-pinned source context for recorded replay datasets.

This module deliberately does not call ``load_dataset``: evaluation labels, raw
benchmark rows, reports, gold patches and tests are never parsed. Only normalized
issues/submissions and public source at each issue's base revision drive selection.
Whole AST units are admitted or explicitly omitted; no unit is silently clipped.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import urllib.error
import urllib.request

from .schema import read_jsonl, write_jsonl

POLICY_VERSION = "pinned-source-context-v2"
DEFAULT_BUDGET = 45_000
MAX_SOURCE_BYTES = 2_000_000
MAX_DEPENDENCY_FILES = 24
CONFIG_FILES = ("pyproject.toml", "setup.py", "setup.cfg")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _safe_path(path):
    p = PurePosixPath(path)
    return bool(path) and not p.is_absolute() and ".." not in p.parts and not any(c in path for c in "\\\x00\n\r")


def _source_path(path):
    p = PurePosixPath(path)
    return _safe_path(path) and not any(x in {"test", "tests", "testing", "__pycache__"} for x in p.parts) and not p.name.startswith("test_") and (path.endswith(".py") or p.name in CONFIG_FILES)


def _download(url):
    try:
        with urllib.request.urlopen(url, timeout=40) as response:
            data = response.read(MAX_SOURCE_BYTES + 1)
        if len(data) > MAX_SOURCE_BYTES:
            raise ValueError("Pinned source exceeds download bound")
        return data
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def patch_changes(patch):
    """Base-side changed line positions, across every submitted patch.

    New files, submitted test files and documentation are not source retrieval
    targets. Added lines are anchored on both adjacent base-side lines.
    """
    result, excluded = {}, set()
    path, old_line, in_hunk = None, 0, False
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            path, in_hunk = None, False
        elif line.startswith("--- "):
            raw = line[4:].split("\t", 1)[0]
            path = raw[2:] if raw.startswith("a/") else None
            if path is not None and not _source_path(path):
                excluded.add(path)
                path = None
            if path is not None:
                result.setdefault(path, set())
            in_hunk = False
        elif line.startswith("@@ "):
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
            if match:
                old_line, in_hunk = int(match.group(1)), True
        elif path and in_hunk:
            if line.startswith("-"):
                result[path].add(max(1, old_line))
                old_line += 1
            elif line.startswith("+"):
                result[path].update((max(1, old_line - 1), max(1, old_line)))
            elif line.startswith(" "):
                old_line += 1
    return result, excluded


def artifact_observation(artifact, source_dir):
    """Acquisition facts come only from a retained public patch object and hash."""
    agent, task = artifact["agent_id"], artifact["task_id"]
    if not all(_safe_path(x) and "/" not in x for x in (agent, task)):
        raise ValueError("Unsafe artifact identity")
    path = Path(source_dir) / "raw" / agent / task / "patch.diff"
    record = dict(observed_patch_object=False, byte_count=None,
                  format="missing_download", content_sha256=None,
                  source_kind="public_submission_patch_object")
    if not path.is_file():
        return record
    data = path.read_bytes()
    sha = _sha(data)
    expected = artifact.get("provenance", {}).get("sha256")
    if expected is not None and sha != expected:
        raise ValueError("Retained patch object hash mismatch: " + artifact["artifact_id"])
    decoded = data.decode("utf-8")
    if decoded != artifact["content"]:
        raise ValueError("Retained patch object differs from normalized content: " + artifact["artifact_id"])
    kind = "observed_empty" if len(data) == 0 else ("unified_diff" if decoded.lstrip().startswith(("diff --git", "--- ")) else "recorded_exception")
    return dict(record, observed_patch_object=True, byte_count=len(data), format=kind, content_sha256=sha)


class SourceCache:
    def __init__(self, root, downloader, seed_manifest=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.downloader = downloader
        self.seeds = {}
        if seed_manifest is not None:
            for item in json.loads(Path(seed_manifest).read_text()):
                self.seeds[item["url"]] = Path(item["path"])

    def get(self, repo, commit, path):
        if not _source_path(path):
            raise ValueError("Not an allowed upstream source path: " + path)
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        key = _sha(url.encode())
        data_path, meta_path = self.root / (key + ".source"), self.root / (key + ".json")
        origin = "download"
        if data_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            data = data_path.read_bytes()
            if meta["url"] != url or meta["sha256"] != _sha(data):
                raise ValueError("Pinned source cache integrity mismatch")
            origin = "verified_cache"
        elif url in self.seeds and self.seeds[url].is_file():
            data = self.seeds[url].read_bytes()
            origin = "pinned_seed_manifest"
        else:
            try:
                data = self.downloader(url)
            except Exception as exc:
                return None, dict(path=path, url=url, status="download_error", error=type(exc).__name__ + ": " + str(exc), sha256=None, bytes=None)
        if data is None:
            return None, dict(path=path, url=url, status="http_404", sha256=None, bytes=None)
        if not isinstance(data, bytes) or len(data) > MAX_SOURCE_BYTES:
            raise ValueError("Downloader must return bounded bytes or None")
        record = dict(path=path, url=url, status="ok", sha256=_sha(data), bytes=len(data), acquisition=origin, cache_path=str(data_path))
        data_path.write_bytes(data)
        meta_path.write_text(json.dumps({"url": url, "sha256": record["sha256"]}))
        try:
            return data.decode("utf-8-sig"), record
        except UnicodeDecodeError:
            return None, dict(record, status="decode_error")


class Module:
    def __init__(self, path, text):
        self.path, self.text, self.lines = path, text, text.splitlines(keepends=True)
        self.tree, self.parse_error = None, None
        self.defs, self.parents = [], {}
        if path.endswith(".py"):
            try:
                self.tree = ast.parse(text)
            except SyntaxError as exc:
                self.parse_error = f"{exc.msg} at line {exc.lineno}"
            if self.tree is not None:
                for node in ast.walk(self.tree):
                    for child in ast.iter_child_nodes(node):
                        self.parents[child] = node
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        self.defs.append(node)

    def span(self, node):
        start = min([node.lineno] + [x.lineno for x in getattr(node, "decorator_list", [])])
        return start, node.end_lineno

    def name(self, node):
        parts = [getattr(node, "name", type(node).__name__)]
        parent = self.parents.get(node)
        while parent:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                parts.insert(0, parent.name)
            parent = self.parents.get(parent)
        return ".".join(parts)

    def source(self, start, end):
        return "".join(self.lines[start - 1:end])

    def symbols(self, node):
        return {x.id for x in ast.walk(node) if isinstance(x, ast.Name)} | {x.attr for x in ast.walk(node) if isinstance(x, ast.Attribute)}

    def changed_units(self, lines):
        selected = []
        for line in sorted(lines):
            callables = [n for n in self.defs if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and self.span(n)[0] <= line <= n.end_lineno]
            # Preserve the enclosing callable, including nested helper closures.
            if callables:
                selected.append(min(callables, key=lambda n: n.lineno))
                continue
            stmts = [n for n in ast.walk(self.tree) if isinstance(n, ast.stmt) and not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.lineno <= line <= n.end_lineno]
            if stmts:
                selected.append(min(stmts, key=lambda n: n.end_lineno - n.lineno))
        return list(dict.fromkeys(selected))

    def imports(self):
        if self.tree is None:
            return []
        # Include conditional/module-level imports; local imports already occur in
        # their complete callables and are also useful for dependency discovery.
        return [n for n in ast.walk(self.tree) if isinstance(n, (ast.Import, ast.ImportFrom))]

    def dependency_paths(self, names, package_roots):
        found = {}
        for node in self.imports():
            aliases = {a.asname or a.name.split(".")[0] for a in node.names}
            relevant = bool(aliases & names)
            modules = []
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = self.path.removesuffix(".py").split("/")[:-1]
                    if parts and parts[0] == "src":
                        parts = parts[1:]
                    base = ".".join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
                modules = [base] if base else []
            else:
                modules = [a.name for a in node.names]
            for module in modules:
                root = module.split(".")[0]
                if root not in package_roots:
                    continue
                prefix = package_roots[root]
                p = prefix + module.replace(".", "/")
                # Module form first; try package form only after a 404.
                found[p + ".py"] = min(found.get(p + ".py", 99), 0 if relevant else 1)
        return found


def _issue_names(question):
    # Named API references in examples/backticks/tracebacks are not labels.
    return set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", question)) | set(re.findall(r"`([A-Za-z_]\w*)`", question)) | set(re.findall(r"\.([A-Za-z_]\w*)", question))


def _build_context(task, artifacts, cache, budget):
    repo, commit = task.get("provenance", {}).get("repo"), task.get("provenance", {}).get("base_commit")
    if not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("Source context requires repository and exact 40-character base commit")
    changes, excluded = {}, set()
    for artifact in artifacts:
        if artifact["task_id"] != task["task_id"]:
            continue
        changed, skipped = patch_changes(artifact["content"])
        excluded.update(skipped)
        for path, lines in changed.items():
            changes.setdefault(path, set()).update(lines)
    modules, sources = {}, []
    def acquire(path):
        text, info = cache.get(repo, commit, path)
        sources.append(info)
        if text is not None:
            modules[path] = Module(path, text)
        return text
    for path in sorted(changes):
        acquire(path)
    names = _issue_names(task["question"])
    keyword_names = set(re.findall(r"\b([A-Za-z_]\w*)\s*=", task["question"]))
    # Operators in examples identify the associated protocol/printer entry points.
    if "*" in task["question"]:
        names.update(("__mul__", "__rmul__", "_print_Mul"))
    if re.search(r"parenthes", task["question"], re.I):
        names.update(("parenthesize", "precedence", "_needs_mul_brackets"))
    package_roots = {}
    for path in changes:
        parts = path.split("/")
        if len(parts) > 1:
            root = parts[1] if parts[0] == "src" and len(parts) > 2 else parts[0]
            package_roots[root] = "src/" if parts[0] == "src" else ""
    # Imports used by changed/issue-named units get first dependency priority.
    seed_nodes, used_names = {}, set(names)
    for path, module in modules.items():
        if module.tree is None:
            continue
        seeds = module.changed_units(changes.get(path, set()))
        seeds += [n for n in module.defs if n.name in names and (not isinstance(n, ast.ClassDef) or len(module.source(*module.span(n))) <= 6000)]
        seed_nodes[path] = list(dict.fromkeys(seeds))
        for node in seeds:
            used_names.update(module.symbols(node))
    dependency_candidates = {}
    for module in list(modules.values()):
        for path, priority in module.dependency_paths(used_names, package_roots).items():
            if path not in modules and _source_path(path):
                dependency_candidates[path] = min(dependency_candidates.get(path, 99), priority)
    # Public API imports appearing only in issue examples also identify source.
    issue_module_paths = set()
    for module_name in re.findall(r"\bfrom\s+([A-Za-z_][\w.]*)\s+import\s+", task["question"]):
        root = module_name.split(".")[0]
        if root in package_roots:
            path = package_roots[root] + module_name.replace(".", "/") + ".py"
            if _source_path(path) and path not in modules:
                dependency_candidates[path] = -2
                issue_module_paths.add(path)
    # Question source paths, normalized from Windows/Unix tracebacks, stay pinned
    # to the task base rather than following issue URLs to another revision.
    normalized_question = task["question"].replace("\\", "/")
    for root, prefix in package_roots.items():
        for path in re.findall(r"\b(" + re.escape(root) + r"/[A-Za-z0-9_./-]+\.py)\b", normalized_question):
            if _source_path(prefix + path) and prefix + path not in modules:
                dependency_candidates[prefix + path] = -1
    planned = sorted(dependency_candidates, key=lambda p: (dependency_candidates[p], p))
    omitted_files = [dict(path=p, reason="uniform_dependency_file_limit") for p in planned[MAX_DEPENDENCY_FILES:]]
    for path in planned[:MAX_DEPENDENCY_FILES]:
        if acquire(path) is None and sources[-1]["status"] == "http_404":
            acquire(path[:-3] + "/__init__.py")
    # Resolve one package re-export hop for issue APIs, within a separate uniform
    # bound. This covers, e.g., a class imported from a package __init__ module.
    reexports = set()
    for path in issue_module_paths:
        module = modules.get(path) or modules.get(path[:-3] + "/__init__.py")
        if module:
            reexports.update(p for p, priority in module.dependency_paths(names, package_roots).items()
                             if priority == 0 and p not in modules and _source_path(p))
    for path in sorted(reexports)[:8]:
        if acquire(path) is None and sources[-1]["status"] == "http_404":
            acquire(path[:-3] + "/__init__.py")
    omitted_files.extend(dict(path=p, reason="uniform_issue_reexport_file_limit") for p in sorted(reexports)[8:])
    for path in CONFIG_FILES:
        if path not in modules:
            acquire(path)
    units = {}
    def add(module, start, end, name, reason, priority):
        key = (module.path, start, end)
        text = module.source(start, end)
        if not text:
            return
        if key in units:
            units[key]["reasons"].add(reason)
            units[key]["priority"] = min(units[key]["priority"], priority)
        else:
            units[key] = dict(path=module.path, start=start, end=end, name=name,
                              text=text, reasons={reason}, priority=priority)
    def add_node(module, node, reason, priority):
        add(module, *module.span(node), module.name(node), reason, priority)
    selected_symbols = set(used_names)
    for path, module in modules.items():
        if module.tree is None:
            if path in changes or path in CONFIG_FILES:
                add(module, 1, len(module.lines), "complete file", "changed_non_python_or_dependency_config", 0 if path in changes else 32)
            continue
        primary = path in changes
        for node in module.imports():
            parent = module.parents.get(node)
            inside_function = False
            while parent:
                inside_function |= isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                parent = module.parents.get(parent)
            if not inside_function:
                add_node(module, node, "module_import", 4 if primary else 28)
        seeds = module.changed_units(changes.get(path, set())) if primary else []
        for node in seeds:
            add_node(module, node, "changed_enclosing_unit", 0)
        issue_nodes = [n for n in module.defs if primary and (n.name in names or (n.name in keyword_names and any(isinstance(d, ast.Name) and d.id == "property" for d in getattr(n, "decorator_list", []))))]
        # Small named classes (e.g. views) are complete; large classes contribute
        # constructors plus their separately selected methods, never a cut prefix.
        for node in issue_nodes:
            start, end = module.span(node)
            if isinstance(node, ast.ClassDef) and len(module.source(start, end)) > 6000:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in {"__init__", "__new__"}:
                        add_node(module, child, "named_class_constructor", 22)
            else:
                add_node(module, node, "issue_named_api", 8)
        seeds += [n for n in issue_nodes if not isinstance(n, ast.ClassDef) or len(module.source(*module.span(n))) <= 6000]
        called = set()
        for node in seeds:
            called.update(module.symbols(node))
            if isinstance(node, ast.ClassDef):
                selected_symbols.update(x.name for x in node.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)))
        selected_symbols.update(called)
        for node in module.defs:
            if isinstance(node, ast.ClassDef):
                continue
            if node.name in called:
                add_node(module, node, "direct_definition", 14)
            elif seeds and module.symbols(node) & ({getattr(n, "name", "") for n in seeds} | {x for x in called if x.startswith("_") and not x.startswith("__")}):
                add_node(module, node, "direct_caller", 18 if any(term[:5] in node.name.lower() for term in re.findall(r"[a-z]{4,}", task["question"].splitlines()[0].lower())) else 24)
        if path in CONFIG_FILES:
            add(module, 1, len(module.lines), "complete dependency configuration", "dependency_config", 30)
    # One-hop definitions in fetched dependency modules, including small modules
    # whose constants and helper dispatch tables explain the imported behavior.
    for path, module in modules.items():
        if path in changes or path in CONFIG_FILES or module.tree is None:
            continue
        if len(module.text) <= 8000:
            add(module, 1, len(module.lines), "complete dependency module", "direct_dependency_module", 12 if Path(path).stem in used_names or Path(path).stem[:4] in task["question"].lower() else 22)
        for node in module.defs:
            enclosing_class = module.parents.get(node)
            while enclosing_class is not None and not isinstance(enclosing_class, ast.ClassDef):
                enclosing_class = module.parents.get(enclosing_class)
            class_referenced = enclosing_class is None or enclosing_class.name in selected_symbols
            if node.name in selected_symbols and not isinstance(node, ast.ClassDef) and class_referenced:
                add_node(module, node, "imported_definition", 16)
            elif isinstance(node, ast.ClassDef) and node.name in selected_symbols:
                if len(module.source(*module.span(node))) <= 5000:
                    add_node(module, node, "imported_class", 17)
                else:
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in selected_symbols:
                            add_node(module, child, "imported_class_method", 17)
        for node in module.tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and module.symbols(node) & selected_symbols:
                add_node(module, node, "dependency_constant", 16)
    header = (f"Pinned upstream source context ({POLICY_VERSION})\nRepository: {repo}\nBase commit: {commit}\n"
              "Only complete source units are shown. Source selection uses the issue and union of all submitted patches; no benchmark truth, gold patches, or hidden tests.\n"
              "This is bounded evidence, not a complete repository. Missing or omitted units are recorded in the context manifest.\n")
    footer_reserve = 300
    remaining = budget - len(header) - footer_reserve
    if remaining <= 0:
        raise ValueError("Context budget is too small")
    included, omitted, chunks = [], [], []
    for unit in sorted(units.values(), key=lambda x: (x["priority"], x["path"], x["start"], -x["end"])):
        desc = {k: v for k, v in unit.items() if k not in {"text", "reasons"}}
        desc["reasons"] = sorted(unit["reasons"])
        if any(x["path"] == unit["path"] and x["start"] <= unit["start"] and x["end"] >= unit["end"] for x in included):
            continue
        chunk = f"\n--- {unit['path']}:{unit['start']}-{unit['end']} | {unit['name']} ---\n" + unit["text"]
        if len(chunk) > remaining:
            omitted.append(dict(desc, reason="whole_unit_exceeds_remaining_budget", characters=len(chunk)))
            continue
        included.append(desc)
        chunks.append(chunk)
        remaining -= len(chunk)
    # An enclosing unit admitted later can make an earlier omission covered.
    omitted = [x for x in omitted if not any(y["path"] == x["path"] and y["start"] <= x["start"] and y["end"] >= x["end"] for y in included)]
    footer = (f"\nContext coverage: {len(included)} complete units; {len(omitted)} candidate units omitted for budget; "
              f"{sum(s['status'] != 'ok' for s in sources)} unavailable source requests; {len(omitted_files)} dependency files omitted by cap.\n")
    context = header + "".join(chunks) + footer
    assert len(context) <= budget
    missing_changed = [p for p in changes if p not in modules]
    record = dict(task_id=task["task_id"], repo=repo, base_commit=commit,
                  character_budget=budget, context_characters=len(context), context_sha256=_sha(context.encode()),
                  changed_source_files=sorted(changes), excluded_submission_paths=sorted(excluded), sources=sources,
                  included_units=included, omitted_units=omitted, omitted_files=omitted_files,
                  missing_changed_files=missing_changed,
                  uncovered_changed_lines={p: [line for line in sorted(lines)
                      if p not in modules or not any(unit["path"] == p and unit["start"] <= line <= unit["end"] for unit in included)]
                      for p, lines in changes.items()},
                  changed_unit_omissions=[x for x in omitted if "changed_enclosing_unit" in x["reasons"]],
                  syntax_parse_failures=[dict(path=p, error=m.parse_error) for p, m in modules.items() if m.parse_error],
                  truncation="whole_unit_omission_only; no snippet prefix truncation")
    return dict(task, context=context), record


def enrich_dataset(source_dir, output_dir, *, budget=DEFAULT_BUDGET, downloader=None,
                   cache_dir=None, seed_manifest=None, workers=6):
    """Create a new enriched dataset; preserve ordering and copy truth opaquely.

    ``downloader(url) -> bytes | None`` is injectable for fully offline tests.
    The only raw acquisition object inspected is each public ``patch.diff``.
    Existing status/label fields never determine an acquisition observation.
    """
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; never overwrite replay evidence")
    if output == source or source in output.parents:
        raise ValueError("Output must not be inside baseline dataset")
    tasks = read_jsonl(source / "tasks.jsonl")
    agents = read_jsonl(source / "agents.jsonl")
    artifacts = read_jsonl(source / "artifacts.jsonl")
    if not tasks or not agents or not artifacts:
        raise ValueError("Context enrichment requires a nonempty recorded dataset")
    enriched_artifacts = [dict(a, observation=artifact_observation(a, source)) for a in artifacts]
    cache = SourceCache(cache_dir or output.parent / ".pinned-source-cache", downloader or _download, seed_manifest)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda task: _build_context(task, artifacts, cache, budget), tasks))
    output.mkdir(parents=True)
    write_jsonl(output / "tasks.jsonl", [t for t, _ in results])
    write_jsonl(output / "agents.jsonl", agents)
    write_jsonl(output / "artifacts.jsonl", enriched_artifacts)
    for filename in ("truth.jsonl", "unavailable_labels.jsonl"):
        if (source / filename).is_file():
            shutil.copyfile(source / filename, output / filename)
    for artifact in enriched_artifacts:
        if artifact["observation"]["observed_patch_object"]:
            relative = Path("raw") / artifact["agent_id"] / artifact["task_id"] / "patch.diff"
            (output / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, output / relative)
    manifest = dict(policy_version=POLICY_VERSION, baseline_directory=str(source),
                    task_count=len(tasks), agent_count=len(agents), artifact_count=len(artifacts),
                    character_budget=budget, dependency_file_limit=MAX_DEPENDENCY_FILES, issue_reexport_file_limit=8,
                    selection="All baseline tasks in unchanged order; union of all submitted existing source changes; AST enclosing callables; issue-named APIs; one-hop definitions/callers/imports; dependency configuration",
                    truth_boundary="Truth and unavailable-label files copied opaquely, never parsed; raw benchmark parquet, evaluation reports, tests and solution patches neither read nor copied",
                    observation_boundary="Only retained public patch object bytes, verified against normalized content and any provenance SHA256; no report or artifact status inference",
                    tasks=[record for _, record in results])
    (output / "context_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({k: v for k, v in manifest.items() if k != "tasks"}, indent=2) + "\n")
    return manifest
