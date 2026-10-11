"""Bounded, read-only public SWE-bench artifact acquisition; no agent execution.

Task data is revision pinned. S3 artifacts are not immutable: every downloaded
object is SHA-256 recorded, and local copies are retained for exact replay.
Licensing of third-party patches is NOT inferred from this project's license.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import urllib.request
import urllib.error
import time
from concurrent.futures import ThreadPoolExecutor

EXCLUDED_CONFIGURATION = "20250726_mini-v1.0.0_gpt-4.1-2025-04-14"
EXCLUDED_LOG_LISTING = "https://swe-bench-submissions.s3.amazonaws.com/?list-type=2&prefix=bash-only/" + EXCLUDED_CONFIGURATION + "/logs/&max-keys=1"

EXPERIMENTS_REVISION = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"
TASK_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
TASK_SHA256 = "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
MODELS = ("gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "claude-sonnet-4-20250514", "o3-2025-04-16", "o4-mini-2025-04-16")
AGENTS = tuple("20250726_mini-v1.0.0_" + model for model in MODELS)
S3 = "https://swe-bench-submissions.s3.amazonaws.com/"
LICENSE_NOTE = "No blanket license verified for submitted artifacts; original repository and submission rights apply. Local research acquisition only; do not redistribute without review."


def normalize_record(task_id, agent_id, patch, report, *, patch_url, report_url):
    """Do not infer failed evaluation from absent labels or absent reports."""
    if report is not None and task_id in report:
        report = report[task_id]
    resolved = report.get("resolved") if isinstance(report, dict) else None
    success = resolved if isinstance(resolved, bool) else None
    # Published exception strings are real failed submissions, not model patches.
    status = "missing" if patch is None else ("completed" if patch.lstrip().startswith(("diff --git", "--- ")) else "failed")
    artifact_id = agent_id + ":" + task_id
    artifact = dict(artifact_id=artifact_id, task_id=task_id, agent_id=agent_id,
                    content=patch or "", status=status, historical_cost=None,
                    provenance={"source_url": patch_url, "license": LICENSE_NOTE})
    raw = None if patch is None else patch.encode("utf-8")
    artifact["observation"] = {
        "observed_patch_object": patch is not None,
        "byte_count": None if raw is None else len(raw),
        "format": "missing_download" if raw is None else "observed_empty" if not raw else "unified_diff" if status == "completed" else "recorded_exception",
        "content_sha256": None if raw is None else hashlib.sha256(raw).hexdigest(),
        "source_kind": "public_submission_patch_object",
    }
    truth = dict(artifact_id=artifact_id, success=success, source_url=report_url,
                 label_status="observed" if success is not None else "unavailable",
                 label_kind="published_submission_evaluation_not_rerun")
    return artifact, truth


def _download(url, path, *, max_bytes=10_000_000, missing_ok=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never silently reuse unverified partially downloaded objects.
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(max_bytes + 1)
            break
        except urllib.error.HTTPError as exc:
            if missing_ok and exc.code == 404:
                return None, {"source_url": url, "status": "http_404"}
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
            time.sleep(1 + attempt)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    if len(data) > max_bytes:
        raise ValueError("Public object exceeds bounded download size: " + url)
    path.write_bytes(data)
    return data, {"source_url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "local_path": str(path)}


def _write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def fetch_swebench_public(output_dir, task_limit=12):
    """Materialize six historical configurations × outcome-blind task sample.

    Requires optional pyarrow only for the 2.1 MB official task parquet. Selection
    uses benchmark task IDs, not successful cases or report availability. Missing
    artifacts remain explicit rows. This is a small readiness sample, not a study.
    """
    if not 3 <= task_limit <= 500:
        raise ValueError("task_limit must be between 3 and 500")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Public acquisition needs optional pyarrow (pip install pyarrow); offline replay does not") from exc
    out = Path(output_dir)
    if any((out / name).exists() for name in ("tasks.jsonl", "artifacts.jsonl", "truth.jsonl")):
        raise FileExistsError("Use a new output directory; never overwrite replay evidence")
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sources = []
    _, exclusion_source = _download(EXCLUDED_LOG_LISTING, raw / "excluded_configuration_logs.xml")
    sources.append(exclusion_source)
    task_url = f"https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/{TASK_REVISION}/data/test-00000-of-00001.parquet"
    _, source = _download(task_url, raw / "tasks.parquet")
    if source["sha256"] != TASK_SHA256:
        raise ValueError("Pinned task parquet content hash mismatch")
    sources.append(source)
    rows = pq.read_table(raw / "tasks.parquet", columns=["instance_id", "problem_statement", "repo", "base_commit"]).to_pylist()
    rows.sort(key=lambda r: hashlib.sha256(("cbp-public-readiness-v1:" + r["instance_id"]).encode()).hexdigest())
    selected = rows[:task_limit]
    tasks = [dict(task_id=r["instance_id"], task_type="code_issue_fix", question=r["problem_statement"], context="", task_group=r["instance_id"], provenance={"source_url": task_url, "repo": r["repo"], "base_commit": r["base_commit"]}) for r in selected]
    agents = []
    integrity_flags = {}
    for agent, model in zip(AGENTS, MODELS):
        url = f"https://raw.githubusercontent.com/SWE-bench/experiments/{EXPERIMENTS_REVISION}/evaluation/verified/{agent}/metadata.yaml"
        _, source = _download(url, raw / agent / "metadata.yaml")
        sources.append(source)
        flag_url = url.rsplit("/", 1)[0] + "/git_peek_suspicious_commits.md"
        flag_data, flag_source = _download(flag_url, raw / agent / "git_peek_suspicious_commits.md", missing_ok=True)
        sources.append(flag_source)
        import re
        flagged = set(re.findall(r"^([A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+)$", flag_data.decode("utf-8") if flag_data else "", flags=re.MULTILINE))
        integrity_flags[agent] = {"source_url": flag_url, "flag_document_available": flag_data is not None,
                                  "selected_flagged_task_ids": sorted(flagged.intersection(t["task_id"] for t in tasks))}
        agents.append(dict(agent_id=agent, system="mini-SWE-agent", model=model,
                           config={"version": "1.0.0", "metadata_url": url}, run=agent))
    def fetch_pair(pair):
        agent, task = pair
        prefix = f"bash-only/{agent}/logs/{task}/"
        patch_url, report_url = S3 + prefix + "patch.diff", S3 + prefix + "report.json"
        patch, ps = _download(patch_url, raw / agent / task / "patch.diff", missing_ok=True)
        report, rs = _download(report_url, raw / agent / task / "report.json", missing_ok=True)
        artifact, truth = normalize_record(task, agent, None if patch is None else patch.decode("utf-8"),
            None if report is None else json.loads(report), patch_url=patch_url, report_url=report_url)
        artifact["provenance"]["sha256"] = ps.get("sha256")
        truth["sha256"] = rs.get("sha256")
        return artifact, truth, ps, rs
    pairs = [(a, t["task_id"]) for a in AGENTS for t in tasks]
    records = list(ThreadPoolExecutor(max_workers=12).map(fetch_pair, pairs))
    artifacts, truths = [], []
    for artifact, truth, ps, rs in records:
        artifacts.append(artifact); truths.append(truth); sources.extend((ps, rs))
    _write_jsonl(out / "tasks.jsonl", tasks)
    _write_jsonl(out / "agents.jsonl", agents)
    _write_jsonl(out / "artifacts.jsonl", artifacts)
    _write_jsonl(out / "truth.jsonl", [t for t in truths if t["success"] is not None])
    _write_jsonl(out / "unavailable_labels.jsonl", [t for t in truths if t["success"] is None])
    manifest = dict(schema_version=2, task_type="code_issue_fix", dataset="SWE-bench_Verified", task_revision=TASK_REVISION,
        experiments_revision=EXPERIMENTS_REVISION, task_count=len(tasks), agent_count=len(agents), artifact_count=len(artifacts),
        excluded_configurations=[{"agent_id": EXCLUDED_CONFIGURATION, "reason": "Availability-based cohort choice: public log prefix was empty during initial acquisition, despite metadata advertising it; no success-rate selection", "source_url": EXCLUDED_LOG_LISTING}],
        task_selection="first N benchmark IDs ordered by SHA256(cbp-public-readiness-v1: + task ID); outcome blind; missing artifacts retained",
        scope="readiness sample; six configurations of one scaffold, not six independent organizations; no live generation, no fresh test execution",
        agent_identity="Public submission identifiers and reported scaffold/model/version; not verified organizational independence or complete regeneration settings",
        judge_context="issue statement and submitted patch only; no repository checkout, gold patch, test patch, hints, test names or labels",
        truth_boundary="truth.jsonl and raw evaluation/task files must never be placed in judge payloads",
        historical_cost="unknown/null per artifact; published aggregate costs are not per-artifact costs",
        integrity_flags=integrity_flags, integrity_note="Published git-peek suspicious-trajectory flags recorded without filtering the sample; absence of a flag is not a clean-run guarantee. Not evidence of uncontaminated generalization.",
        artifact_license=LICENSE_NOTE, task_license="Dataset card at pinned revision has no license field; original upstream repository rights apply",
        artifact_immutability="S3 is mutable, not revision pinned; local bytes and SHA256 in sources establish this acquisition snapshot",
        common_completed_task_count=sum(all(a["status"] == "completed" for a in artifacts if a["task_id"] == t["task_id"]) for t in tasks),
        common_observed_label_task_count=sum(all(t["success"] is not None for a, t in zip(artifacts, truths) if a["task_id"] == task["task_id"]) for task in tasks),
        status_counts={s: sum(a["status"] == s for a in artifacts) for s in ("completed", "failed", "missing")},
        label_counts={"observed": sum(t["success"] is not None for t in truths), "unavailable": sum(t["success"] is None for t in truths)}, sources=sources)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
