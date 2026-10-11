"""Normalized recorded artifacts with separate, evaluation-only benchmark truth."""
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected object")
        rows.append(row)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows))


def index(rows, key):
    result = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip() or value in result:
            raise ValueError(f"Missing or duplicate {key}: {value!r}")
        result[value] = row
    return result


def text_field(row, key):
    if not isinstance(row.get(key), str):
        raise ValueError(f"{key} must be text")


def reject_credentials(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if any(word in key.lower() for word in ("api_key", "password", "credential", "access_token", "secret")):
                raise ValueError("Credentials must not appear in normalized public agent configuration")
            reject_credentials(child)
    elif isinstance(value, list):
        for child in value:
            reject_credentials(child)


def load_dataset(directory):
    root = Path(directory)
    tasks = index(read_jsonl(root / "tasks.jsonl"), "task_id")
    agents = index(read_jsonl(root / "agents.jsonl"), "agent_id")
    artifacts = index(read_jsonl(root / "artifacts.jsonl"), "artifact_id")
    if not tasks or not agents or not artifacts:
        raise ValueError("Tasks, agents, and artifacts must be nonempty")
    task_types = {task.get("task_type") for task in tasks.values() if isinstance(task.get("task_type"), str) and task["task_type"].strip()}
    if len(task_types) != 1 or any(not isinstance(task.get("task_type"), str) or task.get("task_type") not in task_types for task in tasks.values()):
        raise ValueError("A replay market requires exactly one explicit task_type; mixed or missing categories are rejected")
    task_type = next(iter(task_types))
    for task in tasks.values():
        for key in ("question", "context"):
            text_field(task, key)
        task.setdefault("task_group", task["task_id"])
        if not isinstance(task["task_group"], str) or not task["task_group"]:
            raise ValueError("task_group must be a nonempty string")
    for agent in agents.values():
        for key in ("system", "model", "run"):
            text_field(agent, key)
        if not isinstance(agent.get("config"), dict):
            raise ValueError("Agent config must be an object")
        reject_credentials(agent["config"])
    pairs = {}
    for artifact in artifacts.values():
        if artifact.get("task_id") not in tasks or artifact.get("agent_id") not in agents:
            raise ValueError("Unknown artifact task/agent")
        pair = (artifact["agent_id"], artifact["task_id"])
        if pair in pairs:
            raise ValueError("One fixed artifact per agent-task required; encode distinct runs as agents")
        pairs[pair] = artifact
        text_field(artifact, "content")
        if artifact.get("status") not in {"completed", "failed", "missing"}:
            raise ValueError("Artifact status must explicitly include failure/missing outcomes")
        provenance = artifact.get("provenance")
        if not isinstance(provenance, dict) or not all(isinstance(provenance.get(k), str) and provenance[k] for k in ("source_url", "license")):
            raise ValueError("Artifact source_url and license are required")
        cost = artifact.get("historical_cost")
        if cost is not None and (isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0):
            raise ValueError("Historical cost must be nonnegative finite or null")
    coverage = {aid: sum((aid, tid) in pairs for tid in tasks) for aid in agents}
    missing = [(aid, tid) for aid in agents for tid in tasks if (aid, tid) not in pairs]
    # Reject, rather than silently intersect or turn unavailable records into failures.
    if missing:
        raise ValueError(f"Incomplete task intersection: {len(missing)} absent agent-task rows; add explicit missing records, first={missing[0]}")
    truth_path = root / "truth.jsonl"
    truth = index(read_jsonl(truth_path), "artifact_id") if truth_path.exists() else {}
    for aid, label in truth.items():
        if aid not in artifacts or type(label.get("success")) is not bool:
            raise ValueError("Truth requires known artifact_id and strict boolean success")
        if not isinstance(label.get("source_url"), str) or not label["source_url"]:
            raise ValueError("Truth requires independent source_url provenance")
    body = {"task_type": task_type, "tasks": tasks, "agents": agents, "artifacts": artifacts, "truth": truth}
    return {**body, "pairs": pairs, "coverage": coverage, "fingerprint": digest(body)}


def split_tasks(tasks, seed, calibration_fraction=.2, heldout_fraction=.2):
    if not 0 < calibration_fraction < 1 or not 0 < heldout_fraction < 1 or calibration_fraction + heldout_fraction >= 1:
        raise ValueError("Require positive calibration/heldout fractions with sum < 1")
    groups = sorted({t["task_group"] for t in tasks.values()}, key=lambda g: digest([seed, "split", g]))
    if len(groups) < 3:
        raise ValueError("At least three independent task groups required")
    ncal = max(1, int(len(groups)*calibration_fraction))
    nheld = max(1, int(len(groups)*heldout_fraction))
    allocation = {g: "calibration" if i < ncal else "heldout" if i < ncal+nheld else "probe" for i, g in enumerate(groups)}
    return {tid: allocation[t["task_group"]] for tid, t in tasks.items()}
