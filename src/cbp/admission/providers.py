"""Provider-neutral boundary: humans, programs and models return the same artifact.

No provider is invoked over the network by this module. Human adapters may pause
for input; model adapters may wrap the existing optional aq-llm-pivot interface.
"""
from dataclasses import dataclass
from typing import Literal, Protocol
from .protocol import finite


@dataclass(frozen=True)
class Artifact:
    text: str
    source: str
    resource_units: float
    elapsed_seconds: float | None = None
    tokens: int | None = None
    monetary_cost: float | None = None
    currency: str | None = None

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.source:
            raise ValueError("Artifact text and provenance required")
        finite(self.resource_units, "resource_units")
        for name in ("elapsed_seconds", "monetary_cost"):
            if getattr(self, name) is not None:
                finite(getattr(self, name), name)
        if self.monetary_cost is not None and (not isinstance(self.currency, str) or not self.currency.strip()):
            raise ValueError("Known monetary cost requires an explicit currency")
        if self.tokens is not None and (isinstance(self.tokens, bool) or not isinstance(self.tokens, int) or self.tokens < 0):
            raise ValueError("tokens must be a nonnegative integer or unknown (None)")


class Provider(Protocol):
    provider_id: str
    configuration_id: str
    kind: Literal["human", "program", "model"]

    def execute(self, prompt: str) -> Artifact: ...


class ProgramReviewer:
    """Wrap a local score callable without granting it controller state."""
    def __init__(self, reviewer_id, score):
        self.reviewer_id, self.score = reviewer_id, score

    def review(self, task, artifact):
        from .protocol import Review
        return Review(self.reviewer_id, self.score(task, artifact), "program")


class FileReviewer:
    """Human/model-neutral request-response transport; missing responses stay pending."""
    def __init__(self, directory, reviewer_id):
        from pathlib import Path
        self.directory, self.reviewer_id = Path(directory), reviewer_id
        self.directory.mkdir(parents=True, exist_ok=True)

    def submit(self, review_id, task, artifact):
        import json
        import hashlib
        payload = {"review_id": review_id, "reviewer_id": self.reviewer_id, "task": task, "artifact": artifact}
        encoded = json.dumps(payload, sort_keys=True)
        token = hashlib.sha256((self.reviewer_id + ":" + review_id).encode()).hexdigest()
        path = self.directory / (token + ".request.json")
        if path.exists() and path.read_text() != encoded:
            raise ValueError("Cannot mutate an existing request")
        path.write_text(encoded)
        return token

    def poll(self, ticket):
        import json
        import re
        from .protocol import Review
        if not re.fullmatch(r"[a-f0-9]{64}", ticket):
            raise ValueError("Invalid ticket")
        request = json.loads((self.directory / (ticket + ".request.json")).read_text())
        path = self.directory / (ticket + ".response.json")
        if not path.exists():
            return None
        response = json.loads(path.read_text())
        if response.get("review_id") != request["review_id"] or response.get("reviewer_id") != self.reviewer_id:
            raise ValueError("Response identity does not match request")
        if response.get("source") not in {"human", "program", "model", "fixture"}:
            raise ValueError("Explicit provider provenance required")
        return Review(self.reviewer_id, response["score"], response["source"])
