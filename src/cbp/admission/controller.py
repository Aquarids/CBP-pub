"""Single pending trial boundary shared by synchronous and human/file reviewers."""
from dataclasses import dataclass, field
from .protocol import AdmissionState, Cell, Review


@dataclass
class Trial:
    cell: Cell
    probe_id: str
    artifact_id: str
    committee: tuple[str, ...]
    reviews: dict[str, Review] = field(default_factory=dict)
    status: str = "pending"


class ReviewController:
    def __init__(self, state: AdmissionState):
        self.state = state
        self.trials = {}
        self.blocked_cells = set()

    def begin(self, cell, probe_id, artifact_id, committee):
        key = (cell, probe_id)
        if key in self.trials:
            old = self.trials[key]
            if old.artifact_id != artifact_id or old.committee != tuple(committee):
                raise ValueError("A resumed trial cannot change artifact or committee")
            return old
        if cell not in self.state.cells or cell in self.blocked_cells:
            raise ValueError("Unknown or terminal cell")
        if any(t.cell == cell and t.status == "pending" for t in self.trials.values()):
            raise ValueError("Cannot bypass a missing review by starting a fresh trial")
        if not artifact_id or not probe_id:
            raise ValueError("Artifact and probe identity required")
        if (len(committee) != self.state.config.committee_size or len(set(committee)) != len(committee)
                or cell.provider_id in committee or not set(committee) <= self.state.reviewer_ids):
            raise ValueError("Invalid fixed committee")
        if self.state.summary(cell)["n"] >= self.state.config.max_probes_per_cell:
            raise ValueError("Probe resource cap exhausted")
        trial = Trial(cell, probe_id, artifact_id, tuple(committee))
        self.trials[key] = trial
        return trial

    def submit(self, cell, probe_id, artifact_id, review):
        trial = self.trials[(cell, probe_id)]
        if artifact_id != trial.artifact_id or review.reviewer_id not in trial.committee:
            raise ValueError("Review does not match its assigned artifact/committee")
        old = trial.reviews.get(review.reviewer_id)
        if old is not None:
            if old != review:
                raise ValueError("A committed review cannot be changed")
            return trial.status
        if trial.status != "pending":
            raise ValueError("Trial is terminal")
        trial.reviews[review.reviewer_id] = review
        if len(trial.reviews) == len(trial.committee):
            self.state.observe(cell, probe_id, list(trial.reviews.values()))
            trial.status = "completed"
        return trial.status

    def stop(self, cell, probe_id, reason="timeout"):
        if reason not in {"timeout", "withdrawn", "assumption_unavailable"}:
            raise ValueError("Unknown stopping reason")
        trial = self.trials[(cell, probe_id)]
        if trial.status != "pending":
            raise ValueError("Only a pending trial may stop")
        trial.status = "inconclusive" if reason == "timeout" else reason
        self.blocked_cells.add(cell)
        self.state._terminal[cell] = trial.status
        return trial.status
