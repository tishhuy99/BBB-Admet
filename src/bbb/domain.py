"""Applicability domain: is a query molecule close enough to training space?

A prediction is only meaningful if the model has seen something structurally
like the query. This module reports nearest-neighbour Tanimoto similarity
against the training set and compares it to a threshold fitted on the
validation split.

The threshold is fitted on validation, never on test. Choosing a cutoff from
test-set accuracy and then reporting test-set accuracy per bin is tuning on
test, and would undercut the scaffold-split argument the project is built on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bbb.features import featurize, max_similarity, pack

ARTIFACTS = Path("artifacts")
DEFAULT_THRESHOLD = 0.30


@dataclass
class DomainChecker:
    packed_train: np.ndarray
    threshold: float

    @classmethod
    def load(cls, artifacts: Path = ARTIFACTS) -> "DomainChecker":
        packed = np.load(artifacts / "domain.npz")["packed"]
        path = artifacts / "threshold.json"
        thr = json.loads(path.read_text())["threshold"] if path.exists() else DEFAULT_THRESHOLD
        return cls(packed, float(thr))

    def similarity(self, smiles) -> np.ndarray:
        """Nearest-neighbour Tanimoto of each query against the training set."""
        return max_similarity(pack(featurize(smiles)), self.packed_train)

    def check(self, smiles) -> tuple[np.ndarray, np.ndarray]:
        """Returns (max_similarity, in_domain)."""
        sims = self.similarity(smiles)
        return sims, sims >= self.threshold


def accuracy_by_bin(sims: np.ndarray, y_true: np.ndarray, proba: np.ndarray,
                    edges=(0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01)) -> list[dict]:
    """Accuracy within nearest-neighbour similarity bins.

    This is the evidence for whatever threshold gets chosen: if accuracy is
    flat across bins, the applicability domain is not doing any work and the
    README should say so rather than shipping a decorative flag.
    """
    pred = (proba >= 0.5).astype(int)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sims >= lo) & (sims < hi)
        out.append({
            "lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()),
            "accuracy": float((pred[m] == y_true[m]).mean()) if m.any() else float("nan"),
            "mean_proba": float(proba[m].mean()) if m.any() else float("nan"),
        })
    return out

def _merge_small_bins(bins: list[dict], min_bin: int) -> list[dict]:
    """Fold under-populated bins into their neighbour above.

    Empty bins carry accuracy = nan, which would poison a weighted mean
    (nan * 0 is nan, not 0), so they are dropped before merging rather than
    averaged in.
    """
    out: list[dict] = []
    for b in (x for x in bins if x["n"] > 0):
        if out and (b["n"] < min_bin or out[-1]["n"] < min_bin):
            prev = out.pop()
            n = prev["n"] + b["n"]
            out.append({
                "lo": prev["lo"], "hi": b["hi"], "n": n,
                "accuracy": (prev["accuracy"] * prev["n"] + b["accuracy"] * b["n"]) / n,
                "mean_proba": (prev["mean_proba"] * prev["n"] + b["mean_proba"] * b["n"]) / n,
            })
        else:
            out.append(dict(b))
    return out    

def fit_threshold(sims: np.ndarray, y_true: np.ndarray, proba: np.ndarray,
                  drop: float = 0.05, min_bin: int = 20) -> tuple[float, list[dict]]:
    """Pick the cutoff where binned accuracy breaks, not where it dips.

    Walk the bins from most similar downward and stop at the first bin whose
    accuracy falls more than `drop` below the bin immediately above it. That
    bin's upper edge is the threshold. Comparing against the adjacent bin
    rather than the overall mean finds the actual discontinuity: a run of
    flat, slightly-below-average bins is not a domain boundary.

    Bins with fewer than `min_bin` members are merged upward before the walk,
    so a five-molecule tail cannot set the threshold.
    """
    bins = _merge_small_bins(accuracy_by_bin(sims, y_true, proba), min_bin)

    for upper, lower in zip(bins[::-1], bins[-2::-1]):
        if lower["accuracy"] < upper["accuracy"] - drop:
            return float(lower["hi"]), bins
    return float(bins[0]["lo"]), bins


def save_threshold(threshold: float, bins: list[dict], overall_accuracy: float,
                   artifacts: Path = ARTIFACTS) -> None:
    artifacts.mkdir(exist_ok=True)
    (artifacts / "threshold.json").write_text(json.dumps({
        "threshold": threshold,
        "fitted_on": "validation",
        "overall_validation_accuracy": overall_accuracy,
        "bins": bins,
    }, indent=2))
