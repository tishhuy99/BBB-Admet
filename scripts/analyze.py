"""Fit the applicability-domain threshold and produce the two README figures.

    figures/similarity_hist.png   nearest-neighbour similarity, random vs scaffold
    figures/accuracy_by_bin.png   test accuracy within similarity bins
    artifacts/threshold.json      threshold fitted on the validation split

Run after bbb.train:  python scripts/analyze.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")

from bbb.data import load, random_split, scaffold_split          # noqa: E402
from bbb.domain import accuracy_by_bin, fit_threshold, save_threshold  # noqa: E402
from bbb.features import featurize, max_similarity, pack          # noqa: E402
from bbb.train import XGB_GRID, build, run_cell                   # noqa: E402

FIGURES = Path("figures")
SEED = 0


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    ds = load()
    X = featurize(ds.smiles)
    y = ds.y.astype(int)
    packed = pack(X)

    splits = {
        "random": random_split(len(ds), SEED),
        "scaffold": scaffold_split(ds.scaffold, SEED),
    }

    # ---- Figure 1: how far test molecules sit from training space ----
    sims = {}
    for name, (tr, _va, te) in splits.items():
        sims[name] = max_similarity(packed[te], packed[tr])
        leaked = int((ds.scaffold[te][:, None] == ds.scaffold[tr][None, :]).any(1).sum())
        print(f"{name:9s} median NN similarity {np.median(sims[name]):.3f}, "
              f"test molecules sharing a train scaffold: {leaked}")

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)
    for name, color in (("random", "#4C72B0"), ("scaffold", "#C44E52")):
        ax.hist(sims[name], bins=bins, alpha=0.6, label=name, color=color)
    ax.set_xlabel("Nearest-neighbour Tanimoto to training set")
    ax.set_ylabel("Test molecules")
    ax.set_title("Test-set distance from training space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "similarity_hist.png", dpi=150)

    # ---- Fit the threshold on VALIDATION, scaffold split ----
    tr, va, te = splits["scaffold"]
    clf, params, val_auroc, metrics = run_cell(
        "xgboost", XGB_GRID, X, y, tr, va, te, SEED
    )
    print(f"selected {params}, val AUROC {val_auroc:.3f}, test AUROC {metrics['auroc']:.3f}")

    val_sims = max_similarity(packed[va], packed[tr])
    val_proba = clf.predict_proba(X[va])[:, 1]
    threshold, val_bins = fit_threshold(val_sims, y[va], val_proba)
    overall = float(((val_proba >= 0.5).astype(int) == y[va]).mean())
    save_threshold(threshold, val_bins, overall)
    print(f"threshold fitted on validation: {threshold:.2f} "
          f"(overall val accuracy {overall:.3f})")

    # ---- Figure 2: does the threshold hold up on TEST? ----
    test_sims = max_similarity(packed[te], packed[tr])
    test_proba = clf.predict_proba(X[te])[:, 1]
    test_bins = accuracy_by_bin(test_sims, y[te], test_proba)

    labels = [f"{b['lo']:.1f}-{b['hi']:.1f}\nn={b['n']}" for b in test_bins]
    accs = [b["accuracy"] for b in test_bins]
    colors = ["#C44E52" if b["hi"] <= threshold else "#4C72B0" for b in test_bins]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(accs)), accs, color=colors)
    ax.axhline(float(((test_proba >= 0.5).astype(int) == y[te]).mean()),
               ls="--", c="k", lw=1, label="overall test accuracy")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nearest-neighbour Tanimoto to training set")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Accuracy by similarity bin (red = outside domain, < {threshold:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_by_bin.png", dpi=150)

    print("\ntest bins:")
    for b in test_bins:
        print(f"  {b['lo']:.1f}-{b['hi']:.1f}  n={b['n']:4d}  acc={b['accuracy']:.3f}")


if __name__ == "__main__":
    main()
