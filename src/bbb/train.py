"""Train and evaluate across both split types and five seeds.

Protocol: for each (model, split, seed) the hyperparameter grid is scored on
the validation subset and the winner is evaluated once on test. The test set is
touched a single time per cell, which is the whole point of carrying a
validation split through bbb.data.

Outputs:
    results.csv          one row per model x split x seed
    artifacts/model.pkl  the deployable model (scaffold split, seed 0)
    artifacts/domain.npz packed training fingerprints for the AD check
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from bbb.data import load, random_split, scaffold_split
from bbb.features import featurize, pack

SEEDS = range(5)
ARTIFACTS = Path("artifacts")

# Five combos each, hand-chosen. No Optuna: on 1,365 training molecules the
# search space is small enough that a tuner would mostly fit validation noise.
XGB_GRID = [
    {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.10, "subsample": 0.8},
    {"max_depth": 4, "n_estimators": 400, "learning_rate": 0.05, "subsample": 0.8},
    {"max_depth": 6, "n_estimators": 300, "learning_rate": 0.05, "subsample": 0.8},
    {"max_depth": 6, "n_estimators": 600, "learning_rate": 0.03, "subsample": 0.6},
    {"max_depth": 8, "n_estimators": 400, "learning_rate": 0.05, "subsample": 0.6},
    {"max_depth": 10, "n_estimators": 400, "learning_rate": 0.05, "subsample": 0.6},
]
LR_GRID = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0, 100.0)]


def build(model_name: str, params: dict, seed: int):
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed, **params
        )
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=seed,
        **params,
    )


def score(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """AUROC plus AUPRC on both classes.

    At ~76% positive, positive-class AUPRC has a no-skill baseline of 0.76 and
    flatters every model. Minority-class AUPRC (baseline ~0.24) is the number
    that actually moves.
    """
    return {
        "auroc": roc_auc_score(y_true, proba),
        "auprc_pos": average_precision_score(y_true, proba),
        "auprc_neg": average_precision_score(1 - y_true, 1 - proba),
    }


def run_cell(model_name, grid, X, y, tr, va, te, seed):
    """Select hyperparameters on validation, evaluate once on test."""
    best, best_val, best_params = None, -np.inf, grid[0]
    for params in grid:
        clf = build(model_name, params, seed).fit(X[tr], y[tr])
        val_auroc = roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1])
        # Grids are ordered simplest-first. With ~195 validation molecules,
        # AUROC differences under ~0.01 are noise, so a more complex model has
        # to clear the incumbent by that margin to displace it.
        if best is None or val_auroc > best_val + 0.01:
            best, best_val, best_params = clf, val_auroc, params

    metrics = score(y[te], best.predict_proba(X[te])[:, 1])
    return best, best_params, best_val, metrics


def main(out: str = "results.csv"):
    ds = load()
    print(ds.report)

    X = featurize(ds.smiles)
    y = ds.y.astype(int)
    print(f"features {X.shape}, {X.sum(1).mean():.1f} bits set on average")

    splitters = {
        "random": lambda s: random_split(len(ds), s),
        "scaffold": lambda s: scaffold_split(ds.scaffold, s),
    }
    models = {"logreg": LR_GRID, "xgboost": XGB_GRID}

    rows, keep = [], None
    for split_name, splitter in splitters.items():
        for model_name, grid in models.items():
            for seed in SEEDS:
                tr, va, te = splitter(seed)
                clf, params, val_auroc, m = run_cell(
                    model_name, grid, X, y, tr, va, te, seed
                )
                rows.append({
                    "model": model_name, "split": split_name, "seed": seed,
                    "val_auroc": round(val_auroc, 4),
                    **{k: round(v, 4) for k, v in m.items()},
                    "params": json.dumps(params),
                })
                print(f"{split_name:9s} {model_name:8s} seed={seed} "
                      f"AUROC={m['auroc']:.3f} AUPRC+={m['auprc_pos']:.3f} "
                      f"AUPRC-={m['auprc_neg']:.3f}")

                # Ship the scaffold-split seed-0 XGBoost model: the honest one.
                if (split_name, model_name, seed) == ("scaffold", "xgboost", 0):
                    keep = (clf, params, m, tr)

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")

    summarize(rows)

    clf, params, metrics, tr = keep
    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "model.pkl", "wb") as fh:
        pickle.dump({
            "model": clf,
            "params": params,
            "metrics": metrics,
            "split": "scaffold",
            "seed": 0,
            "n_train": int(len(tr)),
            "trained_on": date.today().isoformat(),
        }, fh)

    # Packed training fingerprints for the applicability-domain check.
    np.savez_compressed(ARTIFACTS / "domain.npz", packed=pack(X[tr]))
    print(f"saved artifacts/ ({(ARTIFACTS / 'domain.npz').stat().st_size / 1e3:.0f} KB domain)")


def summarize(rows):
    print("\nmean +/- std over seeds")
    for split_name in ("random", "scaffold"):
        for model_name in ("logreg", "xgboost"):
            sel = [r for r in rows if r["split"] == split_name and r["model"] == model_name]
            a = np.array([r["auroc"] for r in sel])
            n = np.array([r["auprc_neg"] for r in sel])
            print(f"  {split_name:9s} {model_name:8s} "
                  f"AUROC {a.mean():.3f} +/- {a.std():.3f}   "
                  f"AUPRC- {n.mean():.3f} +/- {n.std():.3f}")

    for model_name in ("logreg", "xgboost"):
        r = np.mean([x["auroc"] for x in rows if x["split"] == "random" and x["model"] == model_name])
        s = np.mean([x["auroc"] for x in rows if x["split"] == "scaffold" and x["model"] == model_name])
        print(f"  delta ({model_name}): {r - s:+.3f} AUROC")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results.csv")
    main(**vars(ap.parse_args()))
