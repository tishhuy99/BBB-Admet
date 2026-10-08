"""GATv2 benchmark, run under the same protocol as the fingerprint models.

Same splitters, same five seeds, hyperparameters and early stopping chosen on
validation, test touched once per cell. Anything less and the comparison
against XGBoost says nothing.

CPU only: 1,950 molecules is small, and PyG scatter ops on MPS are unreliable.

    python -m bbb.gat --out results_gat.csv

Runtime is roughly 20-40 minutes for 2 splits x 5 seeds x 3 configurations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool

from bbb.data import load, random_split, scaffold_split
from bbb.graphs import EDGE_DIM, NODE_DIM, build_graphs

DEVICE = torch.device("cpu")
SEEDS = range(5)
MAX_EPOCHS = 200
PATIENCE = 30

# Three configurations, deliberately small. With 1,365 training molecules a
# wider search would fit validation noise, the same failure the XGBoost grid
# showed when it picked max_depth=10.
GRID = [
    {"hidden": 64, "heads": 4, "layers": 2, "dropout": 0.2, "lr": 1e-3},
    {"hidden": 128, "heads": 4, "layers": 2, "dropout": 0.5, "lr": 1e-3},
    {"hidden": 64, "heads": 8, "layers": 3, "dropout": 0.5, "lr": 5e-4},
]


class GAT(nn.Module):
    def __init__(self, hidden=64, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = NODE_DIM
        for _ in range(layers):
            self.convs.append(
                GATv2Conv(in_dim, hidden, heads=heads, dropout=dropout,
                          edge_dim=EDGE_DIM, concat=True)
            )
            self.norms.append(nn.BatchNorm1d(hidden * heads))
            in_dim = hidden * heads

        self.dropout = nn.Dropout(dropout)
        # Mean and sum pooling concatenated: mean is size-invariant, sum keeps
        # molecular size, which matters for permeability.
        self.head = nn.Sequential(
            nn.Linear(in_dim * 2, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = torch.relu(norm(conv(x, ei, ea)))
            x = self.dropout(x)
        pooled = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
        return self.head(pooled).squeeze(-1)


def evaluate(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            probs.append(torch.sigmoid(model(batch)).cpu().numpy())
            ys.append(batch.y.cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs)


def fit_one(graphs, tr, va, params, seed):
    """Train with early stopping on validation AUROC; return the best model."""
    torch.manual_seed(seed)
    train_loader = DataLoader([graphs[i] for i in tr], batch_size=64, shuffle=True)
    val_loader = DataLoader([graphs[i] for i in va], batch_size=256)

    model = GAT(params["hidden"], params["heads"], params["layers"],
                params["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=1e-4)

    # 76.5% positive: weight the minority class so the loss is not dominated
    # by the majority, matching class_weight="balanced" on the linear baseline.
    y_tr = np.array([float(graphs[i].y) for i in tr])
    pos_weight = torch.tensor([(1 - y_tr.mean()) / y_tr.mean()], dtype=torch.float)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=1.0 / pos_weight)

    best_state, best_val, stale = None, -np.inf, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            opt.step()

        y_va, p_va = evaluate(model, val_loader)
        val_auroc = roc_auc_score(y_va, p_va)
        if val_auroc > best_val:
            best_state, best_val, stale = copy.deepcopy(model.state_dict()), val_auroc, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val


def main(out: str = "results_gat.csv"):
    ds = load()
    y = ds.y.astype(int)
    graphs = build_graphs(ds.smiles, y)
    print(f"{len(graphs)} graphs, node dim {NODE_DIM}, edge dim {EDGE_DIM}")

    splitters = {
        "random": lambda s: random_split(len(ds), s),
        "scaffold": lambda s: scaffold_split(ds.scaffold, s),
    }

    rows = []
    for split_name, splitter in splitters.items():
        for seed in SEEDS:
            tr, va, te = splitter(seed)

            best, best_val, best_params = None, -np.inf, GRID[0]
            for params in GRID:
                model, val_auroc = fit_one(graphs, tr, va, params, seed)
                # Same complexity tie-break as the fingerprint models: a larger
                # configuration must clear the incumbent by 0.01 on validation.
                if best is None or val_auroc > best_val + 0.01:
                    best, best_val, best_params = model, val_auroc, params

            y_te, p_te = evaluate(best, DataLoader([graphs[i] for i in te], batch_size=256))
            row = {
                "model": "gat", "split": split_name, "seed": seed,
                "val_auroc": round(best_val, 4),
                "auroc": round(roc_auc_score(y_te, p_te), 4),
                "auprc_pos": round(average_precision_score(y_te, p_te), 4),
                "auprc_neg": round(average_precision_score(1 - y_te, 1 - p_te), 4),
                "params": json.dumps(best_params),
            }
            rows.append(row)
            print(f"{split_name:9s} gat seed={seed} AUROC={row['auroc']:.3f} "
                  f"AUPRC-={row['auprc_neg']:.3f} {best_params}")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nmean +/- std over seeds")
    for split_name in ("random", "scaffold"):
        sel = [r for r in rows if r["split"] == split_name]
        a = np.array([r["auroc"] for r in sel])
        n = np.array([r["auprc_neg"] for r in sel])
        print(f"  {split_name:9s} gat AUROC {a.mean():.3f} +/- {a.std():.3f}   "
              f"AUPRC- {n.mean():.3f} +/- {n.std():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_gat.csv")
    main(**vars(ap.parse_args()))
