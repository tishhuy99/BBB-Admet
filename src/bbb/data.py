"""Load, standardize, deduplicate and split the BBB_Martins dataset.

No pandas: this module is imported by the API, and keeping it stdlib+numpy+rdkit
stops pandas leaking into the runtime image.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()

RAW_CSV = "data/bbb_martins_raw.csv"


@dataclass
class Dataset:
    smiles: np.ndarray      # canonical, standardized
    y: np.ndarray           # int8, 1 = BBB permeable
    inchikey: np.ndarray
    scaffold: np.ndarray    # Murcko scaffold SMILES, or "acyclic:<inchikey>"
    report: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def take(self, idx: np.ndarray) -> "Dataset":
        return Dataset(
            self.smiles[idx], self.y[idx], self.inchikey[idx], self.scaffold[idx],
            dict(self.report, subset_of=len(self.y)),
        )


def standardize(smiles: str):
    """Parse and normalize one SMILES. Returns an RDKit mol, or None if unparseable.

    Largest-fragment selection strips counterions, which matters here: a
    hydrochloride salt and its free base are the same molecule for BBB purposes
    and must not end up on opposite sides of a split.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.Cleanup(mol)
    mol = _LARGEST_FRAGMENT.choose(mol)
    mol = _UNCHARGER.uncharge(mol)
    return mol


def scaffold_of(mol, inchikey: str) -> str:
    """Bemis-Murcko scaffold, with acyclic molecules given their own group.

    GetScaffoldForMol returns an empty molecule for anything with no ring
    system (~99 rows here). Lumping those into a single "" group would put 5%
    of the data in one bucket that is not a scaffold and is not structurally
    coherent, which would leak across the split boundary in the opposite
    direction. Each acyclic molecule becomes its own singleton group instead.
    """
    core = MurckoScaffold.GetScaffoldForMol(mol)
    smi = Chem.MolToSmiles(core)
    return smi if smi else f"acyclic:{inchikey}"


def load(path: str = RAW_CSV, dedupe: bool = True) -> Dataset:
    """Read the raw TDC CSV, standardize, and optionally collapse duplicates.

    Duplicate policy: identical InChIKeys with agreeing labels collapse to one
    row; groups with conflicting labels are dropped entirely, since with 2-3
    members a majority vote is noise. Set dedupe=False to reproduce the TDC
    leaderboard split, which does not deduplicate.
    """
    smiles, ys, keys, scaffolds = [], [], [], []
    n_rows = n_unparseable = 0

    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            mol = standardize(row["Drug"])
            if mol is None:
                n_unparseable += 1
                continue
            key = Chem.MolToInchiKey(mol)
            smiles.append(Chem.MolToSmiles(mol))
            ys.append(int(float(row["Y"])))
            keys.append(key)
            scaffolds.append(scaffold_of(mol, key))

    report = {"rows_in": n_rows, "unparseable": n_unparseable}

    if dedupe:
        by_key: dict[str, list[int]] = defaultdict(list)
        for i, key in enumerate(keys):
            by_key[key].append(i)

        keep, n_conflict, n_collapsed = [], 0, 0
        for members in by_key.values():
            labels = {ys[i] for i in members}
            if len(labels) > 1:
                n_conflict += 1
                continue
            n_collapsed += len(members) - 1
            keep.append(members[0])
        keep.sort()

        smiles = [smiles[i] for i in keep]
        ys = [ys[i] for i in keep]
        keys = [keys[i] for i in keep]
        scaffolds = [scaffolds[i] for i in keep]
        report |= {"conflicting_groups_dropped": n_conflict,
                   "duplicate_rows_collapsed": n_collapsed}

    y = np.asarray(ys, dtype=np.int8)
    report |= {"rows_out": len(y), "positive_fraction": float(y.mean())}

    return Dataset(np.asarray(smiles), y, np.asarray(keys),
                   np.asarray(scaffolds), report)


def random_split(n: int, seed: int, fracs=(0.7, 0.1, 0.2)):
    """Uniform random split. Returns (train_idx, val_idx, test_idx)."""
    assert abs(sum(fracs) - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def scaffold_split(scaffolds: np.ndarray, seed: int, fracs=(0.7, 0.1, 0.2)):
    """Scaffold split with zero scaffold overlap between any two subsets.

    Scaffold groups are shuffled under `seed` and filled greedily into test,
    then val, then train. Groups larger than half the val target are forced
    into train first, so that one 136-member scaffold class cannot swamp a
    small subset. Seeding the group order is what makes five-seed mean +/- std
    meaningful; a size-sorted deterministic split has no variance to report.
    """
    assert abs(sum(fracs) - 1.0) < 1e-9
    n = len(scaffolds)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(scaffolds):
        groups[s].append(i)

    n_val_target = int(round(fracs[1] * n))
    n_test_target = int(round(fracs[2] * n))
    big_cutoff = 0.5 * min(n_val_target, n_test_target)

    # Deterministic base order (size desc, then scaffold string) before shuffling,
    # so the same seed gives the same split regardless of dict iteration order.
    keys = sorted(groups, key=lambda k: (-len(groups[k]), k))
    big = [k for k in keys if len(groups[k]) > big_cutoff]
    small = [k for k in keys if len(groups[k]) <= big_cutoff]

    rng = np.random.default_rng(seed)
    small = [small[i] for i in rng.permutation(len(small))]

    train, val, test = [groups[k] for k in big], [], []
    n_train = sum(len(g) for g in train)
    n_val = n_test = 0

    for k in small:
        g = groups[k]
        if n_test + len(g) <= n_test_target:
            test.append(g)
            n_test += len(g)
        elif n_val + len(g) <= n_val_target:
            val.append(g)
            n_val += len(g)
        else:
            train.append(g)
            n_train += len(g)

    flat = lambda gs: np.asarray(sorted(i for g in gs for i in g), dtype=int)
    return flat(train), flat(val), flat(test)


if __name__ == "__main__":
    ds = load()
    print(ds.report)
    for name, split in (("random", random_split(len(ds), 0)),
                        ("scaffold", scaffold_split(ds.scaffold, 0))):
        tr, va, te = split
        print(f"{name:9s} train={len(tr):5d} val={len(va):4d} test={len(te):4d} "
              f"test_pos={ds.y[te].mean():.3f}")