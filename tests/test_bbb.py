import numpy as np
import pytest
from rdkit import Chem

from bbb.data import (
    Dataset,
    load,
    random_split,
    scaffold_of,
    scaffold_split,
    standardize,
)


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return load()


def test_scaffold_split_has_zero_scaffold_overlap(ds):
    for seed in range(5):
        tr, va, te = scaffold_split(ds.scaffold, seed)
        s_tr, s_va, s_te = (set(ds.scaffold[i]) for i in (tr, va, te))
        assert not s_tr & s_te
        assert not s_tr & s_va
        assert not s_va & s_te


def test_splits_partition_the_dataset(ds):
    for split_fn in (lambda s: random_split(len(ds), s),
                     lambda s: scaffold_split(ds.scaffold, s)):
        for seed in range(3):
            tr, va, te = split_fn(seed)
            union = np.concatenate([tr, va, te])
            assert len(union) == len(ds)
            assert len(set(union.tolist())) == len(ds)


def test_splits_are_deterministic_under_seed(ds):
    a = scaffold_split(ds.scaffold, 7)
    b = scaffold_split(ds.scaffold, 7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    c = scaffold_split(ds.scaffold, 8)
    assert not np.array_equal(a[2], c[2]), "different seeds gave identical test sets"


def test_split_sizes_are_roughly_on_target(ds):
    tr, va, te = scaffold_split(ds.scaffold, 0)
    n = len(ds)
    assert 0.60 <= len(tr) / n <= 0.80
    assert len(te) / n <= 0.22
    assert len(va) > 0


def test_acyclic_molecules_get_their_own_scaffold_group():
    """Empty Murcko cores must not collapse into one group."""
    groups = set()
    for smi in ["CCO", "CC(C)CO", "CCCCCC(=O)O"]:
        mol = standardize(smi)
        groups.add(scaffold_of(mol, Chem.MolToInchiKey(mol)))
    assert len(groups) == 3
    assert all(g.startswith("acyclic:") for g in groups)

    ring = standardize("c1ccccc1O")
    assert not scaffold_of(ring, Chem.MolToInchiKey(ring)).startswith("acyclic:")


def test_standardize_strips_counterions():
    salt = standardize("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
    free = standardize("CC(=O)Oc1ccccc1C(=O)O")
    assert Chem.MolToSmiles(salt) == Chem.MolToSmiles(free)


def test_standardize_rejects_garbage():
    assert standardize("not_a_molecule") is None
    assert standardize("C(C(C") is None


def test_dedupe_removes_duplicate_inchikeys(ds):
    assert len(set(ds.inchikey.tolist())) == len(ds)
    assert ds.report["conflicting_groups_dropped"] > 0
    assert ds.report["duplicate_rows_collapsed"] > 0