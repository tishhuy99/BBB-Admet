"""Tests for the applicability-domain check."""

import numpy as np
import pytest

from bbb.domain import DomainChecker, accuracy_by_bin, fit_threshold
from bbb.features import featurize, pack

TRAIN = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "c1ccc2c(c1)Nc1ccccc1S2",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
]
NEAR = "CC(=O)Oc1ccccc1C(=O)OC"          # aspirin methyl ester
FAR = "CCCCCCCCCCCCCCCCCC(=O)NCCO"       # long-chain fatty amide


@pytest.fixture
def checker() -> DomainChecker:
    return DomainChecker(packed_train=pack(featurize(TRAIN)), threshold=0.30)


def test_training_molecule_is_in_domain(checker):
    sims, in_domain = checker.check([TRAIN[0]])
    assert sims[0] == pytest.approx(1.0)
    assert bool(in_domain[0])


def test_close_analog_is_in_domain(checker):
    sims, in_domain = checker.check([NEAR])
    assert sims[0] > 0.5
    assert bool(in_domain[0])


def test_distant_molecule_is_out_of_domain(checker):
    sims, in_domain = checker.check([FAR])
    assert sims[0] < 0.30
    assert not bool(in_domain[0])


def test_check_handles_batches(checker):
    sims, in_domain = checker.check([TRAIN[0], NEAR, FAR])
    assert sims.shape == (3,) and in_domain.shape == (3,)
    assert in_domain.tolist() == [True, True, False]


def test_accuracy_by_bin_partitions_all_molecules():
    rng = np.random.default_rng(0)
    sims = rng.uniform(0, 1, 500)
    y = rng.integers(0, 2, 500)
    proba = rng.uniform(0, 1, 500)
    bins = accuracy_by_bin(sims, y, proba)
    assert sum(b["n"] for b in bins) == 500


def test_fit_threshold_flags_a_degrading_low_similarity_bin():
    """Low-similarity molecules predicted badly should push the threshold up."""
    rng = np.random.default_rng(0)
    n = 400
    sims = np.concatenate([rng.uniform(0.5, 1.0, 300), rng.uniform(0.0, 0.2, 100)])
    y = rng.integers(0, 2, n)
    proba = np.where(np.arange(n) < 300, y.astype(float), 1.0 - y.astype(float))
    threshold, bins = fit_threshold(sims, y, proba)
    assert threshold > 0.0
    assert sum(b["n"] for b in bins) == n


def test_fit_threshold_returns_floor_when_accuracy_is_flat():
    rng = np.random.default_rng(1)
    n = 400
    sims = rng.uniform(0.0, 1.0, n)
    y = rng.integers(0, 2, n)
    proba = y.astype(float)          # perfect everywhere
    threshold, _ = fit_threshold(sims, y, proba)
    assert threshold == pytest.approx(0.0)
