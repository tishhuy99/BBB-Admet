"""Tests for featurization and similarity.

The RDKit cross-check is the important one: a bit-order mistake in packbits
does not raise, it just quietly degrades every applicability-domain decision.
"""

import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from bbb.features import (
    N_BITS,
    RADIUS,
    featurize,
    max_similarity,
    pack,
    tanimoto,
    unpack,
)

SMILES = [
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",            # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",     # caffeine
    "c1ccc2c(c1)Nc1ccccc1S2",           # phenothiazine core
]


def test_featurize_is_deterministic():
    a = featurize(SMILES)
    b = featurize(SMILES)
    assert np.array_equal(a, b)
    assert a.shape == (len(SMILES), N_BITS)
    assert a.dtype == np.uint8
    assert set(np.unique(a).tolist()) <= {0, 1}


def test_featurize_rejects_bad_smiles():
    with pytest.raises(ValueError, match="unparseable"):
        featurize(["CCO", "not_a_molecule"])


def test_pack_round_trip():
    fps = featurize(SMILES)
    packed = pack(fps)
    assert packed.shape == (len(SMILES), N_BITS // 8)
    assert np.array_equal(unpack(packed), fps)


def test_self_similarity_is_one():
    packed = pack(featurize(SMILES))
    sims = tanimoto(packed, packed)
    assert np.allclose(np.diag(sims), 1.0)


def test_tanimoto_matches_rdkit():
    """Cross-check the packed implementation against RDKit's own metric."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    bitvects = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in SMILES]
    packed = pack(featurize(SMILES))
    ours = tanimoto(packed, packed)

    for i in range(len(SMILES)):
        for j in range(len(SMILES)):
            expected = DataStructs.TanimotoSimilarity(bitvects[i], bitvects[j])
            assert ours[i, j] == pytest.approx(expected, abs=1e-6), (
                f"mismatch at ({i}, {j}): {ours[i, j]} vs {expected}"
            )


def test_similarity_is_symmetric_and_bounded():
    packed = pack(featurize(SMILES))
    sims = tanimoto(packed, packed)
    assert np.allclose(sims, sims.T)
    assert sims.min() >= 0.0 and sims.max() <= 1.0


def test_max_similarity_shape_and_identity():
    packed = pack(featurize(SMILES))
    ref = packed[1:]
    sims = max_similarity(packed[:1], ref)
    assert sims.shape == (1,)
    # A molecule present in the reference set is its own nearest neighbour.
    assert max_similarity(packed[1:2], ref)[0] == pytest.approx(1.0)


def test_dissimilar_molecules_score_low():
    packed = pack(featurize(["CCO", "c1ccc2c(c1)Nc1ccccc1S2"]))
    assert tanimoto(packed[:1], packed[1:])[0, 0] < 0.15
