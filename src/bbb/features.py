"""Morgan fingerprints, bit-packed storage, and vectorized Tanimoto similarity.

Fingerprints are cheap to regenerate (~2,000 molecules in a couple of seconds),
so nothing here is cached to disk except the packed training matrix the
applicability-domain check needs at serving time: 2048 bits -> 256 bytes per
molecule, ~350 KB for the training set.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

N_BITS = 2048
RADIUS = 2

# Built once at import. GetMorganFingerprintAsBitVect is deprecated in current
# RDKit; the generator API is the supported path and is also faster per call.
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


def fingerprint(mol) -> np.ndarray:
    """One molecule -> (N_BITS,) uint8 array of 0/1."""
    return _GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)


def featurize(smiles: np.ndarray | list[str]) -> np.ndarray:
    """SMILES -> (n, N_BITS) uint8 feature matrix.

    Raises on unparseable input rather than silently dropping rows: callers
    upstream of this have already standardized, so a failure here means the
    caller passed something that never went through bbb.data.standardize.
    """
    out = np.zeros((len(smiles), N_BITS), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"unparseable SMILES at index {i}: {smi!r}")
        out[i] = fingerprint(mol)
    return out


def pack(fps: np.ndarray) -> np.ndarray:
    """(n, N_BITS) uint8 of 0/1 -> (n, N_BITS // 8) uint8 of packed bits."""
    return np.packbits(fps, axis=-1)


def unpack(packed: np.ndarray, n_bits: int = N_BITS) -> np.ndarray:
    """Inverse of pack()."""
    return np.unpackbits(packed, axis=-1, count=n_bits)


def tanimoto(query_packed: np.ndarray, ref_packed: np.ndarray) -> np.ndarray:
    """Tanimoto similarity between packed fingerprints.

    query_packed: (n_query, n_bytes) or (n_bytes,)
    ref_packed:   (n_ref, n_bytes)
    returns:      (n_query, n_ref) float32

    np.bitwise_count (numpy >= 2.0) gives popcount without unpacking. The
    intermediate is n_query x n_ref x n_bytes bytes, so chunk the query axis
    if either side grows past a few thousand molecules.
    """
    q = np.atleast_2d(query_packed)
    inter = np.bitwise_count(q[:, None, :] & ref_packed[None, :, :]).sum(-1)
    pop_q = np.bitwise_count(q).sum(-1)[:, None]
    pop_r = np.bitwise_count(ref_packed).sum(-1)[None, :]
    union = pop_q + pop_r - inter
    # Two all-zero fingerprints have an undefined Tanimoto; report 0.0.
    return np.where(union == 0, 0.0, inter / np.maximum(union, 1)).astype(np.float32)


def max_similarity(query_packed: np.ndarray, ref_packed: np.ndarray) -> np.ndarray:
    """Nearest-neighbour Tanimoto of each query against the reference set."""
    return tanimoto(query_packed, ref_packed).max(axis=1)
