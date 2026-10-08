"""Molecule -> graph featurization for message-passing models.

Deliberately plain atom and bond features: the comparison against Morgan
fingerprints is about the model class, so loading the node features with
hand-engineered descriptors would confound it.
"""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

# Covers >99% of this dataset; anything else lands in the "other" slot.
ATOMS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

NODE_DIM = len(ATOMS) + 1 + 6 + 5 + len(HYBRIDIZATIONS) + 1 + 5 + 1 + 1
EDGE_DIM = len(BOND_TYPES) + 2


def _one_hot(value, choices) -> list[float]:
    """One-hot with a trailing 'other' slot."""
    return [float(value == c) for c in choices] + [float(value not in choices)]


def atom_features(atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOMS)
        + _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4])
        + _one_hot(atom.GetFormalCharge(), [-1, 0, 1, 2])
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3])
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond) -> list[float]:
    return _one_hot(bond.GetBondType(), BOND_TYPES) + [float(bond.GetIsConjugated())]


def mol_to_graph(smiles: str, y: int | None = None) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = bond_features(bond)
        # Undirected: both directions, since message passing is directional.
        src += [i, j]
        dst += [j, i]
        attrs += [f, f]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float)
    else:
        # Single-atom molecules have no bonds; GATv2Conv needs the tensors to exist.
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float)
    return data


def build_graphs(smiles: np.ndarray, y: np.ndarray) -> list[Data]:
    return [mol_to_graph(s, int(label)) for s, label in zip(smiles, y)]
