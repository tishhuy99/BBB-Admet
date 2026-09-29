"""API tests.

These run against the real artifacts, so bbb.train must have been run first.
The out-of-domain test is the one that matters: the others check plumbing,
that one checks the feature the README claims.
"""

import pytest
from fastapi.testclient import TestClient

from bbb.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # triggers lifespan, loading the model
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_model_metadata(client):
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["split"] == "scaffold"
    assert body["n_train"] > 1000
    assert 0.0 <= body["domain_threshold"] <= 1.0
    assert "auroc" in body["metrics"]


def test_predict_returns_one_row_per_molecule(client):
    smiles = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"]
    r = client.post("/predict", json={"smiles": smiles})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == len(smiles)
    for p, s in zip(preds, smiles):
        assert p["smiles"] == s
        assert p["prediction"] in (0, 1)
        assert 0.0 <= p["probability"] <= 1.0
        assert 0.0 <= p["max_similarity"] <= 1.0
        assert isinstance(p["in_domain"], bool)


def test_invalid_smiles_returns_422(client):
    r = client.post("/predict", json={"smiles": ["CCO", "not_a_molecule"]})
    assert r.status_code == 422
    assert "unparseable" in r.text


def test_empty_batch_returns_422(client):
    assert client.post("/predict", json={"smiles": []}).status_code == 422


def test_oversized_batch_returns_422(client):
    assert client.post("/predict", json={"smiles": ["CCO"] * 500}).status_code == 422


def test_malformed_body_returns_422(client):
    assert client.post("/predict", json={"molecules": ["CCO"]}).status_code == 422


def test_distant_molecule_is_flagged_out_of_domain(client):
    """A molecule far from drug-like training space must not be reported
    as a confident prediction."""
    far = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)NCCOCCOCCOCCO"
    r = client.post("/predict", json={"smiles": ["CC(=O)Oc1ccccc1C(=O)O", far]})
    assert r.status_code == 200
    body = r.json()
    aspirin, outlier = body["predictions"]
    assert aspirin["max_similarity"] > outlier["max_similarity"]
    assert outlier["in_domain"] is False
    assert body["n_out_of_domain"] >= 1
