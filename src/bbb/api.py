"""FastAPI service for BBB permeability prediction.

Three endpoints:
    GET  /health   liveness
    GET  /model    version, metrics, training date, AD threshold
    POST /predict  batch SMILES -> prediction, probability, in_domain, max_similarity

Every prediction carries an applicability-domain flag. A probability without
one is not an answer: the model was trained on 1,365 molecules and says
nothing trustworthy about chemistry far from them.
"""

from __future__ import annotations

import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from rdkit import Chem, RDLogger

from bbb.domain import DomainChecker
from bbb.features import featurize

RDLogger.DisableLog("rdApp.*")

ARTIFACTS = Path("artifacts")
MAX_BATCH = 256

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and training fingerprints once, at startup."""
    with open(ARTIFACTS / "model.pkl", "rb") as fh:
        STATE["bundle"] = pickle.load(fh)
    STATE["domain"] = DomainChecker.load(ARTIFACTS)
    yield
    STATE.clear()


app = FastAPI(
    title="BBB permeability prediction",
    description="Morgan fingerprints + XGBoost, scaffold-split evaluation, "
                "Tanimoto applicability domain.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    smiles: Annotated[list[str], Field(min_length=1, max_length=MAX_BATCH)]

    @field_validator("smiles")
    @classmethod
    def parseable(cls, v: list[str]) -> list[str]:
        """Reject unparseable SMILES at the boundary, with the offending index.

        Pydantic turns a ValueError here into a 422, which is the right code:
        the request is well-formed JSON but semantically invalid.
        """
        bad = [(i, s) for i, s in enumerate(v) if Chem.MolFromSmiles(s) is None]
        if bad:
            detail = ", ".join(f"[{i}] {s!r}" for i, s in bad[:5])
            raise ValueError(f"unparseable SMILES: {detail}")
        return v


class Prediction(BaseModel):
    smiles: str
    prediction: int = Field(description="1 = BBB permeable")
    probability: float
    in_domain: bool
    max_similarity: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    n_out_of_domain: int


class ModelInfo(BaseModel):
    version: str
    split: str
    seed: int
    n_train: int
    trained_on: str
    params: dict
    metrics: dict
    domain_threshold: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "bundle" in STATE}


@app.get("/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    b = STATE["bundle"]
    return ModelInfo(
        version=app.version,
        split=b["split"],
        seed=b["seed"],
        n_train=b["n_train"],
        trained_on=b["trained_on"],
        params=b["params"],
        metrics=b["metrics"],
        domain_threshold=STATE["domain"].threshold,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if "bundle" not in STATE:
        raise HTTPException(status_code=503, detail="model not loaded")

    X = featurize(req.smiles)
    proba = STATE["bundle"]["model"].predict_proba(X)[:, 1]
    sims, in_domain = STATE["domain"].check(req.smiles)

    preds = [
        Prediction(
            smiles=s,
            prediction=int(p >= 0.5),
            probability=round(float(p), 4),
            in_domain=bool(d),
            max_similarity=round(float(m), 4),
        )
        for s, p, d, m in zip(req.smiles, proba, in_domain, sims)
    ]
    return PredictResponse(
        predictions=preds,
        n_out_of_domain=int(np.sum(~in_domain)),
    )
