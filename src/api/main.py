"""API FastAPI de inferencia de Churn.

Endpoints
---------
GET  /health       — status da aplicacao e modelo carregado
POST /predict      — predicao de churn para um cliente
POST /predict/batch — predicao em lote (ate 1 000 clientes)

Execucao local:
    uvicorn src.api.main:app --reload --port 8000

Variaveis de ambiente:
    ARTIFACTS_DIR  — caminho para a pasta com pipeline.pkl / model.pth / meta.pkl
    LOG_LEVEL      — nivel de log (INFO | DEBUG | WARNING)
    THRESHOLD      — threshold override (float 0-1); se omitido usa meta.pkl
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException

from ..churn.pipeline import load_artifacts, predict_from_dataframe
from .logging_config import configure_logging, get_logger
from .middleware import LatencyLoggingMiddleware
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CustomerFeatures,
    HealthResponse,
    PredictResponse,
)

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
configure_logging(LOG_LEVEL)
logger = get_logger("api.main")

_ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(Path(__file__).parent.parent.parent / "models")))
_DEVICE = torch.device("cpu")

# Estado global (carregado no startup)
_state: dict[str, Any] = {
    "pipeline": None,
    "model":    None,
    "meta":     None,
    "ready":    False,
}

VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega artefatos do modelo no startup; libera no shutdown."""
    try:
        pipeline, model, meta = load_artifacts(_ARTIFACTS_DIR, device=_DEVICE)
        _state["pipeline"] = pipeline
        _state["model"]    = model
        _state["meta"]     = meta
        _state["ready"]    = True
        # Permite override de threshold via variavel de ambiente
        if os.getenv("THRESHOLD"):
            _state["meta"]["threshold"] = float(os.environ["THRESHOLD"])
        logger.info(
            "Modelo carregado. input_dim=%d threshold=%.3f",
            meta["input_dim"],
            meta["threshold"],
        )
    except FileNotFoundError:
        logger.warning(
            "Artefatos nao encontrados em %s. "
            "API iniciada sem modelo (apenas /health disponivel).",
            _ARTIFACTS_DIR,
        )
    yield
    logger.info("API encerrada.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Churn Prediction API",
    description="API de inferencia de churn para clientes Telco — FIAP Tech Challenge Etapa 3",
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(LatencyLoggingMiddleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health() -> HealthResponse:
    """Verifica se a API e o modelo estao operacionais."""
    threshold = _state["meta"]["threshold"] if _state["meta"] else None
    return HealthResponse(
        status="ok" if _state["ready"] else "degraded",
        model_loaded=_state["ready"],
        version=VERSION,
        threshold=threshold,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(customer: CustomerFeatures) -> PredictResponse:
    """Prediz a probabilidade de churn para um unico cliente."""
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Modelo nao carregado")

    X_df = customer.to_dataframe()
    threshold: float = _state["meta"]["threshold"]

    try:
        probs, labels = predict_from_dataframe(
            _state["pipeline"],
            _state["model"],
            X_df,
            threshold=threshold,
            device=_DEVICE,
        )
    except Exception as exc:
        logger.exception("Erro na inferencia: %s", exc)
        raise HTTPException(status_code=422, detail=f"Erro na inferencia: {exc}") from exc

    prob = float(probs[0])
    label = bool(labels[0])
    confidence = _confidence_level(prob, threshold)

    logger.info(
        "predict customer_id=%s prob=%.4f label=%s threshold=%.3f",
        customer.customer_id,
        prob,
        label,
        threshold,
    )

    return PredictResponse(
        customer_id=customer.customer_id,
        churn_probability=prob,
        churn_label=label,
        threshold=threshold,
        confidence=confidence,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Prediz churn para multiplos clientes em uma unica chamada."""
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Modelo nao carregado")

    import pandas as pd

    threshold: float = _state["meta"]["threshold"]
    df = pd.concat(
        [c.to_dataframe() for c in request.customers], ignore_index=True
    )

    try:
        probs, labels = predict_from_dataframe(
            _state["pipeline"],
            _state["model"],
            df,
            threshold=threshold,
            device=_DEVICE,
        )
    except Exception as exc:
        logger.exception("Erro na inferencia em lote: %s", exc)
        raise HTTPException(status_code=422, detail=f"Erro na inferencia: {exc}") from exc

    predictions = [
        PredictResponse(
            customer_id=c.customer_id,
            churn_probability=float(p),
            churn_label=bool(lb),
            threshold=threshold,
            confidence=_confidence_level(float(p), threshold),
        )
        for c, p, lb in zip(request.customers, probs, labels)
    ]

    logger.info("predict_batch n=%d threshold=%.3f", len(predictions), threshold)
    return BatchPredictResponse(predictions=predictions, total=len(predictions))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _confidence_level(prob: float, threshold: float) -> str:
    """Classifica a confianca da predicao em high / medium / low."""
    distance = abs(prob - threshold)
    if distance >= 0.3:
        return "high"
    if distance >= 0.15:
        return "medium"
    return "low"
