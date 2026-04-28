"""Pipeline reprodutivel: sklearn Pipeline + utilitarios de persistencia.

build_pipeline  : cria Pipeline(TelcoEncoder, StandardScaler)
save_artifacts  : persiste scaler (.pkl) e modelo (.pth)
load_artifacts  : carrega scaler + modelo do disco
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ARTIFACTS_DIR
from .model import ChurnMLP
from .preprocessing import TelcoEncoder

_log = logging.getLogger("churn.pipeline")


def build_pipeline() -> Pipeline:
    """Retorna Pipeline(encoder -> scaler) antes do fit."""
    return Pipeline(
        steps=[
            ("encoder", TelcoEncoder()),
            ("scaler", StandardScaler()),
        ]
    )


def save_artifacts(
    pipeline: Pipeline,
    model: ChurnMLP,
    threshold: float,
    out_dir: Path = ARTIFACTS_DIR,
) -> None:
    """Salva pipeline (scaler + encoder) e pesos do modelo.

    Arquivos gerados em ``out_dir/``:
    - ``pipeline.pkl``  — Pipeline sklearn serializado
    - ``model.pth``     — state_dict do ChurnMLP
    - ``meta.pkl``      — metadados (input_dim, hidden_dims, dropout, threshold)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    torch.save(model.state_dict(), out_dir / "model.pth")

    encoder: TelcoEncoder = pipeline.named_steps["encoder"]
    meta = {
        "input_dim":   len(encoder.feature_columns_),
        "hidden_dims": _infer_hidden_dims(model),
        "dropout":     _infer_dropout(model),
        "threshold":   threshold,
        "feature_columns": encoder.feature_columns_,
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    _log.info("Artefatos salvos em: %s", out_dir)


def load_artifacts(
    artifacts_dir: Path = ARTIFACTS_DIR,
    device: torch.device | None = None,
) -> tuple[Pipeline, ChurnMLP, dict]:
    """Carrega pipeline e modelo do disco.

    Returns
    -------
    pipeline  : Pipeline sklearn (encoder + scaler)
    model     : ChurnMLP carregado em ``device``
    meta      : dict com input_dim, threshold, feature_columns, ...
    """
    if device is None:
        device = torch.device("cpu")

    with open(artifacts_dir / "pipeline.pkl", "rb") as f:
        pipeline: Pipeline = pickle.load(f)

    with open(artifacts_dir / "meta.pkl", "rb") as f:
        meta: dict = pickle.load(f)

    model = ChurnMLP(
        input_dim=meta["input_dim"],
        hidden_dims=meta["hidden_dims"],
        dropout=meta["dropout"],
    )
    model.load_state_dict(torch.load(artifacts_dir / "model.pth", map_location=device))
    model.to(device).eval()

    return pipeline, model, meta


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _infer_hidden_dims(model: ChurnMLP) -> list[int]:
    """Extrai hidden_dims inspecionando as camadas Linear do modelo."""
    dims = []
    layers = list(model.net.children())
    for layer in layers[:-1]:  # ultima Linear e a camada de saida
        if isinstance(layer, torch.nn.Linear):
            dims.append(layer.out_features)
    return dims


def _infer_dropout(model: ChurnMLP) -> float:
    for layer in model.net.children():
        if isinstance(layer, torch.nn.Dropout):
            return layer.p
    return 0.0


def predict_from_dataframe(
    pipeline: Pipeline,
    model: ChurnMLP,
    X_raw: "pd.DataFrame",  # noqa: F821
    threshold: float,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Executa inferencia completa: raw DataFrame -> (probs, labels).

    Parameters
    ----------
    X_raw : pd.DataFrame sem colunas de leakage nem a coluna Churn
    threshold : threshold de decisao (0.0–1.0)

    Returns
    -------
    probs  : np.ndarray shape (n,)  — probabilidades P(churn=1)
    labels : np.ndarray shape (n,)  — 0 ou 1
    """
    if device is None:
        device = torch.device("cpu")

    X_proc = pipeline.transform(X_raw)
    tensor = torch.tensor(X_proc, dtype=torch.float32).to(device)
    probs = model.predict_proba(tensor).cpu().numpy()
    labels = (probs >= threshold).astype(int)
    return probs, labels
