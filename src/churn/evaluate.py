"""Metricas de avaliacao e analise de custo de negocio."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .model import ChurnMLP


def get_probs(
    model: ChurnMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Percorre o DataLoader e retorna (y_true, y_prob).

    Returns
    -------
    y_true : np.ndarray shape (n,)  — rotulos reais
    y_prob : np.ndarray shape (n,)  — probabilidades P(churn=1)
    """
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for Xb, yb in loader:
            probs = torch.sigmoid(model(Xb.to(device))).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(yb.tolist())
    return np.array(all_labels), np.array(all_probs)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Calcula 6 metricas de classificacao binaria.

    Metricas
    --------
    accuracy, precision, recall, f1  — sempre calculadas
    auc_roc, pr_auc                  — calculadas se y_prob fornecido
    """
    metrics: dict[str, float] = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"]  = average_precision_score(y_true, y_prob)
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    n_points: int = 100,
) -> tuple[float, float]:
    """Busca o threshold que minimiza o custo total de negocio.

        custo_total = FP * cost_fp + FN * cost_fn

    Returns
    -------
    best_threshold : float
    best_cost      : float
    """
    thresholds = np.linspace(0.05, 0.95, n_points)
    best_thresh, best_cost = 0.5, float("inf")
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost, best_thresh = cost, float(t)
    return best_thresh, best_cost
