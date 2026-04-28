"""Loop de treinamento com early stopping e ReduceLROnPlateau."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .model import ChurnMLP

_log = logging.getLogger("churn.train")


def train(
    model: ChurnMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 150,
    patience: int = 15,
    pos_weight: float | None = None,
) -> dict[str, list[float]]:
    """Treina o modelo com early stopping e monitora val_loss.

    Estrategia
    ----------
    - Otimizador : Adam + weight_decay=1e-4 (regularizacao L2)
    - Loss       : BCEWithLogitsLoss com pos_weight (desbalanceamento)
    - LR decay   : ReduceLROnPlateau (fator 0.5, patience=5)
    - Early stop : interrompe quando val_loss nao melhora por ``patience``
                   epochs; restaura pesos do melhor checkpoint.

    Returns
    -------
    history : dict com listas ``train_loss``, ``val_loss``, ``val_f1``
    """
    pw = (
        torch.tensor([pos_weight], dtype=torch.float32).to(device)
        if pos_weight is not None
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
    }
    best_val_loss = float("inf")
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ------------------------------------------------------------------ #
        # Treino                                                               #
        # ------------------------------------------------------------------ #
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_loader.dataset)  # type: ignore[arg-type]

        # ------------------------------------------------------------------ #
        # Validacao                                                            #
        # ------------------------------------------------------------------ #
        model.eval()
        val_loss = 0.0
        all_probs: list[float] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                val_loss += criterion(logits, yb).item() * len(yb)
                all_probs.extend(torch.sigmoid(logits).cpu().tolist())
                all_labels.extend(yb.cpu().tolist())
        val_loss /= len(val_loader.dataset)  # type: ignore[arg-type]

        val_f1 = f1_score(
            all_labels,
            (np.array(all_probs) >= 0.5).astype(int),
            zero_division=0,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        scheduler.step(val_loss)

        # ------------------------------------------------------------------ #
        # Early stopping                                                       #
        # ------------------------------------------------------------------ #
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            _log.info(
                "Epoch %3d | train=%.4f | val=%.4f | f1=%.4f | patience=%d/%d",
                epoch, train_loss, val_loss, val_f1, no_improve, patience,
            )

        if no_improve >= patience:
            _log.info(
                "Early stopping epoch %d (best val_loss=%.4f)", epoch, best_val_loss
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
