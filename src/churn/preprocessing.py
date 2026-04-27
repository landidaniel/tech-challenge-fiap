"""Preprocessamento do dataset Telco Customer Churn.

Fornece:
- load_raw: carrega o Excel e devolve (X_df, y_series)
- TelcoEncoder: transformador sklearn compativel (fit/transform)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import LEAKAGE_COLS


def load_raw(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega o dataset bruto e separa features do alvo.

    - Remove colunas com leakage (definidas em config.LEAKAGE_COLS)
    - Cria a coluna binaria ``Churn`` a partir de ``Churn Label``
    - Converte ``Total Charges`` para numerico

    Returns
    -------
    X : pd.DataFrame  — features (sem leakage, sem alvo)
    y : pd.Series     — alvo binario 0/1
    """
    df = pd.read_excel(path)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0.0)
    y = (df["Churn Label"] == "Yes").astype(int).rename("Churn")

    drop_cols = [c for c in LEAKAGE_COLS + ["Churn"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y


class TelcoEncoder(BaseEstimator, TransformerMixin):
    """Encoder sklearn-compativel para features categoricas do Telco.

    Aplica ``pd.get_dummies(drop_first=True)`` e alinha as colunas ao
    conjunto de treino para garantir reproducibilidade em producao.

    Parameters
    ----------
    drop_first : bool
        Passa para ``pd.get_dummies``. Default True.
    """

    def __init__(self, drop_first: bool = True) -> None:
        self.drop_first = drop_first
        self.feature_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "TelcoEncoder":
        encoded = pd.get_dummies(X, drop_first=self.drop_first)
        self.feature_columns_ = encoded.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        encoded = pd.get_dummies(X, drop_first=self.drop_first)
        # Alinha colunas: adiciona 0 para colunas ausentes, ignora extras
        encoded = encoded.reindex(columns=self.feature_columns_, fill_value=0)
        return encoded.values.astype(np.float32)

    def get_feature_names_out(self) -> list[str]:
        return self.feature_columns_
