"""Testes unitarios para src/churn/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.churn.config import LEAKAGE_COLS
from src.churn.preprocessing import TelcoEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_df(n: int = 10) -> pd.DataFrame:
    """Cria DataFrame minimo com colunas categoricas e numericas."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Gender":           rng.choice(["Male", "Female"], n),
            "Senior Citizen":   rng.integers(0, 2, n),
            "Partner":          rng.choice(["Yes", "No"], n),
            "Dependents":       rng.choice(["Yes", "No"], n),
            "Tenure Months":    rng.integers(1, 72, n),
            "Phone Service":    rng.choice(["Yes", "No"], n),
            "Internet Service": rng.choice(["DSL", "Fiber optic", "No"], n),
            "Contract":         rng.choice(["Month-to-month", "One year", "Two year"], n),
            "Monthly Charges":  rng.uniform(20, 120, n),
            "Total Charges":    rng.uniform(100, 8000, n),
        }
    )


# ---------------------------------------------------------------------------
# TelcoEncoder
# ---------------------------------------------------------------------------
class TestTelcoEncoder:
    def test_fit_populates_feature_columns(self):
        enc = TelcoEncoder()
        df = _make_df()
        enc.fit(df)
        assert len(enc.feature_columns_) > 0

    def test_transform_returns_numpy_float32(self):
        enc = TelcoEncoder()
        df = _make_df()
        enc.fit(df)
        result = enc.transform(df)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_transform_shape_matches_fit(self):
        df_train = _make_df(20)
        df_test = _make_df(5)
        enc = TelcoEncoder()
        enc.fit(df_train)
        result = enc.transform(df_test)
        assert result.shape[1] == len(enc.feature_columns_)

    def test_unseen_category_filled_with_zero(self):
        """Colunas presentes no treino mas ausentes no teste devem ser zeradas."""
        df_train = pd.DataFrame(
            {
                "Contract": ["Month-to-month", "One year", "Two year"],
                "Gender":   ["Male", "Female", "Male"],
            }
        )
        df_test = pd.DataFrame(
            {
                "Contract": ["Month-to-month"],
                "Gender":   ["Female"],
            }
        )
        enc = TelcoEncoder()
        enc.fit(df_train)
        result = enc.transform(df_test)
        assert result.shape[1] == len(enc.feature_columns_)

    def test_get_feature_names_out(self):
        enc = TelcoEncoder()
        enc.fit(_make_df())
        names = enc.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) == len(enc.feature_columns_)

    def test_fit_transform_idempotent(self):
        """Duas chamadas de transform no mesmo DataFrame retornam resultado identico."""
        enc = TelcoEncoder()
        df = _make_df(10)
        enc.fit(df)
        r1 = enc.transform(df)
        r2 = enc.transform(df)
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# LEAKAGE_COLS
# ---------------------------------------------------------------------------
def test_leakage_cols_contains_churn_score():
    assert "Churn Score" in LEAKAGE_COLS


def test_leakage_cols_contains_cltv():
    assert "CLTV" in LEAKAGE_COLS


def test_leakage_cols_does_not_contain_monthly_charges():
    assert "Monthly Charges" not in LEAKAGE_COLS
