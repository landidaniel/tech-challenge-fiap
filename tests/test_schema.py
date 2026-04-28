"""Validacao de schema do dataset bruto com pandera.

Testa que o DataFrame carregado do Excel satisfaz as restricoes de tipos,
nulos e dominios esperados antes de entrar no pipeline de preprocessamento.
"""

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from pandera.pandas import Check, Column, DataFrameSchema

# ---------------------------------------------------------------------------
# Schema pandera para o dataset Telco (pos-load_raw, sem leakage)
# ---------------------------------------------------------------------------
TELCO_SCHEMA = DataFrameSchema(
    columns={
        "Gender": Column(
            str,
            checks=Check.isin(["Male", "Female"]),
            nullable=False,
        ),
        "Senior Citizen": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
        ),
        "Partner": Column(
            str,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "Dependents": Column(
            str,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "Tenure Months": Column(
            int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "Monthly Charges": Column(
            float,
            checks=Check.greater_than_or_equal_to(0.0),
            nullable=False,
        ),
        "Total Charges": Column(
            float,
            checks=Check.greater_than_or_equal_to(0.0),
            nullable=False,
        ),
        "Contract": Column(
            str,
            checks=Check.isin(["Month-to-month", "One year", "Two year"]),
            nullable=False,
        ),
        "Internet Service": Column(
            str,
            checks=Check.isin(["DSL", "Fiber optic", "No"]),
            nullable=False,
        ),
    },
    strict=False,  # permite colunas extras
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_valid_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Gender":           rng.choice(["Male", "Female"], n),
            "Senior Citizen":   rng.choice([0, 1], n).astype(int),
            "Partner":          rng.choice(["Yes", "No"], n),
            "Dependents":       rng.choice(["Yes", "No"], n),
            "Tenure Months":    rng.integers(0, 72, n).astype(int),
            "Monthly Charges":  rng.uniform(20.0, 120.0, n).astype(float),
            "Total Charges":    rng.uniform(0.0, 8000.0, n).astype(float),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "Internet Service": rng.choice(["DSL", "Fiber optic", "No"], n),
        }
    )


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------
def test_valid_dataframe_passes_schema():
    df = _make_valid_df()
    TELCO_SCHEMA.validate(df)  # nao deve levantar excecao


def test_invalid_gender_fails():
    df = _make_valid_df()
    df.loc[0, "Gender"] = "Unknown"
    with pytest.raises(pa.errors.SchemaError):
        TELCO_SCHEMA.validate(df)


def test_negative_tenure_fails():
    df = _make_valid_df()
    df.loc[0, "Tenure Months"] = -1
    with pytest.raises(pa.errors.SchemaError):
        TELCO_SCHEMA.validate(df)


def test_negative_monthly_charges_fails():
    df = _make_valid_df()
    df.loc[0, "Monthly Charges"] = -10.0
    with pytest.raises(pa.errors.SchemaError):
        TELCO_SCHEMA.validate(df)


def test_invalid_contract_fails():
    df = _make_valid_df()
    df.loc[0, "Contract"] = "Weekly"
    with pytest.raises(pa.errors.SchemaError):
        TELCO_SCHEMA.validate(df)


def test_invalid_internet_service_fails():
    df = _make_valid_df()
    df.loc[0, "Internet Service"] = "5G"
    with pytest.raises(pa.errors.SchemaError):
        TELCO_SCHEMA.validate(df)


def test_senior_citizen_out_of_range_fails():
    df = _make_valid_df()
    df.loc[0, "Senior Citizen"] = 2
    with pytest.raises(pa.errors.SchemaError):
        TELCO_SCHEMA.validate(df)


def test_extra_columns_allowed():
    """strict=False: colunas extras nao devem falhar a validacao."""
    df = _make_valid_df()
    df["Extra Column"] = "foo"
    TELCO_SCHEMA.validate(df)  # nao deve levantar excecao
