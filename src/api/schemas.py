"""Schemas Pydantic para request/response da API de Churn."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    """Campos de um cliente Telco para predicao de churn.

    Todos os campos correspondem ao dataset bruto apos remocao de leakage.
    Valores ausentes sao tratados pelo pipeline de preprocessamento.
    """

    # Identificacao (opcional, nao usada no modelo)
    customer_id: Optional[str] = Field(default=None, description="ID do cliente")

    # Demografico
    gender: str = Field(..., description="Male ou Female")
    senior_citizen: int = Field(..., ge=0, le=1, description="1 se idoso, 0 caso contrario")
    partner: str = Field(..., description="Yes ou No")
    dependents: str = Field(..., description="Yes ou No")

    # Contrato
    tenure: int = Field(..., ge=0, description="Meses como cliente")
    contract: str = Field(..., description="Month-to-month | One year | Two year")
    paperless_billing: str = Field(..., description="Yes ou No")
    payment_method: str = Field(
        ...,
        description=(
            "Electronic check | Mailed check | "
            "Bank transfer (automatic) | Credit card (automatic)"
        ),
    )
    monthly_charges: float = Field(..., ge=0, description="Cobranca mensal em R$")
    total_charges: float = Field(..., ge=0, description="Total cobrado em R$")

    # Servicos de telefonia
    phone_service: str = Field(..., description="Yes ou No")
    multiple_lines: str = Field(
        ..., description="Yes | No | No phone service"
    )

    # Servicos de internet
    internet_service: str = Field(
        ..., description="DSL | Fiber optic | No"
    )
    online_security: str = Field(
        ..., description="Yes | No | No internet service"
    )
    online_backup: str = Field(
        ..., description="Yes | No | No internet service"
    )
    device_protection: str = Field(
        ..., description="Yes | No | No internet service"
    )
    tech_support: str = Field(
        ..., description="Yes | No | No internet service"
    )
    streaming_tv: str = Field(
        ..., description="Yes | No | No internet service"
    )
    streaming_movies: str = Field(
        ..., description="Yes | No | No internet service"
    )

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in {"Male", "Female"}:
            raise ValueError("gender deve ser 'Male' ou 'Female'")
        return v

    @field_validator("partner", "dependents", "phone_service", "paperless_billing")
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        if v not in {"Yes", "No"}:
            raise ValueError(f"Campo deve ser 'Yes' ou 'No', recebido: {v!r}")
        return v

    def to_dataframe(self) -> "pd.DataFrame":  # noqa: F821
        """Converte o schema para DataFrame de uma linha (sem customer_id)."""
        import pandas as pd

        data = self.model_dump(exclude={"customer_id"})
        # Renomeia para os nomes originais do dataset
        rename_map = {
            "gender": "Gender",
            "senior_citizen": "Senior Citizen",
            "partner": "Partner",
            "dependents": "Dependents",
            "tenure": "Tenure Months",
            "contract": "Contract",
            "paperless_billing": "Paperless Billing",
            "payment_method": "Payment Method",
            "monthly_charges": "Monthly Charges",
            "total_charges": "Total Charges",
            "phone_service": "Phone Service",
            "multiple_lines": "Multiple Lines",
            "internet_service": "Internet Service",
            "online_security": "Online Security",
            "online_backup": "Online Backup",
            "device_protection": "Device Protection",
            "tech_support": "Tech Support",
            "streaming_tv": "Streaming TV",
            "streaming_movies": "Streaming Movies",
        }
        renamed = {rename_map.get(k, k): v for k, v in data.items()}
        return pd.DataFrame([renamed])


class PredictResponse(BaseModel):
    """Resposta da API para uma predicao de churn."""

    customer_id: Optional[str] = None
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_label: bool
    threshold: float
    confidence: str = Field(..., description="high | medium | low")


class BatchPredictRequest(BaseModel):
    """Requisicao para predicao em lote."""

    customers: list[CustomerFeatures] = Field(
        ..., min_length=1, max_length=1000
    )


class BatchPredictResponse(BaseModel):
    """Resposta para predicao em lote."""

    predictions: list[PredictResponse]
    total: int


class HealthResponse(BaseModel):
    """Resposta do endpoint /health."""

    status: str
    model_loaded: bool
    version: str
    threshold: Optional[float] = None
