"""Smoke tests para a API FastAPI de churn (sem modelo carregado).

Testa:
- GET /health retorna 200 com campos corretos
- POST /predict retorna 503 quando modelo nao carregado
- POST /predict retorna 422 com payload invalido
- POST /predict/batch retorna 503 quando modelo nao carregado
- Validacao Pydantic: campo obrigatorio ausente gera 422
- Middleware adiciona header X-Latency-Ms
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Payload valido de exemplo (sem modelo carregado -> 503)
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
    "customer_id": "test-001",
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 65.0,
    "total_charges": 780.0,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_response_has_required_fields(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_model_not_loaded_without_artifacts(self):
        data = client.get("/health").json()
        # Em ambiente de teste sem artefatos, modelo nao deve estar carregado
        assert data["model_loaded"] is False

    def test_health_status_degraded_without_model(self):
        data = client.get("/health").json()
        assert data["status"] == "degraded"


# ---------------------------------------------------------------------------
# /predict — sem modelo
# ---------------------------------------------------------------------------
class TestPredictNoModel:
    def test_predict_returns_503_when_model_not_loaded(self):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 503

    def test_predict_error_detail_present(self):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert "detail" in r.json()


# ---------------------------------------------------------------------------
# /predict — validacao Pydantic (422)
# ---------------------------------------------------------------------------
class TestPredictValidation:
    def test_missing_required_field_returns_422(self):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "gender"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_invalid_gender_returns_422(self):
        payload = {**VALID_PAYLOAD, "gender": "Other"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_negative_tenure_returns_422(self):
        payload = {**VALID_PAYLOAD, "tenure": -1}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_senior_citizen_out_of_range_returns_422(self):
        payload = {**VALID_PAYLOAD, "senior_citizen": 5}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_invalid_yes_no_field_returns_422(self):
        payload = {**VALID_PAYLOAD, "partner": "Maybe"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /predict/batch — sem modelo
# ---------------------------------------------------------------------------
class TestBatchPredictNoModel:
    def test_batch_predict_returns_503(self):
        r = client.post("/predict/batch", json={"customers": [VALID_PAYLOAD]})
        assert r.status_code == 503

    def test_batch_empty_list_returns_422(self):
        r = client.post("/predict/batch", json={"customers": []})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
class TestMiddleware:
    def test_latency_header_present(self):
        r = client.get("/health")
        assert "x-latency-ms" in r.headers

    def test_request_id_header_present(self):
        r = client.get("/health")
        assert "x-request-id" in r.headers

    def test_request_id_is_uuid_format(self):
        import uuid
        r = client.get("/health")
        request_id = r.headers.get("x-request-id", "")
        # Deve ser um UUID valido
        uuid.UUID(request_id)
