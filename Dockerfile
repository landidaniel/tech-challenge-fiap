FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.4 CPU-only — compativel com numpy 2.x
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Demais dependencias
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "numpy>=2.0" \
    pandas \
    scikit-learn \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    openpyxl \
    mlflow

COPY src/ ./src/
COPY models/ ./models/

RUN pip install --no-cache-dir -e . --no-deps

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
