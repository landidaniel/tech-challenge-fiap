FROM python:3.11-slim

WORKDIR /app

# Instala dependencias do sistema necessarias para torch e openpyxl
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copia manifesto primeiro para aproveitar cache de camadas
COPY pyproject.toml ./
COPY src/ ./src/
COPY models/ ./models/

# Instala dependencias do projeto (sem extras de dev)
RUN pip install --no-cache-dir -e .

# Copia restante do codigo
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
