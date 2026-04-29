# Churn Prediction — FIAP Postech Tech Challenge

Pipeline de ML end-to-end para previsao de churn de clientes Telco.
Rede neural MLP (PyTorch) comparada com baselines Scikit-Learn, servida via API FastAPI e rastreada no MLflow.

## Estrutura do Projeto

```
tech-challenge-fiap/
├── data/                          # Dataset bruto (Telco Customer Churn - IBM)
├── docs/
│   ├── model_card.md              # Model Card (performance, limitacoes, vieses)
│   └── monitoring_plan.md         # Plano de monitoramento em producao
├── models/                        # Artefatos treinados (pipeline.pkl, model.pth, meta.pkl)
├── notebooks/
│   ├── Etapa_1.ipynb              # EDA + ML Canvas + Baselines
│   └── Etapa_2.ipynb              # MLP PyTorch + comparacao de modelos
├── scripts/
│   ├── train_and_save.py          # Treina o modelo e salva artefatos
│   └── visualize_results.py       # Gera gráficos de prova real (Médias Reais vs Preditas)
├── src/
│   ├── churn/                     # Pacote de modelagem
│   │   ├── config.py              # Constantes globais
│   │   ├── preprocessing.py       # TelcoEncoder (sklearn-compativel)
│   │   ├── model.py               # ChurnMLP (nn.Module) com Residual Connections
│   │   ├── train.py               # Loop de treinamento + early stopping
│   │   ├── evaluate.py            # Metricas + threshold otimo
│   │   └── pipeline.py            # Pipeline sklearn + save/load artefatos
│   ├── evaluation/                # Avaliação estatística
│   │   └── hypothesis_test.py     # Teste T pareado (MLP vs Baseline) com K-Fold em 10-fold
│   └── api/
│       ├── main.py                # FastAPI app (/health, /predict, /predict/batch)
│       ├── schemas.py             # Modelos Pydantic
│       ├── middleware.py          # Middleware de latencia
│       └── logging_config.py      # Logging estruturado (JSON)
├── tests/
│   ├── test_preprocessing.py      # Testes unitarios (TelcoEncoder)
│   ├── test_model.py              # Testes unitarios (ChurnMLP)
│   ├── test_schema.py             # Validacao de schema (pandera)
│   └── test_api.py                # Smoke tests (FastAPI)
├── Makefile                       # Comandos: lint, test, run, train
└── pyproject.toml                 # Dependencias, linting (ruff), pytest
```

## Setup

### Requisitos

- Python 3.11+
- PyTorch (CPU ou CUDA)

### Instalacao

```bash
# Clone o repositorio
git clone <url-do-repositorio>
cd tech-challenge-fiap

# Instala dependencias (incluindo dev)
make install
# ou manualmente:
pip install -e ".[dev]"

# Configure o PYTHONPATH (necessário para reconhecer o pacote 'src')
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Dataset

Baixe o dataset **Telco Customer Churn (IBM)** e coloque em `data/Telco_customer_churn.xlsx`.

## Execucao

### 1. Treinar o modelo

```bash
make train
# ou com parametros customizados:
python scripts/train_and_save.py \
    --data data/Telco_customer_churn.xlsx \
    --hidden 256,128,64 \
    --epochs 150 \
    --no-mlflow
```

Artefatos gerados em `models/`: `pipeline.pkl`, `model.pth`, `meta.pkl`.

### 2. Subir a API

```bash
make run
# API disponivel em http://localhost:8000
# Docs interativas: http://localhost:8000/docs
```

### 3. Rodar os testes

```bash
make test         # com cobertura (htmlcov/)
make test-fast    # sem cobertura
```

### 4. Linting

```bash
make lint     # ruff check
make format   # black

### 5. Teste de Hipótese e Visualização

Para validar a superioridade estatística do modelo MLP contra o baseline e gerar provas reais visuais:

```bash
# Roda o teste T com K-Fold pareado em 10 folds
python src/evaluation/hypothesis_test.py

# Gera o gráfico de Médias Observadas vs Probabilidades do Modelo
python scripts/visualize_results.py
```
```

## Endpoints da API

| Metodo | Endpoint | Descricao |
|--------|----------|-----------|
| GET | `/health` | Status da API e modelo carregado |
| POST | `/predict` | Predicao para um cliente |
| POST | `/predict/batch` | Predicao em lote (ate 1.000 clientes) |
| GET | `/docs` | Swagger UI |

### Exemplo — POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 2,
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 75.0,
    "total_charges": 150.0,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes"
  }'
```

Resposta:

```json
{
  "churn_probability": 0.6399,
  "churn_label": true,
  "threshold": 0.059,
  "confidence": "high"
}
```

## Arquitetura do Modelo

O modelo utiliza uma arquitetura MLP com **Skip Connections** (conexões residuais) para facilitar o fluxo de gradiente e **Inicialização Kaiming (He)** para estabilidade.

```
Input (30 features)
  ├─> Skip Connection (Linear Projection) ────────┐
  └─> [Linear(256) -> BN -> ReLU -> Dropout] ──> (+) ──>
      -> [Linear(128) -> BN -> ReLU -> Dropout] ───────>
      -> [Linear(64)  -> BN -> ReLU -> Dropout] ───────>
      -> Linear(1) -> logit -> sigmoid -> P(churn)
```

**Estrategia de treinamento:**
- Otimizador: Adam + weight_decay=1e-4 (L2)
- Loss: BCEWithLogitsLoss com pos_weight=2.77 (desbalanceamento)
- LR Decay: ReduceLROnPlateau (fator 0.5, patience=5)
- Early Stopping: patience=15, restaura melhor checkpoint

**Threshold otimo:** 0.059 (minimiza custo FN x R$4.400 + FP x R$65)

## Resultados

| Modelo | Accuracy | Recall | F1 | AUC-ROC |
|--------|----------|--------|----|---------|
| DummyClassifier | 0.735 | 0.000 | 0.000 | 0.500 |
| Regressao Logistica | 0.803 | 0.541 | 0.573 | 0.843 |
| Random Forest | 0.791 | 0.489 | 0.543 | 0.830 |
| Gradient Boosting | 0.800 | 0.521 | 0.566 | 0.845 |
| **MLP PyTorch** | **0.466** | **1.000** | **0.498** | **0.853** |

> **Validação Estatística:** O Teste T Pareado (10-fold) resultou em um p-valor de **0.188**, indicando que não há diferença estatisticamente significativa entre a MLP e o Baseline (Regressão Logística). Seguindo a **Navalha de Occam**, o baseline é recomendado para produção devido à sua simplicidade e desempenho equivalente. O uso da MLP com threshold otimizado permanece útil para cenários onde se deseja maximizar o recall (100%) priorizando a redução do custo de perda de clientes (CLTV).

## MLflow

O tracking server roda via Docker (PostgreSQL backend):

```bash
# Na pasta do docker-compose:
docker compose up -d

# Acesse: http://localhost:5000
```

## Variaveis de Ambiente

| Variavel | Padrao | Descricao |
|----------|--------|-----------|
| `ARTIFACTS_DIR` | `models/` | Caminho para artefatos do modelo |
| `THRESHOLD` | valor do meta.pkl | Override do threshold de decisao |
| `LOG_LEVEL` | `INFO` | Nivel de logging |
| `PORT` | `8000` | Porta da API (Makefile) |
| `PYTHONPATH` | `.` | Garante que a pasta `src` seja reconhecida como um pacote |
