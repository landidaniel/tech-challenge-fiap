# Churn Prediction — FIAP Postech Tech Challenge

Pipeline de ML end-to-end para previsão de churn de clientes de telecomunicações.
Rede neural MLP com PyTorch, comparada a baselines Scikit-Learn via teste estatístico, servida via API FastAPI e rastreada no MLflow.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Arquitetura do Modelo](#arquitetura-do-modelo)
- [Setup e Instalação](#setup-e-instalação)
- [Execução](#execução)
- [Endpoints da API](#endpoints-da-api)
- [Resultados e Validação Estatística](#resultados-e-validação-estatística)
- [Threshold e Lógica de Negócio](#threshold-e-lógica-de-negócio)
- [Monitoramento e Deploy](#monitoramento-e-deploy)
- [Variáveis de Ambiente](#variáveis-de-ambiente)
- [Testes](#testes)

---

## Visão Geral

O projeto resolve um problema de **classificação binária**: prever quais clientes de uma operadora de telecomunicações têm probabilidade de cancelar o serviço (churn), de forma que a empresa possa agir preventivamente com campanhas de retenção.

**Dataset:** [Telco Customer Churn (IBM)](https://www.kaggle.com/blastchar/telco-customer-churn)
— 7.043 clientes, 26,5% de churn, 30 features após remoção de colunas com data leakage.

**Stack:**
- Modelagem: PyTorch 2.0+, Scikit-Learn
- Serviço: FastAPI + Uvicorn
- Rastreamento: MLflow
- Deploy: Docker + Render.com

---

## Estrutura do Projeto

```
tech-challenge-fiap/
├── data/
│   └── Telco_customer_churn.xlsx       # Dataset bruto (IBM)
├── docs/
│   ├── model_card.md                   # Model Card (performance, limitações, vieses)
│   ├── monitoring_plan.md              # Plano de monitoramento em produção
│   └── deploy_architecture.md          # Arquitetura de deploy
├── models/                             # Artefatos treinados (versionados no git)
│   ├── pipeline.pkl                    # sklearn Pipeline (TelcoEncoder + StandardScaler)
│   ├── model.pth                       # Pesos do ChurnMLP (state_dict)
│   └── meta.pkl                        # Metadados (input_dim, hidden_dims, threshold, ...)
├── notebooks/
│   ├── Etapa_1.ipynb                   # EDA, ML Canvas, modelos baseline
│   └── Etapa_2.ipynb                   # Treinamento MLP PyTorch + comparação
├── scripts/
│   ├── train_and_save.py               # Script principal de treino (CLI)
│   └── visualize_results.py            # Gráficos de médias observadas vs preditas
├── src/
│   ├── churn/
│   │   ├── config.py                   # Constantes globais (seeds, caminhos, custos)
│   │   ├── preprocessing.py            # TelcoEncoder (sklearn-compatível)
│   │   ├── model.py                    # ChurnMLP (nn.Module com skip connections)
│   │   ├── train.py                    # Loop de treinamento + early stopping
│   │   ├── evaluate.py                 # Métricas + otimização de threshold
│   │   └── pipeline.py                 # Pipeline sklearn + save/load de artefatos
│   ├── churn_baseline/
│   │   └── model.py                    # Baseline: Regressão Logística
│   ├── evaluation/
│   │   └── hypothesis_test.py          # Teste T pareado (MLP vs Baseline, K-Fold)
│   └── api/
│       ├── main.py                     # FastAPI app (/health, /predict, /predict/batch)
│       ├── schemas.py                  # Modelos Pydantic (entrada/saída)
│       ├── middleware.py               # Middleware de latência e request ID
│       └── logging_config.py           # Logging estruturado (JSON)
├── tests/
│   ├── test_preprocessing.py           # TelcoEncoder
│   ├── test_model.py                   # ChurnMLP
│   ├── test_schema.py                  # Schemas Pydantic
│   └── test_api.py                     # Smoke tests da API
├── Dockerfile                          # Imagem Docker de produção
├── render.yaml                         # Configuração Render.com
├── Makefile                            # Comandos de desenvolvimento
└── pyproject.toml                      # Dependências, linting (ruff), pytest
```

---

## Arquitetura do Modelo

### ChurnMLP

O modelo é uma rede neural MLP com **skip connections** (conexões residuais) do input para a primeira camada oculta, facilitando o fluxo de gradiente e prevenindo degradação durante o treinamento.

```
Input (30 features)
  ├─> Skip Connection: Linear(30 → 256) ─────────────────────────┐
  └─> Linear(30 → 256) → BatchNorm1d → ReLU → Dropout           (+)
                                                                   │
                          Linear(256 → 128) → BatchNorm1d → ReLU → Dropout
                          Linear(128 → 64)  → BatchNorm1d → ReLU → Dropout
                          Linear(64 → 1)    → logit → sigmoid → P(churn)
```

**Técnicas aplicadas:**
- **Skip connections:** gradientes fluem diretamente do input sem degradar
- **BatchNorm1d:** estabilidade e convergência mais rápida
- **Inicialização Kaiming (He):** adequada para ativações ReLU
- **Dropout por camada:** regularização configurável (lista ou valor único)

### Estratégia de Treinamento

| Hiperparâmetro | Valor |
|---|---|
| Otimizador | Adam (lr=1e-3, weight_decay=1e-4) |
| Loss | BCEWithLogitsLoss (pos_weight=2.77) |
| LR Scheduler | ReduceLROnPlateau (fator=0.5, patience=5) |
| Early Stopping | patience=15, restaura melhor checkpoint |
| Split | 70% treino / 15% validação / 15% teste (estratificado) |
| Seed | 42 (reprodutibilidade) |

O `pos_weight=2.77` compensa o desbalanceamento de classes (26,5% churn vs 73,5% não-churn).

---

## Setup e Instalação

### Requisitos

- Python 3.11+
- PyTorch 2.0+ (CPU ou CUDA)

### Instalação

```bash
# Clone o repositório
git clone <url-do-repositorio>
cd tech-challenge-fiap

# Instala dependências (incluindo dev)
make install
# ou manualmente:
pip install -e ".[dev]"

# Configure o PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Dataset

Baixe o dataset **Telco Customer Churn (IBM)** e coloque em:
```
data/Telco_customer_churn.xlsx
```

---

## Execução

### 1. Treinar o modelo

```bash
make train

# Ou com parâmetros customizados:
python scripts/train_and_save.py \
    --data data/Telco_customer_churn.xlsx \
    --hidden 256,128,64 \
    --epochs 150 \
    --no-mlflow
```

Artefatos salvos em `models/`: `pipeline.pkl`, `model.pth`, `meta.pkl`.

### 2. Subir a API

```bash
make run
# Modo desenvolvimento com reload — http://localhost:8000
# Documentação interativa: http://localhost:8000/docs

make run-prod
# Modo produção (2 workers, sem reload)
```

### 3. Rodar o teste de hipótese

```bash
# Teste T pareado (10-fold) — MLP vs Baseline
python src/evaluation/hypothesis_test.py
```

### 4. Gerar visualizações

```bash
# Gráfico: médias observadas vs probabilidades preditas
python scripts/visualize_results.py
```

### 5. Linting e formatação

```bash
make lint     # ruff check (auto-fix)
make format   # black
```

---

## Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/health` | Status da API e do modelo carregado |
| `POST` | `/predict` | Predição para um único cliente |
| `POST` | `/predict/batch` | Predição em lote (até 1.000 clientes) |
| `GET` | `/docs` | Swagger UI (documentação interativa) |

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "0.3.0",
  "threshold": 0.059
}
```

### POST /predict

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

```json
{
  "churn_probability": 0.6399,
  "churn_label": true,
  "threshold": 0.059,
  "confidence": "high"
}
```

O campo `confidence` é calculado pela distância entre a probabilidade e o threshold:
- `high`: |prob - threshold| ≥ 0.30
- `medium`: |prob - threshold| ≥ 0.15
- `low`: caso contrário

### POST /predict/batch

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      { "gender": "Female", "senior_citizen": 0, ... },
      { "gender": "Male",   "senior_citizen": 1, ... }
    ]
  }'
```

```json
{
  "predictions": [
    { "customer_id": 0, "churn_probability": 0.64, "churn_label": true,  "threshold": 0.059, "confidence": "high" },
    { "customer_id": 1, "churn_probability": 0.12, "churn_label": true,  "threshold": 0.059, "confidence": "low"  }
  ],
  "total": 2
}
```

### Headers de resposta

Toda resposta inclui:
- `X-Request-ID`: UUID único por requisição (rastreabilidade)
- `X-Latency-Ms`: tempo de processamento em milissegundos

---

## Resultados e Validação Estatística

### Comparação de modelos (conjunto de teste, threshold padrão=0.5)

| Modelo | Accuracy | Recall | F1 | AUC-ROC |
|--------|----------|--------|----|---------|
| DummyClassifier | 0.735 | 0.000 | 0.000 | 0.500 |
| Regressão Logística | 0.803 | 0.541 | 0.573 | 0.843 |
| Random Forest | 0.791 | 0.489 | 0.543 | 0.830 |
| Gradient Boosting | 0.800 | 0.521 | 0.566 | 0.845 |
| **MLP PyTorch** | **0.466** | **1.000** | **0.498** | **0.853** |

> A acurácia baixa do MLP reflete o threshold agressivo (0.059), que prioriza recall = 100%.

### MLP com threshold otimizado (0.059)

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.466 |
| Recall | 1.000 |
| F1 | 0.498 |
| AUC-ROC | 0.853 |
| PR-AUC | 0.650 |

### Validação Estatística

O **Teste T Pareado com 10-Fold Cross-Validation** compara o AUC-ROC do MLP com o da Regressão Logística:

- **p-valor:** 0.188 (> α = 0.05)
- **Conclusão:** Não há diferença estatisticamente significativa entre os modelos.

Seguindo a **Navalha de Occam**, a Regressão Logística é recomendada para produção pela sua simplicidade e desempenho equivalente. A MLP com threshold otimizado é útil quando **recall máximo (100%)** é prioridade — cenários em que o custo de perda de cliente (CLTV) é muito superior ao custo de uma campanha de retenção desnecessária.

---

## Threshold e Lógica de Negócio

O threshold padrão (0.5) não é adequado para este problema. O threshold ótimo é calculado minimizando o custo de negócio:

```
Custo = FP × R$65 + FN × R$4.400
```

| Parâmetro | Valor |
|-----------|-------|
| Custo de falso negativo (FN) | R$4.400 (cliente perdido — CLTV) |
| Custo de falso positivo (FP) | R$65 (campanha de retenção desnecessária) |
| **Threshold ótimo** | **0.059** |
| Economia estimada vs threshold padrão | ~R$189.000 |

Com threshold=0.059, o modelo captura **100% dos clientes em risco** com um volume controlado de falsos positivos.

---

## Monitoramento e Deploy

### Arquitetura de Deploy

A API está em produção no **Render.com** via container Docker. O fluxo completo é:

```
Repositório GitHub (main)
        │
        │  push → build automático
        ▼
┌─────────────────────────────────────┐
│           Render.com                │
│                                     │
│  Docker build (Dockerfile)          │
│    └─ copia src/ + models/          │
│    └─ instala dependências          │
│                                     │
│  Web Service :8000                  │
│    └─ uvicorn src.api.main:app      │
│    └─ health check: GET /health     │
└────────────────┬────────────────────┘
                 │
                 │  HTTPS (TLS gerenciado pelo Render)
                 ▼
          Clientes / CRM / API consumers
```

Os artefatos treinados (`models/pipeline.pkl`, `models/model.pth`, `models/meta.pkl`) são **versionados no git** e copiados para dentro da imagem Docker no build, eliminando a necessidade de um model registry externo em produção.

### Deploy no Render.com

O arquivo `render.yaml` descreve o serviço:

```yaml
services:
  - type: web
    name: churn-prediction-api
    env: docker
    dockerfilePath: ./Dockerfile
    plan: free
    envVars:
      - key: ARTIFACTS_DIR
        value: models
      - key: LOG_LEVEL
        value: INFO
    healthCheckPath: /health
```

Para fazer o deploy:
1. Conecte o repositório ao Render.com
2. O Render detecta o `render.yaml` automaticamente
3. Cada push na branch `main` dispara um novo build e redeploy

### Executar localmente com Docker

```bash
# Build da imagem
docker build -t churn-api .

# Execução local
docker run -p 8000:8000 -e LOG_LEVEL=INFO churn-api
```

### Rastreamento com MLflow

```bash
# Sobe o tracking server com Docker
docker compose up -d

# Acesse: http://localhost:5000
```

### Monitoramento em Produção

Ver [`docs/monitoring_plan.md`](docs/monitoring_plan.md) para o plano completo. Resumo dos alertas:

| Métrica | Alerta |
|---------|--------|
| PSI da probabilidade de churn | > 0.20 (drift significativo) |
| AUC-ROC rolling 30 dias | < 0.80 |
| Recall rolling 30 dias | < 0.85 |
| Latência p95 | > 200ms |
| Taxa de erros HTTP 5xx | > 1% |

---

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `ARTIFACTS_DIR` | `models/` | Caminho para os artefatos do modelo |
| `THRESHOLD` | valor do `meta.pkl` | Override do threshold de decisão |
| `LOG_LEVEL` | `INFO` | Nível de logging (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `PORT` | `8000` | Porta da API |
| `PYTHONPATH` | `.` | Necessário para reconhecer o pacote `src/` |

---

## Testes

```bash
make test         # pytest com cobertura (relatório em htmlcov/)
make test-fast    # pytest sem cobertura
```

Cobertura dos testes:

| Módulo | O que é testado |
|--------|-----------------|
| `test_preprocessing.py` | TelcoEncoder: fit, transform, alinhamento de colunas, categorias não vistas |
| `test_model.py` | ChurnMLP: shapes de saída, inicialização, logits |
| `test_schema.py` | Pydantic: tipos, campos obrigatórios, constraints de valores |
| `test_api.py` | Smoke tests: /health (200), /predict (503 sem modelo), validação de entrada (422) |

---

## Documentação Adicional

| Documento | Descrição |
|-----------|-----------|
| [`docs/model_card.md`](docs/model_card.md) | Ficha do modelo: dados, performance, limitações e vieses |
| [`docs/monitoring_plan.md`](docs/monitoring_plan.md) | Estratégia de monitoramento em produção |
| [`docs/deploy_architecture.md`](docs/deploy_architecture.md) | Arquitetura de infraestrutura e deploy |
| [`notebooks/Etapa_1.ipynb`](notebooks/Etapa_1.ipynb) | EDA, ML Canvas e modelos baseline |
| [`notebooks/Etapa_2.ipynb`](notebooks/Etapa_2.ipynb) | Treinamento MLP PyTorch e comparação de modelos |
