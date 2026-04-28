# Arquitetura de Deploy — Churn Prediction

**Versao:** 1.0.0
**Data:** 2026-04-27

---

## Decisao: Real-time via API REST

**Modo escolhido: Real-time (API REST — FastAPI + uvicorn)**

---

## Justificativa

O negocio de retencao de clientes exige resposta no momento da interacao.
Quando um cliente liga para cancelar ou acessa o portal de autoatendimento,
a equipe de CX precisa do score de churn **imediatamente** para decidir qual
beneficio oferecer. Uma janela de decisao de segundos pode ser a diferenca
entre reter ou perder o cliente.

Processamento em batch (ex: job noturno) entregaria scores com delay de ate
24 horas — inutil para interacoes em tempo real.

---

## Comparativo: Real-time vs Batch

| Criterio | Real-time (escolhido) | Batch |
|---|---|---|
| Latencia de resposta | < 200ms (p95) | Horas / dia seguinte |
| Caso de uso | Atendimento, portal, app | Relatorios, campanhas diarias |
| Complexidade operacional | Media (API + modelo em memoria) | Baixa (job agendado) |
| Custo computacional | Baixo (CPU, modelo pequeno) | Baixo |
| Escala | Horizontal (multiplas instancias) | Vertical (mais CPU/RAM) |
| Adequacao ao problema | Alta — CX em tempo real | Parcial — apenas campanhas |

**Conclusao:** Real-time e a unica arquitetura que atende o caso de uso
principal (retencao no momento da interacao). Batch pode ser usado de forma
complementar para campanhas proativas agendadas.

---

## Arquitetura da Solucao

```
┌─────────────────────────────────────────────────────┐
│                   CLIENTE                            │
│         (App CRM / Portal / Sistema interno)         │
└──────────────────────┬──────────────────────────────┘
                       │  POST /predict (JSON)
                       ▼
┌─────────────────────────────────────────────────────┐
│               LOAD BALANCER                          │
│         (nginx / AWS ALB / GCP Load Balancer)        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│            FastAPI (uvicorn)  :8000                  │
│                                                      │
│  LatencyLoggingMiddleware                            │
│    └─ loga latencia, X-Request-ID, status HTTP       │
│                                                      │
│  POST /predict ──► PredictRequest (Pydantic)         │
│    └─► TelcoEncoder ──► StandardScaler               │
│          └─► ChurnMLP.predict_proba()                │
│                └─► PredictResponse (JSON)            │
│                                                      │
│  GET  /health  ──► HealthResponse                    │
│  POST /predict/batch ──► ate 1.000 clientes          │
└──────┬───────────────────────┬──────────────────────┘
       │                       │
       ▼                       ▼
┌─────────────┐     ┌──────────────────────┐
│  models/    │     │  MLflow Server :5000  │
│ pipeline.pkl│     │  (tracking de        │
│ model.pth   │     │   experimentos e     │
│ meta.pkl    │     │   artefatos)         │
└─────────────┘     └──────────────────────┘
```

---

## Componentes

| Componente | Tecnologia | Responsabilidade |
|---|---|---|
| API | FastAPI + uvicorn | Serve predicoes, valida entrada, loga requisicoes |
| Preprocessamento | sklearn Pipeline (TelcoEncoder + StandardScaler) | Encoding + normalizacao |
| Modelo | ChurnMLP (PyTorch) | Inferencia — P(churn) por cliente |
| Validacao de entrada | Pydantic v2 | Rejeita payloads invalidos antes do modelo |
| Logging | JsonFormatter (stdout) | Logs estruturados para ingestao em observabilidade |
| Rastreamento | MLflow 3.11.1 + PostgreSQL | Historico de experimentos e versoes |

---

## Fluxo de Inferencia

```
1. Cliente envia POST /predict com features do cliente (JSON)
2. Pydantic valida os campos (tipos, dominios, campos obrigatorios)
3. CustomerFeatures.to_dataframe() converte para DataFrame de 1 linha
4. pipeline.transform() aplica TelcoEncoder + StandardScaler
5. ChurnMLP.predict_proba() retorna P(churn) em [0, 1]
6. Compara com threshold otimo (0.059) -> churn_label True/False
7. Retorna PredictResponse com probabilidade, label e confianca
8. Middleware loga: metodo, path, status, latencia, request_id
```

---

## SLOs (Service Level Objectives)

| Metrica | Objetivo |
|---|---|
| Disponibilidade | >= 99,5% (janela 30 dias) |
| Latencia p95 | < 200ms |
| Latencia p99 | < 500ms |
| Taxa de erro 5xx | < 0,5% |

---

## Decisoes de Infraestrutura

**Por que CPU e nao GPU?**
O modelo tem apenas 50.049 parametros. Inferencia em CPU leva ~5ms por
requisicao — bem abaixo do SLO de 200ms. GPU adicionaria custo e
complexidade sem ganho real nesse porte.

**Por que uvicorn e nao gunicorn puro?**
uvicorn e assincrono (ASGI), ideal para FastAPI. Em producao, o padrao e
`gunicorn -k uvicorn.workers.UvicornWorker` para multiplos workers com
gerenciamento de processos robusto.

**Escalonamento horizontal:**
Como o modelo e stateless (carregado em memoria, sem escrita), multiplas
instancias podem rodar em paralelo atras de um load balancer sem
necessidade de sessao compartilhada.

---

## Deploy em Nuvem (Opcional — Bonus)

Para deploy em AWS/GCP/Azure, o caminho mais direto e:

```
# Dockerfile (exemplo)
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
RUN python scripts/train_and_save.py --no-mlflow
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Servicos sugeridos:
- **AWS:** ECS Fargate + ALB + ECR
- **GCP:** Cloud Run (serverless, escala para zero)
- **Azure:** Container Apps
