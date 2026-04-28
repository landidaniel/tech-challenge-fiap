# Plano de Monitoramento — Churn Prediction API

**Versao:** 1.0.0
**Ultima revisao:** 2026-04-27

---

## Arquitetura de Deploy

**Modo escolhido: Real-time (API REST)**

**Justificativa:**
O negocio de retencao de clientes exige resposta imediata. Quando um cliente liga para cancelar ou acessa o portal, a equipe de CX precisa saber o risco de churn em tempo real para oferecer um beneficio no momento certo. Processamento em batch (diario) perderia essa janela de oportunidade.

**Alternativa batch (descartada):**
Adequada para relatorios diarios de risco, mas nao serve para interacoes em tempo real. Pode ser complementar ao modo real-time.

**Infraestrutura:**
```
Cliente (app/CRM)
    |
    v
Load Balancer
    |
    v
FastAPI (uvicorn) -- porta 8000
    |          |
    v          v
models/     MLflow (http://localhost:5000)
pipeline.pkl  tracking de experimentos
model.pth
meta.pkl
```

---

## Metricas de Monitoramento

### 1. Metricas de Infraestrutura (SLIs)

| Metrica | Alerta | Critico | Ferramenta |
|---------|--------|---------|------------|
| Latencia p95 | > 200ms | > 500ms | Middleware X-Latency-Ms |
| Latencia p99 | > 500ms | > 1s | Middleware X-Latency-Ms |
| Taxa de erro HTTP 5xx | > 1% | > 5% | Access log |
| Disponibilidade | < 99.5% | < 99% | Health check |

**SLO:** p95 < 200ms, disponibilidade > 99.5% (medido em janela de 30 dias)

### 2. Metricas de Modelo (Data/Concept Drift)

| Metrica | Frequencia | Alerta | Acao |
|---------|------------|--------|------|
| Distribuicao de `churn_probability` | Diaria | PSI > 0.2 | Investigar drift |
| Taxa de churn predito (% label=True) | Diaria | +/- 10pp vs baseline | Revisar threshold |
| Feature drift (tenure, monthly_charges) | Semanal | KS-test p < 0.05 | Re-treinar |
| AUC-ROC em janela rolante (30 dias) | Mensal | < 0.80 | Re-treinar obrigatorio |
| Recall em janela rolante | Mensal | < 0.85 | Ajustar threshold |

**Baseline de referencia (teste):**
- Churn rate predito: ~68% (threshold=0.059)
- Distribuicao de probabilidades: media=0.38, std=0.25
- AUC-ROC: 0.853

### 3. Metricas de Negocio

| Metrica | Frequencia | Responsavel |
|---------|------------|-------------|
| Taxa de conversao de campanhas de retencao | Mensal | Time de CX |
| Custo real de churn evitado vs previsto | Mensal | Financeiro |
| Precisao real (% alertas que churnam de fato) | Mensal | Data Science |

---

## Alertas e Playbook de Resposta

### Alerta 1: Degradacao de Latencia

**Trigger:** p95 > 200ms por 5 minutos consecutivos

**Investigacao:**
1. Verificar `X-Latency-Ms` no access log
2. Verificar CPU/memoria do servidor
3. Verificar tamanho dos batches de entrada

**Acao:**
- Escalar horizontalmente (adicionar instancias)
- Ou reduzir `max_length` do endpoint `/predict/batch`

---

### Alerta 2: PSI de Probabilidades > 0.2

**Trigger:** Population Stability Index da distribuicao de `churn_probability` ultrapassa 0.2

**Interpretacao:**
- 0.0 – 0.1: Estavel
- 0.1 – 0.2: Monitorar
- > 0.2: Drift significativo — modelo pode estar desatualizado

**Investigacao:**
1. Comparar distribuicao atual vs. distribuicao de treino
2. Identificar features com maior drift (PSI por feature)
3. Verificar se ha mudancas na base de clientes (ex: nova regiao, novo plano)

**Acao:**
- PSI 0.1–0.2: Coletar dados das ultimas 4 semanas e avaliar re-treino
- PSI > 0.2: Re-treinar obrigatorio em < 2 semanas

---

### Alerta 3: AUC-ROC < 0.80

**Trigger:** AUC-ROC calculado em janela rolante de 30 dias cai abaixo de 0.80

**Pre-requisito:** Ter rotulos reais (churn confirmado) para calcular AUC. Coletar via CRM com delay de 30-60 dias.

**Acao:**
1. Executar `scripts/train_and_save.py` com dados dos ultimos 6 meses
2. Comparar metricas novo modelo vs. atual no MLflow
3. Se novo modelo melhorar > 2pp AUC: promover para producao
4. Atualizar `models/` e reiniciar API

---

### Alerta 4: Taxa de Erro > 1%

**Trigger:** HTTP 5xx / total requests > 1% em janela de 10 minutos

**Investigacao:**
1. Verificar logs de erro (JSON structurado no stdout)
2. Checar se `models/pipeline.pkl` e `models/model.pth` existem
3. Verificar se houve deploy de novo codigo sem re-treino do modelo

**Acao:**
- Se artefatos corrompidos: restaurar backup e reiniciar
- Se erro de codigo: rollback para versao anterior

---

## Procedimento de Re-treinamento

```bash
# 1. Coletar dados atualizados em data/
# 2. Treinar novo modelo
python scripts/train_and_save.py \
    --data data/Telco_customer_churn_updated.xlsx \
    --epochs 150

# 3. Comparar no MLflow (http://localhost:5000)
# 4. Se aprovado, a API carrega automaticamente no proximo restart
# 5. Reiniciar API
kill $(lsof -ti:8000)
make run

# 6. Verificar health
curl http://localhost:8000/health
```

---

## Cadencia de Revisao

| Atividade | Frequencia | Responsavel |
|-----------|------------|-------------|
| Review de metricas de infraestrutura | Diaria (automatica) | Ops |
| Review de drift de features | Semanal | Data Science |
| Avaliacao de re-treinamento | Mensal | Data Science + Negocio |
| Revisao do Model Card | A cada re-treinamento | Data Science |
| Auditoria de fairness | Semestral | Data Science + Legal |
