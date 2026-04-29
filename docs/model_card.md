# Model Card — ChurnMLP

**Versao:** 1.0.0
**Data:** 2026-04-27
**Equipe:** FIAP Postech — Tech Challenge Fase 1

---

## Descricao do Modelo

Rede neural MLP (Multi-Layer Perceptron) treinada com PyTorch para classificacao binaria de churn em clientes de telecomunicacoes. Prediz a probabilidade de um cliente cancelar o servico no proximo ciclo.

**Tipo:** Classificacao binaria supervisionada
**Framework:** PyTorch 2.x
**Arquivo de artefatos:** `models/model.pth`, `models/pipeline.pkl`, `models/meta.pkl`

---

## Dados de Treinamento

**Dataset:** Telco Customer Churn (IBM) — publico
**Fonte:** IBM Sample Data Sets
**Registros:** 7.043 clientes
**Features:** 30 (apos encoding de variaveis categoricas)
**Alvo:** `Churn Label` (Yes/No) -> binario 0/1
**Distribuicao:** 26.5% churn (desbalanceado, pos_weight=2.77)

**Split:**
- Treino: 4.932 amostras (70%)
- Validacao: 1.054 amostras (15%)
- Teste: 1.057 amostras (15%)
- Estratificado por `Churn` em todos os splits

**Variaveis removidas (leakage):**
- `Churn Value`, `Churn Score`, `CLTV` — derivadas do alvo
- `CustomerID`, `City`, `Zip Code` — identificadores sem valor preditivo

---

```
Input (30) -> Skip Connection (Linear Projection) ────────┐
           -> Linear(256) -> BN -> ReLU -> Dropout(var) ──> (+)
           -> Linear(128) -> BN -> ReLU -> Dropout(var)
           -> Linear(64)  -> BN -> ReLU -> Dropout(var)
           -> Linear(1)   -> logit
```

**Arquitetura:**
- **Residual Connections:** Implementada conexão residual da entrada para a saída da primeira camada oculta para melhorar a convergência.
- **Layer-wise Dropout:** Suporte a taxas diferenciadas por camada para melhor regularização.
- **Inicialização He/Kaiming:** Aplicada a todas as camadas lineares (nonlinearity='relu').

**Treinamento:**
- Otimizador: Adam (lr=1e-3, weight_decay=1e-4)
- Loss: BCEWithLogitsLoss (pos_weight=2.77)
- Scheduler: ReduceLROnPlateau (fator=0.5, patience=5)
- Early Stopping: patience=15 epochs sem melhora de val_loss
- Seed: 42 (reproducibilidade)
- Epochs treinados: ~50 (configuração atual)

---

## Performance

### Metricas no conjunto de teste (threshold otimo = 0.059)

| Metrica | Valor |
|---------|-------|
| Accuracy | 0.466 |
| Precision | 0.331 |
| Recall | **1.000** |
| F1 | 0.498 |
| AUC-ROC | **0.853** |
| PR-AUC | 0.650 |

### Comparativo com baselines (threshold=0.5)

| Modelo | F1 | AUC-ROC | Recall |
|--------|----|---------|--------|
| DummyClassifier | 0.000 | 0.500 | 0.000 |
| Regressao Logistica | 0.573 | 0.843 | 0.541 |
| Random Forest | 0.543 | 0.830 | 0.489 |
| Gradient Boosting | 0.566 | 0.845 | 0.521 |
| **MLP (threshold otimo)** | 0.498 | **0.853** | **1.000** |

### Validação Estatística

Para garantir que a superioridade da MLP não seja fruto do acaso, realizamos um **Teste T de Student Pareado** (K-fold cross-validation, utilizando 10-folds) comparando o AUC-ROC da MLP contra o Baseline de Regressão Logística.

- **P-valor:** 0.18876 (Alfa = 0.05)
- **Conclusão:** Falhamos em rejeitar a hipótese nula (H0). Não há evidência estatística de que a MLP seja superior à Regressão Logística em termos de AUC-ROC. Seguindo a **Navalha de Occam**, recomenda-se o uso do modelo Baseline por sua menor complexidade e desempenho equivalente.

### Analise de custo de negocio

- Custo Falso Negativo (cliente perdido): R$ 4.400 (CLTV medio)
- Custo Falso Positivo (campanha desnecessaria): R$ 65 (mensalidade media)
- Threshold=0.5 gera custo total estimado: ~R$ 226.000
- Threshold=0.059 reduz custo para ~R$ 37.000 (**economia de 83.6%**)

---

## Limitacoes

1. **Dados de uma unica empresa:** Treinado exclusivamente no dataset IBM Telco. Desempenho pode degradar em operadoras com perfis demograficos ou de servico diferentes.

2. **Variaveis geograficas removidas:** City, State, Latitude/Longitude foram descartadas para evitar bias geografico, mas podem conter sinal util em datasets mais amplos.

3. **Threshold muito agressivo:** O threshold otimizado (0.059) produz alto recall mas baixa precisao (33%). Em cenarios com custo de campanha maior, o threshold ideal e diferente.

4. **Sem dados temporais:** O modelo trata cada cliente como uma observacao estatica. Nao captura padroes de evolucao no tempo (churn gradual vs. abrupto).

5. **Classes desbalanceadas:** 26.5% positivos. O pos_weight mitiga o problema, mas modelos em datasets mais desbalanceados precisam de revisao.

6. **Sem validacao em producao:** Nao foram coletados dados de drift pos-deploy. O plano de monitoramento em `docs/monitoring_plan.md` define como detectar degradacao.

---

## Vieses Conhecidos

- **Genero:** Distribuicao balanceada no dataset (Male/Female). Testado: diferenca de recall entre generos < 2%. Considerado aceitavel.
- **Senior Citizen:** Clientes idosos (16% do dataset) tem churn rate 42% vs 23% dos demais. O modelo captura esse padrao corretamente — nao e vies, e sinal real.
- **Regiao geografica:** Removida do modelo. Nao ha discriminacao por localizacao.

---

## Cenarios de Falha

| Cenario | Impacto | Mitigacao |
|---------|---------|-----------|
| Entrada com coluna ausente | Pipeline preenche com 0 (get_dummies alignment) | Validacao Pydantic na API |
| Distribuicao de entrada muito diferente do treino | Queda de AUC | Monitorar feature drift (ver monitoring_plan.md) |
| `Total Charges` com virgula no lugar de ponto | ValueError no parse | API valida tipo float |
| Modelo desatualizado (> 6 meses) | Degradacao de performance | Re-treinar com dados recentes |
| Servidor sem GPU | Inferencia mais lenta (~50ms vs ~5ms) | Aceitavel para batch; usar cache para real-time |

---

## Uso Pretendido

**Aplicacoes recomendadas:**
- Identificacao proativa de clientes com risco de churn para acoes de retencao
- Priorizacao de equipes de CX (Customer Experience)
- Simulacao de impacto financeiro de campanhas de retencao

**Aplicacoes nao recomendadas:**
- Decisoes automaticas de cancelamento de contratos sem revisao humana
- Uso em dominios fora de telecomunicacoes sem re-treinamento
- Sistemas de credito ou seguros (requer fairness audit adicional)

---

## Contato

Projeto: FIAP Postech — Engenharia de Machine Learning
Repositorio: tech-challenge-fiap
