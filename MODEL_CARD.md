# Model Card — Churn MLP v1.0.0 (Grupo Tech Challenge FIAP)

## 1. Model Details
- **Versão:** 1.0.0
- **Data de treino:** 2026-04-25
- **Arquitetura:** MLP (Rede Neural Multicamadas) treinada com PyTorch.
- **Estrutura:** Camadas [N → 64 → 32 → 1], ativação ReLU, Dropout de 0.3 e Batch Normalization.
- **Licença:** MIT

## 2. Intended Use (Uso Pretendido)
- **Primário:** Estimar a probabilidade de cancelamento (churn) em um horizonte de 30 dias para clientes ativos da operadora de Telecom.
- **Usuário-alvo:** Times de Customer Success e Retenção.
- **Fora de Escopo:** Não deve ser usado para decisões de crédito, cobrança agressiva ou em clientes com menos de 3 meses de histórico.

## 3. Training & Evaluation Data
- **Dataset:** Telco Customer Churn (IBM/Kaggle).
- **Split:** Temporal (Treino: meses 1-9 / Teste: meses 11-12) para evitar vazamento de dados.
- **Pré-processamento:** Escalonamento com StandardScaler e codificação de variáveis categóricas via One-Hot Encoding.

## 4. Metrics (Performance)
Estes são os resultados obtidos no MLflow durante a Etapa 2:

| Métrica | Valor | Observação |
| :--- | :--- | :--- |
| **Recall** | **0.829** | Foco principal do projeto para minimizar Falsos Negativos. |
| **AUC-ROC** | **0.854** | Capacidade de separação entre as classes. |
| **Economia Projetada** | **R$ 186.414,00** | Baseado na análise de custo ajustando o threshold operacional. |

**Threshold Operacional:** Foi definido um corte de **0.059** para atingir o equilíbrio financeiro e maximizar a recuperação de receita.

## 5. Fairness (Considerações Éticas)
- O modelo foi avaliado por subgrupos de gênero e idade.
- **Limitação identificada:** Foi observado um gap de performance em clientes acima de 50 anos, recomendando-se uma revisão de fairness antes do deploy em larga escala para evitar vieses etários.

## 6. Caveats and Recommendations (Ressalvas)
- O modelo não foi testado em períodos de sazonalidade extrema (ex: promoções de Black Friday).
- **Principais Preditores:** O modelo baseia-se fortemente em `Tenure` (tempo de casa), `TotalCharges` e tipo de contrato.
- **Recomendação de Retreino:** A cada 3 meses ou caso a distribuição das features (Data Drift) mude significativamente, impactando a estabilidade do modelo.