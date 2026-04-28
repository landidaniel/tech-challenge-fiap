# Estratégia de Deploy e Arquitetura de Solução

## 1. Arquitetura Escolhida: Real-time (Online Inference)
Optamos pela arquitetura de **Inferência em Tempo Real** utilizando uma **API REST** construída com o framework **FastAPI**.

### Justificativa da Escolha:
* **Necessidade de Negócio:** No setor de Telecomunicações, o churn é um evento sensível ao tempo. A predição precisa estar disponível no momento em que o cliente interage com o Call Center ou o App, permitindo uma ação de retenção imediata.
* **Baixa Latência:** O FastAPI foi escolhido por sua alta performance e suporte a operações assíncronas, garantindo que o tempo de resposta da predição não degrade a experiência do usuário.
* **Integrabilidade:** Diferente do processamento em Batch (lote), a API permite que qualquer sistema da empresa (CRM, Web, Mobile) consuma o modelo de forma padronizada via JSON.

## 2. Estrutura Técnica (Mapeamento do Repositório)
A implementação desta arquitetura está localizada na pasta `src/api/`, seguindo os componentes de um deploy profissional:

* **Entrypoint (`main.py`):** Servidor principal que carrega o modelo PyTorch e expõe o endpoint de predição.
* **Contrato de Dados (`schemas.py`):** Utiliza Pydantic para validar os dados de entrada, garantindo que o modelo receba as 19 features necessárias conforme o treinamento.
* **Middleware (`middleware.py`):** Camada de segurança e monitoramento de performance para cada requisição recebida.
* **Logs (`logging_config.py`):** Implementação de observabilidade para rastrear erros e o comportamento da API em tempo real.

## 3. Fluxo de Operação
1. O sistema cliente envia um POST para `/predict` com os dados do cliente.
2. A API valida o schema, processa os dados através do pipeline de pré-processamento (`src/churn/pipeline.py`).
3. O modelo executa a inferência e aplica o **Threshold de 0.059**.
4. A resposta é retornada em JSON com a probabilidade e a classe (Churn/Não Churn).

## 4. Justificativa contra Batch
O processamento em Batch foi descartado para este caso de uso pois predições geradas uma vez por dia (ou semana) estariam defasadas em relação ao comportamento volátil do cliente de Telecom, impedindo reações em tempo real durante eventos críticos de abandono.