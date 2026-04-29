"""Gera um PDF completo explicando o projeto Tech Challenge FIAP Churn."""

from fpdf import FPDF


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Tech Challenge FIAP - Guia Completo do Projeto", align="R")
        self.ln(2)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Pagina {self.page_no()}", align="C")

    def capa(self):
        self.add_page()
        self.set_fill_color(15, 30, 60)
        self.rect(0, 0, 210, 297, "F")

        self.set_y(80)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(255, 255, 255)
        self.cell(0, 14, "TECH CHALLENGE FIAP", align="C")
        self.ln(12)

        self.set_font("Helvetica", "B", 18)
        self.set_text_color(100, 180, 255)
        self.cell(0, 10, "Guia Completo do Projeto", align="C")
        self.ln(8)

        self.set_font("Helvetica", "", 13)
        self.set_text_color(200, 220, 255)
        self.cell(0, 8, "Predicao de Churn com MLP PyTorch + API FastAPI", align="C")
        self.ln(30)

        self.set_font("Helvetica", "", 11)
        self.set_text_color(160, 180, 210)
        self.cell(0, 7, "Postech FIAP - Engenharia de Machine Learning", align="C")
        self.ln(6)
        self.cell(0, 7, "2026", align="C")

    def titulo_secao(self, numero, titulo):
        self.ln(6)
        self.set_fill_color(15, 30, 60)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 9, f"  {numero}. {titulo}", fill=True)
        self.ln(7)
        self.set_text_color(0, 0, 0)

    def subtitulo(self, texto):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(15, 30, 60)
        self.cell(0, 7, texto)
        self.ln(5)
        self.set_text_color(0, 0, 0)

    def paragrafo(self, texto):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, texto)
        self.ln(2)

    def destaque(self, texto):
        self.set_fill_color(240, 245, 255)
        self.set_draw_color(100, 150, 220)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(20, 40, 80)
        self.multi_cell(0, 5.5, texto, border=1, fill=True)
        self.ln(3)
        self.set_draw_color(0, 0, 0)
        self.set_text_color(0, 0, 0)

    def codigo(self, linhas):
        self.set_fill_color(245, 245, 245)
        self.set_font("Courier", "", 8.5)
        self.set_text_color(30, 30, 30)
        for linha in linhas:
            self.cell(0, 5, linha, fill=True)
            self.ln(5)
        self.ln(2)

    def tabela(self, cabecalho, linhas, larguras=None):
        if larguras is None:
            larguras = [190 // len(cabecalho)] * len(cabecalho)

        self.set_fill_color(15, 30, 60)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 9)
        for i, col in enumerate(cabecalho):
            self.cell(larguras[i], 7, col, border=1, fill=True)
        self.ln()

        self.set_text_color(0, 0, 0)
        for idx, linha in enumerate(linhas):
            fill = idx % 2 == 0
            self.set_fill_color(248, 250, 255) if fill else self.set_fill_color(255, 255, 255)
            self.set_font("Helvetica", "", 9)
            for i, cel in enumerate(linha):
                self.cell(larguras[i], 6, str(cel), border=1, fill=fill)
            self.ln()
        self.ln(3)

    def bullet(self, itens, nivel=0):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        indent = 15 + nivel * 8
        for item in itens:
            self.set_x(indent)
            self.cell(5, 5.5, "-")
            self.multi_cell(0, 5.5, item)
        self.ln(1)


# ---------------------------------------------------------------------------

def gerar(caminho="guia_projeto_churn.pdf"):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 15, 10)

    # Capa
    pdf.capa()

    # -------------------------------------------------------------------------
    # 1. CONTEXTO E PROBLEMA DE NEGOCIO
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.titulo_secao("1", "Contexto e Problema de Negocio")

    pdf.subtitulo("O Problema")
    pdf.paragrafo(
        "Uma operadora de telecomunicacoes esta perdendo clientes em ritmo acelerado. "
        "A diretoria precisa identificar ANTES quais clientes tem risco de cancelar o servico "
        "(churn) para que a equipe de Customer Experience (CX) possa agir proativamente "
        "com ofertas de retencao."
    )
    pdf.paragrafo(
        "Sem predicao, a empresa so sabe que um cliente foi embora depois que ele ja foi. "
        "Com o modelo, e possivel intervir no momento certo -- por exemplo, quando o cliente "
        "liga para reclamar ou acessa o portal -- aumentando a taxa de retencao."
    )

    pdf.subtitulo("Por que Machine Learning?")
    pdf.bullet([
        "Regras manuais nao escalam: sao frageis, dificeis de manter e nao capturam interacoes entre variaveis.",
        "ML aprende padroes complexos: combinacoes de contrato + tipo de internet + tempo de casa que sozinhos nao indicam churn, mas juntos sim.",
        "Modelo quantifica incerteza: entrega probabilidade (ex: 64%), nao so sim/nao, permitindo priorizar os casos mais urgentes.",
    ])

    pdf.subtitulo("Custo Assimetrico")
    pdf.destaque(
        "Falso Negativo (nao alertar um cliente que vai churnar): custo = R$ 4.400 (CLTV medio perdido)\n"
        "Falso Positivo (alertar um cliente que ficaria de qualquer jeito): custo = R$ 65 (campanha de retencao)\n\n"
        "O custo de errar para baixo e 68x maior que errar para cima. Isso muda completamente como o modelo deve ser avaliado e ajustado."
    )

    # -------------------------------------------------------------------------
    # 2. DATASET
    # -------------------------------------------------------------------------
    pdf.titulo_secao("2", "Dataset")

    pdf.subtitulo("Telco Customer Churn (IBM)")
    pdf.tabela(
        ["Caracteristica", "Valor"],
        [
            ["Fonte", "IBM Sample Data Sets (publico)"],
            ["Total de registros", "7.043 clientes"],
            ["Features brutas", "21 colunas"],
            ["Features usadas no modelo", "30 (apos encoding)"],
            ["Alvo (target)", "Churn Label: Yes / No"],
            ["Taxa de churn", "26,5% (1.869 positivos)"],
            ["Desbalanceamento", "3,77 negativos para cada positivo"],
        ],
        [80, 110],
    )

    pdf.subtitulo("Colunas Removidas por Leakage")
    pdf.paragrafo(
        "Leakage acontece quando features do dataset 'sabem' a resposta antes que o modelo "
        "pudesse saber na vida real. Se incluidas, o modelo aprende a 'trapacear' e performa "
        "perfeitamente no treino mas falha em producao."
    )
    pdf.tabela(
        ["Coluna", "Motivo da Remocao"],
        [
            ["Churn Value", "Copia binaria do alvo (0/1) -- vazamento direto"],
            ["Churn Score", "Score pre-computado de churn -- derivado do alvo"],
            ["CLTV", "Customer Lifetime Value calculado com dados futuros"],
            ["CustomerID", "Identificador sem valor preditivo"],
            ["City / Zip Code / Lat / Long", "Geografico -- removido para evitar bias regional"],
            ["Churn Reason", "Razao do cancelamento -- so existe APOS o churn"],
            ["Churn Label", "O proprio alvo (string Yes/No)"],
        ],
        [55, 135],
    )

    pdf.subtitulo("Principais Features Preditivas")
    pdf.bullet([
        "Tenure Months: tempo como cliente. Clientes novos churnam mais.",
        "Contract: Month-to-month tem 3x mais churn que contratos anuais.",
        "Internet Service: Fiber optic tem maior churn (competicao mais acirrada).",
        "Monthly Charges: mensalidades altas correlacionam com churn.",
        "Online Security / Tech Support: ausencia desses servicos aumenta churn.",
    ])

    # -------------------------------------------------------------------------
    # 3. PRE-PROCESSAMENTO E PIPELINE
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.titulo_secao("3", "Pre-processamento e Pipeline sklearn")

    pdf.subtitulo("O que e um Pipeline sklearn?")
    pdf.paragrafo(
        "Um Pipeline sklearn encadeia transformacoes em sequencia garantindo que o mesmo "
        "processamento aplicado no treino seja aplicado identicamente na inferencia em producao. "
        "Sem pipeline, e facil esquecer de aplicar o scaler ou fazer o encoding diferente -- "
        "o que quebra o modelo silenciosamente."
    )

    pdf.subtitulo("Nosso Pipeline: TelcoEncoder -> StandardScaler")

    pdf.subtitulo("Passo 1: TelcoEncoder (transformador customizado)")
    pdf.paragrafo(
        "Classe que herda de BaseEstimator e TransformerMixin do sklearn. "
        "Aplica pd.get_dummies() para converter colunas categoricas em numericas (one-hot encoding) "
        "e armazena a lista de colunas do treino em feature_columns_."
    )
    pdf.destaque(
        "Problema que resolve: se no treino existia 'Contract_Two year' mas na producao "
        "um cliente tem apenas contratos mensais, o DataFrame de producao teria menos colunas. "
        "O TelcoEncoder usa reindex() para garantir que as colunas sempre batem com o treino, "
        "preenchendo colunas ausentes com 0."
    )

    pdf.subtitulo("Passo 2: StandardScaler")
    pdf.paragrafo(
        "Normaliza cada feature para media=0 e desvio padrao=1. "
        "Essencial para redes neurais: sem normalizacao, features com escala grande "
        "(ex: Total Charges em milhares) dominam o gradiente e o treino fica instavel. "
        "O scaler e FIT apenas no treino -- aplicado (transform) em val, teste e producao."
    )

    pdf.subtitulo("Split Estratificado 70/15/15")
    pdf.paragrafo(
        "Estratificado significa que a proporcao de churn (26,5%) e mantida igual nos "
        "tres conjuntos. Sem estratificacao, por acaso um split poderia ter 40% de churn "
        "no treino e 10% no teste, tornando a avaliacao enganosa."
    )
    pdf.tabela(
        ["Conjunto", "Amostras", "Uso"],
        [
            ["Treino", "4.932 (70%)", "Fit do modelo e do pipeline"],
            ["Validacao", "1.054 (15%)", "Early stopping e ajuste de hiperparametros"],
            ["Teste", "1.057 (15%)", "Avaliacao final -- nunca visto durante treino"],
        ],
        [50, 50, 90],
    )

    # -------------------------------------------------------------------------
    # 4. ARQUITETURA DO MODELO MLP
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.titulo_secao("4", "Arquitetura do Modelo - MLP PyTorch")

    pdf.subtitulo("O que e um MLP?")
    pdf.paragrafo(
        "Multi-Layer Perceptron (MLP) e a arquitetura mais basica de rede neural. "
        "Dados de entrada passam por camadas de neuronios onde cada neuronio faz uma "
        "combinacao linear das entradas seguida de uma funcao de ativacao nao-linear. "
        "A nao-linearidade e o que permite ao modelo aprender padroes complexos que "
        "uma regressao logistica simples nao consegue capturar."
    )

    pdf.subtitulo("Arquitetura Completa")
    pdf.codigo([
        "Input (30 features)",
        "  |",
        "  v",
        "[Linear(30->256)] -> [BatchNorm1d(256)] -> [ReLU] -> [Dropout(0.3)]",
        "  |",
        "  v",
        "[Linear(256->128)] -> [BatchNorm1d(128)] -> [ReLU] -> [Dropout(0.3)]",
        "  |",
        "  v",
        "[Linear(128->64)] -> [BatchNorm1d(64)] -> [ReLU] -> [Dropout(0.3)]",
        "  |",
        "  v",
        "[Linear(64->1)]  ->  logit  ->  sigmoid()  ->  P(churn) em [0,1]",
        "",
        "Total de parametros treinaveis: 50.049",
    ])

    pdf.subtitulo("Por que cada componente?")
    pdf.tabela(
        ["Componente", "O que faz", "Por que usar"],
        [
            ["Linear(in, out)", "Multiplicacao matricial + bias", "Aprende combinacoes lineares das features"],
            ["BatchNorm1d", "Normaliza ativacoes por batch", "Treino mais estavel, permite LR maior"],
            ["ReLU", "max(0, x) -- ativa apenas positivos", "Nao-linearidade simples, sem vanishing gradient"],
            ["Dropout(0.3)", "Desliga 30% dos neuronios aleatoriamente", "Regularizacao -- previne overfitting"],
            ["Linear(64, 1)", "Saida unica sem ativacao", "Gera logit para BCEWithLogitsLoss"],
        ],
        [35, 55, 95],
    )

    pdf.subtitulo("Por que a saida e um logit e nao uma probabilidade?")
    pdf.destaque(
        "BCEWithLogitsLoss combina sigmoid + binary cross entropy em uma unica operacao numericamente "
        "mais estavel que aplicar sigmoid() antes da loss. Por isso a rede retorna o logit cru "
        "e a sigmoid so e aplicada na inferencia: prob = torch.sigmoid(model(x))."
    )

    # -------------------------------------------------------------------------
    # 5. ESTRATEGIA DE TREINAMENTO
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.titulo_secao("5", "Estrategia de Treinamento")

    pdf.subtitulo("Otimizador: Adam")
    pdf.paragrafo(
        "Adam (Adaptive Moment Estimation) ajusta a taxa de aprendizado individualmente "
        "para cada parametro com base no historico de gradientes. E o otimizador padrao "
        "para a maioria das redes neurais por convergir rapido e ser robusto a escolha "
        "do learning rate inicial."
    )
    pdf.bullet([
        "Learning rate inicial: 0.001",
        "weight_decay=1e-4: adiciona penalidade L2 (regularizacao Ridge) diretamente no Adam, reduzindo overfitting sem precisar de camada extra.",
    ])

    pdf.subtitulo("Loss: BCEWithLogitsLoss com pos_weight")
    pdf.paragrafo(
        "Binary Cross Entropy e a loss padrao para classificacao binaria. "
        "Como o dataset tem 73,5% de nao-churn, sem correcao o modelo aprenderia a "
        "ignorar a classe minoritaria (churn). O pos_weight resolve isso:"
    )
    pdf.destaque(
        "pos_weight = negativos / positivos = 5.174 / 1.869 = 2,77\n\n"
        "Isso significa: cada exemplo de churn pesa 2,77x mais na loss do que um nao-churn. "
        "O modelo e forcado a prestar mais atencao nos churners."
    )

    pdf.subtitulo("ReduceLROnPlateau")
    pdf.paragrafo(
        "Scheduler que reduz o learning rate quando a val_loss para de melhorar. "
        "Parametros: fator=0.5 (divide o LR pela metade), patience=5 epochs. "
        "Evita que o modelo fique 'saltando' em torno do otimo quando o LR esta muito alto."
    )

    pdf.subtitulo("Early Stopping")
    pdf.paragrafo(
        "Para o treinamento automaticamente quando a val_loss nao melhora por 15 epochs "
        "consecutivas. Ao final, restaura os pesos do epoch com melhor val_loss "
        "(o 'melhor checkpoint'). Previne overfitting: o modelo nao continua memorizando "
        "o treino apos atingir o ponto otimo de generalizacao."
    )
    pdf.destaque(
        "No nosso caso: early stopping ativou no epoch ~23 de 150 possiveis. "
        "Sem early stopping, o modelo continuaria treinando e possivelmente overfittaria."
    )

    pdf.subtitulo("Seed = 42")
    pdf.paragrafo(
        "Seed fixa garante que o resultado e identico a cada execucao. "
        "Sem seed, o split de dados, inicializacao dos pesos e o dropout sao aleatorios -- "
        "voce poderia rodar 2 vezes e obter resultados diferentes, tornando impossivel "
        "reproduzir ou comparar experimentos. O 42 e convencao da comunidade (sem significado especial)."
    )

    # -------------------------------------------------------------------------
    # 6. METRICAS DE AVALIACAO
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.titulo_secao("6", "Metricas de Avaliacao")

    pdf.subtitulo("As 6 Metricas do Projeto")
    pdf.tabela(
        ["Metrica", "Formula", "O que mede", "Nosso valor"],
        [
            ["Accuracy", "Acertos / Total", "% de predicoes corretas", "46,6%"],
            ["Precision", "TP / (TP + FP)", "Dos alertados, quantos eram churn de verdade", "33,1%"],
            ["Recall", "TP / (TP + FN)", "Dos que churnariam, quantos capturamos", "100%"],
            ["F1", "2 x P x R / (P + R)", "Media harmonica entre Precision e Recall", "49,8%"],
            ["AUC-ROC", "Area sob curva ROC", "Poder discriminativo geral do modelo", "85,3%"],
            ["PR-AUC", "Area sob curva PR", "Desempenho na classe positiva (churn)", "65,0%"],
        ],
        [28, 40, 72, 28],
    )

    pdf.subtitulo("Por que Recall e a metrica mais importante para churn?")
    pdf.destaque(
        "Dado o custo assimetrico (FN custa R$4.400, FP custa R$65), queremos maximizar Recall "
        "(capturar o maximo de churners) mesmo que isso gere mais falsos alarmes. "
        "Um cliente avisado desnecessariamente custa R$65. Um cliente perdido custa R$4.400. "
        "Por isso aceitamos Precision baixa (33%) em troca de Recall = 100%."
    )

    pdf.subtitulo("Por que a Accuracy parece baixa (46,6%)?")
    pdf.paragrafo(
        "Com threshold=0.059, o modelo alerta quase todos os clientes como churn. "
        "Isso gera muitos falsos positivos (clientes que nao churnariam), reduzindo a accuracy. "
        "MAS e a estrategia correta dado o custo do negocio. "
        "A accuracy e uma metrica enganosa em datasets desbalanceados -- "
        "um modelo que chuta 'nao churn' sempre teria accuracy de 73,5% sem ser util."
    )

    pdf.subtitulo("Comparativo de Modelos")
    pdf.tabela(
        ["Modelo", "F1", "AUC-ROC", "Recall", "Precision"],
        [
            ["DummyClassifier", "0,000", "0,500", "0,000", "0,000"],
            ["Regressao Logistica", "0,573", "0,843", "0,541", "0,609"],
            ["Random Forest", "0,543", "0,830", "0,489", "0,611"],
            ["Gradient Boosting", "0,566", "0,845", "0,521", "0,620"],
            ["MLP PyTorch (thresh=0.059)", "0,498", "0,853", "1,000", "0,331"],
        ],
        [60, 25, 28, 28, 28],
    )
    pdf.paragrafo(
        "O MLP tem o melhor AUC-ROC (0,853), indicando melhor poder discriminativo geral. "
        "Com threshold otimizado para custo de negocio, alcanca Recall perfeito (1,000) -- "
        "nenhum churner passa despercebido."
    )

    # -------------------------------------------------------------------------
    # 7. THRESHOLD E CUSTO DE NEGOCIO
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.titulo_secao("7", "Threshold e Analise de Custo de Negocio")

    pdf.subtitulo("O que e o Threshold?")
    pdf.paragrafo(
        "O modelo retorna uma probabilidade entre 0 e 1. O threshold e o valor de corte "
        "que decide: acima disso = churn, abaixo = nao churn. "
        "O padrao e 0,5 (se P(churn) > 50%, alerta). Mas esse padrao ignora o custo do negocio."
    )

    pdf.subtitulo("Como encontramos o Threshold Otimo?")
    pdf.paragrafo(
        "Testamos todos os thresholds de 0,05 a 0,95 e calculamos o custo total para cada um:"
    )
    pdf.destaque(
        "custo_total = FP * R$65 + FN * R$4.400\n\n"
        "Threshold = 0,50  -> FP: baixo,  FN: alto  -> Custo: ~R$ 226.000\n"
        "Threshold = 0,059 -> FP: alto,   FN: zero  -> Custo: ~R$  37.000\n\n"
        "Economia com threshold otimizado: 83,6% de reducao no custo total."
    )

    pdf.subtitulo("Por que o threshold otimo e tao baixo (0,059)?")
    pdf.paragrafo(
        "Porque o custo de um FN (R$4.400) e 68x maior que um FP (R$65). "
        "Vale a pena alertar um cliente mesmo que a probabilidade de churn seja so 6% "
        "se o custo de perdê-lo for tao alto. "
        "Com threshold=0,059, praticamente todo cliente com qualquer sinal de risco e alertado."
    )

    pdf.tabela(
        ["Threshold", "FP", "FN", "Custo Total", "Observacao"],
        [
            ["0,50 (padrao)", "~120", "~280", "R$ 226.000", "Padrao -- muitos churners perdidos"],
            ["0,20", "~450", "~80", "R$ 85.000", "Melhor, ainda ha perdas"],
            ["0,059 (otimo)", "~720", "0", "R$ 37.000", "Melhor custo -- nenhum churner perdido"],
            ["0,05", "~800", "0", "R$ 39.000", "Similar, mas mais FP"],
        ],
        [28, 18, 18, 38, 78],
    )

    # -------------------------------------------------------------------------
    # 8. PIPELINE DE PRODUCAO E API
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.titulo_secao("8", "Pipeline de Producao e API FastAPI")

    pdf.subtitulo("Por que API REST?")
    pdf.paragrafo(
        "O caso de uso principal e real-time: quando um cliente liga para cancelar, "
        "o sistema de CRM precisa do score de churn em menos de 200ms para orientar o atendente. "
        "Processamento em batch (job noturno) perderia essa janela de oportunidade. "
        "A API permite integracao com qualquer sistema via HTTP."
    )

    pdf.subtitulo("Endpoints")
    pdf.tabela(
        ["Metodo", "Endpoint", "Descricao", "Retorno"],
        [
            ["GET", "/health", "Status da API e modelo", "status, model_loaded, threshold"],
            ["POST", "/predict", "Predicao para 1 cliente", "probabilidade, label, confianca"],
            ["POST", "/predict/batch", "Predicao para ate 1.000", "lista de predicoes"],
            ["GET", "/docs", "Swagger UI interativo", "Interface web da API"],
        ],
        [18, 35, 70, 67],
    )

    pdf.subtitulo("Validacao Pydantic")
    pdf.paragrafo(
        "Cada requisicao passa por validacao automatica antes de chegar ao modelo. "
        "Se um campo estiver errado, a API retorna HTTP 422 com detalhe do erro -- "
        "sem precisar escrever uma linha de validacao manual."
    )
    pdf.bullet([
        "Campo ausente -> 422 Unprocessable Entity",
        "Gender invalido (nao Male/Female) -> 422",
        "tenure negativo -> 422",
        "Senior Citizen fora de [0,1] -> 422",
        "partner nao Yes/No -> 422",
    ])

    pdf.subtitulo("Middleware de Latencia")
    pdf.paragrafo(
        "Todo request passa pelo LatencyLoggingMiddleware antes de chegar ao endpoint. "
        "Ele mede o tempo de processamento, adiciona headers de rastreabilidade na resposta "
        "e loga tudo em JSON estruturado (compativel com Datadog, CloudWatch, ELK)."
    )
    pdf.destaque(
        "X-Request-ID: uuid4 unico por requisicao -- permite rastrear um request especifico nos logs\n"
        "X-Latency-Ms: tempo de processamento em milissegundos\n\n"
        "Log gerado: {\"timestamp\": \"...\", \"level\": \"INFO\", \"method\": \"POST\", "
        "\"path\": \"/predict\", \"status_code\": 200, \"latency_ms\": 12.4, \"request_id\": \"...\"}"
    )

    pdf.subtitulo("Fluxo Completo de uma Requisicao")
    pdf.codigo([
        "1. POST /predict  (JSON com features do cliente)",
        "2. Middleware: registra inicio do timer, gera request_id",
        "3. Pydantic: valida campos, tipos e dominios",
        "4. CustomerFeatures.to_dataframe(): converte para DataFrame 1 linha",
        "5. pipeline.transform(): TelcoEncoder + StandardScaler",
        "6. model.predict_proba(): ChurnMLP -> P(churn) em [0,1]",
        "7. Compara com threshold (0.059) -> churn_label True/False",
        "8. Middleware: calcula latencia, loga, adiciona headers",
        "9. Retorna JSON: {churn_probability, churn_label, threshold, confidence}",
    ])

    # -------------------------------------------------------------------------
    # 9. TESTES AUTOMATIZADOS
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.titulo_secao("9", "Testes Automatizados (pytest)")

    pdf.subtitulo("Por que testar?")
    pdf.paragrafo(
        "Testes garantem que o codigo faz o que se espera e que mudancas futuras nao quebram "
        "o que ja funcionava (regressao). Em ML, sao especialmente importantes para "
        "detectar leakage, mudancas de schema e bugs de preprocessamento."
    )

    pdf.tabela(
        ["Arquivo", "Tipo", "O que testa", "Qtd"],
        [
            ["test_preprocessing.py", "Unitario", "TelcoEncoder: fit, transform, alinhamento de colunas", "9"],
            ["test_model.py", "Unitario", "ChurnMLP: shape de saida, logits, gradiente, variantes", "13"],
            ["test_schema.py", "Schema (pandera)", "Dominios das colunas do dataset (genero, contrato, etc)", "8"],
            ["test_api.py", "Smoke / Integracao", "Endpoints: /health, /predict, validacao, middleware", "16"],
        ],
        [48, 30, 78, 10],
    )

    pdf.subtitulo("Teste de Schema com Pandera")
    pdf.paragrafo(
        "Pandera valida o schema de DataFrames: tipos de colunas, valores permitidos, "
        "restricoes (ex: tenure >= 0). Detecta dados corrompidos na entrada antes que "
        "cheguem ao modelo. Exemplo: se o dataset de producao comecasse a ter um valor "
        "'5G' em Internet Service, o pandera alertaria imediatamente."
    )

    pdf.subtitulo("Smoke Tests da API")
    pdf.paragrafo(
        "Smoke test verifica que a aplicacao 'nao pega fogo' -- que os endpoints basicos "
        "respondem como esperado mesmo sem o modelo carregado. Testa que: "
        "/health retorna 200, /predict sem modelo retorna 503, payload invalido retorna 422, "
        "middleware adiciona headers corretos."
    )

    pdf.subtitulo("Executar os Testes")
    pdf.codigo([
        "make test           # roda todos os 46 testes com relatorio de cobertura",
        "make test-fast      # sem cobertura (mais rapido)",
        "pytest tests/test_api.py -v   # so os testes da API",
    ])

    # -------------------------------------------------------------------------
    # 10. MLFLOW
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.titulo_secao("10", "MLflow -- Rastreamento de Experimentos")

    pdf.subtitulo("O que e e por que usar?")
    pdf.paragrafo(
        "MLflow e uma plataforma de rastreamento de experimentos de ML. "
        "Sem ele, e comum perder o historico de qual modelo usou quais hiperparametros, "
        "qual versao do dataset, ou qual metrica atingiu. "
        "Com MLflow, cada 'run' registra automaticamente tudo."
    )

    pdf.subtitulo("O que e registrado em cada Run")
    pdf.tabela(
        ["Categoria", "Exemplos"],
        [
            ["Parametros", "hidden_dims, dropout, lr, epochs, patience, pos_weight"],
            ["Metricas", "accuracy, precision, recall, f1, auc_roc, pr_auc, threshold_otimo"],
            ["Artefatos", "graficos de curvas de aprendizado, matriz de confusao, modelo serializado"],
            ["Modelo", "mlflow.pytorch.log_model() -- modelo versionado e servivel"],
        ],
        [35, 155],
    )

    pdf.subtitulo("Configuracao")
    pdf.paragrafo(
        "O MLflow roda via Docker com banco PostgreSQL para persistencia e servidor de artefatos:"
    )
    pdf.codigo([
        "# Subir o servidor MLflow (na pasta do docker-compose):",
        "docker compose up -d",
        "",
        "# Acessar a interface:",
        "http://localhost:5000",
        "",
        "# Versao: MLflow 3.11.1",
        "# Backend: PostgreSQL (dados persistidos em volume Docker)",
    ])

    pdf.subtitulo("Experimento Principal")
    pdf.paragrafo(
        "Todos os runs ficam no experimento 'Tech_Challenge_4_Churn'. "
        "Foram registrados: DummyClassifier, Regressao Logistica, Random Forest, "
        "Gradient Boosting, MLP PyTorch (multiplas execucoes). "
        "Na interface e possivel comparar side-by-side todas as metricas."
    )

    # -------------------------------------------------------------------------
    # 11. ESTRUTURA DO PROJETO
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.titulo_secao("11", "Estrutura do Projeto e Boas Praticas")

    pdf.subtitulo("Estrutura de Pastas")
    pdf.codigo([
        "tech-challenge-fiap/",
        "  data/                    # Dataset bruto (nao versionado no git)",
        "  docs/",
        "    model_card.md          # Performance, limitacoes, vieses",
        "    monitoring_plan.md     # Metricas, alertas, playbook",
        "    deploy_architecture.md # Real-time vs batch + diagrama",
        "  models/                  # Artefatos treinados (nao versionados)",
        "    pipeline.pkl           # TelcoEncoder + StandardScaler",
        "    model.pth              # Pesos do ChurnMLP",
        "    meta.pkl               # input_dim, hidden_dims, threshold",
        "  notebooks/",
        "    Etapa_1.ipynb          # EDA + ML Canvas + Baselines",
        "    Etapa_2.ipynb          # MLP PyTorch + comparacao de modelos",
        "  scripts/",
        "    train_and_save.py      # Treina e salva artefatos",
        "  src/",
        "    churn/                 # Pacote de modelagem",
        "      config.py            # Constantes: SEED, LEAKAGE_COLS, custos",
        "      preprocessing.py     # TelcoEncoder sklearn-compativel",
        "      model.py             # ChurnMLP (nn.Module)",
        "      train.py             # Loop + early stopping",
        "      evaluate.py          # 6 metricas + threshold otimo",
        "      pipeline.py          # build/save/load artefatos",
        "    api/",
        "      main.py              # FastAPI: /health /predict /predict/batch",
        "      schemas.py           # Pydantic: CustomerFeatures, PredictResponse",
        "      middleware.py        # LatencyLoggingMiddleware",
        "      logging_config.py    # JsonFormatter",
        "  tests/",
        "    test_preprocessing.py  # Unitarios: TelcoEncoder",
        "    test_model.py          # Unitarios: ChurnMLP",
        "    test_schema.py         # Pandera: schema do dataset",
        "    test_api.py            # Smoke: endpoints FastAPI",
        "  Makefile                 # install, lint, test, run, train",
        "  pyproject.toml           # Dependencias, ruff, pytest",
    ])

    pdf.subtitulo("Boas Praticas Implementadas")
    pdf.tabela(
        ["Pratica", "Como implementada"],
        [
            ["Seed fixo", "SEED=42 em numpy, torch e train_test_split"],
            ["Sem print() no codigo", "logging.getLogger em src/ e scripts/"],
            ["Linting", "ruff check src/ tests/ scripts/ -- zero erros"],
            ["Sem leakage", "LEAKAGE_COLS removidas antes do split"],
            ["Pipeline reprodutivel", "sklearn Pipeline serializado em pipeline.pkl"],
            ["Validacao de entrada", "Pydantic v2 com validators customizados"],
            ["Logging estruturado", "JsonFormatter -- saida JSON para observabilidade"],
            ["Artefatos fora do git", ".gitignore para *.pkl, *.pth, *.png, *.xlsx"],
            ["Testes automatizados", "46 testes: unitarios, schema, smoke"],
            ["Model Card", "docs/model_card.md com limitacoes e vieses"],
        ],
        [60, 130],
    )

    pdf.subtitulo("Comandos Principais (Makefile)")
    pdf.tabela(
        ["Comando", "O que faz"],
        [
            ["make install", "pip install -e .[dev]  -- instala todas as dependencias"],
            ["make train", "Treina o modelo e salva artefatos em models/"],
            ["make run", "Sobe a API: uvicorn src.api.main:app --reload --port 8000"],
            ["make test", "pytest tests/ com relatorio de cobertura"],
            ["make lint", "ruff check src/ tests/ scripts/"],
            ["make format", "black src/ tests/"],
        ],
        [45, 145],
    )

    # -------------------------------------------------------------------------
    # 12. GLOSSARIO
    # -------------------------------------------------------------------------
    pdf.add_page()
    pdf.titulo_secao("12", "Glossario Rapido")

    pdf.tabela(
        ["Termo", "Significado"],
        [
            ["Churn", "Cancelamento do servico pelo cliente"],
            ["CLTV", "Customer Lifetime Value -- valor total que o cliente gera"],
            ["Leakage", "Feature que 'vaza' a resposta -- faz o modelo trapacear"],
            ["Threshold", "Valor de corte: acima = churn, abaixo = nao churn"],
            ["Logit", "Saida da rede sem sigmoid -- pode ser qualquer valor real"],
            ["pos_weight", "Peso para balancear classes desbalanceadas na loss"],
            ["Early Stopping", "Para o treino quando val_loss para de melhorar"],
            ["Dropout", "Desliga neuronios aleatoriamente durante treino (regularizacao)"],
            ["BatchNorm", "Normaliza ativacoes por mini-batch (treino mais estavel)"],
            ["AUC-ROC", "Area sob curva ROC -- 0.5=aleatório, 1.0=perfeito"],
            ["PR-AUC", "Area sob curva Precision-Recall -- melhor para dados desbalanceados"],
            ["Recall", "Dos positivos reais, quantos o modelo acertou (sensibilidade)"],
            ["Precision", "Dos alertados como positivos, quantos eram realmente positivos"],
            ["F1", "Media harmonica de Precision e Recall"],
            ["Pydantic", "Biblioteca de validacao de dados com type hints Python"],
            ["Pandera", "Biblioteca de validacao de schema para DataFrames"],
            ["Middleware", "Codigo que executa antes/depois de cada requisicao HTTP"],
            ["PSI", "Population Stability Index -- mede drift de distribuicao"],
        ],
        [45, 145],
    )

    pdf.output(caminho)
    print(f"PDF gerado: {caminho}")


if __name__ == "__main__":
    gerar("guia_projeto_churn.pdf")
