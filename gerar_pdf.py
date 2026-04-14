"""Gera o PDF com as explicacoes de conceitos de ML da Etapa 2."""
from fpdf import FPDF

BLUE   = (31, 97, 141)
GRAY   = (44, 62, 80)
LGRAY  = (236, 240, 241)
WHITE  = (255, 255, 255)
GREEN  = (39, 174, 96)
RED    = (192, 57, 43)
ORANGE = (211, 84, 0)
DARK   = (23, 32, 42)

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=18)

    # -- cabecalho de pagina --------------------------------------------------
    def header(self):
        self.set_fill_color(*BLUE)
        self.rect(0, 0, 210, 12, "F")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*WHITE)
        self.set_xy(0, 2)
        self.cell(0, 8, "Tech Challenge FIAP -- Conceitos de Redes Neurais e ML", align="C")
        self.set_text_color(*DARK)
        self.ln(6)

    # -- rodape ---------------------------------------------------------------
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Pagina {self.page_no()}", align="C")

    # -- titulo de secao ------------------------------------------------------
    def section_title(self, num: str, title: str):
        self.ln(4)
        self.set_fill_color(*BLUE)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 10, f"  {num}. {title}", ln=True, fill=True)
        self.set_text_color(*DARK)
        self.ln(2)

    # -- subtitulo ------------------------------------------------------------
    def subtitle(self, text: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*BLUE)
        self.cell(0, 7, text, ln=True)
        self.set_text_color(*DARK)

    # -- paragrafo normal -----------------------------------------------------
    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.multi_cell(0, 6, text)
        self.ln(1)

    # -- bloco de codigo ------------------------------------------------------
    def code_block(self, text: str):
        self.set_fill_color(*LGRAY)
        self.set_draw_color(180, 180, 180)
        self.set_font("Courier", "", 8.5)
        self.set_text_color(40, 40, 40)
        lines = text.strip().split("\n")
        pad = 4
        self.ln(1)
        self.set_x(14)
        total_h = len(lines) * 5 + pad * 2
        self.rect(14, self.get_y(), 182, total_h, "FD")
        self.set_xy(18, self.get_y() + pad)
        for line in lines:
            self.cell(0, 5, line, ln=True)
            self.set_x(18)
        self.set_text_color(*DARK)
        self.ln(2)

    # -- caixa destacada ------------------------------------------------------
    def highlight_box(self, text: str, color=GREEN):
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 10)
        self.multi_cell(0, 8, f"  {text}", fill=True)
        self.set_text_color(*DARK)
        self.ln(2)

    # -- tabela ---------------------------------------------------------------
    def table(self, headers: list, rows: list, col_widths: list):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*BLUE)
        self.set_text_color(*WHITE)
        for h, w in zip(headers, col_widths):
            self.cell(w, 8, f" {h}", border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK)
        for i, row in enumerate(rows):
            fill = i % 2 == 0
            self.set_fill_color(245, 248, 250) if fill else self.set_fill_color(*WHITE)
            for val, w in zip(row, col_widths):
                self.cell(w, 7, f" {val}", border=1, fill=True)
            self.ln()
        self.ln(3)


# ============================================================================
def build(path: str):
    pdf = PDF()
    pdf.set_margins(14, 16, 14)

    # -- CAPA -----------------------------------------------------------------
    pdf.add_page()
    pdf.set_fill_color(*BLUE)
    pdf.rect(0, 0, 210, 297, "F")
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*WHITE)
    pdf.set_y(80)
    pdf.multi_cell(0, 14, "Conceitos de Redes\nNeurais e Machine Learning", align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, "Tech Challenge FIAP -- Etapa 2", align="C", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "MLP com PyTorch para Previsao de Churn", align="C", ln=True)
    pdf.ln(40)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(180, 210, 240)
    pdf.cell(0, 7, "Topicos: Seed  |  Dropout  |  Arquitetura MLP  |  Estrategia de Treino", align="C", ln=True)
    pdf.cell(0, 7, "         Melhor Metrica para Churn  |  Threshold", align="C", ln=True)

    # -- 1. SEED --------------------------------------------------------------
    pdf.add_page()
    pdf.section_title("1", "O que e Seed?")

    pdf.body(
        "Seed e um numero usado para inicializar os geradores de numeros aleatorios. "
        "Operacoes como embaralhar dados, inicializar pesos de uma rede neural ou dividir "
        "treino/teste envolvem aleatoriedade. Sem fixar a seed, cada execucao produziria "
        "resultados diferentes, tornando impossivel reproduzir ou comparar experimentos."
    )

    pdf.subtitle("No codigo, ela e passada para tres lugares:")
    pdf.code_block(
        "SEED = 42\n"
        "\n"
        "np.random.seed(SEED)       # numpy  -- afeta train_test_split, etc.\n"
        "torch.manual_seed(SEED)    # PyTorch -- afeta inicializacao dos pesos\n"
        "random_state=SEED          # scikit-learn -- afeta splits e modelos de arvore"
    )

    pdf.subtitle("Por que o numero 42?")
    pdf.body(
        "Nao ha motivo tecnico. E uma convencao da comunidade de ML, uma referencia ao livro "
        "'O Guia do Mochileiro das Galaxias'. Qualquer numero inteiro funcionaria igualmente bem."
    )

    pdf.highlight_box(
        "Regra: fixar a seed garante que dois pesquisadores rodando o mesmo codigo "
        "obtenham exatamente os mesmos resultados.",
        color=GREEN
    )

    # -- 2. DROPOUT -----------------------------------------------------------
    pdf.add_page()
    pdf.section_title("2", "O que e Dropout?")

    pdf.body(
        "Dropout e uma tecnica de regularizacao que serve para evitar que a rede neural "
        "decore os dados de treino (overfitting). Durante cada batch de treino, o dropout "
        "desliga aleatoriamente uma fracao dos neuronios, zerando sua saida."
    )

    pdf.subtitle("Como funciona com Dropout(0.3):")
    pdf.code_block(
        "Sem Dropout:      [0.8]  [0.3]  [0.9]  [0.1]  [0.6]\n"
        "Com Dropout(0.3): [0.8]  [ 0 ]  [0.9]  [ 0 ]  [0.6]  <- 2 neuronios zerados"
    )

    pdf.subtitle("Por que isso ajuda:")
    pdf.body(
        "- A rede nao consegue depender demais de nenhum neuronio especifico\n"
        "- Ela e forcada a aprender representacoes mais distribuidas e robustas\n"
        "- E como treinar varias redes menores ao mesmo tempo e fazer a media delas"
    )

    pdf.subtitle("Detalhe importante: train vs eval")
    pdf.body(
        "O dropout so e ativado durante o treino. Na avaliacao e inferencia ele e "
        "desligado automaticamente. Por isso o codigo usa model.train() e model.eval():"
    )
    pdf.code_block(
        "model.train()  # dropout ATIVO   -- durante o treinamento\n"
        "model.eval()   # dropout INATIVO -- durante validacao e teste"
    )

    pdf.highlight_box(
        "No projeto: Dropout(0.3) significa que 30% dos neuronios sao desligados "
        "aleatoriamente a cada batch de treino.",
        color=ORANGE
    )

    # -- 3. ARQUITETURA -------------------------------------------------------
    pdf.add_page()
    pdf.section_title("3", "Arquitetura: Input -> [Linear->BatchNorm->ReLU->Dropout] x N -> Linear(1)")

    pdf.body("E uma sequencia de blocos empilhados. Cada camada transforma os dados antes de passar para a proxima.")

    pdf.subtitle("Fluxo completo com os numeros reais do projeto:")
    pdf.code_block(
        "Entrada: 30 features por cliente\n"
        "         |\n"
        "+--------+------ Bloco 1 -------------------------+\n"
        "| Linear(30 -> 256)   multiplica features x pesos |\n"
        "| BatchNorm1d(256)    normaliza a saida do lote    |\n"
        "| ReLU                zera valores negativos       |\n"
        "| Dropout(0.3)        desliga 30% dos neuronios    |\n"
        "+--------------------------------------------------+\n"
        "         |\n"
        "+-------- Bloco 2 ---------------------------------+\n"
        "| Linear(256 -> 128) / BatchNorm / ReLU / Dropout |\n"
        "+--------------------------------------------------+\n"
        "         |\n"
        "+-------- Bloco 3 ---------------------------------+\n"
        "| Linear(128 -> 64) / BatchNorm / ReLU / Dropout  |\n"
        "+--------------------------------------------------+\n"
        "         |\n"
        "  Linear(64 -> 1)    produz um unico numero (logit)\n"
        "         |\n"
        "  sigmoid(logit)     converte para probabilidade [0,1]\n"
        "         |\n"
        "Saida: P(churn) = 0.73  -> 73% de chance de churnar"
    )

    pdf.subtitle("O que cada peca faz:")
    pdf.table(
        ["Camada", "Funcao", "Detalhe"],
        [
            ["Linear",       "Transformacao principal",   "saida = entrada x pesos + bias"],
            ["BatchNorm1d",  "Normaliza o lote",          "Estabiliza e acelera o treino"],
            ["ReLU",         "Ativacao nao-linear",       "max(0, x) -- zera negativos"],
            ["Dropout(0.3)", "Regularizacao",             "Desliga 30% dos neuronios no treino"],
            ["Linear(1)",    "Saida (logit)",             "Um numero: quanto maior, mais churn"],
            ["sigmoid",      "Converte em probabilidade", "sigmoid(2.0) = 0.88 = 88% de churn"],
        ],
        [32, 55, 95]
    )

    pdf.subtitle("Por que afunila (256 -> 128 -> 64)?")
    pdf.body(
        "A rede aprende representacoes cada vez mais abstratas. As primeiras camadas capturam "
        "padroes simples (ex: 'mensalidade alta'). As ultimas combinam esses padroes em "
        "conceitos mais complexos (ex: 'perfil de alto risco de churn')."
    )

    # -- 4. ESTRATEGIA --------------------------------------------------------
    pdf.add_page()
    pdf.section_title("4", "Estrategia de Treinamento")

    pdf.subtitle("4.1  Otimizador: Adam + weight_decay=1e-4")
    pdf.body(
        "Adam e o algoritmo que ajusta os pesos apos cada batch. Ele combina momentum "
        "(guarda a direcao das ultimas atualizacoes) com taxa de aprendizado adaptativa "
        "(passo diferente para cada peso)."
    )
    pdf.body(
        "weight_decay=1e-4 e a regularizacao L2. Penaliza pesos grandes, forcando a rede "
        "a preferir solucoes mais simples e evitar overfitting:"
    )
    pdf.code_block("loss_total = loss_normal + 0.0001 x (soma dos pesos^2)")

    pdf.subtitle("4.2  Loss: BCEWithLogitsLoss com pos_weight")
    pdf.body(
        "BCE (Binary Cross-Entropy) mede o quanto a probabilidade prevista diverge do valor real. "
        "'WithLogits' significa que o sigmoid e aplicado internamente, sendo numericamente mais estavel."
    )
    pdf.body("pos_weight=2.77 resolve o desbalanceamento (73% nao-churn / 27% churn):")
    pdf.code_block(
        "pos_weight = negativos / positivos = 5174 / 1869 = 2.77\n"
        "\n"
        "Erro num churner vale 2.77x mais que erro num nao-churner\n"
        "-> a rede presta mais atencao nos casos de churn"
    )

    pdf.subtitle("4.3  LR Decay: ReduceLROnPlateau")
    pdf.body(
        "Reduz o learning rate automaticamente quando o treino empaca. "
        "Fator 0.5 apos 5 epochs sem melhora na val_loss:"
    )
    pdf.code_block(
        "Epoch 1-5:  val_loss caindo  -> LR = 0.001   (passo normal)\n"
        "Epoch 6-10: val_loss parada  -> LR = 0.001   (aguarda patience=5)\n"
        "Epoch 11:   5 sem melhora    -> LR = 0.0005  (passo menor)\n"
        "Epoch 12+:  refinamento fino dos pesos"
    )

    pdf.subtitle("4.4  Early Stopping")
    pdf.body(
        "Interrompe o treino quando a rede para de melhorar na validacao, "
        "evitando overfitting. Restaura os pesos do melhor checkpoint:"
    )
    pdf.code_block(
        "Epoch 3:  val_loss = 0.71  -> salva checkpoint  (melhor)\n"
        "Epoch 4:  val_loss = 0.72  -> sem melhora (1/15)\n"
        "...\n"
        "Epoch 18: val_loss = 0.75  -> sem melhora (15/15) -> PARA\n"
        "          restaura pesos da Epoch 3 (melhor val_loss=0.71)"
    )

    # -- 5. MELHOR METRICA ----------------------------------------------------
    pdf.add_page()
    pdf.section_title("5", "Melhor Metrica de Avaliacao para Churn")

    pdf.subtitle("Os dois tipos de erro:")
    pdf.code_block(
        "Cliente vai churnar    -> modelo diz NAO = Falso Negativo (FN)\n"
        "                                           perde o cliente para sempre\n"
        "\n"
        "Cliente nao vai churnar -> modelo diz SIM = Falso Positivo (FP)\n"
        "                                            gasta dinheiro em retencao desnecessaria"
    )

    pdf.subtitle("Por que Acuracia e a pior escolha:")
    pdf.body(
        "O dataset tem 73% nao-churn. Um modelo idiota que responde 'ninguem vai churnar' "
        "para todo mundo teria Acuracia = 73%, mas nao identificaria nenhum churner."
    )
    pdf.highlight_box("Acuracia e enganosa em datasets desbalanceados. Nunca use sozinha.", color=RED)

    pdf.subtitle("Comparacao das metricas:")
    pdf.table(
        ["Metrica", "O que mede", "Problema evitado"],
        [
            ["Precision", "Dos previstos como churn, quantos realmente foram?", "Evita retencao desnecessaria (FP)"],
            ["Recall",    "Dos que churnaram, quantos o modelo capturou?",       "Evita perder churners (FN)"],
            ["F1",        "Media harmonica de Precision e Recall",               "Equilibrio geral"],
            ["Acuracia",  "Percentual de acertos geral",                         "Nada -- enganosa aqui"],
        ],
        [26, 78, 74]
    )

    pdf.subtitle("Resposta para churn: Recall e a metrica principal")
    pdf.body("No projeto calculamos que:")
    pdf.code_block(
        "FN custa R$ 4.400  (CLTV perdido)\n"
        "FP custa R$    65  (acao de retencao)\n"
        "\n"
        "FN e 67x mais caro que FP\n"
        "\n"
        "Threshold = 0.50 -> Recall = 0.83  (perde 17% dos churners)\n"
        "Threshold = 0.06 -> Recall = 1.00  (captura todos)  <- custo cai 83%"
    )

    pdf.table(
        ["Situacao", "Melhor Metrica"],
        [
            ["Custo de FN >> FP  (churn, fraude, doenca)", "Recall"],
            ["Custo de FP >> FN  (spam, alarme falso)",    "Precision"],
            ["Custos equilibrados",                         "F1"],
            ["Dataset balanceado e custos iguais",          "Acuracia"],
        ],
        [120, 58]
    )

    # -- 6. THRESHOLD ---------------------------------------------------------
    pdf.add_page()
    pdf.section_title("6", "O que e Threshold?")

    pdf.body(
        "Threshold e o valor de corte que transforma uma probabilidade em uma decisao binaria "
        "(churn / nao churn). A rede neural nao decide diretamente -- ela retorna um numero "
        "entre 0 e 1. O threshold e a linha que separa as duas classes."
    )

    pdf.subtitle("Exemplo com threshold = 0.5:")
    pdf.code_block(
        "Cliente A -> P(churn) = 0.91  >= 0.5  ->  CHURN\n"
        "Cliente B -> P(churn) = 0.63  >= 0.5  ->  CHURN\n"
        "Cliente C -> P(churn) = 0.48  <  0.5  ->  nao churn\n"
        "Cliente D -> P(churn) = 0.12  <  0.5  ->  nao churn"
    )

    pdf.subtitle("O que acontece quando voce move o threshold:")
    pdf.code_block(
        "0.0 -----------------[threshold]---------------- 1.0\n"
        "      <- nao churn ->     ^        <- churn ->\n"
        "\n"
        "Mover para esquerda -> captura mais churners (Recall alto) mas mais FP\n"
        "Mover para direita  -> menos FP mas perde churners (Recall baixo)"
    )

    pdf.table(
        ["", "Threshold baixo", "Threshold 0.5", "Threshold alto"],
        [
            ["Recall",    "alto",  "medio", "baixo"],
            ["Precision", "baixa", "media", "alta"],
            ["Ideal para","churn, fraude, doenca", "custos iguais", "spam, alarme falso"],
        ],
        [30, 52, 46, 50]
    )

    pdf.subtitle("Por que 0.5 nao e o melhor para churn:")
    pdf.code_block(
        "threshold = 0.50:\n"
        "  FP = 220  ->  220 x R$65    = R$  14.300\n"
        "  FN =  48  ->   48 x R$4.400 = R$ 211.200\n"
        "  custo total = R$ 225.500\n"
        "\n"
        "threshold = 0.06:\n"
        "  FP = 565  ->  565 x R$65    = R$  36.700\n"
        "  FN =   0  ->    0 x R$4.400 = R$       0\n"
        "  custo total = R$  36.700  <- 83% mais barato"
    )

    pdf.highlight_box(
        "O threshold nao e um hiperparametro do modelo -- e uma decisao de negocio "
        "baseada no custo relativo de cada tipo de erro.",
        color=BLUE
    )

    pdf.output(path)
    print(f"PDF gerado: {path}")


if __name__ == "__main__":
    build("conceitos_ml_etapa2.pdf")
