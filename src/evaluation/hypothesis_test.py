import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

# Adiciona o diretório 'src' ao path para permitir importações sem o prefixo 'src.'
# Útil quando o script é executado diretamente.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy import stats
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Importações oficiais do projeto
from churn.model import ChurnMLP
from churn.preprocessing import load_raw, TelcoEncoder
from churn.train import train as train_mlp
from churn.config import SEED, DATA_PATH
from churn_baseline.model import get_baseline_model

# ---------------------------------------------------------
# 1. WRAPPER PARA O SEU MODELO REAL (PyTorch)
# ---------------------------------------------------------
# O seu modelo retorna LOGITS, então usaremos BCEWithLogitsLoss no treino.

# Para usar o PyTorch dentro do K-Fold facilmente, criamos uma classe "Wrapper" 
# que se comporta como um modelo do Scikit-Learn (tem fit e predict_proba)
class PyTorchWrapper:
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], epochs=200, lr=0.001, dropout=[0.5, 0.3, 0.2, 0.1]):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.model = ChurnMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)

    def fit(self, X, y):
        # Para um treino idêntico ao do projeto, precisamos de um split de validação interno
        # para o Early Stopping funcionar conforme o src/churn/train.py
        from sklearn.model_selection import train_test_split
        X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

        # Preparando DataLoaders conforme esperado pelo train_mlp
        train_ds = TensorDataset(torch.FloatTensor(X_t), torch.FloatTensor(y_t))
        val_ds = TensorDataset(torch.FloatTensor(X_v), torch.FloatTensor(y_v))
        
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

        # Chamando a função oficial de treinamento do projeto (sem pos_weight para focar em AUC)
        train_mlp(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            lr=self.lr,
            epochs=self.epochs,
            pos_weight=None
        )

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            preds = self.model.predict_proba(X_tensor).numpy()
        preds = preds.reshape(-1, 1)
        return np.hstack((1 - preds, preds))

# ---------------------------------------------------------
# 2. FUNÇÃO DE TESTE DE HIPÓTESE
# ---------------------------------------------------------
def run_hypothesis_test(X, y, baseline_pipeline, mlp_wrapper, n_splits=5, alpha=0.05):
    """
    Roda a validação cruzada K-Fold para dois modelos e faz o Teste-T Pareado.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    auc_baseline = []
    auc_mlp = []
    
    print(f"\n--- Iniciando Validação Cruzada ({n_splits}-Fold) ---")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # Usamos .iloc para manter os dtypes originais do DataFrame
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # --- PREPROCESSAMENTO (Encoding Oficial) ---
        encoder = TelcoEncoder()
        X_train_enc = encoder.fit_transform(X_train)
        X_test_enc = encoder.transform(X_test)

        # 1. Baseline Original (Conforme definido no projeto: Scaler + LogReg)
        # O baseline_pipeline já possui o StandardScaler interno e não usa PCA
        baseline_pipeline.fit(X_train_enc, y_train)
        pred_base = baseline_pipeline.predict_proba(X_test_enc)[:, 1]
        score_base = roc_auc_score(y_test, pred_base)
        auc_baseline.append(score_base)
        
        # 2. Sistema MLP Turbinado (Com Normalização + PCA + Arquitetura Pro)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_enc)
        X_test_scaled = scaler.transform(X_test_enc)

        pca = PCA(n_components=0.95, random_state=SEED)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        if fold == 0:
            print(f"--- Comparativo de Sistemas (Fold 1) ---")
            print(f"Baseline:  Features encoded ({X_train_enc.shape[1]})")
            print(f"MLP Pro:   Features via PCA ({X_train_pca.shape[1]})")

        mlp = PyTorchWrapper(input_dim=X_train_pca.shape[1])
        mlp.fit(X_train_pca, y_train)
        pred_mlp = mlp.predict_proba(X_test_pca)[:, 1]
        score_mlp = roc_auc_score(y_test, pred_mlp)
        auc_mlp.append(score_mlp)
        
        print(f"Fold {fold+1} | AUC Baseline: {score_base:.4f} | AUC MLP: {score_mlp:.4f}")

    # Convertendo resutados para arrays
    scores_base = np.array(auc_baseline)
    scores_mlp = np.array(auc_mlp)
    
    print("\n--- Resultados Finais das Amostras ---")
    print(f"Média Baseline (AUC): {scores_base.mean():.4f}  (Desvio Padrão: {scores_base.std():.4f})")
    print(f"Média MLP (AUC):      {scores_mlp.mean():.4f}  (Desvio Padrão: {scores_mlp.std():.4f})")
    
    # -----------------------------------------
    # TESTE T DE STUDENT PAREADO
    # -----------------------------------------
    # Comparamos as listas de scores geradas em cada fold
    t_stat, p_value = stats.ttest_rel(scores_mlp, scores_base)
    
    print("\n--- Conclusão do Teste de Hipótese ---")
    print(f"Alfa estipulado: {alpha}")
    print(f"P-Valor obtido:  {p_value:.5f}")
    
    if p_value < alpha:
        print("💡 Resultado: REJEITAMOS a Hipótese Nula (H0)")
        if scores_mlp.mean() > scores_base.mean():
             print("Conclusão: A Rede Neural (MLP) é estatisticamente SUPERIOR ao Baseline.")
        else:
             print("Conclusão: O Baseline é estatisticamente SUPERIOR à Rede Neural.")
    else:
        print("⚖️ Resultado: FALHAMOS em rejeitar a Hipótese Nula (H0)")
        print("Conclusão: Não há evidência de diferença significativa entre os modelos.")
        print("           Neste caso, é preferível seguir com o modelo Baseline (Regressão Logística),")
        print("           pois em caso de empate técnico, prefere-se o modelo menos complexo (Navalha de Occam).")


# ---------------------------------------------------------
# 3. CARREGAMENTO E LIMPEZA DOS DADOS REAIS
# ---------------------------------------------------------
def load_telco_data(file_path=DATA_PATH):
    """
    Carrega o dataset usando a função oficial do projeto.
    """
    print(f"Lendo dados de: {file_path}...")
    X_df, y = load_raw(file_path)
    return X_df, y.values

if __name__ == "__main__":
    # Carregando dados reais do curso
    try:
        X_df, y = load_telco_data()
        
        print(f"Dataset carregado: {X_df.shape[0]} amostras e {X_df.shape[1]} colunas brutas.")
        
        # Preparando Baseline centralizado
        baseline = get_baseline_model(max_iter=1000)
        
        # Rodando o Teste de Hipótese (X_df agora é passado como DataFrame completo)
        run_hypothesis_test(X_df, y, baseline, None, n_splits=10, alpha=0.05)
        
    except FileNotFoundError:
        print("❌ Erro: Arquivo 'data/Telco_customer_churn.xlsx' não encontrado.")
        print("Verifique se você está na raiz do projeto.")
    except Exception as e:
        print(f"❌ Ocorreu um erro: {e}")
