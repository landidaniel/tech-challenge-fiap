import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.churn.preprocessing import load_raw, TelcoEncoder
from src.churn.model import ChurnMLP
from src.churn.train import train
from src.churn.evaluate import get_probs
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def run_visualization():
    # 1. Carregar Dados
    X_df, y = load_raw("data/Telco_customer_churn.xlsx")
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    encoder = TelcoEncoder()
    scaler = StandardScaler()
    X_train_enc = encoder.fit_transform(X_train_df)
    X_test_enc = encoder.transform(X_test_df)
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)
    
    # 2. Treinar MLP
    device = torch.device("cpu")
    model = ChurnMLP(input_dim=X_train_scaled.shape[1], hidden_dims=[256, 128, 64])
    train_ds = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train.values))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_ds = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test.values))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    print("Treinando modelo para prova real com agrupamento...")
    train(model, train_loader, test_loader, device, epochs=50)
    
    # 3. Probabilidades Previstas
    y_true, y_prob = get_probs(model, test_loader, device)
    
    # 4. Plotagem com Bins (Médias Reais)
    features_to_plot = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    sns.set_style("whitegrid")
    
    for i, col in enumerate(features_to_plot):
        x_values = X_test_df[col].values
        
        # Criamos os Bins (20 grupos por feature)
        df_plot = pd.DataFrame({'x': x_values, 'y_real': y_true, 'y_prob': y_prob})
        df_plot['bin'] = pd.qcut(df_plot['x'], q=20, duplicates='drop')
        
        # Calculamos a média real e a média prevista por bin
        bin_stats = df_plot.groupby('bin', observed=True).agg({'x': 'mean', 'y_real': 'mean', 'y_prob': 'mean'})
        
        # Plotamos as Médias Reais (Pontos)
        axes[i].scatter(bin_stats['x'], bin_stats['y_real'], color='red', s=80, edgecolors='black', 
                        label='Taxa de Churn Real (Média do Grupo)', zorder=3)
        
        # Plotamos as Médias Previstas pelo Modelo (Linha)
        axes[i].plot(bin_stats['x'], bin_stats['y_prob'], color='blue', lw=3, marker='o',
                     label='Probabilidade Prevista (Média do Modelo)', alpha=0.8)
        
        # Fundo: Pontos originais bem clarinhos só para contexto
        axes[i].scatter(x_values, y_true, alpha=0.05, color='gray', s=5, zorder=1)

        axes[i].set_title(f"Acurácia Visual: {col}", fontsize=14)
        axes[i].set_xlabel(col, fontsize=12)
        axes[i].set_ylabel("Taxa / Probabilidade de Churn", fontsize=12)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].legend(fontsize=9)
        
    plt.suptitle("Prova Real: Médias Observadas vs Probabilidades do Modelo", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = Path("output/results/prova_real_consolidada.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Gráfico consolidado salvo em: {output_path}")

if __name__ == "__main__":
    run_visualization()
