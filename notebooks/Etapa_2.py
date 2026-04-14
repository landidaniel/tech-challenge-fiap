"""
churn_mlp.py — MLP para previsao de Churn (Telco Customer Churn)

Uso:
    python churn_mlp.py --data data/Telco_customer_churn.xlsx
    python churn_mlp.py --data data/Telco_customer_churn.xlsx --epochs 150 --lr 0.001
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

import mlflow
import mlflow.pytorch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SEED = 42
LEAKAGE_COLS = [
    "Churn Label", "Churn Reason",
    "Churn Value", "Churn Score", "CLTV",
    "City", "Zip Code", "Latitude", "Longitude",
    "Count", "Country", "State", "Lat Long", "CustomerID",
]


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def load_and_preprocess(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Carrega o dataset Telco Customer Churn, remove leakage e faz encoding.

    Retorna:
        X : array (n_samples, n_features)
        y : array (n_samples,) binario  0/1
        feature_names : lista de nomes das features
    """
    df = pd.read_excel(path)

    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn Label"] == "Yes").astype(int)

    drop_cols = [c for c in LEAKAGE_COLS + ["Churn"] if c in df.columns]
    X_raw = df.drop(columns=drop_cols)

    X_enc = pd.get_dummies(X_raw, drop_first=True)

    print(f"Dataset: {X_enc.shape[0]} amostras | {X_enc.shape[1]} features")
    print(f"Taxa de churn: {df['Churn'].mean():.1%} ({df['Churn'].sum()} positivos)")

    return X_enc.values, df["Churn"].values, X_enc.columns.tolist()


def make_splits(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.15, val_size: float = 0.176
) -> tuple:
    """
    Split estratificado: ~70% treino | ~15% validacao | ~15% teste.

    val_size=0.176 garante que 0.176 * 0.85 ~ 0.15 do total.
    """
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_size, random_state=SEED, stratify=y_tv
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    print(f"Treino:    {X_train_sc.shape[0]} | Validacao: {X_val_sc.shape[0]} | Teste: {X_test_sc.shape[0]}")
    return X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, scaler


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class ChurnMLP(nn.Module):
    """
    Multi-Layer Perceptron para classificacao binaria de churn.

    Arquitetura:
        Input -> [Linear -> BatchNorm1d -> ReLU -> Dropout] x N -> Linear(1)

    A saida e um logit (sem sigmoid). Use BCEWithLogitsLoss no treino e
    torch.sigmoid na inferencia para obter probabilidades.

    Args:
        input_dim   : numero de features de entrada
        hidden_dims : lista com o numero de neuronios de cada camada oculta
        dropout     : taxa de dropout aplicada apos cada bloco oculto
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidade P(churn=1) para cada amostra."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self(x))


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------
def train(
    model: ChurnMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 150,
    patience: int = 15,
    pos_weight: float | None = None,
) -> dict:
    """
    Loop de treinamento com early stopping.

    Estrategia:
    - Otimizador: Adam + weight_decay=1e-4 (regularizacao L2)
    - Loss:       BCEWithLogitsLoss com pos_weight para desbalanceamento
    - LR decay:   ReduceLROnPlateau (fator 0.5 apos 5 epochs sem melhora)
    - Early stop: interrompe quando val_loss nao melhora por `patience` epochs;
                  restaura os pesos do melhor checkpoint.

    Retorna:
        history : dict com listas 'train_loss', 'val_loss', 'val_f1'
    """
    pw = torch.tensor([pos_weight], dtype=torch.float32).to(device) if pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_loss = float("inf")
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ---- Treino --------------------------------------------------------
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_loader.dataset)

        # ---- Validacao -----------------------------------------------------
        model.eval()
        val_loss = 0.0
        all_probs: list[float] = []
        all_labels: list[int] = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                val_loss += criterion(logits, yb).item() * len(yb)
                all_probs.extend(torch.sigmoid(logits).cpu().tolist())
                all_labels.extend(yb.cpu().tolist())
        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(all_labels, (np.array(all_probs) >= 0.5).astype(int), zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        scheduler.step(val_loss)

        # ---- Early stopping ------------------------------------------------
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | train={train_loss:.4f} | "
                f"val={val_loss:.4f} | f1={val_f1:.4f} | "
                f"patience={no_improve}/{patience}"
            )

        if no_improve >= patience:
            print(f"\nEarly stopping na epoch {epoch} (melhor val_loss={best_val_loss:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    print(f"Treinamento concluido. Melhor val_loss: {best_val_loss:.4f}")
    return history


# ---------------------------------------------------------------------------
# Avaliacao
# ---------------------------------------------------------------------------
def get_probs(model: ChurnMLP, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (y_true, y_prob) para um DataLoader."""
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for Xb, yb in loader:
            probs = torch.sigmoid(model(Xb.to(device))).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(yb.tolist())
    return np.array(all_labels), np.array(all_probs)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    """Calcula 6 metricas de classificacao binaria."""
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        # melhor escolha para negocio de churn: recall (capturar o maximo de clientes que vao churnar)
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"]  = average_precision_score(y_true, y_prob)
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    n_points: int = 100,
) -> tuple[float, float]:
    """
    Busca o threshold que minimiza o custo total de negocio:
        custo_total = FP * cost_fp + FN * cost_fn

    Retorna:
        best_threshold, best_cost
    """
    thresholds = np.linspace(0.05, 0.95, n_points)
    best_thresh, best_cost = 0.5, float("inf")
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost, best_thresh = cost, float(t)
    return best_thresh, best_cost


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_learning_curves(history: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history["train_loss"], label="Treino")
    axes[0].plot(history["val_loss"],   label="Validacao")
    axes[0].set_title("Loss por Epoch (BCEWithLogitsLoss)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_f1"], color="green", label="Val F1")
    axes[1].set_title("F1 Score (Validacao) por Epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "learning_curves.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cost_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    best_thresh: float,
    out_dir: Path,
) -> Path:
    thresholds = np.linspace(0.05, 0.95, 100)
    costs, fps, fns = [], [], []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        costs.append(fp * cost_fp + fn * cost_fn)
        fps.append(int(fp)); fns.append(int(fn))

    best_cost = costs[int(np.argmin(costs))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(thresholds, costs, "b-", lw=2)
    axes[0].axvline(best_thresh, color="r", linestyle="--", label=f"Otimo ({best_thresh:.2f})")
    axes[0].axvline(0.5, color="gray", linestyle=":", label="Default (0.5)")
    axes[0].scatter([best_thresh], [best_cost], color="red", s=120, zorder=5)
    axes[0].set_title("Custo de Negocio vs. Threshold")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Custo Total (R$)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x/1000:.0f}k"))

    axes[1].plot(thresholds, fps, "g-", lw=2, label="Falso Positivo (FP)")
    axes[1].plot(thresholds, fns, "r-", lw=2, label="Falso Negativo (FN)")
    axes[1].axvline(best_thresh, color="purple", linestyle="--", label=f"Otimo ({best_thresh:.2f})")
    axes[1].set_title("FP e FN vs. Threshold")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Numero de Erros")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "cost_analysis.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"Dispositivo: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dados
    X, y, feature_names = load_and_preprocess(args.data)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = make_splits(X, y)

    # 2. Modelo
    hidden = [int(h) for h in args.hidden.split(",")]
    model = ChurnMLP(input_dim=X_train.shape[1], hidden_dims=hidden, dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{model}")
    print(f"Parametros treinaveis: {total_params:,}\n")

    # 3. DataLoaders
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"pos_weight (neg/pos): {pos_weight:.2f}")
    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   args.batch_size, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  args.batch_size, shuffle=False)

    # 4. Treinamento
    print("\n--- Treinamento ---")
    history = train(
        model, train_loader, val_loader, device,
        lr=args.lr, epochs=args.epochs, patience=args.patience,
        pos_weight=pos_weight,
    )

    # 5. Avaliacao
    y_true, y_prob = get_probs(model, test_loader, device)
    y_pred_05 = (y_prob >= 0.5).astype(int)
    metrics_05 = evaluate(y_true, y_pred_05, y_prob)

    # Custo de negocio (estimativa padrao se nao fornecida)
    cost_fn = args.cost_fn if args.cost_fn else 4400.0
    cost_fp = args.cost_fp if args.cost_fp else 65.0
    best_thresh, best_cost = find_optimal_threshold(y_true, y_prob, cost_fp, cost_fn)

    y_pred_opt = (y_prob >= best_thresh).astype(int)
    metrics_opt = evaluate(y_true, y_pred_opt, y_prob)

    print("\n--- Metricas no Teste (threshold=0.5) ---")
    for k, v in metrics_05.items():
        print(f"  {k:12s}: {v:.4f}")

    print(f"\n--- Threshold Otimo (custo minimo) ---")
    print(f"  Threshold:  {best_thresh:.2f}")
    print(f"  Custo:      R$ {best_cost:,.2f}")
    print(f"  F1:         {metrics_opt['f1']:.4f}")
    print(f"  Recall:     {metrics_opt['recall']:.4f}")

    print("\nClassification Report (threshold=0.5):")
    print(classification_report(y_true, y_pred_05, target_names=["No Churn", "Churn"]))

    # 6. Graficos
    lc_path   = plot_learning_curves(history, out_dir)
    cost_path = plot_cost_analysis(y_true, y_prob, cost_fp, cost_fn, best_thresh, out_dir)
    print(f"Graficos salvos em: {out_dir}")

    # 7. MLflow
    if not args.no_mlflow:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("Tech_Challenge_4_Churn")
        with mlflow.start_run(run_name="ChurnMLP_PyTorch"):
            mlflow.log_params({
                "model_type":     "MLP_PyTorch",
                "input_dim":      X_train.shape[1],
                "hidden_dims":    args.hidden,
                "dropout":        args.dropout,
                "batch_size":     args.batch_size,
                "lr":             args.lr,
                "epochs_trained": len(history["train_loss"]),
                "patience":       args.patience,
                "pos_weight":     round(pos_weight, 3),
                "loss_fn":        "BCEWithLogitsLoss",
                "optimizer":      "Adam",
                "weight_decay":   1e-4,
                "device":         str(device),
            })
            mlflow.log_metrics({k: round(float(v), 4) for k, v in metrics_05.items()})
            mlflow.log_metrics({
                "optimal_threshold": round(best_thresh, 4),
                "optimal_f1":        round(float(metrics_opt["f1"]), 4),
                "optimal_recall":    round(float(metrics_opt["recall"]), 4),
                "optimal_precision": round(float(metrics_opt["precision"]), 4),
                "business_cost_opt": round(float(best_cost), 2),
            })
            mlflow.log_artifact(str(lc_path))
            mlflow.log_artifact(str(cost_path))
            mlflow.pytorch.log_model(model, "mlp_model")
            print("\nExperimento registrado no MLflow.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina MLP de Churn com PyTorch")
    p.add_argument("--data",        default="data/Telco_customer_churn.xlsx",
                   help="Caminho para o arquivo Excel do dataset")
    p.add_argument("--hidden",      default="256,128,64",
                   help="Neuronios das camadas ocultas separados por virgula (ex: 256,128,64)")
    p.add_argument("--dropout",     type=float, default=0.3,
                   help="Taxa de dropout (default: 0.3)")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Learning rate do Adam (default: 0.001)")
    p.add_argument("--epochs",      type=int,   default=150,
                   help="Numero maximo de epochs (default: 150)")
    p.add_argument("--patience",    type=int,   default=15,
                   help="Paciencia do early stopping (default: 15)")
    p.add_argument("--batch-size",  type=int,   default=256, dest="batch_size",
                   help="Tamanho do mini-batch (default: 256)")
    p.add_argument("--cost-fn",     type=float, default=None, dest="cost_fn",
                   help="Custo de um Falso Negativo em R$ (default: CLTV medio do dataset)")
    p.add_argument("--cost-fp",     type=float, default=None, dest="cost_fp",
                   help="Custo de um Falso Positivo em R$ (default: mensalidade media)")
    p.add_argument("--output-dir",  default="outputs", dest="output_dir",
                   help="Diretorio para salvar graficos (default: outputs/)")
    p.add_argument("--mlflow-uri",  default="http://localhost:5000", dest="mlflow_uri",
                   help="Tracking URI do servidor MLflow (default: http://localhost:5000)")
    p.add_argument("--no-mlflow",   action="store_true", dest="no_mlflow",
                   help="Desativa o registro no MLflow")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
