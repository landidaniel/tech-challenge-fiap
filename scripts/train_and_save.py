"""Treina o ChurnMLP e salva os artefatos em models/ para a API consumir.

Uso:
    python scripts/train_and_save.py
    python scripts/train_and_save.py --epochs 100 --no-mlflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Garante que src/ esta no path quando executado da raiz do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.churn.config import (
    ARTIFACTS_DIR,
    COST_FN_DEFAULT,
    COST_FP_DEFAULT,
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_URI,
    SEED,
)
from src.churn.evaluate import evaluate, find_optimal_threshold, get_probs
from src.churn.model import ChurnMLP
from src.churn.pipeline import build_pipeline, save_artifacts
from src.churn.preprocessing import load_raw
from src.churn.train import train


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)


def main(args: argparse.Namespace) -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # 1. Dados
    print(f"\nCarregando dados de: {args.data}")
    X_df, y = load_raw(args.data)
    print(f"Dataset: {X_df.shape[0]} amostras | {X_df.shape[1]} colunas brutas")
    print(f"Churn rate: {y.mean():.1%} ({y.sum()} positivos)")

    # 2. Split estratificado 70/15/15
    X_tv, X_test_df, y_tv, y_test = train_test_split(
        X_df, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.176, random_state=SEED, stratify=y_tv
    )

    # 3. Pipeline: encoder + scaler
    pipeline = build_pipeline()
    X_train = pipeline.fit_transform(X_train_df)
    X_val   = pipeline.transform(X_val_df)
    X_test  = pipeline.transform(X_test_df)
    print(f"\nTreino: {X_train.shape[0]} | Val: {X_val.shape[0]} | Teste: {X_test.shape[0]}")
    print(f"Features apos encoding: {X_train.shape[1]}")

    # 4. DataLoaders
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    train_loader = make_loader(X_train, y_train.values, args.batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val.values,   args.batch_size, shuffle=False)
    test_loader  = make_loader(X_test,  y_test.values,  args.batch_size, shuffle=False)

    # 5. Modelo
    hidden = [int(h) for h in args.hidden.split(",")]
    model = ChurnMLP(input_dim=X_train.shape[1], hidden_dims=hidden, dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModelo: {hidden} | Dropout: {args.dropout} | Params: {total_params:,}")

    # 6. Treinamento
    print("\n--- Treinamento ---")
    history = train(
        model, train_loader, val_loader, device,
        lr=args.lr, epochs=args.epochs, patience=args.patience,
        pos_weight=pos_weight,
    )

    # 7. Avaliacao
    y_true, y_prob = get_probs(model, test_loader, device)
    best_thresh, best_cost = find_optimal_threshold(
        y_true, y_prob, COST_FP_DEFAULT, COST_FN_DEFAULT
    )
    y_pred_opt = (y_prob >= best_thresh).astype(int)
    metrics = evaluate(y_true, y_pred_opt, y_prob)

    print("\n--- Metricas no Teste (threshold otimo) ---")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"  threshold   : {best_thresh:.4f}")
    print(f"  custo       : R$ {best_cost:,.2f}")

    # 8. Salvar artefatos
    save_artifacts(pipeline, model, threshold=best_thresh, out_dir=Path(args.artifacts_dir))
    print(f"\nArtefatos prontos para a API em: {args.artifacts_dir}")

    # 9. MLflow (opcional)
    if not args.no_mlflow:
        import mlflow
        import mlflow.pytorch
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="ChurnMLP_API_Deploy"):
            mlflow.log_params({
                "hidden_dims": args.hidden,
                "dropout":     args.dropout,
                "lr":          args.lr,
                "epochs":      len(history["train_loss"]),
                "pos_weight":  round(pos_weight, 3),
                "threshold":   round(best_thresh, 4),
            })
            mlflow.log_metrics({k: round(float(v), 4) for k, v in metrics.items()})
            mlflow.pytorch.log_model(model, "mlp_model")
        print("Run registrado no MLflow.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treina e salva artefatos do ChurnMLP")
    p.add_argument("--data",          default=str(DATA_PATH))
    p.add_argument("--hidden",        default="256,128,64")
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--epochs",        type=int,   default=150)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--batch-size",    type=int,   default=256, dest="batch_size")
    p.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR), dest="artifacts_dir")
    p.add_argument("--mlflow-uri",    default=MLFLOW_URI, dest="mlflow_uri")
    p.add_argument("--no-mlflow",     action="store_true", dest="no_mlflow")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
