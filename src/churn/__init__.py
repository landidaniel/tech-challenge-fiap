"""Churn prediction package — FIAP Tech Challenge Etapa 3."""

from .config import COST_FN_DEFAULT, COST_FP_DEFAULT, LEAKAGE_COLS, SEED
from .evaluate import evaluate, find_optimal_threshold, get_probs
from .model import ChurnMLP
from .pipeline import build_pipeline, load_artifacts, save_artifacts
from .preprocessing import TelcoEncoder, load_raw
from .train import train

__all__ = [
    "SEED",
    "LEAKAGE_COLS",
    "COST_FN_DEFAULT",
    "COST_FP_DEFAULT",
    "ChurnMLP",
    "TelcoEncoder",
    "load_raw",
    "train",
    "evaluate",
    "get_probs",
    "find_optimal_threshold",
    "build_pipeline",
    "save_artifacts",
    "load_artifacts",
]
