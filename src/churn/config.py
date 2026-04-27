"""Constantes e configuracoes globais do projeto."""

from pathlib import Path

# Reproducibilidade
SEED: int = 42

# Colunas que vazam o alvo ou sao identificadores sem valor preditivo
LEAKAGE_COLS: list[str] = [
    "Churn Label",
    "Churn Reason",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
    "Count",
    "Country",
    "State",
    "Lat Long",
    "CustomerID",
]

# Custos de negocio padrao (R$)
COST_FN_DEFAULT: float = 4400.0   # Falso Negativo: cliente perdido (CLTV medio)
COST_FP_DEFAULT: float = 65.0     # Falso Positivo: campanha de retencao

# Caminhos padrao
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
DATA_PATH: Path = PROJECT_ROOT / "data" / "Telco_customer_churn.xlsx"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "models"

# MLflow
MLFLOW_URI: str = "http://localhost:5000"
MLFLOW_EXPERIMENT: str = "Tech_Challenge_4_Churn"
