"""Definição do modelo baseline de churn (Logistic Regression)."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_baseline_model(max_iter: int = 1000) -> Pipeline:
    """
    Retorna o pipeline do modelo baseline conforme definido na Etapa 1.
    
    O pipeline inclui a normalização (StandardScaler) necessária para 
    a convergência da Regressão Logística.
    
    Parameters
    ----------
    max_iter : int
        Número máximo de iterações para o solver da Regressão Logística.
        
    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline configurado com Scaler e Regressor.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=max_iter, random_state=42))
    ])

class ChurnBaseline:
    """
    Classe wrapper para o modelo baseline para manter similaridade 
    com a estrutura do modelo principal (MLP).
    """
    def __init__(self, max_iter: int = 1000):
        self.model = get_baseline_model(max_iter=max_iter)
        
    def fit(self, X, y):
        """Treina o modelo baseline."""
        return self.model.fit(X, y)
        
    def predict_proba(self, X):
        """Retorna probabilidades de churn [P(0), P(1)]."""
        return self.model.predict_proba(X)

    def predict(self, X):
        """Retorna classes preditas."""
        return self.model.predict(X)
