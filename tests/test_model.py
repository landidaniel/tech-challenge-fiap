"""Testes unitarios para src/churn/model.py (ChurnMLP)."""

import pytest
import torch

from src.churn.model import ChurnMLP

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
INPUT_DIM = 30
BATCH = 16
HIDDEN = [64, 32]


@pytest.fixture
def model():
    return ChurnMLP(input_dim=INPUT_DIM, hidden_dims=HIDDEN, dropout=0.3)


@pytest.fixture
def sample_batch():
    torch.manual_seed(42)
    return torch.randn(BATCH, INPUT_DIM)


# ---------------------------------------------------------------------------
# Arquitetura
# ---------------------------------------------------------------------------
class TestChurnMLPArchitecture:
    def test_output_shape(self, model, sample_batch):
        model.eval()
        out = model(sample_batch)
        assert out.shape == (BATCH,), f"Esperado ({BATCH},), obtido {out.shape}"

    def test_output_is_logit_not_probability(self, model, sample_batch):
        """Sem sigmoid aplicado: valores podem ser negativos ou > 1."""
        model.eval()
        with torch.no_grad():
            out = model(sample_batch)
        # logits nao precisam estar em [0, 1]
        assert (out < 0).any() or (out > 1).any(), (
            "Parece que sigmoid foi aplicado no forward"
        )

    def test_predict_proba_in_range(self, model, sample_batch):
        probs = model.predict_proba(sample_batch)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_predict_proba_shape(self, model, sample_batch):
        probs = model.predict_proba(sample_batch)
        assert probs.shape == (BATCH,)

    def test_trainable_parameters_positive(self, model):
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total > 0


# ---------------------------------------------------------------------------
# Variantes de arquitetura
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hidden_dims", [[128], [256, 128, 64], [32, 32, 32, 32]])
def test_various_hidden_dims(hidden_dims):
    model = ChurnMLP(input_dim=INPUT_DIM, hidden_dims=hidden_dims)
    model.eval()
    x = torch.randn(8, INPUT_DIM)
    out = model(x)
    assert out.shape == (8,)


@pytest.mark.parametrize("dropout", [0.0, 0.3, 0.5])
def test_various_dropout(dropout):
    model = ChurnMLP(input_dim=INPUT_DIM, hidden_dims=[64], dropout=dropout)
    model.eval()
    x = torch.randn(4, INPUT_DIM)
    out = model(x)
    assert out.shape == (4,)


# ---------------------------------------------------------------------------
# Gradiente / treino basico
# ---------------------------------------------------------------------------
def test_backward_pass(model, sample_batch):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    labels = torch.randint(0, 2, (BATCH,)).float()

    optimizer.zero_grad()
    loss = criterion(model(sample_batch), labels)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0


def test_single_sample_eval():
    """Modelo deve funcionar com batch_size=1 em modo eval."""
    model = ChurnMLP(input_dim=10, hidden_dims=[16, 8])
    model.eval()
    x = torch.randn(1, 10)
    out = model(x)
    assert out.shape == (1,)
