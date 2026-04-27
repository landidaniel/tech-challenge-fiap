# ===========================================================================
# Makefile — Tech Challenge FIAP Churn (Etapa 3)
# ===========================================================================
# Uso:
#   make install      instala dependencias (incluindo extras de dev)
#   make lint         executa ruff (linting + auto-fix)
#   make format       formata com black
#   make test         roda pytest com cobertura
#   make test-fast    pytest sem cobertura (mais rapido)
#   make run          inicia a API com uvicorn em modo reload
#   make run-prod     inicia a API em modo producao (sem reload, 2 workers)
#   make train        treina o modelo e salva artefatos em models/
#   make clean        remove caches Python e relatorios de cobertura
# ===========================================================================

PYTHON   ?= python
SRC_DIR  := src
TEST_DIR := tests
PORT     ?= 8000

.PHONY: install lint format test test-fast run run-prod train clean help

## Instala o projeto e dependencias de dev
install:
	$(PYTHON) -m pip install -e ".[dev]"

## Linting com ruff (auto-fix habilitado)
lint:
	$(PYTHON) -m ruff check $(SRC_DIR) $(TEST_DIR)

## Formatacao com black
format:
	$(PYTHON) -m black $(SRC_DIR) $(TEST_DIR)

## Roda todos os testes com relatorio de cobertura
test:
	$(PYTHON) -m pytest $(TEST_DIR) \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		-v

## Roda testes sem cobertura (mais rapido para desenvolvimento)
test-fast:
	$(PYTHON) -m pytest $(TEST_DIR) -v --tb=short

## Inicia a API em modo desenvolvimento (reload automatico)
run:
	$(PYTHON) -m uvicorn $(SRC_DIR).api.main:app \
		--reload \
		--host 0.0.0.0 \
		--port $(PORT)

## Inicia a API em modo producao
run-prod:
	$(PYTHON) -m uvicorn $(SRC_DIR).api.main:app \
		--host 0.0.0.0 \
		--port $(PORT) \
		--workers 2

## Treina o modelo e salva artefatos em models/
train:
	$(PYTHON) notebooks/Etapa_2.py \
		--data data/Telco_customer_churn.xlsx \
		--output-dir outputs \
		--no-mlflow

## Remove caches e relatorios temporarios
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true

## Mostra esta ajuda
help:
	@echo ""
	@echo "Comandos disponiveis:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
	@echo ""
