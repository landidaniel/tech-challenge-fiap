"""Configuracao de logging estruturado (JSON) para a API."""

from __future__ import annotations

import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """Formata logs como JSON de linha unica para ingestao em sistemas de observabilidade."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import traceback

        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = traceback.format_exception(*record.exc_info)
        # Campos extras passados via extra={...}
        for key in ("request_id", "path", "method", "status_code", "latency_ms"):
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)
        return json.dumps(log_obj, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Configura o logger raiz com JsonFormatter no stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
