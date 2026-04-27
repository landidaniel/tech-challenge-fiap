"""Middleware de latencia e logging de requisicoes HTTP."""

from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .logging_config import get_logger

logger = get_logger("api.access")


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    """Loga cada requisicao HTTP com metodo, path, status e latencia (ms).

    Adiciona o header ``X-Request-ID`` em todas as respostas para
    rastreabilidade end-to-end.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(latency_ms)

        logger.info(
            "%s %s -> %d (%.2fms)",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            extra={
                "request_id":  request_id,
                "method":      request.method,
                "path":        request.url.path,
                "status_code": response.status_code,
                "latency_ms":  latency_ms,
            },
        )
        return response
