"""
utils/logger.py
===============
Sistema de logging centralizado para 3DaaS.

Estructura de archivos (bajo LOG_DIR, montado como ./logs en el host):

  logs/
    api/
      app.log        ← INFO+ rotado diariamente (30 días)
      error.log      ← ERROR+ rotado diariamente (60 días)
      debug.log      ← DEBUG+ rotado diariamente (7 días) — solo local
    worker/
      app.log
      error.log
      debug.log

Niveles por entorno (APP_ENV):
  local       → consola DEBUG+, archivos DEBUG+/INFO+/ERROR+
  production  → consola INFO+,  archivos INFO+/ERROR+ (sin debug.log)

Con Prefect:
  Las tasks que tienen `log_prints=True` propagan los print() al root logger,
  por lo que llegan automáticamente a los archivos sin cambios en las tasks.

Uso:
  from utils.logger import setup_logging, get_logger

  # Llamar UNA vez al arrancar el proceso
  setup_logging("api")         # en api/main.py
  setup_logging("worker")      # en flows/pipeline.py

  # En cada módulo
  logger = get_logger(__name__)
  logger.info("mensaje")
  logger.debug("detalle")
  logger.error("error", exc_info=True)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

_APP_ENV = os.environ.get("APP_ENV", "production")
_IS_DEV = _APP_ENV != "production"
_LOG_DIR = Path(os.environ.get("LOG_DIR", "/app/logs"))

_FMT = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Librerías de terceros que generan demasiado ruido
_NOISY_LIBS = [
    "google.auth",
    "google.cloud",
    "urllib3",
    "httpx",
    "httpcore",
    "asyncio",
    "prefect.client",
    "grpc",
]

_initialized: set[str] = set()


def setup_logging(component: str = "app") -> None:
    """Configura el root logger con handlers de consola y archivos rotativos.

    Es seguro llamarlo múltiples veces: solo se inicializa una vez por componente.
    Con `disable_existing_loggers=False` de uvicorn, nuestros handlers no se
    eliminan cuando uvicorn configura su propio logging.
    """
    if component in _initialized:
        return
    _initialized.add(component)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Silenciar librerías de terceros
    for lib in _NOISY_LIBS:
        logging.getLogger(lib).setLevel(logging.WARNING)

    log_dir = _LOG_DIR / component
    log_dir.mkdir(parents=True, exist_ok=True)

    def _rotating(filename: str, level: int, days: int) -> logging.Handler:
        h = logging.handlers.TimedRotatingFileHandler(
            log_dir / filename,
            when="midnight",
            backupCount=days,
            encoding="utf-8",
        )
        h.setLevel(level)
        h.setFormatter(_FMT)
        h.suffix = "%Y-%m-%d"
        return h

    new_handlers: list[logging.Handler] = []

    # Consola: DEBUG en local, INFO en prod
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if _IS_DEV else logging.INFO)
    console.setFormatter(_FMT)
    new_handlers.append(console)

    # app.log: INFO+ — 30 días en ambos entornos
    new_handlers.append(_rotating("app.log", logging.INFO, days=30))

    # error.log: ERROR+ — 60 días en ambos entornos
    new_handlers.append(_rotating("error.log", logging.ERROR, days=60))

    # debug.log: DEBUG+ — 7 días, solo en local
    if _IS_DEV:
        new_handlers.append(_rotating("debug.log", logging.DEBUG, days=7))

    # Añadir handlers (sin eliminar los de uvicorn u otros frameworks)
    for h in new_handlers:
        root.addHandler(h)

    root.info(
        f"Logging iniciado — component={component} env={_APP_ENV} "
        f"log_dir={log_dir}"
    )


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger estándar de Python para el módulo dado.

    Los mensajes propagarán al root logger y llegarán a todos los handlers
    configurados por setup_logging().
    """
    return logging.getLogger(name)
