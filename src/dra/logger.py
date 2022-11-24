import os
import logging

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())  # pragma: no cover
logger = logging.getLogger("dra")  # pragma: no cover
