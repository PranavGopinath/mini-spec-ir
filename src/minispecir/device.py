from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(preferred: str | None = None) -> torch.device:
    """Pick execution device: explicit override, else MPS, else CPU."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_device_info(device: torch.device) -> None:
    """Log device choice and relevant backend flags."""
    logger.info("Using device: %s", device)
    logger.info("PyTorch version: %s", torch.__version__)
    if device.type == "mps":
        logger.info("MPS (Metal) backend is active")
    elif device.type == "cuda":
        logger.info("CUDA is available: %s", torch.cuda.is_available())
    else:
        logger.info("Running on CPU")
