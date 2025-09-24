"""
🚀 Sentinel2 - Trading Algorithmique TDD
Architecture refactorisée avec constantes globales et approche TDD
"""

from .constants import CONSTANTS, SentinelConstants
from .config import config

__version__ = "2.0.0"
__author__ = "Sentinel Team"
__description__ = "Trading Algorithmique TDD avec Fusion Adaptative Prix/Sentiment"

# Exports principaux
__all__ = [
    "CONSTANTS",
    "SentinelConstants",
    "config",
    "__version__",
    "__author__",
    "__description__"
]

# Configuration initiale
CONSTANTS.ensure_directories()