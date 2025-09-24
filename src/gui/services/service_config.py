#!/usr/bin/env python3
"""
Configuration centralisée pour les services GUI
"""

import sys
from pathlib import Path

# Ajouter src au path pour tous les services
src_path = str(Path(__file__).parent.parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Imports centralisés pour les services
from src.config import SentinelConfig
from src.constants import CONSTANTS

# Configuration des services
SERVICE_CONFIG = {
    "device": "cpu",
    "cache_size": 100,
    "timeout": 30,
    "retry_attempts": 3
}

def get_service_config():
    """Obtient la configuration des services"""
    config = SentinelConfig()
    return SERVICE_CONFIG

def get_data_file_path(ticker: str, data_type: str) -> str:
    """Obtient le chemin vers un fichier de données"""
    if data_type == "prices":
        return str(CONSTANTS.PRICES_DIR / f"{ticker.lower()}_1min.parquet")
    elif data_type == "news":
        return str(CONSTANTS.NEWS_DIR / f"{ticker.lower()}_news.parquet")
    elif data_type == "sentiment":
        return str(CONSTANTS.SENTIMENT_DIR / f"{ticker.lower()}_sentiment.parquet")
    else:
        return str(CONSTANTS.DATA_ROOT / f"{ticker.lower()}_{data_type}.parquet")

print("🔧 Configuration des services chargée")
