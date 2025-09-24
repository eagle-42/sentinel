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
from config.unified_config import get_config, get_data_file_path, get_model_path, get_feature_columns

# Configuration des services
SERVICE_CONFIG = {
    "device": "cpu",
    "cache_size": 100,
    "timeout": 30,
    "retry_attempts": 3
}

def get_service_config():
    """Obtient la configuration des services"""
    config = get_config()
    return {**SERVICE_CONFIG, **config.services}

print("🔧 Configuration des services chargée")
