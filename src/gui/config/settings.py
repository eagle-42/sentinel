"""
Configuration centralisée pour l'interface Streamlit
Conforme aux bonnes pratiques officielles
"""

from pathlib import Path
from typing import Dict, List, Any
import os

# Chemins de base
BASE_DIR = Path(__file__).parent.parent.parent
GUI_DIR = Path(__file__).parent.parent
ASSETS_DIR = GUI_DIR / "assets"

# Configuration de l'application
APP_CONFIG = {
    "page_title": "Sentinel - Trading Prédictif & Sentiment Analyse",
    "page_icon": "🚀",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Configuration des serveurs
SERVER_CONFIG = {
    "port": int(os.getenv("STREAMLIT_PORT", 8501)),
    "address": os.getenv("STREAMLIT_ADDRESS", "0.0.0.0"),
    "headless": os.getenv("STREAMLIT_HEADLESS", "false").lower() == "true"
}

# Configuration des données
DATA_CONFIG = {
    "historical_path": BASE_DIR / "data" / "historical" / "yfinance",
    "models_path": BASE_DIR / "data" / "models",
    "cache_ttl": 3600,  # 1 heure
    "supported_tickers": ["NVDA", "SPY"]
}

# Configuration des graphiques
CHART_CONFIG = {
    "colors": {
        "primary": "#667eea",
        "secondary": "#764ba2", 
        "success": "#28a745",
        "warning": "#ffc107",
        "danger": "#dc3545",
        "info": "#17a2b8"
    },
    "default_height": 500,
    "prediction_height": 700
}

# Configuration des périodes d'analyse
PERIODS = [
    "7 derniers jours",
    "1 mois", 
    "3 mois",
    "6 derniers mois",
    "1 an",
    "3 ans",
    "5 ans",
    "10 ans",
    "Total (toutes les données)"
]

# Configuration des types d'analyse
ANALYSIS_TYPES = [
    "Prix",
    "Volume", 
    "Sentiment",
    "Prédiction"
]

# Configuration des métriques LSTM
LSTM_METRICS = {
    "score_global": 70.8,
    "precision_direction": 49.7,
    "predictions_precises": 91.9,
    "erreur_moyenne": 0.87,
    "correlation": 0.999,
    "ecart_type": 5.04
}

def get_css_path() -> str:
    """Retourne le chemin vers le fichier CSS"""
    return str(ASSETS_DIR / "custom.css")

def get_data_path(ticker: str) -> str:
    """Retourne le chemin vers les données d'un ticker"""
    return str(DATA_CONFIG["historical_path"] / f"{ticker.lower()}.parquet")

def get_model_path(ticker: str) -> str:
    """Retourne le chemin vers le modèle d'un ticker"""
    return str(DATA_CONFIG["models_path"] / ticker.lower())
