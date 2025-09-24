"""
🔧 Configuration centralisée Sentinel2
Configuration unifiée basée sur les constantes globales
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from .constants import CONSTANTS

# Charger les variables d'environnement
load_dotenv()

class SentinelConfig:
    """Configuration centralisée pour Sentinel2"""
    
    def __init__(self):
        """Initialise la configuration"""
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis les variables d'environnement"""
        return {
            # Tickers
            "tickers": self._get_tickers(),
            
            # Configuration FinBERT
            "finbert": {
                "mode": os.getenv("FINBERT_MODE", CONSTANTS.FINBERT_MODE),
                "timeout_ms": int(os.getenv("FINBERT_TIMEOUT_MS", CONSTANTS.FINBERT_TIMEOUT_MS)),
                "model_name": CONSTANTS.FINBERT_MODEL_NAME,
                "batch_size": CONSTANTS.FINBERT_BATCH_SIZE
            },
            
            # Configuration News
            "news": {
                "interval": int(os.getenv("NEWS_FLOW_INTERVAL", CONSTANTS.NEWS_INTERVAL)),
                "feeds": self._get_news_feeds(),
                "sentiment_window": CONSTANTS.SENTIMENT_WINDOW,
                "newsapi_enabled": os.getenv("NEWSAPI_ENABLED", "false").lower() == "true",
                "newsapi_key": os.getenv("NEWSAPI_KEY"),
                "newsapi_language": os.getenv("NEWSAPI_LANGUAGE", "en"),
                "newsapi_country": os.getenv("NEWSAPI_COUNTRY", "us"),
                "newsapi_sources": os.getenv("NEWSAPI_SOURCES", "reuters,bloomberg,associated-press,cnbc")
            },
            
            # Configuration Trading
            "trading": {
                "price_interval": os.getenv("PRICE_INTERVAL", CONSTANTS.PRICE_INTERVAL),
                "price_period": os.getenv("PRICE_PERIOD", CONSTANTS.PRICE_PERIOD),
                "fusion_mode": os.getenv("FUSION_MODE", CONSTANTS.FUSION_MODE),
                "buy_threshold": CONSTANTS.BUY_THRESHOLD,
                "sell_threshold": CONSTANTS.SELL_THRESHOLD,
                "hold_confidence": CONSTANTS.HOLD_CONFIDENCE
            },
            
            # Configuration LSTM
            "lstm": CONSTANTS.get_lstm_config(),
            
            # Configuration Fusion Adaptative
            "fusion": {
                "mode": os.getenv("FUSION_MODE", CONSTANTS.FUSION_MODE),
                "base_price_weight": CONSTANTS.BASE_PRICE_WEIGHT,
                "base_sentiment_weight": CONSTANTS.BASE_SENTIMENT_WEIGHT,
                "max_weight_change": CONSTANTS.MAX_WEIGHT_CHANGE,
                "regularization_factor": CONSTANTS.REGULARIZATION_FACTOR
            },
            
            # Configuration Chemins
            "paths": {
                "data_root": str(CONSTANTS.DATA_ROOT),
                "prices_dir": str(CONSTANTS.PRICES_DIR),
                "news_dir": str(CONSTANTS.NEWS_DIR),
                "sentiment_dir": str(CONSTANTS.SENTIMENT_DIR),
                "models_dir": str(CONSTANTS.MODELS_DIR),
                "logs_dir": str(CONSTANTS.LOGS_DIR)
            },
            
            # Configuration API
            "api": {
                "host": CONSTANTS.API_HOST,
                "port": CONSTANTS.API_PORT,
                "title": CONSTANTS.API_TITLE,
                "version": CONSTANTS.API_VERSION
            },
            
            # Configuration GUI
            "gui": {
                "host": CONSTANTS.GUI_HOST,
                "port": CONSTANTS.GUI_PORT,
                "title": CONSTANTS.GUI_TITLE,
                "theme": CONSTANTS.GUI_THEME
            },
            
            # Configuration Logging
            "logging": {
                "level": CONSTANTS.LOG_LEVEL,
                "format": CONSTANTS.LOG_FORMAT,
                "rotation": CONSTANTS.LOG_ROTATION,
                "retention": CONSTANTS.LOG_RETENTION
            }
        }
    
    def _get_tickers(self) -> Dict[str, str]:
        """Récupère les tickers depuis les variables d'environnement"""
        tickers_str = os.getenv("TICKERS", ",".join([f"{k}:{v}" for k, v in CONSTANTS.TICKER_NAMES.items()]))
        tickers = {}
        
        for ticker_info in tickers_str.split(","):
            if ":" in ticker_info:
                ticker, name = ticker_info.split(":", 1)
                tickers[ticker.strip()] = name.strip()
            else:
                ticker = ticker_info.strip()
                tickers[ticker] = ticker
        
        return tickers
    
    def _get_news_feeds(self) -> list:
        """Récupère les feeds RSS depuis les variables d'environnement"""
        feeds_str = os.getenv("NEWS_FEEDS", ",".join(CONSTANTS.NEWS_FEEDS))
        return [feed.strip() for feed in feeds_str.split(",") if feed.strip()]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de configuration"""
        keys = key.split(".")
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_tickers(self) -> Dict[str, str]:
        """Retourne les tickers configurés"""
        return self.get("tickers", CONSTANTS.TICKER_NAMES)
    
    def get_finbert_config(self) -> Dict[str, Any]:
        """Retourne la configuration FinBERT"""
        return self.get("finbert", {})
    
    def get_news_config(self) -> Dict[str, Any]:
        """Retourne la configuration des news"""
        return self.get("news", {})
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Retourne la configuration de trading"""
        return self.get("trading", {})
    
    def get_lstm_config(self) -> Dict[str, Any]:
        """Retourne la configuration LSTM"""
        return self.get("lstm", {})
    
    def get_fusion_config(self) -> Dict[str, Any]:
        """Retourne la configuration de fusion"""
        return self.get("fusion", {})
    
    def get_paths(self) -> Dict[str, str]:
        """Retourne les chemins de données"""
        return self.get("paths", {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Retourne la configuration API"""
        return self.get("api", {})
    
    def get_gui_config(self) -> Dict[str, Any]:
        """Retourne la configuration GUI"""
        return self.get("gui", {})
    
    def get_logging_config(self) -> Dict[str, str]:
        """Retourne la configuration de logging"""
        return self.get("logging", {})
    
    def is_finbert_real_mode(self) -> bool:
        """Vérifie si FinBERT est en mode réel"""
        return self.get("finbert.mode") == "real"
    
    def is_fusion_adaptive(self) -> bool:
        """Vérifie si la fusion est adaptative"""
        return self.get("fusion.mode") == "adaptive"
    
    def get_data_path(self, data_type: str, ticker: str = None, interval: str = None) -> Path:
        """Retourne le chemin vers les données"""
        return CONSTANTS.get_data_path(data_type, ticker, interval)
    
    def get_model_path(self, ticker: str, version: int = None) -> Path:
        """Retourne le chemin vers le modèle"""
        return CONSTANTS.get_model_path(ticker, version)
    
    def validate(self) -> bool:
        """Valide la configuration"""
        required_keys = [
            "tickers",
            "finbert.mode",
            "news.interval",
            "trading.price_interval"
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Retourne la configuration complète sous forme de dictionnaire"""
        return self._config.copy()


# Instance globale de configuration
config = SentinelConfig()

# Validation de la configuration
if not config.validate():
    raise ValueError("Configuration invalide - vérifiez les variables d'environnement")
