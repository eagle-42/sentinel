"""
🎯 CONSTANTES GLOBALES SENTINEL2
Toutes les constantes du projet centralisées ici - PAS DE VARIABLES LOCALES
"""

from pathlib import Path
from typing import List, Dict, Any

class SentinelConstants:
    """Constantes globales pour Sentinel2 - Architecture TDD"""
    
    # =============================================================================
    # TICKERS ET MARCHÉS
    # =============================================================================
    TICKERS: List[str] = ["SPY", "NVDA"]
    TICKER_NAMES: Dict[str, str] = {
        "SPY": "S&P 500 ETF",
        "NVDA": "NVIDIA Corporation"
    }
    
    # =============================================================================
    # CONFIGURATION LSTM (BASÉE SUR LES ANALYSES)
    # =============================================================================
    LSTM_SEQUENCE_LENGTH: int = 20  # Optimal selon les analyses
    LSTM_TOP_FEATURES: int = 15     # Features les plus corrélées
    LSTM_PREDICTION_HORIZON: int = 1 # 1 jour optimal pour stabilité
    LSTM_HIDDEN_SIZES: List[int] = [64, 32]  # Architecture optimisée
    LSTM_DROPOUT_RATE: float = 0.2
    LSTM_EPOCHS: int = 100
    LSTM_BATCH_SIZE: int = 32
    LSTM_PATIENCE: int = 15
    LSTM_LEARNING_RATE: float = 0.001
    
    # Top features identifiées par les analyses (par ordre de corrélation)
    TOP_FEATURES: List[str] = [
        'volume_price_trend',    # 0.1069 - Meilleure corrélation
        'price_velocity',        # 0.0841
        'returns_ma_5',          # 0.0739
        'momentum_5',            # 0.0710
        'returns_ma_10',         # 0.0584
        'momentum_10',           # 0.0542
        'ROC_10',                # 0.0542
        'returns_ma_50',         # 0.0508
        'Price_position',        # 0.0495
        'Stoch_K',               # 0.0471
        'returns_ma_20',         # 0.0443
        'momentum_20',           # 0.0411
        'RSI_14',                # 0.0401
        'Williams_R',            # 0.0368
        'BB_position'            # 0.0299
    ]
    
    # =============================================================================
    # CONFIGURATION FINBERT
    # =============================================================================
    FINBERT_MODE: str = "stub"  # "stub" pour tests, "real" pour production
    FINBERT_TIMEOUT_MS: int = 20000
    FINBERT_MODEL_NAME: str = "ProsusAI/finbert"
    FINBERT_BATCH_SIZE: int = 32
    
    # =============================================================================
    # CONFIGURATION CRAWLING
    # =============================================================================
    PRICE_INTERVAL: str = "1m"      # Intervalle des prix
    PRICE_PERIOD: str = "1d"        # Période des données
    NEWS_INTERVAL: int = 240        # 4 minutes en secondes
    SENTIMENT_WINDOW: int = 12      # Fenêtre de sentiment en minutes
    
    # Sources de données
    NEWS_FEEDS: List[str] = [
        "https://www.investing.com/rss/news_25.rss",
        "https://seekingalpha.com/feed.xml",
        "https://feeds.bloomberg.com/markets/news.rss"
    ]
    
    # =============================================================================
    # SEUILS DE TRADING ADAPTATIFS
    # =============================================================================
    # Seuils de base (volatilité normale)
    BASE_BUY_THRESHOLD: float = 0.1      # Seuil d'achat de base
    BASE_SELL_THRESHOLD: float = -0.1    # Seuil de vente de base
    HOLD_CONFIDENCE: float = 0.3         # Confiance pour HOLD
    SUCCESS_THRESHOLD: float = 0.02      # 2% - seuil de réussite des prédictions
    
    # Seuils adaptatifs selon la volatilité
    LOW_VOLATILITY_THRESHOLDS: Dict[str, float] = {
        "buy": 0.05,      # Faible volatilité = seuils bas
        "sell": -0.05
    }
    NORMAL_VOLATILITY_THRESHOLDS: Dict[str, float] = {
        "buy": 0.05,      # Volatilité normale = seuils plus sensibles
        "sell": -0.05
    }
    HIGH_VOLATILITY_THRESHOLDS: Dict[str, float] = {
        "buy": 0.2,       # Haute volatilité = seuils élevés
        "sell": -0.2
    }
    
    # Seuils de détection de volatilité
    VOLATILITY_LOW_THRESHOLD: float = 0.15    # < 15% = faible volatilité
    VOLATILITY_HIGH_THRESHOLD: float = 0.25   # > 25% = haute volatilité
    VOLUME_RATIO_LOW: float = 0.8             # Volume faible
    VOLUME_RATIO_HIGH: float = 1.5            # Volume élevé
    
    # =============================================================================
    # CHEMINS DE DONNÉES - STRUCTURE UNIFIÉE
    # =============================================================================
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_ROOT: Path = PROJECT_ROOT / "data"
    
    # Données historiques
    HISTORICAL_DIR: Path = DATA_ROOT / "historical"
    YFINANCE_DIR: Path = HISTORICAL_DIR / "yfinance"
    FEATURES_DIR: Path = HISTORICAL_DIR / "features"
    
    # Données temps réel
    REALTIME_DIR: Path = DATA_ROOT / "realtime"
    PRICES_DIR: Path = REALTIME_DIR / "prices"
    NEWS_DIR: Path = REALTIME_DIR / "news"
    SENTIMENT_DIR: Path = REALTIME_DIR / "sentiment"
    
    # Modèles et logs
    MODELS_DIR: Path = DATA_ROOT / "models"
    LOGS_DIR: Path = DATA_ROOT / "logs"
    TRADING_DIR: Path = DATA_ROOT / "trading"
    
    # =============================================================================
    # CONFIGURATION FUSION ADAPTATIVE
    # =============================================================================
    FUSION_MODE: str = "adaptive"  # "fixed" ou "adaptive"
    BASE_PRICE_WEIGHT: float = 0.6
    BASE_SENTIMENT_WEIGHT: float = 0.4
    MAX_WEIGHT_CHANGE: float = 0.1
    REGULARIZATION_FACTOR: float = 0.1
    
    # =============================================================================
    # CONFIGURATION TRANSFORMER NVDA (EN ATTENTE)
    # =============================================================================
    TRANSFORMER_NVDA_ENABLED: bool = False  # Pas encore implémenté
    TRANSFORMER_SEQUENCE_LENGTH: int = 50
    TRANSFORMER_HIDDEN_SIZE: int = 128
    TRANSFORMER_NUM_LAYERS: int = 6
    TRANSFORMER_NUM_ATTENTION_HEADS: int = 8
    
    # =============================================================================
    # CONFIGURATION API
    # =============================================================================
    API_TIMEOUT: int = 30  # Timeout API en secondes
    API_RETRY_MAX: int = 3  # Nombre max de tentatives
    API_RETRY_DELAY: float = 1.0  # Délai entre tentatives
    
    # =============================================================================
    # CONFIGURATION TESTS
    # =============================================================================
    TEST_DATA_SIZE: int = 1000
    TEST_SEQUENCE_LENGTH: int = 20
    TEST_BATCH_SIZE: int = 16
    
    # =============================================================================
    # CONFIGURATION LOGGING
    # =============================================================================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "30 days"
    
    # =============================================================================
    # MÉTRIQUES DE PERFORMANCE CIBLES
    # =============================================================================
    TARGET_GLOBAL_SCORE: float = 0.75      # >75% (amélioration de 70.8%)
    TARGET_DIRECTION_ACCURACY: float = 0.55 # >55% (amélioration de 49.7%)
    TARGET_LATENCY_MS: int = 1000          # <1 seconde
    TARGET_MAPE: float = 0.005             # <0.5% (amélioration de 0.87%)
    
    # =============================================================================
    # CONFIGURATION FRONTEND
    # =============================================================================
    GUI_HOST: str = "127.0.0.1"
    GUI_PORT: int = 7867
    GUI_TITLE: str = "Sentinel2 - Trading Algorithmique TDD"
    GUI_THEME: str = "default"
    
    # =============================================================================
    # CONFIGURATION API
    # =============================================================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "Sentinel2 API"
    API_VERSION: str = "2.0.0"
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    @classmethod
    def get_data_path(cls, data_type: str = None, ticker: str = None, interval: str = None) -> Path:
        """Retourne le chemin vers les données"""
        if data_type is None:
            return cls.DATA_ROOT
        
        if data_type == "prices":
            return cls.PRICES_DIR / f"{ticker.lower()}_{interval}.parquet"
        elif data_type == "news":
            return cls.NEWS_DIR / f"{ticker.lower()}_news.parquet"
        elif data_type == "sentiment":
            return cls.SENTIMENT_DIR / f"{ticker.lower()}_sentiment.parquet"
        elif data_type == "models":
            return cls.MODELS_DIR / f"{ticker.lower()}_model.pth"
        else:
            return cls.DATA_ROOT / data_type
    
    @classmethod
    def get_model_path(cls, ticker: str, version: int = None) -> Path:
        """Retourne le chemin vers le modèle"""
        if version:
            return cls.MODELS_DIR / ticker.lower() / f"version{version}"
        else:
            return cls.MODELS_DIR / ticker.lower()
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Crée tous les répertoires nécessaires"""
        for directory in [cls.DATA_ROOT, cls.HISTORICAL_DIR, cls.YFINANCE_DIR, 
                         cls.FEATURES_DIR, cls.REALTIME_DIR, cls.PRICES_DIR, 
                         cls.NEWS_DIR, cls.SENTIMENT_DIR, cls.MODELS_DIR, 
                         cls.LOGS_DIR, cls.TRADING_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_feature_columns(cls) -> List[str]:
        """Retourne les colonnes de features pour le LSTM"""
        return cls.TOP_FEATURES[:cls.LSTM_TOP_FEATURES]
    
    @classmethod
    def get_trading_config(cls) -> Dict[str, Any]:
        """Retourne la configuration de trading"""
        return {
            "base_buy_threshold": cls.BASE_BUY_THRESHOLD,
            "base_sell_threshold": cls.BASE_SELL_THRESHOLD,
            "hold_confidence": cls.HOLD_CONFIDENCE,
            "success_threshold": cls.SUCCESS_THRESHOLD,
            "low_volatility_thresholds": cls.LOW_VOLATILITY_THRESHOLDS,
            "normal_volatility_thresholds": cls.NORMAL_VOLATILITY_THRESHOLDS,
            "high_volatility_thresholds": cls.HIGH_VOLATILITY_THRESHOLDS,
            "volatility_low_threshold": cls.VOLATILITY_LOW_THRESHOLD,
            "volatility_high_threshold": cls.VOLATILITY_HIGH_THRESHOLD,
            "volume_ratio_low": cls.VOLUME_RATIO_LOW,
            "volume_ratio_high": cls.VOLUME_RATIO_HIGH
        }
    
    @classmethod
    def get_adaptive_thresholds(cls, volatility: float, volume_ratio: float) -> Dict[str, float]:
        """Calcule les seuils adaptatifs selon la volatilité et le volume"""
        if volatility < cls.VOLATILITY_LOW_THRESHOLD and volume_ratio < cls.VOLUME_RATIO_LOW:
            return cls.LOW_VOLATILITY_THRESHOLDS.copy()
        elif volatility > cls.VOLATILITY_HIGH_THRESHOLD and volume_ratio > cls.VOLUME_RATIO_HIGH:
            return cls.HIGH_VOLATILITY_THRESHOLDS.copy()
        else:
            return cls.NORMAL_VOLATILITY_THRESHOLDS.copy()
    
    @classmethod
    def get_lstm_config(cls) -> Dict[str, Any]:
        """Retourne la configuration LSTM"""
        return {
            "sequence_length": cls.LSTM_SEQUENCE_LENGTH,
            "top_features": cls.LSTM_TOP_FEATURES,
            "prediction_horizon": cls.LSTM_PREDICTION_HORIZON,
            "hidden_sizes": cls.LSTM_HIDDEN_SIZES,
            "dropout_rate": cls.LSTM_DROPOUT_RATE,
            "epochs": cls.LSTM_EPOCHS,
            "batch_size": cls.LSTM_BATCH_SIZE,
            "patience": cls.LSTM_PATIENCE,
            "learning_rate": cls.LSTM_LEARNING_RATE
        }


# Instance globale des constantes
CONSTANTS = SentinelConstants()

# Créer les répertoires au chargement
CONSTANTS.ensure_directories()
