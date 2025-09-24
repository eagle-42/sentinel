#!/usr/bin/env python3
"""
Configuration centralisée du projet Sentinel
Toutes les constantes, chemins et paramètres du projet
"""

import os
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

# =============================================================================
# CONFIGURATION CENTRALISÉE DU PROJET SENTINEL
# =============================================================================

# Chemins de base
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
TOOLS_DIR = PROJECT_ROOT / "tools"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Chemins des données
DATASET_DIR = DATA_DIR / "dataset"
TRADING_DIR = DATA_DIR / "trading"
BACKUP_DIR = DATA_DIR / "backup"

# Chemins des datasets
YFINANCE_DIR = DATASET_DIR / "1-yfinance"
DELTA_API_DIR = DATASET_DIR / "2-delta-api-15m"
FEATURES_DIR = DATASET_DIR / "3-features"

# Chemins de trading
PRICES_DIR = TRADING_DIR / "prices"
SENTIMENT_DIR = TRADING_DIR / "sentiment"
MODELS_DIR = TRADING_DIR / "models"
DECISIONS_LOG_DIR = TRADING_DIR / "decisions_log"

# Chemins des modèles par ticker
SPY_MODELS_DIR = MODELS_DIR / "spy"
NVDA_MODELS_DIR = MODELS_DIR / "nvda"

# Chemins des prédictions
PREDICTIONS_DIR = TRADING_DIR / "predictions"
SPY_PREDICTIONS_DIR = PREDICTIONS_DIR / "spy"
NVDA_PREDICTIONS_DIR = PREDICTIONS_DIR / "nvda"

# =============================================================================
# CONFIGURATION DES TICKERS
# =============================================================================

@dataclass
class TickerConfig:
    """Configuration pour un ticker"""
    symbol: str
    enabled: bool = True
    confidence_threshold: float = 0.7
    auto_update: bool = True
    version: str = "latest"

# Configuration des tickers supportés
TICKERS = {
    "SPY": TickerConfig(
        symbol="SPY",
        enabled=True,
        confidence_threshold=0.7,
        auto_update=True
    ),
    "NVDA": TickerConfig(
        symbol="NVDA", 
        enabled=True,
        confidence_threshold=0.7,
        auto_update=True
    )
}

# =============================================================================
# CONFIGURATION DES MODÈLES LSTM
# =============================================================================

@dataclass
class LSTMConfig:
    """Configuration LSTM"""
    sequence_length: int = 20
    prediction_horizon: int = 10
    test_size: float = 0.2
    validation_size: float = 0.1
    hidden_sizes: List[int] = None
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    task: str = "regression"
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [64, 32]

# Configuration des modèles
MODEL_CONFIGS = {
    "spy": LSTMConfig(
        sequence_length=20,
        prediction_horizon=10,
        hidden_sizes=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        patience=15
    ),
    "nvda": LSTMConfig(
        sequence_length=20,
        prediction_horizon=10,
        hidden_sizes=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=100,
        patience=15
    )
}

# =============================================================================
# CONFIGURATION DES FEATURES
# =============================================================================

# Features principales utilisées par le modèle LSTM
FEATURE_COLUMNS = [
    'RSI_14', 'MACD', 'BB_position', 'Stoch_K', 'Williams_R',
    'EMA_ratio', 'ATR_normalized', 'Volume_ratio', 'Price_position', 'ROC_10',
    'returns_ma_5', 'returns_std_5', 'volatility_20', 'momentum_5', 'price_velocity'
]

# Features de base (fallback)
BASIC_FEATURE_COLUMNS = [
    'returns', 'volatility', 'momentum_5', 'momentum_10', 'momentum_20',
    'volume_ratio', 'price_velocity', 'high_low_ratio', 'close_open_ratio',
    'volume_price_trend', 'price_position', 'rsi', 'bollinger_position', 'ma_ratio'
]

# =============================================================================
# CONFIGURATION DES PÉRIODES
# =============================================================================

# Périodes de filtrage disponibles
PERIODS = {
    "7 derniers jours": 7,
    "1 mois": 30,
    "3 mois": 90,
    "6 derniers mois": 180,
    "1 an": 365,
    "3 ans": 1095,
    "5 ans": 1825,
    "10 ans": 3650
}

# Période par défaut
DEFAULT_PERIOD = "1 mois"

# =============================================================================
# CONFIGURATION DES GRAPHIQUES
# =============================================================================

@dataclass
class PlotConfig:
    """Configuration des graphiques"""
    figure_size_main: tuple = (15, 10)
    figure_size_final: tuple = (16, 12)
    figure_size_error: tuple = (12, 6)
    style: str = "seaborn-v0_8"
    palette: str = "husl"

PLOT_CONFIG = PlotConfig()

# =============================================================================
# CONFIGURATION DES SEUILS DE PERFORMANCE
# =============================================================================

@dataclass
class PerformanceConfig:
    """Configuration des seuils de performance"""
    success_threshold: float = 0.02  # 2%
    recent_days_display: int = 5
    recent_30_days: int = 30
    recent_90_days: int = 90
    min_correlation: float = 0.1
    max_correlation: float = 0.9
    low_volatility_threshold: float = 0.02
    high_volatility_threshold: float = 0.05

PERFORMANCE_CONFIG = PerformanceConfig()

# =============================================================================
# CONFIGURATION DES CHEMINS DE FICHIERS
# =============================================================================

def get_data_file_path(ticker: str, file_type: str = "yfinance") -> Path:
    """Récupère le chemin d'un fichier de données"""
    if file_type == "yfinance":
        return YFINANCE_DIR / f"{ticker}_1999_2025.parquet"
    elif file_type == "features":
        return FEATURES_DIR / f"{ticker.lower()}_features.parquet"
    elif file_type == "delta":
        return DELTA_API_DIR / f"{ticker.lower()}_15min_delta.parquet"
    else:
        raise ValueError(f"Type de fichier non supporté: {file_type}")

def get_model_path(ticker: str, version: int = None) -> Path:
    """Récupère le chemin d'un modèle"""
    ticker_lower = ticker.lower()
    if ticker_lower == "spy":
        base_dir = SPY_MODELS_DIR
    elif ticker_lower == "nvda":
        base_dir = NVDA_MODELS_DIR
    else:
        raise ValueError(f"Ticker non supporté: {ticker}")
    
    if version is None:
        # Trouver la dernière version
        if not base_dir.exists():
            raise FileNotFoundError(f"Répertoire des modèles non trouvé: {base_dir}")
        
        versions = []
        for item in base_dir.iterdir():
            if item.is_dir() and item.name.startswith('version'):
                try:
                    v_num = int(item.name.replace('version', ''))
                    versions.append(v_num)
                except ValueError:
                    continue
        
        if not versions:
            raise FileNotFoundError(f"Aucune version de modèle trouvée dans {base_dir}")
        
        version = max(versions)
    
    return base_dir / f"version{version}"

def get_prediction_path(ticker: str) -> Path:
    """Récupère le chemin des prédictions"""
    ticker_lower = ticker.lower()
    if ticker_lower == "spy":
        return SPY_PREDICTIONS_DIR
    elif ticker_lower == "nvda":
        return NVDA_PREDICTIONS_DIR
    else:
        raise ValueError(f"Ticker non supporté: {ticker}")

# =============================================================================
# CONFIGURATION DES SERVICES
# =============================================================================

@dataclass
class ServiceConfig:
    """Configuration des services"""
    device: str = "cpu"  # Forcer CPU pour éviter les problèmes MPS
    cache_enabled: bool = True
    auto_load_models: bool = True
    model_validation: bool = True
    debug_mode: bool = False

SERVICE_CONFIG = ServiceConfig()

# =============================================================================
# CONFIGURATION DE L'ENVIRONNEMENT
# =============================================================================

def get_project_root() -> Path:
    """Récupère la racine du projet"""
    return PROJECT_ROOT

def get_data_dir() -> Path:
    """Récupère le répertoire des données"""
    return DATA_DIR

def get_src_dir() -> Path:
    """Récupère le répertoire src"""
    return SRC_DIR

def get_tools_dir() -> Path:
    """Récupère le répertoire tools"""
    return TOOLS_DIR

# =============================================================================
# CONFIGURATION DES LOGS
# =============================================================================

@dataclass
class LogConfig:
    """Configuration des logs"""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}"
    rotation: str = "1 day"
    retention: str = "30 days"
    compression: str = "zip"

LOG_CONFIG = LogConfig()

# =============================================================================
# CONFIGURATION DES TESTS
# =============================================================================

@dataclass
class TestConfig:
    """Configuration des tests"""
    test_data_dir: str = "test_data"
    mock_mode: bool = True
    cleanup_after: bool = True
    verbose: bool = True

TEST_CONFIG = TestConfig()

# =============================================================================
# EXPORT DES CONFIGURATIONS PRINCIPALES
# =============================================================================

__all__ = [
    # Chemins
    "PROJECT_ROOT", "DATA_DIR", "SRC_DIR", "TOOLS_DIR", "SCRIPTS_DIR",
    "DATASET_DIR", "TRADING_DIR", "YFINANCE_DIR", "FEATURES_DIR",
    "SPY_MODELS_DIR", "NVDA_MODELS_DIR", "SPY_PREDICTIONS_DIR", "NVDA_PREDICTIONS_DIR",
    
    # Configurations
    "TICKERS", "MODEL_CONFIGS", "FEATURE_COLUMNS", "BASIC_FEATURE_COLUMNS",
    "PERIODS", "DEFAULT_PERIOD", "PLOT_CONFIG", "PERFORMANCE_CONFIG",
    "SERVICE_CONFIG", "LOG_CONFIG", "TEST_CONFIG",
    
    # Fonctions utilitaires
    "get_data_file_path", "get_model_path", "get_prediction_path",
    "get_project_root", "get_data_dir", "get_src_dir", "get_tools_dir"
]
