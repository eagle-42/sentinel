"""
🧠 Modules Core Sentinel2
Modules fondamentaux : fusion, sentiment, prédiction
"""

from .fusion import AdaptiveFusion, FusionConfig, MarketRegime
from .sentiment import SentimentAnalyzer, FinBertAnalyzer
from .prediction import LSTMPredictor, PredictionEngine

__all__ = [
    "AdaptiveFusion",
    "FusionConfig", 
    "MarketRegime",
    "SentimentAnalyzer",
    "FinBertAnalyzer",
    "LSTMPredictor",
    "PredictionEngine"
]
