"""
🧪 Tests Core Sentinel2
Tests pour les modules fondamentaux
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.constants import CONSTANTS
from src.core.fusion import AdaptiveFusion, FusionConfig, MarketRegime
from src.core.sentiment import SentimentAnalyzer, FinBertAnalyzer
from src.core.prediction import PricePredictor, LSTMPredictor


@pytest.mark.unit
class TestAdaptiveFusion:
    """Tests pour la fusion adaptative"""
    
    def test_fusion_initialization(self):
        """Test l'initialisation de la fusion adaptative"""
        fusion = AdaptiveFusion()
        
        assert fusion.config.base_price_weight == CONSTANTS.BASE_PRICE_WEIGHT
        assert fusion.config.base_sentiment_weight == CONSTANTS.BASE_SENTIMENT_WEIGHT
        assert len(fusion.history) == 0
        assert "price" in fusion.current_weights
        assert "sentiment" in fusion.current_weights
    
    def test_add_signal(self):
        """Test l'ajout d'un signal"""
        fusion = AdaptiveFusion()
        
        result = fusion.add_signal(
            price_signal=0.5,
            sentiment_signal=0.3,
            price_volatility=0.2,
            volume_ratio=1.2
        )
        
        assert "fused_signal" in result
        assert "weights" in result
        assert "regime" in result
        assert len(fusion.history) == 1
        assert isinstance(result["regime"], MarketRegime)
    
    def test_market_regime_detection(self):
        """Test la détection des régimes de marché"""
        fusion = AdaptiveFusion()
        
        # Test basse volatilité
        regime = fusion._detect_market_regime(0.1, 0.8)
        assert regime.volatility_level == "low"
        
        # Test haute volatilité
        regime = fusion._detect_market_regime(0.3, 1.5)
        assert regime.volatility_level == "high"
        
        # Test volatilité moyenne
        regime = fusion._detect_market_regime(0.2, 1.0)
        assert regime.volatility_level == "medium"
    
    def test_weight_adaptation(self):
        """Test l'adaptation des poids"""
        fusion = AdaptiveFusion()
        
        # Régime haute volatilité
        regime = MarketRegime("high", 0.5, 0.8)
        old_weights = fusion.current_weights.copy()
        
        fusion._adapt_weights(regime)
        
        # Les poids doivent avoir changé
        assert fusion.current_weights != old_weights
        # Les poids doivent être normalisés
        total_weight = sum(fusion.current_weights.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_fusion_calculation(self):
        """Test le calcul de la fusion"""
        fusion = AdaptiveFusion()
        
        fused_signal = fusion._calculate_fusion(0.5, 0.3)
        
        expected = (fusion.current_weights["price"] * 0.5 + 
                   fusion.current_weights["sentiment"] * 0.3)
        
        assert abs(fused_signal - expected) < 1e-6
    
    def test_fusion_summary(self):
        """Test le résumé de la fusion"""
        fusion = AdaptiveFusion()
        
        # Ajouter quelques signaux
        for i in range(5):
            fusion.add_signal(0.1 * i, 0.2 * i, 0.1, 1.0)
        
        summary = fusion.get_fusion_summary()
        
        assert summary["total_signals"] == 5
        assert "avg_fused_signal" in summary
        assert "current_weights" in summary
        assert "regime" in summary

@pytest.mark.unit
class TestSentimentAnalyzer:
    """Tests pour l'analyseur de sentiment"""
    
    def test_sentiment_analyzer_initialization(self):
        """Test l'initialisation de l'analyseur de sentiment"""
        analyzer = SentimentAnalyzer()
        
        assert analyzer.window_minutes == CONSTANTS.SENTIMENT_WINDOW
        assert len(analyzer.sentiment_data) == 0
        assert len(analyzer.price_data) == 0
        assert len(analyzer.volume_data) == 0
    
    def test_add_sentiment(self):
        """Test l'ajout de sentiment"""
        analyzer = SentimentAnalyzer()
        
        analyzer.add_sentiment("SPY", 0.5, 0.8)
        
        assert "SPY" in analyzer.sentiment_data
        assert len(analyzer.sentiment_data["SPY"]) == 1
        assert analyzer.sentiment_data["SPY"][0][1] == 0.5  # sentiment
        assert analyzer.sentiment_data["SPY"][0][2] == 0.8  # confidence
    
    def test_add_price(self):
        """Test l'ajout de prix"""
        analyzer = SentimentAnalyzer()
        
        analyzer.add_price("SPY", 100.0)
        
        assert "SPY" in analyzer.price_data
        assert len(analyzer.price_data["SPY"]) == 1
        assert analyzer.price_data["SPY"][0][1] == 100.0
    
    def test_get_sentiment(self):
        """Test la récupération du sentiment"""
        analyzer = SentimentAnalyzer()
        
        # Ajouter des sentiments
        analyzer.add_sentiment("SPY", 0.5, 0.8)
        analyzer.add_sentiment("SPY", 0.3, 0.6)
        
        sentiment = analyzer.get_sentiment("SPY")
        
        # Doit être une moyenne pondérée
        assert isinstance(sentiment, float)
        assert -1 <= sentiment <= 1
    
    def test_get_volatility(self):
        """Test le calcul de la volatilité"""
        analyzer = SentimentAnalyzer()
        
        # Ajouter des prix
        prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105]
        for price in prices:
            analyzer.add_price("SPY", price)
        
        volatility = analyzer.get_volatility("SPY")
        
        assert isinstance(volatility, float)
        assert volatility >= 0
    
    def test_get_volume_ratio(self):
        """Test le calcul du ratio de volume"""
        analyzer = SentimentAnalyzer()
        
        # Ajouter des volumes
        volumes = [1000, 1200, 800, 1500, 900]
        for volume in volumes:
            analyzer.add_volume("SPY", volume)
        
        ratio = analyzer.get_volume_ratio("SPY", 1000)
        
        assert isinstance(ratio, float)
        assert ratio > 0
    
    def test_adaptive_sentiment(self):
        """Test le sentiment adaptatif"""
        analyzer = SentimentAnalyzer()
        
        # Ajouter des données
        analyzer.add_sentiment("SPY", 0.5, 0.8)
        analyzer.add_price("SPY", 100.0)
        
        result = analyzer.get_adaptive_sentiment("SPY", 0.5, 0.2, 1.2)
        
        assert "base_sentiment" in result
        assert "adjusted_sentiment" in result
        assert "confidence" in result
        assert "volatility" in result
        assert "volume_ratio" in result
        assert -1 <= result["adjusted_sentiment"] <= 1
        assert 0 <= result["confidence"] <= 1

@pytest.mark.unit
class TestFinBertAnalyzer:
    """Tests pour l'analyseur FinBERT"""
    
    def test_finbert_initialization(self):
        """Test l'initialisation de FinBERT"""
        analyzer = FinBertAnalyzer()
        
        assert analyzer.mode == CONSTANTS.FINBERT_MODE
        assert analyzer.timeout_ms == CONSTANTS.FINBERT_TIMEOUT_MS
        assert not analyzer._model_loaded
    
    def test_score_texts_stub(self):
        """Test le scoring en mode stub"""
        analyzer = FinBertAnalyzer(mode="stub")
        
        texts = [
            "NVIDIA stock is performing well",
            "Market crash expected",
            "Neutral market conditions"
        ]
        
        scores = analyzer.score_texts(texts)
        
        assert len(scores) == len(texts)
        assert all(-1 <= score <= 1 for score in scores)
    
    def test_score_texts_empty(self):
        """Test le scoring avec liste vide"""
        analyzer = FinBertAnalyzer()
        
        scores = analyzer.score_texts([])
        
        assert scores == []

@pytest.mark.unit
class TestLSTMPredictor:
    """Tests pour le prédicteur LSTM"""
    
    def test_lstm_predictor_initialization(self):
        """Test l'initialisation du prédicteur LSTM"""
        predictor = LSTMPredictor("SPY")
        
        assert predictor.ticker == "SPY"
        assert predictor.feature_columns == CONSTANTS.get_feature_columns()
        assert predictor.sequence_length == CONSTANTS.LSTM_SEQUENCE_LENGTH
        assert not predictor.is_loaded
    
    def test_prepare_features(self):
        """Test la préparation des features"""
        predictor = LSTMPredictor("SPY")
        
        # Créer des données de test
        data = pd.DataFrame({
            col: np.random.randn(100) for col in CONSTANTS.get_feature_columns()
        })
        
        features = predictor.prepare_features(data)
        
        assert features is not None
        assert features.shape[1] == len(CONSTANTS.get_feature_columns())
        assert features.shape[0] == len(data)
    
    def test_create_sequences(self):
        """Test la création des séquences"""
        predictor = LSTMPredictor("SPY")
        
        # Créer des features de test
        features = np.random.randn(50, len(CONSTANTS.get_feature_columns()))
        
        X, y = predictor.create_sequences(features)
        
        if X is not None:
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] == CONSTANTS.LSTM_SEQUENCE_LENGTH
            assert X.shape[2] == len(CONSTANTS.get_feature_columns())
    
    def test_create_sequences_insufficient_data(self):
        """Test la création de séquences avec données insuffisantes"""
        predictor = LSTMPredictor("SPY")
        
        # Pas assez de données
        features = np.random.randn(5, len(CONSTANTS.get_feature_columns()))
        
        X, y = predictor.create_sequences(features)
        
        assert X is None
        assert y is None

@pytest.mark.unit
class TestPricePredictor:
    """Tests pour le prédicteur de prix"""
    
    def test_predictor_initialization(self):
        """Test l'initialisation du prédicteur"""
        predictor = PricePredictor("SPY")
        
        assert predictor.ticker == "SPY"
        assert predictor.is_loaded == False
        assert predictor.sequence_length > 0
    
    def test_predictor_with_features(self):
        """Test le prédicteur avec des features"""
        predictor = PricePredictor("SPY")
        
        assert predictor.feature_columns is not None
        assert len(predictor.feature_columns) > 0
    
    def test_predictor_with_invalid_ticker(self):
        """Test avec un ticker invalide"""
        predictor = PricePredictor("INVALID")
        
        assert predictor.ticker == "INVALID"

# Tests d'intégration
@pytest.mark.integration
class TestIntegration:
    """Tests d'intégration des modules core"""
    
    def test_fusion_with_sentiment(self):
        """Test l'intégration fusion + sentiment"""
        # Créer les composants
        fusion = AdaptiveFusion()
        sentiment_analyzer = SentimentAnalyzer()
        
        # Ajouter des données
        sentiment_analyzer.add_sentiment("SPY", 0.5, 0.8)
        sentiment_analyzer.add_price("SPY", 100.0)
        
        # Obtenir le sentiment adaptatif
        adaptive_sentiment = sentiment_analyzer.get_adaptive_sentiment("SPY", 0.5, 0.2, 1.2)
        
        # Utiliser dans la fusion
        result = fusion.add_signal(
            price_signal=0.3,
            sentiment_signal=adaptive_sentiment["adjusted_sentiment"],
            price_volatility=0.2,
            volume_ratio=1.2
        )
        
        assert "fused_signal" in result
        assert -1 <= result["fused_signal"] <= 1
    
    def test_constants_consistency(self):
        """Test la cohérence des constantes"""
        # Vérifier que les constantes sont cohérentes
        assert len(CONSTANTS.TOP_FEATURES) >= CONSTANTS.LSTM_TOP_FEATURES
        assert CONSTANTS.BASE_PRICE_WEIGHT + CONSTANTS.BASE_SENTIMENT_WEIGHT == 1.0
        assert CONSTANTS.BASE_BUY_THRESHOLD > 0
        assert CONSTANTS.BASE_SELL_THRESHOLD < 0
        assert CONSTANTS.LSTM_SEQUENCE_LENGTH > 0
