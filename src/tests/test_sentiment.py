"""
🧪 Tests Sentiment Sentinel2
Tests pour le module sentiment
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.core.sentiment import FinBertAnalyzer, SentimentAnalyzer
from src.constants import CONSTANTS


@pytest.mark.unit
class TestFinBertAnalyzer:
    """Tests pour l'analyseur FinBERT"""
    
    def test_finbert_initialization(self):
        """Test l'initialisation de FinBERT"""
        analyzer = FinBertAnalyzer()
        
        assert analyzer.mode == CONSTANTS.FINBERT_MODE
        assert analyzer.timeout_ms == CONSTANTS.FINBERT_TIMEOUT_MS
        assert not analyzer._model_loaded
        assert analyzer.model is None
        assert analyzer.tokenizer is None
        assert analyzer.device is None
    
    def test_finbert_initialization_custom(self):
        """Test l'initialisation avec paramètres personnalisés"""
        analyzer = FinBertAnalyzer(mode="real", timeout_ms=5000)
        
        assert analyzer.mode == "real"
        assert analyzer.timeout_ms == 5000
        assert not analyzer._model_loaded
    
    def test_lazy_load_model_stub_mode(self):
        """Test le chargement paresseux en mode stub"""
        analyzer = FinBertAnalyzer(mode="stub")
        
        # Le modèle doit être marqué comme chargé sans charger le vrai modèle
        analyzer._lazy_load_model()
        assert analyzer._model_loaded
        assert analyzer.model is None
        assert analyzer.tokenizer is None
    
    @patch('src.core.sentiment.torch')
    @patch('src.core.sentiment.AutoTokenizer')
    @patch('src.core.sentiment.AutoModelForSequenceClassification')
    def test_lazy_load_model_real_mode(self, mock_model_class, mock_tokenizer_class, mock_torch):
        """Test le chargement paresseux en mode réel"""
        # Mock des dépendances
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        analyzer = FinBertAnalyzer(mode="real")
        analyzer._lazy_load_model()
        
        assert analyzer._model_loaded
        assert analyzer.model == mock_model
        assert analyzer.tokenizer == mock_tokenizer
        assert analyzer.device == "cpu"
    
    def test_score_texts_empty(self):
        """Test le scoring avec liste vide"""
        analyzer = FinBertAnalyzer()
        scores = analyzer.score_texts([])
        
        assert scores == []
    
    def test_score_texts_stub_mode(self):
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
    
    def test_score_texts_stub_positive_keywords(self):
        """Test le scoring stub avec mots-clés positifs"""
        analyzer = FinBertAnalyzer(mode="stub")
        
        positive_texts = [
            "Stock is rising and gaining profits",
            "Strong growth and positive outlook",
            "Bull market continues"
        ]
        
        scores = analyzer.score_texts(positive_texts)
        
        # Les scores doivent être positifs
        assert all(score > 0 for score in scores)
    
    def test_score_texts_stub_negative_keywords(self):
        """Test le scoring stub avec mots-clés négatifs"""
        analyzer = FinBertAnalyzer(mode="stub")
        
        negative_texts = [
            "Stock is falling and losing money",
            "Weak performance and negative outlook",
            "Bear market crash"
        ]
        
        scores = analyzer.score_texts(negative_texts)
        
        # Les scores doivent être négatifs
        assert all(score < 0 for score in scores)
    
    def test_score_texts_stub_neutral_keywords(self):
        """Test le scoring stub avec mots-clés neutres"""
        analyzer = FinBertAnalyzer(mode="stub")
        
        neutral_texts = [
            "Stock price remains stable",
            "No significant changes today",
            "Market is quiet"
        ]
        
        scores = analyzer.score_texts(neutral_texts)
        
        # Les scores doivent être proches de zéro
        assert all(abs(score) < 0.5 for score in scores)
    
    @patch('src.core.sentiment.torch')
    @patch('src.core.sentiment.AutoTokenizer')
    @patch('src.core.sentiment.AutoModelForSequenceClassification')
    def test_score_texts_real_mode(self, mock_model_class, mock_tokenizer_class, mock_torch):
        """Test le scoring en mode réel"""
        # Mock des dépendances
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": "mock", "attention_mask": "mock"}
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = "mock_logits"
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock torch.softmax
        mock_torch.softmax.return_value = [[0.1, 0.2, 0.7]]  # positive sentiment
        mock_torch.no_grad.return_value = MagicMock()
        
        analyzer = FinBertAnalyzer(mode="real")
        scores = analyzer.score_texts(["Test text"])
        
        assert len(scores) == 1
        assert -1 <= scores[0] <= 1


@pytest.mark.unit
class TestSentimentAnalyzer:
    """Tests pour l'analyseur de sentiment"""
    
    def test_sentiment_analyzer_initialization(self):
        """Test l'initialisation de l'analyseur de sentiment"""
        analyzer = SentimentAnalyzer()
        
        assert analyzer.window_minutes == CONSTANTS.SENTIMENT_WINDOW
        assert isinstance(analyzer.finbert, FinBertAnalyzer)
        assert len(analyzer.sentiment_data) == 0
        assert len(analyzer.price_data) == 0
        assert len(analyzer.volume_data) == 0
    
    def test_sentiment_analyzer_initialization_custom(self):
        """Test l'initialisation avec fenêtre personnalisée"""
        analyzer = SentimentAnalyzer(window_minutes=30)
        
        assert analyzer.window_minutes == 30
    
    def test_add_sentiment(self):
        """Test l'ajout de sentiment"""
        analyzer = SentimentAnalyzer()
        
        analyzer.add_sentiment("SPY", 0.5, 0.8)
        
        assert "SPY" in analyzer.sentiment_data
        assert len(analyzer.sentiment_data["SPY"]) == 1
        
        sentiment_entry = analyzer.sentiment_data["SPY"][0]
        assert sentiment_entry[1] == 0.5  # sentiment
        assert sentiment_entry[2] == 0.8  # confidence
        assert isinstance(sentiment_entry[0], datetime)  # timestamp
    
    def test_add_sentiment_multiple(self):
        """Test l'ajout de plusieurs sentiments"""
        analyzer = SentimentAnalyzer()
        
        analyzer.add_sentiment("SPY", 0.5, 0.8)
        analyzer.add_sentiment("SPY", 0.3, 0.6)
        analyzer.add_sentiment("NVDA", 0.7, 0.9)
        
        assert len(analyzer.sentiment_data["SPY"]) == 2
        assert len(analyzer.sentiment_data["NVDA"]) == 1
    
    def test_add_price(self):
        """Test l'ajout de prix"""
        analyzer = SentimentAnalyzer()
        
        analyzer.add_price("SPY", 100.0)
        
        assert "SPY" in analyzer.price_data
        assert len(analyzer.price_data["SPY"]) == 1
        assert analyzer.price_data["SPY"][0][1] == 100.0
        assert isinstance(analyzer.price_data["SPY"][0][0], datetime)
    
    def test_add_volume(self):
        """Test l'ajout de volume"""
        analyzer = SentimentAnalyzer()
        
        analyzer.add_volume("SPY", 1000000)
        
        assert "SPY" in analyzer.volume_data
        assert len(analyzer.volume_data["SPY"]) == 1
        assert analyzer.volume_data["SPY"][0][1] == 1000000
        assert isinstance(analyzer.volume_data["SPY"][0][0], datetime)
    
    def test_get_sentiment_no_data(self):
        """Test la récupération de sentiment sans données"""
        analyzer = SentimentAnalyzer()
        
        sentiment = analyzer.get_sentiment("SPY")
        assert sentiment == 0.0
    
    def test_get_sentiment_with_data(self):
        """Test la récupération de sentiment avec données"""
        analyzer = SentimentAnalyzer()
        
        # Ajouter des sentiments
        analyzer.add_sentiment("SPY", 0.5, 0.8)
        analyzer.add_sentiment("SPY", 0.3, 0.6)
        
        sentiment = analyzer.get_sentiment("SPY")
        
        # Doit être une moyenne pondérée
        assert isinstance(sentiment, float)
        assert -1 <= sentiment <= 1
        assert sentiment > 0  # Les deux sentiments sont positifs
    
    def test_get_sentiment_window_filtering(self):
        """Test le filtrage par fenêtre temporelle"""
        analyzer = SentimentAnalyzer(window_minutes=1)
        
        # Ajouter un sentiment récent
        analyzer.add_sentiment("SPY", 0.5, 0.8)
        
        # Attendre un peu et ajouter un sentiment plus ancien
        import time
        time.sleep(0.1)
        
        # Simuler un sentiment plus ancien en modifiant directement les données
        old_time = datetime.now() - timedelta(minutes=2)
        analyzer.sentiment_data["SPY"].append((old_time, 0.8, 0.9))
        
        sentiment = analyzer.get_sentiment("SPY")
        
        # Seul le sentiment récent doit être pris en compte
        assert sentiment == 0.5
    
    def test_get_volatility_no_data(self):
        """Test le calcul de volatilité sans données"""
        analyzer = SentimentAnalyzer()
        
        volatility = analyzer.get_volatility("SPY")
        assert volatility == 0.0
    
    def test_get_volatility_insufficient_data(self):
        """Test le calcul de volatilité avec données insuffisantes"""
        analyzer = SentimentAnalyzer()
        
        analyzer.add_price("SPY", 100.0)
        volatility = analyzer.get_volatility("SPY")
        
        assert volatility == 0.0
    
    def test_get_volatility_with_data(self):
        """Test le calcul de volatilité avec données"""
        analyzer = SentimentAnalyzer()
        
        # Ajouter des prix pour calculer la volatilité
        prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105]
        for price in prices:
            analyzer.add_price("SPY", price)
        
        volatility = analyzer.get_volatility("SPY")
        
        assert isinstance(volatility, float)
        assert volatility >= 0
    
    def test_get_volume_ratio_no_data(self):
        """Test le calcul du ratio de volume sans données"""
        analyzer = SentimentAnalyzer()
        
        ratio = analyzer.get_volume_ratio("SPY", 1000)
        assert ratio == 1.0
    
    def test_get_volume_ratio_with_data(self):
        """Test le calcul du ratio de volume avec données"""
        analyzer = SentimentAnalyzer()
        
        # Ajouter des volumes
        volumes = [1000, 1200, 800, 1500, 900]
        for volume in volumes:
            analyzer.add_volume("SPY", volume)
        
        ratio = analyzer.get_volume_ratio("SPY", 1000)
        
        assert isinstance(ratio, float)
        assert ratio > 0
    
    def test_get_adaptive_sentiment(self):
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
        assert "window_minutes" in result
        
        assert -1 <= result["adjusted_sentiment"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert result["volatility"] == 0.2
        assert result["volume_ratio"] == 1.2
        assert result["window_minutes"] == analyzer.window_minutes
    
    def test_get_adaptive_sentiment_high_volatility(self):
        """Test le sentiment adaptatif en haute volatilité"""
        analyzer = SentimentAnalyzer()
        
        analyzer.add_sentiment("SPY", 0.5, 0.8)
        
        result = analyzer.get_adaptive_sentiment("SPY", 0.5, 0.4, 1.0)  # Haute volatilité
        
        # En haute volatilité, le sentiment doit être réduit
        assert abs(result["adjusted_sentiment"]) < abs(result["base_sentiment"])
        assert result["confidence"] < 0.8
    
    def test_get_adaptive_sentiment_low_volatility(self):
        """Test le sentiment adaptatif en basse volatilité"""
        analyzer = SentimentAnalyzer()
        
        analyzer.add_sentiment("SPY", 0.5, 0.8)
        
        result = analyzer.get_adaptive_sentiment("SPY", 0.5, 0.05, 1.0)  # Basse volatilité
        
        # En basse volatilité, le sentiment doit être augmenté
        assert abs(result["adjusted_sentiment"]) > abs(result["base_sentiment"])
        assert result["confidence"] > 0.7
    
    def test_get_adaptive_sentiment_high_volume(self):
        """Test le sentiment adaptatif avec volume élevé"""
        analyzer = SentimentAnalyzer()
        
        analyzer.add_sentiment("SPY", 0.5, 0.8)
        
        result = analyzer.get_adaptive_sentiment("SPY", 0.5, 0.2, 2.0)  # Volume élevé
        
        # Avec volume élevé, la confiance doit être augmentée
        assert result["confidence"] > 0.7
    
    def test_analyze_news_batch_empty(self):
        """Test l'analyse de news avec liste vide"""
        analyzer = SentimentAnalyzer()
        
        results = analyzer.analyze_news_batch([])
        assert results == []
    
    def test_analyze_news_batch(self):
        """Test l'analyse de news"""
        analyzer = SentimentAnalyzer()
        
        news_items = [
            {"title": "NVIDIA stock rises", "body": "Strong earnings report"},
            {"title": "Market crash fears", "body": "Economic uncertainty"},
            {"title": "Stable market", "body": "No major changes"}
        ]
        
        results = analyzer.analyze_news_batch(news_items)
        
        assert len(results) == len(news_items)
        for result in results:
            assert "sentiment_score" in result
            assert "sentiment_confidence" in result
            assert -1 <= result["sentiment_score"] <= 1
            assert 0 <= result["sentiment_confidence"] <= 1
    
    def test_get_sentiment_summary_no_data(self):
        """Test le résumé de sentiment sans données"""
        analyzer = SentimentAnalyzer()
        
        summary = analyzer.get_sentiment_summary("SPY")
        
        assert summary["ticker"] == "SPY"
        assert summary["total_sentiments"] == 0
        assert summary["avg_sentiment"] == 0.0
        assert summary["latest_sentiment"] == 0.0
        assert summary["confidence"] == 0.0
    
    def test_get_sentiment_summary_with_data(self):
        """Test le résumé de sentiment avec données"""
        analyzer = SentimentAnalyzer()
        
        # Ajouter des sentiments
        analyzer.add_sentiment("SPY", 0.5, 0.8)
        analyzer.add_sentiment("SPY", 0.3, 0.6)
        analyzer.add_price("SPY", 100.0)
        
        summary = analyzer.get_sentiment_summary("SPY")
        
        assert summary["ticker"] == "SPY"
        assert summary["total_sentiments"] == 2
        assert summary["avg_sentiment"] > 0
        assert summary["latest_sentiment"] == 0.3
        assert summary["confidence"] > 0
        assert "volatility" in summary
        assert "window_minutes" in summary
    
    def test_reset(self):
        """Test la réinitialisation de l'analyseur"""
        analyzer = SentimentAnalyzer()
        
        # Ajouter des données
        analyzer.add_sentiment("SPY", 0.5, 0.8)
        analyzer.add_price("SPY", 100.0)
        analyzer.add_volume("SPY", 1000000)
        
        # Réinitialiser
        analyzer.reset()
        
        assert len(analyzer.sentiment_data) == 0
        assert len(analyzer.price_data) == 0
        assert len(analyzer.volume_data) == 0
    
    def test_data_retention(self):
        """Test la rétention des données (24h)"""
        analyzer = SentimentAnalyzer()
        
        # Ajouter des données
        analyzer.add_sentiment("SPY", 0.5, 0.8)
        analyzer.add_price("SPY", 100.0)
        analyzer.add_volume("SPY", 1000000)
        
        # Simuler des données anciennes
        old_time = datetime.now() - timedelta(hours=25)
        analyzer.sentiment_data["SPY"].append((old_time, 0.8, 0.9))
        analyzer.price_data["SPY"].append((old_time, 95.0))
        analyzer.volume_data["SPY"].append((old_time, 2000000))
        
        # Ajouter de nouvelles données pour déclencher le nettoyage
        analyzer.add_sentiment("SPY", 0.3, 0.6)
        
        # Vérifier que les données anciennes ont été supprimées
        assert len(analyzer.sentiment_data["SPY"]) == 2  # Seulement les 2 récentes
        assert len(analyzer.price_data["SPY"]) == 1  # Seulement la récente
        assert len(analyzer.volume_data["SPY"]) == 1  # Seulement la récente
