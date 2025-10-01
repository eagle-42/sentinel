"""
🧪 Tests Fonctionnels d'Intégration Sentinel2
Tests de qualité qui valident le comportement réel du système
"""

import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import CONSTANTS
from src.core.fusion import AdaptiveFusion
from src.core.prediction import LSTMPredictor
from src.core.sentiment import SentimentAnalyzer
from src.data.unified_storage import UnifiedDataStorage


class TestFunctionalIntegration:
    """Tests fonctionnels d'intégration du système complet"""

    @pytest.fixture
    def temp_data_dir(self):
        """Répertoire temporaire pour les tests"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_price_data(self):
        """Données de prix de test réalistes"""
        dates = pd.date_range(start="2025-01-01", periods=100, freq="1min")
        np.random.seed(42)  # Pour la reproductibilité

        # Générer des données de prix réalistes
        base_price = 100.0
        returns = np.random.normal(0, 0.001, 100)  # Volatilité réaliste
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                "close": prices,
                "volume": np.random.randint(1000, 10000, 100),
                "ticker": "SPY",
            }
        )

        return data

    @pytest.fixture
    def sample_news_data(self):
        """Données de news de test réalistes"""
        news_items = [
            {
                "title": "NVIDIA Stock Surges on Strong Earnings",
                "summary": "NVIDIA reported better than expected earnings",
                "published_at": datetime.now() - timedelta(hours=1),
                "source": "Reuters",
                "ticker": "NVDA",
            },
            {
                "title": "Market Volatility Continues",
                "summary": "Investors remain cautious amid economic uncertainty",
                "published_at": datetime.now() - timedelta(hours=2),
                "source": "Bloomberg",
                "ticker": "SPY",
            },
            {
                "title": "Tech Stocks Lead Market Higher",
                "summary": "Technology sector shows strong performance",
                "published_at": datetime.now() - timedelta(hours=3),
                "source": "CNBC",
                "ticker": "NVDA",
            },
        ]

        return pd.DataFrame(news_items)

    def test_unified_storage_price_workflow(self, temp_data_dir, sample_price_data):
        """Test le workflow complet de stockage des prix"""
        # Créer le stockage avec répertoire temporaire
        storage = UnifiedDataStorage()
        storage.data_root = temp_data_dir
        storage.raw_dir = temp_data_dir / "raw"
        storage.raw_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder les données de prix
        file_path = storage.save_price_data("SPY", sample_price_data, "1min", "yfinance")

        # Vérifier que le fichier existe
        assert file_path.exists()

        # Charger les données
        loaded_data = storage.load_price_data("SPY", "1min", "yfinance")

        # Vérifier l'intégrité des données
        assert len(loaded_data) == len(sample_price_data)
        assert "timestamp" in loaded_data.columns
        assert "close" in loaded_data.columns
        assert loaded_data["ticker"].iloc[0] == "SPY"

        # Vérifier les métadonnées
        assert "ticker" in loaded_data.attrs
        assert loaded_data.attrs["ticker"] == "SPY"

    def test_unified_storage_news_workflow(self, temp_data_dir, sample_news_data):
        """Test le workflow complet de stockage des news"""
        storage = UnifiedDataStorage()
        storage.data_root = temp_data_dir
        storage.raw_dir = temp_data_dir / "raw"
        storage.raw_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder les données de news
        file_path = storage.save_news_data(sample_news_data, "SPY", "rss")

        # Vérifier que le fichier existe
        assert file_path.exists()

        # Charger les données
        loaded_data = storage.load_news_data("SPY", "rss")

        # Vérifier l'intégrité des données
        assert len(loaded_data) == len(sample_news_data)
        assert "title" in loaded_data.columns
        assert "published_at" in loaded_data.columns
        assert loaded_data["ticker"].iloc[0] == "SPY"

    def test_adaptive_fusion_real_scenario(self):
        """Test la fusion adaptative avec un scénario réel"""
        fusion = AdaptiveFusion()

        # Simuler des signaux sur plusieurs périodes
        scenarios = [
            # Période de haute volatilité
            {"price_signal": 0.8, "sentiment_signal": 0.6, "volatility": 0.3, "volume_ratio": 1.5},
            # Période de basse volatilité
            {"price_signal": 0.2, "sentiment_signal": 0.3, "volatility": 0.1, "volume_ratio": 0.8},
            # Période de stress du marché
            {"price_signal": -0.5, "sentiment_signal": -0.7, "volatility": 0.4, "volume_ratio": 2.0},
        ]

        results = []
        for scenario in scenarios:
            result = fusion.add_signal(**scenario)
            results.append(result)

            # Vérifier que la fusion est cohérente
            assert -1 <= result["fused_signal"] <= 1
            assert "weights" in result
            assert "regime" in result

        # Vérifier que les poids s'adaptent
        first_weights = results[0]["weights"]
        last_weights = results[-1]["weights"]

        # Les poids doivent avoir changé entre haute et basse volatilité
        assert first_weights != last_weights

        # Vérifier le résumé
        summary = fusion.get_fusion_summary()
        assert summary["total_signals"] == len(scenarios)
        assert "avg_fused_signal" in summary

    def test_sentiment_analyzer_real_workflow(self):
        """Test l'analyseur de sentiment avec un workflow réel"""
        analyzer = SentimentAnalyzer()

        # Ajouter des données de sentiment
        sentiments = [0.5, -0.3, 0.8, 0.1, -0.6]
        confidences = [0.8, 0.6, 0.9, 0.4, 0.7]

        for sent, conf in zip(sentiments, confidences):
            analyzer.add_sentiment("SPY", sent, conf)

        # Ajouter des prix pour le calcul de volatilité
        prices = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105]
        for price in prices:
            analyzer.add_price("SPY", price)

        # Tester la récupération du sentiment
        avg_sentiment = analyzer.get_sentiment("SPY")
        assert -1 <= avg_sentiment <= 1

        # Tester le calcul de volatilité
        volatility = analyzer.get_volatility("SPY")
        assert volatility >= 0

        # Tester le sentiment adaptatif
        adaptive_result = analyzer.get_adaptive_sentiment("SPY", 0.5, 0.2, 1.2)
        assert "adjusted_sentiment" in adaptive_result
        assert "confidence" in adaptive_result
        assert -1 <= adaptive_result["adjusted_sentiment"] <= 1
        assert 0 <= adaptive_result["confidence"] <= 1

    def test_lstm_predictor_data_preparation(self, sample_price_data):
        """Test la préparation des données pour le LSTM"""
        predictor = LSTMPredictor("SPY")

        # Créer des features techniques basiques
        data = sample_price_data.copy()
        data["returns"] = data["close"].pct_change()
        data["volume_ma"] = data["volume"].rolling(20).mean()
        data["price_ma"] = data["close"].rolling(20).mean()

        # Ajouter quelques features des constantes
        for feature in CONSTANTS.get_feature_columns()[:5]:  # Utiliser les 5 premières
            data[feature] = np.random.randn(len(data))

        # Préparer les features
        features = predictor.prepare_features(data)

        if features is not None:
            assert features.shape[1] == len(CONSTANTS.get_feature_columns())
            assert features.shape[0] == len(data)

            # Vérifier que les features sont normalisées
            assert np.allclose(features.mean(axis=0), 0, atol=1e-6)  # Moyenne proche de 0
            assert np.allclose(features.std(axis=0), 1, atol=1e-6)  # Écart-type proche de 1

    def test_end_to_end_data_pipeline(self, temp_data_dir, sample_price_data, sample_news_data):
        """Test le pipeline complet de données"""
        # Créer le stockage
        storage = UnifiedDataStorage()
        storage.data_root = temp_data_dir
        storage.raw_dir = temp_data_dir / "raw"
        storage.raw_dir.mkdir(parents=True, exist_ok=True)

        # 1. Sauvegarder les prix
        price_file = storage.save_price_data("SPY", sample_price_data, "1min", "yfinance")
        assert price_file.exists()

        # 2. Sauvegarder les news
        news_file = storage.save_news_data(sample_news_data, "SPY", "rss")
        assert news_file.exists()

        # 3. Charger et vérifier les données
        loaded_prices = storage.load_price_data("SPY", "1min", "yfinance")
        loaded_news = storage.load_news_data("SPY", "rss")

        assert len(loaded_prices) == len(sample_price_data)
        assert len(loaded_news) == len(sample_news_data)

        # 4. Tester l'intégration avec l'analyseur de sentiment
        analyzer = SentimentAnalyzer()

        # Ajouter des prix pour la volatilité
        for _, row in loaded_prices.iterrows():
            analyzer.add_price("SPY", row["close"])

        # Ajouter des sentiments simulés
        for _, row in loaded_news.iterrows():
            # Simuler un score de sentiment basé sur le titre
            sentiment = 0.5 if "surge" in row["title"].lower() else -0.3
            analyzer.add_sentiment("SPY", sentiment, 0.8)

        # Vérifier que l'analyseur fonctionne
        avg_sentiment = analyzer.get_sentiment("SPY")
        assert -1 <= avg_sentiment <= 1

        # 5. Tester la fusion adaptative
        fusion = AdaptiveFusion()

        # Simuler un signal de fusion
        volatility = analyzer.get_volatility("SPY")
        volume_ratio = 1.2

        result = fusion.add_signal(
            price_signal=0.3, sentiment_signal=avg_sentiment, price_volatility=volatility, volume_ratio=volume_ratio
        )

        assert "fused_signal" in result
        assert -1 <= result["fused_signal"] <= 1

    def test_performance_benchmarks(self):
        """Test les performances du système"""
        import time

        # Test de performance de la fusion adaptative
        fusion = AdaptiveFusion()

        start_time = time.time()
        for i in range(1000):
            fusion.add_signal(
                price_signal=np.random.normal(0, 0.5),
                sentiment_signal=np.random.normal(0, 0.3),
                price_volatility=np.random.uniform(0.1, 0.4),
                volume_ratio=np.random.uniform(0.5, 2.0),
            )
        fusion_time = time.time() - start_time

        # La fusion doit être rapide (< 1 seconde pour 1000 signaux)
        assert fusion_time < 1.0

        # Test de performance de l'analyseur de sentiment
        analyzer = SentimentAnalyzer()

        start_time = time.time()
        for i in range(100):
            analyzer.add_sentiment("SPY", np.random.normal(0, 0.5), 0.8)
            analyzer.add_price("SPY", 100 + np.random.normal(0, 5))
        sentiment_time = time.time() - start_time

        # L'analyseur doit être rapide (< 0.1 seconde pour 100 opérations)
        assert sentiment_time < 0.1

    def test_error_handling_and_recovery(self):
        """Test la gestion d'erreurs et la récupération"""
        # Test avec des données invalides
        fusion = AdaptiveFusion()

        # Test avec des valeurs extrêmes
        result = fusion.add_signal(
            price_signal=float("inf"), sentiment_signal=float("-inf"), price_volatility=0.0, volume_ratio=0.0
        )

        # Le système doit gérer les valeurs extrêmes
        assert "fused_signal" in result
        assert not np.isnan(result["fused_signal"])
        assert not np.isinf(result["fused_signal"])

        # Test avec des données manquantes
        analyzer = SentimentAnalyzer()

        # Tester la récupération avec des données vides
        sentiment = analyzer.get_sentiment("NONEXISTENT")
        assert sentiment == 0.0

        volatility = analyzer.get_volatility("NONEXISTENT")
        assert volatility == 0.0

    def test_constants_consistency(self):
        """Test la cohérence des constantes globales"""
        # Vérifier que les constantes sont cohérentes
        assert len(CONSTANTS.TOP_FEATURES) >= CONSTANTS.LSTM_TOP_FEATURES
        assert CONSTANTS.BASE_PRICE_WEIGHT + CONSTANTS.BASE_SENTIMENT_WEIGHT == 1.0
        assert CONSTANTS.BASE_BUY_THRESHOLD > 0
        assert CONSTANTS.BASE_SELL_THRESHOLD < 0
        assert CONSTANTS.LSTM_SEQUENCE_LENGTH > 0

        # Vérifier que les chemins existent
        assert CONSTANTS.DATA_ROOT.exists()
        assert CONSTANTS.MODELS_DIR.exists()

        # Vérifier les méthodes utilitaires
        feature_columns = CONSTANTS.get_feature_columns()
        assert len(feature_columns) == CONSTANTS.LSTM_TOP_FEATURES

        trading_config = CONSTANTS.get_trading_config()
        assert "base_buy_threshold" in trading_config
        assert "base_sell_threshold" in trading_config

        lstm_config = CONSTANTS.get_lstm_config()
        assert "sequence_length" in lstm_config
        assert "hidden_sizes" in lstm_config
