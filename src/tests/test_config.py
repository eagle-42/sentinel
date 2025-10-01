"""
🧪 Tests Config Sentinel2
Tests pour le module config
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import SentinelConfig, config
from src.constants import CONSTANTS


@pytest.mark.unit
class TestSentinelConfig:
    """Tests pour la classe SentinelConfig"""

    def test_config_initialization(self):
        """Test l'initialisation de la configuration"""
        test_config = SentinelConfig()

        assert isinstance(test_config._config, dict)
        assert "tickers" in test_config._config
        assert "finbert" in test_config._config
        assert "news" in test_config._config
        assert "trading" in test_config._config
        assert "lstm" in test_config._config
        assert "fusion" in test_config._config
        assert "paths" in test_config._config
        assert "api" in test_config._config
        assert "gui" in test_config._config
        assert "logging" in test_config._config

    def test_get_tickers(self):
        """Test la récupération des tickers"""
        test_config = SentinelConfig()
        tickers = test_config.get_tickers()

        assert isinstance(tickers, dict)
        assert "SPY" in tickers
        assert "NVDA" in tickers
        assert tickers["SPY"] == "S&P 500 ETF"
        assert tickers["NVDA"] == "Nvidia"

    def test_get_finbert_config(self):
        """Test la récupération de la configuration FinBERT"""
        test_config = SentinelConfig()
        finbert_config = test_config.get_finbert_config()

        assert isinstance(finbert_config, dict)
        assert "mode" in finbert_config
        assert "timeout_ms" in finbert_config
        assert "model_name" in finbert_config
        assert "batch_size" in finbert_config

        assert finbert_config["mode"] == CONSTANTS.FINBERT_MODE
        assert finbert_config["timeout_ms"] == CONSTANTS.FINBERT_TIMEOUT_MS
        assert finbert_config["model_name"] == CONSTANTS.FINBERT_MODEL_NAME
        assert finbert_config["batch_size"] == CONSTANTS.FINBERT_BATCH_SIZE

    def test_get_news_config(self):
        """Test la récupération de la configuration des news"""
        test_config = SentinelConfig()
        news_config = test_config.get_news_config()

        assert isinstance(news_config, dict)
        assert "interval" in news_config
        assert "feeds" in news_config
        assert "sentiment_window" in news_config
        assert "newsapi_enabled" in news_config
        assert "newsapi_key" in news_config
        assert "newsapi_language" in news_config
        assert "newsapi_country" in news_config
        assert "newsapi_sources" in news_config

        assert news_config["interval"] == 240  # Valeur du .env corrigée
        assert news_config["sentiment_window"] == CONSTANTS.SENTIMENT_WINDOW
        assert isinstance(news_config["feeds"], list)
        assert len(news_config["feeds"]) > 0

    def test_get_trading_config(self):
        """Test la récupération de la configuration de trading"""
        test_config = SentinelConfig()
        trading_config = test_config.get_trading_config()

        assert isinstance(trading_config, dict)
        assert "price_interval" in trading_config
        assert "price_period" in trading_config
        assert "fusion_mode" in trading_config
        assert "buy_threshold" in trading_config
        assert "sell_threshold" in trading_config
        assert "hold_confidence" in trading_config

        assert trading_config["price_interval"] == CONSTANTS.PRICE_INTERVAL
        assert trading_config["price_period"] == CONSTANTS.PRICE_PERIOD
        assert trading_config["fusion_mode"] == "adaptive"  # Valeur du .env corrigée
        assert trading_config["buy_threshold"] == CONSTANTS.BASE_BUY_THRESHOLD
        assert trading_config["sell_threshold"] == CONSTANTS.BASE_SELL_THRESHOLD
        assert trading_config["hold_confidence"] == CONSTANTS.HOLD_CONFIDENCE

    def test_get_lstm_config(self):
        """Test la récupération de la configuration LSTM"""
        test_config = SentinelConfig()
        lstm_config = test_config.get_lstm_config()

        assert isinstance(lstm_config, dict)
        assert "sequence_length" in lstm_config
        assert "top_features" in lstm_config
        assert "prediction_horizon" in lstm_config
        assert "hidden_sizes" in lstm_config
        assert "dropout_rate" in lstm_config
        assert "epochs" in lstm_config
        assert "batch_size" in lstm_config
        assert "patience" in lstm_config
        assert "learning_rate" in lstm_config

        # Vérifier que la configuration correspond aux constantes
        expected_config = CONSTANTS.get_lstm_config()
        assert lstm_config == expected_config

    def test_get_fusion_config(self):
        """Test la récupération de la configuration de fusion"""
        test_config = SentinelConfig()
        fusion_config = test_config.get_fusion_config()

        assert isinstance(fusion_config, dict)
        assert "mode" in fusion_config
        assert "base_price_weight" in fusion_config
        assert "base_sentiment_weight" in fusion_config
        assert "max_weight_change" in fusion_config
        assert "regularization_factor" in fusion_config

        assert fusion_config["mode"] == "adaptive"  # Valeur du .env corrigée
        assert fusion_config["base_price_weight"] == CONSTANTS.BASE_PRICE_WEIGHT
        assert fusion_config["base_sentiment_weight"] == CONSTANTS.BASE_SENTIMENT_WEIGHT
        assert fusion_config["max_weight_change"] == CONSTANTS.MAX_WEIGHT_CHANGE
        assert fusion_config["regularization_factor"] == CONSTANTS.REGULARIZATION_FACTOR

    def test_get_paths(self):
        """Test la récupération des chemins"""
        test_config = SentinelConfig()
        paths = test_config.get_paths()

        assert isinstance(paths, dict)
        assert "data_root" in paths
        assert "prices_dir" in paths
        assert "news_dir" in paths
        assert "sentiment_dir" in paths
        assert "models_dir" in paths
        assert "logs_dir" in paths

        assert paths["data_root"] == str(CONSTANTS.DATA_ROOT)
        assert paths["prices_dir"] == str(CONSTANTS.PRICES_DIR)
        assert paths["news_dir"] == str(CONSTANTS.NEWS_DIR)
        assert paths["sentiment_dir"] == str(CONSTANTS.SENTIMENT_DIR)
        assert paths["models_dir"] == str(CONSTANTS.MODELS_DIR)
        assert paths["logs_dir"] == str(CONSTANTS.LOGS_DIR)

    def test_get_api_config(self):
        """Test la récupération de la configuration API"""
        test_config = SentinelConfig()
        api_config = test_config.get_api_config()

        assert isinstance(api_config, dict)
        assert "host" in api_config
        assert "port" in api_config
        assert "title" in api_config
        assert "version" in api_config

        assert api_config["host"] == CONSTANTS.API_HOST
        assert api_config["port"] == CONSTANTS.API_PORT
        assert api_config["title"] == CONSTANTS.API_TITLE
        assert api_config["version"] == CONSTANTS.API_VERSION

    def test_get_gui_config(self):
        """Test la récupération de la configuration GUI"""
        test_config = SentinelConfig()
        gui_config = test_config.get_gui_config()

        assert isinstance(gui_config, dict)
        assert "host" in gui_config
        assert "port" in gui_config
        assert "title" in gui_config
        assert "theme" in gui_config

        assert gui_config["host"] == CONSTANTS.GUI_HOST
        assert gui_config["port"] == CONSTANTS.GUI_PORT
        assert gui_config["title"] == CONSTANTS.GUI_TITLE
        assert gui_config["theme"] == CONSTANTS.GUI_THEME

    def test_get_logging_config(self):
        """Test la récupération de la configuration de logging"""
        test_config = SentinelConfig()
        logging_config = test_config.get_logging_config()

        assert isinstance(logging_config, dict)
        assert "level" in logging_config
        assert "format" in logging_config
        assert "rotation" in logging_config
        assert "retention" in logging_config

        assert logging_config["level"] == CONSTANTS.LOG_LEVEL
        assert logging_config["format"] == CONSTANTS.LOG_FORMAT
        assert logging_config["rotation"] == CONSTANTS.LOG_ROTATION
        assert logging_config["retention"] == CONSTANTS.LOG_RETENTION

    def test_get_method(self):
        """Test la méthode get générique"""
        test_config = SentinelConfig()

        # Test récupération simple
        tickers = test_config.get("tickers")
        assert isinstance(tickers, dict)

        # Test récupération imbriquée
        finbert_mode = test_config.get("finbert.mode")
        assert finbert_mode == CONSTANTS.FINBERT_MODE

        # Test avec valeur par défaut
        non_existent = test_config.get("non.existent.key", "default")
        assert non_existent == "default"

        # Test avec clé inexistante sans défaut
        non_existent_none = test_config.get("non.existent.key")
        assert non_existent_none is None

    def test_is_finbert_real_mode(self):
        """Test la vérification du mode FinBERT réel"""
        test_config = SentinelConfig()

        # Par défaut, FinBERT est en mode stub
        assert not test_config.is_finbert_real_mode()

        # Tester avec mode réel
        with patch.object(test_config, "get", return_value="real"):
            assert test_config.is_finbert_real_mode()

    def test_is_fusion_adaptive(self):
        """Test la vérification de la fusion adaptative"""
        test_config = SentinelConfig()

        # Avec .env, la fusion est adaptive
        assert test_config.is_fusion_adaptive()

        # Tester avec mode fixe
        with patch.object(test_config, "get", return_value="fixed"):
            assert not test_config.is_fusion_adaptive()

    def test_get_data_path(self):
        """Test la récupération des chemins de données"""
        test_config = SentinelConfig()

        # Test prix
        price_path = test_config.get_data_path("prices", "SPY", "1m")
        expected_path = CONSTANTS.get_data_path("prices", "SPY", "1m")
        assert price_path == expected_path

        # Test news
        news_path = test_config.get_data_path("news", "NVDA")
        expected_path = CONSTANTS.get_data_path("news", "NVDA")
        assert news_path == expected_path

    def test_get_model_path(self):
        """Test la récupération des chemins de modèles"""
        test_config = SentinelConfig()

        # Test sans version
        model_path = test_config.get_model_path("SPY")
        expected_path = CONSTANTS.get_model_path("SPY")
        assert model_path == expected_path

        # Test avec version
        model_path_v1 = test_config.get_model_path("NVDA", 1)
        expected_path_v1 = CONSTANTS.get_model_path("NVDA", 1)
        assert model_path_v1 == expected_path_v1

    def test_validate(self):
        """Test la validation de la configuration"""
        test_config = SentinelConfig()

        # La configuration par défaut doit être valide
        assert test_config.validate()

        # Tester avec une configuration invalide
        with patch.object(test_config, "get", return_value=None):
            assert not test_config.validate()

    def test_to_dict(self):
        """Test la conversion en dictionnaire"""
        test_config = SentinelConfig()
        config_dict = test_config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict == test_config._config

        # Vérifier que c'est une copie
        assert config_dict is not test_config._config

    @patch.dict(
        os.environ,
        {
            "TICKERS": "AAPL:Apple Inc,MSFT:Microsoft Corporation",
            "FINBERT_MODE": "real",
            "NEWS_FLOW_INTERVAL": "300",
            "PRICE_INTERVAL": "5m",
            "FUSION_MODE": "fixed",
        },
    )
    def test_environment_variables(self):
        """Test la prise en compte des variables d'environnement"""
        test_config = SentinelConfig()

        # Vérifier que les variables d'environnement sont prises en compte
        tickers = test_config.get_tickers()
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert tickers["AAPL"] == "Apple Inc"
        assert tickers["MSFT"] == "Microsoft Corporation"

        assert test_config.get("finbert.mode") == "real"
        assert test_config.get("news.interval") == 300
        assert test_config.get("trading.price_interval") == "5m"
        assert test_config.get("fusion.mode") == "fixed"

    def test_news_feeds_parsing(self):
        """Test le parsing des feeds de news"""
        with patch.dict(os.environ, {"NEWS_FEEDS": "http://feed1.com,http://feed2.com,http://feed3.com"}):
            test_config = SentinelConfig()
            news_config = test_config.get_news_config()

            assert len(news_config["feeds"]) == 3
            assert "http://feed1.com" in news_config["feeds"]
            assert "http://feed2.com" in news_config["feeds"]
            assert "http://feed3.com" in news_config["feeds"]

    def test_tickers_parsing(self):
        """Test le parsing des tickers"""
        with patch.dict(os.environ, {"TICKERS": "AAPL:Apple,MSFT:Microsoft,GOOGL:Google"}):
            test_config = SentinelConfig()
            tickers = test_config.get_tickers()

            assert len(tickers) == 3
            assert tickers["AAPL"] == "Apple"
            assert tickers["MSFT"] == "Microsoft"
            assert tickers["GOOGL"] == "Google"

    def test_global_config_instance(self):
        """Test l'instance globale de configuration"""
        # Vérifier que l'instance globale existe
        assert isinstance(config, SentinelConfig)

        # Vérifier qu'elle est valide
        assert config.validate()

        # Vérifier qu'elle a les propriétés attendues
        assert hasattr(config, "get")
        assert hasattr(config, "get_tickers")
        assert hasattr(config, "get_finbert_config")
        assert hasattr(config, "validate")
