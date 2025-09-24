"""
🧪 Tests Constants Sentinel2
Tests pour le module constants
"""

import pytest
from pathlib import Path
from src.constants import CONSTANTS, SentinelConstants


@pytest.mark.unit
class TestSentinelConstants:
    """Tests pour la classe SentinelConstants"""
    
    def test_constants_initialization(self):
        """Test l'initialisation des constantes"""
        assert isinstance(CONSTANTS, SentinelConstants)
        assert len(CONSTANTS.TICKERS) > 0
        assert "SPY" in CONSTANTS.TICKERS
        assert "NVDA" in CONSTANTS.TICKERS
    
    def test_ticker_names(self):
        """Test les noms des tickers"""
        assert "SPY" in CONSTANTS.TICKER_NAMES
        assert "NVDA" in CONSTANTS.TICKER_NAMES
        assert CONSTANTS.TICKER_NAMES["SPY"] == "S&P 500 ETF"
        assert CONSTANTS.TICKER_NAMES["NVDA"] == "NVIDIA Corporation"
    
    def test_lstm_config(self):
        """Test la configuration LSTM"""
        assert CONSTANTS.LSTM_SEQUENCE_LENGTH > 0
        assert CONSTANTS.LSTM_TOP_FEATURES > 0
        assert CONSTANTS.LSTM_PREDICTION_HORIZON > 0
        assert len(CONSTANTS.LSTM_HIDDEN_SIZES) > 0
        assert 0 <= CONSTANTS.LSTM_DROPOUT_RATE <= 1
        assert CONSTANTS.LSTM_EPOCHS > 0
        assert CONSTANTS.LSTM_BATCH_SIZE > 0
        assert CONSTANTS.LSTM_PATIENCE > 0
        assert CONSTANTS.LSTM_LEARNING_RATE > 0
    
    def test_top_features(self):
        """Test les top features"""
        assert len(CONSTANTS.TOP_FEATURES) > 0
        assert len(CONSTANTS.TOP_FEATURES) >= CONSTANTS.LSTM_TOP_FEATURES
        assert all(isinstance(feature, str) for feature in CONSTANTS.TOP_FEATURES)
    
    def test_finbert_config(self):
        """Test la configuration FinBERT"""
        assert CONSTANTS.FINBERT_MODE in ["stub", "real"]
        assert CONSTANTS.FINBERT_TIMEOUT_MS > 0
        assert isinstance(CONSTANTS.FINBERT_MODEL_NAME, str)
        assert CONSTANTS.FINBERT_BATCH_SIZE > 0
    
    def test_trading_thresholds(self):
        """Test les seuils de trading"""
        assert CONSTANTS.BUY_THRESHOLD > 0
        assert CONSTANTS.SELL_THRESHOLD < 0
        assert 0 <= CONSTANTS.HOLD_CONFIDENCE <= 1
        assert CONSTANTS.SUCCESS_THRESHOLD > 0
    
    def test_paths(self):
        """Test les chemins de données"""
        assert isinstance(CONSTANTS.PROJECT_ROOT, Path)
        assert isinstance(CONSTANTS.DATA_ROOT, Path)
        assert isinstance(CONSTANTS.PRICES_DIR, Path)
        assert isinstance(CONSTANTS.NEWS_DIR, Path)
        assert isinstance(CONSTANTS.SENTIMENT_DIR, Path)
        assert isinstance(CONSTANTS.MODELS_DIR, Path)
        assert isinstance(CONSTANTS.LOGS_DIR, Path)
        
        # Vérifier que les chemins sont relatifs au projet
        assert CONSTANTS.DATA_ROOT == CONSTANTS.PROJECT_ROOT / "data"
        assert CONSTANTS.PRICES_DIR == CONSTANTS.REALTIME_DIR / "prices"
        assert CONSTANTS.NEWS_DIR == CONSTANTS.REALTIME_DIR / "news"
        assert CONSTANTS.SENTIMENT_DIR == CONSTANTS.REALTIME_DIR / "sentiment"
        assert CONSTANTS.MODELS_DIR == CONSTANTS.DATA_ROOT / "models"
        assert CONSTANTS.LOGS_DIR == CONSTANTS.DATA_ROOT / "logs"
    
    def test_fusion_config(self):
        """Test la configuration de fusion"""
        assert CONSTANTS.FUSION_MODE in ["fixed", "adaptive"]
        assert 0 <= CONSTANTS.BASE_PRICE_WEIGHT <= 1
        assert 0 <= CONSTANTS.BASE_SENTIMENT_WEIGHT <= 1
        assert abs(CONSTANTS.BASE_PRICE_WEIGHT + CONSTANTS.BASE_SENTIMENT_WEIGHT - 1.0) < 1e-6
        assert 0 <= CONSTANTS.MAX_WEIGHT_CHANGE <= 1
        assert 0 <= CONSTANTS.REGULARIZATION_FACTOR <= 1
    
    def test_news_feeds(self):
        """Test les feeds de news"""
        assert len(CONSTANTS.NEWS_FEEDS) > 0
        assert all(isinstance(feed, str) for feed in CONSTANTS.NEWS_FEEDS)
        assert all(feed.startswith("http") for feed in CONSTANTS.NEWS_FEEDS)
    
    def test_performance_targets(self):
        """Test les cibles de performance"""
        assert 0 <= CONSTANTS.TARGET_GLOBAL_SCORE <= 1
        assert 0 <= CONSTANTS.TARGET_DIRECTION_ACCURACY <= 1
        assert CONSTANTS.TARGET_LATENCY_MS > 0
        assert CONSTANTS.TARGET_MAPE > 0
    
    def test_gui_config(self):
        """Test la configuration GUI"""
        assert isinstance(CONSTANTS.GUI_HOST, str)
        assert isinstance(CONSTANTS.GUI_PORT, int)
        assert 1024 <= CONSTANTS.GUI_PORT <= 65535
        assert isinstance(CONSTANTS.GUI_TITLE, str)
        assert isinstance(CONSTANTS.GUI_THEME, str)
    
    def test_api_config(self):
        """Test la configuration API"""
        assert isinstance(CONSTANTS.API_HOST, str)
        assert isinstance(CONSTANTS.API_PORT, int)
        assert 1024 <= CONSTANTS.API_PORT <= 65535
        assert isinstance(CONSTANTS.API_TITLE, str)
        assert isinstance(CONSTANTS.API_VERSION, str)
    
    def test_logging_config(self):
        """Test la configuration de logging"""
        assert CONSTANTS.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert isinstance(CONSTANTS.LOG_FORMAT, str)
        assert isinstance(CONSTANTS.LOG_ROTATION, str)
        assert isinstance(CONSTANTS.LOG_RETENTION, str)
    
    def test_get_data_path(self):
        """Test la méthode get_data_path"""
        # Test prix
        price_path = CONSTANTS.get_data_path("prices", "SPY", "1m")
        expected_price_path = CONSTANTS.PRICES_DIR / "spy_1m.parquet"
        assert price_path == expected_price_path
        
        # Test news
        news_path = CONSTANTS.get_data_path("news", "NVDA")
        expected_news_path = CONSTANTS.NEWS_DIR / "nvda_news.parquet"
        assert news_path == expected_news_path
        
        # Test sentiment
        sentiment_path = CONSTANTS.get_data_path("sentiment", "SPY")
        expected_sentiment_path = CONSTANTS.SENTIMENT_DIR / "spy_sentiment.parquet"
        assert sentiment_path == expected_sentiment_path
        
        # Test models
        model_path = CONSTANTS.get_data_path("models", "NVDA")
        expected_model_path = CONSTANTS.MODELS_DIR / "nvda_model.pth"
        assert model_path == expected_model_path
    
    def test_get_model_path(self):
        """Test la méthode get_model_path"""
        # Test sans version
        model_path = CONSTANTS.get_model_path("SPY")
        expected_path = CONSTANTS.MODELS_DIR / "spy"
        assert model_path == expected_path
        
        # Test avec version
        model_path_v1 = CONSTANTS.get_model_path("NVDA", 1)
        expected_path_v1 = CONSTANTS.MODELS_DIR / "nvda" / "version1"
        assert model_path_v1 == expected_path_v1
    
    def test_get_feature_columns(self):
        """Test la méthode get_feature_columns"""
        features = CONSTANTS.get_feature_columns()
        assert len(features) == CONSTANTS.LSTM_TOP_FEATURES
        assert all(feature in CONSTANTS.TOP_FEATURES for feature in features)
    
    def test_get_trading_config(self):
        """Test la méthode get_trading_config"""
        config = CONSTANTS.get_trading_config()
        assert "buy_threshold" in config
        assert "sell_threshold" in config
        assert "hold_confidence" in config
        assert "success_threshold" in config
        
        assert config["buy_threshold"] == CONSTANTS.BUY_THRESHOLD
        assert config["sell_threshold"] == CONSTANTS.SELL_THRESHOLD
        assert config["hold_confidence"] == CONSTANTS.HOLD_CONFIDENCE
        assert config["success_threshold"] == CONSTANTS.SUCCESS_THRESHOLD
    
    def test_get_lstm_config(self):
        """Test la méthode get_lstm_config"""
        config = CONSTANTS.get_lstm_config()
        assert "sequence_length" in config
        assert "top_features" in config
        assert "prediction_horizon" in config
        assert "hidden_sizes" in config
        assert "dropout_rate" in config
        assert "epochs" in config
        assert "batch_size" in config
        assert "patience" in config
        assert "learning_rate" in config
        
        assert config["sequence_length"] == CONSTANTS.LSTM_SEQUENCE_LENGTH
        assert config["top_features"] == CONSTANTS.LSTM_TOP_FEATURES
        assert config["prediction_horizon"] == CONSTANTS.LSTM_PREDICTION_HORIZON
        assert config["hidden_sizes"] == CONSTANTS.LSTM_HIDDEN_SIZES
        assert config["dropout_rate"] == CONSTANTS.LSTM_DROPOUT_RATE
        assert config["epochs"] == CONSTANTS.LSTM_EPOCHS
        assert config["batch_size"] == CONSTANTS.LSTM_BATCH_SIZE
        assert config["patience"] == CONSTANTS.LSTM_PATIENCE
        assert config["learning_rate"] == CONSTANTS.LSTM_LEARNING_RATE
    
    def test_ensure_directories(self):
        """Test la méthode ensure_directories"""
        # Cette méthode est appelée automatiquement lors de l'import
        # Vérifier que les répertoires existent
        assert CONSTANTS.DATA_ROOT.exists()
        assert CONSTANTS.PRICES_DIR.exists()
        assert CONSTANTS.NEWS_DIR.exists()
        assert CONSTANTS.SENTIMENT_DIR.exists()
        assert CONSTANTS.MODELS_DIR.exists()
        assert CONSTANTS.LOGS_DIR.exists()
    
    def test_constants_consistency(self):
        """Test la cohérence des constantes"""
        # Vérifier que les poids de fusion sont cohérents
        total_weight = CONSTANTS.BASE_PRICE_WEIGHT + CONSTANTS.BASE_SENTIMENT_WEIGHT
        assert abs(total_weight - 1.0) < 1e-6
        
        # Vérifier que les seuils de trading sont cohérents
        assert CONSTANTS.BUY_THRESHOLD > CONSTANTS.SELL_THRESHOLD
        
        # Vérifier que les features sont cohérentes
        assert len(CONSTANTS.TOP_FEATURES) >= CONSTANTS.LSTM_TOP_FEATURES
        
        # Vérifier que les ports sont différents
        assert CONSTANTS.GUI_PORT != CONSTANTS.API_PORT
        
        # Vérifier que les chemins sont cohérents
        assert CONSTANTS.PROJECT_ROOT.exists()
        assert CONSTANTS.DATA_ROOT.parent == CONSTANTS.PROJECT_ROOT
