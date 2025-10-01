"""
🧪 Tests Principaux Sentinel2
Point d'entrée pour tous les tests
"""

import sys
from pathlib import Path

import pytest

# Ajouter le répertoire src au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test que tous les modules peuvent être importés"""
    try:
        from src.config import config
        from src.constants import CONSTANTS
        from src.core.fusion import AdaptiveFusion
        from src.core.prediction import LSTMPredictor, PricePredictor
        from src.core.sentiment import FinBertAnalyzer, SentimentAnalyzer
        from src.data.crawler import DataCrawler, NewsCrawler, PriceCrawler
        from src.data.storage import DataStorage, ParquetStorage

        print("✅ Tous les imports réussis")
    except ImportError as e:
        pytest.fail(f"❌ Erreur d'import: {e}")


def test_constants_initialization():
    """Test l'initialisation des constantes"""
    from src.constants import CONSTANTS

    assert CONSTANTS is not None
    assert len(CONSTANTS.TICKERS) > 0
    assert CONSTANTS.PROJECT_ROOT.exists()
    print("✅ Constantes initialisées correctement")


def test_config_initialization():
    """Test l'initialisation de la configuration"""
    from src.config import config

    assert config is not None
    assert config.validate()
    print("✅ Configuration initialisée correctement")


def test_core_modules_initialization():
    """Test l'initialisation des modules core"""
    from src.core.fusion import AdaptiveFusion
    from src.core.prediction import LSTMPredictor, PricePredictor
    from src.core.sentiment import FinBertAnalyzer, SentimentAnalyzer

    # Test fusion
    fusion = AdaptiveFusion()
    assert fusion is not None

    # Test prediction
    predictor = PricePredictor("SPY")
    assert predictor.ticker == "SPY"
    assert predictor is not None

    # Test sentiment
    analyzer = SentimentAnalyzer()
    assert analyzer is not None

    finbert = FinBertAnalyzer()
    assert finbert is not None

    print("✅ Modules core initialisés correctement")


def test_data_modules_initialization():
    """Test l'initialisation des modules data"""
    from src.data.crawler import DataCrawler, NewsCrawler, PriceCrawler
    from src.data.storage import DataStorage, ParquetStorage

    # Test storage
    storage = DataStorage()
    assert storage is not None

    parquet_storage = ParquetStorage()
    assert parquet_storage is not None

    # Test crawler
    crawler = DataCrawler()
    assert crawler is not None

    price_crawler = PriceCrawler()
    assert price_crawler is not None

    news_crawler = NewsCrawler()
    assert news_crawler is not None

    print("✅ Modules data initialisés correctement")


if __name__ == "__main__":
    # Exécuter les tests
    pytest.main([__file__, "-v"])
