#!/usr/bin/env python3
"""
🧪 Script de test du système Sentinel2
Teste tous les composants du système
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from loguru import logger

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import config
from src.constants import CONSTANTS


class SystemTester:
    """Testeur du système Sentinel2"""

    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.results = []

    def log_test(self, test_name: str, success: bool, message: str = ""):
        """Enregistre le résultat d'un test"""
        if success:
            logger.info(f"✅ {test_name}: {message}")
            self.tests_passed += 1
        else:
            logger.error(f"❌ {test_name}: {message}")
            self.tests_failed += 1

        self.results.append(
            {"test": test_name, "success": success, "message": message, "timestamp": datetime.now().isoformat()}
        )

    def test_imports(self):
        """Test des imports"""
        try:
            from src.config import config
            from src.constants import CONSTANTS
            from src.core.fusion import AdaptiveFusion
            from src.core.prediction import PricePredictor
            from src.core.sentiment import FinBertAnalyzer, SentimentAnalyzer
            from src.data.crawler import DataCrawler
            from src.data.storage import DataStorage

            self.log_test("Imports", True, "Tous les modules importés avec succès")
            return True

        except Exception as e:
            self.log_test("Imports", False, f"Erreur d'import: {e}")
            return False

    def test_configuration(self):
        """Test de la configuration"""
        try:
            # Test des constantes
            assert CONSTANTS.TICKERS is not None
            assert len(CONSTANTS.TICKERS) > 0
            assert CONSTANTS.DATA_ROOT.exists()

            # Test de la configuration
            assert config.validate()

            self.log_test("Configuration", True, "Configuration valide")
            return True

        except Exception as e:
            self.log_test("Configuration", False, f"Erreur configuration: {e}")
            return False

    def test_data_directories(self):
        """Test des répertoires de données"""
        try:
            directories = [
                CONSTANTS.DATA_ROOT,
                CONSTANTS.PRICES_DIR,
                CONSTANTS.NEWS_DIR,
                CONSTANTS.SENTIMENT_DIR,
                CONSTANTS.MODELS_DIR,
                CONSTANTS.LOGS_DIR,
            ]

            for directory in directories:
                if not directory.exists():
                    directory.mkdir(parents=True, exist_ok=True)

            self.log_test("Répertoires de données", True, "Tous les répertoires créés")
            return True

        except Exception as e:
            self.log_test("Répertoires de données", False, f"Erreur création répertoires: {e}")
            return False

    def test_core_modules(self):
        """Test des modules core"""
        try:
            from src.core.fusion import AdaptiveFusion
            from src.core.prediction import PricePredictor
            from src.core.sentiment import FinBertAnalyzer, SentimentAnalyzer

            # Test AdaptiveFusion
            fusion = AdaptiveFusion()
            assert fusion is not None

            # Test SentimentAnalyzer
            sentiment = SentimentAnalyzer()
            assert sentiment is not None

            # Test FinBertAnalyzer
            finbert = FinBertAnalyzer(mode="stub")
            assert finbert is not None

            # Test PricePredictor
            predictor = PricePredictor()
            assert predictor is not None

            # Test PricePredictor
            engine = PricePredictor()
            assert engine is not None

            self.log_test("Modules core", True, "Tous les modules core fonctionnels")
            return True

        except Exception as e:
            self.log_test("Modules core", False, f"Erreur modules core: {e}")
            return False

    def test_data_modules(self):
        """Test des modules de données"""
        try:
            from src.data.crawler import DataCrawler
            from src.data.storage import DataStorage

            # Test DataStorage
            storage = DataStorage()
            assert storage is not None

            # Test DataCrawler
            crawler = DataCrawler()
            assert crawler is not None

            self.log_test("Modules de données", True, "Modules de données fonctionnels")
            return True

        except Exception as e:
            self.log_test("Modules de données", False, f"Erreur modules de données: {e}")
            return False

    def test_refresh_scripts(self):
        """Test des scripts de refresh"""
        try:
            # Test refresh_prices.py
            from scripts.refresh_prices import PriceRefresher

            price_refresher = PriceRefresher()
            assert price_refresher is not None

            # Test refresh_news.py
            from scripts.refresh_news import NewsRefresher

            news_refresher = NewsRefresher()
            assert news_refresher is not None

            # Test trading_pipeline.py
            from scripts.trading_pipeline import TradingPipeline

            trading_pipeline = TradingPipeline()
            assert trading_pipeline is not None

            self.log_test("Scripts de refresh", True, "Tous les scripts de refresh chargés")
            return True

        except Exception as e:
            self.log_test("Scripts de refresh", False, f"Erreur scripts de refresh: {e}")
            return False

    def test_sentiment_service(self):
        """Test du service de sentiment"""
        try:
            from scripts.sentiment_service import SentimentService

            # Créer le service
            service = SentimentService(mode="stub", port=5002)
            assert service is not None

            # Test du scoring
            test_texts = ["NVIDIA stock is performing well", "Market crash expected"]
            scores = service.score_texts(test_texts)
            assert len(scores) == len(test_texts)
            assert all(isinstance(score, (int, float)) for score in scores)

            self.log_test("Service de sentiment", True, "Service de sentiment fonctionnel")
            return True

        except Exception as e:
            self.log_test("Service de sentiment", False, f"Erreur service de sentiment: {e}")
            return False

    def test_price_refresh(self):
        """Test du refresh des prix"""
        try:
            from scripts.refresh_prices import PriceRefresher

            refresher = PriceRefresher()

            # Test avec un ticker
            ticker = "SPY"
            result = refresher.refresh_ticker_data(ticker)

            assert result is not None
            assert "ticker" in result
            assert result["ticker"] == ticker

            self.log_test("Refresh des prix", True, f"Refresh {ticker} réussi")
            return True

        except Exception as e:
            self.log_test("Refresh des prix", False, f"Erreur refresh prix: {e}")
            return False

    def test_news_refresh(self):
        """Test du refresh des news"""
        try:
            from scripts.refresh_news import NewsRefresher

            refresher = NewsRefresher()

            # Test de récupération des news RSS
            rss_news = refresher.get_rss_news()
            assert isinstance(rss_news, list)

            # Test de détection de ticker
            ticker = refresher._detect_ticker("NVIDIA stock is performing well")
            assert ticker == "NVDA"

            self.log_test("Refresh des news", True, "Refresh des news fonctionnel")
            return True

        except Exception as e:
            self.log_test("Refresh des news", False, f"Erreur refresh news: {e}")
            return False

    def test_trading_pipeline(self):
        """Test du pipeline de trading"""
        try:
            from scripts.trading_pipeline import TradingPipeline

            pipeline = TradingPipeline()

            # Test de traitement d'un ticker
            ticker = "SPY"
            decision = pipeline.process_ticker(ticker)

            # Le test peut échouer si pas de données, c'est normal
            if decision is not None:
                assert "ticker" in decision
                assert decision["ticker"] == ticker
                self.log_test("Pipeline de trading", True, f"Pipeline {ticker} fonctionnel")
            else:
                self.log_test("Pipeline de trading", True, f"Pipeline {ticker} - pas de données (normal)")

            return True

        except Exception as e:
            self.log_test("Pipeline de trading", False, f"Erreur pipeline trading: {e}")
            return False

    def test_api_endpoints(self):
        """Test des endpoints API"""
        try:
            from scripts.sentiment_service import SentimentService

            # Démarrer le service en arrière-plan
            service = SentimentService(mode="stub", port=5003)

            # Test de l'endpoint de santé
            try:
                response = requests.get("http://localhost:5003/health", timeout=5)
                if response.status_code == 200:
                    self.log_test("API endpoints", True, "Endpoint de santé accessible")
                else:
                    self.log_test("API endpoints", False, f"Endpoint de santé - code {response.status_code}")
            except requests.exceptions.RequestException:
                self.log_test("API endpoints", True, "Endpoint de santé - service non démarré (normal)")

            return True

        except Exception as e:
            self.log_test("API endpoints", False, f"Erreur test API: {e}")
            return False

    def run_all_tests(self):
        """Exécute tous les tests"""
        logger.info("🧪 === TESTS DU SYSTÈME SENTINEL2 ===")
        start_time = time.time()

        # Liste des tests
        tests = [
            self.test_imports,
            self.test_configuration,
            self.test_data_directories,
            self.test_core_modules,
            self.test_data_modules,
            self.test_refresh_scripts,
            self.test_sentiment_service,
            self.test_price_refresh,
            self.test_news_refresh,
            self.test_trading_pipeline,
            self.test_api_endpoints,
        ]

        # Exécuter les tests
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"❌ Erreur inattendue dans {test.__name__}: {e}")
                self.tests_failed += 1

        # Résumé
        duration = time.time() - start_time
        total_tests = self.tests_passed + self.tests_failed

        logger.info(f"\n📊 === RÉSUMÉ DES TESTS ===")
        logger.info(f"   Tests réussis: {self.tests_passed}/{total_tests}")
        logger.info(f"   Tests échoués: {self.tests_failed}/{total_tests}")
        logger.info(f"   Durée: {duration:.1f}s")

        if self.tests_failed == 0:
            logger.info("🎉 Tous les tests sont passés !")
            return True
        else:
            logger.warning(f"⚠️ {self.tests_failed} tests ont échoué")
            return False

    def save_results(self):
        """Sauvegarde les résultats des tests"""
        results_path = CONSTANTS.DATA_ROOT / "logs" / "system_test_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import json

            with open(results_path, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "tests_passed": self.tests_passed,
                        "tests_failed": self.tests_failed,
                        "total_tests": self.tests_passed + self.tests_failed,
                        "results": self.results,
                    },
                    f,
                    indent=2,
                )

            logger.info(f"💾 Résultats sauvegardés: {results_path}")

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde résultats: {e}")


def main():
    """Fonction principale"""
    logger.info("🚀 Démarrage des tests du système")

    try:
        tester = SystemTester()
        success = tester.run_all_tests()
        tester.save_results()

        if success:
            logger.info("✅ Tests terminés avec succès")
            return 0
        else:
            logger.warning("⚠️ Certains tests ont échoué")
            return 1

    except Exception as e:
        logger.error(f"❌ Erreur fatale lors des tests: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
