#!/usr/bin/env python3
"""
🚀 Script principal Sentinel2
Orchestre le système de trading complet avec refresh automatique des données
"""

import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import schedule
from loguru import logger

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import config
from src.constants import CONSTANTS


class SentinelMain:
    """Orchestrateur principal du système Sentinel2"""

    def __init__(self):
        self.running = False
        self.threads = []

        # Configuration des intervalles
        self.price_interval = 15 * 60  # 15 minutes
        self.news_interval = 4 * 60  # 4 minutes
        self.trading_interval = 15 * 60  # 15 minutes

        # Importer les modules de refresh
        self.price_refresher = None
        self.news_refresher = None
        self.trading_pipeline = None

        self._load_modules()

    def _load_modules(self):
        """Charge les modules de refresh"""
        try:
            from scripts.refresh_news import NewsRefresher
            from scripts.refresh_prices import PriceRefresher
            from scripts.trading_pipeline import TradingPipeline

            self.price_refresher = PriceRefresher()
            self.news_refresher = NewsRefresher()
            self.trading_pipeline = TradingPipeline()

            logger.info("✅ Modules de refresh chargés avec succès")

        except Exception as e:
            logger.error(f"❌ Erreur chargement modules: {e}")
            raise

    def refresh_prices_job(self):
        """Job de refresh des prix"""
        try:
            logger.info("🔄 Démarrage refresh des prix")
            state = self.price_refresher.refresh_all_prices()
            logger.info("✅ Refresh des prix terminé")
        except Exception as e:
            logger.error(f"❌ Erreur refresh prix: {e}")

    def refresh_news_job(self):
        """Job de refresh des news"""
        try:
            logger.info("📰 Démarrage refresh des news")
            state = self.news_refresher.refresh_news_and_sentiment()
            logger.info("✅ Refresh des news terminé")
        except Exception as e:
            logger.error(f"❌ Erreur refresh news: {e}")

    def trading_pipeline_job(self):
        """Job de pipeline de trading"""
        try:
            logger.info("🤖 Démarrage pipeline de trading")
            result = self.trading_pipeline.run_trading_pipeline()
            logger.info("✅ Pipeline de trading terminé")
        except Exception as e:
            logger.error(f"❌ Erreur pipeline trading: {e}")

    def validation_job(self):
        """Job de validation des décisions en attente"""
        try:
            logger.info("🔍 Démarrage validation des décisions en attente")

            # Importer le service de validation
            from src.gui.services.decision_validation_service import DecisionValidationService

            validation_service = DecisionValidationService()

            # Traiter les validations en attente
            processed_count = validation_service.process_pending_validations()

            if processed_count > 0:
                logger.info(f"✅ {processed_count} validations traitées")
            else:
                logger.debug("ℹ️ Aucune validation en attente")

        except Exception as e:
            logger.error(f"❌ Erreur validation: {e}")

    def setup_schedule(self):
        """Configure la planification des tâches"""
        # Refresh des prix toutes les 15 minutes
        schedule.every(15).minutes.do(self.refresh_prices_job)

        # Refresh des news toutes les 4 minutes
        schedule.every(4).minutes.do(self.refresh_news_job)

        # Pipeline de trading toutes les 15 minutes
        schedule.every(15).minutes.do(self.trading_pipeline_job)

        # Validation des décisions en attente toutes les 5 minutes
        schedule.every(5).minutes.do(self.validation_job)

        logger.info("📅 Planification configurée:")
        logger.info("   - Prix: toutes les 15 minutes")
        logger.info("   - News: toutes les 4 minutes")
        logger.info("   - Trading: toutes les 15 minutes")
        logger.info("   - Validation: toutes les 5 minutes")

    def run_initial_refresh(self):
        """Exécute un refresh initial de toutes les données"""
        logger.info("🚀 === REFRESH INITIAL ===")

        try:
            # Refresh des prix
            logger.info("1. Refresh des prix...")
            self.refresh_prices_job()

            # Refresh des news
            logger.info("2. Refresh des news...")
            self.refresh_news_job()

            # NE PAS exécuter le pipeline de trading au démarrage
            # Il sera exécuté selon les fenêtres fixes de 15 minutes
            logger.info("3. Pipeline de trading... (attente fenêtre 15min)")

            logger.info("✅ Refresh initial terminé")

        except Exception as e:
            logger.error(f"❌ Erreur refresh initial: {e}")
            raise

    def run_scheduler(self):
        """Exécute le planificateur principal"""
        logger.info("⏰ Démarrage du planificateur")

        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)  # Vérifier toutes les secondes
            except Exception as e:
                logger.error(f"❌ Erreur planificateur: {e}")
                time.sleep(5)  # Attendre 5 secondes avant de réessayer

    def signal_handler(self, signum, frame):
        """Gestionnaire de signaux pour arrêt propre"""
        logger.info(f"🛑 Signal reçu: {signum}")
        self.running = False

    def start(self):
        """Démarre le système Sentinel2"""
        logger.info("🚀 === DÉMARRAGE SENTINEL2 ===")

        # Configuration des signaux
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        try:
            # Refresh initial
            self.run_initial_refresh()

            # Configuration de la planification
            self.setup_schedule()

            # Démarrage du planificateur
            self.running = True
            self.run_scheduler()

        except KeyboardInterrupt:
            logger.info("🛑 Arrêt demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"❌ Erreur fatale: {e}")
        finally:
            self.stop()

    def stop(self):
        """Arrête le système Sentinel2"""
        logger.info("🛑 === ARRÊT SENTINEL2 ===")

        self.running = False

        # Attendre que les threads se terminent
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)

        logger.info("✅ Arrêt terminé")

    def run_once(self):
        """Exécute une seule fois toutes les tâches"""
        logger.info("🔄 === EXÉCUTION UNIQUE ===")

        try:
            self.run_initial_refresh()
            logger.info("✅ Exécution unique terminée")
            return 0
        except Exception as e:
            logger.error(f"❌ Erreur exécution unique: {e}")
            return 1


def main():
    """Fonction principale"""
    import argparse

    parser = argparse.ArgumentParser(description="Système de trading Sentinel2")
    parser.add_argument(
        "--mode",
        choices=["daemon", "once"],
        default="daemon",
        help="Mode d'exécution: daemon (continu) ou once (une fois)",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Niveau de log"
    )

    args = parser.parse_args()

    # Configuration des logs
    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Ajouter un fichier de log
    log_file = CONSTANTS.DATA_ROOT / "logs" / "sentinel_main.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="7 days",
    )

    try:
        sentinel = SentinelMain()

        if args.mode == "daemon":
            logger.info("🚀 Mode daemon - Démarrage continu")
            sentinel.start()
        else:
            logger.info("🔄 Mode once - Exécution unique")
            exit_code = sentinel.run_once()
            sys.exit(exit_code)

    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
