#!/usr/bin/env python3
"""
🤖 Pipeline de trading complet
Exécute le pipeline de trading avec fusion des signaux et prise de décision
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import config
from src.constants import CONSTANTS
from src.core.fusion import AdaptiveFusion
from src.core.prediction import PricePredictor
from src.core.sentiment import SentimentAnalyzer
from src.data.storage import DataStorage


class TradingPipeline:
    """Pipeline de trading complet avec fusion des signaux"""

    def __init__(self):
        self.storage = DataStorage()
        self.tickers = CONSTANTS.TICKERS
        self.fusion = AdaptiveFusion()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.prediction_engine = PricePredictor()

        # Configuration de trading
        self.buy_threshold = CONSTANTS.BASE_BUY_THRESHOLD
        self.sell_threshold = CONSTANTS.BASE_SELL_THRESHOLD
        self.hold_confidence = CONSTANTS.HOLD_CONFIDENCE

        # État de trading
        self.trading_state = self._load_trading_state()

    def _load_trading_state(self) -> Dict[str, Any]:
        """Charge l'état de trading depuis le fichier"""
        state_path = CONSTANTS.DATA_ROOT / "trading" / "decisions_log" / "trading_state.json"

        if state_path.exists():
            try:
                with open(state_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Erreur chargement état trading: {e}")

        # État par défaut
        return {
            "last_update": None,
            "positions": {},
            "decisions": [],
            "performance": {"total_decisions": 0, "correct_decisions": 0, "accuracy": 0.0},
        }

    def _save_trading_state(self):
        """Sauvegarde l'état de trading"""
        state_path = CONSTANTS.DATA_ROOT / "trading" / "decisions_log" / "trading_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(state_path, "w") as f:
                json.dump(self.trading_state, f, indent=2)
            logger.debug(f"💾 État de trading sauvegardé: {state_path}")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde état trading: {e}")

    def get_latest_prices(self, ticker: str) -> Optional[pd.DataFrame]:
        """Récupère les dernières données de prix"""
        try:
            file_path = CONSTANTS.get_data_path("prices", ticker, "15min")
            if not file_path.exists():
                logger.warning(f"⚠️ Fichier de prix manquant pour {ticker}")
                return None

            data = pd.read_parquet(file_path)
            if data.empty:
                logger.warning(f"⚠️ Aucune donnée de prix pour {ticker}")
                return None

            # Trier par date et prendre les dernières données
            data = data.sort_values("ts_utc").tail(20)  # 20 dernières barres
            return data

        except Exception as e:
            logger.error(f"❌ Erreur récupération prix {ticker}: {e}")
            return None

    def get_latest_sentiment(self, ticker: str) -> float:
        """Récupère le dernier sentiment pour un ticker"""
        try:
            # Chercher le fichier de sentiment le plus récent
            sentiment_dir = CONSTANTS.SENTIMENT_DIR
            sentiment_files = list(sentiment_dir.glob("sentiment_*.parquet"))

            if not sentiment_files:
                logger.warning(f"⚠️ Aucun fichier de sentiment pour {ticker}")
                return 0.0

            # Prendre le fichier le plus récent
            latest_file = max(sentiment_files, key=lambda x: x.stat().st_mtime)
            data = pd.read_parquet(latest_file)

            # Filtrer pour le ticker et prendre la dernière valeur
            ticker_data = data[data["ticker"] == ticker]
            if ticker_data.empty:
                logger.warning(f"⚠️ Aucun sentiment pour {ticker}")
                return 0.0

            latest_sentiment = ticker_data.iloc[-1]["sentiment_score"]
            logger.debug(f"📊 Sentiment {ticker}: {latest_sentiment:.3f}")
            return float(latest_sentiment)

        except Exception as e:
            logger.error(f"❌ Erreur récupération sentiment {ticker}: {e}")
            return 0.0

    def calculate_price_signal(self, prices: pd.DataFrame) -> float:
        """Calcule le signal de prix basé sur les données récentes"""
        if prices.empty or len(prices) < 2:
            return 0.0

        try:
            # Calculer le retour sur la période
            current_price = prices["close"].iloc[-1]
            previous_price = prices["close"].iloc[-2]

            # Retour simple
            price_return = (current_price - previous_price) / previous_price

            # Normaliser le signal entre -1 et 1
            price_signal = np.tanh(price_return * 10)  # Amplifier les petits changements

            logger.debug(f"📈 Signal prix: {price_return:.4f} -> {price_signal:.3f}")
            return float(price_signal)

        except Exception as e:
            logger.error(f"❌ Erreur calcul signal prix: {e}")
            return 0.0

    def get_lstm_prediction(self, ticker: str, prices: pd.DataFrame) -> Optional[float]:
        """Récupère la prédiction LSTM pour un ticker"""
        try:
            # Charger les données de features techniques
            features_path = CONSTANTS.get_data_path("features", ticker)
            if not features_path.exists():
                logger.warning(f"⚠️ Données de features non trouvées pour {ticker}")
                return None

            # Charger les features
            features_data = pd.read_parquet(features_path)
            if features_data.empty:
                logger.warning(f"⚠️ Données de features vides pour {ticker}")
                return None

            # Initialiser le prédicteur
            predictor = PricePredictor(ticker)

            # Charger le modèle
            if not predictor.load_model():
                logger.warning(f"⚠️ Impossible de charger le modèle pour {ticker}")
                return None

            # Préparer les features
            features = predictor.prepare_features(features_data)

            if features is None:
                logger.warning(f"⚠️ Impossible de préparer les features pour {ticker}")
                return None

            # Créer les séquences
            X, y = predictor.create_sequences(features)

            if X is None or len(X) == 0:
                logger.warning(f"⚠️ Impossible de créer les séquences pour {ticker}")
                return None

            # Faire la prédiction directe
            with torch.no_grad():
                sequence = torch.FloatTensor(X[-1:]).to(predictor.device)
                pred = predictor.model(sequence)
                prediction_signal = float(pred.cpu().numpy()[0, 0])

            logger.debug(f"🔮 Prédiction LSTM {ticker}: {prediction_signal:.3f}")
            return prediction_signal

        except Exception as e:
            logger.error(f"❌ Erreur prédiction LSTM {ticker}: {e}")
            return None

    def make_trading_decision(
        self, ticker: str, price_signal: float, sentiment_signal: float, prediction_signal: Optional[float] = None
    ) -> Dict[str, Any]:
        """Prend une décision de trading basée sur les signaux"""

        # Préparer les signaux pour la fusion
        signals = {"price": price_signal, "sentiment": sentiment_signal}

        if prediction_signal is not None:
            signals["prediction"] = prediction_signal

        # Fusionner les signaux
        fused_signal = self.fusion.fuse_signals(signals)

        # Prendre la décision
        decision = "HOLD"
        confidence = 0.0

        if fused_signal > self.buy_threshold:
            decision = "BUY"
            confidence = min(fused_signal, 1.0)
        elif fused_signal < self.sell_threshold:
            decision = "SELL"
            confidence = min(abs(fused_signal), 1.0)
        else:
            decision = "HOLD"
            confidence = 1.0 - abs(fused_signal)

        # Créer la décision
        decision_data = {
            "ticker": ticker,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "confidence": confidence,
            "fused_signal": fused_signal,
            "signals": signals,
            "thresholds": {"buy": self.buy_threshold, "sell": self.sell_threshold, "hold": self.hold_confidence},
        }

        logger.info(f"🤖 Décision {ticker}: {decision} (confiance: {confidence:.3f}, signal: {fused_signal:.3f})")

        return decision_data

    def process_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Traite un ticker et génère une décision de trading"""
        logger.info(f"\n📊 Traitement {ticker}")

        # Récupérer les données
        prices = self.get_latest_prices(ticker)
        if prices is None:
            logger.warning(f"⚠️ Impossible de traiter {ticker}: pas de données de prix")
            return None

        # Calculer les signaux
        price_signal = self.calculate_price_signal(prices)
        sentiment_signal = self.get_latest_sentiment(ticker)
        prediction_signal = self.get_lstm_prediction(ticker, prices)

        # Prendre la décision
        decision = self.make_trading_decision(ticker, price_signal, sentiment_signal, prediction_signal)

        return decision

    def _is_decision_window(self) -> bool:
        """Vérifie si on est dans une fenêtre de décision valide (15 minutes)"""
        try:
            import pytz

            # Timezone US Eastern (gère automatiquement EST/EDT)
            us_eastern = pytz.timezone("US/Eastern")
            now_est = datetime.now(us_eastern)

            # Heures de marché US (9:30-16:00)
            market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

            # Vérifier si c'est un jour de semaine
            is_weekday = now_est.weekday() < 5  # 0-4 = lundi-vendredi

            if not is_weekday:
                return False

            if not (market_open <= now_est <= market_close):
                return False

            # Vérifier si on est dans une fenêtre de 15 minutes
            current_minute = now_est.minute
            return current_minute in [30, 45, 0]  # 9:30, 9:45, 10:00, 10:15, etc.

        except ImportError:
            # Fallback si pytz n'est pas disponible
            from datetime import timedelta, timezone

            edt = timezone(timedelta(hours=-4))
            now_est = datetime.now(edt)

            # Heures de marché US (9:30-16:00 EDT)
            market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

            # Vérifier si c'est un jour de semaine
            is_weekday = now_est.weekday() < 5  # 0-4 = lundi-vendredi

            if not is_weekday:
                return False

            if not (market_open <= now_est <= market_close):
                return False

            # Vérifier si on est dans une fenêtre de 15 minutes
            current_minute = now_est.minute
            return current_minute in [30, 45, 0]  # 9:30, 9:45, 10:00, 10:15, etc.

    def run_trading_pipeline(self) -> Dict[str, Any]:
        """Exécute le pipeline de trading complet"""
        logger.info("🤖 === PIPELINE DE TRADING ===")
        start_time = datetime.now()

        # Vérifier si on est dans une fenêtre de décision valide (15 minutes)
        if not self._is_decision_window():
            logger.info("⏰ Pas dans une fenêtre de décision (15min) - Attente")
            return {
                "success": True,
                "decisions": [],
                "tickers_processed": 0,
                "duration": (datetime.now() - start_time).total_seconds(),
            }

        decisions = []
        successful_tickers = 0

        for ticker in self.tickers:
            try:
                decision = self.process_ticker(ticker)
                if decision:
                    decisions.append(decision)
                    successful_tickers += 1
            except Exception as e:
                logger.error(f"❌ Erreur traitement {ticker}: {e}")
                continue

        # Mettre à jour l'état de trading
        self.trading_state["last_update"] = datetime.now().isoformat()
        self.trading_state["decisions"].extend(decisions)

        # Garder seulement les 100 dernières décisions
        if len(self.trading_state["decisions"]) > 100:
            self.trading_state["decisions"] = self.trading_state["decisions"][-100:]

        # Sauvegarder l'état
        self._save_trading_state()

        # Sauvegarder les décisions dans un fichier séparé
        self._save_decisions_log(decisions)

        # Calculer les métriques
        duration = (datetime.now() - start_time).total_seconds()

        result = {
            "timestamp": datetime.now().isoformat(),
            "tickers_processed": successful_tickers,
            "total_tickers": len(self.tickers),
            "decisions": decisions,
            "duration_seconds": duration,
            "status": "success" if successful_tickers > 0 else "no_data",
        }

        logger.info(f"\n📊 Résumé du pipeline:")
        logger.info(f"   Tickers traités: {successful_tickers}/{len(self.tickers)}")
        logger.info(f"   Décisions générées: {len(decisions)}")
        logger.info(f"   Durée: {duration:.1f}s")

        return result

    def _save_decisions_log(self, decisions: List[Dict[str, Any]]):
        """Sauvegarde le log des décisions dans un fichier unifié"""
        if not decisions:
            return

        # Fichier unifié pour toutes les décisions
        log_path = CONSTANTS.DATA_ROOT / "trading" / "decisions_log" / "trading_decisions.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Charger les décisions existantes
        existing_decisions = []
        if log_path.exists():
            try:
                with open(log_path, "r") as f:
                    existing_decisions = json.load(f)
                logger.debug(f"📊 Décisions existantes: {len(existing_decisions)}")
            except Exception as e:
                logger.warning(f"⚠️ Erreur lecture décisions existantes: {e}")

        # Fusionner les nouvelles décisions
        all_decisions = existing_decisions + decisions

        # Garder seulement les 1000 dernières décisions pour éviter la surcharge
        if len(all_decisions) > 1000:
            all_decisions = all_decisions[-1000:]
            logger.info(
                f"📊 Décisions limitées à 1000 (supprimé {len(existing_decisions) + len(decisions) - 1000} anciennes)"
            )

        try:
            with open(log_path, "w") as f:
                json.dump(all_decisions, f, indent=2)
            logger.debug(f"💾 Log des décisions sauvegardé: {log_path} ({len(all_decisions)} total)")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde log décisions: {e}")


def main():
    """Fonction principale"""
    logger.info("🚀 Démarrage du pipeline de trading")

    try:
        pipeline = TradingPipeline()
        result = pipeline.run_trading_pipeline()

        if result["status"] == "success":
            logger.info("✅ Pipeline de trading terminé avec succès")
            return 0
        else:
            logger.warning("⚠️ Pipeline de trading terminé sans données")
            return 1

    except Exception as e:
        logger.error(f"❌ Erreur lors du pipeline de trading: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
