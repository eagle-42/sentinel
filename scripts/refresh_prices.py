#!/usr/bin/env python3
"""
🔄 Script de mise à jour des données de prix
Met à jour les données de prix toutes les 15 minutes depuis Yahoo Finance et Polygon API
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from loguru import logger

# Charger les variables d'environnement
load_dotenv()

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import config
from src.constants import CONSTANTS
from src.data.storage import DataStorage


class PriceRefresher:
    """Gestionnaire de mise à jour des données de prix"""

    def __init__(self):
        self.storage = DataStorage()
        self.tickers = CONSTANTS.TICKERS
        self.polygon_api_key = os.getenv("POLYGON_API_KEY")
        self.yahoo_fallback = True

    def get_yahoo_prices(self, ticker: str, period: str = "7d", interval: str = "15m") -> pd.DataFrame:
        """Récupère les données de prix depuis Yahoo Finance avec retry et backoff"""
        import random
        import time

        max_retries = 3
        base_delay = 5  # secondes

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    logger.info(f"🔄 Tentative {attempt + 1}/{max_retries} pour {ticker} (attente {delay:.1f}s)")
                    time.sleep(delay)

                logger.info(f"📈 Récupération {ticker} depuis Yahoo Finance ({period}, {interval})")

                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(period=period, interval=interval)

                if data.empty:
                    logger.warning(f"⚠️ Aucune donnée Yahoo pour {ticker} (tentative {attempt + 1})")
                    continue

                # Normaliser les données
                data = data.reset_index()
                data["ticker"] = ticker
                data = data.rename(columns={"Datetime": "ts_utc"})

                # S'assurer que la date est en UTC
                if data["ts_utc"].dt.tz is None:
                    data["ts_utc"] = data["ts_utc"].dt.tz_localize("UTC")
                else:
                    data["ts_utc"] = data["ts_utc"].dt.tz_convert("UTC")

                # Sélectionner les colonnes finales
                data = data[["ticker", "ts_utc", "Open", "High", "Low", "Close", "Volume"]]
                data.columns = ["ticker", "ts_utc", "open", "high", "low", "close", "volume"]

                logger.info(f"✅ {ticker}: {len(data)} barres récupérées depuis Yahoo")
                return data

            except Exception as e:
                logger.warning(f"⚠️ Erreur Yahoo pour {ticker} (tentative {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"❌ Échec définitif Yahoo pour {ticker} après {max_retries} tentatives")
                    return pd.DataFrame()
                continue

        return pd.DataFrame()

    def get_polygon_prices(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Récupère les données de prix depuis Polygon API"""
        if not self.polygon_api_key:
            logger.warning("⚠️ Clé API Polygon manquante, utilisation de Yahoo Finance")
            return pd.DataFrame()

        try:
            logger.info(f"📈 Récupération {ticker} depuis Polygon API")

            # Convertir les dates en timestamps Unix
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)

            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/15/minute/{start_ts}/{end_ts}"
            params = {"adjusted": "true", "sort": "asc", "apikey": self.polygon_api_key}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get("status") not in ["OK", "DELAYED"] or not data.get("results"):
                logger.warning(f"⚠️ Aucune donnée Polygon pour {ticker} (status: {data.get('status')})")
                return pd.DataFrame()

            # Convertir les données
            results = data["results"]
            df_data = []

            for bar in results:
                df_data.append(
                    {
                        "ticker": ticker,
                        "ts_utc": pd.to_datetime(bar["t"], unit="ms", utc=True),
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar["v"],
                    }
                )

            df = pd.DataFrame(df_data)
            logger.info(f"✅ {ticker}: {len(df)} barres récupérées depuis Polygon")
            return df

        except Exception as e:
            logger.error(f"❌ Erreur Polygon pour {ticker}: {e}")
            return pd.DataFrame()

    def should_refresh_data(self, ticker: str, interval: str = "15min") -> bool:
        """Vérifie si les données doivent être rafraîchies"""
        try:
            # Vérifier si le fichier existe
            file_path = CONSTANTS.get_data_path("prices", ticker, interval)
            if not file_path.exists():
                logger.info(f"🆕 Nouveau fichier pour {ticker}")
                return True

            # Lire la dernière date
            existing_data = pd.read_parquet(file_path)
            if existing_data.empty:
                logger.info(f"📁 Fichier vide pour {ticker}")
                return True

            last_update = existing_data["ts_utc"].max()
            if hasattr(last_update, "to_pydatetime"):
                last_update = last_update.to_pydatetime()
            if last_update.tzinfo is None:
                last_update = last_update.replace(tzinfo=timezone.utc)

            # Vérifier si les données ont plus de 1 heure
            time_diff = (datetime.now(timezone.utc) - last_update).total_seconds()
            should_refresh = time_diff > 3600  # 1 heure

            if should_refresh:
                logger.info(f"🔄 {ticker}: Données anciennes ({time_diff/3600:.1f}h), refresh nécessaire")
            else:
                logger.info(f"✅ {ticker}: Données récentes ({time_diff/60:.1f}min), pas de refresh")

            return should_refresh

        except Exception as e:
            logger.error(f"❌ Erreur vérification refresh {ticker}: {e}")
            return True

    def merge_price_data(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """Fusionne les données existantes et nouvelles"""
        if existing_data.empty:
            return new_data

        if new_data.empty:
            return existing_data

        # Concaténer et supprimer les doublons
        combined = pd.concat([existing_data, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts_utc"], keep="last")
        combined = combined.sort_values("ts_utc").reset_index(drop=True)

        return combined

    def refresh_ticker_data(self, ticker: str) -> Dict[str, Any]:
        """Met à jour les données pour un ticker spécifique"""
        logger.info(f"\n📈 Traitement {ticker}")

        # Vérifier si refresh nécessaire
        if not self.should_refresh_data(ticker):
            return {"ticker": ticker, "status": "up_to_date", "rows": 0, "new_rows": 0, "source": "none"}

        # Déterminer la période (7 derniers jours)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)

        # Essayer Polygon d'abord, puis Yahoo en fallback
        new_data = self.get_polygon_prices(ticker, start_date, end_date)
        source = "polygon"

        if new_data.empty and self.yahoo_fallback:
            logger.info(f"🔄 Fallback vers Yahoo Finance pour {ticker}")
            new_data = self.get_yahoo_prices(ticker, period="7d", interval="15m")
            source = "yahoo"

        if new_data.empty:
            logger.warning(f"❌ Aucune donnée récupérée pour {ticker}")
            return {"ticker": ticker, "status": "no_data", "rows": 0, "new_rows": 0, "source": "none"}

        # Charger les données existantes
        file_path = CONSTANTS.get_data_path("prices", ticker, "15min")
        existing_data = pd.DataFrame()

        if file_path.exists():
            try:
                existing_data = pd.read_parquet(file_path)
                logger.info(f"📊 {ticker}: {len(existing_data)} lignes existantes")
            except Exception as e:
                logger.warning(f"⚠️ Erreur lecture données existantes {ticker}: {e}")

        # Fusionner les données
        combined_data = self.merge_price_data(existing_data, new_data)

        # Sauvegarder
        try:
            combined_data.to_parquet(file_path, index=False)
            logger.info(f"💾 {ticker}: {len(combined_data)} lignes sauvegardées")
            logger.info(f"📅 Période: {combined_data['ts_utc'].min()} à {combined_data['ts_utc'].max()}")

            return {
                "ticker": ticker,
                "status": "updated",
                "rows": len(combined_data),
                "new_rows": len(new_data),
                "existing_rows": len(existing_data),
                "source": source,
                "start_date": combined_data["ts_utc"].min().isoformat(),
                "end_date": combined_data["ts_utc"].max().isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde {ticker}: {e}")
            return {"ticker": ticker, "status": "error", "rows": 0, "new_rows": 0, "source": source, "error": str(e)}

    def refresh_all_prices(self) -> Dict[str, Any]:
        """Met à jour toutes les données de prix"""
        logger.info("🔄 === MISE À JOUR DES DONNÉES DE PRIX ===")
        start_time = datetime.now()

        results = {}
        total_rows = 0
        total_new_rows = 0

        for ticker in self.tickers:
            result = self.refresh_ticker_data(ticker)
            results[ticker] = result

            if result["status"] == "updated":
                total_rows += result["rows"]
                total_new_rows += result["new_rows"]

        # Sauvegarder l'état
        state = {
            "last_update": datetime.now().isoformat(),
            "granularity": "15min",
            "period": f"7 derniers jours",
            "tickers": results,
            "summary": {
                "total_tickers": len(self.tickers),
                "total_rows": total_rows,
                "total_new_rows": total_new_rows,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            },
        }

        state_path = CONSTANTS.DATA_ROOT / "logs" / "price_refresh_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        import json

        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"\n📊 Résumé de la mise à jour:")
        logger.info(f"   Tickers traités: {len(self.tickers)}")
        logger.info(f"   Lignes totales: {total_rows}")
        logger.info(f"   Nouvelles lignes: {total_new_rows}")
        logger.info(f"   Durée: {(datetime.now() - start_time).total_seconds():.1f}s")
        logger.info(f"   État sauvé: {state_path}")

        return state


def main():
    """Fonction principale"""
    logger.info("🚀 Démarrage du refresh des données de prix")

    try:
        refresher = PriceRefresher()
        state = refresher.refresh_all_prices()

        logger.info("✅ Refresh des données de prix terminé avec succès")
        return 0

    except Exception as e:
        logger.error(f"❌ Erreur lors du refresh des données: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
