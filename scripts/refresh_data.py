#!/usr/bin/env python3
"""
🔄 Script de mise à jour des données de prix
Met à jour les données de prix depuis Yahoo Finance
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def refresh_price_data():
    """Met à jour les données de prix pour SPY et NVDA"""
    try:
        logger.info("🔄 Début de la mise à jour des données de prix")

        # Tickers à mettre à jour
        tickers = ["SPY", "NVDA"]

        for ticker in tickers:
            logger.info(f"📈 Mise à jour de {ticker}")

            # Récupérer les données depuis Yahoo Finance
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period="1y", interval="1d")

            if data.empty:
                logger.warning(f"⚠️ Aucune donnée pour {ticker}")
                continue

            # Normaliser les données
            data = data.reset_index()
            data["ticker"] = ticker
            data = data.rename(columns={"Datetime": "date"})

            # Sauvegarder les données
            output_file = Path(f"data/historical/yfinance/{ticker}_1999_2025.parquet")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            data.to_parquet(output_file, index=False)
            logger.info(f"✅ {ticker} sauvegardé: {len(data)} lignes")
            logger.info(f"   Dernière date: {data['date'].iloc[-1]}")

        logger.info("✅ Mise à jour terminée")
        return True

    except Exception as e:
        logger.error(f"❌ Erreur lors de la mise à jour: {e}")
        return False


if __name__ == "__main__":
    success = refresh_price_data()
    sys.exit(0 if success else 1)
