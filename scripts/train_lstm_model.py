#!/usr/bin/env python3
"""
🚀 Script d'entraînement du modèle LSTM pour Sentinel2
Entraîne le modèle LSTM avec les données de features techniques
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constants import CONSTANTS
from core.prediction import PricePredictor


def train_model(ticker: str = "SPY", epochs: int = 150, use_all_data: bool = True) -> bool:
    """
    Entraîne le modèle LSTM SIMPLE pour un ticker donné

    Basé sur article: "LSTM sans features performe mieux"

    Args:
        ticker: Ticker à entraîner (ex: SPY)
        epochs: Nombre d'époques d'entraînement
        use_all_data: Utiliser TOUTES les données (défaut: True, recommandé par article)
    """
    try:
        logger.info(f"🚀 Début de l'entraînement LSTM SIMPLE (CLOSE ONLY) pour {ticker}")
        logger.info(f"📚 Basé sur recherche: arXiv:2501.17366v1")

        # Charger les données de features
        features_path = CONSTANTS.get_data_path("features", ticker)
        if not features_path.exists():
            logger.error(f"❌ Fichier de features non trouvé: {features_path}")
            return False

        logger.info(f"📊 Chargement des features depuis {features_path}")
        features_df = pd.read_parquet(features_path)

        # PÉRIODE EXACTE DE L'ARTICLE : Oct 2013 - Sept 2024 (11 ans)
        features_df["DATE"] = pd.to_datetime(features_df["DATE"])

        if not use_all_data:
            # Mode legacy: filtrer sur 3 mois (pour comparaison)
            logger.warning("⚠️ Mode legacy: 3 mois seulement (sous-optimal)")
            cutoff = features_df["DATE"].max() - pd.Timedelta(days=90)
            features_df = features_df[features_df["DATE"] >= cutoff].copy()
        else:
            # Période 2019-2024 (5 ans récents, évite distribution shift)
            start_date = "2019-01-01"
            end_date = "2024-09-30"
            features_df = features_df[(features_df["DATE"] >= start_date) & (features_df["DATE"] <= end_date)].copy()
            logger.info(f"✅ Période optimale: 2019-2024 (5 ans, prix cohérents)")

        # FEATURES SÉLECTIONNÉES (|corr| > 0.5 avec CLOSE)
        # Calculer corrélations
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        if "Close" in numeric_cols:
            correlations = features_df[numeric_cols].corr()["Close"].abs()
            selected_features = correlations[correlations > 0.5].index.tolist()

            # Garder DATE + features corrélées
            selected_features = ["DATE"] + [f for f in selected_features if f != "Close"] + ["Close"]
            features_df = features_df[selected_features].copy()

            logger.info(f"✅ Features |corr| > 0.5: {len(selected_features)-2} features + CLOSE")
        else:
            features_df = features_df[["DATE", "Close"]].copy()
            logger.warning("⚠️ CLOSE non trouvé, fallback CLOSE only")

        # Forward-fill (variables non-quotidiennes)
        features_df = features_df.fillna(method="ffill")

        # RETURNS au lieu de prix (stationnarité)
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col != "DATE":
                features_df[f"{col}_RETURN"] = features_df[col].pct_change()

        # Supprimer première ligne (NaN après pct_change)
        features_df = features_df.dropna()

        # Renommer CLOSE_RETURN en TARGET
        features_df.rename(columns={"Close_RETURN": "TARGET"}, inplace=True)

        # Normaliser les colonnes en majuscules
        features_df.columns = features_df.columns.str.upper()

        logger.info(f"✅ Features chargées: {len(features_df)} lignes, {len(features_df.columns)} colonnes")
        logger.info(f"📊 Période: {features_df['DATE'].min().date()} → {features_df['DATE'].max().date()}")
        logger.info(f"💰 Prix moyen: {features_df['CLOSE'].mean():.2f}$ (actuel: {features_df['CLOSE'].iloc[-1]:.2f}$)")

        # Initialiser le prédicteur
        predictor = PricePredictor(ticker)

        # Entraîner le modèle
        logger.info(f"🧠 Début de l'entraînement ({epochs} époques)")
        result = predictor.train(features_df, epochs=epochs)

        if "error" in result:
            logger.error(f"❌ Erreur d'entraînement: {result['error']}")
            return False

        # Sauvegarder le modèle
        model_path = CONSTANTS.get_model_path(ticker) / "version_1" / "model.pkl"
        if predictor.save_model(model_path):
            logger.info(f"✅ Modèle sauvegardé: {model_path}")

            # Afficher les métriques
            logger.info(f"📊 Métriques d'entraînement:")
            logger.info(f"   - Époques entraînées: {result['epochs_trained']}")
            logger.info(f"   - Meilleure validation loss: {result['best_val_loss']:.6f}")
            logger.info(f"   - Mode: CLOSE ONLY (1 feature)")

            return True
        else:
            logger.error("❌ Échec de la sauvegarde du modèle")
            return False

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'entraînement: {e}")
        return False


def main():
    """Fonction principale"""
    import argparse

    parser = argparse.ArgumentParser(description="Entraîner modèle LSTM SIMPLE (CLOSE ONLY - Article Research)")
    parser.add_argument("--ticker", default="SPY", help="Ticker à entraîner (défaut: SPY)")
    parser.add_argument("--epochs", type=int, default=150, help="Nombre d'époques (défaut: 150)")
    parser.add_argument(
        "--all-data", action="store_true", default=True, help="Utiliser toutes les données (recommandé)"
    )

    args = parser.parse_args()

    logger.info(f"🚀 === ENTRAÎNEMENT LSTM SIMPLE (RESEARCH-BASED) ===")
    logger.info(f"📊 Ticker: {args.ticker}")
    logger.info(f"🔄 Époques: {args.epochs}")
    logger.info(f"📅 Données: {'TOUTES (26 ans)' if args.all_data else '3 mois (legacy)'}")
    logger.info(f"📝 Features: CLOSE ONLY (sans indicateurs techniques)")

    success = train_model(args.ticker, args.epochs, args.all_data)

    if success:
        logger.info("✅ Entraînement terminé avec succès!")
        return 0
    else:
        logger.error("❌ Échec de l'entraînement")
        return 1


if __name__ == "__main__":
    sys.exit(main())
