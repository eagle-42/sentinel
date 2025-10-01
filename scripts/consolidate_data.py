"""
Script de consolidation des données Parquet
Consolide tous les fichiers par date en fichiers unifiés
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ajouter le chemin src pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from constants import CONSTANTS


def consolidate_news_data():
    """Consolide tous les fichiers de news en un seul fichier unifié"""
    logger.info("📰 Consolidation des données de news...")

    news_dir = CONSTANTS.NEWS_DIR
    all_news_file = news_dir / "all_news.parquet"

    # Chercher tous les fichiers de news par date
    news_files = list(news_dir.glob("news_*.parquet"))

    if not news_files:
        logger.info("ℹ️ Aucun fichier de news à consolider")
        return

    # Charger et fusionner tous les fichiers
    all_data = []
    for news_file in news_files:
        try:
            df = pd.read_parquet(news_file)
            all_data.append(df)
            logger.info(f"📄 Chargé: {news_file.name} ({len(df)} lignes)")
        except Exception as e:
            logger.error(f"❌ Erreur lecture {news_file.name}: {e}")

    if not all_data:
        logger.warning("⚠️ Aucune donnée de news valide")
        return

    # Fusionner toutes les données
    consolidated_df = pd.concat(all_data, ignore_index=True)

    # Supprimer les doublons
    original_count = len(consolidated_df)
    consolidated_df = consolidated_df.drop_duplicates(subset=["title", "source", "timestamp"], keep="last")
    final_count = len(consolidated_df)

    logger.info(
        f"📊 Consolidation: {original_count} → {final_count} lignes (supprimé {original_count - final_count} doublons)"
    )

    # Sauvegarder le fichier consolidé
    consolidated_df.to_parquet(all_news_file, index=False)
    logger.success(f"✅ News consolidées: {all_news_file} ({final_count} lignes)")

    # Supprimer les anciens fichiers
    for news_file in news_files:
        try:
            news_file.unlink()
            logger.info(f"🗑️ Supprimé: {news_file.name}")
        except Exception as e:
            logger.error(f"❌ Erreur suppression {news_file.name}: {e}")


def consolidate_sentiment_data():
    """Consolide tous les fichiers de sentiment en un seul fichier unifié"""
    logger.info("💭 Consolidation des données de sentiment...")

    sentiment_dir = CONSTANTS.SENTIMENT_DIR
    spy_sentiment_file = sentiment_dir / "spy_sentiment.parquet"

    # Chercher tous les fichiers de sentiment par date
    sentiment_files = list(sentiment_dir.glob("sentiment_*.parquet"))

    if not sentiment_files:
        logger.info("ℹ️ Aucun fichier de sentiment à consolider")
        return

    # Charger et fusionner tous les fichiers
    all_data = []
    for sentiment_file in sentiment_files:
        try:
            df = pd.read_parquet(sentiment_file)
            all_data.append(df)
            logger.info(f"📄 Chargé: {sentiment_file.name} ({len(df)} lignes)")
        except Exception as e:
            logger.error(f"❌ Erreur lecture {sentiment_file.name}: {e}")

    if not all_data:
        logger.warning("⚠️ Aucune donnée de sentiment valide")
        return

    # Fusionner toutes les données
    consolidated_df = pd.concat(all_data, ignore_index=True)

    # Supprimer les doublons
    original_count = len(consolidated_df)
    consolidated_df = consolidated_df.drop_duplicates(subset=["ticker", "ts_utc"], keep="last")
    final_count = len(consolidated_df)

    logger.info(
        f"📊 Consolidation: {original_count} → {final_count} lignes (supprimé {original_count - final_count} doublons)"
    )

    # Sauvegarder le fichier consolidé
    consolidated_df.to_parquet(spy_sentiment_file, index=False)
    logger.success(f"✅ Sentiment consolidé: {spy_sentiment_file} ({final_count} lignes)")

    # Supprimer les anciens fichiers
    for sentiment_file in sentiment_files:
        try:
            sentiment_file.unlink()
            logger.info(f"🗑️ Supprimé: {sentiment_file.name}")
        except Exception as e:
            logger.error(f"❌ Erreur suppression {sentiment_file.name}: {e}")


def main():
    """Fonction principale de consolidation"""
    logger.info("🔄 === CONSOLIDATION DES DONNÉES PARQUET ===")

    try:
        # Consolider les news
        consolidate_news_data()

        # Consolider le sentiment
        consolidate_sentiment_data()

        logger.success("✅ Consolidation terminée avec succès!")

    except Exception as e:
        logger.error(f"❌ Erreur lors de la consolidation: {e}")
        raise


if __name__ == "__main__":
    main()
