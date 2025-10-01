#!/usr/bin/env python3
"""
🔄 Migration vers Stockage Unifié
Migre les données existantes vers la nouvelle architecture de stockage
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.constants import CONSTANTS
from src.data.unified_storage import UnifiedDataStorage


def migrate_existing_data():
    """Migre toutes les données existantes vers le stockage unifié"""
    print("🔄 Migration vers stockage unifié...")

    storage = UnifiedDataStorage()

    # 1. Migrer les données de prix
    migrate_price_data(storage)

    # 2. Migrer les données de news
    migrate_news_data(storage)

    # 3. Migrer les données de features
    migrate_features_data(storage)

    # 4. Migrer les modèles
    migrate_models(storage)

    # 5. Nettoyer les anciens répertoires
    cleanup_old_directories()

    print("✅ Migration terminée avec succès!")


def migrate_price_data(storage):
    """Migre les données de prix"""
    print("📊 Migration des données de prix...")

    # Sources de données existantes
    price_sources = [
        ("data/dataset/1-yfinance/SPY_1999_2025.parquet", "SPY", "1min", "yfinance"),
        ("data/dataset/1-yfinance/NVDA_1999_2025.parquet", "NVDA", "1min", "yfinance"),
        ("data/dataset/2-delta-api-15m/spy_15min_delta.parquet", "SPY", "15min", "delta"),
        ("data/dataset/2-delta-api-15m/nvda_15min_delta.parquet", "NVDA", "15min", "delta"),
        ("data/trading/prices/spy_1min.parquet", "SPY", "1min", "trading"),
        ("data/trading/prices/nvda_1min.parquet", "NVDA", "1min", "trading"),
    ]

    for source_path, ticker, interval, source in price_sources:
        if Path(source_path).exists():
            try:
                # Charger les données
                data = pd.read_parquet(source_path)

                # Normaliser les colonnes
                data = normalize_price_data(data)

                # Sauvegarder dans le stockage unifié
                storage.save_price_data(ticker, data, interval, source)
                print(f"  ✅ {ticker} {interval} ({source}) migré")

            except Exception as e:
                print(f"  ❌ Erreur migration {ticker} {interval}: {e}")


def migrate_news_data(storage):
    """Migre les données de news"""
    print("📰 Migration des données de news...")

    # Chercher les fichiers de news existants
    news_dirs = ["data/news", "data/trading/sentiment"]

    for news_dir in news_dirs:
        if Path(news_dir).exists():
            news_files = list(Path(news_dir).glob("*.parquet"))
            for news_file in news_files:
                try:
                    # Charger les données
                    data = pd.read_parquet(news_file)

                    # Normaliser les colonnes
                    data = normalize_news_data(data)

                    # Déterminer le ticker
                    ticker = extract_ticker_from_filename(news_file.name)

                    # Sauvegarder dans le stockage unifié
                    storage.save_news_data(data, ticker, "migrated")
                    print(f"  ✅ News {ticker} migré depuis {news_file.name}")

                except Exception as e:
                    print(f"  ❌ Erreur migration news {news_file.name}: {e}")


def migrate_features_data(storage):
    """Migre les données de features"""
    print("🔧 Migration des données de features...")

    # Sources de features existantes
    features_sources = [
        ("data/dataset/3-features/spy_features.parquet", "SPY", "technical"),
        ("data/dataset/3-features/nvda_features.parquet", "NVDA", "technical"),
    ]

    for source_path, ticker, feature_type in features_sources:
        if Path(source_path).exists():
            try:
                # Charger les données
                data = pd.read_parquet(source_path)

                # Normaliser les colonnes
                data = normalize_features_data(data)

                # Sauvegarder dans le stockage unifié
                storage.save_features_data(ticker, data, feature_type)
                print(f"  ✅ Features {ticker} migré")

            except Exception as e:
                print(f"  ❌ Erreur migration features {ticker}: {e}")


def migrate_models(storage):
    """Migre les modèles existants"""
    print("🤖 Migration des modèles...")

    # Modèles existants
    model_sources = ["data/models/spy", "data/models/nvda", "data/trading/models/spy", "data/trading/models/nvda"]

    for model_dir in model_sources:
        if Path(model_dir).exists():
            # Copier les modèles vers le nouveau répertoire
            target_dir = storage.models_dir / Path(model_dir).name
            if not target_dir.exists():
                shutil.copytree(model_dir, target_dir)
                print(f"  ✅ Modèles {Path(model_dir).name} migré")


def normalize_price_data(data):
    """Normalise les données de prix"""
    # Mapping des colonnes
    column_mapping = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adj_close",
        "Date": "timestamp",
        "Datetime": "timestamp",
    }

    # Renommer les colonnes
    data = data.rename(columns=column_mapping)

    # S'assurer que timestamp est datetime
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
    elif "Date" in data.columns:
        data["timestamp"] = pd.to_datetime(data["Date"])
        data = data.drop("Date", axis=1)

    # Ajouter ticker si manquant
    if "ticker" not in data.columns:
        # Essayer de deviner le ticker depuis le nom du fichier
        data["ticker"] = "UNKNOWN"

    return data


def normalize_news_data(data):
    """Normalise les données de news"""
    # Mapping des colonnes
    column_mapping = {
        "title": "title",
        "summary": "summary",
        "body": "body",
        "content": "body",
        "link": "url",
        "published": "published_at",
        "publishedAt": "published_at",
        "source": "source",
    }

    # Renommer les colonnes
    data = data.rename(columns=column_mapping)

    # S'assurer que published_at est datetime
    if "published_at" in data.columns:
        data["published_at"] = pd.to_datetime(data["published_at"])

    # Ajouter ticker si manquant
    if "ticker" not in data.columns:
        data["ticker"] = "UNKNOWN"

    return data


def normalize_features_data(data):
    """Normalise les données de features"""
    # Vérifier que les features requises sont présentes
    required_features = CONSTANTS.get_feature_columns()
    missing_features = [f for f in required_features if f not in data.columns]

    if missing_features:
        print(f"  ⚠️ Features manquantes: {missing_features}")

    # Ajouter ticker si manquant
    if "ticker" not in data.columns:
        data["ticker"] = "UNKNOWN"

    return data


def extract_ticker_from_filename(filename):
    """Extrait le ticker depuis le nom de fichier"""
    filename_lower = filename.lower()

    if "spy" in filename_lower:
        return "SPY"
    elif "nvda" in filename_lower:
        return "NVDA"
    else:
        return "UNKNOWN"


def cleanup_old_directories():
    """Nettoie les anciens répertoires après migration"""
    print("🧹 Nettoyage des anciens répertoires...")

    # Répertoires à nettoyer (garder une copie de sauvegarde)
    old_dirs = ["data/dataset", "data/trading", "data/backup"]

    for old_dir in old_dirs:
        if Path(old_dir).exists():
            # Créer une sauvegarde
            backup_dir = f"data/backup_migrated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(old_dir, backup_dir)
            print(f"  ✅ {old_dir} déplacé vers {backup_dir}")


def verify_migration():
    """Vérifie que la migration s'est bien passée"""
    print("🔍 Vérification de la migration...")

    storage = UnifiedDataStorage()
    summary = storage.get_data_summary()

    print("\n📊 Résumé des données migrées:")
    print(f"  Prix: {len(summary['prices'])} tickers")
    print(f"  News: {summary['news'].get('files', 0)} fichiers")
    print(f"  Features: {len(summary['features'])} tickers")
    print(f"  Modèles: {len(summary['models'])} tickers")

    # Vérifier que les données essentielles sont présentes
    if summary["prices"]:
        print("  ✅ Données de prix migrées")
    else:
        print("  ❌ Aucune donnée de prix trouvée")

    if summary["features"]:
        print("  ✅ Données de features migrées")
    else:
        print("  ❌ Aucune donnée de features trouvée")

    if summary["models"]:
        print("  ✅ Modèles migrés")
    else:
        print("  ❌ Aucun modèle trouvé")


if __name__ == "__main__":
    print("🚀 Migration vers stockage unifié Sentinel2")
    print("=" * 50)

    try:
        migrate_existing_data()
        verify_migration()
        print("\n🎉 Migration terminée avec succès!")

    except Exception as e:
        print(f"\n❌ Erreur lors de la migration: {e}")
        sys.exit(1)
