#!/usr/bin/env python3
"""Script pour merger les datasets Kaggle dans 3-kagglefull."""

import sys
from pathlib import Path

import pandas as pd
from loguru import logger


def merge_datasets():
    """Merger les datasets 1-kaggle10years et 2-kagglenvda23-25 dans 3-kagglefull."""

    logger.info("=== MERGE DATASETS KAGGLE ===")

    # Créer le répertoire de destination
    output_dir = Path("data/dataset/3-kagglefull")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Charger les datasets
    logger.info("Chargement des datasets...")

    # Dataset 1 - Kaggle 10 ans
    nvda_1 = pd.read_parquet("data/dataset/1-kaggle10years/nvda_15min_2014_2024.parquet")
    spy_1 = pd.read_parquet("data/dataset/1-kaggle10years/spy_15min_2014_2024.parquet")

    # Dataset 2 - Kaggle récent
    nvda_2 = pd.read_parquet("data/dataset/2-kagglenvda23-25/nvda_15min_2024-03-13_to_latest.parquet")

    logger.info(f"NVDA 1: {len(nvda_1)} lignes ({nvda_1['date'].min()} → {nvda_1['date'].max()})")
    logger.info(f"SPY 1: {len(spy_1)} lignes ({spy_1['date'].min()} → {spy_1['date'].max()})")
    logger.info(f"NVDA 2: {len(nvda_2)} lignes ({nvda_2['date'].min()} → {nvda_2['date'].max()})")

    # 2. Normaliser les colonnes NVDA 2 (Open, High, Low, Close, Volume → open, high, low, close, volume)
    logger.info("Normalisation des colonnes NVDA 2...")
    nvda_2 = nvda_2.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

    # 3. Gérer l'overlap temporel NVDA (garder NVDA 2 pour 2024-03-13+)
    logger.info("Gestion de l'overlap temporel NVDA...")
    nvda_1_clean = nvda_1[nvda_1["date"] < "2024-03-13"]
    logger.info(f"NVDA 1 nettoyé: {len(nvda_1_clean)} lignes (avant 2024-03-13)")

    # 4. Concaténer NVDA (ordre chronologique)
    logger.info("Concaténation NVDA...")
    nvda_full = pd.concat([nvda_1_clean, nvda_2], ignore_index=True)

    # 5. Trier par date CROISSANTE (chronologique pour LSTM)
    logger.info("Tri par date croissante...")
    nvda_full = nvda_full.sort_values("date", ascending=True)
    spy_1 = spy_1.sort_values("date", ascending=True)

    # 6. Vérifications finales
    logger.info("Vérifications finales...")
    logger.info(f"NVDA final: {len(nvda_full)} lignes ({nvda_full['date'].min()} → {nvda_full['date'].max()})")
    logger.info(f"SPY final: {len(spy_1)} lignes ({spy_1['date'].min()} → {spy_1['date'].max()})")

    # Vérifier la continuité temporelle
    nvda_gap = nvda_full["date"].diff().max()
    spy_gap = spy_1["date"].diff().max()
    logger.info(f"Gap max NVDA: {nvda_gap}")
    logger.info(f"Gap max SPY: {spy_gap}")

    # 7. Sauvegarder
    logger.info("Sauvegarde des datasets mergés...")
    nvda_full.to_parquet(output_dir / "nvda_15min_2014_2025.parquet")
    spy_1.to_parquet(output_dir / "spy_15min_2014_2024.parquet")

    # 8. Créer un fichier de métadonnées
    metadata = {
        "nvda": {
            "file": "nvda_15min_2014_2025.parquet",
            "rows": len(nvda_full),
            "date_min": str(nvda_full["date"].min()),
            "date_max": str(nvda_full["date"].max()),
            "sources": ["1-kaggle10years", "2-kagglenvda23-25"],
        },
        "spy": {
            "file": "spy_15min_2014_2024.parquet",
            "rows": len(spy_1),
            "date_min": str(spy_1["date"].min()),
            "date_max": str(spy_1["date"].max()),
            "sources": ["1-kaggle10years"],
        },
    }

    import json

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.success("=== MERGE TERMINÉ ===")
    logger.info(f"Datasets sauvegardés dans: {output_dir}")
    logger.info(f"NVDA: {len(nvda_full)} lignes (2014-2025)")
    logger.info(f"SPY: {len(spy_1)} lignes (2014-2024)")


if __name__ == "__main__":
    merge_datasets()
