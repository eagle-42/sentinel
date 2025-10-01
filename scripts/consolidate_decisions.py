"""
Script de consolidation des décisions de trading
Consolide tous les fichiers de décisions par date en un fichier unifié
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ajouter le chemin src pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from constants import CONSTANTS


def consolidate_decisions():
    """Consolide tous les fichiers de décisions en un seul fichier unifié"""
    logger.info("🤖 Consolidation des décisions de trading...")

    decisions_dir = CONSTANTS.DATA_ROOT / "trading" / "decisions_log"
    unified_file = decisions_dir / "trading_decisions.json"

    # Chercher tous les fichiers de décisions par date
    decision_files = list(decisions_dir.glob("decisions_*.json"))

    if not decision_files:
        logger.info("ℹ️ Aucun fichier de décisions à consolider")
        return

    # Charger et fusionner tous les fichiers
    all_decisions = []
    for decision_file in decision_files:
        try:
            with open(decision_file, "r") as f:
                decisions = json.load(f)
                all_decisions.extend(decisions)
            logger.info(f"📄 Chargé: {decision_file.name} ({len(decisions)} décisions)")
        except Exception as e:
            logger.error(f"❌ Erreur lecture {decision_file.name}: {e}")

    if not all_decisions:
        logger.warning("⚠️ Aucune décision valide")
        return

    # Trier par timestamp pour garder l'ordre chronologique
    all_decisions.sort(key=lambda x: x.get("timestamp", ""))

    # Limiter à 1000 décisions les plus récentes
    if len(all_decisions) > 1000:
        all_decisions = all_decisions[-1000:]
        logger.info(f"📊 Décisions limitées à 1000 (supprimé {len(all_decisions) - 1000} anciennes)")

    # Sauvegarder le fichier consolidé
    with open(unified_file, "w") as f:
        json.dump(all_decisions, f, indent=2)
    logger.success(f"✅ Décisions consolidées: {unified_file} ({len(all_decisions)} décisions)")

    # Supprimer les anciens fichiers
    for decision_file in decision_files:
        try:
            decision_file.unlink()
            logger.info(f"🗑️ Supprimé: {decision_file.name}")
        except Exception as e:
            logger.error(f"❌ Erreur suppression {decision_file.name}: {e}")


def main():
    """Fonction principale de consolidation"""
    logger.info("🔄 === CONSOLIDATION DES DÉCISIONS DE TRADING ===")

    try:
        consolidate_decisions()
        logger.success("✅ Consolidation des décisions terminée avec succès!")

    except Exception as e:
        logger.error(f"❌ Erreur lors de la consolidation: {e}")
        raise


if __name__ == "__main__":
    main()
