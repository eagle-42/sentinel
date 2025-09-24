#!/usr/bin/env python3
"""Script utilitaire pour mettre à jour les données du crawler."""

import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.crawler.providers.polygon.update_15min_data import update_crawler_data

if __name__ == "__main__":
    print("🔄 Mise à jour des données du crawler...")
    results = update_crawler_data()
    
    print("\n📊 Résumé de la mise à jour:")
    for ticker, info in results.items():
        print(f"  {ticker}: {info['rows']} lignes total ({info.get('new_rows', 0)} nouvelles)")
        print(f"    Période: {info['start']} à {info['end']}")
        print(f"    Statut: {info['status']}")
    
    print("\n✅ Mise à jour terminée !")
