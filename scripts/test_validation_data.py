#!/usr/bin/env python3
"""
Script pour tester les données de validation historique
"""

import sys
from pathlib import Path

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui.services.historical_validation_service import HistoricalValidationService


def test_validation_data():
    """Teste les données de validation historique"""

    print("🔍 Test des données de validation historique")
    print("=" * 50)

    # Tester le service
    service = HistoricalValidationService()
    validation_summary = service.get_validation_summary("SPY", days=7)

    print("=== RÉSULTATS DE VALIDATION ===")
    print(f"Status: {validation_summary.get('status', 'N/A')}")
    print(f"Total décisions: {validation_summary.get('total_decisions', 0)}")
    print(f"Décisions correctes: {validation_summary.get('correct_decisions', 0)}")
    print(f"Accuracy rate: {validation_summary.get('accuracy_rate', 0):.3f}")

    validation_results = validation_summary.get("validation_results", [])
    if validation_results:
        print(f"\nPremières décisions:")
        for i, result in enumerate(validation_results[:3]):
            current_price = result.get("current_price", 0)
            future_price = result.get("future_price", 0)
            gain = future_price - current_price
            print(
                f"  {i+1}. {result.get('decision', 'N/A')} - Prix: ${current_price:.2f} → ${future_price:.2f} (Gain: ${gain:+.2f})"
            )

        # Calculer le gain total
        total_gain = sum(r.get("future_price", 0) - r.get("current_price", 0) for r in validation_results)
        print(f"\nGain total: ${total_gain:.2f}")

        # Statistiques détaillées
        buy_decisions = [r for r in validation_results if r.get("decision") == "BUY"]
        sell_decisions = [r for r in validation_results if r.get("decision") == "SELL"]
        hold_decisions = [r for r in validation_results if r.get("decision") == "HOLD"]

        print(f"\nDétail par type de décision:")
        print(f"  BUY: {len(buy_decisions)} décisions")
        print(f"  SELL: {len(sell_decisions)} décisions")
        print(f"  HOLD: {len(hold_decisions)} décisions")

    else:
        print("Aucune donnée de validation trouvée")


if __name__ == "__main__":
    test_validation_data()
