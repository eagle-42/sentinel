#!/usr/bin/env python3
"""
Script de test de validation des décisions
Vérifie que le système détecte correctement les prix +15min
"""

import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui.services.historical_validation_service import HistoricalValidationService

def main():
    """Teste la validation des décisions"""
    print("\n" + "=" * 60)
    print("🧪 TEST VALIDATION DES DÉCISIONS")
    print("=" * 60)
    
    service = HistoricalValidationService()
    
    # Tester avec SPY
    print("\n📊 Validation pour SPY (7 derniers jours)...")
    validation_summary = service.get_validation_summary("SPY", days=7)
    
    print(f"\n✅ Status: {validation_summary.get('status')}")
    print(f"📋 Total décisions: {validation_summary.get('total_decisions', 0)}")
    
    results = validation_summary.get('validation_results', [])
    print(f"📊 Résultats validation: {len(results)}")
    
    if results:
        print("\n" + "-" * 60)
        print("DÉTAIL DES DÉCISIONS:")
        print("-" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.get('timestamp')}")
            print(f"   Décision: {result.get('decision')}")
            print(f"   Prix -15min: ${result.get('current_price', 0):.2f}")
            print(f"   Prix +15min: ${result.get('future_price', 0):.2f}")
            print(f"   Variation: {result.get('price_change', 0):.2f}%")
            print(f"   Résultat: {result.get('status', 'N/A')}")
    
    # Statistiques
    stats = validation_summary.get('summary_stats', {})
    if stats:
        print("\n" + "=" * 60)
        print("📊 STATISTIQUES GLOBALES:")
        print("=" * 60)
        print(f"Décisions correctes: {stats.get('correct_decisions', 0)}/{stats.get('total_decisions', 0)}")
        print(f"Taux de réussite: {stats.get('accuracy_rate', 0)*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ TEST TERMINÉ")
    print("=" * 60)

if __name__ == "__main__":
    main()
