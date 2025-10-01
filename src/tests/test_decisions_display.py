#!/usr/bin/env python3
"""
Test de vérification de l'affichage des décisions dans Streamlit
"""

import sys
from pathlib import Path
import json

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_decisions_file():
    """Vérifie que le fichier de décisions existe et est valide"""
    decisions_file = Path("data/trading/decisions_log/trading_decisions.json")
    
    print("🔍 VÉRIFICATION DES DÉCISIONS")
    print("=" * 60)
    
    if not decisions_file.exists():
        print("❌ Fichier de décisions introuvable")
        return False
    
    print(f"✅ Fichier trouvé: {decisions_file}")
    
    try:
        with open(decisions_file, 'r') as f:
            decisions = json.load(f)
        
        print(f"✅ Fichier JSON valide")
        print(f"📊 Nombre de décisions: {len(decisions)}")
        print()
        
        if decisions:
            print("📋 DERNIÈRES DÉCISIONS:")
            print("-" * 60)
            for i, decision in enumerate(decisions[-5:], 1):  # 5 dernières
                print(f"\n{i}. Décision {decision.get('ticker', 'N/A')}")
                print(f"   Heure: {decision.get('timestamp', 'N/A')}")
                print(f"   Décision: {decision.get('decision', 'N/A')}")
                print(f"   Confiance: {decision.get('confidence', 0)*100:.1f}%")
                print(f"   Signal fusionné: {decision.get('fused_signal', 0):.4f}")
        else:
            print("⚠️ Aucune décision enregistrée")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lecture JSON: {e}")
        return False

def test_service_loading():
    """Teste le chargement des décisions via le service"""
    print("\n" + "=" * 60)
    print("🔍 TEST SERVICE HISTORICAL VALIDATION")
    print("=" * 60)
    
    try:
        from gui.services.historical_validation_service import HistoricalValidationService
        
        service = HistoricalValidationService()
        print("✅ Service initialisé")
        
        # Charger les décisions
        decisions_df = service.load_historical_decisions()
        
        if decisions_df.empty:
            print("⚠️ DataFrame vide (normal si aucune décision)")
        else:
            print(f"✅ DataFrame chargé: {len(decisions_df)} lignes")
            print(f"📊 Colonnes: {list(decisions_df.columns)}")
            print()
            print("📋 APERÇU:")
            print(decisions_df.tail())
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur service: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Exécute tous les tests"""
    print()
    print("🚀 TEST AFFICHAGE DÉCISIONS STREAMLIT")
    print()
    
    # Test 1: Fichier de décisions
    test1 = test_decisions_file()
    
    # Test 2: Service de chargement
    test2 = test_service_loading()
    
    print()
    print("=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    print(f"Fichier décisions: {'✅ OK' if test1 else '❌ ERREUR'}")
    print(f"Service validation: {'✅ OK' if test2 else '❌ ERREUR'}")
    print()
    
    if test1 and test2:
        print("✅ TOUT EST OPÉRATIONNEL")
        print()
        print("💡 Pour voir les décisions dans Streamlit:")
        print("   1. Ouvrir http://localhost:8501")
        print("   2. Aller sur la page 'Production'")
        print("   3. Section 'Décisions Récentes - Synthèse'")
    else:
        print("❌ PROBLÈME DÉTECTÉ - Vérifier les logs ci-dessus")

if __name__ == "__main__":
    main()
