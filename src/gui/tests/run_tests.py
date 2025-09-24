#!/usr/bin/env python3
"""
Script principal pour exécuter tous les tests
"""

import sys
import unittest
from pathlib import Path

# Ajouter le chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_all_tests():
    """Exécute tous les tests"""
    print("🧪 Démarrage des tests Sentinel UI...")
    print("=" * 60)
    
    # Liste des modules de test
    test_modules = [
        'test_data_service',
        'test_prediction_service', 
        'test_chart_component',
        'test_filters_component',
        'test_analysis_page',
        'test_integration'
    ]
    
    # Charger tous les tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for module_name in test_modules:
        try:
            module = __import__(module_name)
            tests = loader.loadTestsFromModule(module)
            suite.addTests(tests)
            print(f"✅ Module {module_name} chargé")
        except Exception as e:
            print(f"❌ Erreur lors du chargement de {module_name}: {e}")
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    print(f"Tests exécutés: {result.testsRun}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    print(f"Succès: {result.testsRun - len(result.failures) - len(result.errors)}")
    
    if result.failures:
        print("\n❌ ÉCHECS:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\n💥 ERREURS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    # Retourner le code de sortie
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())

