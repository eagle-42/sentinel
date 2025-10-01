#!/usr/bin/env python3
"""
Script de test pour valider le fonctionnement de decisions_vs_future_price.py
"""

import subprocess
import sys
from pathlib import Path


def run_test(test_name, args):
    """Exécute un test et vérifie le succès"""
    print(f"\n🧪 TEST: {test_name}")
    print("-" * 50)

    cmd = ["uv", "run", "python", "src/notebooks/decisions_vs_future_price.py"] + args
    print(f"Commande: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/eagle/DevTools/sentinel2")

        if result.returncode == 0:
            print("✅ SUCCÈS")
            print("Sortie:")
            print(result.stdout)
            return True
        else:
            print("❌ ÉCHEC")
            print("Erreur:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ ERREUR: {e}")
        return False


def main():
    print("🚀 TESTS DE VALIDATION DU SCRIPT DECISIONS_VS_FUTURE_PRICE")
    print("=" * 70)

    tests = [
        ("Test par défaut", []),
        ("Test avec horizon 2 pas", ["--horizon-steps", "2"]),
        ("Test avec fenêtre 7 jours", ["--window-days", "7"]),
        (
            "Test avec données historiques",
            [
                "--prices",
                "data/historical/yfinance/SPY_1999_2025.parquet",
                "--time-col",
                "date",
                "--price-col",
                "close",
            ],
        ),
        ("Test avec répertoire de sortie personnalisé", ["--output-dir", "src/notebooks/test_output"]),
    ]

    results = []

    for test_name, args in tests:
        success = run_test(test_name, args)
        results.append((test_name, success))

    # Résumé des tests
    print("\n📊 RÉSUMÉ DES TESTS")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:40} {status}")
        if success:
            passed += 1

    print(f"\nRésultat: {passed}/{total} tests réussis")

    if passed == total:
        print("🎉 TOUS LES TESTS SONT PASSÉS!")
        return 0
    else:
        print("⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
        return 1


if __name__ == "__main__":
    sys.exit(main())
