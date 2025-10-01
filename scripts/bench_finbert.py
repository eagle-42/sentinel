#!/usr/bin/env python3
"""Script de benchmark pour les tests de performance FinBERT."""

import os
import sys
import time
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config.settings import get_settings
from src.models.news.scoring import SCORING_METRICS, get_scorer


def generate_test_texts(n: int) -> list[str]:
    """Génère des textes de test pour le benchmarking."""
    base_texts = [
        "NVIDIA stock is performing very well with strong earnings",
        "The market is showing mixed signals today with volatility",
        "NVDA shares are declining due to market concerns",
        "S&P 500 index reaches new high on positive sentiment",
        "Market volatility continues as investors remain cautious",
        "Tech stocks are leading the market higher today",
        "Financial sector shows weakness amid economic uncertainty",
        "Energy stocks rally on rising oil prices",
        "Healthcare sector mixed with some strong performers",
        "Consumer discretionary stocks show resilience",
    ]

    # Répéter et varier les textes
    texts = []
    for i in range(n):
        base_text = base_texts[i % len(base_texts)]
        # Ajouter de la variation
        if i % 3 == 0:
            texts.append(f"{base_text} and positive outlook")
        elif i % 3 == 1:
            texts.append(f"{base_text} despite challenges")
        else:
            texts.append(base_text)

    return texts


def run_benchmark(mode: str, n_texts: int = 64, timeout_ms: int = 20000):
    """Exécute le benchmark FinBERT."""
    print(f"Exécution du benchmark FinBERT avec le mode {mode}...")
    print(f"Textes: {n_texts}, Timeout: {timeout_ms}ms")
    print("-" * 50)

    # Effacer les métriques précédentes
    SCORING_METRICS.samples.clear()

    # Récupérer le scorer
    try:
        scorer = get_scorer(mode, timeout_ms=timeout_ms)
    except Exception as e:
        print(f"Erreur lors de la création du scorer: {e}")
        return False

    # Générer les textes de test
    texts = generate_test_texts(n_texts)

    # Exécuter le benchmark
    start_time = time.time()

    try:
        scores = scorer.score_texts(texts)
        elapsed_time = time.time() - start_time

        # Calculer les métriques
        elapsed_ms = elapsed_time * 1000
        throughput = n_texts / elapsed_time  # textes par seconde

        p50 = SCORING_METRICS.p50()
        p95 = SCORING_METRICS.p95()
        count = SCORING_METRICS.count()

        # Afficher les résultats
        print(f"Résultats:")
        print(f"  Temps total: {elapsed_ms:.1f}ms")
        print(f"  Débit: {throughput:.1f} textes/sec")
        print(f"  Latence P50: {p50:.1f}ms")
        print(f"  Latence P95: {p95:.1f}ms")
        print(f"  Échantillons: {count}")
        print(f"  Plage des scores: [{min(scores):.3f}, {max(scores):.3f}]")

        # Vérifier la conformité au timeout
        timeout_threshold = timeout_ms * 1.2  # Buffer de 20%
        if p95 > timeout_threshold:
            print(f"\n❌ ÉCHEC: Latence P95 ({p95:.1f}ms) dépasse le seuil ({timeout_threshold:.1f}ms)")
            return False
        else:
            print(f"\n✅ RÉUSSI: Latence P95 ({p95:.1f}ms) dans le seuil ({timeout_threshold:.1f}ms)")
            return True

    except Exception as e:
        print(f"Erreur pendant le benchmark: {e}")
        return False


def main():
    """Fonction principale de benchmark."""
    # Get settings
    settings = get_settings()
    mode = settings.finbert_mode
    timeout_ms = settings.finbert_timeout_ms

    print(f"Benchmark FinBERT")
    print(f"Mode: {mode}")
    print(f"Timeout: {timeout_ms}ms")
    print("=" * 50)

    if mode == "stub":
        print("Exécution en mode stub (rapide, déterministe)")
        success = run_benchmark("stub", n_texts=64, timeout_ms=timeout_ms)
    elif mode == "real":
        print("Exécution en mode réel (nécessite transformers + torch)")
        try:
            success = run_benchmark("real", n_texts=64, timeout_ms=timeout_ms)
        except ImportError as e:
            print(f"Erreur: {e}")
            print("Installer les dépendances avec: pip install transformers torch")
            success = False
    else:
        print(f"Mode inconnu: {mode}")
        success = False

    # Sortir avec le code approprié
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
