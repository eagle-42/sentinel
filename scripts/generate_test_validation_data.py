#!/usr/bin/env python3
"""
Script pour générer des données de test de validation historique
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Ajouter le chemin src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constants import CONSTANTS

def generate_test_validation_data():
    """Génère des données de test pour la validation historique"""
    
    # Créer le répertoire de validation
    validation_path = Path("data/trading/validation_log")
    validation_path.mkdir(parents=True, exist_ok=True)
    
    # Générer des données de test
    np.random.seed(42)  # Pour la reproductibilité
    
    validation_data = []
    base_price = 450.0  # Prix de base pour SPY
    
    # Générer 20 décisions de test sur les 7 derniers jours
    for i in range(20):
        # Timestamp aléatoire dans les 7 derniers jours
        days_ago = np.random.randint(0, 7)
        hours_ago = np.random.randint(0, 24)
        minutes_ago = np.random.randint(0, 60)
        
        timestamp = datetime.now(timezone.utc) - timedelta(
            days=days_ago, 
            hours=hours_ago, 
            minutes=minutes_ago
        )
        
        # Prix actuel avec variation
        price_variation = np.random.normal(0, 0.02)  # 2% de volatilité
        current_price = base_price * (1 + price_variation)
        
        # Prix futur (simulation d'évolution)
        future_change = np.random.normal(0, 0.015)  # 1.5% de changement moyen
        future_price = current_price * (1 + future_change)
        
        # Décision aléatoire
        decisions = ['BUY', 'SELL', 'HOLD']
        decision = np.random.choice(decisions, p=[0.3, 0.3, 0.4])
        
        # Signal de fusion simulé
        fusion_signal = np.random.normal(0, 0.3)
        
        # Confiance
        confidence = np.random.uniform(0.6, 0.9)
        
        # Calculer le changement de prix en pourcentage
        price_change = (future_price - current_price) / current_price * 100
        
        # Évaluer la décision
        is_correct = False
        if decision == 'BUY' and price_change > 0:
            is_correct = True
        elif decision == 'SELL' and price_change < 0:
            is_correct = True
        elif decision == 'HOLD' and abs(price_change) < 1.0:
            is_correct = True
        
        # Calculer l'accuracy
        if decision == 'HOLD':
            accuracy = max(0.5, 1.0 - abs(price_change) / 5.0)  # Plus proche de 0, plus précis
        else:
            accuracy = 0.8 if is_correct else 0.2
        
        # Statut
        if accuracy >= 0.8:
            status = "✅ Correct"
        elif accuracy >= 0.5:
            status = "⚠️ Partiellement correct"
        else:
            status = "❌ Incorrect"
        
        validation_data.append({
            'index': i + 1,
            'timestamp': timestamp,
            'signal': fusion_signal,
            'decision': decision,
            'confidence': confidence,
            'current_price': round(current_price, 2),
            'future_price': round(future_price, 2),
            'price_change': round(price_change, 2),
            'is_correct': is_correct,
            'accuracy': round(accuracy, 3),
            'status': status
        })
    
    # Créer le DataFrame
    df = pd.DataFrame(validation_data)
    
    # Trier par timestamp (plus récent en premier)
    df = df.sort_values('timestamp', ascending=False)
    
    # Sauvegarder
    validation_file = validation_path / "decision_validation_history.parquet"
    df.to_parquet(validation_file, index=False)
    
    print(f"✅ Données de validation générées: {len(df)} décisions")
    print(f"📁 Sauvegardé: {validation_file}")
    
    # Afficher un résumé
    print("\n📊 RÉSUMÉ DES DONNÉES GÉNÉRÉES:")
    print(f"   Décisions BUY: {len(df[df['decision'] == 'BUY'])}")
    print(f"   Décisions SELL: {len(df[df['decision'] == 'SELL'])}")
    print(f"   Décisions HOLD: {len(df[df['decision'] == 'HOLD'])}")
    print(f"   Décisions correctes: {len(df[df['is_correct'] == True])}")
    print(f"   Accuracy moyenne: {df['accuracy'].mean():.3f}")
    print(f"   Prix min: ${df['current_price'].min():.2f}")
    print(f"   Prix max: ${df['current_price'].max():.2f}")
    print(f"   Gain total: ${(df['future_price'] - df['current_price']).sum():.2f}")
    
    return df

if __name__ == "__main__":
    print("🔧 Génération de données de test pour la validation historique")
    print("=" * 60)
    
    df = generate_test_validation_data()
    
    print("\n📋 PREMIÈRES DÉCISIONS:")
    print(df[['timestamp', 'decision', 'current_price', 'future_price', 'price_change', 'status']].head())
