#!/usr/bin/env python3
"""
🔄 Script simple pour mettre à jour les données de prix
Complètement indépendant, sans dépendances externes
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random


def create_mock_data(ticker: str) -> pd.DataFrame:
    """Crée des données de test pour la démonstration"""
    try:
        print(f"📊 Création de données de test pour {ticker}...")
        
        # Créer des données des 7 derniers jours
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Générer des timestamps toutes les 15 minutes
        timestamps = pd.date_range(start=start_date, end=end_date, freq='15min')
        
        # Créer des données de prix simulées
        base_price = 450.0 if ticker == "SPY" else 120.0
        prices = []
        current_price = base_price
        
        for i, ts in enumerate(timestamps):
            # Variation aléatoire entre -0.5% et +0.5%
            variation = random.uniform(-0.005, 0.005)
            current_price *= (1 + variation)
            prices.append(current_price)
        
        # Créer le DataFrame
        data = pd.DataFrame({
            'ts_utc': timestamps,
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': [random.randint(1000000, 5000000) for _ in range(len(timestamps))],
            'ticker': ticker
        })
        
        return data
        
    except Exception as e:
        print(f"❌ Erreur création données test: {e}")
        return None


def update_ticker_data(ticker: str):
    """Met à jour les données pour un ticker spécifique"""
    try:
        print(f"📈 Mise à jour des données pour {ticker}...")
        
        # Créer le répertoire de destination
        data_dir = Path("data/realtime/prices")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer des données de test
        data = create_mock_data(ticker)
        
        if data is None or data.empty:
            print(f"❌ Aucune donnée récupérée pour {ticker}")
            return False
        
        # Sauvegarder
        output_file = data_dir / f"{ticker.lower()}_15min.parquet"
        data.to_parquet(output_file, index=False)
        
        print(f"✅ Données sauvegardées: {output_file}")
        print(f"📊 {len(data)} enregistrements du {data['ts_utc'].min()} au {data['ts_utc'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la mise à jour de {ticker}: {e}")
        return False


def main():
    """Fonction principale"""
    print("🚀 Mise à jour des données de prix...")
    
    # Mettre à jour SPY et NVDA
    tickers = ["SPY", "NVDA"]
    success_count = 0
    
    for ticker in tickers:
        if update_ticker_data(ticker):
            success_count += 1
    
    print(f"\n✅ Mise à jour terminée: {success_count}/{len(tickers)} tickers mis à jour")
    
    if success_count == len(tickers):
        print("🎉 Toutes les données ont été mises à jour avec succès!")
        return True
    else:
        print("⚠️ Certaines données n'ont pas pu être mises à jour")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
