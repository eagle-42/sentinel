#!/usr/bin/env python3
"""
Force le refresh des prix intraday avec yfinance
Récupère les données temps réel d'aujourd'hui
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pytz
from loguru import logger

# Pas besoin d'import UnifiedStorage, on sauvegarde directement

def get_yfinance_intraday(ticker: str = "SPY") -> pd.DataFrame:
    """Récupère les données intraday depuis yfinance"""
    try:
        import yfinance as yf
        
        logger.info(f"📊 Récupération données intraday pour {ticker}")
        
        # Essayer différentes approches
        spy = yf.Ticker(ticker)
        
        # Approche 1: Derniers 2 jours
        try:
            data = spy.history(period='2d', interval='15m')
            if not data.empty:
                logger.info(f"✅ Méthode period='2d' : {len(data)} barres")
                return data
        except Exception as e:
            logger.warning(f"⚠️ Méthode period='2d' échouée: {e}")
        
        # Approche 2: Start/End explicites
        try:
            end = datetime.now()
            start = end - timedelta(days=2)
            data = spy.history(start=start, end=end, interval='15m')
            if not data.empty:
                logger.info(f"✅ Méthode start/end : {len(data)} barres")
                return data
        except Exception as e:
            logger.warning(f"⚠️ Méthode start/end échouée: {e}")
        
        logger.error("❌ Aucune méthode n'a fonctionné")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération yfinance: {e}")
        return pd.DataFrame()

def main():
    """Force le refresh"""
    logger.info("🚀 FORCE REFRESH INTRADAY")
    logger.info("=" * 60)
    
    # Récupérer les données
    data = get_yfinance_intraday("SPY")
    
    if data.empty:
        logger.error("❌ Aucune donnée récupérée")
        return
    
    # Convertir au format unifié
    data = data.reset_index()
    data = data.rename(columns={
        'Datetime': 'ts_utc',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # S'assurer que ts_utc est timezone-aware
    if data['ts_utc'].dt.tz is None:
        data['ts_utc'] = pd.to_datetime(data['ts_utc'], utc=True)
    
    data['ticker'] = 'SPY'
    
    # Afficher aperçu
    logger.info(f"📊 {len(data)} barres récupérées")
    logger.info(f"📅 De {data['ts_utc'].min()} à {data['ts_utc'].max()}")
    
    # Afficher les 5 dernières
    logger.info("\n📋 5 DERNIERS PRIX:")
    paris_tz = pytz.timezone('Europe/Paris')
    for _, row in data.tail(5).iterrows():
        paris_time = row['ts_utc'].astimezone(paris_tz)
        logger.info(f"  {paris_time.strftime('%Y-%m-%d %H:%M')} Paris | ${row['close']:.2f}")
    
    # Charger les données existantes
    existing_file = Path("data/realtime/prices/spy_15min.parquet")
    
    if existing_file.exists():
        existing = pd.read_parquet(existing_file)
        existing['ts_utc'] = pd.to_datetime(existing['ts_utc'])
        logger.info(f"\n📊 Données existantes: {len(existing)} lignes")
        logger.info(f"📅 Dernière donnée existante: {existing['ts_utc'].max()}")
        
        # Fusionner et dédupliquer
        combined = pd.concat([existing, data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['ts_utc'], keep='last')
        combined = combined.sort_values('ts_utc').reset_index(drop=True)
        
        logger.info(f"\n💾 Après fusion: {len(combined)} lignes")
        logger.info(f"📅 Nouvelle période: {combined['ts_utc'].min()} à {combined['ts_utc'].max()}")
        
        # Sauvegarder
        combined.to_parquet(existing_file, index=False)
        logger.info(f"\n✅ Données sauvegardées: {existing_file}")
        
    else:
        data.to_parquet(existing_file, index=False)
        logger.info(f"\n✅ Nouvelles données sauvegardées: {existing_file}")

if __name__ == "__main__":
    main()
