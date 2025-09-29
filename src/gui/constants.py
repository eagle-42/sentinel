"""
Constantes pour l'interface GUI - Normalisation des colonnes
"""

import pandas as pd

# Colonnes normalisées en minuscules
COLUMNS = {
    'DATE': 'date',
    'TS_UTC': 'date', 
    'OPEN': 'open',
    'HIGH': 'high', 
    'LOW': 'low',
    'CLOSE': 'close',
    'VOLUME': 'volume',
    'TICKER': 'ticker'
}

# Mapping des colonnes sources vers colonnes normalisées
COLUMN_MAPPING = {
    'date': 'date',
    'ts_utc': 'date',
    'DATE': 'date',
    'TS_UTC': 'date',
    'open': 'open',
    'OPEN': 'open',
    'high': 'high', 
    'HIGH': 'high',
    'low': 'low',
    'LOW': 'low',
    'close': 'close',
    'CLOSE': 'close',
    'volume': 'volume',
    'VOLUME': 'volume',
    'ticker': 'ticker',
    'TICKER': 'ticker'
}

def normalize_columns(df):
    """Normalise toutes les colonnes d'un DataFrame en minuscules"""
    if df.empty:
        return df
    
    # Renommer les colonnes selon le mapping
    df_normalized = df.rename(columns=COLUMN_MAPPING)
    
    # S'assurer que les colonnes essentielles existent
    required_columns = ['date', 'close']
    missing_columns = [col for col in required_columns if col not in df_normalized.columns]
    
    if missing_columns:
        raise ValueError(f"Colonnes manquantes après normalisation: {missing_columns}")
    
    # Normaliser les timezones des dates
    if 'date' in df_normalized.columns:
        df_normalized['date'] = pd.to_datetime(df_normalized['date'], utc=True)
    
    return df_normalized
