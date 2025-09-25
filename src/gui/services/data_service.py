"""
Service de données pour Streamlit
Chargement et filtrage des données historiques
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional


class DataService:
    """Service de données optimisé pour Streamlit"""
    
    def __init__(self):
        self.data_path = Path("data/historical/yfinance")
        self.cache = {}
        logger.info("📊 Service de données initialisé")
    
    def load_data(self, ticker: str) -> pd.DataFrame:
        """Charge les données pour un ticker avec normalisation complète"""
        try:
            if ticker in self.cache:
                return self.cache[ticker]
            
            file_path = self.data_path / f"{ticker}_1999_2025.parquet"
            
            if not file_path.exists():
                logger.error(f"❌ Fichier non trouvé: {file_path}")
                return pd.DataFrame()
            
            # Chargement avec normalisation des colonnes
            df = pd.read_parquet(file_path)
            
            # Normalisation des colonnes (majuscules)
            df.columns = df.columns.str.upper()
            
            # Conversion des dates en UTC pour éviter les décalages
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'], utc=True)
            
            # Tri par date pour éviter les "zigzags"
            df = df.sort_values('DATE').reset_index(drop=True)
            
            # Validation des données
            df = self._validate_data(df, ticker)
            
            # Cache pour éviter les rechargements
            self.cache[ticker] = df
            
            logger.info(f"✅ Données chargées pour {ticker}: {len(df)} lignes")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement {ticker}: {e}")
            return pd.DataFrame()
    
    def _validate_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Valide et nettoie les données"""
        if df.empty:
            return df
        
        # Vérifier les colonnes requises
        required_cols = ['DATE', 'CLOSE', 'VOLUME']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"⚠️ Colonnes manquantes pour {ticker}: {missing_cols}")
            return pd.DataFrame()
        
        # Nettoyer les valeurs NaN/Inf
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Supprimer les lignes avec des prix invalides
        df = df.dropna(subset=['CLOSE'])
        
        # Vérifier que les prix sont positifs
        df = df[df['CLOSE'] > 0]
        
        if df.empty:
            logger.error(f"❌ Aucune donnée valide pour {ticker}")
            return pd.DataFrame()
        
        logger.info(f"✅ Données validées pour {ticker}: {len(df)} lignes valides")
        return df
    
    def filter_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filtre les données par période"""
        if df.empty:
            return df
        
        periods = {
            "7 derniers jours": 7,
            "1 mois": 30,
            "3 mois": 90,
            "6 derniers mois": 180,
            "1 an": 365,
            "3 ans": 1095,
            "5 ans": 1825,
            "10 ans": 3650,
            "Total (toutes les données)": None
        }
        
        if period not in periods:
            logger.warning(f"⚠️ Période inconnue: {period}")
            return df
        
        days = periods[period]
        if days is None:
            return df
        
        # Utiliser la dernière date des données comme référence
        last_date = df['DATE'].max()
        start_date = last_date - pd.Timedelta(days=days)
        
        # Filtrer et trier
        filtered_df = df[df['DATE'] >= start_date].copy()
        filtered_df = filtered_df.sort_values('DATE').reset_index(drop=True)
        
        logger.info(f"✅ Filtrage {period}: {len(filtered_df)} lignes")
        return filtered_df
    
    def get_available_tickers(self) -> list:
        """Retourne la liste des tickers disponibles"""
        if not self.data_path.exists():
            return []
        
        tickers = []
        for file_path in self.data_path.glob("*.parquet"):
            ticker = file_path.stem.replace("_1999_2025", "")
            tickers.append(ticker)
        
        return sorted(tickers)
