#!/usr/bin/env python3
"""
Service de données unifié - Gestion des données parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

# Import de la configuration centralisée
from .service_config import get_service_config, get_data_file_path


class DataService:
    """Service unifié pour la gestion des données"""
    
    def __init__(self):
        self.data_cache = {}
        logger.info("📊 Service de données initialisé")
    
    def load_data(self, ticker: str) -> pd.DataFrame:
        """Charge les données pour un ticker"""
        try:
            if ticker in self.data_cache:
                return self.data_cache[ticker]
            
            # Utiliser la configuration centralisée
            data_path = get_data_file_path(ticker, "yfinance")
            
            if not data_path.exists():
                raise FileNotFoundError(f"Données non trouvées pour {ticker} à {data_path}")
            
            df = pd.read_parquet(data_path)
            
            # Normaliser les colonnes
            df.columns = df.columns.str.upper()
            
            # Créer la colonne date en majuscules
            if 'DATE' not in df.columns:
                df['DATE'] = pd.to_datetime(df.index) if df.index.name == 'date' else pd.to_datetime(df['date'])
            else:
                df['DATE'] = pd.to_datetime(df['DATE'])
            
            # Mettre en cache
            self.data_cache[ticker] = df
            
            logger.info(f"✅ Données chargées pour {ticker}: {len(df)} lignes")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des données {ticker}: {e}")
            return pd.DataFrame()
    
    def filter_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filtre les données par période"""
        try:
            if df.empty:
                return df
            
            # Utiliser la configuration centralisée pour les périodes
            from config.unified_config import get_config
            config = get_config()
            periods = config.periods
            
            if period not in periods:
                logger.warning(f"⚠️ Période non reconnue: {period}, utilisation de toutes les données")
                return df
            
            today = pd.Timestamp.now()
            days = periods[period]
            start_date = today - pd.Timedelta(days=days)
            
            # Filtrer par date - gérer les timezones
            # Convertir start_date en timezone aware si nécessaire
            if df['DATE'].dt.tz is not None and start_date.tz is None:
                start_date = start_date.tz_localize('UTC')
            elif df['DATE'].dt.tz is None and start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            
            filtered_df = df[df['DATE'] >= start_date].copy()
            
            logger.info(f"✅ Filtrage {period}: {len(filtered_df)} lignes")
            return filtered_df
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du filtrage {period}: {e}")
            return df
    
    def get_price_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Récupère les données de prix filtrées"""
        df = self.load_data(ticker)
        if df.empty:
            return df
        
        return self.filter_by_period(df, period)
    
    def get_volume_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Récupère les données de volume filtrées"""
        return self.get_price_data(ticker, period)  # Même source
    
    def get_prediction_data(self, ticker: str, period: str) -> pd.DataFrame:
        """Récupère les données pour prédiction LSTM"""
        return self.get_price_data(ticker, period)  # Même source
