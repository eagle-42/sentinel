"""
Stockage Unifié Sentinel2
Gestion centralisée du stockage des données
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger

from ..constants import CONSTANTS


class ParquetStorage:
    """Stockage en format Parquet optimisé avec sauvegarde incrémentale"""
    
    def __init__(self, base_path: Path = None):
        """Initialise le stockage Parquet"""
        self.base_path = base_path or CONSTANTS.DATA_ROOT
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Stockage Parquet initialisé: {self.base_path}")
    
    def _save_incremental(self, 
                         data: pd.DataFrame, 
                         file_path: Path,
                         dedup_cols: List[str],
                         sort_col: str = None) -> Path:
        """Méthode générique de sauvegarde incrémentale
        
        Args:
            data: DataFrame à sauvegarder
            file_path: Chemin du fichier parquet
            dedup_cols: Colonnes pour déduplication
            sort_col: Colonne pour le tri (optionnel)
            
        Returns:
            Path du fichier sauvegardé
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.exists():
                try:
                    existing_data = pd.read_parquet(file_path)
                    combined_data = pd.concat([existing_data, data], ignore_index=True)
                    combined_data = combined_data.drop_duplicates(subset=dedup_cols, keep='last')
                    
                    if sort_col and sort_col in combined_data.columns:
                        combined_data = combined_data.sort_values(sort_col)
                    
                    data_to_save = combined_data
                    logger.info(f"💾 Mise à jour incrémentale: {len(existing_data)} → {len(combined_data)} lignes")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lecture fichier existant, sauvegarde complète: {e}")
                    data_to_save = data
            else:
                data_to_save = data
                logger.info(f"💾 Nouveau fichier créé: {len(data)} lignes")
            
            data_to_save.to_parquet(file_path, index=False, compression='snappy')
            logger.info(f"💾 Fichier sauvegardé: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")
            raise
    
    def save_prices(self, 
                   data: pd.DataFrame, 
                   ticker: str, 
                   interval: str = "15min") -> Path:
        """Sauvegarde les données de prix de manière incrémentale"""
        file_path = CONSTANTS.get_data_path("prices", ticker, interval)
        logger.info(f"💾 Sauvegarde prix {ticker} ({interval})")
        return self._save_incremental(data, file_path, dedup_cols=['ts_utc'], sort_col='ts_utc')
    
    def load_prices(self, 
                   ticker: str, 
                   interval: str = "15min") -> pd.DataFrame:
        """Charge les données de prix"""
        file_path = CONSTANTS.get_data_path("prices", ticker, interval)
        return self._load_data(file_path, f"Prix {ticker}")
    
    def save_news(self, 
                 data: pd.DataFrame, 
                 ticker: str = None) -> Path:
        """Sauvegarde les données de news de manière incrémentale"""
        file_path = CONSTANTS.get_data_path("news", ticker) if ticker else CONSTANTS.NEWS_DIR / "all_news.parquet"
        logger.info(f"💾 Sauvegarde news")
        return self._save_incremental(data, file_path, dedup_cols=['timestamp', 'title'], sort_col='timestamp')
    
    def load_news(self, 
                 ticker: str = None) -> pd.DataFrame:
        """Charge les données de news"""
        if ticker:
            file_path = CONSTANTS.get_data_path("news", ticker)
        else:
            news_files = list(CONSTANTS.NEWS_DIR.glob("*.parquet"))
            if not news_files:
                logger.warning("⚠️ Aucun fichier news trouvé")
                return pd.DataFrame()
            file_path = max(news_files, key=lambda x: x.stat().st_mtime)
        
        return self._load_data(file_path, "News")
    
    def save_sentiment(self, 
                      data: pd.DataFrame, 
                      ticker: str) -> Path:
        """Sauvegarde les données de sentiment de manière incrémentale"""
        file_path = CONSTANTS.get_data_path("sentiment", ticker)
        logger.info(f"💾 Sauvegarde sentiment {ticker}")
        return self._save_incremental(data, file_path, dedup_cols=['timestamp'], sort_col='timestamp')
    
    def load_sentiment(self, ticker: str) -> pd.DataFrame:
        """Charge les données de sentiment"""
        file_path = CONSTANTS.get_data_path("sentiment", ticker)
        return self._load_data(file_path, f"Sentiment {ticker}")
    
    def _load_data(self, file_path: Path, data_type: str) -> pd.DataFrame:
        """Méthode générique de chargement de données"""
        try:
            if not file_path.exists():
                logger.warning(f"⚠️ Fichier {data_type} non trouvé: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path)
            logger.info(f"💾 {data_type} chargé: {len(df)} lignes")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement {data_type}: {e}")
            return pd.DataFrame()

class DataStorage:
    """Stockage unifié - Wrapper simple autour de ParquetStorage"""
    
    def __init__(self):
        """Initialise le stockage unifié"""
        self.parquet = ParquetStorage()
        logger.info("💾 Stockage unifié initialisé")
