"""
💾 Stockage Unifié Sentinel2
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
    """Stockage en format Parquet optimisé"""
    
    def __init__(self, base_path: Path = None):
        """Initialise le stockage Parquet"""
        self.base_path = base_path or CONSTANTS.DATA_ROOT
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Stockage Parquet initialisé: {self.base_path}")
    
    def save_prices(self, 
                   data: pd.DataFrame, 
                   ticker: str, 
                   interval: str = "1min") -> Path:
        """Sauvegarde les données de prix de manière incrémentale"""
        try:
            # Chemin de sauvegarde
            file_path = CONSTANTS.get_data_path("prices", ticker, interval)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarde incrémentale : charger les données existantes et ajouter les nouvelles
            if file_path.exists():
                try:
                    existing_data = pd.read_parquet(file_path)
                    # Concaténer et dédupliquer par timestamp
                    combined_data = pd.concat([existing_data, data], ignore_index=True)
                    combined_data = combined_data.drop_duplicates(subset=['ts_utc'], keep='last')
                    combined_data = combined_data.sort_values('ts_utc')
                    data_to_save = combined_data
                    logger.info(f"💾 Prix {ticker} mis à jour incrémentalement: {len(existing_data)} → {len(combined_data)} lignes")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lecture fichier existant {ticker}, sauvegarde complète: {e}")
                    data_to_save = data
            else:
                data_to_save = data
                logger.info(f"💾 Nouveau fichier prix {ticker} créé: {len(data)} lignes")
            
            # Sauvegarder
            data_to_save.to_parquet(file_path, index=False, compression='snappy')
            
            logger.info(f"💾 Prix {ticker} sauvegardé: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde prix {ticker}: {e}")
            raise
    
    def load_prices(self, 
                   ticker: str, 
                   interval: str = "1min") -> pd.DataFrame:
        """Charge les données de prix"""
        try:
            file_path = CONSTANTS.get_data_path("prices", ticker, interval)
            
            if not file_path.exists():
                logger.warning(f"⚠️ Fichier prix non trouvé: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path)
            logger.info(f"💾 Prix {ticker} chargé: {len(df)} lignes")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement prix {ticker}: {e}")
            return pd.DataFrame()
    
    def save_news(self, 
                 data: pd.DataFrame, 
                 ticker: str = None) -> Path:
        """Sauvegarde les données de news de manière incrémentale"""
        try:
            # Chemin de sauvegarde
            if ticker:
                file_path = CONSTANTS.get_data_path("news", ticker)
            else:
                file_path = CONSTANTS.NEWS_DIR / "all_news.parquet"  # Un seul fichier pour toutes les news
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarde incrémentale : charger les données existantes et ajouter les nouvelles
            if file_path.exists():
                try:
                    existing_data = pd.read_parquet(file_path)
                    # Concaténer et dédupliquer par timestamp et title
                    combined_data = pd.concat([existing_data, data], ignore_index=True)
                    combined_data = combined_data.drop_duplicates(subset=['timestamp', 'title'], keep='last')
                    combined_data = combined_data.sort_values('timestamp')
                    data_to_save = combined_data
                    logger.info(f"💾 News mis à jour incrémentalement: {len(existing_data)} → {len(combined_data)} lignes")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lecture fichier news existant, sauvegarde complète: {e}")
                    data_to_save = data
            else:
                data_to_save = data
                logger.info(f"💾 Nouveau fichier news créé: {len(data)} lignes")
            
            # Sauvegarder
            data_to_save.to_parquet(file_path, index=False, compression='snappy')
            
            logger.info(f"💾 News sauvegardé: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde news: {e}")
            raise
    
    def load_news(self, 
                 ticker: str = None,
                 latest: bool = True) -> pd.DataFrame:
        """Charge les données de news"""
        try:
            if ticker:
                file_path = CONSTANTS.get_data_path("news", ticker)
            else:
                # Charger le fichier le plus récent
                news_files = list(CONSTANTS.NEWS_DIR.glob("*.parquet"))
                if not news_files:
                    logger.warning("⚠️ Aucun fichier news trouvé")
                    return pd.DataFrame()
                
                file_path = max(news_files, key=lambda x: x.stat().st_mtime)
            
            if not file_path.exists():
                logger.warning(f"⚠️ Fichier news non trouvé: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path)
            logger.info(f"💾 News chargé: {len(df)} lignes")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement news: {e}")
            return pd.DataFrame()
    
    def save_sentiment(self, 
                      data: pd.DataFrame, 
                      ticker: str) -> Path:
        """Sauvegarde les données de sentiment de manière incrémentale"""
        try:
            file_path = CONSTANTS.get_data_path("sentiment", ticker)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarde incrémentale : charger les données existantes et ajouter les nouvelles
            if file_path.exists():
                try:
                    existing_data = pd.read_parquet(file_path)
                    # Concaténer et dédupliquer par timestamp
                    combined_data = pd.concat([existing_data, data], ignore_index=True)
                    combined_data = combined_data.drop_duplicates(subset=['timestamp'], keep='last')
                    combined_data = combined_data.sort_values('timestamp')
                    data_to_save = combined_data
                    logger.info(f"💾 Sentiment {ticker} mis à jour incrémentalement: {len(existing_data)} → {len(combined_data)} lignes")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lecture fichier sentiment existant {ticker}, sauvegarde complète: {e}")
                    data_to_save = data
            else:
                data_to_save = data
                logger.info(f"💾 Nouveau fichier sentiment {ticker} créé: {len(data)} lignes")
            
            # Sauvegarder
            data_to_save.to_parquet(file_path, index=False, compression='snappy')
            
            logger.info(f"💾 Sentiment {ticker} sauvegardé: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde sentiment {ticker}: {e}")
            raise
    
    def load_sentiment(self, ticker: str) -> pd.DataFrame:
        """Charge les données de sentiment"""
        try:
            file_path = CONSTANTS.get_data_path("sentiment", ticker)
            
            if not file_path.exists():
                logger.warning(f"⚠️ Fichier sentiment non trouvé: {file_path}")
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path)
            logger.info(f"💾 Sentiment {ticker} chargé: {len(df)} lignes")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement sentiment {ticker}: {e}")
            return pd.DataFrame()

class DataStorage:
    """Stockage unifié pour toutes les données Sentinel2"""
    
    def __init__(self):
        """Initialise le stockage unifié"""
        self.parquet = ParquetStorage()
        
        logger.info("💾 Stockage unifié initialisé")
    
    def save_crawl_results(self, 
                          results: Dict[str, Any],
                          timestamp: datetime = None) -> Dict[str, Path]:
        """Sauvegarde les résultats d'un crawling complet"""
        if timestamp is None:
            timestamp = datetime.now()
        
        saved_files = {}
        
        try:
            # Sauvegarder les prix
            for ticker, df in results.get("prices", {}).items():
                if not df.empty:
                    file_path = self.parquet.save_prices(df, ticker)
                    saved_files[f"prices_{ticker}"] = file_path
            
            # Sauvegarder les news
            news = results.get("news", [])
            if news:
                news_df = pd.DataFrame(news)
                file_path = self.parquet.save_news(news_df)
                saved_files["news"] = file_path
            
            # Sauvegarder les métadonnées
            metadata = {
                "timestamp": timestamp.isoformat(),
                "total_tickers": len(results.get("prices", {})),
                "total_news": len(results.get("news", [])),
                "crawl_time": results.get("crawl_time", 0),
                "saved_files": {k: str(v) for k, v in saved_files.items()}
            }
            
            metadata_path = CONSTANTS.DATA_ROOT / f"crawl_metadata_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            saved_files["metadata"] = metadata_path
            
            logger.info(f"💾 Crawling sauvegardé: {len(saved_files)} fichiers")
            return saved_files
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde crawling: {e}")
            raise
    
    def load_latest_data(self, 
                        tickers: List[str] = None,
                        include_news: bool = True) -> Dict[str, Any]:
        """Charge les données les plus récentes"""
        tickers = tickers or CONSTANTS.TICKERS
        data = {}
        
        try:
            # Charger les prix
            for ticker in tickers:
                df = self.parquet.load_prices(ticker)
                if not df.empty:
                    data[f"prices_{ticker}"] = df
            
            # Charger les news
            if include_news:
                news_df = self.parquet.load_news()
                if not news_df.empty:
                    data["news"] = news_df
            
            logger.info(f"💾 Données chargées: {len(data)} datasets")
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement données: {e}")
            return {}
    
    def save_model_artifacts(self, 
                           ticker: str,
                           model: Any,
                           scaler: Any,
                           metrics: Dict[str, float],
                           version: int = None) -> Dict[str, Path]:
        """Sauvegarde les artefacts d'un modèle"""
        try:
            # Chemin du modèle
            if version:
                model_dir = CONSTANTS.get_model_path(ticker, version)
            else:
                # Trouver la prochaine version
                base_dir = CONSTANTS.get_model_path(ticker)
                existing_versions = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("version")]
                version = max([int(d.name.replace("version", "")) for d in existing_versions], default=0) + 1
                model_dir = base_dir / f"version{version}"
            
            model_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = {}
            
            # Sauvegarder le modèle
            model_path = model_dir / "lstm_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size': len(CONSTANTS.get_feature_columns()),
                    'hidden_sizes': CONSTANTS.LSTM_HIDDEN_SIZES,
                    'dropout_rate': CONSTANTS.LSTM_DROPOUT_RATE,
                    'task': 'regression'
                }
            }, model_path)
            saved_files["model"] = model_path
            
            # Sauvegarder le scaler
            scaler_path = model_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            saved_files["scaler"] = scaler_path
            
            # Sauvegarder les métriques
            metrics_path = model_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            saved_files["metrics"] = metrics_path
            
            # Sauvegarder l'historique d'entraînement
            history_path = model_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump({}, f)  # À remplir par le processus d'entraînement
            saved_files["history"] = history_path
            
            logger.info(f"💾 Modèle {ticker} v{version} sauvegardé")
            return saved_files
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde modèle {ticker}: {e}")
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des données disponibles"""
        summary = {
            "prices": {},
            "news": {},
            "sentiment": {},
            "models": {}
        }
        
        try:
            # Résumé des prix
            for ticker in CONSTANTS.TICKERS:
                df = self.parquet.load_prices(ticker)
                if not df.empty:
                    summary["prices"][ticker] = {
                        "rows": len(df),
                        "date_range": f"{df['ts_utc'].min()} to {df['ts_utc'].max()}" if 'ts_utc' in df.columns else "N/A"
                    }
            
            # Résumé des news
            news_df = self.parquet.load_news()
            if not news_df.empty:
                summary["news"] = {
                    "rows": len(news_df),
                    "sources": news_df['source'].nunique() if 'source' in news_df.columns else 0
                }
            
            # Résumé des modèles
            for ticker in CONSTANTS.TICKERS:
                model_dir = CONSTANTS.get_model_path(ticker)
                if model_dir.exists():
                    versions = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("version")]
                    summary["models"][ticker] = {
                        "versions": len(versions),
                        "latest": max([int(d.name.replace("version", "")) for d in versions], default=0) if versions else 0
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Erreur résumé données: {e}")
            return summary
