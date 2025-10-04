"""
💾 Stockage Unifié Sentinel2
Système de stockage cohérent pour données financières et articles
"""

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from ..constants import CONSTANTS


class UnifiedDataStorage:
    """Stockage unifié pour toutes les données Sentinel2"""
    
    def __init__(self):
        """Initialise le stockage unifié"""
        self.data_root = CONSTANTS.DATA_ROOT
        self.historical_dir = CONSTANTS.HISTORICAL_DIR
        self.realtime_dir = CONSTANTS.REALTIME_DIR
        self.models_dir = CONSTANTS.MODELS_DIR
        self.logs_dir = CONSTANTS.LOGS_DIR
        self.trading_dir = CONSTANTS.TRADING_DIR
        
        # Créer les répertoires conformes à l'architecture
        for dir_path in [self.historical_dir, self.realtime_dir, self.models_dir, self.logs_dir, self.trading_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("💾 Stockage unifié initialisé")
    
    def save_price_data(self, 
                       ticker: str, 
                       data: pd.DataFrame, 
                       interval: str = "1min",
                       source: str = "yfinance") -> Path:
        """Sauvegarde les données de prix"""
        try:
            # Normaliser les colonnes
            data = self._normalize_price_columns(data)
            
            # Chemin de sauvegarde
            file_path = self.realtime_dir / "prices" / f"{ticker.lower()}_{interval}.parquet"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ajouter métadonnées
            data.attrs = {
                "ticker": ticker.upper(),
                "interval": interval,
                "source": source,
                "saved_at": datetime.now().isoformat(),
                "rows": len(data)
            }
            
            # Sauvegarder
            data.to_parquet(file_path, index=False, compression='snappy')
            
            logger.info(f"💾 Prix {ticker} sauvegardé: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde prix {ticker}: {e}")
            raise
    
    def load_price_data(self, 
                       ticker: str, 
                       interval: str = "1min",
                       source: str = "yfinance",
                       days: int = None) -> pd.DataFrame:
        """Charge les données de prix"""
        try:
            file_path = self.raw_dir / "prices" / f"{ticker.lower()}_{interval}_{source}.parquet"
            
            if not file_path.exists():
                logger.warning(f"⚠️ Fichier prix non trouvé: {file_path}")
                return pd.DataFrame()
            
            # Charger les données
            data = pd.read_parquet(file_path)
            
            # Filtrer par nombre de jours si spécifié
            if days and 'timestamp' in data.columns:
                cutoff = datetime.now() - timedelta(days=days)
                data = data[data['timestamp'] >= cutoff]
            
            logger.info(f"💾 Prix {ticker} chargé: {len(data)} lignes")
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement prix {ticker}: {e}")
            return pd.DataFrame()
    
    def save_news_data(self, 
                      data: pd.DataFrame, 
                      ticker: str = None,
                      source: str = "rss") -> Path:
        """Sauvegarde les données de news"""
        try:
            # Normaliser les colonnes
            data = self._normalize_news_columns(data)
            
            # Chemin de sauvegarde
            if ticker:
                file_path = self.raw_dir / "news" / f"{ticker.lower()}_{source}_news.parquet"
            else:
                file_path = self.raw_dir / "news" / f"all_{source}_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ajouter métadonnées
            data.attrs = {
                "ticker": ticker,
                "source": source,
                "saved_at": datetime.now().isoformat(),
                "rows": len(data)
            }
            
            # Sauvegarder
            data.to_parquet(file_path, index=False, compression='snappy')
            
            logger.info(f"💾 News sauvegardé: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde news: {e}")
            raise
    
    def load_news_data(self, 
                      ticker: str = None,
                      source: str = "rss",
                      days: int = None) -> pd.DataFrame:
        """Charge les données de news"""
        try:
            if ticker:
                file_path = self.raw_dir / "news" / f"{ticker.lower()}_{source}_news.parquet"
            else:
                # Charger le fichier le plus récent
                news_files = list((self.raw_dir / "news").glob(f"*{source}_news*.parquet"))
                if not news_files:
                    logger.warning("⚠️ Aucun fichier news trouvé")
                    return pd.DataFrame()
                file_path = max(news_files, key=lambda x: x.stat().st_mtime)
            
            if not file_path.exists():
                logger.warning(f"⚠️ Fichier news non trouvé: {file_path}")
                return pd.DataFrame()
            
            # Charger les données
            data = pd.read_parquet(file_path)
            
            # Filtrer par nombre de jours si spécifié
            if days and 'published_at' in data.columns:
                cutoff = datetime.now() - timedelta(days=days)
                data = data[data['published_at'] >= cutoff]
            
            logger.info(f"💾 News chargé: {len(data)} lignes")
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement news: {e}")
            return pd.DataFrame()
    
    def save_features_data(self, 
                          ticker: str, 
                          data: pd.DataFrame,
                          feature_type: str = "technical") -> Path:
        """Sauvegarde les données de features pour LSTM"""
        try:
            # Vérifier que les features requises sont présentes
            required_features = CONSTANTS.get_feature_columns()
            missing_features = [f for f in required_features if f not in data.columns]
            
            if missing_features:
                logger.warning(f"⚠️ Features manquantes: {missing_features}")
            
            # Chemin de sauvegarde
            file_path = self.features_dir / f"{ticker.lower()}_{feature_type}_features.parquet"
            
            # Ajouter métadonnées
            data.attrs = {
                "ticker": ticker.upper(),
                "feature_type": feature_type,
                "saved_at": datetime.now().isoformat(),
                "rows": len(data),
                "features": list(data.columns)
            }
            
            # Sauvegarder
            data.to_parquet(file_path, index=False, compression='snappy')
            
            logger.info(f"💾 Features {ticker} sauvegardé: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde features {ticker}: {e}")
            raise
    
    def load_features_data(self, 
                          ticker: str,
                          feature_type: str = "technical") -> pd.DataFrame:
        """Charge les données de features pour LSTM"""
        try:
            file_path = self.features_dir / f"{ticker.lower()}_{feature_type}_features.parquet"
            
            if not file_path.exists():
                logger.warning(f"⚠️ Fichier features non trouvé: {file_path}")
                return pd.DataFrame()
            
            # Charger les données
            data = pd.read_parquet(file_path)
            
            logger.info(f"💾 Features {ticker} chargé: {len(data)} lignes")
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement features {ticker}: {e}")
            return pd.DataFrame()
    
    def save_sentiment_data(self, 
                           ticker: str, 
                           data: pd.DataFrame) -> Path:
        """Sauvegarde les données de sentiment"""
        try:
            # Normaliser les colonnes
            data = self._normalize_sentiment_columns(data)
            
            # Chemin de sauvegarde
            file_path = self.raw_dir / "sentiment" / f"{ticker.lower()}_sentiment.parquet"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ajouter métadonnées
            data.attrs = {
                "ticker": ticker.upper(),
                "saved_at": datetime.now().isoformat(),
                "rows": len(data)
            }
            
            # Sauvegarder
            data.to_parquet(file_path, index=False, compression='snappy')
            
            logger.info(f"💾 Sentiment {ticker} sauvegardé: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde sentiment {ticker}: {e}")
            raise
    
    def load_sentiment_data(self, ticker: str) -> pd.DataFrame:
        """Charge les données de sentiment"""
        try:
            file_path = self.raw_dir / "sentiment" / f"{ticker.lower()}_sentiment.parquet"
            
            if not file_path.exists():
                logger.warning(f"⚠️ Fichier sentiment non trouvé: {file_path}")
                return pd.DataFrame()
            
            # Charger les données
            data = pd.read_parquet(file_path)
            
            logger.info(f"💾 Sentiment {ticker} chargé: {len(data)} lignes")
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement sentiment {ticker}: {e}")
            return pd.DataFrame()
    
    def save_model_artifacts(self, 
                           ticker: str,
                           model: Any,
                           scaler: Any,
                           metrics: Dict[str, float],
                           version: int = None) -> Dict[str, Path]:
        """Sauvegarde les artefacts d'un modèle LSTM"""
        try:
            # Chemin du modèle
            if version:
                model_dir = self.models_dir / ticker.lower() / f"version{version}"
            else:
                # Trouver la prochaine version
                base_dir = self.models_dir / ticker.lower()
                existing_versions = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("version")]
                version = max([int(d.name.replace("version", "")) for d in existing_versions], default=0) + 1
                model_dir = base_dir / f"version{version}"
            
            model_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = {}
            
            # Sauvegarder le modèle
            model_path = model_dir / "lstm_model.pth"
            import torch
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
    
    def _normalize_price_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalise les colonnes de prix"""
        # Mapping des colonnes communes
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
            'Date': 'timestamp',
            'Datetime': 'timestamp'
        }
        
        # Renommer les colonnes
        data = data.rename(columns=column_mapping)
        
        # S'assurer que timestamp est datetime
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        return data
    
    def _normalize_news_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalise les colonnes de news"""
        # Mapping des colonnes communes
        column_mapping = {
            'title': 'title',
            'summary': 'summary',
            'body': 'body',
            'content': 'body',
            'link': 'url',
            'published': 'published_at',
            'publishedAt': 'published_at',
            'source': 'source'
        }
        
        # Renommer les colonnes
        data = data.rename(columns=column_mapping)
        
        # S'assurer que published_at est datetime
        if 'published_at' in data.columns:
            data['published_at'] = pd.to_datetime(data['published_at'])
        
        return data
    
    def _normalize_sentiment_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalise les colonnes de sentiment"""
        # Mapping des colonnes communes
        column_mapping = {
            'sentiment_score': 'sentiment_score',
            'sentiment': 'sentiment_score',
            'confidence': 'confidence',
            'text': 'text',
            'ticker': 'ticker'
        }
        
        # Renommer les colonnes
        data = data.rename(columns=column_mapping)
        
        return data
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des données disponibles"""
        summary = {
            "prices": {},
            "news": {},
            "sentiment": {},
            "features": {},
            "models": {}
        }
        
        try:
            # Résumé des prix
            for ticker in CONSTANTS.TICKERS:
                price_files = list((self.raw_dir / "prices").glob(f"{ticker.lower()}_*.parquet"))
                if price_files:
                    latest_file = max(price_files, key=lambda x: x.stat().st_mtime)
                    data = pd.read_parquet(latest_file)
                    summary["prices"][ticker] = {
                        "files": len(price_files),
                        "latest_rows": len(data),
                        "latest_file": latest_file.name
                    }
            
            # Résumé des news
            news_files = list((self.raw_dir / "news").glob("*.parquet"))
            if news_files:
                latest_file = max(news_files, key=lambda x: x.stat().st_mtime)
                data = pd.read_parquet(latest_file)
                summary["news"] = {
                    "files": len(news_files),
                    "latest_rows": len(data),
                    "latest_file": latest_file.name
                }
            
            # Résumé des features
            feature_files = list(self.features_dir.glob("*.parquet"))
            for file_path in feature_files:
                ticker = file_path.stem.split('_')[0].upper()
                data = pd.read_parquet(file_path)
                summary["features"][ticker] = {
                    "rows": len(data),
                    "features": len(data.columns),
                    "file": file_path.name
                }
            
            # Résumé des modèles
            for ticker in CONSTANTS.TICKERS:
                model_dir = self.models_dir / ticker.lower()
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
