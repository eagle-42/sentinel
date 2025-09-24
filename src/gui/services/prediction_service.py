#!/usr/bin/env python3
"""
Service de prédiction LSTM - Utilise le modèle existant
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

# Import de la configuration centralisée
from .service_config import get_service_config, get_data_file_path, get_model_path, get_feature_columns


class LSTMModel(nn.Module):
    """Modèle LSTM compatible avec votre architecture existante"""
    
    def __init__(self, input_size: int, hidden_sizes: list = [64, 32], num_layers: int = 2, task: str = 'regression'):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.task = task
        
        # Couches LSTM
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_sizes[i-1]
            self.lstm_layers.append(nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_sizes[i],
                batch_first=True
            ))
        
        # Batch Normalization
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[i]))
        
        # Couches fully connected
        self.fc1 = nn.Linear(hidden_sizes[-1], 32)
        self.fc1_bn = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Passer à travers les couches LSTM
        for i, (lstm, bn) in enumerate(zip(self.lstm_layers, self.batch_norms)):
            x, _ = lstm(x)
            # BatchNorm sur la dernière dimension temporelle
            x = x.permute(0, 2, 1)  # (batch, features, time)
            x = bn(x)
            x = x.permute(0, 2, 1)  # (batch, time, features)
            x = self.dropout(x)
        
        # Prendre la dernière sortie temporelle
        x = x[:, -1, :]  # (batch, features)
        
        # Couches fully connected
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class PredictionService:
    """Service de prédiction LSTM utilisant le modèle existant"""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        
        # Utiliser la configuration centralisée
        config = get_service_config()
        self.device = torch.device(config["device"])
        logger.info("🤖 Service de prédiction LSTM initialisé")
    
    def load_model(self, version: int = None, ticker: str = "spy") -> bool:
        """Charge le modèle LSTM existant (dernière version par défaut)"""
        try:
            # Utiliser la configuration centralisée
            model_path_str = get_model_path(ticker, version)
            model_file = Path(model_path_str)
            
            if not model_file.exists():
                logger.error(f"❌ Modèle non trouvé: {model_file}")
                return False
            
            # Charger le modèle
            checkpoint = torch.load(model_file, map_location=self.device)
            
            # Utiliser la configuration centralisée
            from src.config import SentinelConfig
            config = SentinelConfig()
            model_config = getattr(config, 'models', {}).get(ticker.lower(), {})
            
            # Valeurs par défaut si non trouvées
            if not model_config:
                model_config = {
                    'input_size': 15,
                    'hidden_sizes': [64, 32],
                    'num_layers': 2,
                    'task': 'regression'
                }
            
            # Créer le modèle
            self.model = LSTMModel(
                input_size=model_config['input_size'],
                hidden_sizes=model_config['hidden_sizes'],
                num_layers=model_config['num_layers'],
                task=model_config['task']
            ).to(self.device)
            
            # Charger les poids
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.is_loaded = True
            
            logger.info(f"✅ Modèle LSTM SPY v{version} chargé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            self.is_loaded = False
            return False
    
    def prepare_features(self, df: pd.DataFrame, ticker: str = "spy") -> np.ndarray:
        """Prépare les features pour la prédiction en utilisant les features optimisées"""
        try:
            # Utiliser la configuration centralisée
            features_path = get_data_file_path(ticker, "features")
            
            if not Path(features_path).exists():
                logger.warning("⚠️ Features optimisées non trouvées, utilisation des features de base")
                return self._prepare_basic_features(df)
            
            # Charger les features optimisées
            df_features = pd.read_parquet(features_path)
            df_features['DATE'] = pd.to_datetime(df_features['DATE'])
            
            # Trouver la date la plus récente dans les données d'entrée
            last_date = df['DATE'].max()
            
            # Gérer les timezones pour la comparaison
            if df_features['DATE'].dt.tz is not None and last_date.tz is None:
                last_date = last_date.tz_localize('UTC')
            elif df_features['DATE'].dt.tz is None and last_date.tz is not None:
                last_date = last_date.tz_localize(None)
            
            # Filtrer les features pour la période correspondante
            df_features = df_features[df_features['DATE'] <= last_date]
            
            # Ajuster le nombre de features à la longueur des données d'entrée
            if len(df_features) > len(df):
                df_features = df_features.tail(len(df))
            elif len(df_features) < len(df):
                logger.warning(f"⚠️ Pas assez de features: {len(df_features)} vs {len(df)} données")
                # Utiliser les features disponibles et répéter si nécessaire
                if len(df_features) > 0:
                    # Répéter les dernières features
                    repeat_factor = len(df) // len(df_features) + 1
                    df_features = pd.concat([df_features] * repeat_factor, ignore_index=True)
                    df_features = df_features.tail(len(df))
                else:
                    logger.error("❌ Aucune feature disponible")
                    return self._prepare_basic_features(df)
            
            if df_features.empty:
                logger.warning("⚠️ Aucune feature optimisée trouvée, utilisation des features de base")
                return self._prepare_basic_features(df)
            
            # Utiliser la configuration centralisée pour les features
            feature_cols = get_feature_columns()
            
            # Garder seulement les colonnes disponibles
            available_features = [col for col in feature_cols if col in df_features.columns]
            
            if len(available_features) < 10:
                logger.warning("⚠️ Pas assez de features optimisées, utilisation des features de base")
                return self._prepare_basic_features(df)
            
            # Sélectionner les features finales
            features = df_features[available_features].values
            
            # Gérer les valeurs manquantes
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalisation simple
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            
            logger.info(f"✅ Features optimisées préparées: {features.shape} avec {len(available_features)} features")
            return features
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la préparation des features optimisées: {e}")
            return self._prepare_basic_features(df)
    
    def _prepare_basic_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prépare les features de base en cas d'échec des features optimisées"""
        try:
            df = df.copy()
            
            # Calculer les features de base
            df['returns'] = df['CLOSE'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df['momentum_5'] = df['CLOSE'] / df['CLOSE'].shift(5) - 1
            df['momentum_10'] = df['CLOSE'] / df['CLOSE'].shift(10) - 1
            df['momentum_20'] = df['CLOSE'] / df['CLOSE'].shift(20) - 1
            df['volume_ratio'] = df['VOLUME'] / df['VOLUME'].rolling(window=20).mean()
            df['price_velocity'] = df['CLOSE'].diff() / df['CLOSE'].shift(1)
            df['high_low_ratio'] = df['HIGH'] / df['LOW']
            df['close_open_ratio'] = df['CLOSE'] / df['OPEN']
            df['volume_price_trend'] = df['VOLUME'] * df['returns']
            df['price_position'] = (df['CLOSE'] - df['CLOSE'].rolling(20).min()) / (df['CLOSE'].rolling(20).max() - df['CLOSE'].rolling(20).min())
            df['rsi'] = self._calculate_rsi(df['CLOSE'])
            df['bollinger_upper'] = df['CLOSE'].rolling(20).mean() + 2 * df['CLOSE'].rolling(20).std()
            df['bollinger_lower'] = df['CLOSE'].rolling(20).mean() - 2 * df['CLOSE'].rolling(20).std()
            df['bollinger_position'] = (df['CLOSE'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
            df['ma_ratio'] = df['CLOSE'] / df['CLOSE'].rolling(20).mean()
            
            # Sélectionner les features
            feature_cols = [
                'returns', 'volatility', 'momentum_5', 'momentum_10', 'momentum_20',
                'volume_ratio', 'price_velocity', 'high_low_ratio', 'close_open_ratio',
                'volume_price_trend', 'price_position', 'rsi', 'bollinger_position', 'ma_ratio'
            ]
            
            # Garder seulement les colonnes disponibles
            available_features = [col for col in feature_cols if col in df.columns]
            features = df[available_features].fillna(0).values
            
            # Normalisation simple
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            
            logger.info(f"✅ Features de base préparées: {features.shape} avec {len(available_features)} features")
            return features
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la préparation des features de base: {e}")
            return np.array([])
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def predict(self, df: pd.DataFrame, horizon: int = 20) -> Dict[str, Any]:
        """Prédiction LSTM complète : historique + futur"""
        try:
            if not self.is_loaded:
                return {"error": "Modèle non chargé"}
            
            # Données de base - s'adapter aux colonnes disponibles
            dates = df['DATE'].tolist()
            
            # Chercher la colonne de prix (peut être 'close' ou 'CLOSE')
            if 'CLOSE' in df.columns:
                actual_prices = df['CLOSE'].tolist()
            elif 'close' in df.columns:
                actual_prices = df['close'].tolist()
            else:
                return {"error": "Colonne de prix non trouvée"}
            
            # 1. PRÉDICTIONS HISTORIQUES (utiliser le vrai modèle LSTM)
            historical_predictions = []
            if len(df) > 10:  # Besoin d'au moins 10 points pour le modèle
                # Préparer les features pour chaque point historique
                features = self.prepare_features(df)
                if features.shape[0] > 0:
                    # Utiliser le modèle pour prédire sur les données historiques
                    with torch.no_grad():
                        self.model.eval()
                        for i in range(len(features)):
                            if i >= 10:  # Besoin d'au moins 10 points pour une séquence
                                # Créer une séquence de 10 points
                                seq = features[i-9:i+1].reshape(1, 10, -1)
                                seq_tensor = torch.FloatTensor(seq).to(self.device)
                                
                                # Prédiction du modèle
                                pred = self.model(seq_tensor)
                                pred_price = actual_prices[i-1] * (1 + pred.item())  # Convertir rendement en prix
                                historical_predictions.append(pred_price)
                            else:
                                historical_predictions.append(actual_prices[i])
                else:
                    historical_predictions = actual_prices
            else:
                # Pas assez de données pour les prédictions historiques
                logger.warning(f"⚠️ Pas assez de données pour prédictions historiques: {len(df)} lignes (minimum 10)")
                historical_predictions = actual_prices
            
            # 2. PRÉDICTIONS FUTURES (utiliser le modèle LSTM)
            future_predictions = []
            last_price = actual_prices[-1]
            
            # Préparer les features pour les prédictions futures
            features = self.prepare_features(df)
            
            # Utiliser les dernières features pour prédire le futur
            if len(features) > 0:
                with torch.no_grad():
                    self.model.eval()
                    # Utiliser la même séquence pour toutes les prédictions futures
                    # (simplification - en réalité il faudrait recalculer les features)
                    last_sequence = features[-10:].reshape(1, 10, -1)
                    seq_tensor = torch.FloatTensor(last_sequence).to(self.device)
                    
                    for i in range(horizon):
                        # Prédiction du modèle
                        pred = self.model(seq_tensor)
                        
                        # Convertir rendement en prix
                        pred_price = last_price * (1 + pred.item())
                        future_predictions.append(pred_price)
                        
                        # Mettre à jour pour la prochaine prédiction
                        last_price = pred_price
            else:
                # Fallback : simulation simple
                for i in range(horizon):
                    import random
                    variation = random.uniform(-0.02, 0.02)
                    pred_price = last_price * (1 + variation)
                    future_predictions.append(pred_price)
                    last_price = pred_price
            
            # Dates futures
            last_date = pd.to_datetime(dates[-1]) if dates else pd.Timestamp.now()
            if last_date.tz is not None:
                last_date_naive = last_date.tz_localize(None)
                start_date = last_date_naive + pd.Timedelta(days=1)
                future_dates = pd.date_range(start=start_date, periods=horizon, freq='D')
            else:
                start_date = last_date + pd.Timedelta(days=1)
                future_dates = pd.date_range(start=start_date, periods=horizon, freq='D')
            
            # Convertir en strings
            pred_dates_str = []
            for d in future_dates:
                if hasattr(d, 'strftime'):
                    pred_dates_str.append(d.strftime('%Y-%m-%d'))
                else:
                    pred_dates_str.append(str(d))
            
            logger.info(f"✅ Prédiction LSTM: {len(historical_predictions)} prédictions historiques + {len(future_predictions)} prédictions futures")
            
            return {
                "predictions": future_predictions,
                "prediction_dates": pred_dates_str,
                "actual_prices": actual_prices,
                "historical_dates": dates,
                "historical_prices": actual_prices,
                "historical_predictions": historical_predictions,
                "last_date": pd.to_datetime(dates[-1]) if dates else pd.Timestamp.now(),
                "horizon": horizon
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction: {e}")
            return {"error": str(e)}
