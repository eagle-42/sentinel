"""
🔮 Prédictions LSTM pour Sentinel2
Modèle LSTM optimisé pour la prédiction des prix financiers
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from sklearn.preprocessing import RobustScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import CONSTANTS

class FinancialLSTM(nn.Module):
    """Modèle LSTM optimisé pour la prédiction financière"""
    
    def __init__(self, input_size: int = 15, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super(FinancialLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Architecture LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Prendre la dernière sortie
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class PricePredictor:
    """Prédicteur de prix utilisant un modèle LSTM"""
    
    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker.upper()
        self.model: Optional[nn.Module] = None
        self.scaler: Optional[RobustScaler] = None
        self.is_loaded = False
        self.sequence_length = CONSTANTS.LSTM_SEQUENCE_LENGTH
        self.feature_columns = CONSTANTS.get_feature_columns()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔮 Prédicteur initialisé pour {self.ticker}")

    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """Charge le modèle LSTM"""
        try:
            if model_path is None:
                # Chercher la dernière version
                models_dir = CONSTANTS.get_model_path(self.ticker)
                if not models_dir.exists():
                    logger.error(f"❌ Répertoire modèle non trouvé: {models_dir}")
                    return False
                
                # Trouver la dernière version
                versions = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("version")]
                if not versions:
                    logger.error(f"❌ Aucune version de modèle trouvée pour {self.ticker}")
                    return False
                
                latest_version = max(versions, key=lambda x: int(x.name.replace("version_", "")))
                model_path = latest_version / "model.pkl"
            
            if not model_path.exists():
                logger.error(f"❌ Modèle non trouvé: {model_path}")
                return False
            
            # Créer la classe FinancialLSTM dans __main__ pour la compatibilité
            sys.modules['__main__'].FinancialLSTM = FinancialLSTM
            # Alias pour l'ancien nom
            sys.modules['__main__'].SimpleLSTM = FinancialLSTM
            
            # Charger le modèle
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.to(self.device)
            self.model.eval()
            
            # Charger le scaler
            scaler_path = model_path.parent / "scaler.pkl"
            if scaler_path.exists():
                import pickle
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            self.is_loaded = True
            logger.info(f"✅ Modèle LSTM {self.ticker} chargé depuis {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            return False

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prépare les features pour le modèle LSTM"""
        try:
            # Chercher les colonnes en tenant compte de la casse
            available_cols = []
            for expected_col in self.feature_columns:
                found = False
                for data_col in data.columns:
                    if expected_col.upper() == data_col.upper():
                        available_cols.append(data_col)
                        found = True
                        break
                if not found:
                    logger.warning(f"⚠️ Colonne manquante: {expected_col}")
            
            if not available_cols:
                logger.error("❌ Aucune colonne de feature disponible")
                return None
            
            if len(available_cols) != len(self.feature_columns):
                logger.warning(f"⚠️ Seulement {len(available_cols)}/{len(self.feature_columns)} features trouvées")
            
            feature_data = data[available_cols]
            
            # Normaliser les features
            if hasattr(self.scaler, 'scale_'):
                features_scaled = self.scaler.transform(feature_data)
            else:
                features_scaled = self.scaler.fit_transform(feature_data)
            
            # logger.debug(f"🔮 Features préparées: {features_scaled.shape}")
            return features_scaled
            
        except Exception as e:
            logger.error(f"❌ Erreur préparation features: {e}")
            return None

    def create_sequences(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crée les séquences temporelles pour le LSTM"""
        if len(features) < self.sequence_length:
            logger.warning(f"⚠️ Pas assez de données: {len(features)} < {self.sequence_length}")
            return None, None
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(features[i])  # Prédire la prochaine valeur
        
        return np.array(X), np.array(y)
    
    def predict(self, data: pd.DataFrame, horizon: int = 1) -> Dict[str, Any]:
        """Fait des prédictions avec le modèle LSTM"""
        if not self.is_loaded:
            logger.error("❌ Modèle non chargé")
            return {"error": "Modèle non chargé"}
        
        try:
            # Préparer les features
            features = self.prepare_features(data)
            if features is None:
                return {"error": "Impossible de préparer les features"}
            
            # Créer les séquences
            X, y = self.create_sequences(features)
            if X is None:
                return {"error": "Pas assez de données pour créer les séquences"}
            
            # Cloner les données numpy pour éviter les erreurs de mémoire
            X = X.copy()
            y = y.copy()
            X = np.ascontiguousarray(X)
            y = np.ascontiguousarray(y)
            
            # Sauvegarder une copie de X pour les prédictions futures
            X_backup = X.copy()
            # Prédictions historiques
            historical_predictions = []
            with torch.no_grad():
                for i in range(len(X)):
                    sequence = torch.FloatTensor(X[i:i+1]).to(self.device)
                    pred = self.model(sequence)
                    pred_value = pred.detach().cpu().numpy()[0, 0]
                    historical_predictions.append(pred_value)
            
            # Prédictions futures
            future_predictions = []
            future_dates = []
            
            if horizon > 0:
                # Utiliser la dernière séquence pour prédire l'avenir
                last_sequence = torch.FloatTensor(X_backup[-1:]).to(self.device)
                
                for i in range(horizon):
                    with torch.no_grad():
                        # Vérifier que la séquence n'est pas vide
                        if last_sequence.numel() == 0:
                            logger.error(f"❌ Séquence vide détectée avant prédiction {i+1}")
                            break
                        
                        pred = self.model(last_sequence)
                        pred_value = pred.detach().cpu().numpy()[0, 0]
                        future_predictions.append(pred_value)
                    
                    # Mettre à jour la séquence pour la prochaine prédiction
                    new_sequence_data = last_sequence.cpu().numpy().copy()
                    
                    if new_sequence_data.ndim == 3 and new_sequence_data.shape[1] > 1:
                        # Décaler la séquence d'un pas vers la gauche
                        new_sequence_data[0, :-1, :] = new_sequence_data[0, 1:, :]
                        # Répéter la dernière ligne pour la nouvelle position
                        new_sequence_data[0, -1, :] = new_sequence_data[0, -2, :]
                    else:
                        # Si la forme est incorrecte, recréer la séquence
                        logger.warning(f"⚠️ Forme de séquence incorrecte: {new_sequence_data.shape}")
                        new_sequence_data = X_backup[-1:].copy()
                    
                    # Vérifier que la séquence n'est pas vide avant de créer le tensor
                    if new_sequence_data.size > 0:
                        last_sequence = torch.FloatTensor(new_sequence_data).to(self.device)
                    else:
                        logger.error(f"❌ Séquence vide détectée: {new_sequence_data.shape}")
                        break
                    
                    # Date future
                    if 'DATE' in data.columns:
                        last_date = pd.to_datetime(data['DATE'].iloc[-1])
                        future_date = last_date + pd.Timedelta(days=i+1)
                        future_dates.append(future_date.strftime('%Y-%m-%d'))
                    else:
                        future_dates.append(f"Day +{i+1}")
            
            return {
                "historical_predictions": historical_predictions,
                "predictions": future_predictions,
                "prediction_dates": future_dates,
                "ticker": self.ticker,
                "horizon": horizon,
                "sequence_length": self.sequence_length
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction: {e}")
            return {"error": str(e)}
    
    def predict_with_technical_features(self, features_df: pd.DataFrame, horizon: int = 1) -> Dict[str, Any]:
        """Fait des prédictions avec des features techniques déjà préparées"""
        if not self.is_loaded:
            logger.error("❌ Modèle non chargé")
            return {"error": "Modèle non chargé"}
        
        try:
            # Les features sont déjà préparées, on peut les utiliser directement
            # Garder seulement les colonnes de features techniques
            feature_cols = [col for col in self.feature_columns if col in features_df.columns]
            
            if not feature_cols:
                logger.error("❌ Aucune colonne de feature technique trouvée")
                return {"error": "Aucune colonne de feature technique trouvée"}
            
            # Extraire les features
            features_data = features_df[feature_cols].values
            
            # Normaliser les features avec le scaler
            if hasattr(self.scaler, 'scale_'):
                features_scaled = self.scaler.transform(features_data)
            else:
                features_scaled = self.scaler.fit_transform(features_data)
            
            # Créer les séquences
            X, y = self.create_sequences(features_scaled)
            if X is None:
                return {"error": "Pas assez de données pour créer les séquences"}
            
            # Cloner les données numpy pour éviter les erreurs de mémoire
            X = X.copy()
            y = y.copy()
            X = np.ascontiguousarray(X)
            y = np.ascontiguousarray(y)
            
            # Sauvegarder une copie de X pour les prédictions futures
            X_backup = X.copy()
            
            # Prédictions historiques
            historical_predictions = []
            with torch.no_grad():
                for i in range(len(X)):
                    sequence = torch.FloatTensor(X[i:i+1]).to(self.device)
                    pred = self.model(sequence)
                    pred_value = pred.detach().cpu().numpy()[0, 0]
                    historical_predictions.append(pred_value)
            
            # Prédictions futures
            future_predictions = []
            future_dates = []
            
            if horizon > 0:
                # Utiliser la dernière séquence pour prédire l'avenir
                last_sequence = torch.FloatTensor(X_backup[-1:]).to(self.device)
                
                for i in range(horizon):
                    pred = self.model(last_sequence)
                    pred_value = pred.detach().cpu().numpy()[0, 0]
                    future_predictions.append(pred_value)
                    
                    # Mettre à jour la séquence pour la prochaine prédiction
                    if i < horizon - 1:
                        # Décaler la séquence et ajouter la prédiction
                        last_sequence = torch.cat([
                            last_sequence[:, 1:, :],
                            pred.unsqueeze(1)
                        ], dim=1)
                
                # Générer les dates futures
                if hasattr(features_df.index, 'to_pydatetime'):
                    last_date = features_df.index[-1]
                    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(horizon)]
                else:
                    future_dates = [f"Day_{i+1}" for i in range(horizon)]
            
            logger.info(f"✅ Prédiction LSTM avec features: {len(historical_predictions)} historiques + {len(future_predictions)} futures")
            
            return {
                "historical_predictions": historical_predictions,
                "predictions": future_predictions,
                "prediction_dates": future_dates,
                "features_used": feature_cols,
                "model_loaded": True
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction avec features techniques: {e}")
            return {"error": str(e)}

# Alias pour la compatibilité
LSTMPredictor = PricePredictor