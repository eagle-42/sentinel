"""
🔮 Prédictions LSTM Optimisées
Modèle LSTM optimisé basé sur les insights des analyses
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ..constants import CONSTANTS

class LSTMModel(nn.Module):
    """Modèle LSTM PyTorch optimisé pour Sentinel2"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = None,
                 num_layers: int = 2,
                 dropout_rate: float = None,
                 task: str = 'regression'):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes or CONSTANTS.LSTM_HIDDEN_SIZES
        self.num_layers = num_layers
        self.task = task
        self.dropout_rate = dropout_rate or CONSTANTS.LSTM_DROPOUT_RATE
        
        # Couches LSTM
        self.lstm_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Première couche LSTM
        self.lstm_layers.append(
            nn.LSTM(input_size, self.hidden_sizes[0], 
                   num_layers=1, batch_first=True, dropout=self.dropout_rate)
        )
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_sizes[0]))
        self.dropouts.append(nn.Dropout(self.dropout_rate))
        
        # Couches LSTM supplémentaires
        for i in range(1, len(self.hidden_sizes)):
            self.lstm_layers.append(
                nn.LSTM(self.hidden_sizes[i-1], self.hidden_sizes[i], 
                       num_layers=1, batch_first=True, dropout=self.dropout_rate)
            )
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_sizes[i]))
            self.dropouts.append(nn.Dropout(self.dropout_rate))
        
        # Couches denses
        self.fc1 = nn.Linear(self.hidden_sizes[-1], 32)
        self.fc1_bn = nn.BatchNorm1d(32)
        self.fc1_dropout = nn.Dropout(self.dropout_rate)
        
        # Couche de sortie
        if task == 'regression':
            self.fc2 = nn.Linear(32, 1)
        else:  # classification
            self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        """Forward pass du modèle LSTM"""
        batch_size = x.size(0)
        
        # LSTM layers
        for i, (lstm, bn, dropout) in enumerate(zip(self.lstm_layers, self.batch_norms, self.dropouts)):
            if i == 0:
                lstm_out, _ = lstm(x)
            else:
                lstm_out, _ = lstm(lstm_out)
            
            # Batch normalization sur la dernière dimension temporelle
            lstm_out = lstm_out.contiguous().view(-1, lstm_out.size(-1))
            lstm_out = bn(lstm_out)
            lstm_out = lstm_out.view(batch_size, -1, lstm_out.size(-1))
            
            # Dropout
            lstm_out = dropout(lstm_out)
        
        # Prendre la dernière sortie de la séquence
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Couches denses
        x = torch.relu(self.fc1(last_output))
        x = self.fc1_bn(x)
        x = self.fc1_dropout(x)
        
        # Sortie finale
        output = self.fc2(x)
        
        if self.task == 'classification':
            output = torch.sigmoid(output)
        
        return output

class LSTMPredictor:
    """Prédicteur LSTM optimisé pour Sentinel2"""
    
    def __init__(self, ticker: str = "SPY"):
        """Initialise le prédicteur LSTM"""
        self.ticker = ticker.upper()
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = CONSTANTS.get_feature_columns()
        self.sequence_length = CONSTANTS.LSTM_SEQUENCE_LENGTH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        
        logger.info(f"🔮 Prédicteur LSTM initialisé pour {self.ticker}")
    
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
                
                latest_version = max(versions, key=lambda x: int(x.name.replace("version", "")))
                model_path = latest_version / "lstm_model.pth"
            
            if not model_path.exists():
                logger.error(f"❌ Modèle non trouvé: {model_path}")
                return False
            
            # Charger le modèle
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Reconstruire le modèle
            input_size = len(self.feature_columns)
            self.model = LSTMModel(
                input_size=input_size,
                hidden_sizes=CONSTANTS.LSTM_HIDDEN_SIZES,
                dropout_rate=CONSTANTS.LSTM_DROPOUT_RATE,
                task='regression'
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
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
            # Vérifier que les colonnes nécessaires existent
            missing_cols = [col for col in self.feature_columns if col not in data.columns]
            if missing_cols:
                logger.warning(f"⚠️ Colonnes manquantes: {missing_cols}")
                # Utiliser les colonnes disponibles
                available_cols = [col for col in self.feature_columns if col in data.columns]
                if not available_cols:
                    logger.error("❌ Aucune colonne de feature disponible")
                    return None
                feature_data = data[available_cols]
            else:
                feature_data = data[self.feature_columns]
            
            # Normaliser les features
            if hasattr(self.scaler, 'scale_'):
                # Scaler déjà entraîné
                features_scaled = self.scaler.transform(feature_data)
            else:
                # Premier passage, ajuster le scaler
                features_scaled = self.scaler.fit_transform(feature_data)
            
            logger.debug(f"🔮 Features préparées: {features_scaled.shape}")
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
            
            # Prédictions historiques
            historical_predictions = []
            with torch.no_grad():
                for i in range(len(X)):
                    # Prendre la séquence
                    sequence = torch.FloatTensor(X[i:i+1]).to(self.device)
                    
                    # Prédiction
                    pred = self.model(sequence)
                    pred_value = pred.cpu().numpy()[0, 0]
                    
                    # Dénormaliser (approximation)
                    historical_predictions.append(pred_value)
            
            # Prédictions futures
            future_predictions = []
            future_dates = []
            
            if horizon > 0:
                # Utiliser la dernière séquence pour prédire l'avenir
                last_sequence = torch.FloatTensor(X[-1:]).to(self.device)
                
                for i in range(horizon):
                    with torch.no_grad():
                        pred = self.model(last_sequence)
                        pred_value = pred.cpu().numpy()[0, 0]
                        future_predictions.append(pred_value)
                    
                    # Mettre à jour la séquence pour la prochaine prédiction
                    # (approximation simple)
                    new_sequence = last_sequence.clone()
                    new_sequence[0, :-1] = new_sequence[0, 1:]
                    new_sequence[0, -1] = pred_value
                    last_sequence = new_sequence
                    
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
    
    def evaluate_performance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Évalue la performance du modèle"""
        if not self.is_loaded:
            return {"error": "Modèle non chargé"}
        
        try:
            # Préparer les features
            features = self.prepare_features(data)
            if features is None:
                return {"error": "Impossible de préparer les features"}
            
            # Créer les séquences
            X, y = self.create_sequences(features)
            if X is None:
                return {"error": "Pas assez de données"}
            
            # Prédictions
            predictions = self.predict(data)
            if "error" in predictions:
                return predictions
            
            hist_preds = predictions["historical_predictions"]
            
            # Métriques de performance
            mse = mean_squared_error(y, hist_preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, hist_preds)
            
            # Corrélation
            correlation = np.corrcoef(y, hist_preds)[0, 1] if len(y) > 1 else 0.0
            
            # Précision de direction (pour les prix)
            if len(y) > 1:
                y_direction = np.diff(y) > 0
                pred_direction = np.diff(hist_preds) > 0
                direction_accuracy = np.mean(y_direction == pred_direction)
            else:
                direction_accuracy = 0.0
            
            return {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "correlation": correlation,
                "direction_accuracy": direction_accuracy,
                "total_predictions": len(hist_preds)
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur évaluation: {e}")
            return {"error": str(e)}

class PredictionEngine:
    """Moteur de prédiction unifié pour Sentinel2"""
    
    def __init__(self):
        """Initialise le moteur de prédiction"""
        self.predictors = {}  # {ticker: LSTMPredictor}
        self.is_initialized = False
        
        logger.info("🔮 Moteur de prédiction initialisé")
    
    def initialize_predictor(self, ticker: str) -> bool:
        """Initialise un prédicteur pour un ticker"""
        try:
            predictor = LSTMPredictor(ticker)
            success = predictor.load_model()
            
            if success:
                self.predictors[ticker] = predictor
                logger.info(f"✅ Prédicteur {ticker} initialisé")
                return True
            else:
                logger.error(f"❌ Échec initialisation prédicteur {ticker}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur initialisation {ticker}: {e}")
            return False
    
    def predict(self, ticker: str, data: pd.DataFrame, horizon: int = 1) -> Dict[str, Any]:
        """Fait des prédictions pour un ticker"""
        if ticker not in self.predictors:
            if not self.initialize_predictor(ticker):
                return {"error": f"Impossible d'initialiser le prédicteur {ticker}"}
        
        predictor = self.predictors[ticker]
        return predictor.predict(data, horizon)
    
    def evaluate(self, ticker: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Évalue la performance d'un prédicteur"""
        if ticker not in self.predictors:
            if not self.initialize_predictor(ticker):
                return {"error": f"Impossible d'initialiser le prédicteur {ticker}"}
        
        predictor = self.predictors[ticker]
        return predictor.evaluate_performance(data)
    
    def get_available_tickers(self) -> List[str]:
        """Retourne les tickers disponibles"""
        return list(self.predictors.keys())
    
    def is_ticker_ready(self, ticker: str) -> bool:
        """Vérifie si un ticker est prêt pour les prédictions"""
        return ticker in self.predictors and self.predictors[ticker].is_loaded
