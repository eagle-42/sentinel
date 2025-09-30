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
            
            # Charger le modèle (désactiver weights_only pour permettre le chargement du scaler)
            model_data = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Créer le modèle avec la taille d'entrée sauvegardée
            input_size = model_data.get('input_size', len(model_data['feature_columns']))
            self.model = FinancialLSTM(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                output_size=1
            ).to(self.device)
            
            # Charger les poids
            self.model.load_state_dict(model_data['model_state_dict'])
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.sequence_length = model_data['sequence_length']
            self.model.eval()
            
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

    def create_sequences(self, features: np.ndarray, target_col_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Crée les séquences temporelles pour le LSTM"""
        if len(features) < self.sequence_length:
            logger.warning(f"⚠️ Pas assez de données: {len(features)} < {self.sequence_length}")
            return None, None
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            # Utiliser la colonne spécifiée comme target (par défaut la première)
            y.append(features[i][target_col_idx])
        
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
            
            # Ajouter CLOSE si nécessaire (pour correspondre au modèle entraîné)
            if 'CLOSE' in features_df.columns and 'CLOSE' not in feature_cols:
                feature_cols.append('CLOSE')
                logger.info("🎯 Ajout de CLOSE aux features pour la prédiction")
            
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
                    
                    # Pour les prédictions suivantes, on utilise la même séquence
                    # (pas de mise à jour de séquence pour éviter les erreurs de dimension)
                
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
    
    def train(self, features_df: pd.DataFrame, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2) -> Dict[str, Any]:
        """Entraîne le modèle LSTM avec les données de features"""
        try:
            logger.info(f"🚀 Début de l'entraînement LSTM pour {self.ticker}")
            
            # Préparer les features
            feature_cols = [col for col in self.feature_columns if col in features_df.columns]
            if not feature_cols:
                logger.error("❌ Aucune colonne de feature technique trouvée")
                return {"error": "Aucune colonne de feature technique trouvée"}
            
            # Extraire les features
            features_data = features_df[feature_cols].values
            
            # Trouver l'index de la colonne CLOSE pour le target
            target_col_idx = 0  # Par défaut, utiliser la première colonne
            if 'CLOSE' in features_df.columns:
                # Si CLOSE est dans les features, l'utiliser comme target
                close_idx = list(features_df.columns).index('CLOSE')
                # Ajouter CLOSE aux features si nécessaire
                if 'CLOSE' not in feature_cols:
                    feature_cols.append('CLOSE')
                    # Ajouter la colonne CLOSE aux données
                    close_data = features_df['CLOSE'].values.reshape(-1, 1)
                    features_data = np.hstack([features_data, close_data])
                    logger.info("🎯 Ajout de CLOSE aux features pour le target")
                
                target_col_idx = feature_cols.index('CLOSE')
                logger.info(f"🎯 Utilisation de CLOSE (index {target_col_idx}) comme target")
            
            # Gérer les NaN - remplacer par la moyenne de la colonne
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            features_data = imputer.fit_transform(features_data)
            
            # Normaliser les features
            self.scaler = RobustScaler()
            features_scaled = self.scaler.fit_transform(features_data)
            
            # Créer les séquences
            X, y = self.create_sequences(features_scaled, target_col_idx)
            if X is None:
                return {"error": "Pas assez de données pour créer les séquences"}
            
            # Diviser en train/validation
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Créer le modèle
            input_size = X.shape[2]
            self.model = FinancialLSTM(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                output_size=1
            ).to(self.device)
            
            # Optimiseur et loss
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Entraînement
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience = 15
            patience_counter = 0
            
            for epoch in range(epochs):
                # Mode entraînement
                self.model.train()
                train_loss = 0.0
                
                # Batch training
                for i in range(0, len(X_train), batch_size):
                    batch_X = torch.FloatTensor(X_train[i:i+batch_size]).to(self.device)
                    batch_y = torch.FloatTensor(y_train[i:i+batch_size]).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for i in range(0, len(X_val), batch_size):
                        batch_X = torch.FloatTensor(X_val[i:i+batch_size]).to(self.device)
                        batch_y = torch.FloatTensor(y_val[i:i+batch_size]).to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        val_loss += loss.item()
                
                train_loss /= (len(X_train) // batch_size + 1)
                val_loss /= (len(X_val) // batch_size + 1)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 10 == 0:
                    logger.info(f"📊 Époque {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
                
                if patience_counter >= patience:
                    logger.info(f"⏹️ Arrêt anticipé à l'époque {epoch}")
                    break
            
            # Marquer comme chargé
            self.is_loaded = True
            
            logger.info(f"✅ Entraînement terminé: {len(train_losses)} époques, Best Val Loss: {best_val_loss:.6f}")
            
            return {
                "success": True,
                "epochs_trained": len(train_losses),
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "features_used": feature_cols
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement: {e}")
            return {"error": str(e)}
    
    def save_model(self, model_path: Path) -> bool:
        """Sauvegarde le modèle entraîné"""
        try:
            if not self.is_loaded:
                logger.error("❌ Aucun modèle chargé à sauvegarder")
                return False
            
            # Créer le répertoire si nécessaire
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder le modèle et le scaler
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length,
                'ticker': self.ticker,
                'input_size': self.model.lstm.input_size
            }
            
            torch.save(model_data, model_path)
            logger.info(f"✅ Modèle sauvegardé: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")
            return False

# Alias pour la compatibilité
LSTMPredictor = PricePredictor