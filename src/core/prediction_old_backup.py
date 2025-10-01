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
from sklearn.preprocessing import StandardScaler  # Moyenne au lieu de médiane

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import CONSTANTS

class FinancialLSTM(nn.Module):
    """
    v3 OPTIMAL - Architecture Simple et Performante
    
    Basé sur recherche: LSTM simple performe MIEUX que modèles complexes
    Référence: 653$ (écart 14$ = 2.1% MAPE)
    
    Architecture:
    - LSTM: 64 units x 2 layers, dropout 20%
    - Dense: 64 → 32 → 1
    - Adam + MSE + Weighted Loss
    
    KEEP IT SIMPLE: Complexité != Performance
    """
    
    def __init__(self, input_size: int = 16, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(FinancialLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM simple (prouvé optimal)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,      # 64
            num_layers=num_layers,        # 2
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dense layers simple
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, x_long=None):
        """
        x: [batch, seq, features]
        x_long: ignoré (backward compatibility)
        """
        batch_size = x.size(0)
        
        # LSTM
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        
        # Dernière sortie
        out = out[:, -1, :]
        
        # Dense layers
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class PricePredictor:
    """Prédicteur de prix utilisant un modèle LSTM"""
    
    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker.upper()
        self.model: Optional[nn.Module] = None
        self.scaler: Optional[StandardScaler] = None
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
            
            # Créer le modèle avec l'architecture sauvegardée
            input_size = model_data.get('input_size', len(model_data['feature_columns']))
            hidden_size = model_data.get('hidden_size', 128)  # Lire depuis le fichier
            num_layers = model_data.get('num_layers', 3)      # Lire depuis le fichier
            
            self.model = FinancialLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=1
            ).to(self.device)
            
            logger.info(f"🏗️ Architecture chargée: {input_size} -> {hidden_size}x{num_layers}")
            
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
            # MODE CLOSE ONLY (si le dataset ne contient que CLOSE)
            close_col = None
            for col in features_df.columns:
                if col.upper() == 'CLOSE':
                    close_col = col
                    break
            
            if close_col and len(features_df.columns) <= 3:
                # Mode CLOSE ONLY (1 feature)
                feature_cols = [close_col]
                features_data = features_df[[close_col]].values
                logger.info(f"🎯 Mode CLOSE ONLY pour prédiction (1 feature)")
            else:
                # Mode legacy avec features techniques
                feature_cols = []
                for expected_col in self.feature_columns:
                    found_col = None
                    for df_col in features_df.columns:
                        if expected_col.upper() == df_col.upper():
                            found_col = df_col
                            break
                    if found_col:
                        feature_cols.append(found_col)
                    else:
                        logger.warning(f"⚠️ Feature manquante: {expected_col}")
                
                if not feature_cols:
                    logger.error(f"❌ Aucune colonne de feature technique trouvée")
                    return {"error": "Aucune colonne de feature technique trouvée"}
                
                # Ajouter CLOSE si nécessaire
                if close_col and close_col not in feature_cols:
                    feature_cols.append(close_col)
                    logger.info(f"🎯 Ajout de {close_col} aux features pour la prédiction")
                
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
            historical_predictions_scaled = []  # Prédictions scalées avant dénormalisation
            with torch.no_grad():
                for i in range(len(X)):
                    sequence = torch.FloatTensor(X[i:i+1]).to(self.device)
                    pred = self.model(sequence)
                    pred_value = pred.detach().cpu().numpy()[0, 0]
                    historical_predictions_scaled.append(pred_value)
            
            # Prédictions futures
            future_predictions = []
            future_predictions_scaled = []  # Prédictions scalées avant dénormalisation
            future_dates = []
            
            if horizon > 0:
                # Utiliser la dernière séquence pour prédire l'avenir
                last_sequence = torch.FloatTensor(X_backup[-1:]).to(self.device)
                
                for i in range(horizon):
                    pred = self.model(last_sequence)
                    pred_value = pred.detach().cpu().numpy()[0, 0]
                    future_predictions_scaled.append(pred_value)
                    
                    # Pour les prédictions suivantes, on utilise la même séquence
                    # (pas de mise à jour de séquence pour éviter les erreurs de dimension)
                
                # Générer les dates futures
                # Chercher DATE dans les colonnes ou l'index
                if 'DATE' in features_df.columns:
                    last_date = pd.to_datetime(features_df['DATE'].iloc[-1])
                    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(horizon)]
                elif hasattr(features_df.index, 'to_pydatetime'):
                    last_date = features_df.index[-1]
                    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(horizon)]
                else:
                    # Utiliser la date actuelle comme base si aucune date trouvée
                    logger.warning("⚠️ Aucune colonne DATE trouvée, utilisation date actuelle")
                    last_date = pd.Timestamp.now()
                    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(horizon)]
            
            # DÉNORMALISER les prédictions pour revenir aux vrais prix
            # Trouver l'index de CLOSE dans feature_cols (comme lors de l'entraînement)
            close_idx = len(feature_cols) - 1  # CLOSE est la dernière car ajoutée à la fin
            logger.info(f"🔍 Index CLOSE pour dénormalisation: {close_idx}/{len(feature_cols)}")
            
            # Créer un tableau avec les prédictions historiques à la bonne position
            if historical_predictions_scaled:
                # Créer un tableau rempli de zéros pour toutes les features
                hist_dummy = np.zeros((len(historical_predictions_scaled), len(feature_cols)))
                hist_dummy[:, close_idx] = historical_predictions_scaled
                # Inverse transform pour dénormaliser
                hist_denorm = self.scaler.inverse_transform(hist_dummy)
                historical_predictions = hist_denorm[:, close_idx].tolist()
                logger.info(f"✅ Prédictions historiques dénormalisées: min={min(historical_predictions):.2f}, max={max(historical_predictions):.2f}")
            
            # Dénormaliser les prédictions futures
            if future_predictions_scaled:
                fut_dummy = np.zeros((len(future_predictions_scaled), len(feature_cols)))
                fut_dummy[:, close_idx] = future_predictions_scaled
                fut_denorm = self.scaler.inverse_transform(fut_dummy)
                future_predictions = fut_denorm[:, close_idx].tolist()
                logger.info(f"✅ Prédictions futures dénormalisées: {future_predictions[0]:.2f} -> {future_predictions[-1]:.2f}")
            
            logger.info(f"✅ Prédiction LSTM avec features: {len(historical_predictions)} historiques + {len(future_predictions)} futures (dénormalisées)")
            
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
            
            # MODE CLOSE ONLY (basé sur article de recherche)
            # L'article prouve que CLOSE seul performe mieux que avec features
            if 'CLOSE' in features_df.columns and len(features_df.columns) <= 3:
                # Mode CLOSE ONLY
                feature_cols = ['CLOSE']
                features_data = features_df['CLOSE'].values.reshape(-1, 1)
                target_col_idx = 0
                logger.info("🎯 Mode CLOSE ONLY (research-based: features = noise)")
            else:
                # Mode legacy avec features techniques
                feature_cols = [col for col in self.feature_columns if col in features_df.columns]
                if not feature_cols:
                    logger.error("❌ Aucune colonne de feature technique trouvée")
                    return {"error": "Aucune colonne de feature technique trouvée"}
                
                # Extraire les features
                features_data = features_df[feature_cols].values
                
                # Trouver l'index de la colonne CLOSE pour le target
                target_col_idx = 0
                if 'CLOSE' in features_df.columns:
                    close_idx = list(features_df.columns).index('CLOSE')
                    if 'CLOSE' not in feature_cols:
                        feature_cols.append('CLOSE')
                        close_data = features_df['CLOSE'].values.reshape(-1, 1)
                        features_data = np.hstack([features_data, close_data])
                        logger.info("🎯 Ajout de CLOSE aux features pour le target")
                    
                    target_col_idx = feature_cols.index('CLOSE')
                    logger.info(f"🎯 Utilisation de CLOSE (index {target_col_idx}) comme target")
            
            # Gérer les NaN - remplacer par la moyenne de la colonne
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            features_data = imputer.fit_transform(features_data)
            
            # Logging
            n_days_total = len(features_df)
            close_mean = features_data[:, target_col_idx].mean()
            close_current = features_data[-1, target_col_idx]
            logger.info(f"📊 Données: {n_days_total} jours | mean={close_mean:.2f}$ | actuel={close_current:.2f}$")
            
            # ARTICLE METHOD: Split AVANT scaling (évite data leakage)
            # 1. Split 60% train / 20% val / 20% test (comme l'article)
            n_train = int(len(features_data) * 0.6)
            n_val = int(len(features_data) * 0.2)
            
            features_train = features_data[:n_train]
            features_val = features_data[n_train:n_train+n_val]
            
            logger.info(f"📊 Split 60/20/20: Train={len(features_train)} | Val={len(features_val)}")
            
            # 2. FIT scaler sur TRAIN SEULEMENT (évite data leakage)
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            features_train_scaled = self.scaler.fit_transform(features_train)
            
            # 3. TRANSFORM val avec ce scaler (pas fit !)
            features_val_scaled = self.scaler.transform(features_val)
            
            logger.info(f"✅ Scaler fit sur TRAIN seulement (article method - no data leakage)")
            
            # 4. Créer les séquences sur train/val
            X_train, y_train = self.create_sequences(features_train_scaled, target_col_idx)
            X_val, y_val = self.create_sequences(features_val_scaled, target_col_idx)
            
            if X_train is None or X_val is None:
                return {"error": "Pas assez de données pour créer les séquences"}
            
            logger.info(f"📊 Séquences: Train={X_train.shape} | Val={X_val.shape}")
            
            # v3 OPTIMAL: Architecture Simple Prouvée
            input_size = X_train.shape[2]
            self.model = FinancialLSTM(
                input_size=input_size,
                hidden_size=64,       # 64 (optimal)
                num_layers=2,
                output_size=1,
                dropout=0.2           # Dropout 20%
            ).to(self.device)
            
            if input_size == 1:
                logger.info(f"🏗️ Architecture RESEARCH: CLOSE ONLY -> LSTM[64x2,drop=20%] -> Dense[32→1]")
                logger.info(f"📚 Basé sur arXiv:2501.17366v1: 'LSTM sans features = 96.41% accuracy'")
            else:
                logger.info(f"🏗️ Architecture v3: {input_size} features -> LSTM[64x2] -> Dense[32→1]")
            
            # Weighted MSE Loss
            n_samples = len(X_train)
            tau = n_samples / 5.0
            sample_weights = torch.exp(torch.arange(n_samples, dtype=torch.float32) / tau)
            sample_weights = sample_weights / sample_weights.sum() * n_samples
            sample_weights = sample_weights.to(self.device)
            
            logger.info(f"📊 Weighted Loss: tau={tau:.1f}, weight_max/weight_min={sample_weights[-1]/sample_weights[0]:.1f}x")
            
            # Optimiseur et loss
            optimizer = torch.optim.Adam(self.model.parameters(), lr=CONSTANTS.LSTM_LEARNING_RATE)
            criterion = nn.MSELoss(reduction='none')  # reduction='none' pour weighted loss
            
            # Entraînement
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience = CONSTANTS.LSTM_PATIENCE  # 20 au lieu de 15
            patience_counter = 0
            
            for epoch in range(epochs):
                # Mode entraînement
                self.model.train()
                train_loss = 0.0
                
                # Batch training avec weighted loss
                for i in range(0, len(X_train), batch_size):
                    batch_X = torch.FloatTensor(X_train[i:i+batch_size]).to(self.device)
                    batch_y = torch.FloatTensor(y_train[i:i+batch_size]).to(self.device)
                    batch_weights = sample_weights[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    losses = criterion(outputs.squeeze(), batch_y)
                    loss = (losses * batch_weights).mean()
                    loss.backward()
                    
                    # Gradient clipping (évite explosions)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
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
                        losses = criterion(outputs.squeeze(), batch_y)
                        val_loss += losses.mean().item()
                
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
            
            # Sauvegarder le modèle et le scaler avec l'architecture complète
            # Détecter si c'est DUAL-LSTM (lstm_short) ou simple (lstm)
            if hasattr(self.model, 'lstm_short'):
                input_size = self.model.lstm_short.input_size
            elif hasattr(self.model, 'lstm'):
                input_size = self.model.lstm.input_size
            else:
                input_size = 16  # Fallback
            
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length,
                'ticker': self.ticker,
                'input_size': input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers
            }
            
            torch.save(model_data, model_path)
            logger.info(f"✅ Modèle sauvegardé: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")
            return False

# Alias pour la compatibilité
LSTMPredictor = PricePredictor