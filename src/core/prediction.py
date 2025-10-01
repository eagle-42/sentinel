"""
🔮 LSTM Simple pour Sentinel2 - CLOSE ONLY
Basé sur recherche arXiv:2501.17366v1: LSTM sans features = meilleur
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import CONSTANTS


class FinancialLSTM(nn.Module):
    """
    LSTM Simple - Architecture Optimale
    
    Architecture:
    - LSTM: 64 units x 2 layers, dropout 20%
    - Dense: 64 → 32 → 1
    - Adam + MSE + Weighted Loss
    
    Basé sur: LSTM simple performe MIEUX que modèles complexes
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(FinancialLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class PricePredictor:
    """Prédicteur CLOSE ONLY (simple et performant)"""
    
    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker.upper()
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.sequence_length = CONSTANTS.LSTM_SEQUENCE_LENGTH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔮 Prédicteur CLOSE ONLY initialisé pour {self.ticker}")

    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """Charge le modèle LSTM"""
        try:
            if model_path is None:
                models_dir = CONSTANTS.get_model_path(self.ticker)
                if not models_dir.exists():
                    logger.error(f"❌ Répertoire modèle non trouvé: {models_dir}")
                    return False
                
                versions = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("version")]
                if not versions:
                    logger.error(f"❌ Aucune version de modèle trouvée")
                    return False
                
                latest_version = max(versions, key=lambda x: int(x.name.replace("version_", "")))
                model_path = latest_version / "model.pkl"
            
            if not model_path.exists():
                logger.error(f"❌ Modèle non trouvé: {model_path}")
                return False
            
            model_data = torch.load(model_path, map_location=self.device, weights_only=False)
            
            input_size = model_data.get('input_size', 1)
            hidden_size = model_data.get('hidden_size', 64)
            num_layers = model_data.get('num_layers', 2)
            
            self.model = FinancialLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers
            ).to(self.device)
            
            self.model.load_state_dict(model_data['model_state_dict'])
            self.scaler = model_data['scaler']
            self.sequence_length = model_data['sequence_length']
            self.model.eval()
            self.is_loaded = True
            
            logger.info(f"✅ Modèle chargé: {input_size}D -> LSTM[{hidden_size}x{num_layers}]")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement: {e}")
            return False

    def create_sequences(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Crée les séquences temporelles"""
        if len(features) < self.sequence_length:
            return None, None
        
        X, y = [], []
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(features[i, 0])  # Target = CLOSE
        
        return np.array(X), np.array(y)

    def predict(self, data: pd.DataFrame, horizon: int = 7) -> Dict[str, Any]:
        """Prédiction CLOSE ONLY"""
        if not self.is_loaded:
            logger.error("❌ Modèle non chargé")
            return {"error": "Modèle non chargé"}
        
        try:
            # Extraire CLOSE
            close_col = None
            for col in data.columns:
                if col.upper() == 'CLOSE':
                    close_col = col
                    break
            
            if not close_col:
                logger.error("❌ Colonne CLOSE non trouvée")
                return {"error": "Colonne CLOSE non trouvée"}
            
            features_data = data[[close_col]].values
            logger.info(f"🎯 Mode CLOSE ONLY: {len(features_data)} jours")
            
            # Scaler
            features_scaled = self.scaler.transform(features_data)
            
            # Séquences
            X, y = self.create_sequences(features_scaled)
            if X is None:
                return {"error": "Pas assez de données"}
            
            # Prédictions historiques
            hist_preds_scaled = []
            with torch.no_grad():
                for i in range(len(X)):
                    seq = torch.FloatTensor(X[i:i+1]).to(self.device)
                    pred = self.model(seq).cpu().numpy()[0, 0]
                    hist_preds_scaled.append(pred)
            
            # Prédictions futures
            fut_preds_scaled = []
            last_seq = torch.FloatTensor(X[-1:]).to(self.device)
            
            with torch.no_grad():
                for _ in range(horizon):
                    pred = self.model(last_seq).cpu().numpy()[0, 0]
                    fut_preds_scaled.append(pred)
            
            # Dénormaliser
            hist_dummy = np.array(hist_preds_scaled).reshape(-1, 1)
            hist_preds = self.scaler.inverse_transform(hist_dummy).flatten().tolist()
            
            fut_dummy = np.array(fut_preds_scaled).reshape(-1, 1)
            fut_preds = self.scaler.inverse_transform(fut_dummy).flatten().tolist()
            
            logger.info(f"✅ Prédictions: {len(hist_preds)} hist + {len(fut_preds)} futures")
            
            return {
                "historical_predictions": hist_preds,
                "predictions": fut_preds,
                "ticker": self.ticker,
                "horizon": horizon
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction: {e}")
            return {"error": str(e)}

    def train(self, features_df: pd.DataFrame, epochs: int = 150) -> Dict[str, Any]:
        """Entraîne le modèle LSTM (RETURNS + features corrélées)"""
        try:
            logger.info(f"🚀 Entraînement LSTM ARTICLE pour {self.ticker}")
            
            # TARGET = returns de CLOSE
            if 'TARGET' not in features_df.columns:
                logger.error("❌ Colonne TARGET manquante")
                return {"error": "Colonne TARGET manquante"}
            
            # Extraire features (colonnes avec _RETURN)
            feature_cols = [col for col in features_df.columns if '_RETURN' in col or col == 'TARGET']
            if 'DATE' in feature_cols:
                feature_cols.remove('DATE')
            
            features_data = features_df[feature_cols].values
            n_features = features_data.shape[1]
            logger.info(f"📊 Données: {len(features_data)} jours x {n_features} features (RETURNS)")
            
            # Imputer NaN
            imputer = SimpleImputer(strategy='mean')
            features_data = imputer.fit_transform(features_data)
            
            # Split 60/20/20 AVANT scaling (article method)
            n_train = int(len(features_data) * 0.6)
            n_val = int(len(features_data) * 0.2)
            
            feat_train = features_data[:n_train]
            feat_val = features_data[n_train:n_train+n_val]
            
            logger.info(f"📊 Split 60/20/20: Train={len(feat_train)} | Val={len(feat_val)}")
            
            # Scaler sur TRAIN seulement
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            feat_train_scaled = self.scaler.fit_transform(feat_train)
            feat_val_scaled = self.scaler.transform(feat_val)
            
            logger.info(f"✅ Scaler fit sur TRAIN (no data leakage)")
            
            # Séquences
            X_train, y_train = self.create_sequences(feat_train_scaled)
            X_val, y_val = self.create_sequences(feat_val_scaled)
            
            if X_train is None or X_val is None:
                return {"error": "Pas assez de données"}
            
            logger.info(f"📊 Séquences: Train={X_train.shape} | Val={X_val.shape}")
            
            # Modèle (input_size = nombre de features)
            input_size = X_train.shape[2]
            self.model = FinancialLSTM(input_size=input_size, hidden_size=64, num_layers=2).to(self.device)
            logger.info(f"🏗️ LSTM[64x2] + Dense[32→1] | {input_size} features RETURNS")
            
            # Sauvegarder feature_cols pour predict
            self.feature_cols = feature_cols
            
            # Training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=CONSTANTS.LSTM_LEARNING_RATE)
            criterion = nn.MSELoss()
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                self.model.train()
                
                X_t = torch.FloatTensor(X_train).to(self.device)
                y_t = torch.FloatTensor(y_train).to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_t)
                loss = criterion(outputs.squeeze(), y_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(X_val).to(self.device)
                    y_v = torch.FloatTensor(y_val).to(self.device)
                    val_outputs = self.model(X_v)
                    val_loss = criterion(val_outputs.squeeze(), y_v).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 10 == 0:
                    logger.info(f"📊 Epoch {epoch}: Train={loss.item():.6f}, Val={val_loss:.6f}")
                
                if patience_counter >= CONSTANTS.LSTM_PATIENCE:
                    logger.info(f"⏹️ Early stop epoch {epoch}")
                    break
            
            self.is_loaded = True
            logger.info(f"✅ Entraînement terminé: Best Val Loss={best_val_loss:.6f}")
            
            return {
                "success": True,
                "epochs_trained": epoch + 1,
                "best_val_loss": best_val_loss
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement: {e}")
            return {"error": str(e)}

    def save_model(self, model_path: Path) -> bool:
        """Sauvegarde le modèle"""
        try:
            if not self.is_loaded:
                logger.error("❌ Aucun modèle à sauvegarder")
                return False
            
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            input_size = self.model.lstm.input_size if hasattr(self.model, 'lstm') else 1
            
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'sequence_length': self.sequence_length,
                'ticker': self.ticker,
                'input_size': input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'feature_cols': getattr(self, 'feature_cols', ['TARGET'])
            }
            
            torch.save(model_data, model_path)
            logger.info(f"✅ Modèle sauvegardé: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")
            return False


# Alias
LSTMPredictor = PricePredictor
