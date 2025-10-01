"""
Service de prédiction LSTM pour Streamlit
Utilise le vrai modèle LSTM entraîné depuis data/models/spy
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Ajouter le répertoire src au path pour importer les modules core
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Importer les constantes de normalisation
from gui.constants import normalize_columns

try:
    from core.prediction import PricePredictor
    LSTM_AVAILABLE = True
    logger.info("✅ PricePredictor importé avec succès")
except ImportError as e:
    logger.warning(f"⚠️ PricePredictor non disponible: {e}")
    LSTM_AVAILABLE = False
    PricePredictor = None


class PredictionService:
    """Service de prédiction LSTM utilisant le vrai modèle entraîné"""
    
    def __init__(self):
        self.model_path = Path("data/models/spy")
        self.predictor = None
        self.fallback_mode = False
        logger.info("🤖 Service de prédiction LSTM initialisé")
    
    def _load_model(self) -> bool:
        """Charge le modèle LSTM réel"""
        try:
            if not LSTM_AVAILABLE:
                logger.warning("⚠️ PricePredictor non disponible, passage en mode fallback")
                self.fallback_mode = True
                return False
                
            if self.predictor is None:
                self.predictor = PricePredictor("SPY")
                success = self.predictor.load_model()
                
                if success:
                    logger.info("✅ Modèle LSTM SPY chargé avec succès")
                    return True
                else:
                    logger.warning("⚠️ Échec chargement modèle LSTM, passage en mode fallback")
                    self.fallback_mode = True
                    return False
            
            return self.predictor.is_loaded
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle LSTM: {e}")
            self.fallback_mode = True
            return False
    
    def predict(self, df: pd.DataFrame, horizon: int = 20) -> Dict[str, Any]:
        """Génère des prédictions LSTM avec le vrai modèle"""
        try:
            if df.empty:
                return self._create_empty_prediction()
            
            # Charger le modèle si nécessaire
            if not self._load_model():
                return self._fallback_predict(df, horizon)
            
            # Utiliser le vrai modèle LSTM
            return self._real_predict(df, horizon)
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction: {e}")
            return self._create_empty_prediction()
    
    def predict_with_features(self, ticker: str, horizon: int = 20) -> Dict[str, Any]:
        """Génère des prédictions LSTM en chargeant les features techniques"""
        try:
            # Charger directement les features techniques depuis le fichier
            from constants import CONSTANTS
            features_path = CONSTANTS.get_data_path('features', ticker)
            if not features_path.exists():
                logger.warning(f"⚠️ Fichier de features non trouvé: {features_path}")
                return self._create_empty_prediction()
            
            # Charger les features
            features_df = pd.read_parquet(features_path)
            features_df.columns = features_df.columns.str.upper()
            
            if features_df.empty:
                logger.warning(f"⚠️ Aucune feature trouvée pour {ticker}")
                return self._create_empty_prediction()
            
            # Charger le modèle si nécessaire
            if not self._load_model():
                return self._fallback_predict(features_df, horizon)
            
            # Utiliser le vrai modèle LSTM avec les features
            return self._real_predict(features_df, horizon)
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction avec features: {e}")
            return self._create_empty_prediction()
    
    def _real_predict(self, df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Prédiction avec le vrai modèle LSTM"""
        try:
            # Convertir l'index DATE en colonne si nécessaire
            if 'DATE' in df.index.names:
                df = df.reset_index()
            
            # Utiliser la nouvelle méthode pour les features techniques
            prediction_result = self.predictor.predict_with_technical_features(df, horizon=horizon)
            
            if 'error' in prediction_result:
                logger.warning(f"⚠️ Erreur prédiction: {prediction_result['error']}")
                return self._fallback_predict(df, horizon)
            
            # Extraire les prédictions
            historical_predictions = prediction_result.get('historical_predictions', [])
            future_predictions = prediction_result.get('predictions', [])
            future_dates = prediction_result.get('prediction_dates', [])
            
            # Si pas de dates futures, les générer
            if not future_dates and future_predictions:
                last_date = pd.to_datetime(df['DATE'].iloc[-1])
                future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(len(future_predictions))]
            
            logger.info(f"✅ Prédiction LSTM: {len(historical_predictions)} historiques + {len(future_predictions)} futures")
            
            return {
                'historical_predictions': historical_predictions,
                'predictions': future_predictions,
                'prediction_dates': future_dates,
                'model_type': 'lstm_real',
                'confidence': 0.9
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction LSTM réelle: {e}")
            return self._fallback_predict(df, horizon)
    
    def _fallback_predict(self, df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Prédiction fallback sans simulation - Retourne erreur si modèle non disponible"""
        logger.error("❌ FALLBACK DÉSACTIVÉ - Aucune simulation autorisée selon les règles du projet")
        logger.error("❌ Le modèle LSTM doit être chargé correctement pour faire des prédictions")
        return self._create_empty_prediction()
    
    def _fallback_predict_OLD_DISABLED(self, df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """ANCIEN FALLBACK DÉSACTIVÉ - Violait les règles anti-simulation"""
        try:
            if df.empty:
                return self._create_empty_prediction()
            
            # Si 'date' n'est pas dans les colonnes, essayer de l'extraire de l'index
            if df.index.name == 'DATE' and 'date' not in df.columns:
                df_normalized = df.reset_index()
            else:
                df_normalized = df.copy()
            
            # Pour les features, on n'a pas besoin de normaliser les colonnes
            # car elles sont déjà en majuscules et n'ont pas de colonnes 'date'/'close'
            df_normalized = df_normalized.copy()
            
            # Pour les features, utiliser l'index comme dates
            if df_normalized.index.name == 'DATE':
                df_sorted = df_normalized.reset_index()
                dates = pd.to_datetime(df_sorted['DATE']).tolist()
            else:
                # Utiliser des dates réelles basées sur aujourd'hui
                df_sorted = df_normalized.reset_index(drop=True)
                base_date = pd.Timestamp.now()
                dates = [base_date - pd.Timedelta(days=len(df_sorted)-i-1) for i in range(len(df_sorted))]
            
            # Utiliser une colonne numérique comme référence de prix
            numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                logger.error("❌ Aucune colonne numérique trouvée pour fallback")
                return self._create_empty_prediction()
            
            # Chercher une colonne de prix appropriée (close, open, high, low)
            price_col = None
            for col in ['close', 'open', 'high', 'low', 'CLOSE', 'OPEN', 'HIGH', 'LOW']:
                if col in df_sorted.columns:
                    price_col = col
                    break
            
            # Si aucune colonne de prix trouvée, utiliser la première colonne numérique
            if price_col is None:
                price_col = numeric_cols[0]
            
            prices = df_sorted[price_col].tolist()
            logger.info(f"✅ Utilisation de {price_col} comme prix de référence pour fallback")
            
            # Prédictions historiques (simulation basée sur les prix réels)
            historical_predictions = []
            if len(prices) >= 3:  # Minimum 3 jours pour calculer une tendance
                # Calculer la tendance récente
                recent_returns = []
                for i in range(1, min(6, len(prices))):
                    ret = (prices[-i] - prices[-i-1]) / prices[-i-1]
                    recent_returns.append(ret)
                
                avg_return = np.mean(recent_returns) if recent_returns else 0
                volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.02
                
                # Simuler les prédictions historiques
                for i in range(len(prices)):
                    if i < 3:  # Pour les premiers jours, utiliser le prix réel
                        historical_predictions.append(prices[i])
                    else:
                        # Simulation basée sur la tendance récente
                        noise = np.random.normal(0, volatility * 0.1)
                        pred_return = avg_return + noise
                        pred_price = prices[i-1] * (1 + pred_return)
                        historical_predictions.append(pred_price)
            else:
                historical_predictions = prices
            
            # Prédictions futures
            future_predictions = []
            future_dates = []
            
            if len(prices) > 0:
                last_price = prices[-1]
                # Gérer les dates fictives
                if dates[-1].startswith('Day_'):
                    last_date = pd.Timestamp.now()
                else:
                    last_date = pd.to_datetime(dates[-1])
                
                # Calculer la tendance récente pour les prédictions futures
                recent_returns = []
                if len(prices) >= 5:
                    for i in range(1, min(6, len(prices))):
                        ret = (prices[-i] - prices[-i-1]) / prices[-i-1]
                        recent_returns.append(ret)
                
                avg_return = np.mean(recent_returns) if recent_returns else 0
                volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.02
                
                # Assurer une volatilité minimale réaliste pour les prédictions
                min_volatility = 0.015  # 1.5% de volatilité minimale
                volatility = max(volatility, min_volatility)
                
                # Générer les prédictions futures avec une tendance cohérente et volatilité réaliste
                current_price = last_price
                
                for i in range(1, horizon + 1):
                    # Tendance basée sur la moyenne mobile récente
                    if len(prices) >= 20:
                        ma_20 = np.mean(prices[-20:])
                        trend_factor = 1 + (ma_20 - last_price) / last_price * 0.1
                    else:
                        trend_factor = 1 + avg_return * 0.5
                    
                    # Ajouter de la variabilité réaliste avec plus de volatilité
                    noise = np.random.normal(0, volatility * 0.3)  # Augmenter la volatilité
                    predicted_return = avg_return * trend_factor + noise
                    predicted_price = current_price * (1 + predicted_return)
                    
                    future_predictions.append(predicted_price)
                    future_date = last_date + pd.Timedelta(days=i)
                    future_dates.append(future_date)
                    current_price = predicted_price
            
            logger.warning(f"⚠️ ANCIEN FALLBACK (DÉSACTIVÉ): {len(historical_predictions)} historiques + {len(future_predictions)} futures")
            
            # NE JAMAIS RETOURNER DE SIMULATIONS
            return self._create_empty_prediction()
            
        except Exception as e:
            logger.error(f"❌ Erreur fallback: {e}")
            return self._create_empty_prediction()
    
    def _create_empty_prediction(self) -> Dict[str, Any]:
        """Crée une prédiction vide en cas d'erreur"""
        return {
            'historical_predictions': [],
            'predictions': [],
            'prediction_dates': [],
            'model_type': 'empty',
            'confidence': 0.0
        }
