"""
Service de fusion pour l'onglet Production
Fusion adaptative des signaux (prix + sentiment + prédiction)
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from core.fusion import AdaptiveFusion
    FUSION_AVAILABLE = True
    logger.info("✅ AdaptiveFusion importé avec succès")
except ImportError as e:
    logger.warning(f"⚠️ AdaptiveFusion non disponible: {e}")
    FUSION_AVAILABLE = False
    AdaptiveFusion = None


class FusionService:
    """Service de fusion adaptative pour l'onglet Production"""
    
    def __init__(self):
        self.fusion_engine = None
        self.fusion_history = []
        
        if FUSION_AVAILABLE:
            try:
                self.fusion_engine = AdaptiveFusion()
                logger.info("✅ Moteur de fusion initialisé")
            except Exception as e:
                logger.warning(f"⚠️ Erreur initialisation fusion: {e}")
    
    def calculate_fusion_score(self, 
                             price_signal: float,
                             sentiment_signal: float, 
                             prediction_signal: float,
                             market_regime: str = "normal") -> Dict[str, Any]:
        """Calcule le score de fusion final"""
        try:
            if self.fusion_engine:
                # Utiliser le vrai moteur de fusion
                signals = {
                    "price": price_signal,
                    "sentiment": sentiment_signal,
                    "prediction": prediction_signal
                }
                fusion_score = self.fusion_engine.fuse_signals(signals)
                confidence = 0.8  # Valeur par défaut
                weights = self.fusion_engine.current_weights
            else:
                # Fallback simulation
                fusion_score = self._simulate_fusion_score(
                    price_signal, sentiment_signal, prediction_signal
                )
                confidence = 0.75
                weights = {
                    "price": 0.4,
                    "sentiment": 0.3,
                    "prediction": 0.3
                }
            
            # Déterminer la recommandation
            recommendation = self._get_recommendation(fusion_score, confidence)
            
            # Sauvegarder dans l'historique
            fusion_data = {
                "timestamp": datetime.now(),
                "fusion_score": fusion_score,
                "confidence": confidence,
                "recommendation": recommendation,
                "price_signal": price_signal,
                "sentiment_signal": sentiment_signal,
                "prediction_signal": prediction_signal,
                "weights": weights
            }
            self.fusion_history.append(fusion_data)
            
            # Garder seulement les 100 dernières entrées
            if len(self.fusion_history) > 100:
                self.fusion_history = self.fusion_history[-100:]
            
            return {
                "fusion_score": fusion_score,
                "confidence": confidence,
                "recommendation": recommendation,
                "weights": weights,
                "color": self._get_score_color(fusion_score),
                "label": self._get_score_label(fusion_score)
            }
        except Exception as e:
            logger.error(f"❌ Erreur calcul fusion: {e}")
            return {
                "fusion_score": 0.5,
                "confidence": 0.5,
                "recommendation": "ATTENDRE",
                "weights": {"price": 0.33, "sentiment": 0.33, "prediction": 0.34},
                "color": "gray",
                "label": "Neutre"
            }
    
    def get_multi_signal_chart_data(self, ticker: str = "SPY") -> Dict[str, Any]:
        """Récupère les données pour le graphique multi-signaux"""
        try:
            # Simulation de données (en production, récupérer depuis la base)
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            
            # Signaux simulés
            price_signals = np.random.uniform(0.3, 0.8, 30)
            sentiment_signals = np.random.uniform(0.2, 0.9, 30)
            prediction_signals = np.random.uniform(0.1, 0.7, 30)
            
            # Calculer les scores de fusion
            fusion_signals = []
            for i in range(30):
                fusion_data = self.calculate_fusion_score(
                    price_signals[i], sentiment_signals[i], prediction_signals[i]
                )
                fusion_signals.append(fusion_data["fusion_score"])
            
            return {
                "dates": dates.tolist(),
                "price_signals": price_signals.tolist(),
                "sentiment_signals": sentiment_signals.tolist(),
                "prediction_signals": prediction_signals.tolist(),
                "fusion_signals": fusion_signals
            }
        except Exception as e:
            logger.error(f"❌ Erreur données graphique: {e}")
            return {
                "dates": [],
                "price_signals": [],
                "sentiment_signals": [],
                "prediction_signals": [],
                "fusion_signals": []
            }
    
    def get_fusion_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Récupère l'historique des fusions"""
        try:
            return self.fusion_history[-limit:] if self.fusion_history else []
        except Exception as e:
            logger.error(f"❌ Erreur historique fusion: {e}")
            return []
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques de fusion"""
        try:
            if not self.fusion_history:
                return {
                    "total_signals": 0,
                    "avg_score": 0.5,
                    "last_recommendation": "ATTENDRE",
                    "current_weights": {"price": 0.33, "sentiment": 0.33, "prediction": 0.34}
                }
            
            recent_data = self.fusion_history[-10:]  # 10 dernières entrées
            avg_score = np.mean([d["fusion_score"] for d in recent_data])
            last_recommendation = recent_data[-1]["recommendation"] if recent_data else "ATTENDRE"
            current_weights = recent_data[-1]["weights"] if recent_data else {"price": 0.33, "sentiment": 0.33, "prediction": 0.34}
            
            return {
                "total_signals": len(self.fusion_history),
                "avg_score": avg_score,
                "last_recommendation": last_recommendation,
                "current_weights": current_weights
            }
        except Exception as e:
            logger.error(f"❌ Erreur stats fusion: {e}")
            return {
                "total_signals": 0,
                "avg_score": 0.5,
                "last_recommendation": "ATTENDRE",
                "current_weights": {"price": 0.33, "sentiment": 0.33, "prediction": 0.34}
            }
    
    def _simulate_fusion_score(self, price: float, sentiment: float, prediction: float) -> float:
        """Simule un score de fusion (fallback)"""
        # Poids adaptatifs basés sur la volatilité des signaux
        weights = [0.4, 0.3, 0.3]  # price, sentiment, prediction
        signals = [price, sentiment, prediction]
        
        # Calcul pondéré
        fusion_score = sum(w * s for w, s in zip(weights, signals))
        
        # Normaliser entre 0 et 1
        return max(0, min(1, fusion_score))
    
    def _get_recommendation(self, score: float, confidence: float) -> str:
        """Détermine la recommandation basée sur le score et la confiance"""
        if confidence < 0.6:
            return "ATTENDRE"
        elif score > 0.7:
            return "ACHETER"
        elif score < 0.3:
            return "VENDRE"
        else:
            return "ATTENDRE"
    
    def _get_score_color(self, score: float) -> str:
        """Détermine la couleur basée sur le score"""
        if score > 0.7:
            return "green"
        elif score > 0.5:
            return "blue"
        elif score > 0.3:
            return "orange"
        else:
            return "red"
    
    def _get_score_label(self, score: float) -> str:
        """Convertit le score en label"""
        if score > 0.8:
            return "Très Fort"
        elif score > 0.6:
            return "Fort"
        elif score > 0.4:
            return "Modéré"
        elif score > 0.2:
            return "Faible"
        else:
            return "Très Faible"
