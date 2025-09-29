"""
Service de validation en temps réel des décisions de trading
Valide les décisions BUY/SELL contre l'évolution des prix futurs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import sys
import json

# Ajouter le chemin src pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from constants import CONSTANTS

class DecisionValidationService:
    """Service de validation en temps réel des décisions de trading"""
    
    def __init__(self):
        self.data_path = CONSTANTS.get_data_path()
        self.decisions_path = self.data_path / "trading" / "decisions_log"
        self.validation_path = self.data_path / "trading" / "validation_log"
        self.validation_path.mkdir(parents=True, exist_ok=True)
        
        # Fichier parquet pour l'historique des validations
        self.validation_file = self.validation_path / "decision_validation_history.parquet"
        
        logger.info("🔍 Service de validation des décisions initialisé")
    
    def validate_decision(self, ticker: str, decision: str, fusion_score: float, 
                         current_price: float, timestamp: datetime) -> Dict[str, Any]:
        """
        Valide une décision de trading en temps réel
        
        Args:
            ticker: Symbole de l'action
            decision: Décision prise (BUY/SELL/HOLD)
            fusion_score: Score de fusion utilisé
            current_price: Prix actuel
            timestamp: Timestamp de la décision
            
        Returns:
            Dict contenant les résultats de validation
        """
        try:
            # Pour les décisions HOLD, pas de validation nécessaire
            if decision.upper() == "HOLD":
                return {
                    "status": "hold",
                    "message": "Décision HOLD - Aucune validation nécessaire",
                    "accuracy": None,
                    "price_change": None,
                    "validation_time": None,
                    "is_correct": None
                }
            
            # Attendre un délai pour avoir des données de prix futures
            # Pour la démonstration, on simule avec des données historiques
            validation_result = self._simulate_validation(ticker, decision, current_price, timestamp)
            
            # Sauvegarder la validation
            self._save_validation(ticker, decision, fusion_score, current_price, 
                                timestamp, validation_result)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Erreur validation décision: {e}")
            return {
                "status": "error",
                "message": f"Erreur validation: {str(e)}",
                "accuracy": None,
                "price_change": None,
                "validation_time": None,
                "is_correct": None
            }
    
    def _simulate_validation(self, ticker: str, decision: str, current_price: float, 
                           timestamp: datetime) -> Dict[str, Any]:
        """
        Simule la validation en utilisant des données historiques
        En production, ceci serait remplacé par une validation en temps réel
        """
        try:
            # Charger les données de prix pour simuler l'évolution
            price_data = self._load_price_data(ticker)
            
            if price_data.empty:
                return {
                    "status": "no_data",
                    "message": "Données de prix indisponibles",
                    "accuracy": None,
                    "price_change": None,
                    "validation_time": None,
                    "is_correct": None
                }
            
            # Simuler l'évolution du prix (dans un vrai système, on attendrait 15min)
            # Ici on utilise des données historiques pour la démonstration
            future_price = self._simulate_future_price(price_data, current_price, timestamp)
            
            # Calculer le changement de prix
            price_change = (future_price - current_price) / current_price * 100
            
            # Déterminer si la décision était correcte
            is_correct = self._evaluate_decision_correctness(decision, price_change)
            
            # Calculer la précision
            accuracy = self._calculate_accuracy(decision, price_change)
            
            # Déterminer le statut
            if accuracy >= 0.8:
                status = "✅ Correct"
                message = f"Prix: ${current_price:.2f} → ${future_price:.2f} ({price_change:+.2f}%)"
            elif accuracy >= 0.5:
                status = "⚠️ Partiellement correct"
                message = f"Prix: ${current_price:.2f} → ${future_price:.2f} ({price_change:+.2f}%)"
            else:
                status = "❌ Incorrect"
                message = f"Prix: ${current_price:.2f} → ${future_price:.2f} ({price_change:+.2f}%)"
            
            return {
                "status": status,
                "message": message,
                "accuracy": accuracy,
                "price_change": price_change,
                "validation_time": datetime.now(),
                "is_correct": is_correct,
                "current_price": current_price,
                "future_price": future_price
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur simulation validation: {e}")
            return {
                "status": "error",
                "message": f"Erreur simulation: {str(e)}",
                "accuracy": None,
                "price_change": None,
                "validation_time": None,
                "is_correct": None
            }
    
    def _load_price_data(self, ticker: str) -> pd.DataFrame:
        """Charge les données de prix pour la validation"""
        try:
            # Essayer de charger les données 15min d'abord
            price_file = self.data_path / "realtime" / "prices" / f"{ticker.lower()}_15min.parquet"
            
            if price_file.exists():
                df = pd.read_parquet(price_file)
                if 'ts_utc' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['ts_utc'])
                elif 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            
            # Fallback sur les données historiques
            hist_file = self.data_path / "historical" / "yfinance" / f"{ticker}_1999_2025.parquet"
            if hist_file.exists():
                df = pd.read_parquet(hist_file)
                if 'date' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'])
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement données prix: {e}")
            return pd.DataFrame()
    
    def _simulate_future_price(self, price_data: pd.DataFrame, current_price: float, 
                              timestamp: datetime) -> float:
        """
        Simule l'évolution du prix futur
        En production, ceci attendrait 15min et utiliserait les vrais prix
        """
        try:
            # Pour la démonstration, on simule une évolution basée sur l'historique
            if price_data.empty:
                # Si pas de données, simuler une évolution aléatoire
                volatility = 0.02  # 2% de volatilité
                change = np.random.normal(0, volatility)
                return current_price * (1 + change)
            
            # Utiliser l'historique pour simuler une évolution réaliste
            recent_data = price_data.tail(100)  # 100 derniers points
            
            if len(recent_data) < 10:
                # Pas assez de données, simulation simple
                volatility = 0.01
                change = np.random.normal(0, volatility)
                return current_price * (1 + change)
            
            # Calculer la volatilité historique
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Simuler une évolution basée sur la volatilité historique
            change = np.random.normal(0, volatility)
            return current_price * (1 + change)
            
        except Exception as e:
            logger.error(f"❌ Erreur simulation prix futur: {e}")
            # Fallback: simulation simple
            return current_price * (1 + np.random.normal(0, 0.01))
    
    def _evaluate_decision_correctness(self, decision: str, price_change: float) -> bool:
        """Évalue si une décision était correcte basée sur l'évolution du prix"""
        if decision.upper() == "BUY":
            return price_change > 0.5  # BUY correct si prix monte de plus de 0.5%
        elif decision.upper() == "SELL":
            return price_change < -0.5  # SELL correct si prix baisse de plus de 0.5%
        else:
            return True  # HOLD toujours considéré comme correct
    
    def _calculate_accuracy(self, decision: str, price_change: float) -> float:
        """Calcule la précision d'une décision"""
        if decision.upper() == "BUY":
            if price_change > 1.0:
                return 1.0  # Parfait
            elif price_change > 0.5:
                return 0.8  # Très bon
            elif price_change > 0:
                return 0.6  # Bon
            else:
                return 0.2  # Mauvais
        elif decision.upper() == "SELL":
            if price_change < -1.0:
                return 1.0  # Parfait
            elif price_change < -0.5:
                return 0.8  # Très bon
            elif price_change < 0:
                return 0.6  # Bon
            else:
                return 0.2  # Mauvais
        else:
            return 0.7  # HOLD neutre
    
    def _save_validation(self, ticker: str, decision: str, fusion_score: float, 
                        current_price: float, timestamp: datetime, 
                        validation_result: Dict[str, Any]):
        """Sauvegarde une validation dans le fichier parquet"""
        try:
            validation_data = {
                "timestamp": timestamp.isoformat(),
                "ticker": ticker,
                "decision": decision,
                "fusion_score": fusion_score,
                "current_price": current_price,
                "validation_time": validation_result.get("validation_time", datetime.now()).isoformat(),
                "accuracy": validation_result.get("accuracy"),
                "price_change": validation_result.get("price_change"),
                "is_correct": validation_result.get("is_correct"),
                "status": validation_result.get("status"),
                "message": validation_result.get("message")
            }
            
            # Charger l'historique existant
            if self.validation_file.exists():
                df = pd.read_parquet(self.validation_file)
            else:
                df = pd.DataFrame()
            
            # Ajouter la nouvelle validation
            new_row = pd.DataFrame([validation_data])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Sauvegarder
            df.to_parquet(self.validation_file, index=False)
            logger.info(f"✅ Validation sauvegardée: {ticker} - {decision} - {validation_result.get('status')}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde validation: {e}")
    
    def get_validation_history(self, ticker: str, days: int = 7) -> pd.DataFrame:
        """Récupère l'historique des validations pour un ticker"""
        try:
            if not self.validation_file.exists():
                return pd.DataFrame()
            
            df = pd.read_parquet(self.validation_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filtrer par ticker et période
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_df = df[
                (df['ticker'] == ticker) & 
                (df['timestamp'] >= cutoff_date)
            ].sort_values('timestamp')
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération historique validation: {e}")
            return pd.DataFrame()
    
    def get_validation_stats(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Calcule les statistiques de validation pour un ticker"""
        try:
            df = self.get_validation_history(ticker, days)
            
            if df.empty:
                return {
                    "total_decisions": 0,
                    "total_validations": 0,
                    "correct_decisions": 0,
                    "accuracy_rate": 0.0,
                    "average_accuracy": 0.0,
                    "buy_decisions": 0,
                    "sell_decisions": 0,
                    "hold_decisions": 0,
                    "buy_accuracy": 0.0,
                    "sell_accuracy": 0.0
                }
            
            # Statistiques générales
            total_decisions = len(df)
            total_validations = len(df[df['is_correct'].notna()])
            correct_decisions = len(df[df['is_correct'] == True])
            accuracy_rate = correct_decisions / total_validations if total_validations > 0 else 0.0
            average_accuracy = df['accuracy'].mean() if 'accuracy' in df.columns else 0.0
            
            # Statistiques par type de décision
            buy_decisions = len(df[df['decision'] == 'BUY'])
            sell_decisions = len(df[df['decision'] == 'SELL'])
            hold_decisions = len(df[df['decision'] == 'HOLD'])
            
            buy_accuracy = df[df['decision'] == 'BUY']['accuracy'].mean() if buy_decisions > 0 else 0.0
            sell_accuracy = df[df['decision'] == 'SELL']['accuracy'].mean() if sell_decisions > 0 else 0.0
            
            return {
                "total_decisions": total_decisions,
                "total_validations": total_validations,
                "correct_decisions": correct_decisions,
                "accuracy_rate": round(accuracy_rate, 3),
                "average_accuracy": round(average_accuracy, 3),
                "buy_decisions": buy_decisions,
                "sell_decisions": sell_decisions,
                "hold_decisions": hold_decisions,
                "buy_accuracy": round(buy_accuracy, 3),
                "sell_accuracy": round(sell_accuracy, 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul statistiques validation: {e}")
            return {
                "total_decisions": 0,
                "total_validations": 0,
                "correct_decisions": 0,
                "accuracy_rate": 0.0,
                "average_accuracy": 0.0,
                "buy_decisions": 0,
                "sell_decisions": 0,
                "hold_decisions": 0,
                "buy_accuracy": 0.0,
                "sell_accuracy": 0.0
            }
    
    def get_adaptive_threshold_performance(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Analyse la performance des seuils adaptatifs"""
        try:
            df = self.get_validation_history(ticker, days)
            
            if df.empty:
                return {
                    "threshold_analysis": "Aucune donnée disponible",
                    "recommended_adjustments": [],
                    "performance_score": 0.0
                }
            
            # Analyser les décisions par seuil de fusion
            if 'fusion_score' in df.columns:
                # Grouper par plages de fusion_score
                df['fusion_range'] = pd.cut(df['fusion_score'], 
                                          bins=[-np.inf, -0.1, -0.05, 0.05, 0.1, np.inf],
                                          labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
                
                performance_by_range = df.groupby('fusion_range').agg({
                    'is_correct': ['count', 'sum', 'mean'],
                    'accuracy': 'mean'
                }).round(3)
                
                # Recommandations d'ajustement
                recommendations = []
                if performance_by_range.loc['Positive', ('is_correct', 'mean')] < 0.6:
                    recommendations.append("Considérer augmenter le seuil BUY")
                if performance_by_range.loc['Negative', ('is_correct', 'mean')] < 0.6:
                    recommendations.append("Considérer diminuer le seuil SELL")
                
                performance_score = df['accuracy'].mean() if 'accuracy' in df.columns else 0.0
                
                return {
                    "threshold_analysis": performance_by_range.to_dict(),
                    "recommended_adjustments": recommendations,
                    "performance_score": round(performance_score, 3)
                }
            else:
                return {
                    "threshold_analysis": "Données de fusion_score manquantes",
                    "recommended_adjustments": [],
                    "performance_score": 0.0
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur analyse performance seuils: {e}")
            return {
                "threshold_analysis": f"Erreur: {str(e)}",
                "recommended_adjustments": [],
                "performance_score": 0.0
            }

