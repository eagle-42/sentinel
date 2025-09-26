"""
Service de vérification des décisions de trading
Gère l'historique des décisions et la validation des prédictions
"""

import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import sys

# Ajouter le chemin src pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from constants import CONSTANTS

class VerificationService:
    """Service de vérification des décisions de trading"""
    
    def __init__(self):
        self.data_path = CONSTANTS.get_data_path()
        self.decisions_path = self.data_path / "trading" / "decisions_log"
        self.verification_path = self.data_path / "trading" / "verification_log"
        self.verification_path.mkdir(parents=True, exist_ok=True)
        
        # Fichier parquet pour l'historique des vérifications
        self.verification_file = self.verification_path / "verification_history.parquet"
        
        logger.info("🔍 Service de vérification initialisé")
    
    def get_previous_decision(self, ticker: str) -> Optional[Dict]:
        """Récupère la dernière décision pour un ticker"""
        try:
            # Chercher le dernier fichier de décision
            decision_files = list(self.decisions_path.glob(f"decisions_*.json"))
            if not decision_files:
                return None
            
            # Trier par date de modification
            latest_file = max(decision_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                decisions = json.load(f)
            
            # Chercher la décision pour ce ticker
            for decision in decisions.get('decisions', []):
                if decision.get('ticker') == ticker:
                    return decision
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération décision précédente: {e}")
            return None
    
    def verify_decision(self, ticker: str, current_price: float, 
                       recommendation: str, fusion_score: float) -> Dict:
        """Vérifie une décision en comparant avec les données 15min précédentes"""
        try:
            # Récupérer la décision précédente
            prev_decision = self.get_previous_decision(ticker)
            
            if not prev_decision:
                return {
                    "status": "en_attente",
                    "message": "Aucune décision précédente",
                    "previous_recommendation": None,
                    "previous_price": None,
                    "price_change": None,
                    "accuracy": None
                }
            
            # Calculer le changement de prix
            prev_price = prev_decision.get('price', 0)
            if prev_price > 0:
                price_change = (current_price - prev_price) / prev_price * 100
            else:
                price_change = 0
            
            # Vérifier la cohérence
            prev_recommendation = prev_decision.get('recommendation', '')
            accuracy = self._calculate_accuracy(prev_recommendation, price_change)
            
            # Déterminer le statut
            if accuracy >= 0.7:
                status = "✅ Cohérent"
                color = "#28a745"
            elif accuracy >= 0.4:
                status = "⚠️ Partiellement cohérent"
                color = "#ffc107"
            else:
                status = "❌ Incohérent"
                color = "#dc3545"
            
            # Sauvegarder la vérification
            verification_data = {
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,
                "current_price": current_price,
                "previous_price": prev_price,
                "price_change": price_change,
                "current_recommendation": recommendation,
                "previous_recommendation": prev_recommendation,
                "fusion_score": fusion_score,
                "accuracy": accuracy,
                "status": status
            }
            
            self._save_verification(verification_data)
            
            return {
                "status": status,
                "message": f"Changement: {price_change:+.2f}%",
                "previous_recommendation": prev_recommendation,
                "previous_price": prev_price,
                "price_change": price_change,
                "accuracy": accuracy,
                "color": color
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification décision: {e}")
            return {
                "status": "❌ Erreur",
                "message": str(e),
                "previous_recommendation": None,
                "previous_price": None,
                "price_change": None,
                "accuracy": None,
                "color": "#dc3545"
            }
    
    def _calculate_accuracy(self, recommendation: str, price_change: float) -> float:
        """Calcule la précision d'une recommandation basée sur le changement de prix"""
        if not recommendation or price_change == 0:
            return 0.5  # Neutre si pas de données
        
        # Logique de cohérence
        if recommendation.upper() in ["ACHETER", "BUY"] and price_change > 0:
            return 1.0  # Parfait
        elif recommendation.upper() in ["VENDRE", "SELL"] and price_change < 0:
            return 1.0  # Parfait
        elif recommendation.upper() in ["ATTENDRE", "HOLD", "WAIT"]:
            return 0.7  # Bon pour HOLD
        elif abs(price_change) < 0.5:  # Changement minimal
            return 0.6  # Acceptable
        else:
            return 0.2  # Incohérent
    
    def _save_verification(self, data: Dict):
        """Sauvegarde une vérification dans le fichier parquet"""
        try:
            # Charger l'historique existant
            if self.verification_file.exists():
                df = pd.read_parquet(self.verification_file)
            else:
                df = pd.DataFrame()
            
            # Ajouter la nouvelle vérification
            new_row = pd.DataFrame([data])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Sauvegarder
            df.to_parquet(self.verification_file, index=False)
            logger.info(f"✅ Vérification sauvegardée: {data['ticker']} - {data['status']}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde vérification: {e}")
    
    def get_verification_history(self, ticker: str, days: int = 7) -> pd.DataFrame:
        """Récupère l'historique des vérifications pour un ticker"""
        try:
            if not self.verification_file.exists():
                return pd.DataFrame()
            
            df = pd.read_parquet(self.verification_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filtrer par ticker et période
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_df = df[
                (df['ticker'] == ticker) & 
                (df['timestamp'] >= cutoff_date)
            ].sort_values('timestamp')
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération historique: {e}")
            return pd.DataFrame()
    
    def get_accuracy_stats(self, ticker: str, days: int = 7) -> Dict:
        """Calcule les statistiques de précision pour un ticker"""
        try:
            df = self.get_verification_history(ticker, days)
            
            if df.empty:
                return {
                    "total_decisions": 0,
                    "average_accuracy": 0,
                    "coherent_decisions": 0,
                    "accuracy_rate": 0
                }
            
            total = len(df)
            avg_accuracy = df['accuracy'].mean()
            coherent = len(df[df['accuracy'] >= 0.7])
            accuracy_rate = coherent / total if total > 0 else 0
            
            return {
                "total_decisions": total,
                "average_accuracy": round(avg_accuracy, 3),
                "coherent_decisions": coherent,
                "accuracy_rate": round(accuracy_rate, 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul statistiques: {e}")
            return {
                "total_decisions": 0,
                "average_accuracy": 0,
                "coherent_decisions": 0,
                "accuracy_rate": 0
            }
