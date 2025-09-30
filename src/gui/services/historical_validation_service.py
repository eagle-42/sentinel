"""
Service de validation historique des décisions de trading
Valide les décisions passées contre l'évolution réelle des prix
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

class HistoricalValidationService:
    """Service de validation historique des décisions de trading"""
    
    def __init__(self):
        self.data_path = CONSTANTS.get_data_path()
        self.decisions_path = self.data_path / "trading" / "decisions_log"
        self.validation_path = self.data_path / "trading" / "historical_validation"
        self.validation_path.mkdir(parents=True, exist_ok=True)
        
        # Fichier parquet pour l'historique des validations
        self.validation_file = self.validation_file = self.validation_path / "historical_validation_results.parquet"
        
        logger.info("🔍 Service de validation historique initialisé")
    
    def load_historical_decisions(self) -> pd.DataFrame:
        """Charge les VRAIES décisions historiques depuis les logs de trading"""
        try:
            # Charger les vraies décisions depuis le fichier de trading
            trading_decisions_file = self.data_path / "trading" / "decisions_log" / "trading_decisions.json"
            
            if not trading_decisions_file.exists():
                logger.warning("❌ Aucun fichier de décisions de trading trouvé")
                return pd.DataFrame()
            
            # Charger le JSON
            with open(trading_decisions_file, 'r') as f:
                decisions_data = json.load(f)
            
            if not decisions_data:
                logger.warning("❌ Aucune décision dans le fichier")
                return pd.DataFrame()
            
            # Convertir en DataFrame
            df = pd.DataFrame(decisions_data)
            
            # Convertir les timestamps
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
            
            # Ajouter les colonnes manquantes pour la compatibilité
            if 'fusion_score' not in df.columns:
                df['fusion_score'] = df.get('fused_signal', 0)
            
            logger.info(f"✅ {len(df)} VRAIES décisions chargées depuis le trading pipeline")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement vraies décisions: {e}")
            return pd.DataFrame()
    
    def load_historical_prices(self, ticker: str) -> pd.DataFrame:
        """Charge les données de prix historiques"""
        try:
            # Essayer d'abord les données 15min
            price_file = self.data_path / "realtime" / "prices" / f"{ticker.lower()}_15min.parquet"
            
            if price_file.exists():
                df = pd.read_parquet(price_file)
                if 'ts_utc' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['ts_utc'])
                elif 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
                logger.info(f"✅ {len(df)} lignes de prix chargées pour {ticker}")
                return df
            
            # Fallback sur les données historiques
            hist_file = self.data_path / "historical" / "yfinance" / f"{ticker}_1999_2025.parquet"
            if hist_file.exists():
                df = pd.read_parquet(hist_file)
                if 'date' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'])
                return df
            
            logger.warning(f"Aucune donnée de prix trouvée pour {ticker}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement prix {ticker}: {e}")
            return pd.DataFrame()
    
    def validate_historical_decisions(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """
        Valide les VRAIES décisions historiques contre l'évolution réelle des prix
        
        Args:
            ticker: Symbole de l'action (filtré depuis les vraies décisions)
            days: Nombre de jours à analyser (ignoré, utilise toutes les décisions)
            
        Returns:
            Dict contenant les résultats de validation des vraies décisions
        """
        try:
            # Charger les données
            decisions_df = self.load_historical_decisions()
            prices_df = self.load_historical_prices(ticker)
            
            if decisions_df.empty:
                return {
                    "status": "no_decisions",
                    "message": "Aucune décision historique trouvée",
                    "total_decisions": 0,
                    "validation_results": [],
                    "summary_stats": {}
                }
            
            if prices_df.empty:
                return {
                    "status": "no_prices",
                    "message": "Aucune donnée de prix trouvée",
                    "total_decisions": len(decisions_df),
                    "validation_results": [],
                    "summary_stats": {}
                }
            
            # Filtrer les décisions récentes
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            # S'assurer que les timestamps sont dans le même timezone
            if decisions_df['timestamp'].dt.tz is not None:
                cutoff_date = cutoff_date.astimezone(decisions_df['timestamp'].dt.tz)
            elif decisions_df['timestamp'].dt.tz is None:
                cutoff_date = cutoff_date.replace(tzinfo=None)
            recent_decisions = decisions_df[decisions_df['timestamp'] >= cutoff_date].copy()
            
            if recent_decisions.empty:
                return {
                    "status": "no_recent_decisions",
                    "message": f"Aucune décision récente trouvée (derniers {days} jours)",
                    "total_decisions": len(decisions_df),
                    "validation_results": [],
                    "summary_stats": {}
                }
            
            # Valider chaque décision
            validation_results = []
            
            for i, (_, decision) in enumerate(recent_decisions.iterrows()):
                timestamp = decision.get('timestamp')
                signal = decision.get('fused_signal', decision.get('fusion_score', 0.0))
                decision_type = decision.get('decision', decision.get('recommendation', 'HOLD'))
                confidence = decision.get('confidence', 0.0)
                
                if timestamp is None:
                    continue
                
                # Normaliser les timezones pour la comparaison
                if prices_df['ts_utc'].dt.tz is not None and timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=prices_df['ts_utc'].dt.tz)
                elif prices_df['ts_utc'].dt.tz is None and timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)
                elif prices_df['ts_utc'].dt.tz is not None and timestamp.tzinfo is not None:
                    # Convertir les deux en UTC pour la comparaison
                    timestamp = timestamp.astimezone(timezone.utc)
                    prices_df['ts_utc'] = prices_df['ts_utc'].dt.tz_convert(timezone.utc)
                
                # Calculer la différence de temps
                prices_df['time_diff'] = abs(prices_df['ts_utc'] - timestamp)
                closest_idx = prices_df['time_diff'].idxmin()
                closest_price_data = prices_df.iloc[closest_idx]
                
                current_price = closest_price_data['close']
                
                # Récupérer le prix 15 minutes plus tard depuis les vraies données
                future_price = self._get_future_price(prices_df, closest_price_data, timestamp)
                
                # Calculer le changement de prix
                price_change = (future_price - current_price) / current_price * 100
                
                # Évaluer si la décision était correcte
                is_correct = self._evaluate_decision_correctness(decision_type, price_change)
                accuracy = self._calculate_decision_accuracy(decision_type, price_change)
                
                # Déterminer le statut
                if accuracy >= 0.8:
                    status = "✅ Correct"
                elif accuracy >= 0.5:
                    status = "⚠️ Partiellement correct"
                else:
                    status = "❌ Incorrect"
                
                result = {
                    'index': i + 1,
                    'timestamp': timestamp,
                    'signal': signal,
                    'decision': decision_type,
                    'confidence': confidence,
                    'current_price': current_price,
                    'future_price': future_price,
                    'price_change': price_change,
                    'is_correct': is_correct,
                    'accuracy': accuracy,
                    'status': status
                }
                
                validation_results.append(result)
            
            # Calculer les statistiques globales
            total_decisions = len(validation_results)
            correct_decisions = sum(1 for r in validation_results if r['is_correct'])
            accuracy_rate = correct_decisions / total_decisions if total_decisions > 0 else 0
            avg_accuracy = sum(r['accuracy'] for r in validation_results) / total_decisions if total_decisions > 0 else 0
            
            # Statistiques par type de décision
            buy_decisions = [r for r in validation_results if r['decision'].upper() in ['BUY', 'ACHETER']]
            sell_decisions = [r for r in validation_results if r['decision'].upper() in ['SELL', 'VENDRE']]
            hold_decisions = [r for r in validation_results if r['decision'].upper() in ['HOLD', 'ATTENDRE']]
            
            buy_accuracy = sum(r['accuracy'] for r in buy_decisions) / len(buy_decisions) if buy_decisions else 0
            sell_accuracy = sum(r['accuracy'] for r in sell_decisions) / len(sell_decisions) if sell_decisions else 0
            
            summary_stats = {
                'total_decisions': total_decisions,
                'correct_decisions': correct_decisions,
                'accuracy_rate': accuracy_rate,
                'avg_accuracy': avg_accuracy,
                'buy_decisions': len(buy_decisions),
                'sell_decisions': len(sell_decisions),
                'hold_decisions': len(hold_decisions),
                'buy_accuracy': buy_accuracy,
                'sell_accuracy': sell_accuracy
            }
            
            # Sauvegarder les résultats
            self._save_validation_results(ticker, validation_results, summary_stats)
            
            return {
                "status": "success",
                "message": f"Validation terminée pour {total_decisions} décisions",
                "total_decisions": total_decisions,
                "validation_results": validation_results,
                "summary_stats": summary_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur validation historique: {e}")
            return {
                "status": "error",
                "message": f"Erreur lors de la validation: {str(e)}",
                "total_decisions": 0,
                "validation_results": [],
                "summary_stats": {}
            }
    
    def _get_future_price(self, prices_df: pd.DataFrame, current_price_data: pd.Series, timestamp: datetime) -> float:
        """Récupère le prix 15 minutes plus tard depuis les vraies données"""
        try:
            current_price = current_price_data['close']
            
            # Trouver l'index correspondant au timestamp de la décision
            # Chercher la barre la plus proche du timestamp de la décision
            prices_df['time_diff'] = abs((prices_df['ts_utc'] - timestamp).dt.total_seconds())
            current_idx = prices_df['time_diff'].idxmin()
            
            # Chercher le prix 15 minutes plus tard dans les vraies données
            # Les données sont en 15min, donc +1 = +15min
            future_idx = current_idx + 1
            
            if future_idx < len(prices_df):
                # Prix réel 15min plus tard
                future_price = prices_df.iloc[future_idx]['close']
                logger.info(f"📊 Prix réel: ${current_price:.2f} → ${future_price:.2f} (15min plus tard)")
                return future_price
            else:
                # Pas de données futures, utiliser le dernier prix disponible
                future_price = prices_df.iloc[-1]['close']
                logger.info(f"📊 Prix final: ${current_price:.2f} → ${future_price:.2f} (dernière donnée)")
                return future_price
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération prix futur: {e}")
            # Fallback: retourner le prix actuel (pas de simulation)
            return current_price_data['close']
    
    def _evaluate_decision_correctness(self, decision: str, price_change: float) -> bool:
        """Évalue si une décision était correcte basée sur l'évolution du prix"""
        decision_upper = decision.upper()
        
        if decision_upper in ['BUY', 'ACHETER']:
            return price_change > 0.2  # BUY correct si prix monte de plus de 0.2%
        elif decision_upper in ['SELL', 'VENDRE']:
            return price_change < -0.2  # SELL correct si prix baisse de plus de 0.2%
        else:
            # HOLD correct si le prix reste stable (±0.5%)
            return abs(price_change) <= 0.5
    
    def _calculate_decision_accuracy(self, decision: str, price_change: float) -> float:
        """Calcule la précision d'une décision basée sur l'évolution réelle du prix"""
        decision_upper = decision.upper()
        
        if decision_upper in ['BUY', 'ACHETER']:
            if price_change > 2.0:
                return 1.0  # Excellent (+2%+)
            elif price_change > 1.0:
                return 0.9  # Très bon (+1-2%)
            elif price_change > 0.5:
                return 0.8  # Bon (+0.5-1%)
            elif price_change > 0.2:
                return 0.6  # Correct (+0.2-0.5%)
            elif price_change > 0:
                return 0.4  # Partiellement correct (+0-0.2%)
            else:
                return 0.1  # Incorrect (prix baisse)
        elif decision_upper in ['SELL', 'VENDRE']:
            if price_change < -2.0:
                return 1.0  # Excellent (-2%-)
            elif price_change < -1.0:
                return 0.9  # Très bon (-1 à -2%)
            elif price_change < -0.5:
                return 0.8  # Bon (-0.5 à -1%)
            elif price_change < -0.2:
                return 0.6  # Correct (-0.2 à -0.5%)
            elif price_change < 0:
                return 0.4  # Partiellement correct (0 à -0.2%)
            else:
                return 0.1  # Incorrect (prix monte)
        else:
            # HOLD : correct si prix stable, moins bon si mouvement important
            if abs(price_change) <= 0.5:
                return 0.8  # Très bon (prix stable)
            elif abs(price_change) <= 1.0:
                return 0.6  # Bon (mouvement modéré)
            else:
                return 0.3  # Moyen (mouvement important)
    
    def _save_validation_results(self, ticker: str, validation_results: List[Dict], summary_stats: Dict):
        """Sauvegarde les résultats de validation"""
        try:
            # Créer un DataFrame des résultats
            df_results = pd.DataFrame(validation_results)
            
            # Ajouter les métadonnées
            df_results['ticker'] = ticker
            df_results['validation_date'] = datetime.now()
            
            # Sauvegarder
            df_results.to_parquet(self.validation_file, index=False)
            
            # Sauvegarder aussi les statistiques
            stats_file = self.validation_path / f"{ticker}_validation_stats.json"
            
            # Convertir les résultats pour la sérialisation JSON
            serializable_results = []
            for result in validation_results:
                serializable_result = {}
                for key, value in result.items():
                    if hasattr(value, 'strftime'):  # Timestamp
                        serializable_result[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                    elif hasattr(value, 'item'):  # numpy types
                        serializable_result[key] = value.item()
                    else:
                        serializable_result[key] = value
                serializable_results.append(serializable_result)
            
            with open(stats_file, 'w') as f:
                json.dump({
                    'ticker': ticker,
                    'validation_date': datetime.now().isoformat(),
                    'summary_stats': summary_stats,
                    'validation_results': serializable_results
                }, f, indent=2)
            
            logger.info(f"✅ Résultats de validation sauvegardés pour {ticker}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde résultats: {e}")
    
    def get_validation_summary(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Récupère un résumé de la validation historique"""
        try:
            # Essayer de charger les résultats sauvegardés
            stats_file = self.validation_path / f"{ticker}_validation_stats.json"
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                
                # Vérifier si les données sont récentes (moins de 1 heure)
                validation_date = datetime.fromisoformat(stats_data['validation_date'])
                if datetime.now() - validation_date < timedelta(hours=1):
                    return {
                        "status": "cached",
                        "message": "Résultats de validation récents",
                        "summary_stats": stats_data['summary_stats'],
                        "validation_results": stats_data.get('validation_results', [])
                    }
            
            # Sinon, recalculer
            validation_result = self.validate_historical_decisions(ticker, days)
            
            return {
                "status": validation_result["status"],
                "message": validation_result["message"],
                "summary_stats": validation_result["summary_stats"],
                "validation_results": validation_result["validation_results"]
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération résumé: {e}")
            return {
                "status": "error",
                "message": f"Erreur: {str(e)}",
                "summary_stats": {}
            }

