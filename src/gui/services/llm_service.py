"""
Service LLM pour l'onglet Production
Synthèse LLM (Ollama: Mistral ou Phi3 mini)
"""

import requests
import json
from loguru import logger
from typing import Dict, Any, List, Optional
from datetime import datetime


class LLMService:
    """Service LLM pour la synthèse des décisions de trading"""
    
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.available = self._check_ollama_availability()
        
        if self.available:
            logger.info(f"✅ LLM Service initialisé avec {model}")
        else:
            logger.warning("⚠️ Ollama non disponible, utilisation du mode fallback")
    
    def _check_ollama_availability(self) -> bool:
        """Vérifie si Ollama est disponible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"⚠️ Ollama non accessible: {e}")
            return False
    
    def generate_trading_explanation(self, 
                                   fusion_data: Dict[str, Any],
                                   sentiment_data: Dict[str, Any],
                                   price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Génère une explication LLM de la décision de trading"""
        try:
            if not self.available:
                return self._generate_fallback_explanation(fusion_data, sentiment_data, price_data)
            
            # Construire le prompt
            prompt = self._build_trading_prompt(fusion_data, sentiment_data, price_data)
            
            # Appeler Ollama
            response = self._call_ollama(prompt)
            
            if response:
                return {
                    "explanation": response,
                    "confidence": 0.9,
                    "source": "LLM",
                    "model": self.model,
                    "timestamp": datetime.now()
                }
            else:
                return self._generate_fallback_explanation(fusion_data, sentiment_data, price_data)
                
        except Exception as e:
            logger.error(f"❌ Erreur génération LLM: {e}")
            return self._generate_fallback_explanation(fusion_data, sentiment_data, price_data)
    
    def generate_full_report(self, 
                           fusion_data: Dict[str, Any],
                           sentiment_data: Dict[str, Any],
                           price_data: Dict[str, Any],
                           prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Génère un rapport complet LLM"""
        try:
            if not self.available:
                return self._generate_fallback_report(fusion_data, sentiment_data, price_data, prediction_data)
            
            # Construire le prompt pour rapport complet
            prompt = self._build_full_report_prompt(fusion_data, sentiment_data, price_data, prediction_data)
            
            # Appeler Ollama
            response = self._call_ollama(prompt)
            
            if response:
                return {
                    "full_report": response,
                    "confidence": 0.9,
                    "source": "LLM",
                    "model": self.model,
                    "timestamp": datetime.now()
                }
            else:
                return self._generate_fallback_report(fusion_data, sentiment_data, price_data, prediction_data)
                
        except Exception as e:
            logger.error(f"❌ Erreur rapport complet LLM: {e}")
            return self._generate_fallback_report(fusion_data, sentiment_data, price_data, prediction_data)
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Appelle l'API Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.warning(f"⚠️ Erreur API Ollama: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur appel Ollama: {e}")
            return None
    
    def _build_trading_prompt(self, fusion_data: Dict, sentiment_data: Dict, price_data: Dict) -> str:
        """Construit le prompt pour l'explication de trading"""
        return f"""
Analysez cette situation de trading et fournissez une explication claire et concise :

DONNÉES DE FUSION:
- Score de fusion: {fusion_data.get('fusion_score', 0):.2f}
- Confiance: {fusion_data.get('confidence', 0):.2f}
- Recommandation: {fusion_data.get('recommendation', 'ATTENDRE')}

SENTIMENT DU MARCHÉ:
- Sentiment moyen: {sentiment_data.get('avg_sentiment', 0):.2f}
- Articles analysés: {sentiment_data.get('total_articles', 0)}
- Tendance: {sentiment_data.get('sentiment_trend', 'stable')}

PRIX:
- Dernier prix: ${price_data.get('last_price', 0):.2f}
- Variation: {price_data.get('change_percent', 0):.2f}%

Fournissez une explication en 2-3 phrases expliquant pourquoi cette recommandation est justifiée, en mettant l'accent sur les facteurs clés.
"""
    
    def _build_full_report_prompt(self, fusion_data: Dict, sentiment_data: Dict, 
                                price_data: Dict, prediction_data: Dict) -> str:
        """Construit le prompt pour le rapport complet"""
        return f"""
Générez un rapport d'analyse complet pour cette situation de trading :

DONNÉES DE FUSION:
- Score: {fusion_data.get('fusion_score', 0):.2f}
- Confiance: {fusion_data.get('confidence', 0):.2f}
- Recommandation: {fusion_data.get('recommendation', 'ATTENDRE')}
- Poids: {fusion_data.get('weights', {})}

SENTIMENT:
- Score moyen: {sentiment_data.get('avg_sentiment', 0):.2f}
- Articles: {sentiment_data.get('total_articles', 0)}
- Positifs: {sentiment_data.get('positive_count', 0)}
- Négatifs: {sentiment_data.get('negative_count', 0)}

PRIX:
- Prix: ${price_data.get('last_price', 0):.2f}
- Variation: {price_data.get('change_percent', 0):.2f}%

PRÉDICTION:
- Score: {prediction_data.get('prediction_score', 0):.2f}
- Confiance: {prediction_data.get('confidence', 0):.2f}

Générez un rapport détaillé incluant:
1. Résumé exécutif
2. Analyse des signaux
3. Facteurs de risque
4. Recommandation finale
5. Prochaines étapes
"""
    
    def _generate_fallback_explanation(self, fusion_data: Dict, sentiment_data: Dict, price_data: Dict) -> Dict[str, Any]:
        """Génère une explication de fallback"""
        recommendation = fusion_data.get('recommendation', 'ATTENDRE')
        score = fusion_data.get('fusion_score', 0.5)
        confidence = fusion_data.get('confidence', 0.7)
        
        if recommendation == "ACHETER":
            explanation = f"Recommandation d'ACHAT basée sur un score de fusion de {score:.2f} avec une confiance de {confidence:.1%}. Les signaux de prix et de sentiment sont favorables."
        elif recommendation == "VENDRE":
            explanation = f"Recommandation de VENTE basée sur un score de fusion de {score:.2f} avec une confiance de {confidence:.1%}. Les signaux indiquent une faiblesse du marché."
        else:
            explanation = f"Recommandation d'ATTENTE basée sur un score de fusion de {score:.2f} avec une confiance de {confidence:.1%}. Les signaux sont mitigés, attendre une confirmation."
        
        return {
            "explanation": explanation,
            "confidence": confidence,
            "source": "Fallback",
            "model": "Simulation",
            "timestamp": datetime.now()
        }
    
    def _generate_fallback_report(self, fusion_data: Dict, sentiment_data: Dict, 
                                price_data: Dict, prediction_data: Dict) -> Dict[str, Any]:
        """Génère un rapport de fallback"""
        report = f"""
# RAPPORT D'ANALYSE DE TRADING

## Résumé Exécutif
Recommandation: {fusion_data.get('recommendation', 'ATTENDRE')}
Score de fusion: {fusion_data.get('fusion_score', 0):.2f}
Confiance: {fusion_data.get('confidence', 0):.1%}

## Analyse des Signaux
- **Prix**: ${price_data.get('last_price', 0):.2f} ({price_data.get('change_percent', 0):.2f}%)
- **Sentiment**: {sentiment_data.get('avg_sentiment', 0):.2f} ({sentiment_data.get('total_articles', 0)} articles)
- **Prédiction**: {prediction_data.get('prediction_score', 0):.2f}

## Facteurs de Risque
- Confiance modérée des signaux
- Volatilité du marché
- Données limitées

## Recommandation Finale
{fusion_data.get('recommendation', 'ATTENDRE')} - Attendre une confirmation plus forte des signaux.

## Prochaines Étapes
1. Surveiller l'évolution des prix
2. Analyser de nouveaux articles
3. Réévaluer dans 1-2 heures
"""
        
        return {
            "full_report": report,
            "confidence": 0.7,
            "source": "Fallback",
            "model": "Simulation",
            "timestamp": datetime.now()
        }
