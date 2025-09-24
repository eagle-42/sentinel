#!/usr/bin/env python3
"""
💭 Service de sentiment persistant
Service optimisé pour le scoring de sentiment avec cache et chargement persistant du modèle
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
import threading
import queue
import requests
from flask import Flask, request, jsonify

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.constants import CONSTANTS
from src.core.sentiment import FinBertAnalyzer


class SentimentCache:
    """Cache intelligent pour les scores de sentiment"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_times = {}
        
    def _is_expired(self, key: str) -> bool:
        """Vérifie si une entrée du cache est expirée"""
        if key not in self.access_times:
            return True
        
        age = time.time() - self.access_times[key]
        return age > self.ttl_seconds
    
    def _evict_oldest(self):
        """Supprime l'entrée la plus ancienne du cache"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self.cache.pop(oldest_key, None)
        self.access_times.pop(oldest_key, None)
    
    def get(self, text: str) -> Optional[float]:
        """Récupère un score depuis le cache"""
        key = text.lower().strip()
        
        if key in self.cache and not self._is_expired(key):
            self.access_times[key] = time.time()
            return self.cache[key]
        
        return None
    
    def set(self, text: str, score: float):
        """Ajoute un score au cache"""
        key = text.lower().strip()
        
        # Évincer si nécessaire
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_oldest()
        
        self.cache[key] = score
        self.access_times[key] = time.time()
    
    def clear(self):
        """Vide le cache"""
        self.cache.clear()
        self.access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1),
            "ttl_seconds": self.ttl_seconds
        }


class SentimentService:
    """Service de sentiment persistant avec cache et API REST"""
    
    def __init__(self, mode: str = "stub", port: int = 5001):
        self.mode = mode
        self.port = port
        self.cache = SentimentCache(max_size=1000, ttl_seconds=3600)
        self.analyzer = None
        self.app = Flask(__name__)
        self.running = False
        
        # Métriques
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_texts_processed": 0,
            "average_processing_time": 0.0,
            "start_time": datetime.now().isoformat()
        }
        
        # Configuration de l'API
        self._setup_api()
        
        # Chargement du modèle
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle FinBERT de manière persistante"""
        try:
            logger.info(f"💭 Chargement du modèle FinBERT (mode: {self.mode})")
            self.analyzer = FinBertAnalyzer(mode=self.mode)
            
            # Test du modèle
            test_texts = ["NVIDIA stock is performing well", "Market crash expected"]
            test_scores = self.analyzer.score_texts(test_texts)
            
            logger.info(f"✅ Modèle FinBERT chargé et testé: {len(test_scores)} scores")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle FinBERT: {e}")
            raise
    
    def _setup_api(self):
        """Configure l'API REST"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Endpoint de santé"""
            return jsonify({
                "status": "healthy",
                "mode": self.mode,
                "uptime": (datetime.now() - datetime.fromisoformat(self.metrics["start_time"])).total_seconds(),
                "metrics": self.metrics
            })
        
        @self.app.route('/score', methods=['POST'])
        def score_texts():
            """Endpoint de scoring de sentiment"""
            try:
                data = request.get_json()
                
                if not data or 'texts' not in data:
                    return jsonify({"error": "Missing 'texts' field"}), 400
                
                texts = data['texts']
                if not isinstance(texts, list):
                    return jsonify({"error": "'texts' must be a list"}), 400
                
                # Traitement
                start_time = time.time()
                scores = self.score_texts(texts)
                processing_time = time.time() - start_time
                
                # Mise à jour des métriques
                self.metrics["total_requests"] += 1
                self.metrics["total_texts_processed"] += len(texts)
                self.metrics["average_processing_time"] = (
                    (self.metrics["average_processing_time"] * (self.metrics["total_requests"] - 1) + processing_time) 
                    / self.metrics["total_requests"]
                )
                
                return jsonify({
                    "scores": scores,
                    "processing_time": processing_time,
                    "cache_stats": self.cache.stats()
                })
                
            except Exception as e:
                logger.error(f"❌ Erreur API scoring: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/stats', methods=['GET'])
        def stats():
            """Endpoint de statistiques"""
            return jsonify({
                "metrics": self.metrics,
                "cache_stats": self.cache.stats(),
                "model_mode": self.mode
            })
        
        @self.app.route('/cache/clear', methods=['POST'])
        def clear_cache():
            """Endpoint pour vider le cache"""
            self.cache.clear()
            return jsonify({"message": "Cache cleared"})
    
    def score_texts(self, texts: List[str]) -> List[float]:
        """Score une liste de textes avec cache"""
        if not texts:
            return []
        
        scores = []
        texts_to_process = []
        text_indices = []
        
        # Vérifier le cache
        for i, text in enumerate(texts):
            cached_score = self.cache.get(text)
            if cached_score is not None:
                scores.append(cached_score)
                self.metrics["cache_hits"] += 1
            else:
                scores.append(None)  # Placeholder
                texts_to_process.append(text)
                text_indices.append(i)
                self.metrics["cache_misses"] += 1
        
        # Traiter les textes non cachés
        if texts_to_process:
            try:
                new_scores = self.analyzer.score_texts(texts_to_process)
                
                # Mettre à jour les scores et le cache
                for i, (text, score) in enumerate(zip(texts_to_process, new_scores)):
                    idx = text_indices[i]
                    scores[idx] = score
                    self.cache.set(text, score)
                
            except Exception as e:
                logger.error(f"❌ Erreur scoring: {e}")
                # Remplir avec des scores par défaut
                for i in text_indices:
                    scores[i] = 0.0
        
        return scores
    
    def start_server(self):
        """Démarre le serveur API"""
        try:
            logger.info(f"🚀 Démarrage du service de sentiment sur le port {self.port}")
            self.running = True
            
            # Démarrer le serveur Flask
            self.app.run(
                host='0.0.0.0',
                port=self.port,
                debug=False,
                threaded=True
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur démarrage serveur: {e}")
            raise
    
    def stop_server(self):
        """Arrête le serveur"""
        logger.info("🛑 Arrêt du service de sentiment")
        self.running = False
    
    def save_metrics(self):
        """Sauvegarde les métriques"""
        metrics_path = CONSTANTS.DATA_ROOT / "logs" / "sentiment_service_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.debug(f"💾 Métriques sauvegardées: {metrics_path}")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde métriques: {e}")


def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Service de sentiment Sentinel2")
    parser.add_argument(
        "--mode",
        choices=["stub", "real"],
        default="stub",
        help="Mode FinBERT: stub ou real"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port du serveur API"
    )
    
    args = parser.parse_args()
    
    try:
        service = SentimentService(mode=args.mode, port=args.port)
        service.start_server()
        
    except KeyboardInterrupt:
        logger.info("🛑 Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()