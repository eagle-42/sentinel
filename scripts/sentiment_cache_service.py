#!/usr/bin/env python3
"""
🚀 Service de Cache FinBERT Optimisé
Service de scoring de sentiment avec cache persistant et optimisations
"""

import asyncio
import hashlib
import json
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel


@dataclass
class CacheEntry:
    """Entrée de cache pour les scores de sentiment"""

    scores: List[float]
    timestamp: datetime
    text_hash: str


class ScoreRequest(BaseModel):
    texts: List[str]
    use_cache: bool = True
    cache_ttl_hours: int = 24


class ScoreResponse(BaseModel):
    scores: List[float]
    metrics: Dict
    cache_hits: int
    cache_misses: int


class SentimentCacheService:
    """Service de cache pour FinBERT avec optimisations"""

    def __init__(self):
        self.model = None
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_dir = Path("data/cache/sentiment")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_texts_processed": 0,
            "avg_latency_ms": 0.0,
        }

    def _get_text_hash(self, text: str) -> str:
        """Génère un hash pour le texte"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _is_cache_valid(self, entry: CacheEntry, ttl_hours: int) -> bool:
        """Vérifie si l'entrée de cache est valide"""
        return datetime.now() - entry.timestamp < timedelta(hours=ttl_hours)

    def _load_cache(self):
        """Charge le cache depuis le disque"""
        cache_file = self.cache_dir / "sentiment_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        self.cache[key] = CacheEntry(
                            scores=entry_data["scores"],
                            timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                            text_hash=entry_data["text_hash"],
                        )
                logger.info(f"Cache chargé: {len(self.cache)} entrées")
            except Exception as e:
                logger.warning(f"Erreur chargement cache: {e}")

    def _save_cache(self):
        """Sauvegarde le cache sur le disque"""
        cache_file = self.cache_dir / "sentiment_cache.json"
        try:
            data = {}
            for key, entry in self.cache.items():
                data[key] = {
                    "scores": entry.scores,
                    "timestamp": entry.timestamp.isoformat(),
                    "text_hash": entry.text_hash,
                }
            with open(cache_file, "w") as f:
                json.dump(data, f)
            logger.debug(f"Cache sauvegardé: {len(self.cache)} entrées")
        except Exception as e:
            logger.error(f"Erreur sauvegarde cache: {e}")

    def _cleanup_cache(self, max_age_hours: int = 168):  # 7 jours par défaut
        """Nettoie le cache des entrées anciennes"""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [key for key, entry in self.cache.items() if entry.timestamp < cutoff]

        for key in to_remove:
            del self.cache[key]

        if to_remove:
            logger.info(f"Cache nettoyé: {len(to_remove)} entrées supprimées")

    async def load_model(self):
        """Charge le modèle FinBERT"""
        try:
            from src.core.sentiment import FinBertAnalyzer

            self.model = FinBertAnalyzer(mode="real")
            logger.info("Modèle FinBERT chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {e}")
            raise

    async def score_texts(
        self, texts: List[str], use_cache: bool = True, cache_ttl_hours: int = 24
    ) -> tuple[List[float], Dict]:
        """Score les textes avec cache"""
        start_time = time.time()
        cache_hits = 0
        cache_misses = 0
        scores = []
        texts_to_process = []
        text_indices = []

        # Vérifier le cache pour chaque texte
        for i, text in enumerate(texts):
            if use_cache:
                text_hash = self._get_text_hash(text)
                if text_hash in self.cache:
                    entry = self.cache[text_hash]
                    if self._is_cache_valid(entry, cache_ttl_hours):
                        scores.append(entry.scores[0])  # Un score par texte
                        cache_hits += 1
                        continue

            # Texte non trouvé en cache
            texts_to_process.append(text)
            text_indices.append(i)
            cache_misses += 1

        # Traiter les textes non cachés
        if texts_to_process and self.model:
            try:
                new_scores = self.model.score_texts(texts_to_process)

                # Mettre en cache les nouveaux scores
                for text, score in zip(texts_to_process, new_scores):
                    text_hash = self._get_text_hash(text)
                    self.cache[text_hash] = CacheEntry(scores=[score], timestamp=datetime.now(), text_hash=text_hash)

                # Insérer les scores dans l'ordre correct
                for i, score in zip(text_indices, new_scores):
                    scores.insert(i, score)

            except Exception as e:
                logger.error(f"Erreur scoring: {e}")
                # Fallback: scores neutres
                for i in text_indices:
                    scores.insert(i, 0.0)

        # Calculer les métriques
        duration_ms = (time.time() - start_time) * 1000
        self.metrics["total_requests"] += 1
        self.metrics["cache_hits"] += cache_hits
        self.metrics["cache_misses"] += cache_misses
        self.metrics["total_texts_processed"] += len(texts)

        # Mettre à jour la latence moyenne
        total_requests = self.metrics["total_requests"]
        current_avg = self.metrics["avg_latency_ms"]
        self.metrics["avg_latency_ms"] = (current_avg * (total_requests - 1) + duration_ms) / total_requests

        # Sauvegarder le cache périodiquement
        if total_requests % 10 == 0:
            self._save_cache()

        # Nettoyer le cache périodiquement
        if total_requests % 100 == 0:
            self._cleanup_cache()

        metrics = {
            "duration_ms": duration_ms,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": cache_hits / len(texts) if texts else 0,
            "avg_latency_ms": self.metrics["avg_latency_ms"],
            "total_requests": self.metrics["total_requests"],
        }

        return scores, metrics


# Instance globale du service
_cache_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Événements de démarrage et arrêt"""
    global _cache_service

    # Démarrage
    logger.info("Démarrage du service de cache FinBERT...")
    try:
        _cache_service = SentimentCacheService()
        _cache_service.load_cache()
        await _cache_service.load_model()

        # Échauffement
        dummy_texts = ["NVIDIA stock is performing well"]
        await _cache_service.score_texts(dummy_texts)

        logger.info("Service de cache FinBERT prêt!")
        yield

    except Exception as e:
        logger.error(f"Échec du démarrage du service: {e}")
        raise
    finally:
        # Arrêt
        logger.info("Arrêt du service de cache FinBERT...")
        if _cache_service:
            _cache_service._save_cache()
        _cache_service = None


app = FastAPI(
    title="Sentiment Cache Service",
    description="Service de cache FinBERT optimisé pour Sentinel2",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Endpoint de vérification de santé"""
    return {
        "status": "healthy",
        "model_loaded": _cache_service is not None and _cache_service.model is not None,
        "cache_entries": len(_cache_service.cache) if _cache_service else 0,
        "metrics": _cache_service.metrics if _cache_service else {},
    }


@app.post("/score", response_model=ScoreResponse)
async def score_texts(request: ScoreRequest):
    """Score les textes en utilisant le cache"""
    if _cache_service is None:
        raise HTTPException(status_code=503, detail="Service non initialisé")

    try:
        scores, metrics = await _cache_service.score_texts(request.texts, request.use_cache, request.cache_ttl_hours)

        return ScoreResponse(
            scores=scores, metrics=metrics, cache_hits=metrics["cache_hits"], cache_misses=metrics["cache_misses"]
        )

    except Exception as e:
        logger.error(f"Erreur de scoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
async def get_cache_stats():
    """Retourne les statistiques du cache"""
    if _cache_service is None:
        raise HTTPException(status_code=503, detail="Service non initialisé")

    return {
        "cache_entries": len(_cache_service.cache),
        "metrics": _cache_service.metrics,
        "cache_size_mb": sum(len(str(entry)) for entry in _cache_service.cache.values()) / 1024 / 1024,
    }


@app.post("/cache/clear")
async def clear_cache():
    """Vide le cache"""
    if _cache_service is None:
        raise HTTPException(status_code=503, detail="Service non initialisé")

    _cache_service.cache.clear()
    _cache_service._save_cache()

    return {"message": "Cache vidé avec succès"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")  # Port différent du service principal
