"""Paramètres de configuration chargés depuis les variables d'environnement."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()


@dataclass
class Settings:
    """Paramètres d'application chargés depuis les variables d'environnement."""
    
    # Configuration FinBERT
    finbert_mode: str = "stub"
    finbert_timeout_ms: int = 20000
    
    # Configuration du flux de news
    news_flow_interval: int = 240  # 4 minutes en secondes
    
    # Feeds de news
    news_feeds: List[str] = None
    
    # Configuration NewsAPI
    newsapi_enabled: bool = False
    newsapi_key: str = None
    newsapi_sources: List[str] = None
    newsapi_language: str = "en"
    newsapi_country: str = "us"
    
    
    # Mots-clés pour détection de ticker
    nvda_keywords: List[str] = None
    spy_keywords: List[str] = None
    
    def __post_init__(self):
        """Post-initialisation pour gérer le parsing de listes depuis l'environnement."""
        if self.news_feeds is None:
            feeds_str = os.getenv("NEWS_FEEDS", 
                "https://www.investing.com/rss/news_25.rss,"
                "https://seekingalpha.com/feed.xml,"
                "https://feeds.bloomberg.com/markets/news.rss"
            )
            self.news_feeds = [feed.strip() for feed in feeds_str.split(",") if feed.strip()]
        
        if self.nvda_keywords is None:
            keywords_str = os.getenv("NVDA_KEYWORDS", r"\bNVIDIA\b,\bNVDA\b")
            self.nvda_keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
        
        if self.spy_keywords is None:
            keywords_str = os.getenv("SPY_KEYWORDS", 
                r"\bS&P\s*500\b,\bSPY\b,\bS&P500\b,"
                r"\bmarket\b,\bstocks\b,\bequity\b,\bindex\b"
            )
            self.spy_keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
        
        # Configuration NewsAPI
        if self.newsapi_key is None:
            self.newsapi_key = os.getenv("NEWSAPI_KEY", "")
        
        if self.newsapi_sources is None:
            sources_str = os.getenv("NEWSAPI_SOURCES", "reuters,bloomberg,associated-press,cnbc")
            self.newsapi_sources = [src.strip() for src in sources_str.split(",") if src.strip()]


def get_settings() -> Settings:
    """Récupère les paramètres d'application depuis les variables d'environnement."""
    return Settings(
        finbert_mode=os.getenv("FINBERT_MODE", "stub"),
        finbert_timeout_ms=int(os.getenv("FINBERT_TIMEOUT_MS", "20000")),
        news_flow_interval=int(os.getenv("NEWS_FLOW_INTERVAL", "240")),
        newsapi_enabled=os.getenv("NEWSAPI_ENABLED", "false").lower() == "true",
        newsapi_language=os.getenv("NEWSAPI_LANGUAGE", "en"),
        newsapi_country=os.getenv("NEWSAPI_COUNTRY", "us"),
    )
