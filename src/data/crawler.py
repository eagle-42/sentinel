"""
🕷️ Crawling Temps Réel Sentinel2
Collecte de données prix et news en temps réel
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import feedparser
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from loguru import logger

from ..config import config
from ..constants import CONSTANTS


class PriceCrawler:
    """Crawler pour les données de prix"""
    
    def __init__(self):
        """Initialise le crawler de prix"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        logger.info("💰 Crawler de prix initialisé")
    
    def get_yahoo_prices(self, 
                        ticker: str, 
                        interval: str = None, 
                        period: str = None) -> pd.DataFrame:
        """Récupère les prix depuis Yahoo Finance"""
        try:
            interval = interval or CONSTANTS.PRICE_INTERVAL
            period = period or CONSTANTS.PRICE_PERIOD
            
            logger.info(f"💰 Récupération prix {ticker} - {interval}/{period}")
            
            # Télécharger les données
            stock = yf.Ticker(ticker)
            df = stock.history(interval=interval, period=period)
            
            if df.empty:
                logger.warning(f"⚠️ Aucune donnée pour {ticker}")
                return pd.DataFrame()
            
            # Normaliser les colonnes
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            
            # Ajouter timestamp UTC
            if 'datetime' in df.columns:
                df['ts_utc'] = pd.to_datetime(df['datetime']).dt.tz_convert('UTC')
            else:
                df['ts_utc'] = pd.Timestamp.now(tz='UTC')
            
            # Ajouter ticker
            df['ticker'] = ticker.upper()
            
            # Vérifier les colonnes requises
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"❌ Colonnes manquantes: {missing_cols}")
                return pd.DataFrame()
            
            logger.info(f"✅ {len(df)} lignes récupérées pour {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération prix {ticker}: {e}")
            return pd.DataFrame()
    
    def get_polygon_prices(self, 
                          ticker: str, 
                          api_key: str = None,
                          interval: str = "1min",
                          days: int = 1) -> pd.DataFrame:
        """Récupère les prix depuis Polygon API"""
        try:
            api_key = api_key or os.getenv("POLYGON_API_KEY")
            if not api_key:
                logger.warning("⚠️ Clé API Polygon non fournie")
                return pd.DataFrame()
            
            # Calculer les dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # URL de l'API
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{interval}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            
            params = {
                "apikey": api_key,
                "adjusted": "true",
                "sort": "asc"
            }
            
            logger.info(f"💰 Récupération Polygon {ticker} - {interval}")
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "OK" or not data.get("results"):
                logger.warning(f"⚠️ Aucune donnée Polygon pour {ticker}")
                return pd.DataFrame()
            
            # Convertir en DataFrame
            results = data["results"]
            df_data = []
            
            for item in results:
                df_data.append({
                    'ts_utc': pd.to_datetime(item['t'], unit='ms', utc=True),
                    'open': item['o'],
                    'high': item['h'],
                    'low': item['l'],
                    'close': item['c'],
                    'volume': item['v'],
                    'ticker': ticker.upper()
                })
            
            df = pd.DataFrame(df_data)
            logger.info(f"✅ {len(df)} lignes Polygon récupérées pour {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur Polygon {ticker}: {e}")
            return pd.DataFrame()

class NewsCrawler:
    """Crawler pour les données de news"""
    
    def __init__(self):
        """Initialise le crawler de news"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        logger.info("📰 Crawler de news initialisé")
    
    def get_rss_news(self, urls: List[str], max_items: int = 100) -> List[Dict[str, Any]]:
        """Récupère les news depuis les feeds RSS"""
        all_news = []
        
        for url in urls:
            try:
                logger.info(f"📰 Récupération RSS: {url}")
                
                # Parser le feed RSS
                feed = feedparser.parse(url)
                
                if feed.bozo:
                    logger.warning(f"⚠️ Feed RSS malformé: {url}")
                    continue
                
                # Extraire les articles
                for entry in feed.entries[:max_items]:
                    news_item = {
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': url,
                        'ticker': self._detect_ticker(entry.get('title', '') + ' ' + entry.get('summary', ''))
                    }
                    all_news.append(news_item)
                
                logger.info(f"✅ {len(feed.entries)} articles RSS récupérés de {url}")
                
            except Exception as e:
                logger.error(f"❌ Erreur RSS {url}: {e}")
                continue
        
        logger.info(f"📰 Total: {len(all_news)} articles RSS récupérés")
        return all_news
    
    def get_newsapi_news(self, 
                        api_key: str = None,
                        sources: str = None,
                        language: str = "en",
                        country: str = "us",
                        max_items: int = 100) -> List[Dict[str, Any]]:
        """Récupère les news depuis NewsAPI"""
        try:
            api_key = api_key or os.getenv("NEWSAPI_KEY")
            if not api_key:
                logger.warning("⚠️ Clé API NewsAPI non fournie")
                return []
            
            sources = sources or os.getenv("NEWSAPI_SOURCES", "reuters,bloomberg,associated-press,cnbc")
            
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                "apiKey": api_key,
                "sources": sources,
                "language": language,
                "country": country,
                "pageSize": min(max_items, 100)
            }
            
            logger.info(f"📰 Récupération NewsAPI: {sources}")
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") != "ok":
                logger.error(f"❌ Erreur NewsAPI: {data.get('message', 'Unknown error')}")
                return []
            
            articles = data.get("articles", [])
            news_items = []
            
            for article in articles:
                news_item = {
                    'title': article.get('title', ''),
                    'summary': article.get('description', ''),
                    'body': article.get('content', ''),
                    'link': article.get('url', ''),
                    'published': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', 'NewsAPI'),
                    'ticker': self._detect_ticker(article.get('title', '') + ' ' + article.get('description', ''))
                }
                news_items.append(news_item)
            
            logger.info(f"✅ {len(news_items)} articles NewsAPI récupérés")
            return news_items
            
        except Exception as e:
            logger.error(f"❌ Erreur NewsAPI: {e}")
            return []
    
    def _detect_ticker(self, text: str) -> Optional[str]:
        """Détecte le ticker dans un texte"""
        text_upper = text.upper()
        
        # Mots-clés pour chaque ticker
        ticker_keywords = {
            "SPY": ["SPY", "S&P 500", "SPDR", "SPDR S&P 500", "ETF", "index"],
            "NVDA": ["NVDA", "NVIDIA", "Nvidia", "nvidia", "GPU", "AI", "artificial intelligence"]
        }
        
        for ticker, keywords in ticker_keywords.items():
            if any(keyword in text_upper for keyword in keywords):
                return ticker
        
        return None

class DataCrawler:
    """Crawler principal pour toutes les données"""
    
    def __init__(self):
        """Initialise le crawler principal"""
        self.price_crawler = PriceCrawler()
        self.news_crawler = NewsCrawler()
        
        logger.info("🕷️ Crawler principal initialisé")
    
    def crawl_prices(self, 
                    tickers: List[str] = None,
                    interval: str = None,
                    period: str = None) -> Dict[str, pd.DataFrame]:
        """Crawl les prix pour tous les tickers"""
        tickers = tickers or CONSTANTS.TICKERS
        results = {}
        
        for ticker in tickers:
            logger.info(f"💰 Crawling prix {ticker}")
            
            # Essayer Yahoo Finance d'abord
            df = self.price_crawler.get_yahoo_prices(ticker, interval, period)
            
            if df.empty:
                # Essayer Polygon en fallback
                df = self.price_crawler.get_polygon_prices(ticker)
            
            if not df.empty:
                results[ticker] = df
                logger.info(f"✅ Prix {ticker}: {len(df)} lignes")
            else:
                logger.warning(f"⚠️ Aucun prix récupéré pour {ticker}")
        
        return results
    
    def crawl_news(self, 
                  rss_urls: List[str] = None,
                  newsapi_enabled: bool = None,
                  max_items: int = 100) -> List[Dict[str, Any]]:
        """Crawl les news depuis toutes les sources"""
        all_news = []
        
        rss_urls = rss_urls or CONSTANTS.NEWS_FEEDS
        newsapi_enabled = newsapi_enabled if newsapi_enabled is not None else config.get("news.newsapi_enabled", False)
        
        # RSS Feeds
        if rss_urls:
            logger.info("📰 Crawling RSS feeds")
            rss_news = self.news_crawler.get_rss_news(rss_urls, max_items)
            all_news.extend(rss_news)
        
        # NewsAPI
        if newsapi_enabled:
            logger.info("📰 Crawling NewsAPI")
            newsapi_news = self.news_crawler.get_newsapi_news(max_items=max_items)
            all_news.extend(newsapi_news)
        
        # Filtrer les doublons basés sur le titre
        seen_titles = set()
        unique_news = []
        
        for news in all_news:
            title = news.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(news)
        
        logger.info(f"📰 Total news uniques: {len(unique_news)}")
        return unique_news
    
    def crawl_all(self, 
                 tickers: List[str] = None,
                 price_interval: str = None,
                 price_period: str = None,
                 max_news_items: int = 100) -> Dict[str, Any]:
        """Crawl toutes les données (prix + news)"""
        logger.info("🕷️ Début du crawling complet")
        start_time = time.time()
        
        # Crawler les prix
        prices = self.crawl_prices(tickers, price_interval, price_period)
        
        # Crawler les news
        news = self.crawl_news(max_items=max_news_items)
        
        # Résultats
        results = {
            "prices": prices,
            "news": news,
            "crawl_time": time.time() - start_time,
            "total_tickers": len(prices),
            "total_news": len(news)
        }
        
        logger.info(f"✅ Crawling terminé en {results['crawl_time']:.2f}s")
        return results
