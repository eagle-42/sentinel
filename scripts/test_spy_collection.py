#!/usr/bin/env python3
"""
Script de test pour la collecte d'articles SPY
"""

import pandas as pd
import feedparser
from pathlib import Path

def test_spy_keywords():
    """Teste la détection d'articles SPY avec les nouveaux mots-clés"""
    
    # Mots-clés étendus pour SPY
    spy_keywords = [
        'spy', 's&p', 's&p 500', 'sp500', 'sp-500', 's&p500', 's&p-500',
        'etf', 'index', 'market', 'stock market', 'equity market', 'wall street',
        'dow jones', 'nasdaq', 'broad market', 'market index', 'us market',
        'american market', 'stock index', 'market benchmark', 'market performance',
        'market sentiment', 'market outlook', 'market analysis', 'market trends',
        'market volatility', 'market rally', 'market decline', 'market correction',
        'bull market', 'bear market', 'market cap', 'market capitalization',
        'sector performance', 'market sector', 'financial market', 'securities market'
    ]
    
    # Feeds RSS
    rss_feeds = [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.investing.com/rss/news.rss"
    ]
    
    all_articles = []
    
    for feed_url in rss_feeds:
        print(f"📰 Test du feed: {feed_url}")
        try:
            feed = feedparser.parse(feed_url)
            print(f"   Articles trouvés: {len(feed.entries)}")
            
            for entry in feed.entries:
                title = entry.get('title', '').lower()
                summary = entry.get('summary', '').lower()
                text = f"{title} {summary}"
                
                # Vérifier si l'article contient des mots-clés SPY
                spy_mentions = []
                for keyword in spy_keywords:
                    if keyword.lower() in text:
                        spy_mentions.append(keyword)
                
                if spy_mentions:
                    all_articles.append({
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': feed_url,
                        'spy_keywords': ', '.join(spy_mentions),
                        'ticker': 'SPY'
                    })
                    print(f"   ✅ Article SPY: {entry.get('title', '')[:60]}...")
                    print(f"      Mots-clés: {', '.join(spy_mentions)}")
        
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    print(f"\n📊 Total articles SPY trouvés: {len(all_articles)}")
    
    if all_articles:
        # Sauvegarder les articles SPY
        df = pd.DataFrame(all_articles)
        output_path = Path("data/analysis/spy_articles_test.parquet")
        df.to_parquet(output_path, index=False)
        print(f"💾 Articles SPY sauvegardés: {output_path}")
        
        # Afficher les articles
        print("\n📋 ARTICLES SPY DÉTECTÉS:")
        for i, article in enumerate(all_articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   Mots-clés: {article['spy_keywords']}")
            print(f"   Source: {article['source']}")
            print()
    
    return all_articles

if __name__ == "__main__":
    print("🔍 Test de collecte d'articles SPY avec mots-clés étendus")
    print("=" * 60)
    articles = test_spy_keywords()
