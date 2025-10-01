#!/usr/bin/env python3
"""
🕷️ Yahoo Finance Scraper
Scrape les données de prix directement depuis Yahoo Finance
Contourne les limitations des APIs gratuites
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
import time
import random

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from constants import CONSTANTS


class YahooFinanceScraper:
    """Scraper pour Yahoo Finance"""
    
    def __init__(self):
        self.base_url = "https://finance.yahoo.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Récupère le prix actuel via yfinance (plus fiable que scraping)"""
        try:
            import yfinance as yf
            
            logger.info(f"📊 Récupération prix actuel pour {ticker} via yfinance")
            
            # Utiliser l'API fast_info de yfinance qui est plus stable
            stock = yf.Ticker(ticker)
            
            # Essayer fast_info d'abord
            try:
                price = stock.fast_info['lastPrice']
                if price and price > 0:
                    logger.info(f"✅ Prix actuel {ticker}: ${price:.2f}")
                    return float(price)
            except:
                pass
            
            # Fallback: info dict
            try:
                info = stock.info
                price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
                if price and price > 0:
                    logger.info(f"✅ Prix actuel {ticker} (info): ${price:.2f}")
                    return float(price)
            except:
                pass
            
            # Dernier fallback: history 1 jour
            try:
                hist = stock.history(period='1d', interval='1m')
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    logger.info(f"✅ Prix actuel {ticker} (history): ${price:.2f}")
                    return float(price)
            except:
                pass
            
            logger.warning(f"⚠️ Impossible de récupérer le prix pour {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération prix {ticker}: {e}")
            return None
    
    def get_historical_data(self, ticker: str, period: str = "1d", interval: str = "15m") -> pd.DataFrame:
        """
        Récupère les données historiques via l'API JSON Yahoo cachée
        
        Args:
            ticker: Symbole du ticker (ex: SPY)
            period: Période (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Intervalle (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        """
        try:
            # Convertir period en timestamps Unix
            end_time = int(datetime.now().timestamp())
            
            period_map = {
                '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
                '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
            }
            days = period_map.get(period, 5)  # Par défaut 5 jours
            start_time = int((datetime.now() - timedelta(days=days)).timestamp())
            
            # Utiliser l'API JSON cachée de Yahoo Finance v8
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            params = {
                'period1': start_time,
                'period2': end_time,
                'interval': interval,
                'includePrePost': 'false',
                'events': 'div,splits'
            }
            
            logger.info(f"📊 Requête API Yahoo pour {ticker} ({period}, {interval})")
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data_json = response.json()
            
            # Vérifier si erreur
            if 'chart' not in data_json:
                logger.warning(f"⚠️ Format JSON invalide pour {ticker}")
                return pd.DataFrame()
            
            chart = data_json['chart']
            
            if 'error' in chart and chart['error']:
                logger.error(f"❌ Erreur API: {chart['error']}")
                return pd.DataFrame()
            
            if not chart.get('result'):
                logger.warning(f"⚠️ Aucun résultat pour {ticker}")
                return pd.DataFrame()
            
            result = chart['result'][0]
            
            # Extraire les données
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            data = []
            for i, ts in enumerate(timestamps):
                # Vérifier que les valeurs ne sont pas None
                if (quotes['close'][i] is not None and 
                    quotes['open'][i] is not None):
                    
                    data.append({
                        'ts_utc': datetime.fromtimestamp(ts, tz=timezone.utc),
                        'open': float(quotes['open'][i]),
                        'high': float(quotes['high'][i]),
                        'low': float(quotes['low'][i]),
                        'close': float(quotes['close'][i]),
                        'volume': int(quotes['volume'][i]) if quotes['volume'][i] else 0
                    })
            
            if not data:
                logger.warning(f"⚠️ Aucune donnée valide pour {ticker}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['ticker'] = ticker
            
            logger.info(f"✅ {len(df)} barres récupérées pour {ticker}")
            logger.info(f"📅 Période: {df['ts_utc'].min()} à {df['ts_utc'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur API Yahoo {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def scrape_with_retry(self, ticker: str, period: str = "1d", interval: str = "15m", 
                         max_retries: int = 3) -> pd.DataFrame:
        """Scrape avec retry et backoff"""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = 2 ** attempt + random.uniform(0, 1)
                    logger.info(f"🔄 Tentative {attempt + 1}/{max_retries} (attente {delay:.1f}s)")
                    time.sleep(delay)
                
                df = self.get_historical_data(ticker, period, interval)
                
                if not df.empty:
                    return df
                
            except Exception as e:
                logger.warning(f"⚠️ Tentative {attempt + 1} échouée: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"❌ Échec définitif après {max_retries} tentatives")
        
        return pd.DataFrame()


def refresh_prices_scraper(ticker: str = "SPY") -> bool:
    """
    Scrape UNIQUEMENT le prix actuel et l'ajoute à l'historique
    Approche simple : on construit notre propre historique 15min
    """
    try:
        logger.info(f"🕷️ Scraping prix actuel pour {ticker}")
        
        scraper = YahooFinanceScraper()
        
        # Récupérer le prix actuel
        current_price = scraper.get_current_price(ticker)
        
        if current_price is None:
            logger.error(f"❌ Impossible de récupérer le prix pour {ticker}")
            return False
        
        # Créer une barre pour maintenant
        now = datetime.now(timezone.utc)
        # Arrondir à la fenêtre 15min la plus proche
        minutes = (now.minute // 15) * 15
        ts_rounded = now.replace(minute=minutes, second=0, microsecond=0)
        
        new_bar = pd.DataFrame([{
            'ts_utc': ts_rounded,
            'open': current_price,
            'high': current_price,
            'low': current_price,
            'close': current_price,
            'volume': 0,  # On n'a pas le volume en scrapant
            'ticker': ticker
        }])
        
        logger.info(f"📊 Nouvelle barre: {ts_rounded} | ${current_price:.2f}")
        
        # Charger les données existantes
        file_path = Path(f"data/realtime/prices/{ticker.lower()}_15min.parquet")
        
        if file_path.exists():
            existing_data = pd.read_parquet(file_path)
            existing_data['ts_utc'] = pd.to_datetime(existing_data['ts_utc'])
            logger.info(f"📊 Données existantes: {len(existing_data)} lignes")
            
            # Vérifier si on a déjà cette timestamp
            if ts_rounded in existing_data['ts_utc'].values:
                # Mettre à jour la barre existante (close, high, low)
                idx = existing_data[existing_data['ts_utc'] == ts_rounded].index[0]
                existing_data.loc[idx, 'close'] = current_price
                existing_data.loc[idx, 'high'] = max(existing_data.loc[idx, 'high'], current_price)
                existing_data.loc[idx, 'low'] = min(existing_data.loc[idx, 'low'], current_price)
                combined = existing_data
                logger.info(f"🔄 Barre {ts_rounded} mise à jour")
            else:
                # Ajouter la nouvelle barre
                combined = pd.concat([existing_data, new_bar], ignore_index=True)
                combined = combined.sort_values('ts_utc').reset_index(drop=True)
                logger.info(f"➕ Nouvelle barre ajoutée")
        else:
            combined = new_bar
            logger.info(f"💾 Premier enregistrement pour {ticker}")
        
        # Sauvegarder
        file_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(file_path, index=False)
        
        logger.info(f"✅ Données sauvegardées: {file_path}")
        logger.info(f"📅 Total: {len(combined)} barres de {combined['ts_utc'].min()} à {combined['ts_utc'].max()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur refresh scraper: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Point d'entrée du script"""
    logger.info("🚀 Yahoo Finance Scraper")
    logger.info("=" * 60)
    
    success = refresh_prices_scraper("SPY")
    
    if success:
        logger.info("\n✅ Scraping terminé avec succès")
    else:
        logger.error("\n❌ Échec du scraping")
        sys.exit(1)


if __name__ == "__main__":
    main()
