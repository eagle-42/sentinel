#!/usr/bin/env python3
"""
🌐 Google Finance Scraper
Récupère le prix SPY depuis Google Finance (plus stable que Yahoo)
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import time
import random

import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class GoogleFinanceScraper:
    """Scraper pour Google Finance"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
    
    def get_price_from_google_search(self, ticker: str) -> Optional[float]:
        """Récupère le prix depuis Google Search (SPY stock price)"""
        try:
            # URL de recherche Google pour le prix
            query = f"{ticker} stock price"
            url = f"https://www.google.com/search?q={query}"
            
            logger.info(f"🌐 Recherche Google: {query}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Google affiche le prix dans plusieurs éléments possibles
            # Chercher les patterns courants
            
            # Pattern 1: div avec classe contenant "price" ou "stock"
            price_divs = soup.find_all('div', {'class': lambda x: x and ('price' in x.lower() or 'stock' in x.lower())})
            for div in price_divs:
                text = div.get_text().strip()
                # Essayer d'extraire un nombre
                import re
                numbers = re.findall(r'\d+\.\d+', text)
                if numbers:
                    price = float(numbers[0])
                    if 500 < price < 1000:  # SPY est dans cette fourchette
                        logger.info(f"✅ Prix trouvé (pattern 1): ${price:.2f}")
                        return price
            
            # Pattern 2: span avec data-attrib
            spans = soup.find_all('span', {'data-attrib': True})
            for span in spans:
                text = span.get_text().strip()
                try:
                    price = float(text.replace(',', ''))
                    if 500 < price < 1000:
                        logger.info(f"✅ Prix trouvé (pattern 2): ${price:.2f}")
                        return price
                except:
                    continue
            
            # Pattern 3: Chercher "USD" suivi d'un nombre
            text_full = soup.get_text()
            import re
            usd_pattern = re.findall(r'USD\s*(\d+\.\d+)', text_full)
            if usd_pattern:
                price = float(usd_pattern[0])
                if 500 < price < 1000:
                    logger.info(f"✅ Prix trouvé (pattern 3): ${price:.2f}")
                    return price
            
            # Pattern 4: Chercher juste des nombres dans une fourchette réaliste
            all_numbers = re.findall(r'\b(\d{3}\.\d{2})\b', text_full)
            for num in all_numbers:
                price = float(num)
                if 600 < price < 700:  # SPY est autour de 665
                    logger.info(f"✅ Prix trouvé (pattern 4): ${price:.2f}")
                    return price
            
            logger.warning(f"⚠️ Prix non trouvé pour {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur Google search: {e}")
            return None
    
    def get_price_from_google_finance(self, ticker: str, exchange: str = "NYSEARCA") -> Optional[float]:
        """Récupère le prix depuis Google Finance direct"""
        try:
            # URL Google Finance
            url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"
            
            logger.info(f"🌐 Google Finance: {url}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Chercher le span avec jsname="L3mUVe" et classe "uRbih"
            price_span = soup.find('span', {'jsname': 'L3mUVe', 'class': 'uRbih'})
            if price_span:
                price_text = price_span.get_text().strip()
                # Nettoyer (remplacer la virgule européenne par un point)
                price_clean = price_text.replace(',', '.')
                price = float(price_clean)
                logger.info(f"✅ Prix Google Finance: ${price:.2f}")
                return price
            
            # Fallback 1: Chercher juste avec jsname
            price_span = soup.find('span', {'jsname': 'L3mUVe'})
            if price_span:
                price_text = price_span.get_text().strip()
                price_clean = price_text.replace(',', '.')
                price = float(price_clean)
                logger.info(f"✅ Prix Google Finance (jsname): ${price:.2f}")
                return price
            
            # Fallback 2: Chercher juste avec la classe
            price_span = soup.find('span', {'class': 'uRbih'})
            if price_span:
                price_text = price_span.get_text().strip()
                price_clean = price_text.replace(',', '.')
                # Vérifier que c'est un prix réaliste
                try:
                    price = float(price_clean)
                    if 500 < price < 1000:  # SPY est dans cette fourchette
                        logger.info(f"✅ Prix Google Finance (classe): ${price:.2f}")
                        return price
                except:
                    pass
            
            logger.warning(f"⚠️ Prix non trouvé sur Google Finance")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur Google Finance: {e}")
            return None
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Récupère le prix actuel (essaie plusieurs méthodes)"""
        # Méthode 1: Google Finance direct
        price = self.get_price_from_google_finance(ticker)
        if price:
            return price
        
        # Méthode 2: Google Search
        time.sleep(1)  # Éviter le rate limit
        price = self.get_price_from_google_search(ticker)
        if price:
            return price
        
        return None


def refresh_prices_google(ticker: str = "SPY") -> bool:
    """
    Récupère le prix actuel depuis Google et l'ajoute à l'historique
    """
    try:
        logger.info(f"🌐 Récupération prix Google pour {ticker}")
        
        scraper = GoogleFinanceScraper()
        
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
            'volume': 0,  # On n'a pas le volume
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
                # Mettre à jour la barre existante
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
        logger.error(f"❌ Erreur refresh Google: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Point d'entrée du script"""
    logger.info("🚀 Google Finance Scraper")
    logger.info("=" * 60)
    
    success = refresh_prices_google("SPY")
    
    if success:
        logger.info("\n✅ Récupération Google Finance terminée avec succès")
    else:
        logger.error("\n❌ Échec de la récupération Google Finance")
        sys.exit(1)


if __name__ == "__main__":
    main()
