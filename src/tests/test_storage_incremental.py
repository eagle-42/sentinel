#!/usr/bin/env python3
"""
Script de test pour la sauvegarde incrémentale
Teste sur données réelles SANS casser les données de production
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
from loguru import logger

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.constants import CONSTANTS
from src.data.storage import ParquetStorage
import yfinance as yf


class IncrementalStorageTest:
    """Test de sauvegarde incrémentale SAFE"""
    
    def __init__(self):
        self.storage = ParquetStorage()
        self.test_marker = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.test_rows_added = []
        
        logger.info("🧪 Test de sauvegarde incrémentale initialisé")
        logger.info(f"🏷️ Marqueur de test: {self.test_marker}")
    
    def test_prices_incremental(self) -> bool:
        """Teste la sauvegarde incrémentale des prix"""
        try:
            ticker = "SPY"
            logger.info(f"\n1️⃣ TEST PRIX - {ticker}")
            logger.info("=" * 60)
            
            # Charger données existantes
            file_path = CONSTANTS.get_data_path("prices", ticker, "15min")
            existing_count = 0
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                existing_count = len(existing_df)
                logger.info(f"📊 Données existantes: {existing_count} lignes")
                logger.info(f"📅 Période: {existing_df['ts_utc'].min()} → {existing_df['ts_utc'].max()}")
            else:
                logger.info("📊 Aucune donnée existante")
            
            # Créer données de TEST manuellement (marché fermé = pas de yfinance)
            logger.info(f"\n📥 Création données TEST manuelles...")
            now = datetime.now(timezone.utc)
            test_data = pd.DataFrame([
                {
                    'ts_utc': now,
                    'open': 450.00,
                    'high': 451.50,
                    'low': 449.50,
                    'close': 450.75,
                    'volume': 1000000,
                    'ticker': ticker,
                    'test_marker': self.test_marker
                },
                {
                    'ts_utc': now - timedelta(minutes=15),
                    'open': 449.50,
                    'high': 450.00,
                    'low': 449.00,
                    'close': 449.75,
                    'volume': 950000,
                    'ticker': ticker,
                    'test_marker': self.test_marker
                }
            ])
            
            if test_data.empty:
                logger.error("❌ Erreur création données test")
                return False
            
            logger.info(f"✅ Données TEST: {len(test_data)} lignes")
            logger.info(f"📅 Période TEST: {test_data['ts_utc'].min()} → {test_data['ts_utc'].max()}")
            
            # SAUVEGARDER (incrémental)
            logger.info(f"\n💾 Sauvegarde incrémentale...")
            result_path = self.storage.save_prices(test_data, ticker, "15min")
            
            # Vérifier résultat
            final_df = pd.read_parquet(file_path)
            final_count = len(final_df)
            test_rows = len(final_df[final_df.get('test_marker') == self.test_marker])
            
            logger.info(f"\n✅ RÉSULTAT:")
            logger.info(f"   Avant: {existing_count} lignes")
            logger.info(f"   Ajoutées: {test_rows} lignes TEST")
            logger.info(f"   Après: {final_count} lignes")
            logger.info(f"   Fichier: {result_path}")
            
            # Sauvegarder info pour nettoyage
            self.test_rows_added.append({
                "type": "prices",
                "ticker": ticker,
                "interval": "15min",
                "rows": test_rows,
                "file_path": file_path
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur test prix: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_news_incremental(self) -> bool:
        """Teste la sauvegarde incrémentale des news"""
        try:
            logger.info(f"\n2️⃣ TEST NEWS")
            logger.info("=" * 60)
            
            # Charger données existantes
            file_path = CONSTANTS.NEWS_DIR / "all_news.parquet"
            existing_count = 0
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                existing_count = len(existing_df)
                logger.info(f"📊 News existantes: {existing_count} lignes")
            else:
                logger.info("📊 Aucune news existante")
            
            # Créer données de TEST
            test_news = pd.DataFrame([
                {
                    "title": f"TEST Article 1 - {self.test_marker}",
                    "summary": "Test summary 1",
                    "body": "Test body 1",
                    "link": f"https://test.com/article1_{self.test_marker}",
                    "timestamp": datetime.now(timezone.utc),
                    "source": "TEST_SOURCE",
                    "ticker": "SPY",
                    "test_marker": self.test_marker
                },
                {
                    "title": f"TEST Article 2 - {self.test_marker}",
                    "summary": "Test summary 2",
                    "body": "Test body 2",
                    "link": f"https://test.com/article2_{self.test_marker}",
                    "timestamp": datetime.now(timezone.utc) - timedelta(minutes=5),
                    "source": "TEST_SOURCE",
                    "ticker": "SPY",
                    "test_marker": self.test_marker
                }
            ])
            
            logger.info(f"✅ Données TEST: {len(test_news)} articles")
            
            # SAUVEGARDER (incrémental)
            logger.info(f"\n💾 Sauvegarde incrémentale...")
            result_path = self.storage.save_news(test_news)
            
            # Vérifier résultat
            final_df = pd.read_parquet(file_path)
            final_count = len(final_df)
            test_rows = len(final_df[final_df.get('test_marker') == self.test_marker])
            
            logger.info(f"\n✅ RÉSULTAT:")
            logger.info(f"   Avant: {existing_count} lignes")
            logger.info(f"   Ajoutées: {test_rows} lignes TEST")
            logger.info(f"   Après: {final_count} lignes")
            logger.info(f"   Fichier: {result_path}")
            
            # Sauvegarder info pour nettoyage
            self.test_rows_added.append({
                "type": "news",
                "rows": test_rows,
                "file_path": file_path
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur test news: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_sentiment_incremental(self) -> bool:
        """Teste la sauvegarde incrémentale du sentiment"""
        try:
            ticker = "SPY"
            logger.info(f"\n3️⃣ TEST SENTIMENT - {ticker}")
            logger.info("=" * 60)
            
            # Charger données existantes
            file_path = CONSTANTS.get_data_path("sentiment", ticker)
            existing_count = 0
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                existing_count = len(existing_df)
                logger.info(f"📊 Sentiment existant: {existing_count} lignes")
            else:
                logger.info("📊 Aucun sentiment existant")
            
            # Créer données de TEST
            test_sentiment = pd.DataFrame([
                {
                    "ticker": ticker,
                    "timestamp": datetime.now(timezone.utc),
                    "sentiment_score": 0.75,
                    "confidence": 0.85,
                    "article_count": 5,
                    "test_marker": self.test_marker
                },
                {
                    "ticker": ticker,
                    "timestamp": datetime.now(timezone.utc) - timedelta(minutes=10),
                    "sentiment_score": 0.60,
                    "confidence": 0.80,
                    "article_count": 3,
                    "test_marker": self.test_marker
                }
            ])
            
            logger.info(f"✅ Données TEST: {len(test_sentiment)} entrées")
            
            # SAUVEGARDER (incrémental)
            logger.info(f"\n💾 Sauvegarde incrémentale...")
            result_path = self.storage.save_sentiment(test_sentiment, ticker)
            
            # Vérifier résultat
            final_df = pd.read_parquet(file_path)
            final_count = len(final_df)
            test_rows = len(final_df[final_df.get('test_marker') == self.test_marker])
            
            logger.info(f"\n✅ RÉSULTAT:")
            logger.info(f"   Avant: {existing_count} lignes")
            logger.info(f"   Ajoutées: {test_rows} lignes TEST")
            logger.info(f"   Après: {final_count} lignes")
            logger.info(f"   Fichier: {result_path}")
            
            # Sauvegarder info pour nettoyage
            self.test_rows_added.append({
                "type": "sentiment",
                "ticker": ticker,
                "rows": test_rows,
                "file_path": file_path
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur test sentiment: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup_test_data(self) -> bool:
        """Nettoie UNIQUEMENT les données de test (pas toutes les données)"""
        try:
            logger.info(f"\n🧹 NETTOYAGE DES DONNÉES DE TEST")
            logger.info("=" * 60)
            logger.info(f"🏷️ Suppression des lignes avec marqueur: {self.test_marker}")
            
            for test_info in self.test_rows_added:
                file_path = test_info["file_path"]
                
                logger.info(f"\n📝 Nettoyage {test_info['type']}...")
                
                # Charger toutes les données
                df = pd.read_parquet(file_path)
                before_count = len(df)
                
                # Supprimer UNIQUEMENT les lignes de test
                if 'test_marker' in df.columns:
                    df_cleaned = df[df['test_marker'] != self.test_marker]
                    # Supprimer la colonne test_marker si elle existe
                    df_cleaned = df_cleaned.drop(columns=['test_marker'])
                else:
                    logger.warning(f"⚠️ Colonne test_marker non trouvée dans {file_path}")
                    df_cleaned = df
                
                after_count = len(df_cleaned)
                removed = before_count - after_count
                
                # Sauvegarder
                df_cleaned.to_parquet(file_path, index=False, compression='snappy')
                
                logger.info(f"✅ {test_info['type'].upper()}")
                logger.info(f"   Avant: {before_count} lignes")
                logger.info(f"   Supprimées: {removed} lignes TEST")
                logger.info(f"   Après: {after_count} lignes")
            
            logger.info(f"\n✅ Nettoyage terminé - Données de production préservées")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur nettoyage: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """Exécute tous les tests"""
        logger.info("\n" + "=" * 80)
        logger.info("🧪 TEST SAUVEGARDE INCRÉMENTALE - DONNÉES RÉELLES")
        logger.info("=" * 80)
        logger.info(f"⚠️ Les données de production ne seront PAS supprimées")
        logger.info(f"⚠️ Seules les lignes de test (marqueur: {self.test_marker}) seront nettoyées")
        logger.info("=" * 80)
        
        results = {}
        
        # Test 1: Prix
        results['prices'] = self.test_prices_incremental()
        
        # Test 2: News
        results['news'] = self.test_news_incremental()
        
        # Test 3: Sentiment
        results['sentiment'] = self.test_sentiment_incremental()
        
        # Résumé
        logger.info(f"\n" + "=" * 80)
        logger.info("📊 RÉSUMÉ DES TESTS")
        logger.info("=" * 80)
        for test_name, success in results.items():
            status = "✅ SUCCÈS" if success else "❌ ÉCHEC"
            logger.info(f"   {test_name.upper()}: {status}")
        
        all_success = all(results.values())
        
        if all_success:
            logger.info(f"\n✅ Tous les tests sont passés !")
        else:
            logger.warning(f"\n⚠️ Certains tests ont échoué")
        
        # Nettoyage
        logger.info(f"\n⏳ Nettoyage des données de test dans 3 secondes...")
        import time
        time.sleep(3)
        
        cleanup_success = self.cleanup_test_data()
        
        if cleanup_success:
            logger.info(f"\n✅ TEST COMPLET - Données de production intactes")
        else:
            logger.error(f"\n❌ Erreur nettoyage - Vérifier manuellement")
        
        return all_success and cleanup_success


def main():
    """Fonction principale"""
    tester = IncrementalStorageTest()
    success = tester.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
