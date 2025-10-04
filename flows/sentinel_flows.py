"""
Prefect Flows pour Sentinel2
Architecture orchestration complète avec monitoring
"""

from prefect import flow, task
from datetime import timedelta
from loguru import logger
import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@task(name="refresh-prices-15min", retries=2, retry_delay_seconds=30, log_prints=True)
def refresh_prices_15min_task():
    """Task: Rafraîchir les prix 15min (SPY) via Finnhub API"""
    logger.info("📊 Refresh prix 15min (Finnhub)...")
    try:
        result = subprocess.run(
            ["uv", "run", "python", "scripts/finnhub_scraper.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            logger.info("✅ Prix 15min rafraîchis (Finnhub)")
            return {"success": True, "output": result.stdout}
        else:
            logger.error(f"❌ Erreur refresh prix Finnhub: {result.stderr}")
            return {"success": False, "error": result.stderr}
    except Exception as e:
        logger.error(f"❌ Exception refresh prix: {e}")
        raise


@task(name="refresh-news-sentiment", retries=2, retry_delay_seconds=30, log_prints=True)
def refresh_news_task():
    """Task: Rafraîchir les news et sentiment (RSS + NewsAPI + FinBERT)"""
    logger.info("📰 Refresh news + sentiment...")
    try:
        result = subprocess.run(
            ["uv", "run", "python", "scripts/refresh_news.py"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            logger.info("✅ News + sentiment rafraîchis")
            return {"success": True, "output": result.stdout}
        else:
            logger.error(f"❌ Erreur refresh news: {result.stderr}")
            return {"success": False, "error": result.stderr}
    except Exception as e:
        logger.error(f"❌ Exception refresh news: {e}")
        raise


@task(name="trading-decision", retries=1, retry_delay_seconds=60, log_prints=True)
def trading_decision_task(force: bool = False):
    """Task: Générer décision de trading (LSTM + Fusion + Décision)"""
    logger.info("🤖 Génération décision trading...")
    try:
        cmd = ["uv", "run", "python", "scripts/trading_pipeline.py"]
        if force:
            cmd.append("--force")
        
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            logger.info("✅ Décision générée")
            return {"success": True, "output": result.stdout}
        else:
            logger.warning(f"⚠️ Warning trading: {result.stderr}")
            return {"success": True, "output": result.stdout, "warning": result.stderr}
    except Exception as e:
        logger.error(f"❌ Exception trading: {e}")
        raise


@flow(name="📊 Prix 15min Flow", log_prints=True)
def prices_15min_flow():
    """
    Flow: Rafraîchir prix 15min uniquement
    Exécution: Toutes les 15 minutes
    """
    logger.info("🚀 Démarrage Prix 15min Flow")
    result = refresh_prices_15min_task()
    logger.info("✅ Prix 15min Flow terminé")
    return result


@flow(name="📰 News + Sentiment Flow", log_prints=True)
def news_sentiment_flow():
    """
    Flow: Rafraîchir news + sentiment
    Exécution: Toutes les 4 minutes
    """
    logger.info("🚀 Démarrage News + Sentiment Flow")
    result = refresh_news_task()
    logger.info("✅ News + Sentiment Flow terminé")
    return result


@flow(name="🤖 Trading Flow", log_prints=True)
def trading_flow(force: bool = False):
    """
    Flow: Pipeline de trading complet
    Exécution: Toutes les 15 minutes (heures marché)
    
    Args:
        force: Bypass fenêtre 15min (pour tests)
    """
    logger.info("🚀 Démarrage Trading Flow")
    
    # 1. Rafraîchir prix
    prices_result = refresh_prices_15min_task()
    
    # 2. Rafraîchir news
    news_result = refresh_news_task()
    
    # 3. Générer décision
    decision_result = trading_decision_task(force=force)
    
    logger.info("✅ Trading Flow terminé")
    return {
        "prices": prices_result,
        "news": news_result,
        "decision": decision_result
    }


@flow(name="🚀 Full System Flow", log_prints=True)
def full_system_flow():
    """
    Flow: Système complet Sentinel2
    Exécution: Démarrage initial
    """
    logger.info("=" * 60)
    logger.info("🚀 SENTINEL2 - DÉMARRAGE COMPLET")
    logger.info("=" * 60)
    
    # 1. Refresh prix
    logger.info("\n1️⃣ Refresh prix 15min...")
    prices_result = refresh_prices_15min_task()
    
    # 2. Refresh news
    logger.info("\n2️⃣ Refresh news + sentiment...")
    news_result = refresh_news_task()
    
    # 3. Première décision
    logger.info("\n3️⃣ Première décision trading...")
    decision_result = trading_decision_task(force=True)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ SYSTÈME SENTINEL2 OPÉRATIONNEL")
    logger.info("=" * 60)
    
    return {
        "prices": prices_result,
        "news": news_result,
        "decision": decision_result
    }


if __name__ == "__main__":
    logger.info("🧪 Test local des flows Sentinel2")
    logger.info("=" * 60)
    
    # Test full system
    result = full_system_flow()
    print("\n📊 Résultat:")
    print(f"  Prices: {result['prices'].get('success', False)}")
    print(f"  News: {result['news'].get('success', False)}")
    print(f"  Decision: {result['decision'].get('success', False)}")
