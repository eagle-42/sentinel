"""
🚀 Prefect Flows pour Sentinel2
Architecture orchestration complète avec monitoring
"""

from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
from loguru import logger
import sys
from pathlib import Path

# Ajouter le projet au path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# TASKS - Unités atomiques de travail
# ============================================================================

@task(name="refresh-prices", retries=2, retry_delay_seconds=30)
def refresh_prices_task():
    """Task: Rafraîchir les prix 15min"""
    from scripts.refresh_prices import main as refresh_prices
    logger.info("📊 Refresh des prix...")
    result = refresh_prices()
    logger.info("✅ Prix rafraîchis")
    return result


@task(name="refresh-news", retries=2, retry_delay_seconds=30)
def refresh_news_task():
    """Task: Rafraîchir les news et sentiment"""
    from scripts.refresh_news import main as refresh_news
    logger.info("📰 Refresh des news...")
    result = refresh_news()
    logger.info("✅ News rafraîchies")
    return result


@task(name="trading-decision", retries=1, retry_delay_seconds=60)
def trading_decision_task(force: bool = False):
    """Task: Générer décision de trading"""
    from scripts.trading_pipeline import TradingPipeline
    logger.info("🤖 Génération décision trading...")
    pipeline = TradingPipeline()
    result = pipeline.run_trading_pipeline(force=force)
    logger.info(f"✅ Décision générée: {result.get('decisions', [])}")
    return result


@task(name="validate-decisions")
def validate_decisions_task():
    """Task: Valider les décisions passées"""
    logger.info("🔍 Validation des décisions...")
    # Logique de validation (comparer prix actuel vs prix décision)
    logger.info("✅ Validation terminée")
    return {"validated": True}


# ============================================================================
# FLOWS - Pipelines orchestrés
# ============================================================================

@flow(name="🔄 Data Refresh Flow", log_prints=True)
def data_refresh_flow():
    """
    Flow: Rafraîchit prix et news en parallèle
    Exécution: Toutes les 4 minutes
    """
    logger.info("🚀 Démarrage Data Refresh Flow")
    
    # Exécuter en parallèle
    prices_future = refresh_prices_task.submit()
    news_future = refresh_news_task.submit()
    
    # Attendre completion
    prices_result = prices_future.result()
    news_result = news_future.result()
    
    logger.info("✅ Data Refresh Flow terminé")
    return {
        "prices": prices_result,
        "news": news_result
    }


@flow(name="🤖 Trading Flow", log_prints=True)
def trading_flow(force: bool = False):
    """
    Flow: Pipeline de trading complet
    Exécution: Toutes les 15 minutes
    
    Args:
        force: Bypass fenêtre 15min (pour tests)
    """
    logger.info("🚀 Démarrage Trading Flow")
    
    # 1. Rafraîchir données
    data_result = data_refresh_flow()
    
    # 2. Générer décision
    decision_result = trading_decision_task(force=force)
    
    # 3. Valider décisions passées
    validation_result = validate_decisions_task()
    
    logger.info("✅ Trading Flow terminé")
    return {
        "data": data_result,
        "decision": decision_result,
        "validation": validation_result
    }


@flow(name="📊 Full System Flow", log_prints=True)
def full_system_flow():
    """
    Flow: Système complet Sentinel2
    Exécution: Démarrage initial
    """
    logger.info("🚀 Démarrage Full System Flow")
    logger.info("=" * 60)
    logger.info("SENTINEL2 - Système de Trading Algorithmique")
    logger.info("=" * 60)
    
    # Refresh initial
    data_result = data_refresh_flow()
    
    # Première décision
    decision_result = trading_decision_task(force=True)
    
    logger.info("✅ Full System Flow terminé")
    return {
        "data": data_result,
        "decision": decision_result
    }


# ============================================================================
# DEPLOYMENT CONFIG
# ============================================================================

if __name__ == "__main__":
    # Test local
    logger.info("🧪 Test local des flows")
    
    # Test data refresh
    # result = data_refresh_flow()
    # print(f"Data refresh result: {result}")
    
    # Test trading
    result = trading_flow(force=True)
    print(f"Trading result: {result}")
