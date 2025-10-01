"""
🚀 Déploiements Prefect pour Sentinel2
TOUS LES CRAWLERS configurés avec schedules optimaux
"""

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from sentinel_flows import (
    prices_15min_flow,
    news_sentiment_flow,
    trading_flow,
    historical_update_flow,
    full_system_flow
)


# SCHEDULES

# Prix 15min: Toutes les 15 minutes
prices_15min_schedule = CronSchedule(
    cron="*/15 * * * *",  # Toutes les 15min
    timezone="America/New_York"
)

# News + Sentiment: Toutes les 4 minutes (news refresh rapide)
news_sentiment_schedule = CronSchedule(
    cron="*/4 * * * *",  # Toutes les 4 minutes
    timezone="America/New_York"
)

# Trading: Toutes les 15 minutes (heures marché uniquement)
trading_schedule = CronSchedule(
    cron="*/15 9-16 * * 1-5",  # 9h-16h, Lun-Ven
    timezone="America/New_York"
)

# Historical: 1 fois par jour après fermeture marché
historical_schedule = CronSchedule(
    cron="30 16 * * 1-5",  # 16h30 ET, après fermeture
    timezone="America/New_York"
)


# DEPLOYMENTS

# 1. Prix 15min (toutes les 15min)
prices_15min_deployment = Deployment.build_from_flow(
    flow=prices_15min_flow,
    name="prices-15min-production",
    version="2.0.0",
    tags=["production", "prices", "15min"],
    schedule=prices_15min_schedule,
    work_pool_name="sentinel-pool",
    description="Rafraîchit prix 15min toutes les 15 minutes"
)

# 2. News + Sentiment (toutes les 4min)
news_sentiment_deployment = Deployment.build_from_flow(
    flow=news_sentiment_flow,
    name="news-sentiment-production",
    version="2.0.0",
    tags=["production", "news", "sentiment"],
    schedule=news_sentiment_schedule,
    work_pool_name="sentinel-pool",
    description="Rafraîchit news + sentiment toutes les 4 minutes"
)

# 3. Trading (15min pendant heures marché)
trading_deployment = Deployment.build_from_flow(
    flow=trading_flow,
    name="trading-production",
    version="2.0.0",
    tags=["production", "trading", "decisions"],
    schedule=trading_schedule,
    parameters={"force": False},
    work_pool_name="sentinel-pool",
    description="Pipeline trading 15min (heures marché US)"
)

# 4. Historical (1x/jour après clôture)
historical_deployment = Deployment.build_from_flow(
    flow=historical_update_flow,
    name="historical-daily",
    version="2.0.0",
    tags=["production", "historical", "daily"],
    schedule=historical_schedule,
    work_pool_name="sentinel-pool",
    description="Mise à jour prix historiques 1D (16h30 ET)"
)

# 5. Full System (manuel - démarrage)
full_system_deployment = Deployment.build_from_flow(
    flow=full_system_flow,
    name="full-system-startup",
    version="2.0.0",
    tags=["production", "startup", "manual"],
    work_pool_name="sentinel-pool",
    description="Démarrage complet système (manuel)"
)


# DEPLOYMENT REGISTRY

def deploy_all():
    """Déploie tous les flows (5 déploiements)"""
    deployments = [
        prices_15min_deployment,
        news_sentiment_deployment,
        trading_deployment,
        historical_deployment,
        full_system_deployment
    ]
    
    print("🚀 Déploiement des flows Sentinel2...")
    print("=" * 60)
    
    for i, deployment in enumerate(deployments, 1):
        deployment.apply()
        print(f"{i}. ✅ {deployment.name}")
    
    print("=" * 60)
    print("🎉 5 déploiements créés avec succès !")
    print("\n📊 Flows disponibles:")
    print("  1. Prix 15min       → Toutes les 15min")
    print("  2. News + Sentiment → Toutes les 4min")
    print("  3. Trading         → Toutes les 15min (heures marché)")
    print("  4. Historical      → 1x/jour (16h30 ET)")
    print("  5. Full System     → Manuel")
    print("\n🔧 Démarrer le worker:")
    print("  prefect worker start --pool sentinel-pool")


if __name__ == "__main__":
    deploy_all()
