"""
🚀 Déploiements Prefect pour Sentinel2
Configure les schedules et les déploiements automatiques
"""

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from sentinel_flows import data_refresh_flow, trading_flow, full_system_flow
from datetime import timedelta


# ============================================================================
# SCHEDULES
# ============================================================================

# Data Refresh: Toutes les 4 minutes (news)
data_refresh_schedule = CronSchedule(
    cron="*/4 * * * *",  # Toutes les 4 minutes
    timezone="America/New_York"
)

# Trading: Toutes les 15 minutes (pendant heures marché)
trading_schedule = CronSchedule(
    cron="*/15 9-16 * * 1-5",  # 9h-16h, Lun-Ven (heures marché US)
    timezone="America/New_York"
)

# Validation: Toutes les 5 minutes
validation_schedule = CronSchedule(
    cron="*/5 * * * *",
    timezone="America/New_York"
)


# ============================================================================
# DEPLOYMENTS
# ============================================================================

# Déploiement 1: Data Refresh (fréquent)
data_refresh_deployment = Deployment.build_from_flow(
    flow=data_refresh_flow,
    name="data-refresh-production",
    version="1.0.0",
    tags=["production", "data", "refresh"],
    schedule=data_refresh_schedule,
    work_pool_name="sentinel-pool",
    description="Rafraîchit les prix et news toutes les 4 minutes"
)

# Déploiement 2: Trading (heures marché)
trading_deployment = Deployment.build_from_flow(
    flow=trading_flow,
    name="trading-production",
    version="1.0.0",
    tags=["production", "trading", "decisions"],
    schedule=trading_schedule,
    parameters={"force": False},
    work_pool_name="sentinel-pool",
    description="Pipeline trading complet toutes les 15min (heures marché)"
)

# Déploiement 3: Full System (démarrage)
full_system_deployment = Deployment.build_from_flow(
    flow=full_system_flow,
    name="full-system-startup",
    version="1.0.0",
    tags=["production", "startup"],
    work_pool_name="sentinel-pool",
    description="Démarrage complet du système (manuel)"
)


# ============================================================================
# DEPLOYMENT REGISTRY
# ============================================================================

def deploy_all():
    """Déploie tous les flows"""
    deployments = [
        data_refresh_deployment,
        trading_deployment,
        full_system_deployment
    ]
    
    for deployment in deployments:
        deployment.apply()
        print(f"✅ Déploiement: {deployment.name}")
    
    print("\n🎉 Tous les déploiements sont créés !")
    print("\nPour démarrer le worker:")
    print("  prefect worker start --pool sentinel-pool")


if __name__ == "__main__":
    deploy_all()
