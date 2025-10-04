"""
Déploiement Prefect 3.0+ pour Sentinel2
"""

from sentinel_flows import (
    prices_15min_flow,
    news_sentiment_flow,
    trading_flow,
    full_system_flow
)

if __name__ == "__main__":
    print("🚀 Déploiement des flows Sentinel2 (Prefect 3.0)")
    print("=" * 60)
    
    # Flow 1: Prix 15min
    print("\n1️⃣ Déploiement Prix 15min...")
    prices_15min_flow.serve(
        name="prices-15min-production",
        cron="*/15 * * * *",
        tags=["production", "prices", "15min"]
    )
    
    print("\n✅ Flows configurés pour serve mode !")
    print("\n📊 Pour démarrer, lancer dans des terminaux séparés:")
    print("  Terminal 1: make prefect-ui")
    print("  Terminal 2: uv run python flows/deploy_flows.py")
