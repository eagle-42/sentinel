#!/bin/bash
# 🚀 Démarrage Prefect pour Sentinel2

echo "🚀 Démarrage Prefect Sentinel2"
echo "================================"

# 1. Démarrer le serveur Prefect
echo ""
echo "1️⃣ Démarrage serveur Prefect..."
prefect server start &
SERVER_PID=$!
echo "   PID: $SERVER_PID"

# Attendre que le serveur soit prêt
sleep 5

# 2. Créer le work pool
echo ""
echo "2️⃣ Création work pool..."
prefect work-pool create sentinel-pool --type process || echo "   Work pool existe déjà"

# 3. Déployer les flows
echo ""
echo "3️⃣ Déploiement des flows..."
cd flows
uv run python deployments.py
cd ..

# 4. Démarrer le worker
echo ""
echo "4️⃣ Démarrage worker..."
prefect worker start --pool sentinel-pool &
WORKER_PID=$!
echo "   PID: $WORKER_PID"

echo ""
echo "✅ Prefect démarré !"
echo ""
echo "📊 Dashboard: http://localhost:4200"
echo ""
echo "🛑 Pour arrêter:"
echo "   kill $SERVER_PID $WORKER_PID"
echo ""

# Garder le script actif
wait
