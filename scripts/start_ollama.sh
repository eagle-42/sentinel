#!/bin/bash

# Script de démarrage d'Ollama pour Sentinel2
# Démarre Ollama avec le modèle phi3:mini

echo "🚀 Démarrage d'Ollama pour Sentinel2..."

# Vérifier si Ollama est installé
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama n'est pas installé. Veuillez l'installer depuis https://ollama.ai/"
    exit 1
fi

# Vérifier si le modèle phi3:mini est disponible
if ! ollama list | grep -q "phi3:mini"; then
    echo "📥 Téléchargement du modèle phi3:mini..."
    ollama pull phi3:mini
fi

# Démarrer Ollama en arrière-plan
echo "🔄 Démarrage du serveur Ollama..."
ollama serve &

# Attendre que le serveur soit prêt
echo "⏳ Attente du démarrage du serveur..."
sleep 5

# Vérifier que le serveur répond
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama est prêt sur http://localhost:11434"
    echo "📊 Modèle disponible: phi3:mini"
    echo "🔧 Pour arrêter: pkill -f ollama"
else
    echo "❌ Erreur: Ollama n'a pas démarré correctement"
    exit 1
fi
