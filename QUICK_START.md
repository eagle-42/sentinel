# 🚀 Quick Start - Sentinel2

## **Démarrage en 3 commandes**

### **1️⃣ Démarrer l'application complète**

```bash
make start
```

**Cette commande lance automatiquement** :
- ✅ Ollama (LLM)
- ✅ Prefect Server (orchestration)
- ✅ Orchestrateur (sentinel_main.py)
- ✅ Streamlit (interface)

**Attendre ~15 secondes** que tout démarre.

---

### **2️⃣ Déployer les flows Prefect**

```bash
uv run prefect deploy --all
```

**Appuyer sur Entrée** pour accepter les valeurs par défaut.

**5 flows seront déployés** :
- 📊 Prix 15min (toutes les 15min)
- 📰 News + Sentiment (toutes les 4min)
- 🤖 Trading (toutes les 15min, heures marché)
- 📈 Historical (1x/jour, 16h30 ET)
- 🚀 Full System (manuel)

---

### **3️⃣ Démarrer le worker Prefect**

**Ouvrir un nouveau terminal** :

```bash
cd /Users/eagle/DevTools/sentinel2
uv run prefect worker start --pool sentinel
```

**Laisser ce terminal ouvert** - Le worker exécute les flows.

---

## **📊 Accès aux dashboards**

### **Streamlit (Interface principale)**
```
http://localhost:8501
```

### **Prefect (Orchestration)**
```
http://localhost:4200
```

---

## **✅ Vérifier le statut**

```bash
make status
```

**Output attendu** :
```
📊 Statut de Sentinel2:

Prefect Server:
  ✅ En cours d'exécution
   Dashboard: http://localhost:4200

Orchestrateur:
  ✅ En cours d'exécution

Streamlit:
  ✅ En cours d'exécution
   URL: http://localhost:8501

Ollama:
  ✅ En cours d'exécution
   Port: 11434
```

---

## **🛑 Arrêter l'application**

```bash
make stop
```

**Arrête TOUT** :
- Streamlit
- Orchestrateur  
- Prefect
- Ollama

---

## **🔧 Commandes utiles**

### **Redémarrer**
```bash
make restart
```

### **Logs**
```bash
# Logs Prefect
tail -f data/logs/prefect_server.log

# Logs orchestrateur
tail -f data/logs/sentinel_orchestrator.log

# Logs trading
tail -f data/logs/trading_decisions.log
```

### **Nettoyer les logs**
```bash
make clean-logs
```

### **Ouvrir Prefect UI**
```bash
make prefect-ui
```

---

## **📈 Flows Prefect**

### **Exécuter manuellement un flow**

```bash
# Full system (init complet)
uv run prefect deployment run '🚀 Full System Flow/full-system-startup'

# Trading immédiat
uv run prefect deployment run '🤖 Trading Flow/trading-production'

# Prix 15min
uv run prefect deployment run '📊 Prix 15min Flow/prices-15min-production'

# News + Sentiment
uv run prefect deployment run '📰 News + Sentiment Flow/news-sentiment-production'
```

### **Voir les flows actifs**

```bash
uv run prefect deployment ls
```

---

## **🐛 Troubleshooting**

### **Prefect ne démarre pas**

```bash
# Tuer processus Prefect
pkill -f "prefect server"

# Redémarrer
make start-prefect-server
```

### **Worker ne se connecte pas**

```bash
# Vérifier work pool
uv run prefect work-pool ls

# Créer si manquant
uv run prefect work-pool create sentinel --type process
```

### **Streamlit ne charge pas**

```bash
# Redémarrer Streamlit
make stop-streamlit
make start-streamlit
```

---

## **🎯 Utilisation normale**

1. **Démarrer** : `make start` (une fois au début)
2. **Worker** : `uv run prefect worker start --pool sentinel` (dans un terminal dédié)
3. **Utiliser** : Ouvrir http://localhost:8501
4. **Arrêter** : `make stop` (à la fin)

**Les flows s'exécutent automatiquement selon leur schedule !**

---

## **📚 Documentation complète**

- **Prefect** : `PREFECT_GUIDE.md`
- **Architecture** : `README_ARCHITECTURE.md`
- **Plan nettoyage** : `PLAN_NETTOYAGE.md`

---

**🎉 Sentinel2 est maintenant opérationnel !**
