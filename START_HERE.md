# 🚀 START HERE - Sentinel2

## **Démarrage ULTRA RAPIDE (1 commande)**

```bash
make start
```

**TOUT démarre automatiquement** :
- ✅ Ollama (LLM)
- ✅ Prefect Server (orchestration)
- ✅ **Prefect Worker (CRITIQUE !)**
- ✅ Orchestrateur (sentinel_main.py)
- ✅ Streamlit (interface)

**Attendre ~20 secondes** que tout démarre.

---

## **✅ Vérifier que tout tourne**

```bash
make status
```

**Output attendu** :
```
📊 Statut de Sentinel2:

Prefect Server:
  ✅ En cours d'exécution
   Dashboard: http://localhost:4200

Prefect Worker:
  ✅ En cours d'exécution

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

## **📊 Accès aux interfaces**

### **Streamlit (Interface principale)**
```
http://localhost:8501
```

### **Prefect (Dashboard orchestration)**
```
http://localhost:4200
```

---

## **🔄 Les flows s'exécutent automatiquement !**

Une fois le worker démarré, les flows s'exécutent selon leurs schedules :

| Flow | Fréquence |
|------|-----------|
| **📊 Prix 15min** | Toutes les 15 minutes |
| **📰 News + Sentiment** | Toutes les 4 minutes |
| **🤖 Trading** | Toutes les 15 minutes (heures marché) |
| **📈 Historical** | 1x/jour à 16h30 ET |

---

## **🐛 Troubleshooting**

### **Prefect dashboard montre "0 completed runs"**

**Cause** : Le worker n'est pas démarré

**Solution** :
```bash
make status  # Vérifier si worker actif
make start-prefect-worker  # Si pas actif
```

### **Pas de décisions dans Streamlit**

**Cause** : Aucun flow trading n'a encore tourné

**Solution** :
```bash
# Exécuter manuellement le flow full system
uv run prefect deployment run '🚀 Full System Flow/full-system-startup'
```

### **"N/A" partout dans Prefect**

**Cause** : Worker pas connecté

**Solution** :
```bash
# Redémarrer worker
pkill -f "prefect worker"
make start-prefect-worker
```

---

## **🛑 Arrêter l'application**

```bash
make stop
```

---

## **🔧 Commandes utiles**

| Commande | Action |
|----------|--------|
| `make start` | Tout démarrer |
| `make stop` | Tout arrêter |
| `make restart` | Redémarrer |
| `make status` | Statut complet |
| `make check-prod` | Vérification production |
| `make prefect-ui` | Ouvrir Prefect |

---

## **📚 Documentation complète**

- `QUICK_START.md` → Guide détaillé
- `PREFECT_GUIDE.md` → Guide Prefect complet
- `README.md` → Architecture générale

---

**🎉 C'est tout ! Sentinel2 tourne automatiquement !**
