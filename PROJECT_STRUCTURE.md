# 📁 STRUCTURE DU PROJET - Sentinel2

**Dernière mise à jour** : 2025-10-01  
**Version** : 2.0

---

## 📂 ARBORESCENCE PRINCIPALE

```
sentinel2/
├── .windsurf/                  # Configuration Windsurf
│   ├── README_WINDSURF.md      # Documentation Windsurf
│   └── WINDSURF_GLOBAL_MEMORIES.md  # Mémoires globales
├── .windsurfrules              # Règles Windsurf/Cascade
├── archive_docs/               # Documents archivés
│   ├── AUDIT_COMPLET.md
│   ├── PLAN_NETTOYAGE.md
│   ├── README_ARCHITECTURE.md
│   └── README_OLD.md
├── config/                     # Configuration
│   ├── config.json
│   ├── models.json
│   └── settings.py
├── data/                       # Données (gitignored)
│   ├── realtime/              # Données temps réel
│   ├── trading/               # Décisions trading
│   ├── models/                # Modèles ML
│   └── logs/                  # Logs système
├── flows/                      # Prefect Flows
│   ├── sentinel_flows.py      # 5 flows principaux
│   └── deployments.py         # Configuration déploiements
├── scripts/                    # Scripts essentiels (6)
│   ├── refresh_prices.py      # Prefect
│   ├── refresh_news.py        # Prefect
│   ├── trading_pipeline.py    # Prefect
│   ├── update_prices_simple.py # Prefect
│   ├── check_production.sh    # Vérification
│   └── clean_logs.sh          # Nettoyage
├── src/                        # Code source
│   ├── core/                  # ML (LSTM, Sentiment, Fusion)
│   ├── data/                  # Gestion données
│   ├── gui/                   # Interface Streamlit
│   ├── models/                # Modèles ML
│   ├── tests/                 # Tests unitaires/intégration
│   └── constants.py           # Constantes globales
├── COPYRIGHT                   # Avis de copyright
├── LICENSE                     # Licence propriétaire
├── Makefile                    # Commandes simplifiées
├── PREFECT_GUIDE.md           # Guide Prefect
├── QUICK_START.md             # Guide démarrage
├── README.md                  # Documentation principale
└── START_HERE.md              # Démarrage ultra-rapide
```

---

## 📊 STATISTIQUES

| Catégorie | Nombre |
|-----------|--------|
| **Scripts actifs** | 6 |
| **Flows Prefect** | 5 |
| **Documents actifs** | 4 |
| **Documents archivés** | 4 |
| **Fichiers configuration** | 3 |

---

## 🎯 DOCUMENTS ACTIFS

### **📚 Documentation principale**

| Fichier | Description | Priorité |
|---------|-------------|----------|
| `START_HERE.md` | ⭐ Démarrage 1 commande | **HAUTE** |
| `README.md` | Documentation complète | **HAUTE** |
| `QUICK_START.md` | Guide détaillé | Moyenne |
| `PREFECT_GUIDE.md` | Guide Prefect | Moyenne |

### **🔒 Légal**

| Fichier | Description |
|---------|-------------|
| `LICENSE` | Licence propriétaire restrictive |
| `COPYRIGHT` | Avis de copyright |

### **⚙️ Configuration**

| Fichier | Description |
|---------|-------------|
| `.windsurfrules` | Règles Windsurf/Cascade |
| `Makefile` | Commandes simplifiées |
| `prefect.yaml` | Configuration Prefect |

---

## 📁 DOSSIERS SPÉCIAUX

### **.windsurf/**
Documentation et configuration Windsurf/Cascade.

### **archive_docs/**
Anciens documents conservés pour référence historique.

### **data/** (gitignored)
Données générées par l'application :
- Prix temps réel
- News + sentiment
- Décisions trading
- Modèles entraînés
- Logs

---

## 🚀 SCRIPTS ESSENTIELS

### **Python (4 scripts Prefect)**
- `refresh_prices.py` - Flow "Prix 15min"
- `refresh_news.py` - Flow "News + Sentiment"
- `trading_pipeline.py` - Flow "Trading"
- `update_prices_simple.py` - Flow "Historical"

### **Shell (2 scripts)**
- `check_production.sh` - Vérification production
- `clean_logs.sh` - Nettoyage logs

---

## 🔧 FICHIERS DE CONFIGURATION

### **Principaux**
- `pyproject.toml` - Dépendances Python (uv)
- `.env` - Variables d'environnement (gitignored)
- `config/config.json` - Configuration application
- `prefect.yaml` - Configuration Prefect

### **IDE**
- `.windsurf/` - Windsurf/Cascade
- `.gitignore` - Git

---

## 📝 ORDRE DE LECTURE

Pour un nouveau développeur :

1. **START_HERE.md** - Démarrage rapide
2. **README.md** - Vue d'ensemble complète
3. **LICENSE** - Restrictions d'usage
4. **QUICK_START.md** - Guide détaillé
5. **PREFECT_GUIDE.md** - Orchestration

---

## 🔒 SÉCURITÉ

### **Fichiers sensibles (gitignored)**
- `.env` - Variables d'environnement
- `data/` - Toutes les données
- `*.log` - Logs
- `.venv/` - Environnement virtuel

### **Fichiers publics**
- Documentation (README, guides)
- Code source (src/, flows/, scripts/)
- Configuration (sauf .env)

---

## ✅ MAINTENANCE

### **Fichiers à maintenir**
- ✅ README.md - Documentation principale
- ✅ START_HERE.md - Démarrage
- ✅ LICENSE - Licence
- ✅ Scripts dans scripts/

### **Fichiers archivés (ne pas modifier)**
- ❌ archive_docs/* - Référence uniquement

---

**Dernière mise à jour** : 2025-10-01  
**Maintenu par** : Eagle42
