# 🔍 AUDIT DES SCRIPTS - Sentinel2

**Date** : 2025-10-01  
**Objectif** : Identifier scripts inutilisés

---

## 📊 **SCRIPTS ACTUELS (13 fichiers)**

### **✅ UTILISÉS PAR PREFECT (4 essentiels)**

| Script | Utilisé par | Status |
|--------|-------------|--------|
| `refresh_prices.py` | Flow Prefect "Prix 15min" | ✅ **CONSERVER** |
| `refresh_news.py` | Flow Prefect "News + Sentiment" | ✅ **CONSERVER** |
| `trading_pipeline.py` | Flow Prefect "Trading" | ✅ **CONSERVER** |
| `update_prices_simple.py` | Flow Prefect "Historical" | ✅ **CONSERVER** |

### **✅ UTILISÉS PAR MAKEFILE (1)**

| Script | Utilisé par | Status |
|--------|-------------|--------|
| `check_production.sh` | `make check-prod` | ✅ **CONSERVER** |

### **❓ UTILITAIRES (5 - à vérifier)**

| Script | Usage | Dernière utilisation | Décision |
|--------|-------|---------------------|----------|
| `train_lstm_model.py` | Ré-entraînement manuel LSTM | Jamais référencé | ❌ **ARCHIVER** |
| `test_system.py` | Tests système | Jamais référencé | ❌ **ARCHIVER** |
| `refresh_data.py` | Refresh global | Jamais référencé | ❌ **ARCHIVER** |
| `clean_comments.py` | Nettoyage commentaires | One-shot fait | ❌ **ARCHIVER** |
| `clean_logs.sh` | Nettoyage logs | `make clean-logs` existe | ⚠️ **VÉRIFIER** |

### **❌ SCRIPTS OBSOLÈTES (3)**

| Script | Raison | Décision |
|--------|--------|----------|
| `start_ollama.sh` | Makefile gère le démarrage | ❌ **SUPPRIMER** |
| `start_prefect.sh` | Makefile gère le démarrage | ❌ **SUPPRIMER** |
| `sentinel2.sh` | Remplacé par Makefile | ❌ **SUPPRIMER** |

---

## 🎯 **RECOMMANDATIONS**

### **À ARCHIVER (5 scripts)**
```bash
scripts/train_lstm_model.py    # Utile mais jamais automatisé
scripts/test_system.py          # Tests manuels
scripts/refresh_data.py         # Redondant avec Prefect
scripts/clean_comments.py       # One-shot terminé
```

### **À SUPPRIMER (3 scripts)**
```bash
scripts/start_ollama.sh         # Makefile le fait
scripts/start_prefect.sh        # Makefile le fait
scripts/sentinel2.sh            # Obsolète
```

### **À CONSERVER (5 scripts)**
```bash
# Python
scripts/refresh_prices.py       # Prefect
scripts/refresh_news.py         # Prefect
scripts/trading_pipeline.py     # Prefect
scripts/update_prices_simple.py # Prefect

# Shell
scripts/check_production.sh     # Make
```

---

## 📈 **APRÈS NETTOYAGE**

**Avant** : 13 scripts  
**Après** : 5 scripts  
**Gain** : **-62% de scripts**

**Structure finale** :
```
scripts/
├── refresh_prices.py        # Prefect Flow
├── refresh_news.py          # Prefect Flow
├── trading_pipeline.py      # Prefect Flow
├── update_prices_simple.py  # Prefect Flow
└── check_production.sh      # Makefile
```

**Scripts archivés** : `scripts/archive/`

---

## ✅ **VALIDATION**

Après nettoyage, vérifier :
```bash
make check-all  # Tous services OK
```

Aucun script ne devrait être référencé nulle part sauf dans :
- `flows/sentinel_flows.py` (4 scripts Python)
- `Makefile` (1 script shell)
