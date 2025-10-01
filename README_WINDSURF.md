# 🌊 Configuration Windsurf/Cascade pour Sentinel2

## ✅ Fichiers de Configuration

### **1. Règles du Projet** (`.windsurfrules`)
- **Emplacement** : `.windsurfrules` à la racine
- **Statut** : ✅ Configuré
- **Portée** : Règles spécifiques à Sentinel2
- **Usage** : Cascade les lit automatiquement pour ce projet

### **2. Règles Globales** (Memories dans l'IDE)
- **Emplacement** : Interface Windsurf → Cascade → Memories
- **Guide** : Voir `WINDSURF_GLOBAL_MEMORIES.md`
- **Portée** : Tous vos projets Windsurf
- **Action requise** : 👉 À configurer manuellement dans l'IDE

---

## 📋 Checklist de Configuration

### ✅ Fait
- [x] Fichier `.windsurfrules` créé et configuré
- [x] Fichier `.env` nettoyé (suppression des doublons)
- [x] Guide des Memories globales créé (`WINDSURF_GLOBAL_MEMORIES.md`)
- [x] Backup de l'ancien `.env` créé (`.env.backup`)

### 📝 À Faire par Vous
- [ ] Ouvrir Windsurf IDE
- [ ] Accéder à Cascade → Memories
- [ ] Copier-coller les 7 memories depuis `WINDSURF_GLOBAL_MEMORIES.md`
- [ ] Sauvegarder et activer les memories
- [ ] Tester avec une commande simple à Cascade

---

## 🎯 Structure Finale

```
sentinel2/
├── .windsurfrules                    # ✅ Règles projet (auto-détecté)
├── WINDSURF_GLOBAL_MEMORIES.md       # 📖 Guide memories globales
├── README_WINDSURF.md                # 📖 Ce fichier
├── .env                              # ✅ Configuration nettoyée
├── .env.backup                       # 💾 Backup
└── .cursor/
    └── rules                         # ℹ️  Règles Cursor (conservées)
```

---

## 🚀 Comment Utiliser

### **Lancer Cascade avec les règles**
1. Ouvrir Windsurf sur le projet Sentinel2
2. Ouvrir Cascade (assistant IA)
3. Les règles `.windsurfrules` sont chargées automatiquement
4. Cascade respecte l'architecture TDD de Sentinel2

### **Exemples de commandes**
```
"Crée un nouveau service pour l'analyse technique"
→ Cascade appliquera automatiquement les règles TDD, architecture, etc.

"Ajoute une feature de notification"
→ Cascade demandera confirmation (changement majeur)

"Optimise le chargement des données"
→ Cascade respectera la structure Parquet, pas de doublons
```

---

## 📚 Hiérarchie des Règles

1. **Memories Globales** (toujours actives)
   - Style de code général
   - Workflow de développement
   - Technologies préférées

2. **`.windsurfrules`** (projet Sentinel2)
   - Architecture TDD spécifique
   - Interdictions (simulations, doublons Parquet)
   - Structure des features

3. **README.md** (documentation de référence)
   - Architecture complète
   - Commandes et scripts
   - Historique et métriques

---

## ⚡ Commandes Rapides

### **Tests**
```bash
# Tests complets
uv run python scripts/test_system.py

# Tests unitaires
uv run pytest src/tests/unit/ -v
```

### **Lancement**
```bash
# Mode production (avec Ollama)
caffeinate -d ./scripts/sentinel2.sh prod

# Mode développement
./scripts/sentinel2.sh dev

# Arrêt
./scripts/sentinel2.sh stop
```

### **Vérification**
```bash
# Voir les règles du projet
cat .windsurfrules

# Voir les variables d'environnement
cat .env
```

---

## 🔧 Dépannage

### **Cascade ne respecte pas les règles**
1. Vérifier que `.windsurfrules` existe : `ls -la .windsurfrules`
2. Relancer Windsurf/Cascade
3. Vérifier les Memories globales dans l'IDE

### **Erreurs de configuration**
1. Vérifier `.env` : `cat .env`
2. Comparer avec `env.example` : `diff .env env.example`
3. Restaurer backup si nécessaire : `cp .env.backup .env`

---

## 📖 Documentation Complète

- **README.md** : Documentation principale du projet
- **README_ARCHITECTURE.md** : Architecture détaillée
- **READMEFUSION.md** : Documentation fusion adaptative
- **WINDSURF_GLOBAL_MEMORIES.md** : Guide configuration Memories
- **.windsurfrules** : Règles projet (ce fichier)

---

**Projet** : Sentinel2  
**Version** : 2.0  
**Statut** : ✅ Finalisé et Validé  
**Assistant** : Cascade (Windsurf)
