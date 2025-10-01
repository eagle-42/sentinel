# 🌍 Configuration des Memories Globales Windsurf/Cascade

Ce fichier contient les règles globales à configurer dans Windsurf pour tous vos projets.

---

## 📋 Comment configurer les Memories dans Windsurf

### **Étape 1 : Accéder aux Memories**
1. Ouvrir Windsurf IDE
2. Cliquer sur l'icône **Cascade** (en bas à gauche ou dans la barre latérale)
3. Chercher l'option **"Memories"** ou **"Rules"** dans les paramètres
4. Cliquer sur **"Create Memory"** ou **"Add Global Rule"**

### **Étape 2 : Copier-coller les règles ci-dessous**

---

## 🚀 RÈGLES GLOBALES À AJOUTER DANS MEMORIES

### **Memory 1 : Préférences de Langue et Communication**

```markdown
# Préférences de Communication

- **Langue** : Français pour les interactions
- **Style** : Direct, concis, sans phrases d'acquiescement
- **Documentation** : Code en anglais, commentaires en français si nécessaire
- **Explication** : Toujours expliquer POURQUOI et pas seulement COMMENT
```

---

### **Memory 2 : Architecture et Bonnes Pratiques**

```markdown
# Architecture et Bonnes Pratiques Générales

## Principes de Code
- **TDD obligatoire** : Tests AVANT implémentation
- **Fonctions courtes** : Maximum 100 lignes
- **Single Responsibility** : Une fonction/classe = une responsabilité
- **DRY** : Don't Repeat Yourself - pas de duplication
- **Type hints** : Toujours annoter les types en Python

## Structure de Projet
- **Configuration centralisée** : Jamais de valeurs en dur
- **Variables d'environnement** : Pour toute configuration sensible
- **Séparation des responsabilités** : Logique métier séparée de l'UI
- **Documentation** : README.md à jour + docstrings complètes

## Imports et Organisation
- **Imports en haut** : JAMAIS d'imports dans les fonctions
- **Ordre des imports** : Standard → Tiers → Local
- **Un import par ligne** : Pas de imports multiples
```

---

### **Memory 3 : Workflow de Développement**

```markdown
# Workflow de Développement

## Avant toute modification
1. Lire la documentation existante (README.md)
2. Comprendre l'architecture actuelle
3. Vérifier les tests existants
4. Identifier les constantes/config à utiliser

## Pendant le développement
1. **Tests d'abord** : Écrire le test avant le code
2. **Petits commits** : Modifications incrémentales
3. **Documentation inline** : Docstrings + type hints
4. **Respect de l'architecture** : Suivre la structure existante

## Après modification
1. **Tests complets** : Vérifier 100% de succès
2. **Vérification réelle** : Tester l'application
3. **Documentation** : Mettre à jour si nécessaire
4. **Confirmation** : Valider avec l'utilisateur
```

---

### **Memory 4 : Interdictions et Règles Critiques**

```markdown
# Interdictions Strictes (TOUJOURS RESPECTER)

## ❌ JAMAIS FAIRE
- **Pas de code sans tests** : TDD obligatoire
- **Pas de valeurs en dur** : Toujours utiliser config/constantes
- **Pas d'imports dans le code** : Tous en haut du fichier
- **Pas de "c'est fini" sans vérifier** : Toujours tester réellement
- **Pas de changements majeurs sans confirmation** : Demander validation

## ✅ TOUJOURS FAIRE
- **Demander confirmation** : Pour tout changement de logique métier
- **Vérifier le fonctionnement** : Tester avant de confirmer
- **Expliquer l'impact** : Décrire les conséquences des changements
- **Proposer des alternatives** : Donner plusieurs options
- **Documenter** : Code clair + docstrings + README
```

---

### **Memory 5 : Technologies et Outils Préférés**

```markdown
# Technologies et Outils (Python Focus)

## Gestionnaires de Dépendances
- **Préférence** : `uv` (ultra-rapide)
- **Alternative** : `poetry` ou `pip`
- **Fichiers** : `pyproject.toml` pour la config moderne

## Tests
- **Framework** : pytest
- **Coverage** : pytest-cov
- **Structure** : tests/unit, tests/integration, tests/e2e
- **Objectif** : 80% de couverture minimum

## Linting et Formatage
- **Linter** : ruff (rapide et moderne)
- **Formatter** : black
- **Type checking** : mypy
- **Pre-commit** : Vérifications automatiques

## Data Science / ML
- **Data** : pandas, polars (pour performances)
- **ML** : scikit-learn, PyTorch/TensorFlow
- **Viz** : plotly (interactif), matplotlib (statique)

## Web / API
- **API** : FastAPI (moderne et rapide)
- **UI** : Streamlit (dashboards rapides), Gradio (ML UI)
- **Async** : asyncio, aiohttp
```

---

### **Memory 6 : Sécurité et Configuration**

```markdown
# Sécurité et Configuration

## Variables d'Environnement
- **JAMAIS en dur** : Utiliser `.env` + python-dotenv
- **Template** : Toujours fournir `env.example`
- **Git** : `.env` TOUJOURS dans `.gitignore`
- **Validation** : Vérifier les variables au démarrage

## Clés API et Secrets
- **Stockage** : Variables d'environnement uniquement
- **Rotation** : Prévoir le changement facile des clés
- **Logs** : JAMAIS logger les secrets
- **Code** : JAMAIS commiter les clés

## Gestion d'Erreurs
- **Try/catch** : Toujours gérer les exceptions
- **Logs structurés** : Utiliser logging avec niveaux appropriés
- **Messages clairs** : Erreurs compréhensibles pour debug
- **Fallback** : Stratégie de repli si possible
```

---

### **Memory 7 : Performance et Optimisation**

```markdown
# Performance et Optimisation

## Principes
- **Profiler d'abord** : Mesurer avant d'optimiser
- **Optimisation prématurée** : Éviter, se concentrer sur la clarté
- **Bottlenecks** : Identifier les vrais goulots d'étranglement
- **Trade-offs** : Équilibrer performance vs maintenabilité

## Techniques
- **Cache intelligent** : Mémorisation des calculs coûteux
- **Lazy loading** : Charger uniquement ce qui est nécessaire
- **Async/await** : Pour les I/O (réseau, fichiers)
- **Vectorisation** : numpy/pandas pour traitement de données
- **Batch processing** : Traiter par lots si possible

## Monitoring
- **Logs** : Enregistrer les performances critiques
- **Métriques** : Temps d'exécution, mémoire, CPU
- **Alertes** : Définir des seuils acceptables
```

---

## 🎯 Instructions d'Installation

### **Dans Windsurf IDE :**

1. **Ouvrir Cascade** (assistant IA)
2. **Accéder aux Memories** (via paramètres ou commande)
3. **Créer 7 memories** (une par section ci-dessus)
4. **Copier-coller** le contenu de chaque memory
5. **Sauvegarder** et vérifier qu'elles sont actives

### **Structure finale :**

```
Windsurf Global Memories/
├── Memory 1: Préférences de Langue et Communication
├── Memory 2: Architecture et Bonnes Pratiques
├── Memory 3: Workflow de Développement
├── Memory 4: Interdictions et Règles Critiques
├── Memory 5: Technologies et Outils Préférés
├── Memory 6: Sécurité et Configuration
└── Memory 7: Performance et Optimisation
```

---

## ✅ Vérification

Après configuration, tester avec Cascade :
- Demander à Cascade de créer une fonction Python
- Vérifier qu'il applique les règles (TDD, type hints, docstrings)
- Vérifier qu'il communique en français
- Vérifier qu'il demande confirmation pour changements majeurs

---

**Ces règles globales s'appliqueront à TOUS vos projets Windsurf automatiquement.**  
**Les règles spécifiques du projet (`.windsurfrules`) les complètent pour Sentinel2.**

---

📝 **Note** : Ce fichier est juste un guide. Les vraies Memories doivent être configurées dans l'IDE Windsurf via l'interface graphique.
