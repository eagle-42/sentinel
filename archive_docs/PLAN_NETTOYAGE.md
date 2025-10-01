# 🧹 PLAN DE NETTOYAGE SENTINEL2

**Date**: 2025-10-01  
**Note actuelle**: D ❌ (159 issues)  
**Objectif**: A ✅ (<20 issues)

---

## 📊 **RÉSUMÉ AUDIT**

| Catégorie | Issues | Priorité | Temps |
|-----------|--------|----------|-------|
| **Imports inutilisés** | 106 | 🟡 Moyenne | 30min |
| **Fichiers sans tests** | 25 | 🔴 Haute | 2h |
| **Manque documentation** | 17 | 🟡 Moyenne | 1h |
| **Bonnes pratiques** | 15 | 🟢 Basse | 30min |
| **Fonctions complexes** | 8 | 🟡 Moyenne | 1h |
| **Code dupliqué** | 10 | 🟡 Moyenne | 1h |

**Total temps estimé**: 6 heures

---

## 🎯 **PHASE 1: NETTOYAGE RAPIDE** (1h)

### 1.1 Supprimer imports inutilisés (30min)
**Fichiers prioritaires**:
- `src/core/sentiment.py` - numpy, pandas non utilisés
- `src/core/prediction.py` - pandas non utilisé
- `src/core/fusion.py` - numpy non utilisé
- Tous les tests avec imports inutilisés

**Action**:
```bash
# Script automatique
python scripts/clean_unused_imports.py
```

### 1.2 Corriger print() → logger (15min)
**Fichiers**:
- `src/tests/run_tests.py` (35 print)
- `src/notebooks/*.py` (OK, notebooks peuvent garder print)
- `src/gui/launch.py` (6 print → logger)

**Action**: Remplacer `print(` par `logger.info(`

### 1.3 Nettoyer fichiers temporaires (5min)
```bash
# Supprimer .DS_Store
find . -name ".DS_Store" -delete

# Vérifier logs
ls -lh data/logs/
```

### 1.4 Formater code (10min)
```bash
# Black (formatage automatique)
pip install black
black src/ --line-length 120

# isort (tri imports)
pip install isort
isort src/
```

---

## 🎯 **PHASE 2: SIMPLIFICATION CODE** (2h) - ✅ COMPLÉTÉE

### ✅ 2.1 Nettoyage fichiers orphelins
- ✅ Suppression fichiers backup (déjà fait)
- ✅ Nettoyage __pycache__ (déjà fait)
- ✅ .gitignore complet (__pycache__, .streamlit, backups)
- ✅ Commit evaluate_lstm_article.py (déjà fait)

### ✅ 2.2 Suppression code mort
- ✅ Aucun TODO/FIXME non résolu
- ✅ Aucun import inutilisé (vérifié avec ruff)
- ✅ Code prediction.py optimisé (656→335 lignes)

### 🔄 2.3 Refactoring complexité (EN COURS)
**Fonctions complexes identifiées** (>10 branches):
1. `production_page.py::render_production_page` (~1300 lignes)
2. `decision_validation_service.py::validate_decision` (~650 lignes)
3. `chart_service.py::create_main_chart` (~500 lignes)

**Actions à faire** (optionnel - n'affecte pas fonctionnement):
- Découper pages GUI en composants
- Extraire logique conditionnelle
- Créer fonctions helper si nécessaire

**Note**: Ces fichiers fonctionnent parfaitement. Refactoring optionnel pour maintenabilité future.

---

## 🎯 **PHASE 3: AJOUTER TESTS** (2h)

### 3.1 Tests manquants prioritaires
**Fichiers SANS tests** (top 10):
1. `src/gui/pages/production_page.py` (1298 lignes !)
2. `src/gui/services/decision_validation_service.py` (655 lignes)
3. `src/gui/services/chart_service.py` (498 lignes)
4. `src/data/unified_storage.py` (452 lignes)
5. `src/gui/pages/logs_page.py` (447 lignes)
6. `src/gui/pages/analysis_page.py` (430 lignes)
7. `src/gui/services/historical_validation_service.py` (422 lignes)
8. `src/gui/services/fusion_service.py` (378 lignes)
9. `src/data/storage.py` (386 lignes)
10. `src/core/prediction.py` (342 lignes)

**Actions**:
- `test_prediction.py` ✅ (core critique)
- `test_unified_storage.py` ✅ (data critique)
- `test_decision_validation_service.py` (GUI critique)

**Tests à créer**:
```python
# tests/unit/test_prediction_article.py
def test_prediction_returns_mode():
    """Test modèle avec RETURNS"""
    pass

def test_prediction_price_conversion():
    """Test conversion RETURNS → Prix"""
    pass
```

---

## 🎯 **PHASE 4: DOCUMENTATION** (1h)

### 4.1 Ajouter docstrings manquants
**Fichiers prioritaires**:
- `src/core/prediction.py` - fonctions publiques
- `src/gui/services/*.py` - toutes les classes
- `src/data/unified_storage.py` - API principale

**Format docstring**:
```python
def my_function(param1: str, param2: int) -> bool:
    """
    Description courte de la fonction
    
    Args:
        param1: Description param1
        param2: Description param2
        
    Returns:
        bool: Description retour
        
    Raises:
        ValueError: Quand param1 invalide
    """
    pass
```

### 4.2 Mettre à jour README.md
**Sections à ajouter**:
- Architecture finale (avec schéma)
- Guide installation
- Guide contribution
- Résultats modèle LSTM (99.32% accuracy !)

---

---

## 📊 **STATUT ACTUEL DU PROJET**

### ✅ **PHASES COMPLÉTÉES**
- ✅ **Phase 1** : Quick wins (formatting, imports, tests unitaires)
- ✅ **Phase 2** : Simplification code (nettoyage, suppression code mort)

### 🔄 **PHASES EN COURS**
- 🔄 **Phase 3** : Tests (couverture à améliorer)
- 🔄 **Phase 4** : Documentation (docstrings à compléter)

### 📈 **MÉTRIQUES**
- **Tests** : 110 tests (11 intégration, 99 unit)
- **Coverage** : ~60% (objectif 80%)
- **Code quality** : Ruff clean, Black formaté
- **Architecture** : TDD, modulaire, propre

### 🚀 **PROCHAINES ÉTAPES**
1. Ajouter tests GUI critiques
2. Compléter docstrings
3. Intégrer Prefect (orchestration visuelle)

---

## 🎯 **ORDRE D'EXÉCUTION RECOMMANDÉ**

### **🚀 Sprint 1: Quick Wins (1h)**
1. ✅ Supprimer imports inutilisés (automatique)
2. ✅ Formater code (black + isort)
3. ✅ Corriger print() → logger
4. ✅ Nettoyer .DS_Store

**Résultat attendu**: -100 issues → Note C ⚠️

### **🔧 Sprint 2: Refactoring (2h)**
1. Réduire complexité top 3 fonctions
2. Factoriser code dupliqué
3. Créer `src/utils/` si nécessaire

**Résultat attendu**: -18 issues → Note B ⚠️

### **🧪 Sprint 3: Tests + Docs (2h)**
1. Tests unitaires manquants (core)
2. Docstrings fonctions publiques
3. Mise à jour README

**Résultat attendu**: -30 issues → Note A ✅

---

## 📋 **CHECKLIST ACTIONS IMMÉDIATES**

### **Maintenant** (5 min)
- [ ] Créer branch `feature/nettoyage-complet`
- [ ] Commiter état actuel (sauvegarde)

### **Sprint 1** (1h)
- [ ] Exécuter `clean_unused_imports.py`
- [ ] Exécuter `black src/ --line-length 120`
- [ ] Exécuter `isort src/`
- [ ] Remplacer print() par logger dans launch.py
- [ ] `find . -name ".DS_Store" -delete`
- [ ] Relancer audit → vérifier amélioration
- [ ] Commit "🧹 Sprint 1: Quick wins"

### **Sprint 2** (2h)
- [ ] Refactor `production_page.py::render_production_page`
- [ ] Factoriser validation commune
- [ ] Créer `src/utils/date_utils.py`
- [ ] Commit "♻️ Sprint 2: Refactoring"

### **Sprint 3** (2h)
- [ ] Créer `test_prediction_article.py`
- [ ] Créer `test_unified_storage_complete.py`
- [ ] Ajouter docstrings manquants
- [ ] Mettre à jour README.md
- [ ] Commit "📚 Sprint 3: Tests + Docs"

### **Validation Finale**
- [ ] Relancer audit → Note A ✅
- [ ] Tests: `uv run pytest` → 100% pass
- [ ] Merge dans main

---

## 🎯 **OBJECTIF FINAL**

**Note visée**: A ✅  
**Issues max**: < 20  
**Tests coverage**: > 60% (actuellement 43%)  
**Temps total**: 6 heures réparties sur 2-3 jours

---

**Status**: 🟡 EN ATTENTE  
**Prochaine action**: Créer branch + Sprint 1
