# 🏗️ Principes d'Architecture Sentinel2

## 1. DRY (Don't Repeat Yourself)

### ❌ ÉVITER : Duplication de code
```python
# MAUVAIS - 3 méthodes qui font la même chose
def save_prices(data, ticker):
    # 30 lignes de logique incrémentale
    
def save_news(data, ticker):
    # 30 lignes de logique incrémentale (DUPLIQUÉE)
    
def save_sentiment(data, ticker):
    # 30 lignes de logique incrémentale (DUPLIQUÉE)
```

### ✅ APPLIQUER : Méthode générique unique
```python
# BON - Une méthode, logique centralisée
def save_data(data, data_type, ticker=None):
    if data_type == "prices":
        dedup_cols = ['ts_utc']
    elif data_type == "news":
        dedup_cols = ['timestamp', 'title']
    # ...
    return self._save_incremental(data, dedup_cols)
```

---

## 2. API Unifiée et Cohérente

### ❌ ÉVITER : Multiples méthodes similaires
```python
storage.save_prices(df, "SPY")
storage.save_news(df)
storage.save_sentiment(df, "SPY")
```

### ✅ APPLIQUER : Interface unique claire
```python
storage.save_data(df, data_type="prices", ticker="SPY")
storage.save_data(df, data_type="news")
storage.save_data(df, data_type="sentiment", ticker="SPY")
```

**Avantages** :
- Interface prévisible
- Validation centralisée
- Extensibilité facile

---

## 3. Validation Stricte

### ✅ TOUJOURS valider les entrées
```python
def save_data(self, data_type: str, ticker: str = None):
    if data_type not in ["prices", "news", "sentiment"]:
        raise ValueError(f"data_type invalide: {data_type}")
    
    if data_type in ["prices", "sentiment"] and not ticker:
        raise ValueError(f"ticker requis pour {data_type}")
```

---

## 4. Architecture en Couches

```
Scripts (finnhub_scraper.py, refresh_news.py)
    ↓ appellent
Storage API publique (save_data, load_data)
    ↓ appelle
Méthodes privées (_save_incremental, _load_data)
```

**Règle** : 
- Méthodes publiques = API simple et cohérente
- Méthodes privées = Logique réutilisable

---

## 5. Documentation Complète

### ✅ Toujours documenter
```python
def save_data(self, data_type: str, ticker: str = None) -> Path:
    """Sauvegarde générique de données
    
    Args:
        data_type: Type ("prices", "news", "sentiment")
        ticker: Symbole (requis pour prices/sentiment)
        
    Returns:
        Path du fichier sauvegardé
        
    Raises:
        ValueError: Si data_type invalide ou ticker manquant
    """
```

---

## 6. Refactoring Progressif

**Méthodologie appliquée** :
1. Identifier duplication
2. Créer méthode générique
3. Migrer les appelants un par un
4. Tester à chaque étape
5. Supprimer ancien code

---

## 7. Tests Avant Production

**Toujours** :
- Créer test incrémental (`test_storage_incremental.py`)
- Valider sur données réelles
- Nettoyer les données de test
- Vérifier aucune régression

---

## 8. Sécurité

### ❌ JAMAIS de secrets en dur
```python
# INTERDIT
API_KEY = "abc123def456"
```

### ✅ Variables d'environnement (.env)
```python
# BON
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY manquante dans .env")
```

---

## Checklist de Révision

Avant de valider du code, vérifier :
- [ ] Pas de duplication (DRY)
- [ ] API cohérente et simple
- [ ] Validation des entrées
- [ ] Documentation complète
- [ ] Tests couvrant le code
- [ ] Architecture en couches respectée
- [ ] Aucun secret en dur
- [ ] Imports en haut du fichier

---

## Exemples Appliqués dans Sentinel2

### ✅ storage.py
- Méthode unique `save_data()` au lieu de 6 méthodes
- Validation stricte du `data_type`
- Documentation complète avec docstrings

### ✅ finnhub_scraper.py
- Pas de clé API en dur
- Validation `.env` obligatoire
- Utilisation API storage centralisée

### ✅ refresh_news.py
- Migration vers `save_data()`
- Séparation responsabilités (fetch → analyze → save)
