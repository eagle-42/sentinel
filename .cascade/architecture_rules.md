# Règles d'Architecture Sentinel2 (Active Memory)

## RÈGLE 1 : DRY (Don't Repeat Yourself) - OBLIGATOIRE

❌ INTERDICTION STRICTE de dupliquer la logique
✅ TOUJOURS créer une méthode générique réutilisable

Exemple appliqué : `storage.py`
- AVANT : `save_prices()`, `save_news()`, `save_sentiment()` (3x30 lignes dupliquées)
- APRÈS : `save_data(data_type="prices"|"news"|"sentiment")` (1 méthode unique)

## RÈGLE 2 : API Unifiée - OBLIGATOIRE

❌ INTERDICTION de créer multiples méthodes similaires
✅ TOUJOURS utiliser une interface unique avec paramètre `type`

Pattern :
```python
# BON
def save_data(self, data, data_type: str, **kwargs):
    if data_type == "prices":
        # logique prices
    elif data_type == "news":
        # logique news
    # ...
```

## RÈGLE 3 : Validation Stricte - OBLIGATOIRE

✅ TOUJOURS valider les paramètres d'entrée
✅ TOUJOURS lever ValueError si invalide
✅ TOUJOURS documenter les valeurs acceptées

```python
if data_type not in ["prices", "news", "sentiment"]:
    raise ValueError(f"data_type invalide: {data_type}. Acceptées: 'prices', 'news', 'sentiment'")
```

## RÈGLE 4 : Architecture en Couches - OBLIGATOIRE

```
Scripts (finnhub_scraper.py, refresh_news.py)
    ↓ utilisent
API Publique (save_data, load_data)
    ↓ utilisent
Méthodes Privées (_save_incremental, _load_data)
```

Règle :
- Méthodes publiques = API simple, validation, routing
- Méthodes privées (_*) = Logique réutilisable

## RÈGLE 5 : Refactoring Progressif - PROCESS OBLIGATOIRE

Quand duplication détectée :
1. Identifier la duplication
2. Créer méthode générique
3. Écrire tests pour la nouvelle méthode
4. Migrer les appelants UN PAR UN
5. Tester après chaque migration
6. Supprimer ancien code seulement quand tout migré

❌ JAMAIS de refactoring "big bang" sans tests

## RÈGLE 6 : Documentation Complète - OBLIGATOIRE

Toute méthode publique DOIT avoir :
```python
def method(self, param: str) -> Type:
    """Description courte
    
    Args:
        param: Description du paramètre
        
    Returns:
        Description du retour
        
    Raises:
        ValueError: Conditions d'erreur
    """
```

## RÈGLE 7 : Tests Avant Modification - OBLIGATOIRE

Avant tout refactoring :
1. Créer test incrémental (ex: test_storage_incremental.py)
2. Valider sur données RÉELLES
3. Nettoyer automatiquement les données de test
4. Vérifier aucune régression

## RÈGLE 8 : Sécurité - INTERDICTIONS ABSOLUES

❌ JAMAIS de secrets/clés API en dur dans le code
❌ JAMAIS de valeur par défaut pour les secrets

✅ TOUJOURS os.getenv() SANS défaut
✅ TOUJOURS valider que la variable existe

```python
# BON
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY manquante dans .env")
```

## EXEMPLES CONCRETS APPLIQUÉS

### ✅ storage.py (2024-10-04)
- Refactoring de 6 méthodes → 2 méthodes (save_data, load_data)
- Réduction de 389 → 180 lignes (-54%)
- API unifiée avec validation stricte

### ✅ finnhub_scraper.py (2024-10-04)
- Suppression clé API hardcodée
- Ajout validation .env obligatoire
- Migration vers storage.save_data()

### ✅ refresh_news.py (2024-10-04)
- Suppression 70 lignes de logique dupliquée
- Migration vers storage.save_data()
- Utilisation API unifiée

## CHECKLIST OBLIGATOIRE AVANT COMMIT

- [ ] Aucune duplication de code (DRY)
- [ ] API cohérente et simple
- [ ] Validation stricte des entrées
- [ ] Documentation complète (docstrings)
- [ ] Tests couvrent le nouveau code
- [ ] Architecture en couches respectée
- [ ] Aucun secret en dur
- [ ] Imports en haut du fichier

## PROCESS DE RÉVISION

Si Cascade détecte :
1. Code dupliqué → Proposer refactoring avec méthode générique
2. Méthodes similaires → Proposer API unifiée
3. Pas de validation → Ajouter ValueError
4. Pas de tests → Créer tests avant modification
5. Secret en dur → Migrer vers .env + validation

Ces règles doivent être appliquées AUTOMATIQUEMENT par Cascade sans qu'il soit nécessaire de les rappeler.
