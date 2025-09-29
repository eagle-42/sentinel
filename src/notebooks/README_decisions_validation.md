# 📊 Validation des Décisions de Trading

Ce script valide les décisions de trading par rapport aux prix futurs et génère des métriques de performance.

## 🎯 Objectif

Créer un graphique unique qui valide les décisions de trading en comparant :
- Les décisions prises (BUY/SELL/HOLD)
- Les prix futurs réels
- La performance de chaque décision

## 📁 Fichiers

- `decisions_vs_future_price.py` - Script principal
- `test_decisions_validation.py` - Script de test
- `decisions_vs_future_price.png` - Graphique généré
- `decisions_vs_future_price_summary.csv` - Données et métriques

## 🚀 Utilisation

### Commande de base
```bash
uv run python src/notebooks/decisions_vs_future_price.py
```

### Options disponibles
```bash
uv run python src/notebooks/decisions_vs_future_price.py \
  --decisions data/trading/decisions_log/trading_decisions.json \
  --prices data/realtime/prices/spy_15min.parquet \
  --horizon-steps 1 \
  --window-days 14 \
  --output-dir src/notebooks
```

### Paramètres

| Option | Description | Défaut |
|--------|-------------|---------|
| `--decisions` | Fichier de décisions JSON | `data/trading/decisions_log/trading_decisions.json` |
| `--prices` | Fichier de prix (Parquet/CSV) | `data/realtime/prices/spy_15min.parquet` |
| `--horizon-steps` | Pas dans le futur pour calculer le rendement | `1` |
| `--window-days` | Fenêtre d'affichage (jours, 0=tout) | `14` |
| `--time-col` | Nom de la colonne de temps | Auto-détection |
| `--price-col` | Nom de la colonne de prix | Auto-détection |
| `--output-dir` | Répertoire de sortie | `src/notebooks` |

## 📊 Données d'entrée

### Décisions (JSON)
```json
[
  {
    "ticker": "SPY",
    "timestamp": "2025-09-24T07:57:41.906704+00:00",
    "decision": "HOLD",
    "confidence": 0.9969224699700235,
    "fused_signal": 0.0030775300299764673
  }
]
```

### Prix (Parquet/CSV)
Colonnes requises :
- `timestamp` ou `date` - Timestamp des prix
- `close` - Prix de clôture

## 📈 Sorties

### Graphique PNG
- Courbe de prix SPY en noir
- Marqueurs de décisions :
  - 🔺 BUY (triangle haut)
  - 🔻 SELL (triangle bas)  
  - ⚪ HOLD (rond)
- Couleurs :
  - 🟢 Vert = Décision correcte
  - 🔴 Rouge = Décision incorrecte

### CSV de résumé
Colonnes :
- `timestamp` - Timestamp de la décision
- `decision` - Type de décision (BUY/SELL/HOLD)
- `confidence` - Confiance de la décision
- `fused_signal` - Signal fusionné
- `close` - Prix au moment de la décision
- `ret_next` - Rendement futur
- `is_correct` - Décision correcte (True/False)

### KPIs par type de décision
- `n` - Nombre de décisions
- `win_rate` - Taux de réussite
- `mean_ret_next` - Rendement moyen
- `median_ret_next` - Rendement médian
- `mean_confidence` - Confiance moyenne
- `mean_fused_signal` - Signal fusionné moyen

## 🔍 Logique de validation

### Règles de validation
- **BUY** : Correct si `ret_next > 0`
- **SELL** : Correct si `ret_next < 0`
- **HOLD** : Correct si `|ret_next| < médiane(|ret_next|)`

### Calcul du rendement futur
```python
ret_next = close(t+h) / close(t) - 1
```
Où `h` est le nombre de pas dans le futur (`horizon-steps`).

## 🧪 Tests

Exécuter tous les tests :
```bash
uv run python src/notebooks/test_decisions_validation.py
```

Tests inclus :
- Test par défaut
- Test avec horizon 2 pas
- Test avec fenêtre 7 jours
- Test avec données historiques
- Test avec répertoire personnalisé

## 📊 Exemple de résultats

```
📊 MÉTRIQUES DE PERFORMANCE:
------------------------------------------------------------
  HOLD:   7 décisions | Win Rate:  28.6% | Ret Moyen:  -0.000 | Conf Moy:  0.949
```

## ⚠️ Contraintes

- Gestion automatique des timezones (UTC)
- Auto-détection des colonnes de temps et prix
- Filtrage automatique SPY si présent
- Gestion d'erreurs avec codes de sortie appropriés
- Support Parquet et CSV

## 🔧 Dépendances

- `pandas` - Manipulation de données
- `numpy` - Calculs numériques
- `matplotlib` - Graphiques
- `pyarrow` - Support Parquet
- `argparse` - Interface CLI

## 📝 Notes

- Le script gère automatiquement les différences de timezone
- Les décisions sont alignées avec le prix le plus proche dans le temps
- La fenêtre d'affichage peut être ajustée pour se concentrer sur des périodes récentes
- Les métriques sont calculées séparément pour chaque type de décision

