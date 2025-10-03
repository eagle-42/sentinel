# 📊 Analyse de Performance du Modèle LSTM

**Notebook pour Présentation de Mémoire**

---

## 🎯 Objectif

Analyser les performances du modèle LSTM Sentinel2 pour votre présentation de mémoire avec :
- Métriques de performance détaillées
- Visualisations professionnelles haute résolution
- Tableaux exportables pour inclusion dans le mémoire

---

## 🚀 Utilisation Rapide

### 1. Lancer Jupyter

```bash
cd src/notebooks
uv run jupyter notebook model_performance_analysis.ipynb
```

### 2. Exécuter toutes les cellules

Dans Jupyter : `Cell` → `Run All`

### 3. Récupérer les fichiers générés

Les fichiers suivants seront créés dans `src/notebooks/` :

**Graphiques (PNG 300 DPI):**
- `performance_overview.png` - Vue d'ensemble des performances
- `decisions_vs_price.png` - Décisions vs prix réels
- `returns_distribution.png` - Distribution des rendements
- `cumulative_return.png` - Rendement cumulé

**Tableaux (CSV):**
- `performance_metrics.csv` - Métriques par type de décision
- `validation_results.csv` - Résultats détaillés

---

## 📊 Contenu du Notebook

### Section 1: Chargement des Données
- Charge les décisions de trading (`trading_decisions.json`)
- Charge les prix SPY (`spy_15min.parquet`)
- Affiche un aperçu des données

### Section 2: Calcul des Rendements Futurs
- Calcule les rendements futurs réels (horizon 15 min)
- Statistiques de base (moyenne, volatilité)

### Section 3: Validation des Décisions
- Compare décisions vs prix futurs
- Détermine si chaque décision est correcte
- Règles:
  - **BUY**: Correct si prix monte > 0.1%
  - **SELL**: Correct si prix baisse > 0.1%
  - **HOLD**: Correct si variation < 0.1%

### Section 4: Métriques de Performance
Tableau complet avec:
- Win Rate par type de décision
- Rendement moyen et médian
- Volatilité
- Confiance moyenne du modèle

### Section 5: Visualisations
4 graphiques professionnels:
1. **Distribution des décisions** - Répartition BUY/SELL/HOLD
2. **Décisions vs Prix** - Carte des décisions sur la courbe de prix
3. **Distribution rendements** - Histogramme et boxplots
4. **Rendement cumulé** - Évolution de la performance

### Section 6: Export
Sauvegarde automatique de tous les graphiques et tableaux

### Section 7: Résumé
Résumé synthétique pour présentation orale

---

## 📁 Fichiers Requis

Le notebook utilise ces fichiers (chemins relatifs):
```
../../data/trading/decisions_log/trading_decisions.json
../../data/realtime/prices/spy_15min.parquet
```

**Important**: Le notebook doit être exécuté depuis `src/notebooks/`

---

## 🎨 Personnalisation

### Changer la période d'analyse

Dans la cellule "Graphique 2":
```python
window_days = 7  # Modifier ici (ex: 14 pour 2 semaines)
```

### Changer le seuil de validation

Dans la cellule "Validation":
```python
THRESHOLD_PCT = 0.001  # 0.1% par défaut
```

### Ajuster la taille des graphiques

En début de notebook:
```python
plt.rcParams['figure.figsize'] = (14, 8)  # Modifier ici
```

---

## 📈 Métriques Expliquées

### Win Rate
Pourcentage de décisions correctes. 
- > 50% = Modèle meilleur que hasard
- > 60% = Bon modèle
- > 70% = Excellent modèle

### Sharpe Ratio
Rendement ajusté du risque.
- > 1 = Bon
- > 2 = Très bon
- > 3 = Excellent

### Rendement Cumulé
Performance totale sur la période.
- Positif = Gain net
- Négatif = Perte nette

---

## 🔧 Dépendances

Installées automatiquement via `uv`:
- `pandas` - Manipulation données
- `numpy` - Calculs numériques
- `matplotlib` - Graphiques
- `seaborn` - Visualisations avancées
- `jupyter` - Environnement notebook

---

## ⚠️ Troubleshooting

### Erreur "Fichier non trouvé"
Vérifiez que vous êtes dans `src/notebooks/` et que les données existent:
```bash
ls -la ../../data/trading/decisions_log/trading_decisions.json
ls -la ../../data/realtime/prices/spy_15min.parquet
```

### Erreur "Module not found"
Réinstallez les dépendances:
```bash
uv sync
```

### Graphiques ne s'affichent pas
Dans Jupyter, ajoutez en début de notebook:
```python
%matplotlib inline
```

---

## 📝 Utilisation pour le Mémoire

### 1. Graphiques
Les PNG générés sont en **haute résolution (300 DPI)** et peuvent être directement inclus dans votre mémoire (Word, LaTeX, etc.)

### 2. Tableaux
Les CSV peuvent être ouverts dans Excel et formatés pour inclusion:
```bash
open performance_metrics.csv
```

### 3. Statistiques
Copiez les métriques de la Section 7 (Résumé) pour votre présentation orale.

---

## 📞 Support

Pour toute question sur l'analyse ou modifications spécifiques pour votre présentation, référez-vous au code commenté dans le notebook.

**Bon courage pour votre mémoire ! 🎓**
