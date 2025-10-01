from config.project_config import DATA_DIR, FEATURES_DIR, MODELS_DIR, PROJECT_ROOT, YFINANCE_DIR
from config.unified_config import get_config, get_data_file_path, get_model_path

#!/usr/bin/env python3
"""
Script de déploiement pour la fusion adaptative
Phase: Infrastructure - Intégration - Calibration
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveFusionDeployer:
    """Déployeur pour le système de fusion adaptative"""

    def __init__(self, base_path: str = "PROJECT_ROOT"):
        self.base_path = Path(base_path)
        self.deployment_log = []

    def log_step(self, step: str, status: str, details: str = ""):
        """Enregistre une étape du déploiement"""
        log_entry = {"timestamp": datetime.now().isoformat(), "step": step, "status": status, "details": details}
        self.deployment_log.append(log_entry)
        logger.info(f"[{status.upper()}] {step}: {details}")

    def create_directory_structure(self):
        """Crée la structure de répertoires nécessaire"""
        self.log_step("Création structure répertoires", "start", "Création des répertoires pour la fusion adaptative")

        directories = [
            "PREDICTIONS_DIR/nvda",
            "PREDICTIONS_DIR/spy",
            "data/trading/adaptive_fusion",
            "data/trading/calibration",
            "logs/adaptive_fusion",
            "config/adaptive_fusion",
        ]

        for directory in directories:
            dir_path = self.base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.log_step(f"Répertoire {directory}", "created", str(dir_path))

        self.log_step("Création structure répertoires", "completed", f"{len(directories)} répertoires créés")

    def create_config_files(self):
        """Crée les fichiers de configuration"""
        self.log_step("Création fichiers config", "start", "Création des fichiers de configuration")

        # Configuration par défaut de la fusion adaptative
        default_config = {
            "fusion_config": {
                "lookback_window": 100,
                "min_samples": 50,
                "min_correlation": 0.1,
                "max_correlation": 0.9,
                "base_price_weight": 0.6,
                "base_sentiment_weight": 0.4,
                "low_volatility_threshold": 0.02,
                "high_volatility_threshold": 0.05,
                "low_volume_threshold": 0.8,
                "high_volume_threshold": 1.2,
                "regularization_factor": 0.1,
                "max_weight_change": 0.2,
            },
            "price_predictor": {"base_path": "PREDICTIONS_DIR", "auto_load_models": True, "model_validation": True},
            "calibration": {
                "walk_forward_months": 6,
                "test_months": 1,
                "optimization_metric": "composite_score",
                "max_iterations": 1000,
            },
            "gui": {"default_fusion_mode": "fixed", "auto_refresh_interval": 5, "show_debug_info": True},
        }

        config_file = self.base_path / "config/adaptive_fusion/config.json"
        with open(config_file, "w") as f:
            json.dump(default_config, f, indent=2)

        self.log_step("Configuration par défaut", "created", str(config_file))

        # Configuration des modèles de prédiction
        model_config = {
            "nvda": {"enabled": True, "version": "latest", "auto_update": True, "confidence_threshold": 0.7},
            "spy": {"enabled": True, "version": "latest", "auto_update": True, "confidence_threshold": 0.7},
        }

        model_config_file = self.base_path / "config/adaptive_fusion/models.json"
        with open(model_config_file, "w") as f:
            json.dump(model_config, f, indent=2)

        self.log_step("Configuration modèles", "created", str(model_config_file))

        self.log_step("Création fichiers config", "completed", "Fichiers de configuration créés")

    def create_deployment_scripts(self):
        """Crée les scripts de déploiement et de maintenance"""
        self.log_step("Création scripts déploiement", "start", "Création des scripts de déploiement")

        # Script de démarrage
        startup_script = """#!/bin/bash
# Script de démarrage pour la fusion adaptative

echo "🚀 Démarrage du système de fusion adaptative..."

# Vérifier les dépendances
python -c "import torch, pandas, numpy, scipy" || {
    echo "❌ Dépendances manquantes. Installez avec: pip install torch pandas numpy scipy"
    exit 1
}

# Créer les répertoires si nécessaire
mkdir -p PREDICTIONS_DIR/{nvda,spy}
mkdir -p data/trading/adaptive_fusion
mkdir -p logs/adaptive_fusion

# Démarrer les services
echo "📊 Démarrage des services de fusion adaptative..."
python scripts/test_fusion_simple.py

echo "✅ Système de fusion adaptative démarré!"
"""

        startup_file = self.base_path / "scripts/start_adaptive_fusion.sh"
        with open(startup_file, "w") as f:
            f.write(startup_script)
        startup_file.chmod(0o755)

        self.log_step("Script de démarrage", "created", str(startup_file))

        # Script de test
        test_script = """#!/bin/bash
# Script de test pour la fusion adaptative

echo "🧪 Tests du système de fusion adaptative..."

# Test des composants de base
python scripts/test_fusion_simple.py

# Test de l'interface GUI (optionnel)
if [ "$1" = "--gui" ]; then
    echo "🖥️  Démarrage de l'interface GUI..."
    python scripts/gui_fusion_test.py
fi

echo "✅ Tests terminés!"
"""

        test_file = self.base_path / "scripts/test_adaptive_fusion_system.sh"
        with open(test_file, "w") as f:
            f.write(test_script)
        test_file.chmod(0o755)

        self.log_step("Script de test", "created", str(test_file))

        # Script de calibration
        calibration_script = """#!/bin/bash
# Script de calibration pour la fusion adaptative

echo "🎯 Démarrage de la calibration de la fusion adaptative..."

# Vérifier la présence des données
if [ ! -f "data/dataset/3-kagglefull/nvda_15min_2014_2025.parquet" ]; then
    echo "❌ Données NVDA manquantes. Veuillez les télécharger d'abord."
    exit 1
fi

if [ ! -f "data/dataset/3-kagglefull/spy_15min_2014_2024.parquet" ]; then
    echo "❌ Données SPY manquantes. Veuillez les télécharger d'abord."
    exit 1
fi

# Lancer la calibration
python scripts/calibrate_adaptive_fusion.py

echo "✅ Calibration terminée!"
"""

        calibration_file = self.base_path / "scripts/calibrate_adaptive_fusion_system.sh"
        with open(calibration_file, "w") as f:
            f.write(calibration_script)
        calibration_file.chmod(0o755)

        self.log_step("Script de calibration", "created", str(calibration_file))

        self.log_step("Création scripts déploiement", "completed", "Scripts de déploiement créés")

    def create_documentation(self):
        """Crée la documentation du système"""
        self.log_step("Création documentation", "start", "Création de la documentation")

        # README pour la fusion adaptative
        readme_content = """# Système de Fusion Adaptative

## Vue d'ensemble

Le système de fusion adaptative combine intelligemment les signaux de prix et de sentiment pour prendre des décisions de trading optimisées. Il s'adapte dynamiquement aux conditions de marché en ajustant les poids des signaux.

## Fonctionnalités

### 🔄 Fusion Adaptative
- **Détection de régime de marché** : Volatilité, volume, tendance
- **Ajustement dynamique des poids** : Prix vs Sentiment
- **Normalisation des signaux** : Z-scores et statistiques glissantes
- **Régularisation** : Évite l'overfitting

### 🔧 Fusion Fixe
- **Logique traditionnelle** : Règles fixes basées sur sentiment et prix
- **Compatibilité** : Maintient la logique existante
- **Comparaison** : Permet de comparer avec la fusion adaptative

### 📊 Price Predictor
- **Chargement automatique** : Récupère les modèles depuis `PREDICTIONS_DIR`
- **Support multi-tickers** : NVDA, SPY, etc.
- **Validation** : Vérifie l'intégrité des modèles

## Structure

```
src/
├── features/
│   ├── adaptive_fusion.py      # Système de fusion adaptative
│   ├── rolling_stats.py        # Statistiques glissantes
│   └── sentiment_window.py     # Analyse de sentiment étendue
├── models/predictions/
│   └── price_predictor.py      # Prédicteur de prix
└── pipeline/
    └── decision_flow.py        # Flux de décision mis à jour
```

## Utilisation

### Test Simple
```bash
python scripts/test_fusion_simple.py
```

### Interface GUI
```bash
python scripts/gui_fusion_test.py
```

### Calibration
```bash
python scripts/calibrate_adaptive_fusion_system.sh
```

## Configuration

Les paramètres sont dans `config/adaptive_fusion/config.json` :

- `lookback_window` : Fenêtre de lookback (défaut: 100)
- `base_price_weight` : Poids de base pour le prix (défaut: 0.6)
- `base_sentiment_weight` : Poids de base pour le sentiment (défaut: 0.4)
- `min_correlation` : Corrélation minimum (défaut: 0.1)
- `max_correlation` : Corrélation maximum (défaut: 0.9)

## Modes de Fusion

### Mode Fixe
- Logique traditionnelle
- Règles basées sur des seuils fixes
- Compatible avec l'existant

### Mode Adaptatif
- Ajustement dynamique des poids
- Détection de régime de marché
- Normalisation des signaux
- Régularisation

## Métriques

- **Accuracy** : Précision des décisions
- **Sharpe Ratio** : Rendement ajusté du risque
- **Max Drawdown** : Perte maximale
- **Volatilité** : Écart-type des retours

## Déploiement

1. **Infrastructure** : Création des répertoires et configurations
2. **Intégration** : Tests et validation des composants
3. **Calibration** : Optimisation des paramètres avec walk-forward

## Support

Pour toute question ou problème, consultez les logs dans `logs/adaptive_fusion/`.
"""

        readme_file = self.base_path / "src/features/README_ADAPTIVE_FUSION.md"
        with open(readme_file, "w") as f:
            f.write(readme_content)

        self.log_step("Documentation", "created", str(readme_file))

        self.log_step("Création documentation", "completed", "Documentation créée")

    def validate_deployment(self):
        """Valide le déploiement"""
        self.log_step("Validation déploiement", "start", "Validation du déploiement")

        # Vérifier les fichiers créés
        required_files = [
            "src/features/adaptive_fusion.py",
            "src/features/rolling_stats.py",
            "src/models/predictions/price_predictor.py",
            "scripts/test_fusion_simple.py",
            "scripts/gui_fusion_test.py",
            "scripts/calibrate_adaptive_fusion.py",
            "config/adaptive_fusion/config.json",
        ]

        missing_files = []
        for file_path in required_files:
            if not (self.base_path / file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            self.log_step("Validation déploiement", "failed", f"Fichiers manquants: {missing_files}")
            return False

        # Tester l'import des modules
        try:
            import sys

            sys.path.append(str(self.base_path))

            from src.core.adaptive_fusion.fusion_engine import AdaptiveFusion, FusionMode
            from src.core.adaptive_fusion.rolling_stats import RollingStats
            from src.core.price_predictor.predictor import PricePredictor

            self.log_step("Import modules", "success", "Tous les modules s'importent correctement")

        except Exception as e:
            self.log_step("Import modules", "failed", f"Erreur d'import: {e}")
            return False

        self.log_step("Validation déploiement", "completed", "Déploiement validé avec succès")
        return True

    def save_deployment_log(self):
        """Sauvegarde le log de déploiement"""
        log_file = self.base_path / "logs/adaptive_fusion/deployment_log.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "w") as f:
            json.dump(self.deployment_log, f, indent=2)

        self.log_step("Sauvegarde log", "completed", f"Log sauvegardé dans {log_file}")

    def deploy(self):
        """Lance le déploiement complet"""
        print("🚀 Démarrage du déploiement de la fusion adaptative...")
        print("=" * 60)

        try:
            # Étape 1: Structure de répertoires
            self.create_directory_structure()

            # Étape 2: Fichiers de configuration
            self.create_config_files()

            # Étape 3: Scripts de déploiement
            self.create_deployment_scripts()

            # Étape 4: Documentation
            self.create_documentation()

            # Étape 5: Validation
            if self.validate_deployment():
                print("\n✅ Déploiement terminé avec succès!")
                print("\n📋 Prochaines étapes:")
                print("   1. Tester le système: python scripts/test_fusion_simple.py")
                print("   2. Lancer l'interface GUI: python scripts/gui_fusion_test.py")
                print("   3. Calibrer les paramètres: python scripts/calibrate_adaptive_fusion_system.sh")
                print("   4. Intégrer dans le flux principal: Modifier decision_flow.py")
            else:
                print("\n❌ Déploiement échoué. Vérifiez les logs.")

            # Sauvegarder le log
            self.save_deployment_log()

        except Exception as e:
            self.log_step("Déploiement", "failed", f"Erreur: {e}")
            print(f"\n❌ Erreur lors du déploiement: {e}")
            import traceback

            traceback.print_exc()


def main():
    """Fonction principale"""
    deployer = AdaptiveFusionDeployer()
    deployer.deploy()


if __name__ == "__main__":
    main()
