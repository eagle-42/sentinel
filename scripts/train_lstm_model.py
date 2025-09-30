#!/usr/bin/env python3
"""
🚀 Script d'entraînement du modèle LSTM pour Sentinel2
Entraîne le modèle LSTM avec les données de features techniques
"""

import sys
from pathlib import Path
import pandas as pd
from loguru import logger

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.prediction import PricePredictor
from constants import CONSTANTS

def train_model(ticker: str = "SPY", epochs: int = 100) -> bool:
    """Entraîne le modèle LSTM pour un ticker donné"""
    try:
        logger.info(f"🚀 Début de l'entraînement LSTM pour {ticker}")
        
        # Charger les données de features
        features_path = CONSTANTS.get_data_path("features", ticker)
        if not features_path.exists():
            logger.error(f"❌ Fichier de features non trouvé: {features_path}")
            return False
        
        logger.info(f"📊 Chargement des features depuis {features_path}")
        features_df = pd.read_parquet(features_path)
        
        # Normaliser les colonnes en majuscules
        features_df.columns = features_df.columns.str.upper()
        
        logger.info(f"✅ Features chargées: {len(features_df)} lignes, {len(features_df.columns)} colonnes")
        
        # Initialiser le prédicteur
        predictor = PricePredictor(ticker)
        
        # Entraîner le modèle
        logger.info(f"🧠 Début de l'entraînement ({epochs} époques)")
        result = predictor.train(features_df, epochs=epochs)
        
        if "error" in result:
            logger.error(f"❌ Erreur d'entraînement: {result['error']}")
            return False
        
        # Sauvegarder le modèle
        model_path = CONSTANTS.get_model_path(ticker) / "version_1" / "model.pkl"
        if predictor.save_model(model_path):
            logger.info(f"✅ Modèle sauvegardé: {model_path}")
            
            # Afficher les métriques
            logger.info(f"📊 Métriques d'entraînement:")
            logger.info(f"   - Époques entraînées: {result['epochs_trained']}")
            logger.info(f"   - Meilleure validation loss: {result['best_val_loss']:.6f}")
            logger.info(f"   - Features utilisées: {len(result['features_used'])}")
            
            return True
        else:
            logger.error("❌ Échec de la sauvegarde du modèle")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'entraînement: {e}")
        return False

def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraîner le modèle LSTM")
    parser.add_argument("--ticker", default="SPY", help="Ticker à entraîner (défaut: SPY)")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'époques (défaut: 100)")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 === ENTRAÎNEMENT LSTM SENTINEL2 ===")
    logger.info(f"📊 Ticker: {args.ticker}")
    logger.info(f"🔄 Époques: {args.epochs}")
    
    success = train_model(args.ticker, args.epochs)
    
    if success:
        logger.info("✅ Entraînement terminé avec succès!")
        return 0
    else:
        logger.error("❌ Échec de l'entraînement")
        return 1

if __name__ == "__main__":
    sys.exit(main())
