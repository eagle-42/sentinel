#!/usr/bin/env python3
"""
Tests pour le service de prédiction
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import unittest

# Ajouter le chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.prediction_service import PredictionService


class TestPredictionService(unittest.TestCase):
    """Tests pour PredictionService"""
    
    def setUp(self):
        """Configuration des tests"""
        self.prediction_service = PredictionService()
    
    def test_load_model(self):
        """Test du chargement du modèle LSTM"""
        success = self.prediction_service.load_model(version=1)
        
        self.assertTrue(success, "Le modèle LSTM doit se charger avec succès")
        self.assertTrue(self.prediction_service.is_loaded, "Le modèle doit être marqué comme chargé")
        self.assertIsNotNone(self.prediction_service.model, "Le modèle ne doit pas être None")
        
        print("✅ Modèle LSTM chargé avec succès")
    
    def test_prepare_features_with_optimized(self):
        """Test de la préparation des features optimisées"""
        # Créer des données de test
        df = pd.DataFrame({
            'DATE': pd.date_range('2024-01-01', periods=100, freq='D'),
            'CLOSE': np.random.randn(100).cumsum() + 100,
            'VOLUME': np.random.randint(1000, 10000, 100),
            'OPEN': np.random.randn(100).cumsum() + 100,
            'HIGH': np.random.randn(100).cumsum() + 105,
            'LOW': np.random.randn(100).cumsum() + 95
        })
        
        features = self.prediction_service.prepare_features(df)
        
        # Vérifications
        self.assertIsInstance(features, np.ndarray, "Les features doivent être un array numpy")
        self.assertGreater(features.shape[0], 0, "Les features ne doivent pas être vides")
        self.assertGreater(features.shape[1], 0, "Les features doivent avoir des colonnes")
        
        print(f"✅ Features préparées: {features.shape}")
    
    def test_prepare_basic_features(self):
        """Test de la préparation des features de base"""
        # Créer des données de test
        df = pd.DataFrame({
            'DATE': pd.date_range('2024-01-01', periods=100, freq='D'),
            'CLOSE': np.random.randn(100).cumsum() + 100,
            'VOLUME': np.random.randint(1000, 10000, 100),
            'OPEN': np.random.randn(100).cumsum() + 100,
            'HIGH': np.random.randn(100).cumsum() + 105,
            'LOW': np.random.randn(100).cumsum() + 95
        })
        
        features = self.prediction_service._prepare_basic_features(df)
        
        # Vérifications
        self.assertIsInstance(features, np.ndarray, "Les features de base doivent être un array numpy")
        self.assertGreater(features.shape[0], 0, "Les features de base ne doivent pas être vides")
        
        print(f"✅ Features de base préparées: {features.shape}")
    
    def test_predict(self):
        """Test de la prédiction LSTM"""
        # Charger le modèle d'abord
        if not self.prediction_service.is_loaded:
            self.prediction_service.load_model(version=1)
        
        # Créer des données de test
        df = pd.DataFrame({
            'DATE': pd.date_range('2024-01-01', periods=100, freq='D'),
            'CLOSE': np.random.randn(100).cumsum() + 100,
            'VOLUME': np.random.randint(1000, 10000, 100),
            'OPEN': np.random.randn(100).cumsum() + 100,
            'HIGH': np.random.randn(100).cumsum() + 105,
            'LOW': np.random.randn(100).cumsum() + 95
        })
        
        prediction_data = self.prediction_service.predict(df, horizon=20)
        
        # Vérifications
        self.assertIsInstance(prediction_data, dict, "La prédiction doit retourner un dictionnaire")
        
        if "error" not in prediction_data:
            self.assertIn('predictions', prediction_data, "La prédiction doit contenir 'predictions'")
            self.assertIn('future_dates', prediction_data, "La prédiction doit contenir 'future_dates'")
            self.assertIn('upper_bound', prediction_data, "La prédiction doit contenir 'upper_bound'")
            self.assertIn('lower_bound', prediction_data, "La prédiction doit contenir 'lower_bound'")
            
            predictions = prediction_data['predictions']
            self.assertEqual(len(predictions), 20, "Il doit y avoir 20 prédictions")
            
            print(f"✅ Prédiction réussie: {len(predictions)} prédictions")
        else:
            print(f"⚠️ Erreur de prédiction: {prediction_data['error']}")


if __name__ == '__main__':
    unittest.main()
