#!/usr/bin/env python3
"""
Tests pour le composant graphique
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import unittest

# Ajouter le chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.chart_component import ChartComponent


class TestChartComponent(unittest.TestCase):
    """Tests pour ChartComponent"""
    
    def setUp(self):
        """Configuration des tests"""
        self.chart = ChartComponent()
        
        # Créer des données de test
        self.test_df = pd.DataFrame({
            'DATE': pd.date_range('2024-01-01', periods=100, freq='D'),
            'CLOSE': np.random.randn(100).cumsum() + 100,
            'VOLUME': np.random.randint(1000, 10000, 100),
            'OPEN': np.random.randn(100).cumsum() + 100,
            'HIGH': np.random.randn(100).cumsum() + 105,
            'LOW': np.random.randn(100).cumsum() + 95
        })
    
    def test_create_price_chart(self):
        """Test de la création du graphique de prix"""
        fig = self.chart.create_price_chart(self.test_df, "SPY", "7 derniers jours")
        
        # Vérifications
        self.assertIsNotNone(fig, "Le graphique de prix ne doit pas être None")
        
        # Vérifier que c'est un objet Plotly Figure
        from plotly.graph_objects import Figure
        self.assertIsInstance(fig, Figure, "Le graphique doit être une Figure Plotly")
        
        print("✅ Graphique de prix créé avec succès")
    
    def test_create_volume_chart(self):
        """Test de la création du graphique de volume"""
        fig = self.chart.create_volume_chart(self.test_df, "SPY", "7 derniers jours")
        
        # Vérifications
        self.assertIsNotNone(fig, "Le graphique de volume ne doit pas être None")
        
        from plotly.graph_objects import Figure
        self.assertIsInstance(fig, Figure, "Le graphique doit être une Figure Plotly")
        
        print("✅ Graphique de volume créé avec succès")
    
    def test_create_prediction_chart(self):
        """Test de la création du graphique de prédiction"""
        # Créer des données de prédiction de test
        prediction_data = {
            'predictions': [100, 101, 102, 103, 104],
            'future_dates': pd.date_range('2024-04-10', periods=5, freq='D'),
            'upper_bound': [105, 106, 107, 108, 109],
            'lower_bound': [95, 96, 97, 98, 99],
            'last_date': pd.Timestamp('2024-04-09')
        }
        
        fig = self.chart.create_prediction_chart(self.test_df, prediction_data, "SPY", "7 derniers jours")
        
        # Vérifications
        self.assertIsNotNone(fig, "Le graphique de prédiction ne doit pas être None")
        
        from plotly.graph_objects import Figure
        self.assertIsInstance(fig, Figure, "Le graphique doit être une Figure Plotly")
        
        print("✅ Graphique de prédiction créé avec succès")
    
    def test_create_sentiment_chart(self):
        """Test de la création du graphique de sentiment"""
        fig = self.chart.create_sentiment_chart(self.test_df, "SPY", "7 derniers jours")
        
        # Vérifications
        self.assertIsNotNone(fig, "Le graphique de sentiment ne doit pas être None")
        
        from plotly.graph_objects import Figure
        self.assertIsInstance(fig, Figure, "Le graphique doit être une Figure Plotly")
        
        print("✅ Graphique de sentiment créé avec succès")
    
    def test_create_error_chart(self):
        """Test de la création du graphique d'erreur"""
        fig = self.chart._create_error_chart("Test error message")
        
        # Vérifications
        self.assertIsNotNone(fig, "Le graphique d'erreur ne doit pas être None")
        
        from plotly.graph_objects import Figure
        self.assertIsInstance(fig, Figure, "Le graphique doit être une Figure Plotly")
        
        print("✅ Graphique d'erreur créé avec succès")


if __name__ == '__main__':
    unittest.main()
