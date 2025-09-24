#!/usr/bin/env python3
"""
Tests pour la page d'analyse
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import unittest

# Ajouter le chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pages.analysis_page import AnalysisPage


class TestAnalysisPage(unittest.TestCase):
    """Tests pour AnalysisPage"""
    
    def setUp(self):
        """Configuration des tests"""
        self.analysis_page = AnalysisPage()
        
        # Créer des données de test
        self.test_df = pd.DataFrame({
            'DATE': pd.date_range('2024-01-01', periods=100, freq='D'),
            'CLOSE': np.random.randn(100).cumsum() + 100,
            'VOLUME': np.random.randint(1000, 10000, 100),
            'OPEN': np.random.randn(100).cumsum() + 100,
            'HIGH': np.random.randn(100).cumsum() + 105,
            'LOW': np.random.randn(100).cumsum() + 95
        })
    
    def test_initialization(self):
        """Test de l'initialisation"""
        self.assertIsNotNone(self.analysis_page.filters, "Les filtres doivent être initialisés")
        self.assertIsNotNone(self.analysis_page.chart, "Le composant graphique doit être initialisé")
        self.assertIsNotNone(self.analysis_page.data_service, "Le service de données doit être initialisé")
        self.assertIsNotNone(self.analysis_page.prediction_service, "Le service de prédiction doit être initialisé")
        
        print("✅ Page d'analyse initialisée correctement")
    
    def test_analyze_data_price(self):
        """Test de l'analyse des données pour les prix"""
        result_text, chart = self.analysis_page.analyze_data("NVDA", "Prix", "7 derniers jours")
        
        # Vérifications
        self.assertIsInstance(result_text, str, "Le texte de résultat doit être une chaîne")
        self.assertIsNotNone(chart, "Le graphique ne doit pas être None")
        
        # Vérifier que le texte contient des informations sur NVDA
        self.assertIn("NVDA", result_text, "Le texte doit contenir le ticker")
        
        print("✅ Analyse des prix réussie")
    
    def test_analyze_data_volume(self):
        """Test de l'analyse des données pour le volume"""
        result_text, chart = self.analysis_page.analyze_data("SPY", "Volume", "1 mois")
        
        # Vérifications
        self.assertIsInstance(result_text, str, "Le texte de résultat doit être une chaîne")
        self.assertIsNotNone(chart, "Le graphique ne doit pas être None")
        
        print("✅ Analyse du volume réussie")
    
    def test_analyze_data_prediction(self):
        """Test de l'analyse des données pour les prédictions"""
        result_text, chart = self.analysis_page.analyze_data("SPY", "Prédiction", "7 derniers jours")
        
        # Vérifications
        self.assertIsInstance(result_text, str, "Le texte de résultat doit être une chaîne")
        self.assertIsNotNone(chart, "Le graphique ne doit pas être None")
        
        print("✅ Analyse des prédictions réussie")
    
    def test_analyze_data_sentiment(self):
        """Test de l'analyse des données pour le sentiment"""
        result_text, chart = self.analysis_page.analyze_data("NVDA", "Sentiment", "3 mois")
        
        # Vérifications
        self.assertIsInstance(result_text, str, "Le texte de résultat doit être une chaîne")
        self.assertIsNotNone(chart, "Le graphique ne doit pas être None")
        
        print("✅ Analyse du sentiment réussie")
    
    def test_create_result_text(self):
        """Test de la création du texte de résultat"""
        result_text = self.analysis_page._create_result_text("SPY", "7 derniers jours", "Prix", self.test_df)
        
        # Vérifications
        self.assertIsInstance(result_text, str, "Le texte de résultat doit être une chaîne")
        self.assertIn("SPY", result_text, "Le texte doit contenir le ticker")
        self.assertIn("7 derniers jours", result_text, "Le texte doit contenir la période")
        self.assertIn("Prix", result_text, "Le texte doit contenir le type d'analyse")
        
        print("✅ Création du texte de résultat réussie")


if __name__ == '__main__':
    unittest.main()
