#!/usr/bin/env python3
"""
Tests pour le composant de filtres
"""

import sys
from pathlib import Path
import unittest

# Ajouter le chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.filters_component import FiltersComponent


class TestFiltersComponent(unittest.TestCase):
    """Tests pour FiltersComponent"""
    
    def setUp(self):
        """Configuration des tests"""
        self.filters = FiltersComponent()
    
    def test_initialization(self):
        """Test de l'initialisation"""
        self.assertIsNotNone(self.filters.ticker_choices, "Les choix de ticker doivent être définis")
        self.assertIsNotNone(self.filters.period_choices, "Les choix de période doivent être définis")
        self.assertIsNotNone(self.filters.analysis_choices, "Les choix d'analyse doivent être définis")
        
        # Vérifier les valeurs par défaut
        self.assertIn("NVDA", self.filters.ticker_choices, "NVDA doit être dans les choix de ticker")
        self.assertIn("SPY", self.filters.ticker_choices, "SPY doit être dans les choix de ticker")
        self.assertIn("7 derniers jours", self.filters.period_choices, "7 derniers jours doit être dans les choix de période")
        self.assertIn("Prix", self.filters.analysis_choices, "Prix doit être dans les choix d'analyse")
        
        print("✅ Composant de filtres initialisé correctement")
    
    def test_create_filters(self):
        """Test de la création des filtres"""
        ticker_radio, analysis_radio, period_radio = self.filters.create_filters()
        
        # Vérifications
        self.assertIsNotNone(ticker_radio, "Le radio ticker ne doit pas être None")
        self.assertIsNotNone(analysis_radio, "Le radio analyse ne doit pas être None")
        self.assertIsNotNone(period_radio, "Le radio période ne doit pas être None")
        
        print("✅ Filtres créés avec succès")
    
    def test_validate_filters_valid(self):
        """Test de la validation avec des filtres valides"""
        # Test avec des valeurs valides
        self.assertTrue(self.filters.validate_filters("NVDA", "Prix", "7 derniers jours"))
        self.assertTrue(self.filters.validate_filters("SPY", "Volume", "1 mois"))
        self.assertTrue(self.filters.validate_filters("NVDA", "Prédiction", "Total (toutes les données)"))
        
        print("✅ Validation des filtres valides réussie")
    
    def test_validate_filters_invalid(self):
        """Test de la validation avec des filtres invalides"""
        # Test avec des valeurs invalides
        self.assertFalse(self.filters.validate_filters("INVALID", "Prix", "7 derniers jours"))
        self.assertFalse(self.filters.validate_filters("NVDA", "INVALID", "7 derniers jours"))
        self.assertFalse(self.filters.validate_filters("NVDA", "Prix", "INVALID"))
        
        print("✅ Validation des filtres invalides réussie")
    
    def test_get_default_values(self):
        """Test des valeurs par défaut"""
        ticker, period, analysis_type = self.filters.get_default_values()
        
        # Vérifications
        self.assertEqual(ticker, "NVDA", "Le ticker par défaut doit être NVDA")
        self.assertEqual(period, "7 derniers jours", "La période par défaut doit être 7 derniers jours")
        self.assertEqual(analysis_type, "Prix", "Le type d'analyse par défaut doit être Prix")
        
        print("✅ Valeurs par défaut correctes")


if __name__ == '__main__':
    unittest.main()
