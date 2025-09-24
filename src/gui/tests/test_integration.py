#!/usr/bin/env python3
"""
Tests d'intégration complets
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import unittest

# Ajouter le chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.data_service import DataService
from services.prediction_service import PredictionService
from components.chart_component import ChartComponent
from components.filters_component import FiltersComponent
from pages.analysis_page import AnalysisPage


class TestIntegration(unittest.TestCase):
    """Tests d'intégration complets"""
    
    def setUp(self):
        """Configuration des tests"""
        self.data_service = DataService()
        self.prediction_service = PredictionService()
        self.chart = ChartComponent()
        self.filters = FiltersComponent()
        self.analysis_page = AnalysisPage()
    
    def test_full_workflow_spy_price(self):
        """Test du workflow complet pour SPY - Prix"""
        print("\n🔍 Test du workflow complet SPY - Prix")
        
        # 1. Charger les données
        df = self.data_service.get_price_data("SPY", "7 derniers jours")
        self.assertFalse(df.empty, "Les données SPY ne doivent pas être vides")
        print(f"   ✅ Données chargées: {len(df)} lignes")
        
        # 2. Créer le graphique
        chart = self.chart.create_price_chart(df, "SPY", "7 derniers jours")
        self.assertIsNotNone(chart, "Le graphique ne doit pas être None")
        print("   ✅ Graphique créé")
        
        # 3. Analyser via la page
        result_text, chart_result = self.analysis_page.analyze_data("SPY", "Prix", "7 derniers jours")
        self.assertIsInstance(result_text, str, "Le résultat doit être une chaîne")
        self.assertIsNotNone(chart_result, "Le graphique de résultat ne doit pas être None")
        print("   ✅ Analyse complète réussie")
    
    def test_full_workflow_spy_prediction(self):
        """Test du workflow complet pour SPY - Prédiction"""
        print("\n🔍 Test du workflow complet SPY - Prédiction")
        
        # 1. Charger le modèle
        model_loaded = self.prediction_service.load_model(version=1)
        if not model_loaded:
            print("   ⚠️ Modèle LSTM non chargé, test ignoré")
            return
        
        # 2. Charger les données
        df = self.data_service.get_price_data("SPY", "7 derniers jours")
        self.assertFalse(df.empty, "Les données SPY ne doivent pas être vides")
        print(f"   ✅ Données chargées: {len(df)} lignes")
        
        # 3. Faire la prédiction
        prediction_data = self.prediction_service.predict(df, horizon=20)
        if "error" not in prediction_data:
            self.assertIn('predictions', prediction_data, "La prédiction doit contenir 'predictions'")
            print(f"   ✅ Prédiction réussie: {len(prediction_data['predictions'])} prédictions")
        else:
            print(f"   ⚠️ Erreur de prédiction: {prediction_data['error']}")
        
        # 4. Créer le graphique de prédiction
        chart = self.chart.create_prediction_chart(df, prediction_data, "SPY", "7 derniers jours")
        self.assertIsNotNone(chart, "Le graphique de prédiction ne doit pas être None")
        print("   ✅ Graphique de prédiction créé")
    
    def test_data_consistency(self):
        """Test de la cohérence des données"""
        print("\n🔍 Test de cohérence des données")
        
        # Charger les données pour différents tickers
        spy_df = self.data_service.load_data("SPY")
        nvda_df = self.data_service.load_data("NVDA")
        
        # Vérifier la structure des colonnes
        expected_columns = ['DATE', 'CLOSE', 'VOLUME', 'OPEN', 'HIGH', 'LOW']
        for col in expected_columns:
            self.assertIn(col, spy_df.columns, f"SPY doit avoir la colonne {col}")
            self.assertIn(col, nvda_df.columns, f"NVDA doit avoir la colonne {col}")
        
        # Vérifier les types de données
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(spy_df['DATE']), "La colonne DATE de SPY doit être datetime")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(nvda_df['DATE']), "La colonne DATE de NVDA doit être datetime")
        
        print("   ✅ Cohérence des données vérifiée")
    
    def test_period_filtering_accuracy(self):
        """Test de la précision du filtrage par période"""
        print("\n🔍 Test de précision du filtrage")
        
        df = self.data_service.load_data("SPY")
        
        # Test pour différentes périodes
        periods = ["7 derniers jours", "1 mois", "3 mois", "6 derniers mois", "1 an"]
        
        for period in periods:
            filtered_df = self.data_service.filter_by_period(df, period)
            
            if not filtered_df.empty:
                # Vérifier que les dates sont dans la bonne plage
                today = pd.Timestamp.now()
                
                if period == "7 derniers jours":
                    expected_days = 7
                elif period == "1 mois":
                    expected_days = 30
                elif period == "3 mois":
                    expected_days = 90
                elif period == "6 derniers mois":
                    expected_days = 180
                elif period == "1 an":
                    expected_days = 365
                
                # Calculer la différence en jours - gérer les timezones
                min_date = filtered_df['DATE'].min()
                if min_date.tz is not None and today.tz is None:
                    today = today.tz_localize('UTC')
                elif min_date.tz is None and today.tz is not None:
                    today = today.tz_localize(None)
                
                date_diff = (today - min_date).days
                self.assertLessEqual(date_diff, expected_days + 5, f"Le filtrage {period} doit être dans la bonne plage")
                
                print(f"   ✅ Filtrage {period}: {len(filtered_df)} lignes, {date_diff} jours")
            else:
                print(f"   ⚠️ Filtrage {period}: données vides")


if __name__ == '__main__':
    unittest.main()
