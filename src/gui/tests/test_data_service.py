#!/usr/bin/env python3
"""
Tests pour le service de données
"""

import sys
import pandas as pd
from pathlib import Path
import unittest
from datetime import datetime, timedelta

# Ajouter le chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.data_service import DataService


class TestDataService(unittest.TestCase):
    """Tests pour DataService"""
    
    def setUp(self):
        """Configuration des tests"""
        self.data_service = DataService()
    
    def test_load_data_spy(self):
        """Test du chargement des données SPY"""
        df = self.data_service.load_data("SPY")
        
        # Vérifications de base
        self.assertFalse(df.empty, "Les données SPY ne doivent pas être vides")
        self.assertIn('DATE', df.columns, "La colonne DATE doit exister")
        self.assertIn('CLOSE', df.columns, "La colonne CLOSE doit exister")
        self.assertIn('VOLUME', df.columns, "La colonne VOLUME doit exister")
        
        # Vérifier le type de la colonne DATE
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['DATE']), 
                       "La colonne DATE doit être de type datetime")
        
        print(f"✅ Données SPY chargées: {len(df)} lignes")
        print(f"   Colonnes: {df.columns.tolist()}")
        print(f"   Période: {df['DATE'].min()} à {df['DATE'].max()}")
    
    def test_load_data_nvda(self):
        """Test du chargement des données NVDA"""
        df = self.data_service.load_data("NVDA")
        
        # Vérifications de base
        self.assertFalse(df.empty, "Les données NVDA ne doivent pas être vides")
        self.assertIn('DATE', df.columns, "La colonne DATE doit exister")
        
        print(f"✅ Données NVDA chargées: {len(df)} lignes")
    
    def test_filter_by_period_7_days(self):
        """Test du filtrage sur 7 derniers jours"""
        df = self.data_service.load_data("SPY")
        filtered_df = self.data_service.filter_by_period(df, "7 derniers jours")
        
        # Vérifications
        self.assertFalse(filtered_df.empty, "Le filtrage 7 jours ne doit pas être vide")
        self.assertLessEqual(len(filtered_df), len(df), "Le filtrage doit réduire le nombre de lignes")
        
        # Vérifier que les dates sont dans la bonne plage
        today = pd.Timestamp.now()
        seven_days_ago = today - pd.Timedelta(days=7)
        
        # Convertir les dates en timezone naive pour la comparaison
        filtered_dates = pd.to_datetime(filtered_df['DATE']).dt.tz_localize(None)
        seven_days_ago_naive = seven_days_ago.tz_localize(None)
        
        self.assertTrue(all(filtered_dates >= seven_days_ago_naive), 
                       "Toutes les dates filtrées doivent être dans les 7 derniers jours")
        
        print(f"✅ Filtrage 7 jours: {len(filtered_df)} lignes")
    
    def test_filter_by_period_1_month(self):
        """Test du filtrage sur 1 mois"""
        df = self.data_service.load_data("SPY")
        filtered_df = self.data_service.filter_by_period(df, "1 mois")
        
        self.assertFalse(filtered_df.empty, "Le filtrage 1 mois ne doit pas être vide")
        print(f"✅ Filtrage 1 mois: {len(filtered_df)} lignes")
    
    def test_filter_by_period_total(self):
        """Test du filtrage sur toutes les données"""
        df = self.data_service.load_data("SPY")
        filtered_df = self.data_service.filter_by_period(df, "Total (toutes les données)")
        
        # Pour "Total", on doit avoir toutes les données
        self.assertEqual(len(filtered_df), len(df), "Le filtrage Total doit garder toutes les données")
        print(f"✅ Filtrage Total: {len(filtered_df)} lignes")
    
    def test_get_price_data(self):
        """Test de la méthode get_price_data"""
        df = self.data_service.get_price_data("SPY", "7 derniers jours")
        
        self.assertFalse(df.empty, "get_price_data ne doit pas retourner de données vides")
        self.assertIn('DATE', df.columns, "get_price_data doit inclure la colonne DATE")
        self.assertIn('CLOSE', df.columns, "get_price_data doit inclure la colonne CLOSE")
        
        print(f"✅ get_price_data: {len(df)} lignes")


if __name__ == '__main__':
    unittest.main()
