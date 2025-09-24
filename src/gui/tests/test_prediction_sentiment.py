#!/usr/bin/env python3
"""
Test spécifique pour la prédiction et le sentiment
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ajouter le chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent))

from services.data_service import DataService
from services.prediction_service import PredictionService
from components.chart_component import ChartComponent
from pages.analysis_page import AnalysisPage

def test_prediction_spy():
    """Test de la prédiction SPY"""
    print("🔍 Test de la prédiction SPY...")
    
    # Initialiser les services
    data_service = DataService()
    prediction_service = PredictionService()
    chart = ChartComponent()
    analysis_page = AnalysisPage()
    
    # Charger le modèle
    if not prediction_service.load_model(version=1):
        print("❌ Impossible de charger le modèle LSTM")
        return False
    
    # Charger les données SPY
    df = data_service.get_price_data("SPY", "7 derniers jours")
    print(f"✅ Données SPY chargées: {len(df)} lignes")
    
    # Faire la prédiction
    prediction_data = prediction_service.predict(df, horizon=20)
    
    if "error" in prediction_data:
        print(f"❌ Erreur de prédiction: {prediction_data['error']}")
        return False
    
    print(f"✅ Prédiction réussie: {len(prediction_data['predictions'])} prédictions")
    print(f"   Période: {prediction_data['future_dates'][0]} à {prediction_data['future_dates'][-1]}")
    print(f"   Prix actuel: {df['CLOSE'].iloc[-1]:.2f}")
    print(f"   Prix prédit (jour 1): {prediction_data['predictions'][0]:.2f}")
    print(f"   Prix prédit (jour 20): {prediction_data['predictions'][-1]:.2f}")
    
    # Créer le graphique de prédiction
    try:
        chart_fig = chart.create_prediction_chart(df, prediction_data, "SPY", "7 derniers jours")
        print("✅ Graphique de prédiction créé avec succès")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la création du graphique: {e}")
        return False

def test_sentiment_spy():
    """Test du sentiment SPY"""
    print("\n🔍 Test du sentiment SPY...")
    
    # Initialiser les services
    data_service = DataService()
    chart = ChartComponent()
    analysis_page = AnalysisPage()
    
    # Charger les données SPY
    df = data_service.get_price_data("SPY", "7 derniers jours")
    print(f"✅ Données SPY chargées: {len(df)} lignes")
    
    # Créer le graphique de sentiment
    try:
        chart_fig = chart.create_sentiment_chart(df, "SPY", "7 derniers jours")
        print("✅ Graphique de sentiment créé avec succès")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la création du graphique de sentiment: {e}")
        return False

def test_analysis_page():
    """Test de la page d'analyse complète"""
    print("\n🔍 Test de la page d'analyse...")
    
    analysis_page = AnalysisPage()
    
    # Test prédiction SPY
    result_text, chart = analysis_page.analyze_data("SPY", "Prédiction", "7 derniers jours")
    print(f"✅ Analyse prédiction SPY: {len(result_text)} caractères")
    
    # Test sentiment SPY
    result_text, chart = analysis_page.analyze_data("SPY", "Sentiment", "7 derniers jours")
    print(f"✅ Analyse sentiment SPY: {len(result_text)} caractères")
    
    return True

if __name__ == "__main__":
    print("🧪 Test des fonctionnalités de prédiction et sentiment")
    print("=" * 60)
    
    # Tests
    pred_success = test_prediction_spy()
    sent_success = test_sentiment_spy()
    analysis_success = test_analysis_page()
    
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    print(f"Prédiction SPY: {'✅' if pred_success else '❌'}")
    print(f"Sentiment SPY: {'✅' if sent_success else '❌'}")
    print(f"Page d'analyse: {'✅' if analysis_success else '❌'}")
    
    if all([pred_success, sent_success, analysis_success]):
        print("\n🎉 Tous les tests sont réussis !")
    else:
        print("\n⚠️ Certains tests ont échoué.")

