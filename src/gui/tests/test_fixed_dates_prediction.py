#!/usr/bin/env python3
"""
Test de génération du graphique de prédiction avec dates corrigées
Vérification que les dates s'affichent correctement sur l'axe horizontal
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ajouter le path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.gui.services.data_service import DataService
from src.gui.services.chart_service import ChartService
from src.gui.services.prediction_service import PredictionService

def test_prediction_chart_with_fixed_dates():
    """Test du graphique de prédiction avec dates corrigées"""
    
    print("🧪 Test du graphique de prédiction avec dates corrigées")
    print("=" * 60)
    
    # Initialisation des services
    data_service = DataService()
    chart_service = ChartService()
    prediction_service = PredictionService()
    
    # Chargement des données SPY
    print("📊 Chargement des données SPY...")
    df = data_service.load_data("SPY")
    
    if df.empty:
        print("❌ Aucune donnée SPY disponible")
        return
    
    print(f"✅ Données chargées: {len(df)} lignes")
    print(f"📅 Période: {df['DATE'].min()} à {df['DATE'].max()}")
    
    # Filtrage sur 1 mois
    print("\n🔍 Filtrage sur 1 mois...")
    df_filtered = data_service.filter_by_period(df, "1 mois")
    print(f"✅ Données filtrées: {len(df_filtered)} lignes")
    
    # Génération des prédictions
    print("\n🔮 Génération des prédictions LSTM...")
    prediction_data = prediction_service.predict(df_filtered, 20)
    
    if not prediction_data['predictions']:
        print("❌ Aucune prédiction générée")
        return
    
    print(f"✅ Prédictions générées: {len(prediction_data['predictions'])} points")
    print(f"📅 Dates futures: {prediction_data['prediction_dates'][0]} à {prediction_data['prediction_dates'][-1]}")
    
    # Création du graphique
    print("\n📈 Création du graphique...")
    fig = chart_service.create_prediction_chart(
        df_filtered, 
        prediction_data, 
        "SPY", 
        "1 mois"
    )
    
    # Vérification de la configuration des dates
    print("\n🔍 Vérification de la configuration des dates...")
    xaxis_config = fig.layout.xaxis
    
    print(f"✅ Type d'axe: {xaxis_config.type}")
    print(f"✅ Format des ticks: {xaxis_config.tickformat}")
    print(f"✅ Angle des ticks: {xaxis_config.tickangle}")
    print(f"✅ Intervalle des ticks: {xaxis_config.dtick}")
    
    # Sauvegarde du graphique
    output_path = Path(__file__).parent / "spy_fixed_dates_prediction.png"
    fig.write_image(str(output_path), width=1200, height=700, scale=2)
    
    print(f"\n💾 Graphique sauvegardé: {output_path}")
    print("✅ Test terminé avec succès!")
    
    # Affichage des métriques
    if 'historical_predictions' in prediction_data and prediction_data['historical_predictions']:
        hist_preds = prediction_data['historical_predictions']
        real_prices = df_filtered['CLOSE'].values
        
        mae_hist = np.mean(np.abs(np.array(hist_preds) - real_prices))
        mape_hist = np.mean(np.abs((np.array(hist_preds) - real_prices) / real_prices)) * 100
        correlation_hist = np.corrcoef(hist_preds, real_prices)[0, 1]
        
        print(f"\n📊 MÉTRIQUES HISTORIQUES:")
        print(f"• Erreur Moyenne: {mae_hist:.2f}$")
        print(f"• Erreur Relative: {mape_hist:.1f}%")
        print(f"• Corrélation: {correlation_hist:.3f}")
    
    if 'predictions' in prediction_data and prediction_data['predictions']:
        future_preds = prediction_data['predictions']
        last_real_price = df_filtered['CLOSE'].iloc[-1]
        
        price_change = (future_preds[-1] - last_real_price) / last_real_price * 100
        volatility_future = np.std(np.diff(future_preds)) / np.mean(future_preds) * 100
        
        print(f"\n🔮 MÉTRIQUES FUTURES:")
        print(f"• Prix initial: ${last_real_price:.2f}")
        print(f"• Prix final prédit: ${future_preds[-1]:.2f}")
        print(f"• Variation attendue: {price_change:+.1f}%")
        print(f"• Volatilité prédite: {volatility_future:.1f}%")

if __name__ == "__main__":
    test_prediction_chart_with_fixed_dates()
