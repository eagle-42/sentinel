#!/usr/bin/env python3
"""
Script pour générer le graphique LSTM en PNG
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger

# Ajouter le répertoire racine au path
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

from tools.gui.services.data_service import DataService
from tools.gui.services.prediction_service import PredictionService

def generate_lstm_png():
    """Génère le graphique LSTM et le sauvegarde en PNG"""
    
    logger.info("🔮 GÉNÉRATION DU GRAPHIQUE LSTM EN PNG")
    logger.info("=" * 50)
    
    # 1. Services
    data_service = DataService()
    prediction_service = PredictionService()
    
    # 2. Données
    df = data_service.load_data('SPY')
    df = data_service.filter_by_period(df, '1 mois')
    logger.info(f"📊 Données: {len(df)} lignes")
    
    # 3. Modèle
    success = prediction_service.load_model()
    if not success:
        logger.error("❌ Impossible de charger le modèle")
        return None
    
    # 4. Prédiction
    result = prediction_service.predict(df, horizon=10)
    if 'error' in result:
        logger.error(f"❌ Erreur: {result['error']}")
        return None
    
    # 5. Création du graphique
    fig = create_prediction_chart(df, result)
    
    # 6. Sauvegarde en PNG
    png_path = Path("lstm_predictions_spy.png")
    fig.write_image(png_path, width=1200, height=700, scale=2)
    
    logger.info(f"💾 Graphique sauvegardé: {png_path.absolute()}")
    logger.info(f"📏 Taille: 1200x700 pixels (scale 2x)")
    
    if png_path.exists():
        logger.info(f"✅ Fichier créé: {png_path.stat().st_size / 1024:.1f} KB")
        
        # Analyse des résultats
        if 'historical_predictions' in result and len(result['historical_predictions']) > 0:
            actual_prices = result['actual_prices']
            hist_preds = result['historical_predictions']
            
            if len(hist_preds) == len(actual_prices):
                errors = [abs(actual - pred) for actual, pred in zip(actual_prices, hist_preds)]
                mae = np.mean(errors)
                mape = np.mean([abs(actual - pred) / actual * 100 for actual, pred in zip(actual_prices, hist_preds)])
                correlation = np.corrcoef(actual_prices, hist_preds)[0, 1]
                
                logger.info("📊 RÉSULTATS:")
                logger.info(f"   - MAE: ${mae:.2f}")
                logger.info(f"   - MAPE: {mape:.2f}%")
                logger.info(f"   - Corrélation: {correlation:.3f}")
                
                if correlation > 0.8:
                    logger.info("   ✅ Excellente performance!")
                elif correlation > 0.6:
                    logger.info("   ✅ Bonne performance")
                else:
                    logger.info("   ⚠️ Performance à améliorer")
        
        return png_path
    else:
        logger.error("❌ Erreur lors de la sauvegarde")
        return None

def create_prediction_chart(df, result):
    """Crée le graphique de prédiction complet"""
    
    fig = go.Figure()
    
    # 1. Prix réel (bleu)
    fig.add_trace(go.Scatter(
        x=df['DATE'],
        y=df['CLOSE'],
        mode='lines+markers',
        name='Prix Réel',
        line=dict(color='blue', width=3),
        marker=dict(size=8),
        hovertemplate='<b>Prix Réel</b><br>Date: %{x}<br>Prix: $%{y:.2f}<extra></extra>'
    ))
    
    # 2. Prédictions historiques (vert) - si disponibles
    if 'historical_predictions' in result and len(result['historical_predictions']) > 0:
        hist_preds = result['historical_predictions']
        
        if len(hist_preds) == len(df):
            fig.add_trace(go.Scatter(
                x=df['DATE'],
                y=hist_preds,
                mode='lines+markers',
                name='Prédictions Historiques (LSTM)',
                line=dict(color='green', width=2, dash='dot'),
                marker=dict(size=6),
                hovertemplate='<b>Prédiction Historique</b><br>Date: %{x}<br>Prix prédit: $%{y:.2f}<extra></extra>'
            ))
    
    # 3. Prédictions futures (rouge)
    if 'prediction_dates' in result and 'predictions' in result:
        pred_dates = [pd.to_datetime(d) for d in result['prediction_dates']]
        predictions = result['predictions']
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=predictions,
            mode='lines+markers',
            name='Prédictions Futures (LSTM)',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate='<b>Prédiction Future</b><br>Date: %{x}<br>Prix prédit: $%{y:.2f}<extra></extra>'
        ))
        
        # Ligne de séparation
        last_hist_date = df['DATE'].iloc[-1]
        if hasattr(last_hist_date, 'timestamp'):
            last_hist_date_ts = last_hist_date.timestamp()
        else:
            last_hist_date_ts = last_hist_date
        fig.add_vline(
            x=last_hist_date_ts,
            line_dash="dot",
            line_color="gray",
            line_width=2,
            annotation_text="Début prédiction future",
            annotation_position="top"
        )
    
    # Mise en forme
    fig.update_layout(
        title={
            'text': '🔮 PRÉDICTIONS LSTM SPY - Modèle v4',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Date",
        yaxis_title="Prix ($)",
        hovermode='x unified',
        showlegend=True,
        height=700,
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        ),
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text="<b>Légende:</b><br>🔵 Prix réel<br>🟢 Prédictions historiques<br>🔴 Prédictions futures",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1
            )
        ]
    )
    
    return fig

if __name__ == "__main__":
    png_path = generate_lstm_png()
    if png_path:
        print(f"\n🎉 Graphique généré avec succès: {png_path}")
    else:
        print("\n❌ Échec de la génération du graphique")

