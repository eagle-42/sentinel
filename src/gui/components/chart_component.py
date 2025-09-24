#!/usr/bin/env python3
"""
Composant graphique unifié - Un seul composant pour tous les graphiques
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger


class ChartComponent:
    """Composant graphique unifié et réutilisable"""
    
    def __init__(self):
        logger.info("📊 Composant graphique initialisé")
    
    def create_price_chart(self, df: pd.DataFrame, ticker: str, period: str) -> go.Figure:
        """Crée un graphique de prix avec indicateurs"""
        try:
            if df.empty:
                return self._create_error_chart("Aucune donnée disponible")
            
            fig = go.Figure()
            
            # Prix historiques
            fig.add_trace(go.Scatter(
                x=df['DATE'],
                y=df['CLOSE'],
                mode='lines',
                name=f'{ticker} - Prix',
                line=dict(color='blue', width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Prix: $%{y:.2f}<extra></extra>'
            ))
            
            # Moyennes mobiles
            if len(df) >= 20:
                df['MA_20'] = df['CLOSE'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=df['DATE'],
                    y=df['MA_20'],
                    mode='lines',
                    name='MA 20',
                    line=dict(color='orange', width=1, dash='dash'),
                    hovertemplate='<b>MA 20</b><br>Date: %{x}<br>Prix: $%{y:.2f}<extra></extra>'
                ))
            
            if len(df) >= 50:
                df['MA_50'] = df['CLOSE'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=df['DATE'],
                    y=df['MA_50'],
                    mode='lines',
                    name='MA 50',
                    line=dict(color='red', width=1, dash='dot'),
                    hovertemplate='<b>MA 50</b><br>Date: %{x}<br>Prix: $%{y:.2f}<extra></extra>'
                ))
            
            # Mise en forme
            fig.update_layout(
                title=f"{ticker} - Prix avec Moyennes Mobiles ({period})",
                xaxis_title="Date",
                yaxis_title="Prix ($)",
                hovermode='x unified',
                showlegend=True,
                height=500,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création du graphique de prix: {e}")
            return self._create_error_chart(str(e))
    
    def create_volume_chart(self, df: pd.DataFrame, ticker: str, period: str) -> go.Figure:
        """Crée un graphique de volume avec volatilité"""
        try:
            if df.empty:
                return self._create_error_chart("Aucune donnée disponible")
            
            fig = go.Figure()
            
            # Volume
            fig.add_trace(go.Bar(
                x=df['DATE'],
                y=df['VOLUME'],
                name=f'{ticker} - Volume',
                marker_color='lightblue',
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Volume: %{y:,.0f}<extra></extra>'
            ))
            
            # Volatilité du volume
            if len(df) >= 20:
                df['VOLUME_VOLATILITY'] = df['VOLUME'].rolling(window=20).std()
                fig.add_trace(go.Scatter(
                    x=df['DATE'],
                    y=df['VOLUME_VOLATILITY'],
                    mode='lines',
                    name='Volatilité Volume',
                    line=dict(color='red', width=2),
                    yaxis='y2',
                    hovertemplate='<b>Volatilité Volume</b><br>Date: %{x}<br>Volatilité: %{y:,.0f}<extra></extra>'
                ))
            
            # Mise en forme
            fig.update_layout(
                title=f"{ticker} - Volume avec Volatilité ({period})",
                xaxis_title="Date",
                yaxis_title="Volume",
                yaxis2=dict(
                    title="Volatilité Volume",
                    overlaying="y",
                    side="right"
                ),
                hovermode='x unified',
                showlegend=True,
                height=500,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création du graphique de volume: {e}")
            return self._create_error_chart(str(e))
    
    def create_prediction_chart(self, df: pd.DataFrame, prediction_data: Dict[str, Any], ticker: str, period: str) -> go.Figure:
        """Graphique ultra-simple : Prix réel + Prédictions futures"""
        try:
            if df.empty:
                return self._create_error_chart("Aucune donnée disponible")
            
            if "error" in prediction_data:
                return self._create_error_chart(prediction_data["error"])
            
            fig = go.Figure()
            
            # Prix réel (bleu)
            fig.add_trace(go.Scatter(
                x=df['DATE'],
                y=df['CLOSE'],
                mode='lines',
                name=f'{ticker} - Prix réel',
                line=dict(color='blue', width=2),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Prix: $%{y:.2f}<extra></extra>'
            ))
            
            # Prédictions historiques (vert pointillé) - si disponibles
            if 'historical_predictions' in prediction_data and len(prediction_data['historical_predictions']) > 0:
                hist_predictions = prediction_data['historical_predictions']
                # Ajuster la longueur pour correspondre aux dates
                if len(hist_predictions) == len(df['DATE']):
                    fig.add_trace(go.Scatter(
                        x=df['DATE'],
                        y=hist_predictions,
                        mode='lines',
                        name=f'{ticker} - Prédictions historiques',
                        line=dict(color='green', width=2, dash='dot'),
                        hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Prix prédit: $%{y:.2f}<extra></extra>'
                    ))
            
            # Prédictions futures (rouge pointillé)
            if 'prediction_dates' in prediction_data and 'predictions' in prediction_data:
                pred_dates = [pd.to_datetime(d) for d in prediction_data['prediction_dates']]
                predictions = prediction_data['predictions']
                
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=predictions,
                    mode='lines',
                    name=f'{ticker} - Prédictions futures',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Prix prédit: $%{y:.2f}<extra></extra>'
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
                    line_width=1,
                    annotation_text="Début prédiction future"
                )
            
            # Mise en forme
            fig.update_layout(
                title=f"{ticker} - Prix Réel vs Prédictions ({period})",
                xaxis_title="Date",
                yaxis_title="Prix ($)",
                hovermode='x unified',
                showlegend=True,
                height=600,
                template="plotly_white",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création du graphique de prédiction: {e}")
            return self._create_error_chart(str(e))
    
    def create_sentiment_chart(self, df: pd.DataFrame, ticker: str, period: str) -> go.Figure:
        """Crée un graphique de sentiment normalisé avec log-rendement et Z-score"""
        try:
            if df.empty:
                return self._create_error_chart("Aucune donnée disponible")
            
            # Calculer le sentiment normalisé selon la méthode demandée
            df_sentiment = df.copy()
            
            # 1. Calcul du log-rendement: r_t = ln(P_t / P_{t-1})
            df_sentiment['log_returns'] = np.log(df_sentiment['CLOSE'] / df_sentiment['CLOSE'].shift(1))
            
            # 2. Standardisation glissante (Z-score) sur fenêtre de 20 jours
            window = 20
            df_sentiment['mean_returns'] = df_sentiment['log_returns'].rolling(window=window).mean()
            df_sentiment['std_returns'] = df_sentiment['log_returns'].rolling(window=window).std()
            df_sentiment['z_score'] = (df_sentiment['log_returns'] - df_sentiment['mean_returns']) / df_sentiment['std_returns']
            
            # 3. Compression par tanh: s_t = tanh(α * z_t)
            alpha = 2.0  # Facteur de compression
            df_sentiment['sentiment_score'] = np.tanh(alpha * df_sentiment['z_score'])
            
            # 4. Conversion en pourcentage
            df_sentiment['sentiment_pct'] = df_sentiment['sentiment_score'] * 100
            
            # 5. Décision de trading
            tau = 0.3  # Seuil de décision
            df_sentiment['decision'] = df_sentiment['sentiment_score'].apply(
                lambda x: 'ACHETER' if x > tau else 'VENDRE' if x < -tau else 'HOLD'
            )
            
            # Créer la figure simplifiée
            fig = go.Figure()
            
            # Score de sentiment normalisé uniquement
            fig.add_trace(go.Scatter(
                x=df_sentiment['DATE'],
                y=df_sentiment['sentiment_pct'],
                mode='lines',
                name=f'{ticker} - Score Sentiment (%)',
                line=dict(color='lightblue', width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Score: %{y:.1f}%<br>' +
                             'Décision: ' + df_sentiment['decision'].iloc[0] + '<extra></extra>'
            ))
            
            # Lignes de référence pour les décisions - positionnées en dehors de la zone de tracé
            fig.add_hline(y=tau * 100, line_dash="dash", line_color="green", 
                         annotation_text="Seuil ACHETER", annotation_position="top right",
                         annotation_x=0.95, annotation_y=0.95)
            fig.add_hline(y=-tau * 100, line_dash="dash", line_color="red", 
                         annotation_text="Seuil VENDRE", annotation_position="bottom right",
                         annotation_x=0.95, annotation_y=0.05)
            fig.add_hline(y=0, line_dash="dot", line_color="gray", 
                         annotation_text="Neutre", annotation_position="top left",
                         annotation_x=0.05, annotation_y=0.95)
            
            # Mise en forme
            fig.update_layout(
                title=f"{ticker} - Score de Sentiment Normalisé ({period})",
                xaxis_title="Date",
                yaxis_title="Score de Sentiment (%)",
                height=500,
                template="plotly_white",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                # Ajuster les marges pour éviter la superposition
                margin=dict(l=50, r=100, t=50, b=50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création du graphique de sentiment: {e}")
            return self._create_error_chart(str(e))
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calcule le RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Valeur neutre par défaut
    
    def _create_error_chart(self, error_msg: str) -> go.Figure:
        """Crée un graphique d'erreur"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Erreur: {error_msg}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Erreur de graphique",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=500
        )
        return fig
