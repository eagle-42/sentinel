#!/usr/bin/env python3
"""
Page d'analyse modulaire - Interface d'analyse des données
"""

import gradio as gr
import pandas as pd
from typing import Tuple, Any
from loguru import logger

from src.gui.components.filters_component import FiltersComponent
from src.gui.components.chart_component import ChartComponent
from src.gui.services.data_service import DataService
from src.gui.services.prediction_service import PredictionService


class AnalysisPage:
    """Page d'analyse modulaire et maintenable"""
    
    def __init__(self):
        self.filters = FiltersComponent()
        self.chart = ChartComponent()
        self.data_service = DataService()
        self.prediction_service = PredictionService()
        
        # Charger le modèle LSTM
        self.prediction_service.load_model(version=1)
        
        logger.info("📊 Page d'analyse initialisée")
    
    def create_page(self) -> Tuple[Any, Any, Any, Any]:
        """Crée la page d'analyse complète"""
        try:
            with gr.Blocks() as analysis_page:
                gr.Markdown("### 📊 Analyse d'action")
                
                # Créer les filtres
                ticker_radio, analysis_radio, period_radio = self.filters.create_filters()
                
                # Composants de sortie
                result_text = gr.Markdown("### En attente d'analyse...")
                chart_plot = gr.Plot(label="Graphique")
                
                # Événements
                ticker_radio.change(
                    self.analyze_data,
                    inputs=[ticker_radio, analysis_radio, period_radio],
                    outputs=[result_text, chart_plot]
                )
                
                analysis_radio.change(
                    self.analyze_data,
                    inputs=[ticker_radio, analysis_radio, period_radio],
                    outputs=[result_text, chart_plot]
                )
                
                period_radio.change(
                    self.analyze_data,
                    inputs=[ticker_radio, analysis_radio, period_radio],
                    outputs=[result_text, chart_plot]
                )
                
                # Chargement initial
                analysis_page.load(
                    self.analyze_data,
                    inputs=[ticker_radio, analysis_radio, period_radio],
                    outputs=[result_text, chart_plot]
                )
            
            logger.info("✅ Page d'analyse créée")
            return analysis_page, ticker_radio, analysis_radio, period_radio
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de la page d'analyse: {e}")
            return None, None, None, None
    
    def analyze_data(self, ticker: str, analysis_type: str, period: str) -> Tuple[str, Any]:
        """Analyse les données selon les filtres"""
        try:
            # Valider les filtres
            if not self.filters.validate_filters(ticker, analysis_type, period):
                return "❌ **Erreur de validation des filtres**", self.chart._create_error_chart("Filtres invalides")
            
            # Charger les données
            df = self.data_service.get_price_data(ticker, period)
            
            if df.empty:
                return f"❌ **Aucune donnée disponible** pour {ticker}", self.chart._create_error_chart("Données non disponibles")
            
            # Créer le texte de résultat
            result_text = self._create_result_text(ticker, period, analysis_type, df)
            
            # Créer le graphique selon le type
            if analysis_type == "Prix":
                chart = self.chart.create_price_chart(df, ticker, period)
            elif analysis_type == "Volume":
                chart = self.chart.create_volume_chart(df, ticker, period)
            elif analysis_type == "Prédiction":
                if ticker == "SPY":
                    # Utiliser les données filtrées par période pour la prédiction
                    prediction_data = self.prediction_service.predict(df, horizon=20)
                    chart = self.chart.create_prediction_chart(df, prediction_data, ticker, period)
                else:
                    chart = self.chart._create_error_chart("Prédictions LSTM disponibles uniquement pour SPY")
            elif analysis_type == "Sentiment":
                chart = self.chart.create_sentiment_chart(df, ticker, period)
            else:
                chart = self.chart._create_error_chart("Type d'analyse non supporté")
            
            return result_text, chart
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse des données: {e}")
            return f"❌ **Erreur d'analyse** : {str(e)}", self.chart._create_error_chart(str(e))
    
    def _create_result_text(self, ticker: str, period: str, analysis_type: str, df: pd.DataFrame) -> str:
        """Crée le texte de résultat"""
        try:
            # Informations de base
            start_date = df['DATE'].min().strftime('%Y-%m-%d')
            end_date = df['DATE'].max().strftime('%Y-%m-%d')
            data_points = len(df)
            
            # Prix actuel
            current_price = df['CLOSE'].iloc[-1]
            price_change = df['CLOSE'].iloc[-1] - df['CLOSE'].iloc[0]
            price_change_pct = (price_change / df['CLOSE'].iloc[0]) * 100
            
            # Volume moyen
            avg_volume = df['VOLUME'].mean()
            
            # Volatilité
            returns = df['CLOSE'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5) * 100
            
            result_text = f"""
### {ticker} - Analyse et Prédiction

**Période** : {period} ({start_date} à {end_date})  
**Points de données** : {data_points:,}

**📊 Métriques clés** :
- **Prix actuel** : ${current_price:.2f}
- **Variation** : ${price_change:+.2f} ({price_change_pct:+.2f}%)
- **Volume moyen** : {avg_volume:,.0f}
- **Volatilité** : {volatility:.2f}%

**🔍 Analyse** :
- **Tendance** : {'📈 Haussière' if price_change > 0 else '📉 Baissière' if price_change < 0 else '➡️ Stable'}
- **Volatilité** : {'🔴 Élevée' if volatility > 30 else '🟡 Modérée' if volatility > 15 else '🟢 Faible'}
- **Volume** : {'🔴 Élevé' if avg_volume > df['VOLUME'].quantile(0.8) else '🟡 Normal' if avg_volume > df['VOLUME'].quantile(0.2) else '🟢 Faible'}

**🤖 Méthode de prédiction** : Modèle LSTM avec 15 features optimisées, prédiction sur 20 jours
**😊 Méthode de sentiment** : Log-rendement + Z-score + compression tanh, seuils ACHETER/VENDRE à ±30%
"""
            
            return result_text
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création du texte de résultat: {e}")
            return f"❌ **Erreur lors de la génération du rapport** : {str(e)}"
