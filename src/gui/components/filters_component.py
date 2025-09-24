#!/usr/bin/env python3
"""
Composant de filtres unifié - Gestion des filtres
"""

import gradio as gr
from typing import Tuple, Any
from loguru import logger

from ..gui_config import TICKER_CHOICES, PERIOD_CHOICES, ANALYSIS_CHOICES, DEFAULT_TICKER, DEFAULT_PERIOD, DEFAULT_ANALYSIS


class FiltersComponent:
    """Composant de filtres unifié et réutilisable"""
    
    def __init__(self):
        self.ticker_choices = TICKER_CHOICES
        self.period_choices = PERIOD_CHOICES
        self.analysis_choices = ANALYSIS_CHOICES
        logger.info("🔧 Composant de filtres initialisé")
    
    def create_filters(self) -> Tuple[Any, Any, Any]:
        """Crée les composants de filtres"""
        try:
            # Radio pour le ticker
            ticker_radio = gr.Radio(
                choices=self.ticker_choices,
                value=DEFAULT_TICKER,
                label="Action à analyser",
                info="Sélectionnez l'action à analyser"
            )
            
            # Radio pour le type d'analyse
            analysis_radio = gr.Radio(
                choices=self.analysis_choices,
                value=DEFAULT_ANALYSIS,
                label="Type d'analyse",
                info="Sélectionnez le type d'analyse"
            )
            
            # Radio pour la période (sous le type d'analyse)
            period_radio = gr.Radio(
                choices=self.period_choices,
                value=DEFAULT_PERIOD,
                label="Période d'analyse",
                info="Sélectionnez la période d'analyse"
            )
            
            logger.info("✅ Composants de filtres créés")
            return ticker_radio, analysis_radio, period_radio
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création des filtres: {e}")
            return None, None, None
    
    def get_default_values(self) -> Tuple[str, str, str]:
        """Retourne les valeurs par défaut"""
        return "NVDA", "7 derniers jours", "Prix"
    
    def validate_filters(self, ticker: str, analysis_type: str, period: str) -> bool:
        """Valide les filtres sélectionnés"""
        try:
            if ticker not in self.ticker_choices:
                logger.warning(f"⚠️ Ticker invalide: {ticker}")
                return False
            
            if period not in self.period_choices:
                logger.warning(f"⚠️ Période invalide: {period}")
                return False
            
            if analysis_type not in self.analysis_choices:
                logger.warning(f"⚠️ Type d'analyse invalide: {analysis_type}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la validation des filtres: {e}")
            return False
