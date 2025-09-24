#!/usr/bin/env python3
"""
Configuration centralisée pour l'interface GUI Sentinel
"""

# Configuration des tickers
TICKER_CHOICES = ["NVDA", "SPY"]

# Configuration des périodes
PERIOD_CHOICES = [
    "7 derniers jours",
    "1 mois", 
    "3 mois",
    "6 derniers mois",
    "1 an",
    "3 ans",
    "5 ans",
    "10 ans",
    "Total (toutes les données)"
]

# Configuration des types d'analyse
ANALYSIS_CHOICES = ["Prix", "Volume", "Prédiction", "Sentiment"]

# Configuration des modes de fusion
FUSION_MODES = ["Prix seul", "Sentiment seul", "Fusion adaptative"]

# Configuration par défaut
DEFAULT_TICKER = "NVDA"
DEFAULT_PERIOD = "7 derniers jours"
DEFAULT_ANALYSIS = "Prix"
DEFAULT_FUSION_MODE = "Fusion adaptative"

# Configuration de l'interface
INTERFACE_CONFIG = {
    "title": "Sentinel - Trading Algorithmique",
    "description": "Interface de trading algorithmique avec fusion adaptative",
    "port": 7867,
    "host": "127.0.0.1",
    "theme": "default",
    "share": False
}

# Configuration des graphiques
CHART_CONFIG = {
    "width": 800,
    "height": 400,
    "theme": "plotly_white",
    "show_legend": True,
    "show_grid": True
}

print("🎨 Configuration GUI chargée")
