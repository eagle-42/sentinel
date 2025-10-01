"""
Interface Streamlit principale - Sentinel Trading
Structure conforme aux bonnes pratiques officielles Streamlit
"""

import sys
from pathlib import Path

import streamlit as st

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration des logs AVANT tout import
from gui.config.logging_config import setup_logging

setup_logging()

from gui.config.settings import APP_CONFIG, get_css_path

# Imports au niveau du module (bonnes pratiques officielles)
from gui.pages.analysis_page import show_analysis_page
from gui.pages.logs_page import show_logs_page
from gui.pages.production_page import show_production_page


def inject_css() -> None:
    """Injection contrôlée du CSS centralisé (bonnes pratiques officielles)"""
    css_path = get_css_path()
    if Path(css_path).exists():
        css_content = Path(css_path).read_text(encoding="utf-8")
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def main():
    """Fonction principale selon les bonnes pratiques officielles Streamlit"""

    # Configuration de la page au tout début (Get started + API reference)
    st.set_page_config(**APP_CONFIG)

    # Injection du CSS centralisé
    inject_css()

    # Header principal
    st.markdown(
        """
    <div class="main-header">
        <h1>🚀 Sentinel - Trading Prédictif & Sentiment Analyse</h1>
        <p>Système d'analyse et de prédiction des marchés financiers</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Onglets avec Default tab (bonnes pratiques officielles)
    tabs = st.tabs(["📊 Analysis", "🚀 Production", "📋 Logs"])

    with tabs[0]:
        show_analysis_page()

    with tabs[1]:
        show_production_page()

    with tabs[2]:
        show_logs_page()


# Appel explicite de la fonction principale (DCO recommandation 1)
if __name__ == "__main__":
    main()
