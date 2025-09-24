#!/usr/bin/env python3
"""
Sentinel UI - Interface modulaire et maintenable
Architecture claire avec composants réutilisables
"""

import sys
from pathlib import Path
import gradio as gr
from loguru import logger

# Ajouter le répertoire racine au path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.gui.pages.analysis_page import AnalysisPage


def create_interface() -> gr.Blocks:
    """Crée l'interface principale modulaire"""
    
    try:
        # Initialiser les pages
        analysis_page = AnalysisPage()
        
        with gr.Blocks(
            title="Sentinel - Trading Algorithmique",
            theme=gr.themes.Soft(
                primary_hue="purple",
                secondary_hue="purple"
            ),
            css="""
            .analysis-box {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                padding: 1rem;
                border-radius: 0.5rem;
                border: 2px solid #e9d5ff;
                margin: 1rem 0;
            }
            """
        ) as demo:
            
            # En-tête principal
            gr.Markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="margin: 0;">🚀 Sentinel - Trading Algorithmique</h1>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Système de trading avec prédictions LSTM</p>
            </div>
            """)
            
            # Onglets
            with gr.Tabs():
                
                # Onglet Trading
                with gr.Tab("📈 Trading"):
                    gr.Markdown("### 🚀 Trading en Temps Réel")
                    gr.Markdown("""
                    <div class="analysis-box">
                        <h4>Fonctionnalité en développement</h4>
                        <p>Le module de trading sera disponible dans une future version.</p>
                    </div>
                    """)
                
                # Onglet Analysis
                with gr.Tab("📊 Analysis"):
                    analysis_page.create_page()
                
                # Onglet Logs
                with gr.Tab("📋 Logs"):
                    gr.Markdown("### 📋 Logs système")
                    gr.Markdown("""
                    <div class="analysis-box">
                        <h4>Fonctionnalité en développement</h4>
                        <p>Le module de logs sera disponible dans une future version.</p>
                    </div>
                    """)
        
        logger.info("✅ Interface Sentinel créée avec succès")
        return demo
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création de l'interface: {e}")
        return None


def main():
    """Point d'entrée principal"""
    try:
        logger.info("🚀 Démarrage de Sentinel UI...")
        
        # Créer l'interface
        demo = create_interface()
        
        if demo is None:
            logger.error("❌ Impossible de créer l'interface")
            return
        
        # Lancer l'interface ATTENTION MODIFIER EN PROD
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            inbrowser=True,
            debug=True,  # Activation du mode debug
            show_api=True  # Affichage de l'API pour le debug
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du lancement: {e}")


if __name__ == "__main__":
    main()
