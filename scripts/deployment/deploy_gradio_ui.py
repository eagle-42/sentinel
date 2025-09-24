#!/usr/bin/env python3
"""
Script de déploiement pour l'interface Gradio Sentinel
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire racine au path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from loguru import logger

def deploy_gradio_ui():
    """Déploie l'interface Gradio Sentinel"""
    
    logger.info("🚀 Déploiement de l'interface Gradio Sentinel...")
    
    try:
        # Vérifier que Gradio est installé
        import gradio as gr
        logger.success("✅ Gradio installé")
        
        # Vérifier que Plotly est installé
        import plotly
        logger.success("✅ Plotly installé")
        
        # Lancer l'interface
        from tools.gui.fusion_test_gradio import main
        
        logger.info("🌐 Lancement de l'interface sur http://127.0.0.1:7860")
        logger.info("📱 Interface accessible via navigateur web")
        logger.info("🛑 Ctrl+C pour arrêter")
        
        main()
        
    except ImportError as e:
        logger.error(f"❌ Dépendance manquante: {e}")
        logger.info("💡 Installez avec: uv add gradio plotly")
        return False
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du déploiement: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = deploy_gradio_ui()
    if not success:
        sys.exit(1)
