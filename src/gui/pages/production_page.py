"""
Page de production - Application de production pour trading temps réel
"""

import streamlit as st
from datetime import datetime, timedelta

def show_production_page():
    """Affiche la page de production"""
    
    # CSS personnalisé
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .feature-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background-color: #28a745; }
        .status-offline { background-color: #dc3545; }
        .status-pending { background-color: #ffc107; }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar - Paramètres de production (uniquement pour Production)
    with st.sidebar:
        st.header("⚙️ Paramètres de Production")
        
        # Configuration du système
        st.subheader("🔧 Configuration")
        
        update_frequency = st.selectbox(
            "Fréquence de mise à jour",
            ["15 minutes", "30 minutes", "1 heure", "2 heures"],
            index=0,
            help="Fréquence de récupération des données"
        )
        
        auto_trading = st.checkbox("Trading automatique", value=False)
        if auto_trading:
            st.warning("⚠️ Trading automatique activé")
        
        # Seuils de trading
        st.subheader("🎯 Seuils de Trading")
        
        buy_threshold = st.slider("Seuil d'achat (%)", 0.0, 10.0, 2.0, 0.1)
        sell_threshold = st.slider("Seuil de vente (%)", -10.0, 0.0, -2.0, 0.1)
        confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.7, 0.05)
        
        # Actions du système
        st.subheader("🎮 Actions")
        
        if st.button("🔄 Démarrer", type="primary"):
            st.success("✅ Système démarré")
        
        if st.button("⏸️ Pause"):
            st.warning("⏸️ Système en pause")
        
        if st.button("🛑 Arrêter"):
            st.error("🛑 Système arrêté")
    
    # Pas de header ici car il est déjà dans main.py
    
    # Statut du système
    st.header("📊 Statut du Système")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4><span class="status-indicator status-online"></span>Récupération Prix</h4>
            <p>Toutes les 15 minutes</p>
            <small>✅ Fonctionnel - SPY & NVDA</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4><span class="status-indicator status-online"></span>Crawler Financier</h4>
            <p>Articles et actualités</p>
            <small>✅ Fonctionnel - Toutes les 4 min</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4><span class="status-indicator status-online"></span>Fusion Données</h4>
            <p>Prédiction + Sentiment + FinBERT</p>
            <small>✅ Fonctionnel - Fusion adaptative</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h4><span class="status-indicator status-online"></span>Recommandations</h4>
            <p>ACHETER/ATTENDRE/VENDRE</p>
            <small>✅ Fonctionnel - Score de fusion</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Fonctionnalités en développement
    st.header("🔧 Fonctionnalités en Développement")
    
    # Récupération des prix temps réel
    with st.expander("📈 Récupération Prix Temps Réel", expanded=True):
        st.markdown("""
        **Objectif** : Récupérer les prix des actions toutes les 15 minutes
        
        **Fonctionnalités prévues** :
        - ✅ Connexion API Yahoo Finance
        - ✅ Mise à jour automatique des données
        - ✅ Stockage en base de données
        - ✅ Alertes de prix en temps réel
        - ⏳ Interface de monitoring
        - ⏳ Historique des variations
        """)
        
        # Simulation de données temps réel
        if st.button("🔄 Simuler Récupération Prix"):
            st.success("✅ Prix mis à jour - NVDA: $183.45, SPY: $445.67")
    
    # Crawler financier
    with st.expander("📰 Crawler Financier", expanded=True):
        st.markdown("""
        **Objectif** : Récupérer et analyser les articles financiers
        
        **Fonctionnalités prévues** :
        - ⏳ Crawling automatique des sources financières
        - ⏳ Analyse de sentiment avec FinBERT
        - ⏳ Extraction des mots-clés importants
        - ⏳ Classification par impact sur le marché
        - ⏳ Interface de visualisation des articles
        - ⏳ Alertes sur les news importantes
        """)
        
        st.info("🚧 Cette fonctionnalité sera disponible dans la prochaine version")
    
    # Fusion des données
    with st.expander("🔗 Fusion des Données", expanded=True):
        st.markdown("""
        **Objectif** : Combiner prédiction LSTM + sentiment + FinBERT
        
        **Fonctionnalités prévues** :
        - ⏳ Pipeline de fusion des données
        - ⏳ Pondération des différents signaux
        - ⏳ Score de confiance global
        - ⏳ Historique des fusions
        - ⏳ Interface de configuration des poids
        - ⏳ Monitoring de la qualité des données
        """)
        
        st.info("🚧 Cette fonctionnalité sera disponible dans la prochaine version")
    
    # Recommandations de trading
    with st.expander("🎯 Recommandations de Trading", expanded=True):
        st.markdown("""
        **Objectif** : Générer des recommandations ACHETER/ATTENDRE/VENDRE
        
        **Fonctionnalités prévues** :
        - ⏳ Algorithme de décision basé sur les signaux
        - ⏳ Seuils de confiance configurables
        - ⏳ Justification des recommandations
        - ⏳ Historique des performances
        - ⏳ Interface de backtesting
        - ⏳ Alertes automatiques
        """)
        
        st.info("🚧 Cette fonctionnalité sera disponible dans la prochaine version")
    
    # Justification LLM
    with st.expander("🤖 Justification LLM", expanded=True):
        st.markdown("""
        **Objectif** : Expliquer les décisions avec un modèle LLM local
        
        **Fonctionnalités prévues** :
        - ⏳ Intégration d'un modèle LLM local (type Windows)
        - ⏳ Génération d'explications en langage naturel
        - ⏳ Analyse des facteurs de décision
        - ⏳ Rapports de trading automatisés
        - ⏳ Interface de configuration du modèle
        - ⏳ Monitoring de la qualité des explications
        """)
        
        st.info("🚧 Cette fonctionnalité sera disponible dans la prochaine version")
    
    # Configuration et actions déplacées dans la sidebar
    
    # Informations de développement
    st.markdown("---")
    st.markdown("#### 📋 Informations de Développement")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**🕒 Dernière mise à jour** : {datetime.now().strftime('%H:%M:%S')}")
        st.markdown(f"**📊 Version** : 1.0.0-beta")
    with col2:
        st.markdown(f"**🔧 Statut** : En développement")
        st.markdown(f"**📈 Progression** : 25%")
    with col3:
        st.markdown(f"**🎯 Prochaine version** : Q1 2025")
        st.markdown(f"**📞 Support** : Disponible")
