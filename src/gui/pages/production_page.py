"""
Page de production - Dashboard Trading
Interface optimisée selon les spécifications utilisateur
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys
import requests
import time

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.services.data_service import DataService
from gui.services.chart_service import ChartService
from gui.services.prediction_service import PredictionService
from gui.services.sentiment_service import SentimentService
from gui.services.fusion_service import FusionService
from gui.services.llm_service import LLMService
from gui.services.monitoring_service import MonitoringService
from gui.services.data_monitor_service import DataMonitorService


def show_production_page():
    """Affiche la page de production optimisée"""
    
    # Initialiser les services
    if 'production_services' not in st.session_state:
        st.session_state.production_services = {
            'data_service': DataService(),
            'chart_service': ChartService(),
            'prediction_service': PredictionService(),
            'sentiment_service': SentimentService(),
            'fusion_service': FusionService(),
            'llm_service': LLMService(),
            'monitoring_service': MonitoringService(),
            'data_monitor_service': DataMonitorService()
        }
    
    services = st.session_state.production_services
    
    # CSS personnalisé optimisé
    st.markdown("""
    <style>
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .feature-card:hover {
            transform: translateY(-2px);
        }
        .price-card {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        }
        .sentiment-card {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        }
        .fusion-card {
            background: linear-gradient(135deg, #e2e3f0 0%, #d1d5f0 100%);
        }
        .llm-card {
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        }
        .monitoring-card {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
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
        .status-warning { background-color: #ffc107; }
        .gauge-container {
            text-align: center;
            padding: 1rem;
        }
        .recommendation-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1.1rem;
        }
        .recommendation-buy {
            background-color: #28a745;
            color: white;
        }
        .recommendation-sell {
            background-color: #dc3545;
            color: white;
        }
        .recommendation-wait {
            background-color: #ffc107;
            color: #212529;
        }
        .action-buttons {
            display: flex;
            gap: 10px;
            margin-left: 20px;
        }
        .error-alert {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .market-status {
            text-align: center;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .market-open {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .market-closed {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .service-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .service-card h6 {
            margin: 0 0 0.5rem 0;
            font-size: 0.9rem;
            color: #666;
        }
        .service-card p {
            margin: 0.25rem 0;
            font-size: 1rem;
        }
        .service-card small {
            color: #888;
            font-size: 0.8rem;
        }
        .status-line {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            font-size: 1.1rem;
            text-align: center;
        }
        .timeline-container {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .timeline-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid #e9ecef;
        }
        .timeline-item:last-child {
            border-bottom: none;
        }
        .kpi-box {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        .kpi-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar - Nettoyée selon demande
    with st.sidebar:
        st.header("⚙️ Configuration Production")
        
        # Configuration simple
        ticker = st.selectbox(
            "Ticker",
            ["SPY", "NVDA"],
            index=0,
            help="Symbole de l'action à analyser. SPY: Prédictions disponibles. NVDA: Analyse uniquement."
        )
        
        period = st.selectbox(
            "Période",
            ["7 derniers jours", "1 mois", "3 mois", "6 mois", "1 an"],
            index=1,
            help="Période d'analyse des données historiques."
        )
        
        # Actions du système
        st.subheader("🎮 Actions")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Démarrer Services", type="primary", key="start_services"):
                st.success("✅ Services démarrés")
                st.session_state.services_running = True
        with col2:
            if st.button("🛑 Arrêter Services", type="secondary", key="stop_services"):
                st.warning("⚠️ Services arrêtés")
                st.session_state.services_running = False
        with col3:
            if st.button("🔄 Rafraîchir", type="secondary", key="refresh_page"):
                st.rerun()
        
        # Mise à jour des données
        st.subheader("📊 Données")
        
        if st.button("📈 Mettre à jour les prix", type="secondary", key="update_prices"):
            with st.spinner("Mise à jour des données de prix..."):
                try:
                    # Utiliser le service de monitoring pour la mise à jour
                    success = services['data_monitor_service'].trigger_data_refresh(ticker)
                    
                    if success:
                        st.success("✅ Données de prix mises à jour")
                        st.rerun()  # Recharger pour voir les nouvelles données
                    else:
                        st.warning("⚠️ Mise à jour partielle - Vérifiez les logs")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la mise à jour: {e}")
                    st.info("ℹ️ Utilisation des données en cache")
    
    # 1. COMPTE RENDU D'ACTIVITÉ - Version restructurée en haut
    st.header("📊 Compte Rendu d'Activité")
    
    # Vérification de l'état du marché
    market_status = _check_market_status()
    
    # Vérification des dépendances critiques
    fusion_available, fusion_errors = _check_fusion_dependencies(services, ticker, market_status)
    
    # Affichage des erreurs centralisées - Un seul message par module
    if fusion_errors:
        critical_errors = [e for e in fusion_errors if e['severity'] == 'critical']
        if critical_errors:
            st.error(f"🚨 **{critical_errors[0]['title']}** – {critical_errors[0]['message']}")
    
    # Calculer fusion_data avant l'affichage des KPIs
    fusion_data = None
    if fusion_available:
        try:
            price_signal = _get_price_signal(services, ticker)
            sentiment_signal = _get_sentiment_signal(services, ticker)
            prediction_signal = _get_prediction_signal(services, ticker)
            
            fusion_data = services['fusion_service'].calculate_fusion_score(
                price_signal, sentiment_signal, prediction_signal
            )
        except Exception as e:
            st.error(f"Erreur calcul fusion: {e}")
            fusion_data = None
    
    # KPI principaux en haut - Ordre réorganisé
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        # 1. Statut marché
        market_color = "#28a745" if market_status["is_open"] else "#dc3545"
        market_text = "Ouvert" if market_status["is_open"] else "Fermé"
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-value" style="color: {market_color};">{market_text}</div>
            <div class="kpi-label">Marché</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # 2. Dernière MAJ + Prochain Update
        current_time = datetime.now().strftime("%H:%M:%S")
        next_update = (datetime.now() + timedelta(minutes=15)).strftime("%H:%M")
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-value" style="color: #667eea;">{current_time}</div>
            <div class="kpi-label">Dernière MAJ</div>
            <div class="kpi-label" style="font-size: 0.8rem; color: #888; margin-top: 0.5rem;">Prochain: {next_update}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # 3. Confiance avec tooltip
        if fusion_available and fusion_data:
            try:
                st.markdown(f"""
                <div class="kpi-box" title="Niveau de confiance basé sur la cohérence des signaux (prix, sentiment, prédiction)">
                    <div class="kpi-value">{fusion_data['confidence']:.1%}</div>
                    <div class="kpi-label">Confiance</div>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-value" style="color: #dc3545;">N/A</div>
                    <div class="kpi-label">Confiance</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #dc3545;">N/A</div>
                <div class="kpi-label">Confiance</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # 4. Score de fusion avec détails
        if fusion_available and fusion_data:
            try:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-value">{fusion_data['fusion_score']:.2f}</div>
                    <div class="kpi-label">Score Fusion</div>
                    <div class="kpi-label" style="font-size: 0.7rem; color: #888; margin-top: 0.3rem;">
                        Prix: {price_signal:.2f} | News: {sentiment_signal:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-value" style="color: #dc3545;">N/A</div>
                    <div class="kpi-label">Score Fusion</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #dc3545;">N/A</div>
                <div class="kpi-label">Score Fusion</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        # 5. Recommandation - Logique corrigée selon état du marché
        if fusion_available and market_status["is_open"] and ticker == "SPY":
            try:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-value" style="color: {fusion_data['color']};">{fusion_data['recommendation']}</div>
                    <div class="kpi-label">Recommandation</div>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-value" style="color: #dc3545;">N/A</div>
                    <div class="kpi-label">Recommandation</div>
                </div>
                """, unsafe_allow_html=True)
        elif ticker == "NVDA":
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #ffc107;">N/A</div>
                <div class="kpi-label">Recommandation</div>
                <div class="kpi-label" style="font-size: 0.8rem; color: #888;">NVDA: Pas de prédiction</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #dc3545;">N/A</div>
                <div class="kpi-label">Recommandation</div>
                <div class="kpi-label" style="font-size: 0.8rem; color: #888;">Marché fermé</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col6:
        # 6. Box de vérification de la décision avec historique
        if fusion_available and market_status["is_open"] and ticker == "SPY":
            try:
                from gui.services.verification_service import VerificationService
                verification_service = VerificationService()
                
                current_price = services['data_service'].load_data(ticker)['CLOSE'].iloc[-1]
                verification_result = verification_service.verify_decision(
                    ticker, current_price, fusion_data['recommendation'], fusion_data['score']
                )
                
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-value" style="color: {verification_result.get('color', '#6c757d')};">
                        {verification_result['status']}
                    </div>
                    <div class="kpi-label">Vérification</div>
                    <div class="kpi-label" style="font-size: 0.7rem; color: #888; margin-top: 0.3rem;">
                        {verification_result['message']}
                    </div>
                    {f'<div class="kpi-label" style="font-size: 0.6rem; color: #666; margin-top: 0.2rem;">Précédent: {verification_result.get("previous_recommendation", "N/A")}</div>' if verification_result.get('previous_recommendation') else ''}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-value" style="color: #dc3545;">Erreur</div>
                    <div class="kpi-label">Vérification</div>
                    <div class="kpi-label" style="font-size: 0.7rem; color: #888; margin-top: 0.3rem;">
                        {str(e)[:50]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #6c757d;">-</div>
                <div class="kpi-label">Vérification</div>
            </div>
            """, unsafe_allow_html=True)
    
    # 2. GRAPHIQUE 15MIN DU JOUR - Données réelles avec monitoring
    st.header("📈 Graphique 15min du Jour")
    
    # Vérifier l'état des données 15min
    data_summary = services['data_monitor_service'].get_data_summary(ticker)
    
    # Afficher le statut des données
    col_status1, col_status2, col_status3 = st.columns([2, 1, 1])
    
    with col_status1:
        status_color = data_summary.get('status_color', 'red')
        status_text = data_summary.get('status_text', '❌ Données indisponibles')
        st.markdown(f"""
        <div style="background: {'#d4edda' if status_color == 'green' else '#fff3cd' if status_color == 'orange' else '#f8d7da'}; 
                    padding: 0.5rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;">
            <strong>{status_text}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status2:
        if data_summary.get('needs_update', False):
            if st.button("🔄 Actualiser", type="secondary", key="refresh_15min"):
                with st.spinner("Mise à jour des données 15min..."):
                    success = services['data_monitor_service'].trigger_data_refresh(ticker)
                    if success:
                        st.success("✅ Données mises à jour")
                        st.rerun()
                    else:
                        st.error("❌ Échec de la mise à jour")
    
    with col_status3:
        if data_summary.get('available', False):
            last_update = data_summary.get('last_update', datetime.now())
            if isinstance(last_update, str):
                last_update = datetime.now()
            st.markdown(f"""
            <div style="text-align: center; font-size: 0.9rem; color: #666;">
                Dernière MAJ:<br>
                <strong>{last_update.strftime('%H:%M')}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Afficher le graphique si les données sont disponibles
    if data_summary.get('available', False):
        try:
            data_15min, metadata = services['data_monitor_service'].get_latest_15min_data(ticker)
            
            if not data_15min.empty:
                # Filtrer les données des 7 derniers jours
                seven_days_ago = datetime.now() - timedelta(days=7)
                # Convertir les timestamps en datetime naif pour la comparaison
                data_15min['ts_utc_naive'] = data_15min['ts_utc'].apply(lambda x: x.replace(tzinfo=None) if x.tzinfo is not None else x)
                data_recent = data_15min[data_15min['ts_utc_naive'] >= seven_days_ago]
                
                if not data_recent.empty:
                    # Créer le graphique 15min
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=data_recent['ts_utc'],
                        y=data_recent['close'],
                        mode='lines+markers',
                        name=f'{ticker} - Prix 15min',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=4),
                        hovertemplate='<b>%{fullData.name}</b><br>Heure: %{x}<br>Prix: $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Ajouter les moyennes mobiles
                    if len(data_recent) > 20:
                        data_recent['ma_20'] = data_recent['close'].rolling(window=20).mean()
                        fig.add_trace(go.Scatter(
                            x=data_recent['ts_utc'],
                            y=data_recent['ma_20'],
                            mode='lines',
                            name='MA 20',
                            line=dict(color='#ff7f0e', width=1, dash='dash'),
                            hovertemplate='<b>MA 20</b><br>Heure: %{x}<br>Prix: $%{y:.2f}<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title=f"Prix {ticker} - 15min (7 derniers jours)",
                        xaxis_title="Heure",
                        yaxis_title="Prix ($)",
                        height=400,
                        showlegend=True,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Afficher les statistiques
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Dernier Prix", f"${data_recent['close'].iloc[-1]:.2f}")
                    with col2:
                        price_change = data_summary.get('price_change_24h', 0)
                        st.metric("Variation 24h", f"{price_change:+.2f}$")
                    with col3:
                        st.metric("Volume Moyen", f"{data_summary.get('volume_avg', 0):.0f}")
                    with col4:
                        st.metric("Total Records", f"{data_summary.get('total_records', 0)}")
                else:
                    st.info("Aucune donnée 15min récente disponible")
            else:
                st.warning("Données 15min vides")
        except Exception as e:
            st.error(f"Erreur graphique 15min: {e}")
    else:
        st.warning(f"Données 15min non disponibles: {data_summary.get('message', 'Erreur inconnue')}")
    
    # 3. PARTIE LLM ET SENTIMENT - Séparées en deux colonnes
    st.header("🤖 Services d'Analyse")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Service LLM (Ollama)
        st.subheader("🤖 Service LLM (Ollama)")
        
        # Vérifier l'état d'Ollama
        ollama_online = False
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                ollama_online = any(m.get('name') == 'phi3:mini' for m in models)
        except:
            ollama_online = False
        
        # Boutons LLM
        col_llm1, col_llm2 = st.columns(2)
        with col_llm1:
            if st.button("🚀 Démarrer Phi-3", type="primary", key="start_phi3"):
                try:
                    response = requests.post("http://localhost:11434/api/generate", json={
                        "model": "phi3:mini",
                        "prompt": "Test de connexion - prêt pour la production?",
                        "stream": False
                    }, timeout=30)
                    
                    if response.status_code == 200:
                        st.success("✅ Phi-3 Mini démarré et prêt !")
                        st.session_state.llm_ready = True
                    else:
                        st.error(f"❌ Erreur API Ollama : {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Ollama non accessible. Vérifiez que le service est démarré.")
                except Exception as e:
                    st.error(f"Erreur Ollama : {e}")
        
        with col_llm2:
            if st.button("🛑 Arrêter LLM", key="stop_llm"):
                st.warning("⚠️ Service LLM arrêté (simulé).")
                st.session_state.llm_ready = False
        
        # État du service
        status_color = "#28a745" if ollama_online else "#dc3545"
        status_text = "✅ En ligne" if ollama_online else "❌ Hors ligne"
        st.markdown(f"""
        <div style="background: white; padding: 0.5rem; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-size: 0.9rem;"><strong>Statut :</strong> <span style="color: {status_color};">{status_text}</span></p>
            <p style="margin: 0; font-size: 0.8rem; color: #666;">phi3:mini</p>
        </div>
        """, unsafe_allow_html=True)
    
        # Rapport LLM avec toggle
        if ollama_online and fusion_available:
            if st.button("📄 Générer Rapport avec Phi-3", type="secondary", key="generate_report"):
                try:
                    with st.spinner("Génération du rapport avec Phi-3..."):
                        response = requests.post("http://localhost:11434/api/generate", json={
                            "model": "phi3:mini",
                            "prompt": f"""
                            Analysez la situation de trading pour {ticker}:
                            - Score de fusion: {fusion_data.get('fusion_score', 0):.2f}
                            - Recommandation: {fusion_data.get('recommendation', 'N/A')}
                            - Confiance: {fusion_data.get('confidence', 0):.1%}
                            - Statut marché: {'Ouvert' if market_status['is_open'] else 'Fermé'}
                            
                            Fournissez une analyse détaillée et des recommandations.
                            """,
                            "stream": False
                        }, timeout=60)
                        
                        if response.status_code == 200:
                            result = response.json()
                            rapport = result.get('response', 'Erreur de génération')
                            
                            st.markdown(f"""
                            <div class="feature-card llm-card">
                                <h4>📋 Rapport Phi-3</h4>
                                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; max-height: 300px; overflow-y: auto; border: 1px solid #e0e0e0;">
                                    <p>{rapport}</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"Erreur API Phi-3: {response.status_code}")
                except Exception as e:
                    st.error(f"Erreur génération rapport: {e}")
        else:
            if not ollama_online:
                st.info("ℹ️ LLM non disponible - Ollama hors ligne")
            else:
                st.info("ℹ️ Fusion non disponible - Impossible de générer le rapport")
    
    with col2:
        # Analyse de Sentiment
        st.subheader("💭 Analyse de Sentiment")
        
        try:
            articles = services['sentiment_service'].get_news_articles(ticker, 10)
            
            if articles:
                # Afficher le dernier article analysé
                dernier_article = articles[0]
                sentiment = services['sentiment_service'].analyze_article_sentiment(dernier_article)
                
                # Score avec justification
                st.markdown(f"""
                **Dernier sentiment :** {sentiment['label']} ({sentiment['sentiment_score']:.2f}) • {dernier_article['timestamp'].strftime('%H:%M')}
                """)
                
                # Justification du sentiment
                justification_elements = []
                if sentiment['sentiment_score'] > 0.1:
                    justification_elements.append("📈 Tendance positive")
                elif sentiment['sentiment_score'] < -0.1:
                    justification_elements.append("📉 Tendance négative")
                else:
                    justification_elements.append("📊 Tendance neutre")
                
                if len(articles) > 5:
                    justification_elements.append("📰 Volume d'articles élevé")
                elif len(articles) > 2:
                    justification_elements.append("📰 Volume d'articles modéré")
                else:
                    justification_elements.append("📰 Volume d'articles faible")
                
                st.markdown(f"""
                **Justification :** {' | '.join(justification_elements)}
                """)
                
                # Bouton pour afficher/masquer l'historique
                if 'show_sentiment_history' not in st.session_state:
                    st.session_state.show_sentiment_history = False
                
                if st.button("📜 Afficher/Masquer l'historique des articles", key="toggle_sentiment_history"):
                    st.session_state.show_sentiment_history = not st.session_state.show_sentiment_history
                
                # Container avec scroll vertical pour l'historique
                if st.session_state.show_sentiment_history:
                    st.markdown("""
                    <div style="max-height: 300px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem;">
                    """, unsafe_allow_html=True)
                    
                    for i, article in enumerate(articles[1:], 1):
                        try:
                            sentiment = services['sentiment_service'].analyze_article_sentiment(article)
                            
                            st.markdown(f"""
                            <div class="feature-card sentiment-card" style="margin-bottom: 1rem;">
                                <h5>{article['title']}</h5>
                                <p><small>{article['source']} • {article['timestamp'].strftime('%H:%M')}</small></p>
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <span style="font-size: 1.5rem;">{sentiment['emoji']}</span>
                                    <span><strong>{sentiment['label']}</strong> ({sentiment['sentiment_score']:.2f})</span>
                                    <div style="flex: 1; background: #e9ecef; height: 8px; border-radius: 4px;">
                                        <div style="width: {(sentiment['sentiment_score'] + 1) * 50}%; background: {sentiment['color']}; height: 100%; border-radius: 4px;"></div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.markdown(f"""
                            <div class="feature-card sentiment-card" style="margin-bottom: 1rem; border-left-color: #dc3545;">
                                <h5>{article['title']}</h5>
                                <p><small>{article['source']} • {article['timestamp'].strftime('%H:%M')}</small></p>
                                <p style="color: #dc3545;"><strong>❌ Erreur d'analyse</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Aucun article disponible")
                
        except Exception as e:
            st.error(f"Erreur Analyse Sentiment: {str(e)}")
    
    # 4. SERVICE LLM - Synthèses concises avec limitation de tokens
    st.header("🧠 Service LLM - Synthèse de Trading")
    
    col_llm1, col_llm2 = st.columns([2, 1])
    
    with col_llm1:
        try:
            llm_service = LLMService()
            
            # Vérifier le statut du service
            llm_status = llm_service.check_service_status()
            
            if llm_status['online'] and llm_status['model_available']:
                st.success(f"✅ {llm_status['status']}")
                
                # Boutons de contrôle
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                
                with col_btn1:
                    if st.button("🔄 Générer Synthèse", key="generate_synthesis"):
                        if fusion_available and market_status["is_open"] and ticker == "SPY":
                            with st.spinner("Génération de la synthèse..."):
                                current_price = services['data_service'].load_data(ticker)['CLOSE'].iloc[-1]
                                sentiment_score = services['sentiment_service'].get_sentiment_score(ticker)
                                
                                synthesis = llm_service.generate_trading_synthesis(
                                    ticker, fusion_data['recommendation'], 
                                    fusion_data['score'], current_price, sentiment_score
                                )
                                
                                if synthesis['success']:
                                    st.session_state['llm_synthesis'] = synthesis
                                    llm_service.save_synthesis(ticker, synthesis)
                                    st.success("✅ Synthèse générée et sauvegardée")
                                else:
                                    st.error(f"❌ Erreur: {synthesis['synthesis']}")
                        else:
                            st.warning("⚠️ Synthèse disponible uniquement pour SPY en marché ouvert")
                
                with col_btn2:
                    if st.button("📊 Statistiques", key="llm_stats"):
                        st.session_state['show_llm_stats'] = not st.session_state.get('show_llm_stats', False)
                
                with col_btn3:
                    if st.button("🗑️ Effacer", key="clear_synthesis"):
                        if 'llm_synthesis' in st.session_state:
                            del st.session_state['llm_synthesis']
                        st.rerun()
                
                # Afficher la synthèse si disponible
                if 'llm_synthesis' in st.session_state:
                    synthesis = st.session_state['llm_synthesis']
                    st.markdown(f"""
                    <div class="feature-card" style="margin-top: 1rem;">
                        <h5>📝 Synthèse LLM ({synthesis['model']})</h5>
                        <p style="font-size: 0.9rem; line-height: 1.4;">{synthesis['synthesis']}</p>
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #666; margin-top: 0.5rem;">
                            <span>Tokens: {synthesis['tokens_used']}</span>
                            <span>{synthesis['timestamp'][:19]}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Afficher les statistiques si demandées
                if st.session_state.get('show_llm_stats', False):
                    st.markdown("**📊 Statistiques LLM:**")
                    st.json({
                        "Modèle": llm_status['model'],
                        "Statut": llm_status['status'],
                        "Modèles disponibles": llm_status['models'][:3]  # Limiter l'affichage
                    })
            
            else:
                st.error(f"❌ {llm_status['status']}")
                st.info("💡 Pour activer le service LLM: `ollama pull phi3:mini`")
        
        except Exception as e:
            st.error(f"❌ Erreur Service LLM: {str(e)}")
    
    with col_llm2:
        # Statistiques de vérification
        try:
            from gui.services.verification_service import VerificationService
            verification_service = VerificationService()
            
            stats = verification_service.get_accuracy_stats(ticker, 7)
            
            st.markdown(f"""
            <div class="feature-card">
                <h6>📈 Précision (7j)</h6>
                <div style="font-size: 0.9rem;">
                    <div>Décisions: {stats['total_decisions']}</div>
                    <div>Précision: {stats['accuracy_rate']:.1%}</div>
                    <div>Cohérentes: {stats['coherent_decisions']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"⚠️ Statistiques indisponibles: {str(e)}")
    
    
    # 5. ÉTATS DES SERVICES - Déplacé en fin de page et corrigé
    st.header("🔧 États des Services")
    
    try:
        # Récupérer les statuts des services
        articles = services['sentiment_service'].get_news_articles(ticker, 10)
        article_count = len(articles) if articles else 0
        data = services['data_service'].load_data(ticker)
        
        # Corriger l'erreur strftime
        if not data.empty and hasattr(data.index[-1], 'strftime'):
            last_update = data.index[-1].strftime('%H:%M')
        elif not data.empty:
            # Si l'index n'a pas strftime, utiliser la dernière date disponible
            try:
                last_update = data.index[-1].strftime('%H:%M')
            except:
                last_update = 'N/A'
        else:
            last_update = 'N/A'
        
        # Construire la ligne de statut
        if market_status["is_open"]:
            market_status_text = "🟢 Marché Ouvert (EST)"
            market_time = f"Heure: {market_status['current_time']}"
            next_info = f"Fermeture: {market_status['next_close']}"
            warning = ""
        else:
            market_status_text = "🔴 Marché Fermé (EST)"
            market_time = f"Heure: {market_status['current_time']}"
            next_info = f"Ouverture: {market_status['next_open']}"
            warning = "⚠️ Aucune prédiction disponible en dehors des heures de marché"
        
        # Services status - Logique corrigée selon état du marché
        crawler_status = "🟢" if article_count > 0 else "🔴"
        
        # Prix : données historiques toujours disponibles, mais pas de prix en temps réel si marché fermé
        if not data.empty:
            if market_status["is_open"]:
                price_status = "🟢"  # Données historiques + marché ouvert
            else:
                price_status = "🟡"  # Données historiques seulement (pas de temps réel)
        else:
            price_status = "🔴"  # Pas de données du tout
        
        # Fusion : seulement si données prix disponibles ET marché ouvert
        if market_status["is_open"] and not data.empty:
            fusion_status = "🟢"
        else:
            fusion_status = "🔴"
        
        status_line = f"{market_status_text} | {market_time} | {next_info}"
        if warning:
            status_line += f" | {warning}"
        status_line += f" | Articles: {crawler_status} {article_count} | Prix: {price_status} {last_update} | Fusion: {fusion_status}"
        
        st.markdown(f"""
        <div class="status-line">
            {status_line}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Erreur statut services: {e}")


def _check_fusion_dependencies(services, ticker, market_status):
    """Vérifie les dépendances critiques pour le calcul de fusion"""
    errors = []
    fusion_available = True
    
    try:
        # 1. Vérifier la disponibilité des données de prix
        try:
            data = services['data_service'].load_data(ticker)
            if data.empty:
                errors.append({
                    'severity': 'critical',
                    'title': 'Données Prix Manquantes',
                    'message': 'Aucune donnée de prix disponible pour le calcul de fusion.'
                })
                fusion_available = False
        except Exception as e:
            errors.append({
                'severity': 'critical',
                'title': 'Erreur Chargement Prix',
                'message': f'Impossible de charger les données de prix: {str(e)}'
            })
            fusion_available = False
        
        # 2. Vérifier la prédiction (seulement pour SPY)
        if ticker == "SPY":
            try:
                prediction = services['prediction_service'].predict(data, horizon=20)
                if not prediction or 'predictions' not in prediction:
                    errors.append({
                        'severity': 'warning',
                        'title': 'Prédiction Indisponible',
                        'message': 'Les prédictions LSTM ne sont pas disponibles. Utilisation de valeurs par défaut.'
                    })
            except Exception as e:
                errors.append({
                    'severity': 'warning',
                    'title': 'Erreur Prédiction',
                    'message': f'Erreur lors du calcul de prédiction: {str(e)}'
                })
        
        # 3. Vérifier le sentiment
        try:
            articles = services['sentiment_service'].get_news_articles(ticker, 5)
            if not articles:
                errors.append({
                    'severity': 'warning',
                    'title': 'Articles Manquants',
                    'message': 'Aucun article disponible pour l\'analyse de sentiment.'
                })
        except Exception as e:
            errors.append({
                'severity': 'warning',
                'title': 'Erreur Sentiment',
                'message': f'Erreur lors de la récupération des articles: {str(e)}'
            })
        
        # 4. Vérifier l'état du marché
        if not market_status["is_open"] and ticker == "SPY":
            errors.append({
                'severity': 'info',
                'title': 'Marché Fermé',
                'message': 'Le marché est fermé. Les prédictions peuvent être moins fiables.'
            })
    
    except Exception as e:
        errors.append({
            'severity': 'critical',
            'title': 'Erreur Système',
            'message': f'Erreur lors de la vérification des dépendances: {str(e)}'
        })
        fusion_available = False
    
    return fusion_available, errors


def _get_price_signal(services, ticker):
    """Récupère le signal de prix"""
    try:
        data = services['data_service'].load_data(ticker)
        if data.empty:
            return 0.0
        
        # Calculer la tendance sur les 5 dernières périodes
        recent_data = data.tail(5)
        if len(recent_data) < 2:
            return 0.0
        
        # Tendance basée sur la variation moyenne
        price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
        
        # Normaliser entre 0 et 1 (0.5 = neutre)
        signal = 0.5 + (price_change * 2)  # Multiplier par 2 pour amplifier
        return max(0.0, min(1.0, signal))  # Clamper entre 0 et 1
    
    except Exception:
        return 0.0


def _get_sentiment_signal(services, ticker):
    """Récupère le signal de sentiment"""
    try:
        sentiment_summary = services['sentiment_service'].get_sentiment_summary(ticker)
        if 'avg_sentiment' in sentiment_summary:
            # Normaliser de [-1, 1] vers [0, 1]
            return (sentiment_summary['avg_sentiment'] + 1) / 2
        return 0.5  # Neutre par défaut
    except Exception:
        return 0.5


def _get_prediction_signal(services, ticker):
    """Récupère le signal de prédiction"""
    try:
        if ticker != "SPY":
            return 0.5  # Neutre pour NVDA
        
        data = services['data_service'].load_data(ticker)
        prediction = services['prediction_service'].predict(data, horizon=20)
        
        if prediction and 'predictions' in prediction and prediction['predictions']:
            # Utiliser la première prédiction
            pred_value = prediction['predictions'][0] if prediction['predictions'] else 0.5
            return max(0.0, min(1.0, pred_value))
        
        return 0.5  # Neutre par défaut
    except Exception:
        return 0.5


def _check_market_status():
    """Vérifie l'état du marché (ouvert/fermé) - Horaires US (EST)"""
    from datetime import timezone, timedelta
    
    # Créer un timezone EST (UTC-5) ou EDT (UTC-4) selon la saison
    # Pour simplifier, on utilise EST (UTC-5)
    est = timezone(timedelta(hours=-5))
    now_est = datetime.now(est)
    current_time = now_est.strftime("%H:%M")
    
    # Heures de marché US (9h30 - 16h00 EST, du lundi au vendredi)
    market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Vérifier si c'est un jour de semaine
    is_weekday = now_est.weekday() < 5  # 0-4 = lundi-vendredi
    
    if is_weekday and market_open <= now_est <= market_close:
        return {
            "is_open": True,
            "current_time": current_time,
            "timezone": "EST",
            "next_close": market_close.strftime("%H:%M"),
            "next_open": "09:30" if now_est.date() == market_open.date() else "Lundi 09:30"
        }
    else:
        # Calculer la prochaine ouverture
        if now_est.weekday() >= 5:  # Weekend
            days_until_monday = 7 - now_est.weekday()
            next_open = now_est + timedelta(days=days_until_monday)
        elif now_est < market_open:  # Avant l'ouverture
            next_open = now_est
        else:  # Après la fermeture
            next_open = now_est + timedelta(days=1)
        
        next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)
        
        return {
            "is_open": False,
            "current_time": current_time,
            "timezone": "EST",
            "next_open": next_open.strftime("%A %H:%M"),
            "next_close": "16:00"
        }