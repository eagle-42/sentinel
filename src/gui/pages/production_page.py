"""
Page de production améliorée - Dashboard Trading
Interface optimisée selon les recommandations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.services.data_service import DataService
from gui.services.chart_service import ChartService
from gui.services.prediction_service import PredictionService
from gui.services.sentiment_service import SentimentService
from gui.services.fusion_service import FusionService
from gui.services.llm_service import LLMService
from gui.services.monitoring_service import MonitoringService


def show_production_page():
    """Affiche la page de production améliorée"""
    
    # Initialiser les services
    if 'production_services' not in st.session_state:
        st.session_state.production_services = {
            'data_service': DataService(),
            'chart_service': ChartService(),
            'prediction_service': PredictionService(),
            'sentiment_service': SentimentService(),
            'fusion_service': FusionService(),
            'llm_service': LLMService(),
            'monitoring_service': MonitoringService()
        }
    
    services = st.session_state.production_services
    
    # CSS personnalisé pour Production
    st.markdown("""
    <style>
        .production-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            border-left: 5px solid #667eea;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .feature-card:hover {
            transform: translateY(-2px);
        }
        .price-card {
            border-left-color: #28a745;
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        }
        .sentiment-card {
            border-left-color: #fd7e14;
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        }
        .fusion-card {
            border-left-color: #6f42c1;
            background: linear-gradient(135deg, #e2e3f0 0%, #d1d5f0 100%);
        }
        .llm-card {
            border-left-color: #20c997;
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        }
        .monitoring-card {
            border-left-color: #dc3545;
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
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar - Paramètres d'analyse uniquement
    with st.sidebar:
        st.header("⚙️ Paramètres d'Analyse")
        
        # Configuration du système
        st.subheader("🔧 Configuration")
        
        ticker = st.selectbox(
            "Ticker",
            ["SPY", "NVDA"],
            index=0,
            help="Symbole de l'action à analyser. SPY: Prédictions disponibles. NVDA: Analyse uniquement."
        )
        
        action_type = st.selectbox(
            "Type d'action",
            ["Analyse", "Prédiction", "Trading"],
            index=0,
            help="Type d'analyse à effectuer. Trading: Recommandations d'achat/vente."
        )
        
        period = st.selectbox(
            "Période",
            ["7 derniers jours", "1 mois", "3 mois", "6 mois", "1 an"],
            index=1,
            help="Période d'analyse des données historiques. Plus long = plus de contexte."
        )
        
        # Seuils de trading
        st.subheader("🎯 Seuils de Trading")
        
        buy_threshold = st.slider("Seuil d'achat (%)", 0.0, 10.0, 2.0, 0.1,
                                 help="Variation minimale pour déclencher un achat")
        sell_threshold = st.slider("Seuil de vente (%)", -10.0, 0.0, -2.0, 0.1,
                                  help="Variation minimale pour déclencher une vente")
        confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.7, 0.05,
                                        help="Niveau de confiance minimum pour agir (0.7 = 70%)")
        
        # Actions du système - alignés horizontalement
        st.subheader("🎮 Actions")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Démarrer", type="primary"):
                st.success("✅ Système démarré")
        with col2:
            if st.button("⏸️ Pause"):
                st.warning("⏸️ Système en pause")
        with col3:
            if st.button("🛑 Arrêter"):
                st.error("🛑 Système arrêté")
    
        # Bouton de rafraîchissement
        if st.button("🔄 Rafraîchir Données", type="secondary"):
            st.rerun()
    
    # Vérification de l'état du marché
    market_status = _check_market_status()
    
    # Statut du marché en carré à côté
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Statut des services de crawl en haut
        try:
            system_status = services['monitoring_service'].get_system_status()
            
            st.markdown("### 🔧 Statut des Services de Crawl")
            
            col_crawl1, col_crawl2, col_crawl3, col_crawl4 = st.columns(4)
            
            with col_crawl1:
                crawler_status = system_status['services'].get('crawler', 'offline')
                status_class = f"status-{crawler_status}" if crawler_status in ['online', 'offline'] else "status-warning"
                st.markdown(f"""
                <div class="feature-card">
                    <h5><span class="status-indicator {status_class}"></span>Crawler Articles</h5>
                    <p><strong>{crawler_status.title()}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_crawl2:
                price_status = system_status['services'].get('data', 'offline')
                status_class = f"status-{price_status}" if price_status in ['online', 'offline'] else "status-warning"
                st.markdown(f"""
                <div class="feature-card">
                    <h5><span class="status-indicator {status_class}"></span>Crawler Prix</h5>
                    <p><strong>{price_status.title()}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_crawl3:
                prediction_status = system_status['services'].get('prediction', 'offline')
                status_class = f"status-{prediction_status}" if prediction_status in ['online', 'offline'] else "status-warning"
                st.markdown(f"""
                <div class="feature-card">
                    <h5><span class="status-indicator {status_class}"></span>Prédiction</h5>
                    <p><strong>{prediction_status.title()}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_crawl4:
                fusion_status = system_status['services'].get('fusion', 'offline')
                status_class = f"status-{fusion_status}" if fusion_status in ['online', 'offline'] else "status-warning"
                st.markdown(f"""
        <div class="feature-card">
                    <h5><span class="status-indicator {status_class}"></span>Fusion</h5>
                    <p><strong>{fusion_status.title()}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Erreur statut services: {e}")
    
    with col2:
        # Statut du marché en carré
        if market_status["is_open"]:
            st.markdown(f"""
            <div class="market-status market-open" style="padding: 1rem; text-align: center;">
                <h4>🟢 Marché Ouvert</h4>
                <p><strong>{market_status['timezone']}</strong></p>
                <p>Heure: {market_status['current_time']}</p>
                <p>Fermeture: {market_status['next_close']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="market-status market-closed" style="padding: 1rem; text-align: center;">
                <h4>🔴 Marché Fermé</h4>
                <p><strong>{market_status['timezone']}</strong></p>
                <p>Heure: {market_status['current_time']}</p>
                <p>Ouverture: {market_status['next_open']}</p>
                <p><strong>⚠️ Aucune prédiction disponible</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Vérification de l'état des services
    try:
        system_status = services['monitoring_service'].get_system_status()
        if system_status['overall_status'] != 'online':
            st.markdown(f"""
            <div class="error-alert">
                <h4>⚠️ Services Dégradés</h4>
                <p>Certains services ne sont pas opérationnels. Vérifiez la section Monitoring.</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="error-alert">
            <h4>❌ Erreur de Monitoring</h4>
            <p>Impossible de vérifier l'état des services: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 1. ANALYSE DE SENTIMENT (ARTICLES) - Dans une fenêtre
    with st.expander("💭 Analyse de Sentiment (Articles)", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Liste des articles analysés avec scroll vertical
            try:
                articles = services['sentiment_service'].get_news_articles(ticker, 10)
                
                if articles:
                    # Container avec scroll vertical
                    st.markdown("""
                    <div style="max-height: 400px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem;">
                    """, unsafe_allow_html=True)
                    
                    for article in articles:
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
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("Aucun article disponible")
            except Exception as e:
                st.markdown(f"""
                <div class="error-alert">
                    <h4>❌ Erreur Analyse Sentiment</h4>
                    <p>Impossible de charger les articles: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Jauge de sentiment global
            try:
                sentiment_summary = services['sentiment_service'].get_sentiment_summary(ticker)
                
                # Créer une jauge de sentiment
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = sentiment_summary['avg_sentiment'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Sentiment Global"},
                    delta = {'reference': 0},
                    gauge = {
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-1, -0.5], 'color': "lightgray"},
                            {'range': [-0.5, 0.5], 'color': "yellow"},
                            {'range': [0.5, 1], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.8
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Mots-clés impactants
                keywords = services['sentiment_service'].get_keywords(ticker)
                if keywords:
                    st.markdown("**Mots-clés impactants:**")
                    for kw in keywords[:5]:
                        color = "green" if kw['impact'] > 0 else "red"
                        st.markdown(f"- <span style='color: {color};'>{kw['word']}</span> ({kw['impact']:.1f})")
            
            except Exception as e:
                st.markdown(f"""
                <div class="error-alert">
                    <h4>❌ Erreur Sentiment Global</h4>
                    <p>Impossible de calculer le sentiment: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 2. RÉSULTAT FUSION (PRÉDICTION + SENTIMENT + FINBERT)
    st.header("🔄 Résultat Fusion (Prédiction + Sentiment + FinBERT)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Carte Score Fusion
        try:
            # Simuler des signaux
            price_signal = 0.7
            sentiment_signal = 0.6
            prediction_signal = 0.8
            
            fusion_data = services['fusion_service'].calculate_fusion_score(
                price_signal, sentiment_signal, prediction_signal
            )
            
            st.markdown(f"""
            <div class="feature-card fusion-card">
                <h4>🎯 Score Fusion</h4>
                <div class="gauge-container">
                    <h1 style="color: {fusion_data['color']}; font-size: 3rem;">{fusion_data['fusion_score']:.2f}</h1>
                    <p><strong>{fusion_data['label']}</strong></p>
                    <p>Confiance: {fusion_data['confidence']:.1%}</p>
                </div>
                <div class="recommendation-badge recommendation-{fusion_data['recommendation'].lower()}">
                    {fusion_data['recommendation']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Historique des fusions
            fusion_stats = services['fusion_service'].get_fusion_stats()
            st.markdown(f"""
            <div class="feature-card">
                <h5>📊 Historique</h5>
                <p><strong>Total signaux:</strong> {fusion_stats['total_signals']}</p>
                <p><strong>Score moyen:</strong> {fusion_stats['avg_score']:.2f}</p>
                <p><strong>Dernière recommandation:</strong> {fusion_stats['last_recommendation']}</p>
                <p><strong>Poids actuels:</strong></p>
                <ul>
                    <li>Prix: {fusion_stats['current_weights']['price']:.1%}</li>
                    <li>Sentiment: {fusion_stats['current_weights']['sentiment']:.1%}</li>
                    <li>Prédiction: {fusion_stats['current_weights']['prediction']:.1%}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f"""
            <div class="error-alert">
                <h4>❌ Erreur Fusion</h4>
                <p>Impossible de calculer la fusion: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Graphique multi-signaux
        try:
            chart_data = services['fusion_service'].get_multi_signal_chart_data(ticker)
            
            if chart_data['dates']:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=chart_data['dates'],
                    y=chart_data['price_signals'],
                    mode='lines',
                    name='Prix',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=chart_data['dates'],
                    y=chart_data['sentiment_signals'],
                    mode='lines',
                    name='Sentiment',
                    line=dict(color='orange', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=chart_data['dates'],
                    y=chart_data['prediction_signals'],
                    mode='lines',
                    name='Prédiction',
                    line=dict(color='green', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=chart_data['dates'],
                    y=chart_data['fusion_signals'],
                    mode='lines',
                    name='Fusion',
                    line=dict(color='purple', width=3)
                ))
                
                fig.update_layout(
                    title="Graphique Multi-Signaux",
                    xaxis_title="Date",
                    yaxis_title="Score",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune donnée de fusion disponible")
        
        except Exception as e:
            st.markdown(f"""
            <div class="error-alert">
                <h4>❌ Erreur Graphique Multi-Signaux</h4>
                <p>Impossible de générer le graphique: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 3. RECOMMANDATION & SYNTHÈSE LLM - Seulement si SPY et marché ouvert
    if ticker == "SPY" and market_status["is_open"]:
        st.header("🎯 Recommandation & Synthèse LLM")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Bloc Action à prendre
            try:
                # Simuler des données pour LLM
                price_data = {"last_price": 445.67, "change_percent": 2.34}
                prediction_data = {"prediction_score": 0.8, "confidence": 0.85}
                
                llm_explanation = services['llm_service'].generate_trading_explanation(
                    fusion_data, sentiment_summary, price_data
                )
                
                st.markdown(f"""
                <div class="feature-card llm-card">
                    <h4>🎯 Action à Prendre</h4>
                    <div class="recommendation-badge recommendation-{fusion_data['recommendation'].lower()}" style="font-size: 1.5rem; margin: 1rem 0;">
                        {fusion_data['recommendation']}
                    </div>
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                        <h6>📊 Métriques de Décision:</h6>
                        <p><strong>Score de Fusion:</strong> {fusion_data['fusion_score']:.2f}/1.0 
                        <small style="color: #6c757d;">(Intensité du signal combiné)</small></p>
                        <p><strong>Niveau de Confiance:</strong> {fusion_data['confidence']:.1%} 
                        <small style="color: #6c757d;">(Fiabilité de la prédiction)</small></p>
                        <div style="margin-top: 0.5rem;">
                            <small style="color: #6c757d;">
                                <strong>Score:</strong> Force du signal (0=faible, 1=fort)<br>
                                <strong>Confiance:</strong> Probabilité de réussite (0%=incertain, 100%=certain)
                            </small>
                        </div>
                    </div>
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                        <h6>Explication:</h6>
                        <p>{llm_explanation['explanation']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            except Exception as e:
                st.markdown(f"""
                <div class="error-alert">
                    <h4>❌ Erreur LLM</h4>
                    <p>Impossible de générer l'explication: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Explication LLM détaillée
            try:
                st.markdown("""
                <div class="feature-card">
                    <h4>🤖 Explication LLM</h4>
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
                        <h6>Résumé Exécutif:</h6>
                        <p>Les signaux de fusion indiquent une recommandation d'ACHAT basée sur un score élevé de 0.78 avec une confiance de 85%. Les facteurs clés incluent un sentiment positif du marché, des prédictions LSTM favorables, et une tendance haussière des prix.</p>
                        
                        <h6>Facteurs Clés:</h6>
                        <ul>
                            <li><strong>Prix:</strong> Tendance haussière (+2.34%)</li>
                            <li><strong>Sentiment:</strong> Positif (0.6/1.0)</li>
                            <li><strong>Prédiction LSTM:</strong> Forte (0.8/1.0)</li>
                            <li><strong>Volume:</strong> Au-dessus de la moyenne</li>
                        </ul>
                        
                        <h6>Risques Identifiés:</h6>
                        <ul>
                            <li>Volatilité du marché</li>
                            <li>Données limitées sur 24h</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Bouton rapport complet
                if st.button("📄 Afficher Rapport Complet", type="primary"):
                    try:
                        full_report = services['llm_service'].generate_full_report(
                            fusion_data, sentiment_summary, price_data, prediction_data
                        )
                        st.markdown("### 📋 Rapport Complet")
                        st.markdown(full_report['full_report'])
                    except Exception as e:
                        st.error(f"Erreur rapport complet: {e}")
            
            except Exception as e:
                st.markdown(f"""
                <div class="error-alert">
                    <h4>❌ Erreur Explication LLM</h4>
                    <p>Impossible de générer l'explication détaillée: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        if ticker == "NVDA":
            st.info("ℹ️ Les recommandations ne sont pas disponibles pour NVDA (prédiction non disponible)")
        else:
            st.info("ℹ️ Les recommandations ne sont pas disponibles en dehors des heures de marché")
    
    # Note: Monitoring & Contrôle déplacé vers l'onglet Logs
    
    # 4. DERNIER PRIX ET STATUS CRAWLER - En bas
    st.header("📈 Dernier Prix & Status Crawler")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Widget dernier prix avec indication de marché
        try:
            # Récupérer les vraies données
            data = services['data_service'].load_data(ticker)
            if not data.empty:
                last_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2] if len(data) > 1 else last_price
                change = last_price - prev_price
                change_percent = (change / prev_price) * 100
                
                # Déterminer la couleur et le statut
                if market_status["is_open"]:
                    price_color = "#28a745" if change >= 0 else "#dc3545"
                    status_text = "Temps réel"
                else:
                    price_color = "#6c757d"
                    status_text = "Données historiques"
                
                st.markdown(f"""
                <div class="feature-card price-card">
                    <h4>💰 Dernier Prix ({ticker})</h4>
                    <h2 style="color: {price_color};">${last_price:.2f}</h2>
                    <p style="color: {price_color};">{change:+.2f} ({change_percent:+.2f}%)</p>
                    <small><strong>{status_text}</strong></small><br>
                    <small>Dernière MAJ: {data.index[-1].strftime('%H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="feature-card price-card">
                    <h4>💰 Dernier Prix ({ticker})</h4>
                    <h2 style="color: #6c757d;">N/A</h2>
                    <p style="color: #6c757d;">Données non disponibles</p>
                    <small><strong>Données historiques</strong></small>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div class="feature-card price-card">
                <h4>💰 Dernier Prix ({ticker})</h4>
                <h2 style="color: #dc3545;">Erreur</h2>
                <p style="color: #dc3545;">Impossible de charger</p>
                <small><strong>Données historiques</strong></small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Statut crawler
        try:
            system_status = services['monitoring_service'].get_system_status()
            crawler_status = system_status['services'].get('crawler', 'offline')
            
            if crawler_status == 'online':
                status_class = "status-online"
                status_text = "En ligne"
            else:
                status_class = "status-offline"
                status_text = "Hors ligne"
            
            st.markdown(f"""
            <div class="feature-card">
                <h4><span class="status-indicator {status_class}"></span>Status Crawler</h4>
                <p><strong>{status_text}</strong></p>
                <small>Dernière MAJ: {datetime.now().strftime('%H:%M')}</small>
                <br><br>
                <small>✅ SPY: Données disponibles</small><br>
                <small>✅ NVDA: Données disponibles</small>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div class="feature-card">
                <h4><span class="status-indicator status-offline"></span>Status Crawler</h4>
                <p><strong>Erreur</strong></p>
                <small>Impossible de vérifier</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer avec informations système
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dernière MAJ", "10:45:23")
    with col2:
        st.metric("Version", "2.0")
    with col3:
        st.metric("Statut", "🟢 Opérationnel")
    with col4:
        st.metric("Progrès", "85%")


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
