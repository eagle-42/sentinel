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
import json

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import du gestionnaire de services centralisé
from gui.services.service_manager import service_manager
from gui.config.logging_config import get_logger

# Logger pour cette page
logger = get_logger("production_page")


def show_production_page():
    """Affiche la page de production optimisée"""
    
    logger.info("🔄 Chargement de la page de production")
    
    # Auto-refresh si le marché est ouvert (toutes les 30 secondes)
    market_status = _check_market_status()
    if market_status["is_open"]:
        # Auto-refresh toutes les 30 secondes quand le marché est ouvert
        if 'last_auto_refresh' not in st.session_state:
            st.session_state.last_auto_refresh = datetime.now()
        
        time_since_refresh = datetime.now() - st.session_state.last_auto_refresh
        if time_since_refresh.total_seconds() > 30:  # 30 secondes
            st.session_state.last_auto_refresh = datetime.now()
            
            # Mise à jour automatique des données toutes les 15 minutes
            # Utiliser l'heure UTC pour la cohérence avec les données
            now_utc = datetime.utcnow()
            
            if 'last_data_update' not in st.session_state:
                st.session_state.last_data_update = now_utc
            
            time_since_data_update = now_utc - st.session_state.last_data_update
            if time_since_data_update.total_seconds() > 900:  # 15 minutes
                st.session_state.last_data_update = now_utc
                logger.info(f"🔄 Mise à jour automatique des données 15min (UTC: {now_utc.strftime('%H:%M')})...")
                # Déclencher la mise à jour des données
                try:
                    from gui.services.data_monitor_service import DataMonitorService
                    data_monitor = DataMonitorService()
                    data_monitor.trigger_data_refresh("SPY")
                    data_monitor.trigger_data_refresh("NVDA")
                    logger.info("✅ Données mises à jour automatiquement")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur mise à jour auto: {e}")
            
            st.rerun()
    
    # Vérifier l'état des services AVANT d'initialiser quoi que ce soit
    services_stopped = not st.session_state.get('services_running', True)
    logger.info(f"🔧 Services arrêtés: {services_stopped}")
    
    # Utiliser le gestionnaire de services centralisé
    if not services_stopped:
        services = service_manager.get_services()
    else:
        # Services arrêtés - utiliser des services vides
        services = {
            'data_service': None,
            'chart_service': None,
            'prediction_service': None,
            'sentiment_service': None,
            'fusion_service': None,
            'llm_service': None,
            'monitoring_service': None,
            'data_monitor_service': None
        }
    
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
        
        # Période fixe à 7 jours pour l'analyse
        period = "7 derniers jours"
        
        # Configuration des seuils adaptatifs
        st.subheader("🎯 Seuils Adaptatifs")
        
        # Afficher les seuils actuels
        if services['fusion_service']:
            fusion_stats = services['fusion_service'].get_fusion_stats()
            current_thresholds = fusion_stats.get('current_thresholds', {'buy': 0.1, 'sell': -0.1})
        else:
            current_thresholds = {'buy': 0.1, 'sell': -0.1}
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Seuil BUY",
                f"{current_thresholds.get('buy', 0.1):.3f}",
                help="Signal > seuil → ACHETER"
            )
        with col2:
            st.metric(
                "Seuil SELL", 
                f"{current_thresholds.get('sell', -0.1):.3f}",
                help="Signal < seuil → VENDRE"
            )
        
        # Information sur l'adaptation
        st.info("🔄 Les seuils s'adaptent automatiquement selon la volatilité du marché")
        
        # Actions du système
        st.subheader("🎮 Actions")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Démarrer Services", type="primary", key="start_services", help="Démarre tous les services de trading (crawler, prédiction, sentiment, fusion)"):
                logger.info("🚀 Démarrage des services demandé par l'utilisateur")
                st.success("✅ Services démarrés")
                st.session_state.services_running = True
                logger.info("✅ Services démarrés avec succès")
        with col2:
            services_running = st.session_state.get('services_running', True)
            if st.button("🛑 Arrêter Services", type="secondary", key="stop_services", help="Arrête tous les services de trading et libère la mémoire", disabled=not services_running):
                logger.info("🛑 Arrêt des services demandé par l'utilisateur")
                # Vraiment arrêter les services et libérer la mémoire
                service_manager.stop_all_services()
                st.warning("⚠️ Services arrêtés - Mémoire libérée")
                st.session_state.services_running = False
                logger.info("✅ Services arrêtés avec succès")
                st.rerun()  # Recharger pour voir l'effet immédiat
        with col3:
            if st.button("🔄 Rafraîchir", type="secondary", key="refresh_page", help="Recharge la page et actualise tous les affichages"):
                st.rerun()
        
        # Mise à jour des données
        st.subheader("📊 Données")
        
        if st.button("📈 Mettre à jour les prix", type="secondary", key="update_prices", help="Met à jour les données de prix 15min depuis l'API et recharge les graphiques", disabled=services_stopped):
            if services_stopped:
                st.warning("⚠️ Services arrêtés - Impossible de mettre à jour les données")
            else:
                with st.spinner("Mise à jour des données de prix..."):
                    try:
                        except Exception as e:
                            logger.error(f"Erreur mise à jour données: {e}")
                            success = False
                        # Utiliser le service de monitoring pour la mise à jour
                        if services['data_monitor_service']:
                            success = services['data_monitor_service'].trigger_data_refresh(ticker)
                        else:
                            success = False
                    
                    if success:
                        # Vider tous les caches Streamlit pour forcer le rechargement
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        st.success("✅ Données de prix mises à jour")
                        st.rerun()  # Recharger pour voir les nouvelles données
                    else:
                        st.warning("⚠️ Mise à jour partielle - Vérifiez les logs")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la mise à jour: {e}")
                    st.info("ℹ️ Utilisation des données en cache")
        
        # Section d'aide
        st.subheader("❓ Aide")
        with st.expander("💡 Que font ces boutons ?", expanded=False):
            st.markdown("""
            **🎮 Actions :**
            - **🚀 Démarrer Services** : Lance tous les services de trading (crawler, prédiction, sentiment, fusion)
            - **🛑 Arrêter Services** : Arrête tous les services (simulation pour l'interface)
            - **🔄 Rafraîchir** : Recharge la page et actualise tous les affichages
            
            **📊 Données :**
            - **📈 Mettre à jour les prix** : Met à jour les données de prix 15min depuis l'API et recharge les graphiques
            """)
    
            # Afficher un avertissement si les services sont arrêtés
            if services_stopped:
                st.warning("🛑 **Services arrêtés** - Les services de trading sont actuellement arrêtés. Seules les données historiques sont affichées.")
            else:
                # Indicateur de mise à jour automatique
                if market_status["is_open"]:
                    st.info("🔄 **Mise à jour automatique activée** - Les données se mettent à jour automatiquement toutes les 15 minutes pendant les heures de marché US.")
                else:
                    st.info("⏰ **Marché fermé** - Mise à jour automatique suspendue. Les données seront mises à jour à la prochaine ouverture du marché.")
    
    
    # 1. COMPTE RENDU D'ACTIVITÉ - Version restructurée en haut
    st.header("📊 Compte Rendu d'Activité")
    
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
            
            if services['fusion_service']:
            fusion_data = services['fusion_service'].calculate_fusion_score(
                price_signal, sentiment_signal, prediction_signal
            )
            else:
                fusion_data = {'recommendation': 'HOLD', 'fusion_score': 0.0}
        except Exception as e:
            st.error(f"Erreur calcul fusion: {e}")
            fusion_data = None
    
    # KPI principaux en haut - Ordre réorganisé
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        # 1. Statut marché avec détails
        market_color = "#28a745" if market_status["is_open"] else "#dc3545"
        market_text = "Ouvert" if market_status["is_open"] else "Fermé"
        next_action = market_status.get("next_close", "16:00") if market_status["is_open"] else market_status.get("next_open", "09:30")
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-value" style="color: {market_color};">{market_text}</div>
            <div class="kpi-label">Marché {market_status.get('timezone', 'EST')}</div>
            <div class="kpi-label" style="font-size: 0.8rem; color: #888; margin-top: 0.3rem;">
                {next_action}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # 2. Dernière MAJ + Prochain Update (avec fuseaux horaires)
        import pytz
        try:
            # Heure locale française
            fr_tz = pytz.timezone('Europe/Paris')
            current_time_fr = datetime.now(fr_tz).strftime("%H:%M:%S")
            
            # Heure US (EST/EDT)
            us_tz = pytz.timezone('US/Eastern')
            current_time_us = datetime.now(us_tz).strftime("%H:%M:%S")
            
            next_update = (datetime.now() + timedelta(minutes=15)).strftime("%H:%M")
            
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #667eea;">{current_time_fr}</div>
                <div class="kpi-label">Dernière MAJ (FR)</div>
                <div class="kpi-label" style="font-size: 0.7rem; color: #888; margin-top: 0.3rem;">US: {current_time_us}</div>
                <div class="kpi-label" style="font-size: 0.7rem; color: #888; margin-top: 0.2rem;">Prochain: {next_update}</div>
            </div>
            """, unsafe_allow_html=True)
        except ImportError:
            # Fallback si pytz n'est pas disponible
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
        # 5. Recommandation - Afficher le dernier résultat ou statut approprié
        if fusion_available and market_status["is_open"] and ticker == "SPY" and fusion_data:
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
                    <div class="kpi-value" style="color: #ffc107;">En cours</div>
                    <div class="kpi-label">Recommandation</div>
                </div>
                """, unsafe_allow_html=True)
        elif ticker == "NVDA":
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #ffc107;">Analyse</div>
                <div class="kpi-label">Recommandation</div>
                <div class="kpi-label" style="font-size: 0.8rem; color: #888;">NVDA: Analyse uniquement</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #ffc107;">En attente</div>
                <div class="kpi-label">Recommandation</div>
                <div class="kpi-label" style="font-size: 0.8rem; color: #888;">Ouverture marché</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col6:
                # 6. % de réussite des décisions (au lieu du seuil)
                try:
                    # Récupérer les données de validation historique
                    from gui.services.historical_validation_service import HistoricalValidationService
                    historical_validation = HistoricalValidationService()
                    validation_summary = historical_validation.get_validation_summary(ticker, days=7)
                    summary_stats = validation_summary.get('summary_stats', {})
                    
                    # Calculer le % de réussite
                    total_decisions = summary_stats.get('total_decisions', 0)
                    correct_decisions = summary_stats.get('correct_decisions', 0)
                    success_rate = (correct_decisions / total_decisions * 100) if total_decisions > 0 else 0
                    
                    # Couleur basée sur la performance
                    if success_rate >= 70:
                        color = "#28a745"  # Vert pour bonne performance
                    elif success_rate >= 50:
                        color = "#ffc107"  # Orange pour performance moyenne
        else:
                        color = "#dc3545"  # Rouge pour performance faible
                    
                st.markdown(f"""
                <div class="kpi-box">
                        <div class="kpi-value" style="color: {color};">{success_rate:.1f}%</div>
                        <div class="kpi-label">Réussite</div>
                        <div class="kpi-label" style="font-size: 0.8rem; color: #888;">{total_decisions} décisions</div>
                </div>
                """, unsafe_allow_html=True)
                except Exception as e:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #6c757d;">N/A</div>
                        <div class="kpi-label">Réussite</div>
                <div class="kpi-label" style="font-size: 0.8rem; color: #888;">Non disponible</div>
            </div>
            """, unsafe_allow_html=True)
    
    
    
    # 3. GRAPHIQUE 15MIN DU JOUR - Données réelles avec monitoring
    st.header("📈 Graphique 15min du Jour")
    
    # Vérifier l'état des données 15min
    if services['data_monitor_service']:
    data_summary = services['data_monitor_service'].get_data_summary(ticker)
                    else:
        data_summary = {'available': False, 'message': 'Service de monitoring non disponible'}
    
    # Afficher le graphique si les données sont disponibles
    if data_summary.get('available', False):
        try:
            if services['data_monitor_service']:
            data_15min, metadata = services['data_monitor_service'].get_latest_15min_data(ticker)
            else:
                data_15min = pd.DataFrame()
                metadata = {'last_update': datetime.now()}
            
            if not data_15min.empty:
                # Filtrer les données des 7 derniers jours
                seven_days_ago = datetime.now() - timedelta(days=7)
                # Créer une copie pour éviter le warning SettingWithCopyWarning
                data_15min_copy = data_15min.copy()
                # Convertir les timestamps en datetime naif pour la comparaison
                data_15min_copy = data_15min_copy.copy()
                data_15min_copy['ts_utc_naive'] = data_15min_copy['ts_utc'].apply(lambda x: x.replace(tzinfo=None) if x.tzinfo is not None else x)
                data_recent = data_15min_copy[data_15min_copy['ts_utc_naive'] >= seven_days_ago].copy()
                
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
    
    
    # Layout en 3 colonnes : Analyse sentiment, Service LLM, Graphiques de performance
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    
    with col1:
        # Analyse de Sentiment
        st.subheader("💭 Analyse de Sentiment")
        
        # Analyse de sentiment simple
        try:
            if services['sentiment_service']:
            articles = services['sentiment_service'].get_news_articles(ticker, 10)
            else:
                articles = []
            
            if articles:
                # Trier les articles par date (plus récent en premier)
                articles = sorted(articles, key=lambda x: x['timestamp'], reverse=True)
                
                # Afficher le dernier article analysé
                dernier_article = articles[0]
                if services['sentiment_service']:
                sentiment = services['sentiment_service'].analyze_article_sentiment(dernier_article)
                else:
                    sentiment = {'sentiment_score': 0.0, 'sentiment_label': 'Neutre', 'article_count': 0, 'timestamp': 'N/A'}
                
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
                
                # Affichage simple
                st.markdown(f"**Dernier sentiment :** {sentiment['label']} ({sentiment['sentiment_score']:.2f}) • {dernier_article['timestamp'].strftime('%H:%M')}")
                st.markdown(f"**Justification :** {' | '.join(justification_elements)}")
                
                # Liste des articles
                st.markdown("**📰 Liste des articles :**")
                for i, article in enumerate(articles[:5], 1):  # Limiter à 5 articles
                    try:
                        if services['sentiment_service']:
                        article_sentiment = services['sentiment_service'].analyze_article_sentiment(article)
                        else:
                            article_sentiment = {'sentiment_score': 0.0, 'sentiment_label': 'Neutre'}
                        with st.expander(f"Article {i}: {article['title'][:50]}...", expanded=False):
                            st.write(f"**Source:** {article['source']} • {article['timestamp'].strftime('%H:%M')}")
                            st.write(f"**Sentiment:** {article_sentiment['emoji']} {article_sentiment['label']} ({article_sentiment['sentiment_score']:.2f})")
                    except Exception as e:
                        with st.expander(f"Article {i}: {article['title'][:50]}...", expanded=False):
                            st.error(f"❌ Erreur d'analyse: {str(e)}")
            else:
                st.info("Aucun article disponible")
                
        except Exception as e:
            st.error(f"Erreur Analyse Sentiment: {str(e)}")
    
    with col2:
        # Service LLM
        st.subheader("🧠 Service LLM")
        
        # Service LLM simple
        try:
            if services['llm_service']:
                llm_service = services['llm_service']
            else:
                llm_service = None
            if llm_service:
            llm_status = llm_service.check_service_status()
            else:
                llm_status = {'online': False, 'model_available': False, 'status': 'Service non disponible'}
            
            # Affichage du statut
            if llm_status['online'] and llm_status['model_available']:
                st.success("✅ Service LLM actif")
                
                # Génération automatique de synthèse si fusion disponible
                if fusion_available and market_status["is_open"] and ticker == "SPY" and fusion_data:
                    try:
                        if services['data_service']:
                        current_price = services['data_service'].load_data(ticker)['CLOSE'].iloc[-1]
                        else:
                            current_price = 0.0
                        
                        if services['sentiment_service']:
                        sentiment_score = services['sentiment_service'].get_sentiment_score(ticker)
                        else:
                            sentiment_score = 0.0
                        
                        synthesis = llm_service.generate_trading_synthesis(
                            ticker, fusion_data['recommendation'], 
                            fusion_data.get('fusion_score', 0.0), current_price, sentiment_score
                        )
                        
                        if synthesis['success']:
                            st.session_state['llm_synthesis'] = synthesis
                            llm_service.save_synthesis(ticker, synthesis)
                            
                            with st.expander("📝 Analyse LLM Automatique", expanded=True):
                                st.write(synthesis['synthesis'])
                                st.caption(f"Modèle: {synthesis['model']} • {synthesis['timestamp'][:19]}")
                        else:
                            st.warning(f"⚠️ {synthesis['synthesis']}")
                            
                    except Exception as e:
                        if "timeout" in str(e).lower():
                            st.warning("⚠️ Service LLM occupé - Réessayez plus tard")
                        else:
                            st.warning(f"⚠️ Erreur génération: {str(e)[:50]}...")
                else:
                    # Marché fermé ou pas de données de fusion
                    if not market_status["is_open"]:
                        st.info("🕐 Marché fermé - Synthèses disponibles uniquement en heures d'ouverture")
                    elif ticker != "SPY":
                        st.info("ℹ️ Synthèses LLM disponibles uniquement pour SPY")
                    else:
                        st.info("ℹ️ En attente des données de fusion...")
                
                # Afficher l'historique des synthèses
                try:
                    from constants import CONSTANTS
                    synthesis_path = CONSTANTS.get_data_path() / "trading" / "llm_synthesis"
                    synthesis_files = list(synthesis_path.glob(f"{ticker}_synthesis_*.json"))
                    
                    if synthesis_files:
                        # Prendre le fichier le plus récent
                        latest_file = max(synthesis_files, key=lambda x: x.stat().st_mtime)
                        
                        with open(latest_file, 'r') as f:
                            syntheses = json.load(f)
                        
                        if syntheses:
                            st.markdown("**📚 Historique des Synthèses:**")
                            for i, synthesis in enumerate(syntheses[-3:], 1):
                                with st.expander(f"Synthèse #{len(syntheses) - 3 + i}", expanded=False):
                                    st.write(synthesis['synthesis'])
                                    st.caption(f"{synthesis['timestamp'][:19]} • {synthesis['tokens_used']} mots")
                        else:
                            st.info("Aucune synthèse disponible dans l'historique")
                    else:
                        st.info("Aucun historique de synthèses trouvé")
                        
                except Exception as e:
                    st.warning(f"⚠️ Erreur chargement historique: {str(e)}")
            
            else:
                st.error(f"❌ {llm_status['status']}")
                st.info("💡 Pour activer le service LLM: `ollama pull phi3:mini`")
            
        
        except Exception as e:
            st.error(f"❌ Erreur Service LLM: {str(e)}")
    
                    with col3:
        # Graphiques de Performance
        st.subheader("📊 Graphiques de Performance")
        
        try:
            # Récupérer les données de validation historique
            from gui.services.historical_validation_service import HistoricalValidationService
            historical_validation = HistoricalValidationService()
            validation_summary = historical_validation.get_validation_summary(ticker, days=7)
            summary_stats = validation_summary.get('summary_stats', {})
            
            if summary_stats.get('total_decisions', 0) > 0:
                st.info(f"📊 {summary_stats.get('total_decisions', 0)} décisions analysées sur 7 jours")
                # Graphique de répartition des décisions
                import plotly.express as px
                
                decision_types = ['BUY', 'SELL', 'HOLD']
                decision_counts = [
                    summary_stats.get('buy_decisions', 0),
                    summary_stats.get('sell_decisions', 0),
                    summary_stats.get('hold_decisions', 0)
                ]
                
                fig = px.bar(
                    x=decision_types,
                    y=decision_counts,
                    title="Répartition des Décisions",
                    color=decision_types,
                    color_discrete_map={'BUY': '#28a745', 'SELL': '#dc3545', 'HOLD': '#ffc107'}
                )
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Graphique de précision par type
                if summary_stats.get('buy_decisions', 0) > 0 or summary_stats.get('sell_decisions', 0) > 0:
                    accuracy_data = []
                    if summary_stats.get('buy_decisions', 0) > 0:
                        accuracy_data.append({'Type': 'BUY', 'Précision': summary_stats.get('buy_accuracy', 0.0) * 100})
                    if summary_stats.get('sell_decisions', 0) > 0:
                        accuracy_data.append({'Type': 'SELL', 'Précision': summary_stats.get('sell_accuracy', 0.0) * 100})
                    
                    if accuracy_data:
                        df_accuracy = pd.DataFrame(accuracy_data)
                        fig2 = px.bar(
                            df_accuracy,
                            x='Type',
                            y='Précision',
                            title="Précision par Type",
                            color='Type',
                            color_discrete_map={'BUY': '#28a745', 'SELL': '#dc3545'}
                        )
                        fig2.update_layout(showlegend=False, height=300, yaxis_title="Précision (%)")
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                        st.info("Aucune décision BUY/SELL pour afficher la précision")
                else:
                    st.info("Aucune décision BUY/SELL pour afficher la précision")
                
                        # Tableau des 5 décisions les plus récentes - Déplacé en bas
                pass  # Le tableau sera affiché en bas de page
            else:
                st.info("📈 Aucune donnée de performance disponible")
                st.markdown("""
                **💡 Pour voir les graphiques de performance :**
                - Attendez que des décisions de trading soient prises
                - Les graphiques apparaîtront automatiquement
                - Les données sont mises à jour toutes les 15 minutes
                """)
                
        except Exception as e:
            st.warning(f"⚠️ Erreur chargement graphiques: {str(e)}")
    
    
            # Section États des Services supprimée de cette position - sera déplacée en bas
    
    # Métriques de performance supprimées selon demande utilisateur
    
    # TABLEAU DES DÉCISIONS RÉCENTES - VERSION SIMPLIFIÉE
    st.header("📋 Décisions Récentes - Synthèse Clean")
    
    try:
        # Récupérer les données de validation historique
        from gui.services.historical_validation_service import HistoricalValidationService
        historical_validation = HistoricalValidationService()
        validation_summary = historical_validation.get_validation_summary(ticker, days=1)  # Seulement le 29 septembre
        validation_results = validation_summary.get('validation_results', [])
        
        if validation_results:
            # Trier par timestamp pour avoir les plus récentes en premier
            recent_decisions = sorted(validation_results, key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # Limiter à 10 lignes pour l'affichage
            display_decisions = recent_decisions[:10]
            
            # Créer le tableau avec les colonnes spécifiées
            table_data = []
            for i, decision in enumerate(display_decisions):
                # Gérer l'affichage de l'heure correctement (Heure de Paris)
                timestamp = decision.get('timestamp', 'N/A')
                if hasattr(timestamp, 'strftime'):
                    # Objet datetime - convertir en heure de Paris
                    import pytz
                    try:
                        # Si c'est déjà en UTC, convertir en Paris
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=pytz.UTC)
                        paris_tz = pytz.timezone('Europe/Paris')
                        paris_time = timestamp.astimezone(paris_tz)
                        heure_str = paris_time.strftime('%H:%M')
                        date_str = paris_time.strftime('%d/%m')
                    except:
                        heure_str = timestamp.strftime('%H:%M')
                        date_str = timestamp.strftime('%d/%m')
                elif isinstance(timestamp, str):
                    # Chaîne de caractères - essayer de parser
                    try:
                        if 'T' in timestamp:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        else:
                            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                        
                        # Convertir en heure de Paris
                        import pytz
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=pytz.UTC)
                        paris_tz = pytz.timezone('Europe/Paris')
                        paris_time = dt.astimezone(paris_tz)
                        heure_str = paris_time.strftime('%H:%M')
                        date_str = paris_time.strftime('%d/%m')
                    except Exception as e:
                        if ' ' in timestamp and ':' in timestamp:
                            time_part = timestamp.split(' ')[1]
                            heure_str = time_part[:5]
                            date_str = timestamp.split(' ')[0][5:]
                        else:
                            heure_str = str(timestamp)[:5]
                            date_str = "N/A"
                else:
                    heure_str = str(timestamp)[:5]
                    date_str = "N/A"
                
                # Calculer les prix et le gain
                current_price = decision.get('current_price', 0)
                future_price = decision.get('future_price', 0)
                price_change = decision.get('price_change', 0)
                
                # Calculer le gain en dollars
                gain_dollars = future_price - current_price
                
                # Déterminer le résultat
                decision_type = decision.get('decision', 'N/A')
                if decision_type == 'BUY':
                    is_positive = price_change > 0
                    result_text = "Positif" if is_positive else "Négatif"
                elif decision_type == 'SELL':
                    is_positive = price_change < 0
                    result_text = "Positif" if is_positive else "Négatif"
                else:  # HOLD - Logique corrigée
                    # Pour HOLD : positif seulement si le prix reste vraiment stable (±0.2%)
                    # Toute variation significative est une perte d'opportunité
                    if abs(price_change) <= 0.2:  # Prix vraiment stable → Positif
                        is_positive = True
                    else:  # Toute variation > 0.2% → Négatif (perte d'opportunité)
                        is_positive = False
                    result_text = "Positif" if is_positive else "Négatif"
                
                table_data.append({
                    'N°': i + 1,
                    'Date': date_str,
                    'Heure': heure_str,
                    'Prix -15min': f"${current_price:.2f}",
                    'Prix +15min': f"${future_price:.2f}",
                    'Décision': decision_type,
                    'Résultat': result_text,
                    'Gain': f"${gain_dollars:+.2f}"
                })
            
            if table_data:
                df_table = pd.DataFrame(table_data)
                
                # Afficher le tableau en pleine largeur
                st.dataframe(
                    df_table, 
                    use_container_width=True,
                    height=300,
                    column_config={
                        "N°": st.column_config.NumberColumn("N°", width="small"),
                        "Date": st.column_config.TextColumn("Date", width="small"),
                        "Heure": st.column_config.TextColumn("Heure", width="small"),
                        "Prix -15min": st.column_config.TextColumn("Prix -15min", width="medium"),
                        "Prix +15min": st.column_config.TextColumn("Prix +15min", width="medium"),
                        "Décision": st.column_config.TextColumn("Décision", width="small"),
                        "Résultat": st.column_config.TextColumn("Résultat", width="medium"),
                        "Gain": st.column_config.TextColumn("Gain", width="medium")
                    }
                )
                
                # Statistiques simplifiées
                col1, col2, col3, col4 = st.columns(4)
                    with col1:
                    st.metric("Total Décisions", len(table_data))
                    with col2:
                    positive_count = sum(1 for d in table_data if d['Résultat'] == 'Positif')
                    st.metric("Résultats Positifs", f"{positive_count}/{len(table_data)}")
                    with col3:
                    total_gain = sum(float(d['Gain'].replace('$', '').replace('+', '')) for d in table_data)
                    st.metric("Gain Total", f"${total_gain:+.2f}")
                with col4:
                    # Bouton de téléchargement pour toutes les données
                    if len(validation_results) > 10:
                        # Créer un DataFrame complet pour le téléchargement
                        full_table_data = []
                        for i, decision in enumerate(validation_results):
                            # Même logique que pour l'affichage mais pour toutes les données
                            timestamp = decision.get('timestamp', 'N/A')
                            if hasattr(timestamp, 'strftime'):
                                import pytz
                                try:
                                    if timestamp.tzinfo is None:
                                        timestamp = timestamp.replace(tzinfo=pytz.UTC)
                                    paris_tz = pytz.timezone('Europe/Paris')
                                    paris_time = timestamp.astimezone(paris_tz)
                                    heure_str = paris_time.strftime('%H:%M')
                                    date_str = paris_time.strftime('%d/%m')
                                except:
                                    heure_str = timestamp.strftime('%H:%M')
                                    date_str = timestamp.strftime('%d/%m')
                            elif isinstance(timestamp, str):
                                try:
                                    if 'T' in timestamp:
                                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                    else:
                                        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                                    import pytz
                                    if dt.tzinfo is None:
                                        dt = dt.replace(tzinfo=pytz.UTC)
                                    paris_tz = pytz.timezone('Europe/Paris')
                                    paris_time = dt.astimezone(paris_tz)
                                    heure_str = paris_time.strftime('%H:%M')
                                    date_str = paris_time.strftime('%d/%m')
                                except:
                                    if ' ' in timestamp and ':' in timestamp:
                                        time_part = timestamp.split(' ')[1]
                                        heure_str = time_part[:5]
                                        date_str = timestamp.split(' ')[0][5:]
                                    else:
                                        heure_str = str(timestamp)[:5]
                                        date_str = "N/A"
                            else:
                                heure_str = str(timestamp)[:5]
                                date_str = "N/A"
                            
                            current_price = decision.get('current_price', 0)
                            future_price = decision.get('future_price', 0)
                            price_change = decision.get('price_change', 0)
                            gain_dollars = future_price - current_price
                            
                            decision_type = decision.get('decision', 'N/A')
                            if decision_type == 'BUY':
                                is_positive = price_change > 0
                                result_text = "Positif" if is_positive else "Négatif"
                            elif decision_type == 'SELL':
                                is_positive = price_change < 0
                                result_text = "Positif" if is_positive else "Négatif"
                            else:  # HOLD - Logique corrigée
                                # Pour HOLD : positif seulement si le prix reste vraiment stable (±0.2%)
                                # Toute variation significative est une perte d'opportunité
                                if abs(price_change) <= 0.2:  # Prix vraiment stable → Positif
                                    is_positive = True
                                else:  # Toute variation > 0.2% → Négatif (perte d'opportunité)
                                    is_positive = False
                                result_text = "Positif" if is_positive else "Négatif"
                            
                            full_table_data.append({
                                'N°': i + 1,
                                'Date': date_str,
                                'Heure': heure_str,
                                'Prix -15min': f"${current_price:.2f}",
                                'Prix +15min': f"${future_price:.2f}",
                                'Décision': decision_type,
                                'Résultat': result_text,
                                'Gain': f"${gain_dollars:+.2f}"
                            })
                        
                        # Créer le CSV pour le téléchargement
                        full_df = pd.DataFrame(full_table_data)
                        csv = full_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Télécharger toutes les données",
                            data=csv,
                            file_name=f"decisions_historiques_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            help=f"Télécharge toutes les {len(validation_results)} décisions historiques"
                        )
                    else:
                        st.info("Toutes les données sont affichées")
            else:
                st.info("Aucune décision récente disponible")
        else:
            st.info("Aucune donnée de validation disponible")
    except Exception as e:
        st.warning(f"⚠️ Erreur chargement tableau: {str(e)}")
    
    # ÉTATS DES SERVICES - DÉPLACÉ EN BAS DE PAGE
    st.header("🔧 États des Services")
    
    try:
        # Récupérer les statuts des services
        if services['sentiment_service']:
        articles = services['sentiment_service'].get_news_articles(ticker, 10)
        else:
            articles = []
        article_count = len(articles) if articles else 0
        
        if services['data_service']:
        data = services['data_service'].load_data(ticker)
        else:
            data = pd.DataFrame()
        
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
            if services['data_service']:
            data = services['data_service'].load_data(ticker)
            else:
                data = pd.DataFrame()
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
                if services['prediction_service']:
                prediction = services['prediction_service'].predict(data, horizon=20)
                else:
                    prediction = None
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
            if services['sentiment_service']:
            articles = services['sentiment_service'].get_news_articles(ticker, 5)
            else:
                articles = []
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
        if services['data_service']:
        data = services['data_service'].load_data(ticker)
        else:
            data = pd.DataFrame()
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
        if services['sentiment_service']:
        sentiment_summary = services['sentiment_service'].get_sentiment_summary(ticker)
        else:
            sentiment_summary = {'avg_sentiment': 0.0}
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
        
        if services['data_service']:
        data = services['data_service'].load_data(ticker)
        else:
            data = pd.DataFrame()
        
        if services['prediction_service']:
        prediction = services['prediction_service'].predict(data, horizon=20)
        else:
            prediction = None
        
        if prediction and 'predictions' in prediction and prediction['predictions']:
            # Utiliser la première prédiction
            pred_value = prediction['predictions'][0] if prediction['predictions'] else 0.5
            return max(0.0, min(1.0, pred_value))
        
        return 0.5  # Neutre par défaut
    except Exception:
        return 0.5


def _check_market_status():
    """Vérifie l'état du marché (ouvert/fermé) - Horaires US (EST/EDT)"""
    from datetime import timezone, timedelta
    import pytz
    
    try:
        # Utiliser pytz pour gérer correctement EST/EDT
        us_eastern = pytz.timezone('US/Eastern')
        now_est = datetime.now(us_eastern)
        current_time = now_est.strftime("%H:%M")
        
        # Heures de marché US (9h30 - 16h00 EST/EDT, du lundi au vendredi)
        market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Vérifier si c'est un jour de semaine
        is_weekday = now_est.weekday() < 5  # 0-4 = lundi-vendredi
        
        # Debug info
        print(f"🔍 Debug marché: {now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"🔍 Jour semaine: {is_weekday}, Heure: {now_est.hour}:{now_est.minute}")
        print(f"🔍 Marché ouvert: {market_open} - {market_close}")
        
        if is_weekday and market_open <= now_est <= market_close:
            return {
                "is_open": True,
                "current_time": current_time,
                "timezone": "EST/EDT",
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
                "timezone": "EST/EDT",
                "next_open": next_open.strftime("%A %H:%M"),
                "next_close": "16:00"
            }
    except ImportError:
        # Fallback si pytz n'est pas disponible
        print("⚠️ pytz non disponible, utilisation du fallback")
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


# Fonction show_performance_metrics supprimée selon demande utilisateur