"""
Page de logs - Affichage des logs de l'application + Monitoring
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path
import sys

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.services.monitoring_service import MonitoringService

def show_logs_page():
    """Affiche la page de logs avec monitoring"""
    
    # Initialiser le service de monitoring
    if 'monitoring_service' not in st.session_state:
        st.session_state.monitoring_service = MonitoringService()
    
    monitoring_service = st.session_state.monitoring_service
    
    # CSS personnalisé
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #6c757d 0%, #495057 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .log-entry {
            background: #f8f9fa;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 4px;
            border-left: 4px solid #6c757d;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
        }
        .log-info { border-left-color: #17a2b8; }
        .log-success { border-left-color: #28a745; }
        .log-warning { border-left-color: #ffc107; }
        .log-error { border-left-color: #dc3545; }
        .log-debug { border-left-color: #6c757d; }
        .monitoring-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            border-left: 5px solid #6c757d;
            margin: 1rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
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
    </style>
    """, unsafe_allow_html=True)
    
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>📋 Logs & Monitoring</h1>
        <p>Surveillance et diagnostic du système Sentinel</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section Monitoring & Contrôle (en premier)
    st.header("📊 Monitoring & Contrôle")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Statut des services
        try:
            system_status = monitoring_service.get_system_status()
            
            st.markdown("""
            <div class="monitoring-card">
                <h4>🔧 Statut des Services</h4>
            """, unsafe_allow_html=True)
            
            for service, status in system_status['services'].items():
                status_class = f"status-{status}" if status in ['online', 'offline'] else "status-warning"
                st.markdown(f"""
                <p><span class="status-indicator {status_class}"></span>{service.title()}: <strong>{status}</strong></p>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <p><strong>Statut global:</strong> {system_status['overall_status']}</p>
                <p><strong>Services en ligne:</strong> {system_status['online_count']}/{system_status['total_count']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f"""
            <div class="monitoring-card">
                <h4>❌ Erreur Statut Services</h4>
                <p>Impossible de vérifier l'état: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Mini-log des actions
        try:
            decisions = monitoring_service.get_recent_trading_decisions(5)
            
            st.markdown("""
            <div class="monitoring-card">
                <h4>📝 Actions Récentes</h4>
            """, unsafe_allow_html=True)
            
            for decision in decisions:
                color = "green" if decision['action'] == "ACHETER" else "red" if decision['action'] == "VENDRE" else "orange"
                st.markdown(f"""
                <p style="color: {color};">
                    <strong>{decision['action']}</strong> {decision['ticker']} @ ${decision['price']}<br>
                    <small>{decision['timestamp'].strftime('%H:%M')} - {decision['reason']}</small>
                </p>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f"""
            <div class="monitoring-card">
                <h4>❌ Erreur Actions Récentes</h4>
                <p>Impossible de charger les actions: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Alertes
        try:
            alerts = monitoring_service.get_alerts()
            
            st.markdown("""
            <div class="monitoring-card">
                <h4>🚨 Alertes</h4>
            """, unsafe_allow_html=True)
            
            if alerts:
                for alert in alerts[-3:]:  # 3 dernières alertes
                    severity_color = "red" if alert['severity'] == "critical" else "orange" if alert['severity'] == "warning" else "blue"
                    st.markdown(f"""
                    <p style="color: {severity_color};">
                        <strong>{alert['type'].title()}</strong><br>
                        <small>{alert['message']}</small>
                    </p>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: green;'>✅ Aucune alerte</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f"""
            <div class="monitoring-card">
                <h4>❌ Erreur Alertes</h4>
                <p>Impossible de charger les alertes: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Métriques de performance
    st.header("📈 Métriques de Performance")
    
    try:
        metrics = monitoring_service.get_performance_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("CPU", f"{metrics['cpu_percent']:.1f}%")
        with col2:
            st.metric("Mémoire", f"{metrics['memory_percent']:.1f}%")
        with col3:
            st.metric("Disque", f"{metrics['disk_percent']:.1f}%")
        with col4:
            st.metric("Dernière MAJ", metrics['timestamp'].strftime('%H:%M'))
    
    except Exception as e:
        st.error(f"Impossible de charger les métriques: {e}")
    
    st.markdown("---")
    
    # Filtres de logs
    st.header("🔍 Filtres de Logs")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        log_level = st.selectbox(
            "Niveau de log",
            ["Tous", "INFO", "SUCCESS", "WARNING", "ERROR", "DEBUG"],
            index=0
        )
    
    with col2:
        time_range = st.selectbox(
            "Période",
            ["Dernière heure", "Dernières 6 heures", "Dernier jour", "Dernière semaine"],
            index=1
        )
    
    with col3:
        component = st.selectbox(
            "Composant",
            ["Tous", "DataService", "ChartService", "PredictionService", "Main", "GUI"],
            index=0
        )
    
    with col4:
        if st.button("🔄 Actualiser"):
            st.rerun()
    
    # Simulation de logs (en attendant la vraie implémentation)
    def generate_sample_logs():
        """Génère des logs d'exemple"""
        logs = []
        
        # Logs récents
        now = datetime.now()
        for i in range(50):
            timestamp = now - timedelta(minutes=i*2)
            
            # Niveaux de log variés
            levels = ["INFO", "SUCCESS", "WARNING", "ERROR", "DEBUG"]
            level = levels[i % len(levels)]
            
            # Composants variés
            components = ["DataService", "ChartService", "PredictionService", "Main", "GUI"]
            comp = components[i % len(components)]
            
            # Messages variés
            messages = [
                f"Données chargées pour NVDA: {6708 - i*10} lignes",
                f"Graphique {comp.lower()} créé avec succès",
                f"Prédiction LSTM générée: {20 - i//3} prédictions futures",
                f"Erreur lors du chargement du modèle LSTM",
                f"Cache mis à jour pour {comp.lower()}",
                f"Interface utilisateur mise à jour",
                f"Filtrage des données: {30 - i//2} lignes",
                f"Calcul des moyennes mobiles terminé",
                f"Analyse de sentiment en cours...",
                f"Validation des données réussie"
            ]
            
            message = messages[i % len(messages)]
            
            logs.append({
                'timestamp': timestamp.strftime('%H:%M:%S'),
                'level': level,
                'component': comp,
                'message': message
            })
        
        return logs
    
    # Affichage des logs
    st.header("📊 Journal des Logs")
    
    # Générer les logs d'exemple
    logs = generate_sample_logs()
    
    # Filtrer les logs
    filtered_logs = logs
    
    if log_level != "Tous":
        filtered_logs = [log for log in filtered_logs if log['level'] == log_level]
    
    if component != "Tous":
        filtered_logs = [log for log in filtered_logs if log['component'] == component]
    
    # Afficher les logs
    if filtered_logs:
        st.markdown(f"**{len(filtered_logs)} entrées trouvées**")
        
        # Pagination
        logs_per_page = 20
        total_pages = (len(filtered_logs) - 1) // logs_per_page + 1
        
        if total_pages > 1:
            page = st.selectbox("Page", range(1, total_pages + 1), index=0)
            start_idx = (page - 1) * logs_per_page
            end_idx = start_idx + logs_per_page
            page_logs = filtered_logs[start_idx:end_idx]
        else:
            page_logs = filtered_logs
        
        # Afficher les logs de la page
        for log in page_logs:
            level_class = f"log-{log['level'].lower()}"
            st.markdown(f"""
            <div class="log-entry {level_class}">
                <strong>{log['timestamp']}</strong> | 
                <strong>{log['level']}</strong> | 
                <strong>{log['component']}</strong> | 
                {log['message']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Aucun log trouvé avec les filtres sélectionnés")
    
    # Statistiques des logs
    st.header("📈 Statistiques des Logs")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Compter les logs par niveau
    level_counts = {}
    for log in logs:
        level = log['level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    with col1:
        st.metric("INFO", level_counts.get('INFO', 0))
    with col2:
        st.metric("SUCCESS", level_counts.get('SUCCESS', 0))
    with col3:
        st.metric("WARNING", level_counts.get('WARNING', 0))
    with col4:
        st.metric("ERROR", level_counts.get('ERROR', 0))
    
    # Actions sur les logs
    st.header("🎮 Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Exporter les Logs"):
            # Créer un DataFrame des logs
            df_logs = pd.DataFrame(logs)
            csv = df_logs.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv,
                file_name=f"sentinel_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🗑️ Nettoyer les Logs"):
            st.warning("⚠️ Cette action supprimera tous les logs. Êtes-vous sûr ?")
            if st.button("✅ Confirmer la suppression"):
                st.success("✅ Logs nettoyés")
    
    with col3:
        if st.button("📊 Graphique des Logs"):
            # Créer un graphique des logs par heure
            df_logs = pd.DataFrame(logs)
            df_logs['hour'] = pd.to_datetime(df_logs['timestamp'], format='%H:%M:%S').dt.hour
            
            hourly_counts = df_logs.groupby(['hour', 'level']).size().unstack(fill_value=0)
            
            st.bar_chart(hourly_counts)
    
    
    # Informations système
    st.markdown("---")
    st.markdown("#### 📋 Informations Système")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**🕒 Dernière actualisation** : {datetime.now().strftime('%H:%M:%S')}")
        st.markdown(f"**📊 Total des logs** : {len(logs)}")
    with col2:
        st.markdown(f"**💾 Taille du fichier** : ~2.3 MB")
        st.markdown(f"**🔄 Rotation** : Quotidienne")
    with col3:
        st.markdown(f"**📁 Emplacement** : data/logs/")
        st.markdown(f"**🔧 Format** : Loguru")
