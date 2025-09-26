"""
Page d'analyse - Interface d'analyse des données avec graphiques optimisés
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.services.data_service import DataService
from gui.services.chart_service import ChartService
from gui.services.prediction_service import PredictionService

def show_analysis_page():
    """Affiche la page d'analyse"""
    
    # CSS personnalisé
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        }
        .definition-box {
            background: #e3f2fd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Pas de header ici car il est déjà dans main.py
    
    # Initialisation des services
    @st.cache_resource
    def init_services():
        """Initialise les services avec cache"""
        return {
            'data_service': DataService(),
            'chart_service': ChartService(),
            'prediction_service': PredictionService()
        }
    
    # Chargement des données avec cache
    @st.cache_data
    def load_ticker_data(ticker: str):
        """Charge les données d'un ticker avec cache"""
        services = init_services()
        
        # Essayer d'abord les données 15min récentes
        try:
            from gui.services.data_monitor_service import DataMonitorService
            data_monitor = DataMonitorService()
            data_15min, metadata = data_monitor.get_latest_15min_data(ticker)
            
            if not data_15min.empty and metadata.get('status') == 'ok':
                # Convertir les données 15min au format attendu par data_service
                data_15min = data_15min.rename(columns={
                    'ts_utc': 'DATE',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                # Normaliser les colonnes en majuscules comme data_service
                data_15min.columns = data_15min.columns.str.upper()
                # Conversion des dates en UTC
                data_15min['DATE'] = pd.to_datetime(data_15min['DATE'], utc=True)
                # Tri par date
                data_15min = data_15min.sort_values('DATE').reset_index(drop=True)
                return data_15min
        except Exception as e:
            st.warning(f"⚠️ Impossible de charger les données 15min: {e}")
        
        # Fallback sur les données historiques
        return services['data_service'].load_data(ticker)
    
    # Initialisation des services
    services = init_services()
    data_service = services['data_service']
    chart_service = services['chart_service']
    prediction_service = services['prediction_service']
    
    # Sidebar - Paramètres d'analyse (uniquement pour Analysis)
    with st.sidebar:
        st.header("⚙️ Paramètres d'analyse")
        
        # Sélection du ticker
        ticker = st.selectbox(
            "Action à analyser",
            ["NVDA", "SPY"],
            index=0,
            help="Sélectionnez l'action à analyser"
        )
        
        # Sélection du type d'analyse
        analysis_type = st.selectbox(
            "Type d'analyse",
            ["Prix", "Volume", "Sentiment", "Prédiction"],
            index=0,
            help="Sélectionnez le type d'analyse"
        )
        
        # Sélection de la période
        periods = [
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
        
        period = st.selectbox(
            "Période d'analyse",
            periods,
            index=1,
            help="Sélectionnez la période d'analyse"
        )
        
        # Définitions des indicateurs
        st.header("📚 Définitions des indicateurs")
        
        with st.expander("💰 Prix - Moyenne Mobile", expanded=False):
            st.markdown("""
            **Moyenne Mobile (MA)** : Indicateur technique qui calcule la moyenne des prix de clôture sur une période donnée.
            
            - **MA 20** : Moyenne sur 20 jours (tendance court terme)
            - **MA 50** : Moyenne sur 50 jours (tendance moyen terme)  
            - **MA 100** : Moyenne sur 100 jours (tendance long terme)
            
            **Calcul** : MA(n) = (P1 + P2 + ... + Pn) / n
            """)
        
        with st.expander("📊 Volume - Volatilité", expanded=False):
            st.markdown("""
            **Volatilité du Volume** : Mesure la variabilité du volume de trading.
            
            - Indique l'incertitude des investisseurs
            - Volume élevé = forte activité, Volume faible = faible activité
            
            **Calcul** : Écart-type du volume sur une fenêtre glissante
            """)
        
        with st.expander("🔮 Prédiction - LSTM", expanded=False):
            st.markdown("""
            **LSTM (Long Short-Term Memory)** : Réseau de neurones spécialisé pour les séquences temporelles.
            
            - **Apprentissage** : Le modèle apprend les patterns historiques des prix
            - **Prédiction** : Utilise 15 features techniques pour prédire les prix futurs
            - **Logique** : "Si le passé ressemble à X, alors le futur sera probablement Y"
            
            **Disponible uniquement pour SPY**
            """)
        
        with st.expander("😊 Sentiment - Analyse exploratoire", expanded=False):
            st.markdown("""
            **Sentiment basé sur le prix** : Analyse les patterns de mouvement des prix.
            
            - **Log-rendement** : ln(Prix_t / Prix_t-1) pour normaliser
            - **Z-score** : Standardisation pour comparer les périodes
            - **Compression tanh** : Limite les valeurs entre -1 et +1
            
            **Objectif** : Identifier les patterns de sentiment du marché
            """)
    
    # Chargement des données
    with st.spinner(f"Chargement des données {ticker}..."):
        df = load_ticker_data(ticker)
        
        if df.empty:
            st.error(f"❌ Aucune donnée disponible pour {ticker}")
            return
        
        # Filtrage par période
        filtered_df = data_service.filter_by_period(df, period)
        
        if filtered_df.empty:
            st.error(f"❌ Aucune donnée pour la période {period}")
            return
    
    # Affichage des métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Prix actuel",
            f"${filtered_df['CLOSE'].iloc[-1]:.2f}",
            f"{((filtered_df['CLOSE'].iloc[-1] / filtered_df['CLOSE'].iloc[0]) - 1) * 100:+.2f}%"
        )
    
    with col2:
        st.metric(
            "Volume moyen",
            f"{filtered_df['VOLUME'].mean():,.0f}",
            f"{filtered_df['VOLUME'].iloc[-1]:,.0f}"
        )
    
    with col3:
        volatility = (filtered_df['CLOSE'].std() / filtered_df['CLOSE'].mean()) * 100
        st.metric(
            "Volatilité",
            f"{volatility:.2f}%",
            "Élevée" if volatility > 30 else "Modérée" if volatility > 15 else "Faible"
        )
    
    with col4:
        st.metric(
            "Points de données",
            len(filtered_df),
            f"{period}"
        )
    
    # Création du graphique selon le type d'analyse
    st.header(f"📊 {analysis_type} - {ticker}")
    
    if analysis_type == "Prix":
        # Graphique de prix avec moyennes mobiles
        chart = chart_service.create_price_chart(filtered_df, ticker, period)
        st.plotly_chart(chart, use_container_width=True)
        
        # Métriques de prix
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📈 Métriques de prix")
            current_price = filtered_df['CLOSE'].iloc[-1]
            ma_20 = filtered_df['CLOSE'].rolling(20, min_periods=1).mean().iloc[-1]
            ma_50 = filtered_df['CLOSE'].rolling(50, min_periods=1).mean().iloc[-1] if len(filtered_df) >= 50 else None
            
            st.markdown(f"**💰 Prix actuel** : ${current_price:.2f}")
            st.markdown(f"**📈 MA 20** : ${ma_20:.2f}")
            if ma_50:
                st.markdown(f"**📊 MA 50** : ${ma_50:.2f}")
        
        with col2:
            st.markdown("#### 🎯 Recommandations")
            if current_price > ma_20:
                st.success("📈 **Tendance haussière** - Prix au-dessus de la MA 20")
            else:
                st.warning("📉 **Tendance baissière** - Prix en-dessous de la MA 20")
    
    elif analysis_type == "Volume":
        # Graphique de volume avec volatilité
        chart = chart_service.create_volume_chart(filtered_df, ticker, period)
        st.plotly_chart(chart, use_container_width=True)
        
        # Métriques de volume
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Métriques de volume")
            avg_volume = filtered_df['VOLUME'].mean()
            current_volume = filtered_df['VOLUME'].iloc[-1]
            
            st.markdown(f"**📊 Volume moyen** : {avg_volume:,.0f}")
            st.markdown(f"**📈 Volume actuel** : {current_volume:,.0f}")
        
        with col2:
            st.markdown("#### 🎯 Recommandations")
            if current_volume > avg_volume * 1.5:
                st.success("🔥 **Volume élevé** - Forte activité de trading")
            elif current_volume < avg_volume * 0.5:
                st.warning("😴 **Volume faible** - Faible activité de trading")
            else:
                st.info("📊 **Volume normal** - Activité de trading standard")
    
    elif analysis_type == "Sentiment":
        # Graphique de sentiment
        chart = chart_service.create_sentiment_chart(filtered_df, ticker, period)
        st.plotly_chart(chart, use_container_width=True)
        
        # Calcul du sentiment
        log_returns = np.log(filtered_df['CLOSE'] / filtered_df['CLOSE'].shift(1)).fillna(0)
        window = min(20, len(filtered_df))
        z_score = (log_returns - log_returns.rolling(window, min_periods=1).mean()) / log_returns.rolling(window, min_periods=1).std()
        sentiment = np.tanh(2.0 * z_score.fillna(0)) * 100
        current_sentiment = sentiment.iloc[-1]
        
        # Métriques de sentiment
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 😊 Métriques de sentiment")
            st.markdown(f"**😊 Score sentiment** : {current_sentiment:.1f}%")
            st.markdown(f"**📊 Tendance** : {'Positive' if current_sentiment > 0 else 'Négative'}")
        
        with col2:
            st.markdown("#### 🎯 Recommandations")
            if current_sentiment > 30:
                st.success("🟢 **ACHETER** - Sentiment très positif")
            elif current_sentiment < -30:
                st.error("🔴 **VENDRE** - Sentiment très négatif")
            else:
                st.info("🟡 **HOLD** - Sentiment neutre")
    
    elif analysis_type == "Prédiction":
        if ticker == "SPY":
            # Graphique de prédiction LSTM
            with st.spinner("Génération des prédictions LSTM..."):
                prediction_data = prediction_service.predict(filtered_df, horizon=20)
                chart = chart_service.create_prediction_chart(filtered_df, prediction_data, ticker, period)
                st.plotly_chart(chart, use_container_width=True)
            
            # Métriques de prédiction
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔮 Métriques de prédiction")
                current_price = filtered_df['CLOSE'].iloc[-1]
                if prediction_data.get('predictions'):
                    future_price = prediction_data['predictions'][-1]
                    change_pct = ((future_price / current_price) - 1) * 100
                    st.markdown(f"**💰 Prix actuel** : ${current_price:.2f}")
                    st.markdown(f"**🔮 Prix prédit** : ${future_price:.2f}")
                    st.markdown(f"**📈 Variation prédite** : {change_pct:+.2f}%")
                else:
                    st.markdown(f"**💰 Prix actuel** : ${current_price:.2f}")
                    st.error("**❌ Prédiction non disponible**")
            
            with col2:
                st.markdown("#### 🎯 Recommandations")
                if prediction_data.get('predictions'):
                    future_price = prediction_data['predictions'][-1]
                    change_pct = ((future_price / current_price) - 1) * 100
                    if change_pct > 5:
                        st.success("🚀 **FORTE HAUSSE PRÉVUE** - Considérer un achat")
                    elif change_pct < -5:
                        st.error("📉 **FORTE BAISSE PRÉVUE** - Considérer une vente")
                    else:
                        st.info("📊 **ÉVOLUTION MODÉRÉE** - Maintenir la position")
                else:
                    st.warning("❌ Aucune prédiction disponible")
        else:
            st.error("❌ Prédictions LSTM disponibles uniquement pour SPY")
            st.info("💡 Utilisez SPY pour les prédictions LSTM")
    
    # Informations sur les données
    st.markdown("---")
    st.markdown("#### 📋 Informations sur les données")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**📅 Période** : {period}")
        st.markdown(f"**📊 Points de données** : {len(filtered_df)}")
    with col2:
        st.markdown(f"**📈 Date de début** : {filtered_df['DATE'].min().strftime('%Y-%m-%d')}")
        st.markdown(f"**📉 Date de fin** : {filtered_df['DATE'].max().strftime('%Y-%m-%d')}")
    with col3:
        st.markdown(f"**💰 Prix min** : ${filtered_df['CLOSE'].min():.2f}")
        st.markdown(f"**💰 Prix max** : ${filtered_df['CLOSE'].max():.2f}")
