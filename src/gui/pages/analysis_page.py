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

# Import du gestionnaire de services centralisé
from gui.services.service_manager import service_manager
from gui.constants import normalize_columns

def show_analysis_page():
    """Affiche la page d'analyse"""
    
    # Vérifier l'état des services
    services_running = st.session_state.get('services_running', True)
    
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
    
    # Initialisation des services SANS cache pour la réactivité
    def init_services():
        """Initialise les services sans cache pour la réactivité"""
        if not services_running:
            return {
                'data_service': None,
                'chart_service': None,
                'prediction_service': None
            }
        # Utiliser le gestionnaire de services centralisé
        all_services = service_manager.get_services()
        return {
            'data_service': all_services.get('data_service'),
            'chart_service': all_services.get('chart_service'),
            'prediction_service': all_services.get('prediction_service')
        }
    
    # Chargement des données SANS cache pour permettre le filtrage réactif
    def load_ticker_data(ticker: str):
        """Charge les données d'un ticker sans cache pour réactivité"""
        services = init_services()
        
        # Vérifier si les services sont disponibles
        if not services['data_service']:
            return pd.DataFrame()  # Retourner un DataFrame vide si services arrêtés
        
        # Pour l'analyse, utiliser TOUJOURS les données historiques qui ont plus de données
        # Les données 15min sont limitées à quelques jours seulement
        historical_data = services['data_service'].load_data(ticker)
        
        # Normaliser les colonnes en minuscules
        if not historical_data.empty:
            historical_data = normalize_columns(historical_data)
            # Conversion des dates
            historical_data['date'] = pd.to_datetime(historical_data['date'], utc=True)
            # Tri par date
            historical_data = historical_data.sort_values('date').reset_index(drop=True)
        
        return historical_data
    
    # Fonction pour filtrer les données par période (sans cache pour réactivité)
    def filter_data_by_period(data: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filtre les données par période de manière réactive"""
        if data.empty:
            return data
        
        # Utiliser la DERNIÈRE DATE DISPONIBLE dans les données comme référence
        # Pas la date actuelle qui peut être après les données disponibles
        last_date = data['date'].max()
        
        if period == "7 derniers jours":
            start_date = last_date - pd.Timedelta(days=7)
        elif period == "1 mois":
            start_date = last_date - pd.Timedelta(days=30)
        elif period == "1 an":
            start_date = last_date - pd.Timedelta(days=365)
        else:  # Total
            return data
        
        # Filtrer les données
        if 'date' in data.columns:
            return data[data['date'] >= start_date].copy()
        else:
            return data
    
    # Initialisation des services
    services = init_services()
    data_service = services['data_service']
    chart_service = services['chart_service']
    prediction_service = services['prediction_service']
    
    # Afficher un message si les services sont arrêtés
    if not services_running:
        st.warning("🔧 **Services arrêtés** - Seules les données historiques sont disponibles pour l'analyse.")
    
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
            "1 an"
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
    
    # Filtrage par période - Utilisation de la fonction réactive
    with st.spinner(f"Filtrage des données pour {period}..."):
        filtered_df = filter_data_by_period(df, period)
        
        if filtered_df.empty:
            st.error(f"❌ Aucune donnée pour la période {period}")
            return
    
    # Indicateur de mise à jour avec debug
    st.success(f"✅ Données historiques chargées : {len(filtered_df)} points pour {period}")
    
    # Affichage des métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Prix actuel",
            f"${filtered_df['close'].iloc[-1]:.2f}",
            f"{((filtered_df['close'].iloc[-1] / filtered_df['close'].iloc[0]) - 1) * 100:+.2f}%"
        )
    
    with col2:
        st.metric(
            "Volume moyen",
            f"{filtered_df['volume'].mean():,.0f}",
            f"{filtered_df['volume'].iloc[-1]:,.0f}"
        )
    
    with col3:
        volatility = (filtered_df['close'].std() / filtered_df['close'].mean()) * 100
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
    
    # Utiliser la période comme clé pour forcer la mise à jour
    chart_key = f"{ticker}_{analysis_type}_{period}_{len(filtered_df)}"
    
    if analysis_type == "Prix":
        # Graphique de prix avec moyennes mobiles
        if chart_service:
            chart = chart_service.create_price_chart(filtered_df, ticker, period)
            st.plotly_chart(chart, use_container_width=True, key=f"price_chart_{chart_key}")
        else:
            st.info("🔧 Service de graphiques non disponible - Services arrêtés")
        
        # Métriques de prix
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📈 Métriques de prix")
            current_price = filtered_df['close'].iloc[-1]
            ma_20 = filtered_df['close'].rolling(20, min_periods=1).mean().iloc[-1]
            ma_50 = filtered_df['close'].rolling(50, min_periods=1).mean().iloc[-1] if len(filtered_df) >= 50 else None
            
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
        if chart_service:
            chart = chart_service.create_volume_chart(filtered_df, ticker, period)
            st.plotly_chart(chart, use_container_width=True, key=f"volume_chart_{chart_key}")
        else:
            st.info("🔧 Service de graphiques non disponible - Services arrêtés")
        
        # Métriques de volume
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Métriques de volume")
            avg_volume = filtered_df['volume'].mean()
            current_volume = filtered_df['volume'].iloc[-1]
            
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
        if chart_service:
            chart = chart_service.create_sentiment_chart(filtered_df, ticker, period)
            st.plotly_chart(chart, use_container_width=True, key=f"sentiment_chart_{chart_key}")
        else:
            st.info("🔧 Service de graphiques non disponible - Services arrêtés")
        
        # Calcul du sentiment
        log_returns = np.log(filtered_df['close'] / filtered_df['close'].shift(1)).fillna(0)
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
        if not prediction_service or not chart_service:
            st.info("🔧 Services de prédiction non disponibles - Services arrêtés")
        elif ticker == "SPY":
            # Graphique de prédiction LSTM
            with st.spinner("Génération des prédictions LSTM..."):
                prediction_data = prediction_service.predict(filtered_df, horizon=20)
                chart = chart_service.create_prediction_chart(filtered_df, prediction_data, ticker, period)
                st.plotly_chart(chart, use_container_width=True, key=f"prediction_chart_{chart_key}")
            
            # Métriques de prédiction
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔮 Métriques de prédiction")
                current_price = filtered_df['close'].iloc[-1]
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
    
