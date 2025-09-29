#!/usr/bin/env python3
"""
Script pour valider les décisions de trading par rapport aux prix futurs
Génère un graphique et des métriques de performance
"""

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

def load_decisions(decisions_path):
    """Charge les décisions de trading depuis un fichier JSON"""
    try:
        with open(decisions_path, 'r') as f:
            data = json.load(f)
        
        # Gérer différents formats de données
        if isinstance(data, list):
            decisions = pd.DataFrame(data)
        elif isinstance(data, dict) and 'decisions' in data:
            decisions = pd.DataFrame(data['decisions'])
        else:
            decisions = pd.DataFrame([data])
        
        # Normaliser les colonnes
        decisions['timestamp'] = pd.to_datetime(decisions['timestamp'])
        
        # Filtrer SPY si présent
        if 'ticker' in decisions.columns:
            decisions = decisions[decisions['ticker'] == 'SPY'].copy()
        
        print(f"✅ {len(decisions)} décisions SPY chargées")
        return decisions
        
    except Exception as e:
        print(f"❌ Erreur chargement décisions: {e}")
        sys.exit(1)

def load_prices(prices_path, time_col=None, price_col=None):
    """Charge les données de prix SPY"""
    try:
        if prices_path.endswith('.parquet'):
            prices = pd.read_parquet(prices_path)
        elif prices_path.endswith('.csv'):
            prices = pd.read_csv(prices_path)
        else:
            print(f"❌ Format de fichier non supporté: {prices_path}")
            sys.exit(1)
        
        # Auto-détection des colonnes
        if time_col is None:
            time_candidates = ['timestamp', 'ts_utc', 'date', 'time', 'datetime']
            time_col = next((col for col in time_candidates if col in prices.columns), None)
            if time_col is None:
                print(f"❌ Colonne de temps non trouvée. Colonnes disponibles: {list(prices.columns)}")
                sys.exit(1)
        
        if price_col is None:
            price_candidates = ['close', 'Close', 'CLOSE', 'price', 'Price']
            price_col = next((col for col in price_candidates if col in prices.columns), None)
            if price_col is None:
                print(f"❌ Colonne de prix non trouvée. Colonnes disponibles: {list(prices.columns)}")
                sys.exit(1)
        
        # Normaliser les colonnes
        prices = prices.rename(columns={time_col: 'timestamp', price_col: 'close'})
        prices['timestamp'] = pd.to_datetime(prices['timestamp'])
        
        # S'assurer que les timestamps sont timezone-aware (UTC)
        if prices['timestamp'].dt.tz is None:
            prices['timestamp'] = prices['timestamp'].dt.tz_localize('UTC')
        
        # Filtrer SPY si présent
        if 'ticker' in prices.columns:
            prices = prices[prices['ticker'] == 'SPY'].copy()
        
        # Trier par timestamp
        prices = prices.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ {len(prices)} prix SPY chargés")
        print(f"   Période: {prices['timestamp'].min()} à {prices['timestamp'].max()}")
        return prices
        
    except Exception as e:
        print(f"❌ Erreur chargement prix: {e}")
        sys.exit(1)

def calculate_future_returns(prices, horizon_steps=1):
    """Calcule les rendements futurs pour chaque prix"""
    prices = prices.copy()
    prices['ret_next'] = prices['close'].pct_change(periods=horizon_steps).shift(-horizon_steps)
    return prices

def validate_decisions(decisions, prices, horizon_steps=1):
    """Valide les décisions par rapport aux prix futurs"""
    # Calculer les rendements futurs
    prices_with_returns = calculate_future_returns(prices, horizon_steps)
    
    # Fusionner les décisions avec les prix
    results = []
    
    for _, decision in decisions.iterrows():
        # Trouver le prix le plus proche dans le temps
        time_diff = abs(prices_with_returns['timestamp'] - decision['timestamp'])
        closest_idx = time_diff.idxmin()
        closest_price = prices_with_returns.iloc[closest_idx]
        
        # Vérifier si on a un rendement futur valide
        if pd.isna(closest_price['ret_next']):
            continue
        
        # Déterminer si la décision est correcte
        ret_next = closest_price['ret_next']
        decision_type = decision['decision']
        
        # Règles de validation
        if decision_type == 'BUY':
            is_correct = ret_next > 0
        elif decision_type == 'SELL':
            is_correct = ret_next < 0
        elif decision_type == 'HOLD':
            # Pour HOLD, utiliser la médiane des rendements absolus comme seuil
            median_abs_ret = prices_with_returns['ret_next'].abs().median()
            is_correct = abs(ret_next) < median_abs_ret
        else:
            is_correct = False
        
        results.append({
            'timestamp': decision['timestamp'],
            'decision': decision_type,
            'confidence': decision['confidence'],
            'fused_signal': decision['fused_signal'],
            'close': closest_price['close'],
            'ret_next': ret_next,
            'is_correct': is_correct
        })
    
    return pd.DataFrame(results)

def calculate_kpis(results):
    """Calcule les KPIs par type de décision"""
    kpis = []
    
    for decision_type in results['decision'].unique():
        subset = results[results['decision'] == decision_type]
        
        if len(subset) == 0:
            continue
        
        kpi = {
            'decision': decision_type,
            'n': len(subset),
            'win_rate': subset['is_correct'].mean(),
            'mean_ret_next': subset['ret_next'].mean(),
            'median_ret_next': subset['ret_next'].median(),
            'mean_confidence': subset['confidence'].mean(),
            'mean_fused_signal': subset['fused_signal'].mean()
        }
        kpis.append(kpi)
    
    return pd.DataFrame(kpis)

def create_plot(results, prices, window_days=14):
    """Crée le graphique de validation des décisions"""
    # Filtrer les données récentes
    if window_days > 0:
        cutoff_date = results['timestamp'].max() - timedelta(days=window_days)
        recent_results = results[results['timestamp'] >= cutoff_date].copy()
        recent_prices = prices[prices['timestamp'] >= cutoff_date].copy()
    else:
        recent_results = results.copy()
        recent_prices = prices.copy()
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Tracer la courbe de prix
    ax.plot(recent_prices['timestamp'], recent_prices['close'], 
            'k-', linewidth=2, label='Prix SPY', alpha=0.7)
    
    # Définir les marqueurs et couleurs
    markers = {'BUY': '^', 'SELL': 'v', 'HOLD': 'o'}
    colors = {'correct': 'green', 'incorrect': 'red'}
    
    # Tracer les décisions
    for _, row in recent_results.iterrows():
        marker = markers.get(row['decision'], 'o')
        color = colors['correct'] if row['is_correct'] else colors['incorrect']
        
        ax.scatter(row['timestamp'], row['close'], 
                  marker=marker, s=100, c=color, 
                  edgecolors='black', linewidth=1, alpha=0.8)
    
    # Configuration du graphique
    ax.set_title('Décisions vs Prix Futur - Validation des Prédictions', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Temps', fontsize=12)
    ax.set_ylabel('Prix SPY ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Formatage de l'axe des temps
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Légende personnalisée
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label='Décision Correcte'),
        Patch(facecolor='red', alpha=0.8, label='Décision Incorrecte'),
        plt.Line2D([0], [0], marker='^', color='black', linestyle='None', 
                  markersize=8, label='BUY'),
        plt.Line2D([0], [0], marker='v', color='black', linestyle='None', 
                  markersize=8, label='SELL'),
        plt.Line2D([0], [0], marker='o', color='black', linestyle='None', 
                  markersize=8, label='HOLD')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Valide les décisions de trading par rapport aux prix futurs')
    parser.add_argument('--decisions', default='data/trading/decisions_log/trading_decisions.json',
                       help='Chemin vers le fichier de décisions')
    parser.add_argument('--prices', default='data/realtime/prices/spy_15min.parquet',
                       help='Chemin vers le fichier de prix')
    parser.add_argument('--horizon-steps', type=int, default=1,
                       help='Nombre de pas dans le futur pour calculer le rendement')
    parser.add_argument('--window-days', type=int, default=14,
                       help='Fenêtre de temps à afficher (jours, 0 pour tout)')
    parser.add_argument('--time-col', help='Nom de la colonne de temps dans les prix')
    parser.add_argument('--price-col', help='Nom de la colonne de prix dans les prix')
    parser.add_argument('--output-dir', default='src/notebooks',
                       help='Répertoire de sortie')
    
    args = parser.parse_args()
    
    print("🚀 VALIDATION DES DÉCISIONS DE TRADING")
    print("=" * 50)
    
    # Charger les données
    print("\n📊 Chargement des données...")
    decisions = load_decisions(args.decisions)
    prices = load_prices(args.prices, args.time_col, args.price_col)
    
    # Valider les décisions
    print("\n🔍 Validation des décisions...")
    results = validate_decisions(decisions, prices, args.horizon_steps)
    
    if len(results) == 0:
        print("❌ Aucune décision valide trouvée")
        sys.exit(1)
    
    print(f"✅ {len(results)} décisions validées")
    
    # Calculer les KPIs
    print("\n📈 Calcul des métriques...")
    kpis = calculate_kpis(results)
    
    # Afficher les KPIs
    print("\n📊 MÉTRIQUES DE PERFORMANCE:")
    print("-" * 60)
    for _, kpi in kpis.iterrows():
        print(f"{kpi['decision']:>6}: {kpi['n']:>3} décisions | "
              f"Win Rate: {kpi['win_rate']:>6.1%} | "
              f"Ret Moyen: {kpi['mean_ret_next']:>7.3f} | "
              f"Conf Moy: {kpi['mean_confidence']:>6.3f}")
    
    # Créer le graphique
    print(f"\n🎨 Génération du graphique (fenêtre: {args.window_days} jours)...")
    fig = create_plot(results, prices, args.window_days)
    
    # Sauvegarder les fichiers
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Sauvegarder le graphique
    plot_path = output_dir / 'decisions_vs_future_price.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {plot_path}")
    
    # Sauvegarder le CSV avec les KPIs
    csv_path = output_dir / 'decisions_vs_future_price_summary.csv'
    
    # Ajouter les KPIs au CSV
    with open(csv_path, 'w') as f:
        # Écrire les résultats
        results.to_csv(f, index=False)
        
        # Ajouter une ligne vide
        f.write('\n')
        
        # Écrire les KPIs
        f.write('# KPIs par type de décision\n')
        kpis.to_csv(f, index=False)
    
    print(f"✅ Résultats sauvegardés: {csv_path}")
    
    # Afficher le graphique
    plt.show()
    
    print("\n🎉 VALIDATION TERMINÉE AVEC SUCCÈS!")
    print(f"📁 Fichiers générés dans: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
