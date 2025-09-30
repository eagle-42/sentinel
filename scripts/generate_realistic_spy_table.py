#!/usr/bin/env python3
"""
Script pour générer un tableau réaliste montrant l'absence d'articles SPY individuels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_data():
    """Charge toutes les données disponibles"""
    news_df = pd.read_parquet('data/realtime/news/all_news.parquet')
    sentiment_df = pd.read_parquet('data/realtime/sentiment/spy_sentiment.parquet')
    
    return news_df, sentiment_df

def create_realistic_table(news_df, sentiment_df):
    """Crée un tableau montrant la réalité des données"""
    
    # Données SPY agrégées
    spy_data = sentiment_df[sentiment_df['ticker'] == 'SPY'].iloc[0]
    
    # Créer le tableau de synthèse
    table_data = {
        'Aspect': [
            'Articles individuels SPY disponibles',
            'Score de sentiment SPY (agrégé)',
            'Nombre d\'articles analysés pour SPY',
            'Niveau de confiance',
            'Timestamp d\'analyse',
            'Articles NVDA disponibles',
            'Problème identifié'
        ],
        'Valeur': [
            '0 (Aucun)',
            f"{spy_data['sentiment_score']:.4f}",
            f"{spy_data['article_count']}",
            f"{spy_data['confidence']:.3f}",
            str(spy_data['ts_utc']),
            f"{len(news_df)}",
            'Tous les articles sont étiquetés NVDA'
        ]
    }
    
    return pd.DataFrame(table_data)

def create_limitation_chart(news_df, sentiment_df):
    """Crée un graphique montrant les limitations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graphique 1: Distribution des articles par ticker
    ticker_counts = news_df['ticker'].value_counts()
    colors = ['red' if ticker == 'NVDA' else 'blue' for ticker in ticker_counts.index]
    
    bars = ax1.bar(ticker_counts.index, ticker_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Ticker')
    ax1.set_ylabel('Nombre d\'Articles')
    ax1.set_title('Distribution des Articles par Ticker')
    ax1.grid(True, alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(height)}', ha='center', va='bottom')
    
    # Graphique 2: Comparaison des scores de sentiment
    spy_score = sentiment_df[sentiment_df['ticker'] == 'SPY']['sentiment_score'].iloc[0]
    nvda_scores = news_df['sentiment_score'].values
    
    ax2.hist(nvda_scores, bins=10, alpha=0.7, color='red', label='NVDA (articles individuels)', edgecolor='black')
    ax2.axvline(spy_score, color='blue', linestyle='--', linewidth=3, label=f'SPY (agrégé): {spy_score:.4f}')
    ax2.set_xlabel('Score de Sentiment')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Comparaison des Scores de Sentiment')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generate_latex_table(df):
    """Génère le code LaTeX pour le tableau"""
    
    latex = """
\\begin{table}[H]
\\centering
\\caption{Analyse de Sentiment FinBERT - Limitation des Données SPY}
\\label{tab:sentiment_limitation_spy}
\\begin{tabular}{p{6cm}|p{8cm}}
\\toprule
\\textbf{Aspect} & \\textbf{Valeur} \\\\
\\midrule
"""
    
    for _, row in df.iterrows():
        aspect = row['Aspect']
        value = row['Valeur'].replace('&', '\\&').replace('%', '\\%').replace('$', '\\$')
        
        latex += f"{aspect} & {value} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    return latex

def main():
    """Fonction principale"""
    print("📊 Génération du tableau réaliste pour SPY")
    print("=" * 60)
    
    # Charger les données
    news_df, sentiment_df = load_data()
    
    print(f"✅ {len(news_df)} articles NVDA chargés")
    print(f"✅ Données de sentiment SPY chargées")
    
    # Créer le tableau réaliste
    print("📋 Création du tableau de limitations...")
    table_df = create_realistic_table(news_df, sentiment_df)
    
    # Créer le graphique
    print("📈 Création du graphique de limitations...")
    fig = create_limitation_chart(news_df, sentiment_df)
    
    # Sauvegarder
    output_dir = Path("data/analysis/spy")
    output_dir.mkdir(exist_ok=True)
    
    # Tableau en CSV
    table_df.to_csv(output_dir / "tableau_limitations_spy.csv", index=False, encoding='utf-8')
    
    # Graphique
    fig.savefig(output_dir / "graphique_limitations_spy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Générer le LaTeX
    print("📝 Génération du code LaTeX...")
    latex_code = generate_latex_table(table_df)
    
    # Sauvegarder LaTeX
    with open(output_dir / "tableau_latex_limitations_spy.tex", 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    # Afficher le tableau
    print("\n" + "=" * 80)
    print("📊 TABLEAU DE LIMITATIONS - DONNÉES SPY")
    print("=" * 80)
    print(table_df.to_string(index=False))
    
    print(f"\n✅ Fichiers sauvegardés dans: {output_dir}")
    print("   - tableau_limitations_spy.csv")
    print("   - graphique_limitations_spy.png")
    print("   - tableau_latex_limitations_spy.tex")
    
    # Afficher le code LaTeX
    print("\n" + "=" * 60)
    print("📝 CODE LATEX POUR LE MÉMOIRE")
    print("=" * 60)
    print(latex_code)

if __name__ == "__main__":
    main()
