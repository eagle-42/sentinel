#!/usr/bin/env python3
"""
Script pour générer un tableau d'analyse de sentiment SPY basé sur les données agrégées
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_spy_sentiment_data():
    """Charge les données de sentiment SPY"""
    sentiment_path = Path("data/realtime/sentiment/spy_sentiment.parquet")
    
    if not sentiment_path.exists():
        print("❌ Fichier de sentiment SPY non trouvé")
        return None
    
    df = pd.read_parquet(sentiment_path)
    spy_data = df[df['ticker'] == 'SPY'].iloc[0]
    
    return spy_data

def create_spy_summary_table(spy_data):
    """Crée un tableau de synthèse pour SPY"""
    
    # Catégoriser le sentiment
    score = spy_data['sentiment_score']
    if score > 0.1:
        category = "🟢 Positif"
        justification = "Sentiment positif détecté par FinBERT"
    elif score < -0.1:
        category = "🔴 Négatif" 
        justification = "Sentiment négatif détecté par FinBERT"
    else:
        category = "🟡 Neutre"
        justification = "Sentiment neutre détecté par FinBERT"
    
    # Créer le tableau
    table_data = {
        'Métrique': [
            'Score de Sentiment',
            'Catégorie',
            'Justification',
            'Niveau de Confiance',
            'Nombre d\'Articles Analysés',
            'Timestamp d\'Analyse'
        ],
        'Valeur': [
            f"{score:.4f}",
            category,
            justification,
            f"{spy_data['confidence']:.3f}",
            f"{spy_data['article_count']}",
            str(spy_data['ts_utc'])
        ]
    }
    
    return pd.DataFrame(table_data)

def create_spy_chart(spy_data):
    """Crée un graphique pour SPY"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graphique en barre du score
    score = spy_data['sentiment_score']
    colors = ['green' if score > 0.1 else 'red' if score < -0.1 else 'orange']
    
    bars = ax1.bar(['SPY'], [score], color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Score de Sentiment')
    ax1.set_title('Score de Sentiment SPY')
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Seuil Positif')
    ax1.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5, label='Seuil Négatif')
    
    # Ajouter la valeur sur la barre
    height = bars[0].get_height()
    ax1.text(bars[0].get_x() + bars[0].get_width()/2., height + 0.02 if height >= 0 else height - 0.02,
            f'{score:.4f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Graphique de confiance
    confidence = spy_data['confidence']
    ax2.bar(['SPY'], [confidence], color='blue', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Niveau de Confiance')
    ax2.set_title('Niveau de Confiance SPY')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Ajouter la valeur sur la barre
    ax2.text(0, confidence + 0.02, f'{confidence:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def generate_latex_table(df):
    """Génère le code LaTeX pour le tableau SPY"""
    
    latex = """
\\begin{table}[H]
\\centering
\\caption{Analyse de Sentiment FinBERT - Synthèse SPY}
\\label{tab:sentiment_analysis_spy}
\\begin{tabular}{p{6cm}|p{8cm}}
\\toprule
\\textbf{Métrique} & \\textbf{Valeur} \\\\
\\midrule
"""
    
    for _, row in df.iterrows():
        metric = row['Métrique']
        value = row['Valeur'].replace('&', '\\&').replace('%', '\\%').replace('$', '\\$')
        
        latex += f"{metric} & {value} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    return latex

def main():
    """Fonction principale"""
    print("📊 Génération du tableau d'analyse de sentiment SPY")
    print("=" * 60)
    
    # Charger les données SPY
    spy_data = load_spy_sentiment_data()
    
    if spy_data is None:
        return
    
    print("✅ Données SPY chargées")
    
    # Créer le tableau de synthèse
    print("📋 Création du tableau de synthèse...")
    table_df = create_spy_summary_table(spy_data)
    
    # Créer le graphique SPY
    print("📈 Création du graphique SPY...")
    fig = create_spy_chart(spy_data)
    
    # Sauvegarder
    output_dir = Path("data/analysis/spy")
    output_dir.mkdir(exist_ok=True)
    
    # Tableau en CSV
    table_df.to_csv(output_dir / "tableau_sentiment_spy.csv", index=False, encoding='utf-8')
    
    # Graphique
    fig.savefig(output_dir / "graphique_global_sentiment_spy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Générer le LaTeX
    print("📝 Génération du code LaTeX...")
    latex_code = generate_latex_table(table_df)
    
    # Sauvegarder LaTeX
    with open(output_dir / "tableau_latex_spy.tex", 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    # Afficher le tableau
    print("\n" + "=" * 80)
    print("📊 TABLEAU D'ANALYSE DE SENTIMENT SPY")
    print("=" * 80)
    print(table_df.to_string(index=False))
    
    print(f"\n✅ Fichiers sauvegardés dans: {output_dir}")
    print("   - tableau_sentiment_spy.csv")
    print("   - graphique_global_sentiment_spy.png")
    print("   - tableau_latex_spy.tex")
    
    # Afficher le code LaTeX
    print("\n" + "=" * 60)
    print("📝 CODE LATEX POUR LE MÉMOIRE")
    print("=" * 60)
    print(latex_code)

if __name__ == "__main__":
    main()
