#!/usr/bin/env python3
"""
Script pour générer un tableau d'analyse de sentiment SPY uniquement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_sentiment_data():
    """Charge les données de sentiment"""
    news_path = Path("data/realtime/news/all_news.parquet")
    
    if not news_path.exists():
        print("❌ Fichier de news non trouvé")
        return None
    
    df = pd.read_parquet(news_path)
    return df

def create_spy_table(df):
    """Crée un tableau simple avec texte et score pour SPY uniquement"""
    # Filtrer seulement SPY
    spy_df = df[df['ticker'] == 'SPY'].copy()
    
    if len(spy_df) == 0:
        print("❌ Aucun article SPY trouvé")
        return None, None
    
    # Sélectionner les colonnes importantes
    table_data = []
    
    for _, row in spy_df.iterrows():
        # Catégoriser le sentiment
        score = row['sentiment_score']
        if score > 0.1:
            category = "🟢 Positif"
            justification = "Sentiment positif détecté par FinBERT"
        elif score < -0.1:
            category = "🔴 Négatif" 
            justification = "Sentiment négatif détecté par FinBERT"
        else:
            category = "🟡 Neutre"
            justification = "Sentiment neutre détecté par FinBERT"
        
        table_data.append({
            'Titre de l\'article': row['title'],
            'Score': f"{score:.4f}",
            'Catégorie': category,
            'Justification': justification,
            'Confiance': f"{row['sentiment_confidence']:.3f}",
            'Ticker': row['ticker']
        })
    
    # Trier par score (plus négatif en premier)
    table_df = pd.DataFrame(table_data)
    table_df = table_df.sort_values('Score')
    
    return table_df, spy_df

def create_spy_chart(spy_df):
    """Crée un graphique global simple des catégories pour SPY"""
    # Compter les catégories
    def categorize_sentiment(score):
        if score > 0.1:
            return "Positif"
        elif score < -0.1:
            return "Négatif"
        else:
            return "Neutre"
    
    spy_df['category'] = spy_df['sentiment_score'].apply(categorize_sentiment)
    category_counts = spy_df['category'].value_counts()
    
    # Créer le graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graphique en barres
    colors = {'Positif': 'green', 'Négatif': 'red', 'Neutre': 'orange'}
    category_colors = [colors[cat] for cat in category_counts.index]
    
    bars = ax1.bar(category_counts.index, category_counts.values, color=category_colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Catégorie de Sentiment')
    ax1.set_ylabel('Nombre d\'Articles')
    ax1.set_title('Répartition des Articles SPY par Sentiment')
    ax1.grid(True, alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(height)}', ha='center', va='bottom')
    
    # Graphique en secteurs
    wedges, texts, autotexts = ax2.pie(category_counts.values, labels=category_counts.index, 
                                      colors=category_colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribution des Sentiments SPY')
    
    plt.tight_layout()
    return fig

def generate_latex_table(df):
    """Génère le code LaTeX pour le tableau SPY"""
    
    latex = """
\\begin{table}[H]
\\centering
\\caption{Analyse de Sentiment FinBERT - Articles SPY}
\\label{tab:sentiment_analysis_spy}
\\begin{tabular}{p{8cm}|c|c|p{4cm}|c}
\\toprule
\\textbf{Titre de l'Article} & \\textbf{Score} & \\textbf{Catégorie} & \\textbf{Justification} & \\textbf{Confiance} \\\\
\\midrule
"""
    
    for _, row in df.iterrows():
        # Échapper les caractères spéciaux LaTeX
        title = row['Titre de l\'article'].replace('&', '\\&').replace('%', '\\%').replace('$', '\\$')
        if len(title) > 80:
            title = title[:77] + "..."
        
        score = row['Score']
        category = row['Catégorie']
        justification = row['Justification']
        confidence = row['Confiance']
        
        latex += f"{title} & {score} & {category} & {justification} & {confidence} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    return latex

def main():
    """Fonction principale"""
    print("📊 Génération du tableau d'analyse de sentiment SPY")
    print("=" * 60)
    
    # Charger les données
    df = load_sentiment_data()
    
    if df is None:
        return
    
    print(f"✅ {len(df)} articles chargés")
    
    # Créer le tableau SPY
    print("📋 Création du tableau SPY...")
    table_df, spy_df = create_spy_table(df)
    
    if table_df is None:
        return
    
    print(f"✅ {len(table_df)} articles SPY trouvés")
    
    # Créer le graphique global SPY
    print("📈 Création du graphique global SPY...")
    fig = create_spy_chart(spy_df)
    
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
