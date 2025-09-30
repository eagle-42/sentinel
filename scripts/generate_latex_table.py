#!/usr/bin/env python3
"""
Script pour générer un tableau LaTeX formaté
"""

import pandas as pd
from pathlib import Path

def load_simple_table():
    """Charge le tableau simple"""
    table_path = Path("data/analysis/simple/tableau_sentiment_simple.csv")
    
    if not table_path.exists():
        print("❌ Tableau simple non trouvé")
        return None
    
    df = pd.read_csv(table_path)
    return df

def generate_latex_table(df):
    """Génère le code LaTeX pour le tableau"""
    
    latex = """
\\begin{table}[H]
\\centering
\\caption{Analyse de Sentiment FinBERT - Exemples d'Articles}
\\label{tab:sentiment_analysis}
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
    print("📝 Génération du tableau LaTeX")
    print("=" * 40)
    
    # Charger le tableau
    df = load_simple_table()
    
    if df is None:
        return
    
    print(f"✅ {len(df)} articles chargés")
    
    # Générer le LaTeX
    print("📝 Génération du code LaTeX...")
    latex_code = generate_latex_table(df)
    
    # Sauvegarder
    output_path = Path("data/analysis/simple/tableau_latex.tex")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"✅ Code LaTeX sauvegardé: {output_path}")
    
    # Afficher le code
    print("\n" + "=" * 60)
    print("📝 CODE LATEX POUR LE MÉMOIRE")
    print("=" * 60)
    print(latex_code)

if __name__ == "__main__":
    main()
