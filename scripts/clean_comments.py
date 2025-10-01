#!/usr/bin/env python3
"""
Script de nettoyage des commentaires inutiles
Supprime les commentaires évidents et redondants
"""

import re
from pathlib import Path
from typing import List

# Patterns de commentaires à supprimer (commentaires évidents)
USELESS_PATTERNS = [
    r"^\s*# Imports?\s*$",
    r"^\s*# Configuration\s*$",
    r"^\s*# Initialisation\s*$",
    r"^\s*# Variables?\s*$",
    r"^\s*# Functions?\s*$",
    r"^\s*# Main\s*$",
    r"^\s*# End\s*$",
    r"^\s*# --+\s*$",
    r"^\s*# ===+\s*$",
    r"^\s*#\s*$",  # Lignes vides avec juste #
]

def should_remove_comment(line: str) -> bool:
    """Vérifie si un commentaire doit être supprimé"""
    for pattern in USELESS_PATTERNS:
        if re.match(pattern, line):
            return True
    return False

def clean_file(file_path: Path) -> int:
    """Nettoie un fichier Python de ses commentaires inutiles"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        cleaned_lines = []
        removed_count = 0
        
        for line in lines:
            if should_remove_comment(line):
                removed_count += 1
                continue
            cleaned_lines.append(line)
        
        if removed_count > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)
            print(f"✅ {file_path}: {removed_count} commentaires supprimés")
        
        return removed_count
        
    except Exception as e:
        print(f"❌ Erreur {file_path}: {e}")
        return 0

def main():
    """Nettoie tous les fichiers Python du projet"""
    project_root = Path(__file__).parent.parent
    
    # Dossiers à traiter
    directories = [
        project_root / "src",
        project_root / "scripts",
        project_root / "flows",
    ]
    
    total_removed = 0
    files_processed = 0
    
    print("🧹 Nettoyage des commentaires inutiles...")
    print("=" * 60)
    
    for directory in directories:
        if not directory.exists():
            continue
            
        for py_file in directory.rglob("*.py"):
            # Ignorer les fichiers dans __pycache__ et .venv
            if "__pycache__" in str(py_file) or ".venv" in str(py_file):
                continue
            
            removed = clean_file(py_file)
            if removed > 0:
                total_removed += removed
                files_processed += 1
    
    print("=" * 60)
    print(f"✅ Nettoyage terminé:")
    print(f"   Fichiers modifiés: {files_processed}")
    print(f"   Commentaires supprimés: {total_removed}")

if __name__ == "__main__":
    main()
