#!/usr/bin/env python3
"""
🧹 Nettoyage imports inutilisés - Phase 1
ATTENTION: Script automatique, vérifier les changements avant commit !
"""

import re
from pathlib import Path
from typing import List, Set

# Imports à NE PAS supprimer même si non utilisés dans certains cas
SAFE_IMPORTS = {
    "pytest",
    "unittest",
    "Mock",
    "MagicMock",  # Tests
    "logger",
    "loguru",  # Logging (peut être dans config)
    "TYPE_CHECKING",  # Type hints
}


def get_imports(content: str) -> List[tuple]:
    """Extrait tous les imports d'un fichier"""
    imports = []
    lines = content.split("\n")

    for i, line in enumerate(lines, 1):
        line = line.strip()

        # Import simple: import module
        if line.startswith("import ") and " as " not in line:
            match = re.match(r"import\s+([a-zA-Z_][a-zA-Z0-9_.]*)", line)
            if match:
                module = match.group(1).split(".")[0]
                imports.append((i, line, module, "simple"))

        # Import from: from module import X
        elif line.startswith("from ") and " import " in line:
            match = re.match(r"from\s+[\w.]+\s+import\s+(.+)", line)
            if match:
                imported = match.group(1).split(",")
                for imp in imported:
                    name = imp.strip().split(" as ")[0].strip()
                    if name not in SAFE_IMPORTS:
                        imports.append((i, line, name, "from"))

    return imports


def is_import_used(content: str, import_name: str, import_line: str) -> bool:
    """Vérifie si un import est utilisé (heuristique)"""
    # Compter occurrences hors ligne d'import
    lines = content.split("\n")
    count = 0

    for line in lines:
        if line.strip() != import_line.strip():  # Pas la ligne d'import elle-même
            # Recherche le nom utilisé
            if re.search(rf"\b{re.escape(import_name)}\b", line):
                count += 1

    return count > 0


def clean_file(file_path: Path, dry_run: bool = True) -> dict:
    """Nettoie les imports inutilisés d'un fichier"""
    with open(file_path) as f:
        content = f.read()

    imports = get_imports(content)
    to_remove = []

    for line_num, line, name, import_type in imports:
        if name in SAFE_IMPORTS:
            continue

        if not is_import_used(content, name, line):
            to_remove.append((line_num, line, name))

    if to_remove and not dry_run:
        lines = content.split("\n")
        # Supprimer lignes (en commençant par la fin)
        for line_num, line, name in reversed(to_remove):
            lines[line_num - 1] = f"# REMOVED: {line}  # Unused import"

        new_content = "\n".join(lines)
        with open(file_path, "w") as f:
            f.write(new_content)

    return {"file": str(file_path), "removed": len(to_remove), "imports": to_remove}


def main(dry_run: bool = True):
    """Lance le nettoyage sur tous les fichiers Python"""
    print("=" * 80)
    print("🧹 NETTOYAGE IMPORTS INUTILISÉS")
    print("=" * 80)
    print(f"\nMode: {'DRY RUN (simulation)' if dry_run else 'REAL (modifications)'}")

    src_dir = Path("src")
    py_files = list(src_dir.rglob("*.py"))

    results = []
    total_removed = 0

    for file in py_files:
        result = clean_file(file, dry_run=dry_run)
        if result["removed"] > 0:
            results.append(result)
            total_removed += result["removed"]

    print(f"\n📊 RÉSULTATS:")
    print(f"   Fichiers analysés: {len(py_files)}")
    print(f"   Fichiers avec imports inutilisés: {len(results)}")
    print(f"   Total imports à supprimer: {total_removed}")

    if results:
        print(f"\n⚠️ TOP FICHIERS:")
        for r in sorted(results, key=lambda x: x["removed"], reverse=True)[:10]:
            print(f"   - {r['file']}: {r['removed']} imports")

    if dry_run:
        print(f"\n💡 Pour appliquer les changements:")
        print(f"   python scripts/clean_unused_imports.py --apply")
    else:
        print(f"\n✅ Changements appliqués !")
        print(f"⚠️ VÉRIFIER avec: git diff src/")

    print("=" * 80)


if __name__ == "__main__":
    import sys

    apply = "--apply" in sys.argv
    main(dry_run=not apply)
