#!/usr/bin/env python3
"""
🔍 AUDIT QUALITÉ CODE - Sentinel2
Analyse complète: erreurs, duplications, complexité, bonnes pratiques
"""

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class CodeAuditor:
    def __init__(self, root_dir: str = "src"):
        self.root = Path(root_dir)
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)

    def audit_all(self):
        """Lance tous les audits"""
        print("=" * 80)
        print("🔍 AUDIT QUALITÉ CODE - SENTINEL2")
        print("=" * 80)

        py_files = list(self.root.rglob("*.py"))
        print(f"\n📊 Fichiers Python: {len(py_files)}")

        # 1. Syntaxe
        print("\n1️⃣  VÉRIFICATION SYNTAXE...")
        self.check_syntax(py_files)

        # 2. Imports inutilisés
        print("\n2️⃣  IMPORTS INUTILISÉS...")
        self.check_unused_imports(py_files)

        # 3. Code dupliqué
        print("\n3️⃣  CODE DUPLIQUÉ...")
        self.check_duplicates(py_files)

        # 4. Complexité
        print("\n4️⃣  COMPLEXITÉ FONCTIONS...")
        self.check_complexity(py_files)

        # 5. Bonnes pratiques
        print("\n5️⃣  BONNES PRATIQUES...")
        self.check_best_practices(py_files)

        # 6. Tests coverage
        print("\n6️⃣  FICHIERS SANS TESTS...")
        self.check_test_coverage(py_files)

        # 7. Documentation
        print("\n7️⃣  DOCUMENTATION...")
        self.check_documentation(py_files)

        # Rapport final
        self.print_report()

    def check_syntax(self, files: List[Path]):
        """Vérifie syntaxe Python"""
        for file in files:
            try:
                with open(file) as f:
                    ast.parse(f.read())
                self.stats["files_ok"] += 1
            except SyntaxError as e:
                self.errors.append(f"❌ SYNTAX: {file}:{e.lineno} - {e.msg}")
                self.stats["syntax_errors"] += 1
            except Exception as e:
                self.errors.append(f"❌ ERROR: {file} - {e}")

        print(f"   ✅ {self.stats['files_ok']} fichiers OK")
        if self.stats["syntax_errors"]:
            print(f"   ❌ {self.stats['syntax_errors']} erreurs de syntaxe")

    def check_unused_imports(self, files: List[Path]):
        """Détecte imports potentiellement inutilisés"""
        unused_count = 0
        for file in files:
            try:
                with open(file) as f:
                    content = f.read()

                tree = ast.parse(content)
                imports = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            imports.append(alias.name)

                # Vérifier usage (heuristique simple)
                for imp in imports:
                    if imp and content.count(imp) == 1:  # Seulement dans import
                        self.warnings.append(f"⚠️ UNUSED?: {file.relative_to(self.root)} - {imp}")
                        unused_count += 1

            except Exception:
                pass

        print(f"   ⚠️ {unused_count} imports potentiellement inutilisés")
        self.stats["unused_imports"] = unused_count

    def check_duplicates(self, files: List[Path]):
        """Détecte code dupliqué"""
        function_signatures = defaultdict(list)

        for file in files:
            try:
                with open(file) as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Signature simple: nom + nb params
                        sig = f"{node.name}_{len(node.args.args)}"
                        function_signatures[sig].append(str(file.relative_to(self.root)))

            except Exception:
                pass

        # Trouver duplications
        duplicates = {sig: files for sig, files in function_signatures.items() if len(files) > 1}

        if duplicates:
            count = 0
            for sig, files_list in list(duplicates.items())[:10]:
                if len(set(files_list)) > 1:  # Vraiment dans fichiers différents
                    self.warnings.append(f"⚠️ DUPLICATE: {sig} dans {len(set(files_list))} fichiers")
                    count += 1
            print(f"   ⚠️ {count} fonctions potentiellement dupliquées")
        else:
            print(f"   ✅ Pas de duplication détectée")

        self.stats["duplicates"] = len(duplicates)

    def check_complexity(self, files: List[Path]):
        """Mesure complexité cyclomatique (simple)"""
        complex_functions = []

        for file in files:
            try:
                with open(file) as f:
                    lines = f.readlines()

                for i, line in enumerate(lines):
                    if "def " in line:
                        # Compter branches (if, for, while, try, except)
                        func_lines = []
                        indent = len(line) - len(line.lstrip())

                        for j in range(i, min(i + 100, len(lines))):
                            if lines[j].strip() and not lines[j].strip().startswith("#"):
                                line_indent = len(lines[j]) - len(lines[j].lstrip())
                                if line_indent <= indent and j > i:
                                    break
                                func_lines.append(lines[j])

                        complexity = sum(
                            1 for l in func_lines if any(k in l for k in ["if ", "for ", "while ", "except ", "elif "])
                        )

                        if complexity > 10:
                            func_name = line.split("def ")[1].split("(")[0]
                            complex_functions.append((file.relative_to(self.root), func_name, complexity))

            except Exception:
                pass

        if complex_functions:
            for file, func, complexity in complex_functions[:10]:
                self.warnings.append(f"⚠️ COMPLEX: {file}::{func} - complexité {complexity}")
            print(f"   ⚠️ {len(complex_functions)} fonctions très complexes (>10 branches)")
        else:
            print(f"   ✅ Complexité acceptable")

        self.stats["complex_functions"] = len(complex_functions)

    def check_best_practices(self, files: List[Path]):
        """Vérifie bonnes pratiques Python"""
        issues = []

        for file in files:
            try:
                with open(file) as f:
                    content = f.read()
                    lines = content.split("\n")

                # Vérifier print() (devrait utiliser logger)
                if "print(" in content and "test" not in str(file) and "notebook" not in str(file):
                    print_count = content.count("print(")
                    issues.append(f"⚠️ PRINT: {file.relative_to(self.root)} - {print_count} print() (utiliser logger)")

                # Vérifier lignes trop longues
                for i, line in enumerate(lines, 1):
                    if len(line) > 120:
                        issues.append(f"⚠️ LONG LINE: {file.relative_to(self.root)}:{i} - {len(line)} chars")
                        break  # Une seule warning par fichier

                # Vérifier exception bare
                if "except:" in content:
                    issues.append(f"⚠️ BARE EXCEPT: {file.relative_to(self.root)} - utiliser Exception explicite")

            except Exception:
                pass

        print(f"   ⚠️ {len(issues)} problèmes de bonnes pratiques")
        self.warnings.extend(issues[:20])
        self.stats["best_practice_issues"] = len(issues)

    def check_test_coverage(self, files: List[Path]):
        """Vérifie fichiers sans tests"""
        src_files = [f for f in files if "test" not in str(f) and "notebook" not in str(f)]
        test_files = [f for f in files if "test" in str(f)]

        # Fichiers src sans test correspondant
        untested = []
        for src_file in src_files:
            test_name = f"test_{src_file.stem}.py"
            if not any(test_name in str(tf) for tf in test_files):
                untested.append(src_file.relative_to(self.root))

        print(f"   📊 {len(test_files)} fichiers de tests")
        print(f"   ⚠️ {len(untested)} fichiers sans tests")

        for file in untested[:10]:
            self.warnings.append(f"⚠️ NO TEST: {file}")

        self.stats["files_without_tests"] = len(untested)

    def check_documentation(self, files: List[Path]):
        """Vérifie documentation (docstrings)"""
        missing_docs = []

        for file in files:
            if "test" in str(file):
                continue

            try:
                with open(file) as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        if not ast.get_docstring(node):
                            missing_docs.append((file.relative_to(self.root), node.name))

            except Exception:
                pass

        print(f"   ⚠️ {len(missing_docs)} fonctions/classes sans docstring")

        for file, name in missing_docs[:10]:
            self.warnings.append(f"⚠️ NO DOC: {file}::{name}")

        self.stats["missing_docs"] = len(missing_docs)

    def print_report(self):
        """Affiche rapport final"""
        print("\n" + "=" * 80)
        print("📊 RAPPORT FINAL")
        print("=" * 80)

        # Erreurs critiques
        if self.errors:
            print(f"\n❌ ERREURS CRITIQUES ({len(self.errors)}):")
            for err in self.errors[:10]:
                print(f"   {err}")
        else:
            print(f"\n✅ Aucune erreur critique")

        # Top warnings
        if self.warnings:
            print(f"\n⚠️ WARNINGS TOP 20 ({len(self.warnings)} total):")
            for warn in self.warnings[:20]:
                print(f"   {warn}")

        # Stats
        print(f"\n📊 STATISTIQUES:")
        print(f"   - Fichiers analysés: {self.stats['files_ok']}")
        print(f"   - Imports inutilisés: {self.stats['unused_imports']}")
        print(f"   - Fonctions complexes: {self.stats['complex_functions']}")
        print(f"   - Problèmes bonnes pratiques: {self.stats['best_practice_issues']}")
        print(f"   - Fichiers sans tests: {self.stats['files_without_tests']}")
        print(f"   - Manque documentation: {self.stats['missing_docs']}")

        # Note globale
        total_issues = len(self.errors) + len(self.warnings)
        if total_issues == 0:
            grade = "A+ 🏆"
        elif total_issues < 20:
            grade = "A ✅"
        elif total_issues < 50:
            grade = "B ⚠️"
        elif total_issues < 100:
            grade = "C ⚠️"
        else:
            grade = "D ❌"

        print(f"\n🎯 NOTE GLOBALE: {grade}")
        print(f"   Total issues: {total_issues}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    auditor = CodeAuditor("src")
    auditor.audit_all()
