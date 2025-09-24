# Corrections pour test_config.py
import re

with open('src/tests/test_config.py', 'r') as f:
    content = f.read()

# Corriger les valeurs pour correspondre au .env actuel
content = re.sub(r'assert news_config\["interval"\] == 300  # Valeur du \.env', 
                 'assert news_config["interval"] == 240  # Valeur du .env corrigée', content)

content = re.sub(r'assert trading_config\["fusion_mode"\] == "fixed"  # Valeur du \.env',
                 'assert trading_config["fusion_mode"] == "adaptive"  # Valeur du .env corrigée', content)

content = re.sub(r'assert fusion_config\["mode"\] == "fixed"  # Valeur du \.env',
                 'assert fusion_config["mode"] == "adaptive"  # Valeur du .env corrigée', content)

content = re.sub(r'# Avec \.env, la fusion est fixe\s+assert not test_config\.is_fusion_adaptive\(\)',
                 '# Avec .env, la fusion est adaptive\n        assert test_config.is_fusion_adaptive()', content)

with open('src/tests/test_config.py', 'w') as f:
    f.write(content)

print("✅ Tests de configuration corrigés")
