# Sentinel2 - Système de Trading Algorithmique
# Makefile pour la gestion complète de l'application

.PHONY: help install start stop restart clean status logs

# Variables
APP_NAME = sentinel2
STREAMLIT_PORT = 8501
OLLAMA_PORT = 11434
VENV_DIR = .venv

# Couleurs pour les messages
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

help: ## Afficher l'aide
	@echo "$(GREEN)🚀 Sentinel2 - Système de Trading Algorithmique$(NC)"
	@echo ""
	@echo "$(YELLOW)Commandes disponibles:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Installer les dépendances
	@echo "$(YELLOW)📦 Installation des dépendances...$(NC)"
	uv sync
	@echo "$(GREEN)✅ Dépendances installées$(NC)"

start: ## Démarrer l'application (avec Ollama)
	@echo "$(YELLOW)🚀 Démarrage de Sentinel2...$(NC)"
	@make start-ollama
	@echo "$(YELLOW)⏳ Attente du démarrage d'Ollama...$(NC)"
	@sleep 3
	@make start-orchestrator
	@echo "$(YELLOW)⏳ Attente du démarrage de l'orchestrateur...$(NC)"
	@sleep 2
	@make start-streamlit
	@echo "$(GREEN)✅ Application démarrée sur http://localhost:$(STREAMLIT_PORT)$(NC)"

start-ollama: ## Démarrer Ollama en arrière-plan
	@echo "$(YELLOW)🧠 Démarrage d'Ollama...$(NC)"
	@if ! pgrep -f "ollama serve" > /dev/null; then \
		ollama serve > /dev/null 2>&1 & \
		echo "$(GREEN)✅ Ollama démarré$(NC)"; \
	else \
		echo "$(YELLOW)⚠️ Ollama déjà en cours d'exécution$(NC)"; \
	fi

start-orchestrator: ## Démarrer l'orchestrateur (sentinel_main)
	@echo "$(YELLOW)🤖 Démarrage de l'orchestrateur...$(NC)"
	@if ! pgrep -f "sentinel_main.py" > /dev/null; then \
		nohup uv run python scripts/sentinel_main.py > data/logs/sentinel_orchestrator.log 2>&1 & \
		echo "$(GREEN)✅ Orchestrateur démarré$(NC)"; \
	else \
		echo "$(YELLOW)⚠️ Orchestrateur déjà en cours d'exécution$(NC)"; \
	fi

start-streamlit: ## Démarrer Streamlit
	@echo "$(YELLOW)📊 Démarrage de Streamlit...$(NC)"
	@if ! pgrep -f "streamlit run" > /dev/null; then \
		uv run streamlit run src/gui/main.py --server.port $(STREAMLIT_PORT) > /dev/null 2>&1 & \
		echo "$(GREEN)✅ Streamlit démarré$(NC)"; \
	else \
		echo "$(YELLOW)⚠️ Streamlit déjà en cours d'exécution$(NC)"; \
	fi

stop: ## Arrêter l'application
	@echo "$(YELLOW)🛑 Arrêt de Sentinel2...$(NC)"
	@make stop-streamlit
	@make stop-orchestrator
	@make stop-ollama
	@echo "$(GREEN)✅ Application arrêtée$(NC)"

stop-streamlit: ## Arrêter Streamlit
	@echo "$(YELLOW)📊 Arrêt de Streamlit...$(NC)"
	@pkill -f "streamlit run" || true
	@echo "$(GREEN)✅ Streamlit arrêté$(NC)"

stop-orchestrator: ## Arrêter l'orchestrateur
	@echo "$(YELLOW)🤖 Arrêt de l'orchestrateur...$(NC)"
	@pkill -f "sentinel_main.py" || true
	@echo "$(GREEN)✅ Orchestrateur arrêté$(NC)"

stop-ollama: ## Arrêter Ollama
	@echo "$(YELLOW)🧠 Arrêt d'Ollama...$(NC)"
	@pkill -f "ollama serve" || true
	@echo "$(GREEN)✅ Ollama arrêté$(NC)"

restart: ## Redémarrer l'application
	@echo "$(YELLOW)🔄 Redémarrage de Sentinel2...$(NC)"
	@make stop
	@sleep 2
	@make start
	@echo "$(GREEN)✅ Application redémarrée$(NC)"

status: ## Vérifier le statut de l'application
	@echo "$(YELLOW)📊 Statut de Sentinel2:$(NC)"
	@echo ""
	@echo "$(YELLOW)Orchestrateur:$(NC)"
	@if pgrep -f "sentinel_main.py" > /dev/null; then \
		echo "  $(GREEN)✅ En cours d'exécution$(NC)"; \
	else \
		echo "  $(RED)❌ Arrêté$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Streamlit:$(NC)"
	@if pgrep -f "streamlit run" > /dev/null; then \
		echo "  $(GREEN)✅ En cours d'exécution$(NC)"; \
		echo "  $(GREEN)   URL: http://localhost:$(STREAMLIT_PORT)$(NC)"; \
	else \
		echo "  $(RED)❌ Arrêté$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Ollama:$(NC)"
	@if pgrep -f "ollama serve" > /dev/null; then \
		echo "  $(GREEN)✅ En cours d'exécution$(NC)"; \
		echo "  $(GREEN)   Port: $(OLLAMA_PORT)$(NC)"; \
	else \
		echo "  $(RED)❌ Arrêté$(NC)"; \
	fi

logs: ## Afficher les logs de l'application
	@echo "$(YELLOW)📋 Logs de Sentinel2:$(NC)"
	@if [ -f "data/logs/sentinel_main.log" ]; then \
		tail -f data/logs/sentinel_main.log; \
	else \
		echo "$(RED)❌ Aucun fichier de log trouvé$(NC)"; \
	fi

clean: ## Nettoyer les caches et fichiers temporaires
	@echo "$(YELLOW)🧹 Nettoyage de Sentinel2...$(NC)"
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

clean-all: clean ## Nettoyage complet (supprime .venv)
	@echo "$(YELLOW)🧹 Nettoyage complet...$(NC)"
	@rm -rf $(VENV_DIR) 2>/dev/null || true
	@echo "$(GREEN)✅ Nettoyage complet terminé$(NC)"

clean-logs: ## Nettoyer les logs et décisions (INTERACTIF)
	@echo "$(YELLOW)🧹 Nettoyage des logs et décisions...$(NC)"
	@bash scripts/clean_logs.sh

dev: ## Mode développement (sans Ollama)
	@echo "$(YELLOW)🔧 Mode développement...$(NC)"
	@make stop
	@make start-streamlit
	@echo "$(GREEN)✅ Mode développement activé$(NC)"

prod: ## Mode production (avec Ollama)
	@echo "$(YELLOW)🚀 Mode production...$(NC)"
	@make start
	@echo "$(GREEN)✅ Mode production activé$(NC)"

test: ## Lancer les tests
	@echo "$(YELLOW)🧪 Lancement des tests...$(NC)"
	uv run python -m pytest tests/ -v
	@echo "$(GREEN)✅ Tests terminés$(NC)"

# Commande par défaut
.DEFAULT_GOAL := help
