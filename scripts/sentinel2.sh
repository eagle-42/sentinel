#!/bin/bash

# Sentinel2 - Script de gestion complet
# Usage: ./scripts/sentinel2.sh [start|stop|restart|status|dev|prod]

set -e

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Variables
APP_NAME="Sentinel2"
STREAMLIT_PORT=8501
OLLAMA_PORT=11434
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Fonctions
log_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_dependencies() {
    log_info "Vérification des dépendances..."
    
    if ! command -v uv &> /dev/null; then
        log_error "uv n'est pas installé. Installez-le depuis https://docs.astral.sh/uv/"
        exit 1
    fi
    
    if ! command -v ollama &> /dev/null; then
        log_error "Ollama n'est pas installé. Installez-le depuis https://ollama.ai/"
        exit 1
    fi
    
    log_success "Toutes les dépendances sont installées"
}

start_ollama() {
    log_info "Démarrage d'Ollama..."
    
    if pgrep -f "ollama serve" > /dev/null; then
        log_info "Ollama est déjà en cours d'exécution"
        return 0
    fi
    
    # Démarrer Ollama en arrière-plan
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    
    # Attendre que le serveur soit prêt
    log_info "Attente du démarrage d'Ollama..."
    for i in {1..10}; do
        if curl -s http://localhost:$OLLAMA_PORT/api/tags > /dev/null 2>&1; then
            log_success "Ollama est prêt sur le port $OLLAMA_PORT"
            return 0
        fi
        sleep 1
    done
    
    log_error "Ollama n'a pas démarré correctement"
    exit 1
}

stop_ollama() {
    log_info "Arrêt d'Ollama..."
    
    if pgrep -f "ollama serve" > /dev/null; then
        pkill -f "ollama serve"
        log_success "Ollama arrêté"
    else
        log_info "Ollama n'était pas en cours d'exécution"
    fi
}

start_streamlit() {
    log_info "Démarrage de Streamlit..."
    
    if pgrep -f "streamlit run" > /dev/null; then
        log_info "Streamlit est déjà en cours d'exécution"
        return 0
    fi
    
    cd "$PROJECT_DIR"
    uv run streamlit run src/gui/main.py --server.port $STREAMLIT_PORT > /dev/null 2>&1 &
    STREAMLIT_PID=$!
    
    # Attendre que Streamlit soit prêt
    log_info "Attente du démarrage de Streamlit..."
    for i in {1..15}; do
        if curl -s http://localhost:$STREAMLIT_PORT > /dev/null 2>&1; then
            log_success "Streamlit est prêt sur http://localhost:$STREAMLIT_PORT"
            return 0
        fi
        sleep 1
    done
    
    log_error "Streamlit n'a pas démarré correctement"
    exit 1
}

stop_streamlit() {
    log_info "Arrêt de Streamlit..."
    
    if pgrep -f "streamlit run" > /dev/null; then
        pkill -f "streamlit run"
        log_success "Streamlit arrêté"
    else
        log_info "Streamlit n'était pas en cours d'exécution"
    fi
}

show_status() {
    log_info "Statut de $APP_NAME:"
    echo ""
    
    echo -e "${YELLOW}Streamlit:${NC}"
    if pgrep -f "streamlit run" > /dev/null; then
        echo -e "  ${GREEN}✅ En cours d'exécution${NC}"
        echo -e "  ${GREEN}   URL: http://localhost:$STREAMLIT_PORT${NC}"
    else
        echo -e "  ${RED}❌ Arrêté${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}Ollama:${NC}"
    if pgrep -f "ollama serve" > /dev/null; then
        echo -e "  ${GREEN}✅ En cours d'exécution${NC}"
        echo -e "  ${GREEN}   Port: $OLLAMA_PORT${NC}"
    else
        echo -e "  ${RED}❌ Arrêté${NC}"
    fi
}

# Fonction principale
main() {
    case "${1:-help}" in
        start)
            log_info "Démarrage de $APP_NAME..."
            check_dependencies
            start_ollama
            start_streamlit
            log_success "$APP_NAME démarré avec succès!"
            echo -e "${GREEN}🌐 Accédez à l'application: http://localhost:$STREAMLIT_PORT${NC}"
            ;;
        stop)
            log_info "Arrêt de $APP_NAME..."
            stop_streamlit
            stop_ollama
            log_success "$APP_NAME arrêté"
            ;;
        restart)
            log_info "Redémarrage de $APP_NAME..."
            stop_streamlit
            stop_ollama
            sleep 2
            start_ollama
            start_streamlit
            log_success "$APP_NAME redémarré"
            ;;
        status)
            show_status
            ;;
        dev)
            log_info "Mode développement (sans Ollama)..."
            check_dependencies
            stop_streamlit
            start_streamlit
            log_success "Mode développement activé"
            echo -e "${GREEN}🌐 Accédez à l'application: http://localhost:$STREAMLIT_PORT${NC}"
            ;;
        prod)
            log_info "Mode production (avec Ollama)..."
            check_dependencies
            start_ollama
            start_streamlit
            log_success "Mode production activé"
            echo -e "${GREEN}🌐 Accédez à l'application: http://localhost:$STREAMLIT_PORT${NC}"
            ;;
        help|*)
            echo -e "${GREEN}🚀 $APP_NAME - Système de Trading Algorithmique${NC}"
            echo ""
            echo -e "${YELLOW}Usage: $0 [commande]${NC}"
            echo ""
            echo -e "${YELLOW}Commandes disponibles:${NC}"
            echo -e "  ${GREEN}start${NC}     Démarrer l'application (avec Ollama)"
            echo -e "  ${GREEN}stop${NC}      Arrêter l'application"
            echo -e "  ${GREEN}restart${NC}   Redémarrer l'application"
            echo -e "  ${GREEN}status${NC}    Afficher le statut"
            echo -e "  ${GREEN}dev${NC}       Mode développement (sans Ollama)"
            echo -e "  ${GREEN}prod${NC}      Mode production (avec Ollama)"
            echo -e "  ${GREEN}help${NC}      Afficher cette aide"
            ;;
    esac
}

# Exécuter la fonction principale
main "$@"
