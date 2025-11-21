#!/bin/bash
# Deployment Script for Modular Semantic Discovery System
# Agent 10: Documentation & Deployment Manager
# Version: 1.0.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ENV=${1:-production}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-}
VERSION=${VERSION:-latest}

echo "=================================================="
echo "  MIDI DNA API Deployment"
echo "  Environment: $DEPLOYMENT_ENV"
echo "  Version: $VERSION"
echo "=================================================="
echo

# Function to print colored messages
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    log_info "✓ Prerequisites met"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."

    cd "$(dirname "$0")/../../.."

    docker build \
        -f midi_generator/deployment/docker/Dockerfile \
        -t midi-dna-api:${VERSION} \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VERSION=${VERSION} \
        .

    log_info "✓ Image built successfully"
}

# Push to registry (if configured)
push_image() {
    if [ -n "$DOCKER_REGISTRY" ]; then
        log_info "Pushing image to registry..."

        docker tag midi-dna-api:${VERSION} ${DOCKER_REGISTRY}/midi-dna-api:${VERSION}
        docker push ${DOCKER_REGISTRY}/midi-dna-api:${VERSION}

        log_info "✓ Image pushed to registry"
    else
        log_warn "No registry configured, skipping push"
    fi
}

# Download pre-trained models
download_models() {
    log_info "Downloading pre-trained models..."

    MODELS_DIR="midi_generator/models"
    mkdir -p "$MODELS_DIR"

    # Check if models exist
    if [ -f "$MODELS_DIR/modular_semantic_discovery_v1.pth" ]; then
        log_info "✓ Models already exist"
    else
        log_warn "Pre-trained models not found"
        log_info "To train models, run:"
        log_info "  python -m midi_generator.learning.train_modular_pipeline \\"
        log_info "    --corpus data/midi_corpus \\"
        log_info "    --output output/training \\"
        log_info "    --features 120 \\"
        log_info "    --epochs 100 \\"
        log_info "    --device cuda"
    fi
}

# Start services
start_services() {
    log_info "Starting services..."

    cd midi_generator/deployment/docker

    docker-compose up -d

    log_info "✓ Services started"
}

# Stop services
stop_services() {
    log_info "Stopping services..."

    cd midi_generator/deployment/docker

    docker-compose down

    log_info "✓ Services stopped"
}

# Health check
health_check() {
    log_info "Checking service health..."

    max_attempts=30
    attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log_info "✓ API is healthy"
            return 0
        fi

        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done

    log_error "Health check failed"
    return 1
}

# Run tests
run_tests() {
    log_info "Running tests..."

    docker-compose exec api python -m pytest tests/ -v

    log_info "✓ Tests passed"
}

# Deploy
deploy() {
    log_info "Starting deployment..."

    check_prerequisites
    download_models
    build_image
    push_image
    stop_services
    start_services
    sleep 5
    health_check

    log_info "✓ Deployment complete"

    echo
    echo "=================================================="
    echo "  Deployment Summary"
    echo "=================================================="
    echo "API endpoint:  http://localhost:8000"
    echo "API docs:      http://localhost:8000/docs"
    echo "Health check:  http://localhost:8000/health"
    echo "=================================================="
    echo
    echo "To view logs:"
    echo "  docker-compose -f midi_generator/deployment/docker/docker-compose.yml logs -f"
    echo
}

# Rollback
rollback() {
    log_warn "Rolling back deployment..."

    PREVIOUS_VERSION=${2:-previous}

    docker-compose down
    docker tag midi-dna-api:${PREVIOUS_VERSION} midi-dna-api:${VERSION}
    docker-compose up -d

    log_info "✓ Rollback complete"
}

# Main
case "${2:-deploy}" in
    deploy)
        deploy
        ;;
    build)
        check_prerequisites
        build_image
        ;;
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        start_services
        ;;
    health)
        health_check
        ;;
    test)
        run_tests
        ;;
    rollback)
        rollback
        ;;
    *)
        echo "Usage: $0 [environment] {deploy|build|start|stop|restart|health|test|rollback}"
        exit 1
        ;;
esac

exit 0
