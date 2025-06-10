#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create logs directory and log file
mkdir -p logs
LOG_FILE="logs/deploy_$(date +%Y%m%d_%H%M%S).log"

# Redirect all output to both console and log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo -e "${GREEN}🚀 Deploying PDF RAG App (Simple - No ConfigMap)${NC}"

# Check if required commands exist
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

if ! command_exists kubectl; then
    echo -e "${RED}❌ kubectl not found${NC}"
    exit 1
fi

if ! command_exists minikube; then
    echo -e "${RED}❌ minikube not found${NC}"
    exit 1
fi

if ! command_exists docker; then
    echo -e "${RED}❌ docker not found${NC}"
    exit 1
fi

# Test Docker access
if ! docker ps &> /dev/null; then
    echo -e "${RED}❌ Docker permission denied. Please run: newgrp docker${NC}"
    exit 1
fi

# Check if PDF file exists
PDF_FILE="random_machine_learing.pdf"
if [ ! -f "$PDF_FILE" ]; then
    echo -e "${RED}❌ PDF file '$PDF_FILE' not found!${NC}"
    echo -e "${YELLOW}Current directory: $(pwd)${NC}"
    echo -e "${YELLOW}Available files:${NC}"
    ls -la *.pdf 2>/dev/null || echo "No PDF files found"
    exit 1
fi

echo -e "${GREEN}✅ Found PDF file: $PDF_FILE ($(du -h "$PDF_FILE" | cut -f1))${NC}"

# Check if minikube is running
echo -e "${YELLOW}Checking Minikube status...${NC}"
if ! minikube status &> /dev/null; then
    echo -e "${YELLOW}Starting Minikube...${NC}"
    minikube start --memory=4096 --cpus=2 --driver=docker
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to start Minikube${NC}"
        exit 1
    fi
fi

# Set docker environment to use minikube's docker daemon
echo -e "${YELLOW}Setting Docker environment for Minikube...${NC}"
eval $(minikube docker-env)

# Clean up any existing resources
echo -e "${YELLOW}Cleaning up existing resources...${NC}"
kubectl delete deployment pdf-rag-app --ignore-not-found=true
kubectl delete service pdf-rag-service --ignore-not-found=true
kubectl delete configmap pdf-configmap --ignore-not-found=true

# Wait for cleanup
sleep 3

# Apply secrets
echo -e "${YELLOW}Applying secrets...${NC}"
echo -e "${RED}⚠️  Make sure you've updated secrets.yaml with your actual OpenAI API key!${NC}"
kubectl apply -f secrets.yaml
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to apply secrets${NC}"
    exit 1
fi

# Build Docker image (this will include the PDF file)
echo -e "${YELLOW}Building Docker image with embedded PDF...${NC}"
docker build -f Dockerfile -t pdf-rag-app:latest .
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker image built successfully${NC}"

# Apply deployment (without ConfigMap)
echo -e "${YELLOW}Applying deployment...${NC}"
kubectl apply -f deployment.yaml
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to apply deployment${NC}"
    exit 1
fi

# Wait for deployment to be ready
echo -e "${YELLOW}Waiting for deployment to be ready...${NC}"
kubectl wait --for=condition=available --timeout=600s deployment/pdf-rag-app

# Check pod status
echo -e "${YELLOW}Checking pod status:${NC}"
kubectl get pods -l app=pdf-rag-app

# Show pod events if needed
POD_STATUS=$(kubectl get pods -l app=pdf-rag-app -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
if [ "$POD_STATUS" != "Running" ]; then
    echo -e "${RED}❌ Pod is not running. Status: $POD_STATUS${NC}"
    echo -e "${YELLOW}Pod logs:${NC}"
    kubectl logs -l app=pdf-rag-app --tail=20
    exit 1
fi

# Get service URL
echo -e "${GREEN}✅ Deployment complete!${NC}"


# Check all service statuses
echo -e "${YELLOW}Checking service status:${NC}"
echo -e "${GREEN}SERVICE                    READY   STATUS      AGE     ACCESS URL${NC}"
echo -e "${GREEN}========================================================================${NC}"

# Get PDF RAG app status
RAG_STATUS=$(kubectl get pods -l app=pdf-rag-app -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "NotFound")
RAG_READY=$(kubectl get pods -l app=pdf-rag-app -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null || echo "false")
RAG_AGE=$(kubectl get pods -l app=pdf-rag-app -o jsonpath='{.items[0].metadata.creationTimestamp}' 2>/dev/null | xargs -I {} date -d {} +%s 2>/dev/null || echo "0")
RAG_URL=$(minikube service pdf-rag-service --url 2>/dev/null)

# Get Prometheus status
PROM_STATUS=$(kubectl get pods -l app.kubernetes.io/instance=prometheus -l app.kubernetes.io/name=prometheus -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "NotFound")
PROM_READY=$(kubectl get pods -l app.kubernetes.io/instance=prometheus -l app.kubernetes.io/name=prometheus -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null || echo "false")
PROM_AGE=$(kubectl get pods -l app.kubernetes.io/instance=prometheus -l app.kubernetes.io/name=prometheus -o jsonpath='{.items[0].metadata.creationTimestamp}' 2>/dev/null | xargs -I {} date -d {} +%s 2>/dev/null || echo "0")
PROM_URL=$(minikube service prometheus-server-np --url 2>/dev/null)

# Get Grafana status
GRAFANA_STATUS=$(kubectl get pods -l app.kubernetes.io/instance=grafana -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "NotFound")
GRAFANA_READY=$(kubectl get pods -l app.kubernetes.io/instance=grafana -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>/dev/null || echo "false")
GRAFANA_AGE=$(kubectl get pods -l app.kubernetes.io/instance=grafana -o jsonpath='{.items[0].metadata.creationTimestamp}' 2>/dev/null | xargs -I {} date -d {} +%s 2>/dev/null || echo "0")
GRAFANA_URL=$(minikube service grafana-np --url 2>/dev/null)

# Calculate ages in human readable format
CURRENT_TIME=$(date +%s)
RAG_AGE_HUMAN="Unknown"
PROM_AGE_HUMAN="Unknown"
GRAFANA_AGE_HUMAN="Unknown"

if [ "$RAG_AGE" != "0" ]; then
    RAG_AGE_DIFF=$((CURRENT_TIME - RAG_AGE))
    RAG_AGE_HUMAN="${RAG_AGE_DIFF}s"
fi

if [ "$PROM_AGE" != "0" ]; then
    PROM_AGE_DIFF=$((CURRENT_TIME - PROM_AGE))
    PROM_AGE_HUMAN="${PROM_AGE_DIFF}s"
fi

if [ "$GRAFANA_AGE" != "0" ]; then
    GRAFANA_AGE_DIFF=$((CURRENT_TIME - GRAFANA_AGE))
    GRAFANA_AGE_HUMAN="${GRAFANA_AGE_DIFF}s"
fi

# Format ready status
RAG_READY_STATUS="0/1"
if [ "$RAG_READY" = "true" ]; then RAG_READY_STATUS="1/1"; fi

PROM_READY_STATUS="0/1"
if [ "$PROM_READY" = "true" ]; then PROM_READY_STATUS="1/1"; fi

GRAFANA_READY_STATUS="0/1"
if [ "$GRAFANA_READY" = "true" ]; then GRAFANA_READY_STATUS="1/1"; fi

# Display status table
printf "%-20s %-7s %-10s %-7s %s\n" "pdf-rag-app" "$RAG_READY_STATUS" "$RAG_STATUS" "$RAG_AGE_HUMAN" "$RAG_URL"
printf "%-20s %-7s %-10s %-7s %s\n" "prometheus" "$PROM_READY_STATUS" "$PROM_STATUS" "$PROM_AGE_HUMAN" "$PROM_URL"
printf "%-20s %-7s %-10s %-7s %s\n" "grafana" "$GRAFANA_READY_STATUS" "$GRAFANA_STATUS" "$GRAFANA_AGE_HUMAN" "$GRAFANA_URL"

echo -e "${GREEN}========================================================================${NC}"

# Show quick access commands
echo -e "${GREEN}🎉 Your services are running!${NC}"

echo "Log file saved to: $LOG_FILE"