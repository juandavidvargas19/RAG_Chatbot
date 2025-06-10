#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create logs directory and log file
mkdir -p logs
LOG_FILE="logs/setup_$(date +%Y%m%d_%H%M%S).log"

# Redirect all output to both console and log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo -e "${GREEN}🔧 Setting up environment for PDF RAG App deployment${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if running on Ubuntu/Debian
if [[ -f /etc/debian_version ]]; then
    DISTRO="debian"
elif [[ -f /etc/redhat-release ]]; then
    DISTRO="redhat"
else
    DISTRO="unknown"
fi

echo -e "${YELLOW}Detected distribution: $DISTRO${NC}"

# Fix Docker permissions
echo -e "${YELLOW}Fixing Docker permissions...${NC}"
if ! groups $USER | grep -q docker; then
    echo -e "${YELLOW}Adding user to docker group...${NC}"
    sudo usermod -aG docker $USER
    echo -e "${RED}⚠️  You need to log out and log back in for group changes to take effect${NC}"
    echo -e "${RED}   Or run: newgrp docker${NC}"
fi

# Start Docker service if not running
echo -e "${YELLOW}Starting Docker service...${NC}"
sudo systemctl start docker
sudo systemctl enable docker

# Install kubectl if not present
if ! command_exists kubectl; then
    echo -e "${YELLOW}Installing kubectl...${NC}"
    if [[ "$DISTRO" == "debian" ]]; then
        # For Ubuntu/Debian
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
        rm kubectl
    else
        echo -e "${RED}Please install kubectl manually for your distribution${NC}"
        echo -e "${YELLOW}Visit: https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/${NC}"
    fi
else
    echo -e "${GREEN}✅ kubectl already installed${NC}"
fi

# Install minikube if not present
if ! command_exists minikube; then
    echo -e "${YELLOW}Installing minikube...${NC}"
    if [[ "$DISTRO" == "debian" ]]; then
        curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
        sudo install minikube-linux-amd64 /usr/local/bin/minikube
        rm minikube-linux-amd64
    else
        echo -e "${RED}Please install minikube manually for your distribution${NC}"
        echo -e "${YELLOW}Visit: https://minikube.sigs.k8s.io/docs/start/${NC}"
    fi
else
    echo -e "${GREEN}✅ minikube already installed${NC}"
fi

# Check Docker installation
if ! command_exists docker; then
    echo -e "${YELLOW}Installing Docker...${NC}"
    if [[ "$DISTRO" == "debian" ]]; then
        # Install Docker on Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
        
        # Add Docker's official GPG key
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        
        # Set up stable repository
        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        
        # Install Docker Engine
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io
        
        # Add user to docker group
        sudo usermod -aG docker $USER
    else
        echo -e "${RED}Please install Docker manually for your distribution${NC}"
    fi
else
    echo -e "${GREEN}✅ Docker already installed${NC}"
fi

echo -e "${GREEN}🎉 Setup complete!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Log out and log back in (or run 'newgrp docker')"
echo -e "2. Run './deploy.sh' to deploy your application"

# Test Docker access
echo -e "${YELLOW}Testing Docker access...${NC}"
if docker ps &> /dev/null; then
    echo -e "${GREEN}✅ Docker is accessible${NC}"
else
    echo -e "${RED}❌ Docker permission issue. Please run 'newgrp docker' or log out/in${NC}"
fi

echo "Log file saved to: $LOG_FILE"