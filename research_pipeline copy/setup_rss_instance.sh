#!/bin/bash

# EC2 instance details
EC2_HOST="ubuntu@3.144.215.99"
KEY_PATH="~/.ssh/rss-feed-kp.pem"

echo "Setting up new RSS scraper instance..."

# SSH into EC2 and setup Docker
ssh -i $KEY_PATH $EC2_HOST << 'EOF'
    # Update system
    echo "Updating system packages..."
    sudo apt-get update && sudo apt-get upgrade -y

    # Install Docker
    echo "Installing Docker..."
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io

    # Add ubuntu user to docker group
    sudo usermod -aG docker ubuntu

    # Start Docker service
    sudo systemctl start docker
    sudo systemctl enable docker

    echo "Docker installation complete"
    docker --version
EOF

echo "Setup complete!"
