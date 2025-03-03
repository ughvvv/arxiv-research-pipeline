#!/bin/bash

# EC2 instance details
EC2_HOST="ubuntu@3.144.215.99"
KEY_PATH="/Users/blakecole/.ssh/arxiv-analyzer-sk.pem"

echo "Deploying Research Pipeline to EC2..."

# Copy project files to EC2
echo "Copying project files..."
scp -i $KEY_PATH -r \
    main.py \
    config.py \
    arxiv_client.py \
    semanticscholar_client.py \
    openai_utils.py \
    analyze_scores.py \
    novelty_detector.py \
    requirements.txt \
    .env \
    Dockerfile \
    crontab.txt \
    $EC2_HOST:~/research-pipeline/

# Create cache directory on EC2 if it doesn't exist
ssh -i $KEY_PATH $EC2_HOST "mkdir -p ~/research-pipeline/cache"

# SSH into EC2 and build/run Docker container
ssh -i $KEY_PATH $EC2_HOST << 'EOF'
    cd ~/research-pipeline
    
    # Build Docker image
    echo "Building Docker image..."
    sudo docker build -t research-pipeline .
    
    # Stop any existing container
    echo "Stopping existing container..."
    sudo docker stop research-pipeline || true
    sudo docker rm research-pipeline || true
    
    # Run new container
    echo "Starting new container..."
    sudo docker run -d \
        --name research-pipeline \
        --restart unless-stopped \
        -v ~/research-pipeline/cache:/app/cache \
        research-pipeline
    
    # Check container status
    echo "Container status:"
    sudo docker ps -a | grep research-pipeline
    
    # Show initial logs
    echo "Container logs:"
    sudo docker logs research-pipeline
EOF

echo "Deployment complete!"
