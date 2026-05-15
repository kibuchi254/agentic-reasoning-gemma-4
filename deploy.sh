#!/bin/bash
# Lightning AI Production Deployment Script
# This script is executed by the webhook endpoint in server.py

set -e # Exit immediately if a command exits with a non-zero status

# Path in Lightning Studio
PROJECT_DIR="/teamspace/studios/this_studio"

echo "Starting deployment at $(date)..."
cd $PROJECT_DIR

# 1. Pull the latest code from GitHub
echo "Pulling latest code from origin/main..."
git pull origin main

# 2. Install/update dependencies
# Lightning AI uses the base python environment usually, but we will make sure everything is installed
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Any additional setups (e.g., migrations, pre-computes) can go here
# ...

# 4. Restart the Supervisor process
# The webhook is actually running *inside* this process, so restarting it 
# will kill the webhook response but supervisor will bring it right back up!
echo "Restarting the application via Supervisor..."
sudo supervisorctl restart gemma-reasoning

echo "Deployment completed successfully!"
