#!/bin/bash

# Simple RAG server update script
# Usage: ./update-rag-server.sh

set -e  # Exit on any error

# Load environment variables if .env exists
if [ -f ".env" ]; then
  source .env
fi

# Use environment variables or defaults
BRANCH=${GIT_BRANCH:-main}
REPO=${GIT_REPO_URL:-"origin"}

echo "=== RAG Server Update ==="
echo "Using branch: ${BRANCH}"

# Update code using deploy key if configured
if [ -n "$GIT_DEPLOY_KEY_PATH" ] && [ -f "$GIT_DEPLOY_KEY_PATH" ]; then
  echo "Using deploy key for Git operations"
  GIT_SSH_COMMAND="ssh -i $GIT_DEPLOY_KEY_PATH -o StrictHostKeyChecking=no" git pull $REPO $BRANCH
else
  echo "Using default SSH configuration"
  git pull $REPO $BRANCH
fi

# Rebuild and restart Docker containers
echo "Rebuilding and restarting Docker containers..."
docker-compose down
docker-compose build
docker-compose up -d

echo "=== Update completed ==="
echo "Service is running at: http://localhost:${SERVER_PORT:-8000}" 