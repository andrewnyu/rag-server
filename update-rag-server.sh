#!/bin/bash

# Simple RAG server update script with disk cleanup
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

# Check disk space before starting
df -h . | grep -v "Filesystem"
echo ""

# Clean up Docker resources to free disk space
echo "Cleaning up Docker resources..."
# Stop and remove all containers associated with this project
docker-compose down

# Clean up unused Docker resources
echo "Removing dangling images..."
docker image prune -f
echo "Removing unused containers..."
docker container prune -f
echo "Removing build cache..."
docker builder prune -f

# Report disk space after cleanup
echo "Disk space after cleanup:"
df -h . | grep -v "Filesystem"
echo ""

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
docker-compose build
docker-compose up -d

# Final cleanup after build
echo "Performing final cleanup..."
docker image prune -f

echo "=== Update completed ==="
echo "Service is running at: http://localhost:${SERVER_PORT:-8000}"

# Report final disk space
echo "Final disk space:"
df -h . | grep -v "Filesystem" 