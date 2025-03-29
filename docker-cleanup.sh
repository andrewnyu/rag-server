#!/bin/bash

# docker-cleanup.sh - Aggressive Docker cleanup script
# Use this when you need to reclaim maximum disk space
# Usage: ./docker-cleanup.sh

echo "=== Docker Deep Cleanup ==="
echo "This will remove all unused Docker resources to free up disk space."
echo "Current disk space usage:"
df -h . | grep -v "Filesystem"
echo ""

echo "1. Stopping all running containers..."
docker-compose down 2>/dev/null || true
docker stop $(docker ps -aq) 2>/dev/null || true

echo "2. Removing all containers..."
docker rm $(docker ps -aq) 2>/dev/null || true

echo "3. Removing all unused images (including intermediates)..."
docker image prune -af

echo "4. Removing all build cache..."
docker builder prune -af

echo "5. Removing all volumes not used by at least one container..."
docker volume prune -f

echo "6. Removing all unused networks..."
docker network prune -f

echo "7. System prune (a final sweep)..."
docker system prune -af

echo ""
echo "Cleanup completed."
echo "Current disk space usage:"
df -h . | grep -v "Filesystem"

echo ""
echo "To rebuild and restart your RAG server:"
echo "  ./update-rag-server.sh" 