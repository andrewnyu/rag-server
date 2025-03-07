#!/bin/bash

# update-rag-server.sh - Script to update and deploy the RAG server
# Usage: ./update-rag-server.sh [branch_name]

set -e  # Exit on any error

# Default branch is main if not specified
BRANCH=${1:-main}
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="${SCRIPT_DIR}/backups/${TIMESTAMP}"

# Print header
echo "=========================================="
echo "RAG Server Update Script"
echo "=========================================="
echo "Current directory: ${SCRIPT_DIR}"
echo "Using branch: ${BRANCH}"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================="

# Function to check if Docker is running
check_docker() {
  if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running or not installed."
    exit 1
  fi
}

# Function to create backup
create_backup() {
  echo "Creating backup..."
  mkdir -p "${BACKUP_DIR}"
  
  # Backup uploads directory
  if [ -d "${SCRIPT_DIR}/uploads" ]; then
    echo "Backing up uploads directory..."
    cp -r "${SCRIPT_DIR}/uploads" "${BACKUP_DIR}/"
  fi
  
  # Backup db directory
  if [ -d "${SCRIPT_DIR}/db" ]; then
    echo "Backing up db directory..."
    cp -r "${SCRIPT_DIR}/db" "${BACKUP_DIR}/"
  fi
  
  echo "Backup created at: ${BACKUP_DIR}"
}

# Function to update code
update_code() {
  echo "Updating code from repository..."
  
  # Check if .git directory exists
  if [ -d "${SCRIPT_DIR}/.git" ]; then
    # Save any local changes
    if git diff --quiet; then
      echo "No local changes detected."
    else
      echo "Local changes detected, creating patch..."
      git diff > "${BACKUP_DIR}/local_changes.patch"
      echo "Local changes saved to ${BACKUP_DIR}/local_changes.patch"
      
      # Ask if user wants to stash changes
      read -p "Do you want to stash local changes? (y/n): " STASH_CHANGES
      if [[ "$STASH_CHANGES" =~ ^[Yy]$ ]]; then
        git stash
        echo "Local changes stashed."
      else
        echo "Warning: Proceeding with local changes. Pull might fail."
      fi
    fi
    
    # Pull latest code
    git fetch origin
    git checkout ${BRANCH}
    git pull origin ${BRANCH}
    echo "Code updated to latest version of ${BRANCH}."
  else
    echo "Not a git repository. Skipping code update."
  fi
}

# Function to rebuild and restart Docker containers
rebuild_docker() {
  echo "Rebuilding and restarting Docker containers..."
  
  # Stop existing containers
  if docker-compose ps | grep -q "rag-server"; then
    echo "Stopping existing containers..."
    docker-compose down
  fi
  
  # Build and start containers
  echo "Building new containers..."
  docker-compose build
  
  echo "Starting containers..."
  docker-compose up -d
  
  # Check if containers started successfully
  if docker-compose ps | grep -q "rag-server" && docker-compose ps | grep -q "Up"; then
    echo "Containers started successfully."
  else
    echo "Error: Containers failed to start. Check docker-compose logs."
    docker-compose logs
    exit 1
  fi
}

# Function to verify the service is running
verify_service() {
  echo "Verifying service is running..."
  
  # Wait for service to start
  echo "Waiting for service to start (10 seconds)..."
  sleep 10
  
  # Check if service is responding
  if curl -s http://localhost:8000/ > /dev/null; then
    echo "Service is running and responding."
  else
    echo "Warning: Service is not responding. Check logs with: docker-compose logs"
  fi
}

# Main execution
main() {
  # Check Docker
  check_docker
  
  # Create backup
  create_backup
  
  # Update code
  update_code
  
  # Rebuild and restart Docker
  rebuild_docker
  
  # Verify service
  verify_service
  
  echo "=========================================="
  echo "Update completed successfully!"
  echo "Service is running at: http://localhost:8000"
  echo "=========================================="
}

# Run the main function
main 