#!/bin/bash
set -e

echo "Cleaning up containers and volumes..."
docker-compose down -v
echo "Clean complete!"
echo ""

echo "Building & starting container..."
docker-compose up -d --build
echo "Container started!"
echo ""

echo "Viewing logs..."
docker-compose logs -f

# Remove dead/orphan/duplicate code. Optimize & refactor configuration & code (both backend and frontend).

# Update /api/docs with all available endpoints and help me find which are public (used by frontend) & which are private (used by backend) & which are not used.
