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
# docker-compose logs -f assistant

# Remove dead/orphan/duplicate code. Optimize & refactor configuration & code (both backend and frontend).

# Update /api/docs with all available endpoints and help me find which are public (used by frontend) & which are private (used by backend) & which are not used.

# Make the UI simple, Use the generate code UI. Add some option that user can select between code generation or presentation workflow. Even if user doesn't select anything, the backend should be smart enough to figure out what the user is trying to do.

# create presentation slide for android lifecycle, create a nice flowchart for demonstration

# Create presentation slide for Android lifecycle with flowchart

# Create a sequence diagram showing user authentication flow

# Create a Gantt chart for a 3-month software project

# Create an ER diagram for a blog database with users, posts, and comments

# Create a state diagram for an order processing system
