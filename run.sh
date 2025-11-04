#!/bin/bash

# DeepSeek Coder: Algorithms, optimization, complex logic, performance
# CodeLlama: Refactoring, best practices, clean code, documentation
# Qwen2.5-Coder: Explanations, tutorials, learning, reasoning
# StarCoder2: Multi-language, frameworks, templates, APIs

# Build and run script for C++ Assistant
set -e

docker-compose down -v && docker-compose up -d --build && docker-compose logs -f

# echo "Cleaning up containers and volumes..."
# docker-compose down -v
# echo "Clean complete!"
# echo ""

# echo "Building container with podman..."
# docker-compose build
# echo "Build complete!"
# echo ""

# echo "Starting container..."
# docker-compose up -d
# echo "Container started!"
# echo ""

# echo "Viewing logs..."
# docker-compose logs -f
# docker-compose logs -f assistant

# Remove dead/orphan code, optimize the code (backend and frontend). Refactor code. Move duplicate code to utils or something similar. DO we still need setup_model.sh. I think first time setup is already covered.

# Write a c++ function that exposes some functions (fun1, fun2, fun3) as a object (cppObject) to javascript using v8

# Find all avalable API endpoints, is any of them orphaned?

# Update /docs with all available endpoints and help me find which are public (used by frontend) & which are private (used by backend) & which are not used.
