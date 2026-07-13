#!/bin/bash
# Script to check and compare Docker image sizes

set -e

echo "🔍 Docker Image Size Analysis"
echo "=============================="
echo ""

# Check if image exists
if docker images ai-assistant:latest --format "{{.Repository}}" | grep -q "ai-assistant"; then
    # Get image details
    IMAGE_SIZE=$(docker images ai-assistant:latest --format "{{.Size}}")
    IMAGE_ID=$(docker images ai-assistant:latest --format "{{.ID}}")
    CREATED=$(docker images ai-assistant:latest --format "{{.CreatedSince}}")
    
    echo "📦 Current Image:"
    echo "   Name:    ai-assistant:latest"
    echo "   ID:      $IMAGE_ID"
    echo "   Size:    $IMAGE_SIZE"
    echo "   Created: $CREATED"
    echo ""
    
    # Show layer breakdown
    echo "📊 Layer Breakdown:"
    docker history ai-assistant:latest --human=true --format "table {{.Size}}\t{{.CreatedBy}}" | head -20
    echo ""
    
    # Compare with baseline if exists
    if [ -f .docker-size-baseline ]; then
        BASELINE=$(cat .docker-size-baseline)
        echo "📈 Comparison:"
        echo "   Baseline: $BASELINE"
        echo "   Current:  $IMAGE_SIZE"
        echo ""
    else
        echo "💾 Saving current size as baseline..."
        echo "$IMAGE_SIZE" > .docker-size-baseline
        echo "   Baseline saved to .docker-size-baseline"
        echo ""
    fi
    
    # Show all related images
    echo "🗂️  All AI Assistant Images:"
    docker images | grep -E "REPOSITORY|ai-assistant|my_llm"
    echo ""
    
    # Disk usage
    echo "💽 Docker Disk Usage:"
    docker system df
    echo ""
    
    echo "✅ Analysis complete!"
    echo ""
    echo "💡 Tips:"
    echo "   - To rebuild: docker-compose build --no-cache"
    echo "   - To clean up: docker image prune -a"
    echo "   - To reset baseline: rm .docker-size-baseline"
    
else
    echo "❌ Image 'ai-assistant:latest' not found!"
    echo ""
    echo "Build the image first:"
    echo "   docker-compose build"
    exit 1
fi

