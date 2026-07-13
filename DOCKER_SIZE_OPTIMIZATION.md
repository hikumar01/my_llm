# Docker Image Size Optimization Guide

## 🎯 Current Optimizations Applied

### ✅ 1. Multi-Stage Build (30-50% reduction)
- **Before**: Build tools (clang, gcc, g++) included in final image
- **After**: Build tools only in builder stage, runtime only has minimal dependencies
- **Savings**: ~200-500 MB

### ✅ 2. CPU-Only PyTorch (~1 GB reduction)
- **Before**: Full PyTorch with CUDA support (~2 GB)
- **After**: CPU-only PyTorch (~800 MB)
- **Savings**: ~1-1.2 GB

### ✅ 3. Enhanced .dockerignore
- Excludes docs/ folder and unnecessary scripts
- **Savings**: ~50-100 MB

---

## 📊 Expected Results

| Optimization | Before | After | Savings |
|--------------|--------|-------|---------|
| Multi-stage build | ~3.5 GB | ~2.5 GB | ~1 GB |
| CPU-only PyTorch | ~2.5 GB | ~1.3 GB | ~1.2 GB |
| .dockerignore | ~1.3 GB | ~1.2 GB | ~100 MB |
| **TOTAL** | **~3.5 GB** | **~1.2 GB** | **~2.3 GB (66%)** |

---

## 🚀 How to Apply

### Step 1: Rebuild the image
```bash
# Clean rebuild with new optimizations
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Step 2: Verify image size
```bash
# Check image size
docker images | grep ai-assistant

# Expected output:
# ai-assistant   latest   abc123   1.2GB
```

### Step 3: Clean up old images
```bash
# Remove dangling images
docker image prune -f

# Remove all unused images
docker image prune -a
```

---

## 🔍 Additional Optimizations (Optional)

### Option 4: Use Alpine Linux (Advanced - Can save another 200-300 MB)
**Warning**: Alpine uses musl instead of glibc, which can cause compatibility issues with some Python packages.

```dockerfile
FROM python:3.12-alpine AS builder
# ... requires additional build dependencies
```

### Option 5: Remove Unnecessary Dependencies
Review if you really need all these packages:

```python
# Heavy packages in requirements.txt:
torch>=2.0.0              # ~800 MB (CPU-only)
sentence-transformers     # ~500 MB (includes models)
faiss-cpu                 # ~50 MB
```

**Alternative**: Use Ollama's built-in embeddings instead of sentence-transformers:
```bash
# Ollama can generate embeddings directly
curl http://ollama:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "your text here"
}'
```

This would eliminate the need for PyTorch and sentence-transformers entirely (~1.3 GB savings).

### Option 6: Layer Caching Optimization
Already implemented in the Dockerfile:
- ✅ Copy requirements.txt before source code
- ✅ Install dependencies in separate layer
- ✅ Copy source code last (changes most frequently)

---

## 🧪 Testing After Optimization

### 1. Verify functionality
```bash
# Check if the container starts
docker-compose logs assistant

# Test API endpoints
curl http://localhost:8080/api/health
curl http://localhost:8080/api/models
```

### 2. Monitor resource usage
```bash
# Check container stats
docker stats ai-assistant

# Expected memory usage: 500MB-1GB (depending on loaded models)
```

### 3. Test embedding generation
```bash
# Verify sentence-transformers still works with CPU-only PyTorch
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test search", "repo_path": "/repos"}'
```

---

## 📈 Monitoring Image Size Over Time

### Create a baseline
```bash
# Save current image size
docker images ai-assistant:latest --format "{{.Size}}" > .docker-size-baseline
```

### Compare after changes
```bash
# Compare with baseline
echo "Baseline: $(cat .docker-size-baseline)"
echo "Current:  $(docker images ai-assistant:latest --format '{{.Size}}')"
```

---

## 🎓 Best Practices Going Forward

1. **Always use multi-stage builds** for compiled dependencies
2. **Use --no-cache-dir** with pip to avoid caching packages
3. **Clean up apt cache** with `rm -rf /var/lib/apt/lists/*`
4. **Minimize layers** by combining RUN commands
5. **Use .dockerignore** aggressively
6. **Consider CPU-only versions** of ML libraries when GPU isn't needed
7. **Review dependencies regularly** - remove unused packages

---

## 🔧 Troubleshooting

### Issue: Build fails with CPU-only PyTorch
**Solution**: Some packages might require specific PyTorch versions. Revert to:
```txt
torch>=2.0.0
```

### Issue: sentence-transformers doesn't work
**Solution**: Ensure you're using a compatible version:
```bash
pip install sentence-transformers --no-deps
pip install torch transformers tokenizers
```

### Issue: libclang not found
**Solution**: Ensure libclang1 is installed in runtime stage:
```dockerfile
RUN apt-get install -y --no-install-recommends libclang1
```

---

## 📝 Summary

**Applied optimizations:**
1. ✅ Multi-stage Docker build
2. ✅ CPU-only PyTorch
3. ✅ Enhanced .dockerignore

**Expected reduction:** ~66% (from ~3.5 GB to ~1.2 GB)

**Next steps:**
1. Rebuild the image: `docker-compose build --no-cache`
2. Verify size: `docker images | grep ai-assistant`
3. Test functionality: `docker-compose up -d && docker-compose logs -f`

**Future consideration:**
- Replace sentence-transformers with Ollama embeddings API (saves another ~1.3 GB)

