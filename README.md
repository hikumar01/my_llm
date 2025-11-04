# AI Code Assistant

A full-stack web application for AI-powered code generation and analysis using local LLM models.

## Features

- **Modern Web Interface**: Responsive single-page app with dark/light theme
- **Multi-Model Support**: 4 local LLM models (DeepSeek, CodeLlama, Qwen, StarCoder)
- **Model Management**: Download/remove models directly from the UI
- **Semantic Search**: FAISS-based vector search across code repositories
- **Symbol Extraction**: Extract and index code symbols using libclang
- **Real-time Streaming**: Server-Sent Events for live progress updates
- **Containerized**: Docker ready with volume persistence
- **100% Free**: All models run locally, no API keys needed

## Quick Start

```bash
# Start containers
docker-compose up -d

# Access the application
open http://localhost:8080
```

**First-time startup**: Models (~16 GB) will download automatically (10-30 minutes).

### Web Interface Tabs
- **Generate Code** - AI code generation with streaming support
- **Search Symbols** - Search indexed code repositories
- **Index Repository** - Index new repositories with smart incremental updates
- **Manage Models** - Download/remove models

### API Documentation
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## Usage Guide

### Generate Code
1. Enter a prompt (e.g., "Write a function to reverse a string")
2. Select a model from dropdown
3. Adjust **Max Tokens** (100-8000) and **Temperature** (0.0-2.0)
   - Recommended for code: Temperature 0.1-0.3, Max Tokens 2000
4. Toggle **Enable Streaming** for real-time token generation
5. Click "Generate Code" or press `Ctrl+Enter`

### Search Symbols
- Semantic search across indexed code repositories
- Find functions, classes, variables by description
- Filter by repository

### Index Repository
- Index code repositories for search
- Supports parallel indexing and smart incremental updates (git-aware)
- Real-time progress tracking via Server-Sent Events

### Manage Models
- View all available models with download status
- Download/remove models from UI
- See model size, license, and description

## Architecture

**Frontend**: Vanilla JavaScript (index.html, styles.css, app.js)
**Backend**: Python FastAPI (api_server.py, llm_client.py, vector_store.py, database.py)
**LLM Engine**: Ollama (runs in separate container)
**Storage**: SQLite (symbols), FAISS (vector search)
**Indexing**: Smart incremental indexer (git-aware, only processes changed files)

```
Browser (8080) → FastAPI Server → Ollama Container → Local Models
                      ↓
                SQLite + FAISS
```

### Docker Architecture

- **Container 1 (ollama)**: Ollama LLM server
  - Port: 11434
  - Volume: `.llm_models` → `/root/.ollama` (model cache)

- **Container 2 (ai-assistant)**: FastAPI application
  - Port: 8080
  - Volumes:
    - `assistant_data` → `/app/data` (persistent data: database, FAISS index, logs)
    - `~/src` → `/repos` (read-only: source code repositories)
    - `./src` → `/app/src` (dev mode: live code updates)
    - `./frontend` → `/app/frontend` (dev mode: live frontend updates)

## API Endpoints

**Base URL**: `http://localhost:8080`

**Total**: 14 active endpoints (all used by frontend)

### Endpoints by Category

```bash
# Frontend (1)
GET  /                                      # Serve frontend HTML

# Health & Status (1)
GET  /health                                # Health check and system status

# Model Management (4)
GET    /models                              # List all models with download status
GET    /models/downloaded                   # List only downloaded models
POST   /models/{model_key}/download         # Download a model
DELETE /models/{model_key}                  # Delete a downloaded model

# Code Generation (2)
POST /generate                              # Non-streaming code generation
POST /generate/stream                       # Streaming code generation (SSE)

# Symbol Search (2)
GET  /symbols/stats                         # Get symbol statistics
GET  /symbols/repo/{repo_name}              # Get symbols from repository

# Repository Management (2)
GET    /repositories                        # List all indexed repositories
DELETE /repositories/{repo_name}            # Delete repository from index

# Indexing (2)
POST /index                                 # Start repository indexing
GET  /index/progress/{repo_name}/stream     # Stream indexing progress (SSE)
```

**Full API Documentation**: http://localhost:8080/docs

## Supported Models

| Model | Size | License | Best For |
|-------|------|---------|----------|
| **DeepSeek Coder** | 3.8 GB | MIT | General code generation (recommended) |
| **CodeLlama** | 3.8 GB | Llama 2 | Meta's code specialist |
| **Qwen2.5 Coder** | 4.7 GB | Apache 2.0 | Fast & efficient |
| **StarCoder2** | 4.0 GB | BigCode OpenRAIL-M | Multi-language support |

**Total**: ~16 GB for all models
**Storage**: Models cached in `.llm_models/` (persists across restarts)
**Download**: Automatic on first startup (configurable via `.env`)

## Configuration

Edit `.env` file to customize:

```bash
# Docker Configuration
COMPOSE_PROJECT_NAME=ai-assistant        # Docker Compose project name
HOST_REPOS_DIR=~/src                     # Source code repositories directory

# Model Configuration
ENABLED_MODELS=all                       # or comma-separated: deepseek-coder,codellama
AUTO_PULL_MODELS=true                    # Auto-download models on startup

# Performance Tuning (optional)
SYMBOL_BATCH_SIZE=100                    # Symbols per database batch
EMBEDDING_BATCH_SIZE=32                  # Symbols per embedding batch
SYMBOL_EXTRACTOR_WORKERS=4               # Parallel extraction workers
WATCH_DEBOUNCE_SECONDS=2.0               # File change debounce delay
```

### Data Persistence

| Location | Type | Purpose |
|----------|------|---------|
| `.llm_models/` | Host directory | Ollama model cache (~16 GB) |
| `assistant_data` | Docker volume | Database, FAISS index, logs |
| `~/src/` | Host directory | Source code repositories (read-only) |

**Note**: The `assistant_data` Docker volume persists across container restarts and is mounted at `/app/data` inside the container.

## Troubleshooting

```bash
# Check running containers
docker ps

# View logs
docker-compose logs -f

# Check Ollama models
docker exec -it ollama ollama list

# Restart services
docker-compose restart

# Clean restart (removes volumes)
docker-compose down -v
docker-compose up -d --build

# Permission errors
chmod 777 .llm_models

# Port already in use
# Edit docker-compose.yml: ports: - "8081:8080"
```

## Best Practices

**Temperature Settings**:
- Production code: 0.1-0.2 (deterministic, consistent)
- Exploring solutions: 0.4-0.6 (more creative)
- Recommended: 0.2

**Model Selection**:
- **DeepSeek Coder**: Best all-around (recommended)
- **CodeLlama**: Standard algorithms and patterns
- **Qwen2.5**: Fastest inference speed
- **StarCoder2**: Complex multi-file projects

**Indexing**:
- Use smart incremental indexing (default) for faster updates
- Force full re-index only when necessary
- Index repositories before using symbol search

---

## Get Started

```bash
# Clone and start
git clone <your-repo>
cd my_llm
docker-compose up -d

# Access the application
open http://localhost:8080
```

**What you get:**
- Zero-config startup
- Auto model downloads
- Modern web interface
- 14 REST API endpoints
- 4 local LLM models
- 100% free & local
- No API keys needed
