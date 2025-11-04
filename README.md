# C++ AI Assistant - Full Stack Web Application

A modern full-stack web application for AI-powered C++ code generation and analysis, featuring multiple free local LLM models, semantic code search, and an intuitive web interface.

## 🎯 Features

- **🎨 Modern Web Interface**: Responsive single-page app with dark/light theme
- **🤖 Multi-Model Support**: 4 local LLM models (DeepSeek, CodeLlama, Qwen, StarCoder)
- **⚖️ Model Comparison**: Compare outputs from multiple models side-by-side
- **📥 Model Management**: Download/remove models directly from the UI
- **🔍 Semantic Search**: FAISS-based vector search across C++ codebases
- **📊 Symbol Extraction**: Extract and index C++ symbols using libclang
- **⚡ Real-time Streaming**: WebSocket-based live code generation
- **🎛️ Interactive Controls**: Sliders for temperature and max tokens with helpful hints
- **🔄 Progress Indicators**: Real-time feedback during code generation
- **🐳 Containerized**: Docker/Podman ready with volume persistence
- **🆓 100% Free**: All models run locally, no API keys needed

## 🚀 Quick Start

### 1. Clone and Navigate

```bash
cd /path/to/llm-cpp
```

### 2. Start Containers

```bash
# Using Podman
podman-compose up -d

# Or using Docker
docker-compose up -d
```

**First-time startup**: Models (~16 GB) will download automatically. This takes 10-30 minutes depending on your internet speed.

### 3. Access the Application

**🌐 Open in your browser**: http://localhost:8080

That's it! You'll see the web interface with 5 tabs:
- **Generate Code** - Single model code generation
- **Compare Models** - Side-by-side comparison
- **Search Symbols** - Search your C++ codebase
- **Index Repository** - Index new repositories
- **Manage Models** - Download/remove models

### 4. Verify Everything Works

Open http://localhost:8080 and you should see:
- ✅ Green "Healthy" status indicator
- ✅ Model dropdown populated with downloaded models
- ✅ All 5 tabs accessible

**Advanced users** can access API docs at:
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc
```

---

## 📖 Web Interface Guide

### **Main Features**

#### 1. **Generate Code** Tab
- Enter a prompt (e.g., "Write a function to reverse a string")
- Select a model from dropdown (only downloaded models shown)
- **Max Tokens Slider**: Control output length (100-8000)
  - 100 = Short snippets
  - 2000 = Balanced (default, recommended)
  - 8000 = Comprehensive with examples
- **Temperature Slider**: Control creativity (0.0-2.0)
  - 0.1-0.3 = Deterministic (recommended for C++)
  - 0.5-0.7 = Balanced
  - 1.0+ = Creative/experimental
- Click "Generate Code" or press `Ctrl+Enter`
- Watch real-time progress indicator
- View syntax-highlighted code with clang-tidy warnings

#### 2. **Compare Models** Tab
- Select multiple models to compare (2-4 recommended)
- Configure max tokens and temperature (same for all models for fair comparison)
- Click "Compare Models"
- Watch progress for each model independently
- View side-by-side comparison cards showing:
  - Generated code with syntax highlighting
  - Generation time
  - Code length
  - Model description
  - Error/timeout indicators

#### 3. **Search Symbols** Tab
- Semantic search across indexed C++ codebases
- Find functions, classes, variables by description
- Filter by repository
- View file location and signatures

#### 4. **Index Repository** Tab
- Index C++ repositories for search
- Background processing with progress
- Parallel indexing support

#### 5. **Manage Models** Tab ⭐ NEW
- View all available models in table
- See download status (green = downloaded)
- Download/remove models from UI
- Bulk operations support

### **UI Features**

- 🌓 **Dark/Light Theme**: Toggle in header
- 📱 **Responsive**: Works on mobile devices
- ⚡ **Real-time Status**: Connection indicator in header
- 💾 **Persistent Settings**: Theme saved to localStorage
- 🎛️ **Interactive Sliders**: Visual controls for temperature and max tokens
- 📊 **Progress Indicators**: Real-time feedback during generation
- 🔄 **Auto-refresh**: Model dropdown updates after downloads

---

## 🏗️ Architecture

### **Technology Stack**

**Frontend** (Vanilla JavaScript - No frameworks!)
- `index.html` - Single-page application structure
- `styles.css` - Modern CSS with variables, animations
- `app.js` - Event handling, API calls, WebSocket client

**Backend** (Python FastAPI)
- `api_server.py` - REST API + WebSocket server
- `llm_client.py` - Ollama integration
- `vector_store.py` - FAISS semantic search
- `database.py` - SQLite symbol storage

**LLM Engine** (Ollama)
- Runs in separate container
- Serves 4 local models
- Accessed via HTTP API

### **WebSocket Usage**

**Purpose**: Real-time streaming code generation (optional feature)

**Who uses it**:
- Frontend `app.js` can optionally use WebSocket for streaming
- Currently disabled by default (uses HTTP POST instead)
- Available at `ws://localhost:8080/ws/generate`

**Why it exists**:
- Allows character-by-character streaming (like ChatGPT)
- Better UX for long generations
- Can be enabled in future for live code streaming

**Current behavior**:
- Frontend uses HTTP POST `/generate` (simpler, more reliable)
- WebSocket endpoint available but not actively used
- Can be enabled by setting `useStream = true` in `app.js`

---

## 🔌 API Reference (Advanced Users)

This is a **web application**, not an API service. However, the REST API is available for advanced users who want to integrate with other tools.

### **Base URL**: `http://localhost:8080`

### **Key Endpoints**

#### Health & Status

```bash
# Health check
GET /health

# List all models with download status
GET /models

# List only downloaded models
GET /models/downloaded
```

#### Model Management

```bash
# Download a model
POST /models/{model_key}/download

# Remove a model
DELETE /models/{model_key}
```

#### Code Generation

```bash
# Generate code with single model
POST /generate
{
  "prompt": "Write a C++ function to reverse a string",
  "model": "deepseek-coder",
  "temperature": 0.2,
  "max_tokens": 2000,
  "format_code": true,
  "check_code": true
}

# Compare multiple models
POST /compare
{
  "prompt": "Write a binary search function",
  "models": ["deepseek-coder", "codellama"],
  "parallel": true
}
```

#### Symbol Search

```bash
# Search symbols
POST /search
{
  "query": "function to parse JSON",
  "top_k": 10,
  "repo_filter": "my-project"
}
```

#### Repository Indexing

```bash
# Index a repository
POST /index
{
  "repo_path": "my-cpp-project",
  "parallel": true,
  "rebuild_vectors": true
}

# Get database statistics
GET /stats
```

**📚 Full API Documentation**:
- Interactive Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

**Note**: Most users should use the web interface at http://localhost:8080 instead of calling the API directly.

---

## 🤖 Supported Models

| Model | Size | License | Best For |
|-------|------|---------|----------|
| **DeepSeek Coder** | 3.8 GB | MIT | General code generation ⭐ |
| **CodeLlama** | 3.8 GB | Llama 2 | Meta's code specialist |
| **Qwen2.5 Coder** | 4.7 GB | Apache 2.0 | Fast & efficient |
| **StarCoder2** | 4.0 GB | BigCode OpenRAIL-M | Multi-language support |

**Total**: ~16 GB for all models

### **Model Management**

Models are automatically downloaded on first startup. You can also:

1. **Via Web UI**: Go to "Manage Models" tab
2. **Via Script**: Run `./setup_models.sh`
3. **Via API**: `POST /models/{model_key}/download`

Models are cached in `.llm_models/` and persist across container restarts.

---

## ⚙️ Configuration

### **Environment Variables**

Edit `.env` file to customize:

```bash
# Model Configuration
ENABLED_MODELS=all  # or comma-separated: deepseek-coder,codellama
AUTO_PULL_MODELS=true  # Auto-download models on startup

# Ollama Configuration
OLLAMA_URL=http://ollama:11434

# Performance Tuning (optional)
SYMBOL_BATCH_SIZE=100
EMBEDDING_BATCH_SIZE=32
SYMBOL_EXTRACTOR_WORKERS=4
```

### **Multi-Directory Support**

Mount multiple source directories:

```yaml
# docker-compose.yml
volumes:
  - ~/src:/repos/src1
  - ~/projects:/repos/src2
  - ~/work:/repos/src3
```

---

## 🏗️ Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│  Web Browser        │  HTTP   │  Assistant          │
│  (Port 8080)        │ ──────> │  Container          │
│                     │         │  - FastAPI Server   │
│  - Modern UI        │         │  - Python Code      │
│  - 5 Tabs           │         │  - SQLite DB        │
│  - Model Mgmt       │         │  - FAISS Index      │
└─────────────────────┘         └──────────┬──────────┘
                                           │ HTTP API
                                           ▼
                                ┌─────────────────────┐
                                │  Ollama             │
                                │  Container          │
                                │  - Model Storage    │
                                │  - Inference Engine │
                                └──────────┬──────────┘
                                           │
                                           ▼
                                ┌─────────────────────┐
                                │  .llm_models/       │
                                │  (Host Storage)     │
                                │  ~16 GB cached      │
                                └─────────────────────┘
```

### **Data Persistence**

| Directory | Purpose | Size |
|-----------|---------|------|
| `.llm_models/` | Ollama model cache | ~16 GB |
| `.host_data/` | SQLite DB, FAISS index, comparisons | ~100 MB |
| `~/src/` | Your C++ source code (read-only) | Variable |

---

## 🧪 Testing

See **TESTING.md** for comprehensive testing procedures including:
- 20+ test scenarios
- API endpoint testing
- Model comparison testing
- Performance benchmarks
- Troubleshooting guide

---

## 📚 Additional Documentation

- **TESTING.md** - Comprehensive testing guide with 20+ test procedures
- **QUICK_REFERENCE.md** - Fast reference for common tasks and configurations

---

## 🛠️ Troubleshooting

### **Models not showing as downloaded**

```bash
# Check Ollama has models
podman exec -it llm-cpp-ollama-1 ollama list

# Restart containers
podman-compose restart
```

### **Frontend not loading**

```bash
# Rebuild container (includes frontend files)
podman-compose build assistant
podman-compose up -d
```

### **Permission errors**

```bash
# Fix permissions
chmod 777 .host_data
chmod 777 .llm_models
```

### **Port already in use**

```bash
# Change port in docker-compose.yml
ports:
  - "8081:8080"  # Use 8081 instead
```

---

## 🎯 Best Practices

### **Temperature Settings for C++**

| Use Case | Temperature | Why |
|----------|-------------|-----|
| Production code | 0.1 - 0.2 | Deterministic, reliable |
| Bug fixes | 0.1 | Maximum consistency |
| Exploring solutions | 0.4 - 0.6 | See alternatives |
| Learning | 0.3 - 0.5 | Balance of standard + creative |

**Recommendation**: Use 0.2 for most C++ code generation.

### **Model Selection**

- **DeepSeek Coder**: Best all-around choice ⭐
- **CodeLlama**: Good for standard algorithms
- **Qwen2.5**: Fastest inference
- **StarCoder2**: Best for complex multi-file projects

---

## 📦 Project Structure

```
llm-cpp/
├── src/                    # Python source code
│   ├── api_server.py      # FastAPI server (854 lines)
│   ├── llm_client.py      # Ollama client (619 lines)
│   ├── vector_store.py    # FAISS search
│   ├── symbol_extractor.py # libclang parser
│   ├── database.py        # SQLite operations
│   └── ...
├── frontend/              # Web interface
│   ├── index.html        # Main HTML (338 lines)
│   ├── styles.css        # Styles (737 lines)
│   └── app.js            # JavaScript (774 lines)
├── examples/             # Example clients
│   ├── api_client.py     # Python client
│   └── api_examples.sh   # Bash examples
├── .env                  # Configuration
├── docker-compose.yml    # Container orchestration
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── setup_models.sh      # Manual model download script
├── README.md           # This file
├── TESTING.md          # Testing guide
└── QUICK_REFERENCE.md  # Quick reference
```

---

## 🚀 Summary

**What You Get:**

✅ **Zero-config startup** - Just run `podman-compose up -d`
✅ **Auto model downloads** - No manual setup required
✅ **Modern web interface** - Beautiful, responsive, intuitive
✅ **4 LLM models** - Compare and choose the best
✅ **Interactive controls** - Sliders for temperature and max tokens
✅ **Real-time feedback** - Progress indicators and streaming
✅ **100% local & free** - No API keys, no cloud dependencies
✅ **Production-ready** - Containerized, persistent, scalable

**Get started in 3 commands:**

```bash
cd /path/to/llm-cpp
podman-compose up -d
open http://localhost:8080
```

**That's it!** 🎉

