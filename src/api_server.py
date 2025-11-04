#!/usr/bin/env python3
"""
FastAPI REST API Server for C++ AI Assistant.
Provides endpoints for code generation, model comparison, symbol search, and indexing.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import time
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import json
from pathlib import Path
import os

# Import our modules
from llm_client import (
    OllamaClient,
    AVAILABLE_LOCAL_MODELS,
    get_default_model_key,
    get_installed_models,
    get_all_models_with_status,
    get_downloaded_models,
    pull_model,
    remove_model
)
from vector_store import search_symbols as search_symbols_semantic, build_index
from database import (
    get_symbol_count,
    get_symbols_by_repo,
    get_database_stats,
    search_symbols as search_symbols_by_name,
    init_database,
    get_repositories,
    delete_symbols_by_repo
)
from symbol_extractor import index_repo
from smart_incremental_indexer import smart_index_repo
from constants import (
    COMPARISONS_DIR,
    OLLAMA_URL,
    ensure_directories
)

# ============================================================================
# GLOBAL STATE FOR INDEXING PROGRESS
# ============================================================================

# Dictionary to track indexing progress: {repo_name: {status, progress, message}}
indexing_progress = {}

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class CodeGenerationRequest(BaseModel):
    """Request schema for code generation."""
    prompt: str = Field(..., description="The code generation prompt", min_length=1)
    model: str = Field(default_factory=get_default_model_key, description="Model to use (defaults to first available model)")
    max_tokens: int = Field(default=2000, ge=100, le=8000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")

class CodeGenerationResponse(BaseModel):
    """Response schema for code generation."""
    success: bool
    model: str
    response: str  # Full markdown response from LLM
    generation_time: float
    error: Optional[str] = None

class SymbolSearchRequest(BaseModel):
    """Request schema for symbol search."""
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    repo: Optional[str] = Field(default=None, description="Filter by repository")

class IndexRepoRequest(BaseModel):
    """Request schema for repository indexing."""
    repo_path: str = Field(..., description="Path to repository (relative to /repos)")
    parallel: bool = Field(default=True, description="Use parallel processing")
    incremental: bool = Field(default=True, description="Use smart incremental indexing (only re-index changed files)")
    force_full: bool = Field(default=False, description="Force full re-index (ignore incremental logic)")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_available: bool
    database_stats: Dict[str, Any]
    available_models: List[str]

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="C++ AI Assistant API",
    description="""
## C++ AI Assistant REST API

AI-powered code generation, repository indexing, and symbol search for C++ projects.

### Features
- 🤖 **Code Generation**: Generate C++ code using local LLM models (streaming & non-streaming)
- 📚 **Repository Indexing**: Index C++ repositories for symbol search
- 🔍 **Symbol Search**: Search for functions, classes, and variables
- 📦 **Model Management**: Download and manage local LLM models

### Endpoint Categories
- **🟢 Public**: Used by frontend UI (14 endpoints)
- **🔴 Unused**: Available but not currently used (3 endpoints)

### Technology Stack
- **Backend**: FastAPI + Python 3.12
- **LLMs**: Ollama (DeepSeek Coder, CodeLlama, Qwen2.5-Coder, StarCoder2)
- **Storage**: SQLite + FAISS vector index
- **Frontend**: Vanilla JavaScript + marked.js + highlight.js
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "🟢 Frontend",
            "description": "**PUBLIC** - Serve frontend HTML and static files. Used by browser."
        },
        {
            "name": "🟢 Health & Status",
            "description": "**PUBLIC** - Health check endpoint. Used by frontend for status monitoring."
        },
        {
            "name": "🔴 API Info",
            "description": "**UNUSED** - API information endpoint. Not currently used but useful for API discovery."
        },
        {
            "name": "🟢 Model Management",
            "description": "**PUBLIC** - Download, list, and delete LLM models. Used by frontend Models tab."
        },
        {
            "name": "🔴 Model Discovery",
            "description": "**UNUSED** - List all available models. Frontend uses /models/downloaded instead."
        },
        {
            "name": "🟢 Code Generation",
            "description": "**PUBLIC** - Generate C++ code using LLM models. Used by frontend Generate tab."
        },
        {
            "name": "🟢 Symbol Search",
            "description": "**PUBLIC** - Search indexed symbols and get statistics. Used by frontend Index tab."
        },
        {
            "name": "🟢 Repository Management",
            "description": "**PUBLIC** - List and delete indexed repositories. Used by frontend Index tab."
        },
        {
            "name": "🟢 Indexing",
            "description": "**PUBLIC** - Index C++ repositories and stream progress. Used by frontend Index tab."
        },
        {
            "name": "🔴 Indexing (Polling)",
            "description": "**UNUSED** - Polling-based progress endpoint. Frontend uses SSE stream instead."
        }
    ]
)

# CORS middleware - allow all origins (configure as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist on startup
ensure_directories()

# ============================================================================
# STARTUP EVENT - AUTO-DOWNLOAD MODELS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    Initializes database and checks for required models.
    """
    print("\n" + "="*60)
    print("🚀 AI Assistant - Server Starting Up")
    print("="*60)

    # Initialize database
    if init_database():
        print("✅ Database initialized successfully")
    else:
        print("⚠️ Database initialization failed (will retry on first use)")

    print("\n📦 Model Configuration:")
    print(f"   Ollama URL: {OLLAMA_URL}")
    print(f"   Shared model storage: ./.llm_models")
    print(f"   Available models: {list(AVAILABLE_LOCAL_MODELS.keys())}")
    print(f"   ℹ️  Models in shared storage will be used automatically")
    print(f"   ℹ️  Download models using the 'Manage Models' tab in the UI")
    print("="*60)

    print("\n✅ Server startup complete!\n")

# ============================================================================
# STATIC FILES & FRONTEND
# ============================================================================

# Mount static files for frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["🟢 Frontend"],
    summary="Serve Frontend HTML",
    description="**PUBLIC** - Serves the main frontend HTML page. Used by browser on initial page load."
)
async def serve_frontend():
    """Serve the frontend HTML."""
    frontend_file = frontend_path / "index.html"
    if frontend_file.exists():
        return FileResponse(frontend_file)
    else:
        return HTMLResponse(content="<h1>C++ AI Assistant API</h1><p>Frontend not found. Visit <a href='/docs'>/docs</a> for API documentation.</p>")

# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get(
    "/api",
    response_model=Dict[str, str],
    tags=["🔴 API Info"],
    summary="API Information",
    description="**UNUSED** - Returns API metadata. Not currently used by frontend but useful for API discovery."
)
async def api_info():
    """API information endpoint."""
    return {
        "name": "C++ AI Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["🟢 Health & Status"],
    summary="Health Check",
    description="**PUBLIC** - Verify all services are running. Used by frontend for status monitoring."
)
async def health_check():
    """Health check endpoint - verify all services are running."""
    try:
        # Check Ollama availability across all instances
        import requests
        ollama_available = True

        # Check Ollama instance
        try:
            print(f"Calling Ollama at {OLLAMA_URL}/api/tags with timeout 2s...")
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if response.status_code != 200:
                print(f"⚠️  Ollama is not responding")
                ollama_available = False
        except Exception as e:
            print(f"⚠️  Ollama is not available: {e}")
            ollama_available = False

        # Get database stats
        db_stats = get_database_stats()

        # Get available models
        available_models = list(AVAILABLE_LOCAL_MODELS.keys())

        return HealthResponse(
            status="healthy" if ollama_available else "degraded",
            ollama_available=ollama_available,
            database_stats=db_stats,
            available_models=available_models
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get(
    "/models",
    response_model=Dict[str, Any],
    tags=["🔴 Model Discovery"],
    summary="List All Models",
    description="""**UNUSED** - List all available models with metadata and download status.

Frontend uses `/models/downloaded` instead. This endpoint is useful for:
- Discovering all available models (not just downloaded)
- Getting full model metadata
- API clients that need complete model list
    """
)
async def list_models():
    """
    List all available models with their metadata and download status.

    Returns a list of models with:
    - key: Model identifier
    - name: Full model name
    - description: Model description
    - size: Model size
    - license: Model license
    - downloaded: Whether the model is currently downloaded
    """
    try:
        models_with_status = get_all_models_with_status()
        downloaded_count = sum(1 for m in models_with_status if m["downloaded"])

        return {
            "models": models_with_status,
            "total": len(models_with_status),
            "downloaded": downloaded_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@app.get(
    "/models/downloaded",
    response_model=Dict[str, Any],
    tags=["🟢 Model Management"],
    summary="List Downloaded Models",
    description="**PUBLIC** - List only downloaded models. Used by frontend Models tab to populate model dropdown."
)
async def list_downloaded_models(no_cache: bool = False, metadata: bool = False):
    """
    List only the models that are currently downloaded.

    Args:
        no_cache: If True, bypass cache and fetch fresh data
        metadata: If True, include model metadata (name, description)

    Returns model keys (or model objects if metadata=true) that are available for use.
    """
    try:
        downloaded = get_downloaded_models(use_cache=not no_cache, include_metadata=metadata)
        return {
            "models": downloaded,
            "count": len(downloaded)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get downloaded models: {str(e)}")


@app.post(
    "/models/{model_key}/download",
    response_model=Dict[str, Any],
    tags=["🟢 Model Management"],
    summary="Download Model",
    description="**PUBLIC** - Download a specific model. Used by frontend Models tab when user clicks download button."
)
async def download_model(model_key: str, background_tasks: BackgroundTasks):
    """
    Download a specific model.

    Args:
        model_key: The model key from AVAILABLE_LOCAL_MODELS

    Returns:
        Status of the download operation
    """
    if model_key not in AVAILABLE_LOCAL_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")

    # Check if already downloaded
    downloaded = get_downloaded_models()
    if model_key in downloaded:
        return {
            "status": "already_downloaded",
            "model": model_key,
            "message": f"Model '{model_key}' is already downloaded"
        }

    # Start download in background
    def download_task():
        success = pull_model(model_key)
        if success:
            print(f"✅ Background download completed: {model_key}")
        else:
            print(f"❌ Background download failed: {model_key}")

    background_tasks.add_task(download_task)

    return {
        "status": "downloading",
        "model": model_key,
        "message": f"Download started for '{model_key}'. Check logs for progress."
    }


@app.delete(
    "/models/{model_key}",
    response_model=Dict[str, Any],
    tags=["🟢 Model Management"],
    summary="Delete Model",
    description="**PUBLIC** - Remove/delete a specific model. Used by frontend Models tab when user clicks delete button."
)
async def delete_model(model_key: str):
    """
    Remove/delete a specific model.

    Args:
        model_key: The model key from AVAILABLE_LOCAL_MODELS

    Returns:
        Status of the removal operation
    """
    if model_key not in AVAILABLE_LOCAL_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")

    # Check if model is downloaded
    downloaded = get_downloaded_models()
    if model_key not in downloaded:
        return {
            "status": "not_downloaded",
            "model": model_key,
            "message": f"Model '{model_key}' is not downloaded"
        }

    try:
        success = remove_model(model_key)
        if success:
            return {
                "status": "removed",
                "model": model_key,
                "message": f"Model '{model_key}' removed successfully"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to remove model '{model_key}'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing model: {str(e)}")


# ============================================================================
# INTELLIGENT MODEL SELECTION ENDPOINTS
# ============================================================================

# ============================================================================
# CODE GENERATION ENDPOINTS
# ============================================================================

@app.post(
    "/generate",
    response_model=CodeGenerationResponse,
    tags=["🟢 Code Generation"],
    summary="Generate Code (Non-Streaming)",
    description="""**PUBLIC** - Generate C++ code using a single model (non-streaming).

Used by frontend Generate tab when streaming is disabled. Returns complete response at once.
For real-time streaming, use `/generate/stream` instead.
    """
)
async def generate_code(request: CodeGenerationRequest):
    """
    Generate C++ code using a single model.

    Example:
        POST /generate
        {
            "prompt": "Write a C++ function to reverse a string",
            "model": "deepseek-coder",
            "max_tokens": 2000,
            "temperature": 0.2
        }
    """
    try:
        # Validate model exists
        if request.model not in AVAILABLE_LOCAL_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available: {list(AVAILABLE_LOCAL_MODELS.keys())}"
            )

        # Check if model is downloaded
        downloaded = get_downloaded_models()
        if request.model not in downloaded:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not downloaded. Please download it first from the Manage Models tab."
            )

        # Generate code
        client = OllamaClient(AVAILABLE_LOCAL_MODELS[request.model]["name"])

        import time
        start_time = time.time()
        raw_response = client.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        generation_time = time.time() - start_time

        if not raw_response:
            return CodeGenerationResponse(
                success=False,
                model=request.model,
                response="",
                generation_time=generation_time,
                error="Model failed to generate code"
            )

        # Backend only generates - frontend handles formatting/rendering
        return CodeGenerationResponse(
            success=True,
            model=request.model,
            response=raw_response,
            generation_time=generation_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")


@app.post(
    "/generate/stream",
    tags=["🟢 Code Generation"],
    summary="Generate Code (Streaming)",
    description="""**PUBLIC** - Generate C++ code with real-time streaming using Server-Sent Events (SSE).

Used by frontend Generate tab when streaming is enabled. Tokens are sent as they're generated,
providing real-time feedback to the user. Frontend renders markdown incrementally every 100ms.
    """
)
async def generate_code_stream(request: CodeGenerationRequest):
    """
    Generate C++ code using a single model with streaming enabled.
    Returns Server-Sent Events (SSE) stream of tokens as they're generated.

    Example:
        POST /generate/stream
        {
            "prompt": "Write a C++ function to reverse a string",
            "model": "deepseek-coder",
            "max_tokens": 2000,
            "temperature": 0.2
        }

    Response: Server-Sent Events stream
        data: {"token": "void", "done": false}
        data: {"token": " reverse", "done": false}
        ...
        data: {"done": true, "total_time": 45.2}
    """
    try:
        # Validate model exists
        if request.model not in AVAILABLE_LOCAL_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available: {list(AVAILABLE_LOCAL_MODELS.keys())}"
            )

        # Check if model is downloaded
        downloaded = get_downloaded_models()
        if request.model not in downloaded:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is not downloaded. Please download it first from the Manage Models tab."
            )

        # Get model info
        model_info = AVAILABLE_LOCAL_MODELS[request.model]
        model_name = model_info["name"]

        # Create client (uses default OLLAMA_URL)
        client = OllamaClient(model_name)

        # Generator function for SSE
        async def event_generator():
            """Generate Server-Sent Events from the streaming response."""
            start_time = time.time()
            full_response = ""

            try:
                # Send initial event
                print(f"[SSE] Sending start event for model {request.model}")
                yield f"data: {json.dumps({'status': 'started', 'model': request.model})}\n\n"

                # Stream tokens from Ollama
                token_count = 0
                for chunk in client.generate_stream(
                    request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                ):
                    # Check for errors
                    if "error" in chunk:
                        print(f"[SSE ERROR] {chunk['error']}")
                        yield f"data: {json.dumps({'error': chunk['error'], 'done': True})}\n\n"
                        return

                    # Send token if present
                    if "response" in chunk:
                        token = chunk["response"]
                        full_response += token
                        token_count += 1
                        yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

                    # Check if done
                    if chunk.get("done", False):
                        elapsed_time = time.time() - start_time
                        print(f"[SSE] Stream complete: {token_count} tokens in {elapsed_time:.2f}s")

                        # Send completion event with metadata
                        # Frontend will render markdown, no need to extract code
                        completion_data = {
                            "done": True,
                            "total_time": elapsed_time,
                            "model": request.model,
                            "response": full_response
                        }
                        yield f"data: {json.dumps(completion_data)}\n\n"
                        return

            except Exception as e:
                print(f"[SSE EXCEPTION] {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

        # Return streaming response
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable buffering in nginx
                "Transfer-Encoding": "chunked"  # Enable chunked transfer
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SYMBOL SEARCH ENDPOINTS
# ============================================================================

@app.get(
    "/symbols/stats",
    response_model=Dict[str, Any],
    tags=["🟢 Symbol Search"],
    summary="Get Symbol Statistics",
    description="**PUBLIC** - Get statistics about indexed symbols. Used by frontend Index tab to display stats."
)
async def get_symbol_stats():
    """Get statistics about indexed symbols."""
    try:
        stats = get_database_stats()

        # If stats are empty, database might not be initialized
        if not stats:
            print("Database not initialized, attempting to initialize...")
            if init_database():
                stats = get_database_stats()

        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get(
    "/symbols/repo/{repo_name}",
    response_model=Dict[str, Any],
    tags=["🟢 Symbol Search"],
    summary="Get Repository Symbols",
    description="**PUBLIC** - Get all symbols from a specific repository. Used by frontend when viewing repository symbols."
)
async def get_repo_symbols(repo_name: str, limit: int = Query(100, ge=1, le=1000)):
    """Get all symbols from a specific repository."""
    try:
        symbols = get_symbols_by_repo(repo_name, limit=limit)
        return {
            "success": True,
            "repo": repo_name,
            "count": len(symbols),
            "symbols": symbols
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get symbols: {str(e)}")

@app.get(
    "/repositories",
    response_model=Dict[str, Any],
    tags=["🟢 Repository Management"],
    summary="List Repositories",
    description="**PUBLIC** - Get list of all indexed repositories. Used by frontend Index tab to display repository list."
)
async def list_repositories():
    """
    Get list of all indexed repositories with their statistics.

    Returns:
        List of repositories with symbol count, file count, and last indexed time
    """
    try:
        repositories = get_repositories()
        return {
            "success": True,
            "count": len(repositories),
            "repositories": repositories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get repositories: {str(e)}")

@app.delete(
    "/repositories/{repo_name}",
    response_model=Dict[str, Any],
    tags=["🟢 Repository Management"],
    summary="Delete Repository",
    description="**PUBLIC** - Delete an indexed repository from database and FAISS index. Used by frontend Index tab when user clicks delete button."
)
async def delete_repository(repo_name: str):
    """
    Delete an indexed repository from database and FAISS index.

    Args:
        repo_name: Name of the repository to delete

    Returns:
        Success status and number of symbols deleted
    """
    try:
        # Delete from database
        deleted_count = delete_symbols_by_repo(repo_name)

        # Rebuild FAISS index to remove deleted symbols
        try:
            build_index()
            index_rebuilt = True
        except Exception as e:
            print(f"Warning: Failed to rebuild FAISS index after deletion: {e}")
            index_rebuilt = False

        return {
            "success": True,
            "message": f"Repository '{repo_name}' deleted",
            "symbols_deleted": deleted_count,
            "index_rebuilt": index_rebuilt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete repository: {str(e)}")

# ============================================================================
# INDEXING ENDPOINTS
# ============================================================================

@app.post(
    "/index",
    response_model=Dict[str, Any],
    tags=["🟢 Indexing"],
    summary="Index Repository",
    description="""**PUBLIC** - Index a C++ repository (extract symbols and build vector index).

Used by frontend Index tab when user clicks "Index Repository" button. This is a long-running
background operation. Use `/index/progress/{repo_name}/stream` to monitor progress in real-time.

By default, uses smart incremental indexing (git-aware) that only re-indexes changed files.
    """
)
async def index_repository(request: IndexRepoRequest, background_tasks: BackgroundTasks):
    """
    Index a C++ repository (extract symbols and build vector index).
    This is a long-running operation, so it runs in the background.

    By default, uses smart incremental indexing that only re-indexes changed files.
    Set force_full=true to force a complete re-index.

    Example (incremental):
        POST /index
        {
            "repo_path": "my_project",
            "parallel": true,
            "incremental": true
        }

    Example (force full re-index):
        POST /index
        {
            "repo_path": "my_project",
            "parallel": true,
            "force_full": true
        }
    """
    try:
        # Validate repo path
        from constants import CONTAINER_REPOS_DIR
        full_path = os.path.join(CONTAINER_REPOS_DIR, request.repo_path)
        if not os.path.exists(full_path):
            raise HTTPException(
                status_code=404,
                detail=f"Repository not found: {request.repo_path}"
            )

        # Initialize progress tracking
        repo_name = request.repo_path
        mode = "FULL RE-INDEX" if request.force_full else ("INCREMENTAL" if request.incremental else "STANDARD")
        indexing_progress[repo_name] = {
            "status": "running",
            "progress": 0,
            "message": f"Starting {mode} indexing...",
            "start_time": time.time(),
            "mode": mode
        }

        # Run indexing in background
        def index_task():
            try:
                # Update progress: Indexing symbols
                indexing_progress[repo_name].update({
                    "progress": 10,
                    "message": "Analyzing repository and extracting symbols..."
                })

                # Choose indexing strategy
                if request.incremental:
                    # Smart incremental indexing (git-aware)
                    smart_index_repo(
                        full_path,
                        force_full=request.force_full,
                        parallel=request.parallel,
                        rebuild_vectors=True
                    )
                else:
                    # Legacy full indexing
                    index_repo(full_path, parallel=request.parallel)

                    indexing_progress[repo_name].update({
                        "progress": 70,
                        "message": "Symbol extraction complete. Building vector index..."
                    })

                    # Rebuild vector index
                    build_index()

                # Mark as complete
                elapsed = time.time() - indexing_progress[repo_name]["start_time"]
                indexing_progress[repo_name].update({
                    "status": "complete",
                    "progress": 100,
                    "message": f"{mode} indexing complete in {elapsed:.1f}s",
                    "elapsed_time": elapsed
                })
            except Exception as e:
                print(f"Background indexing failed: {e}")
                import traceback
                traceback.print_exc()
                indexing_progress[repo_name].update({
                    "status": "error",
                    "progress": 0,
                    "message": f"Error: {str(e)}"
                })

        background_tasks.add_task(index_task)

        return {
            "success": True,
            "message": f"{mode} indexing started for {request.repo_path}",
            "repo_path": request.repo_path,
            "status": "running",
            "mode": mode
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start indexing: {str(e)}")

@app.get(
    "/index/progress/{repo_name}",
    response_model=Dict[str, Any],
    tags=["🔴 Indexing (Polling)"],
    summary="Get Indexing Progress (Polling)",
    description="""**UNUSED** - Get current indexing progress using polling.

Frontend uses `/index/progress/{repo_name}/stream` (SSE) instead for real-time updates.
This endpoint is useful for clients that don't support Server-Sent Events.
    """
)
async def get_indexing_progress(repo_name: str):
    """
    Get the current indexing progress for a repository.

    Args:
        repo_name: Name of the repository being indexed

    Returns:
        Progress information including status, progress percentage, and message
    """
    if repo_name not in indexing_progress:
        return {
            "success": False,
            "message": "No indexing in progress for this repository"
        }

    progress_info = indexing_progress[repo_name].copy()
    return {
        "success": True,
        **progress_info
    }

@app.get(
    "/index/progress/{repo_name}/stream",
    tags=["🟢 Indexing"],
    summary="Stream Indexing Progress (SSE)",
    description="""**PUBLIC** - Stream real-time indexing progress using Server-Sent Events (SSE).

Used by frontend Index tab to show real-time progress updates during repository indexing.
Updates are sent every 500ms with status, progress percentage, and current message.
    """
)
async def stream_indexing_progress(repo_name: str):
    """
    Stream real-time indexing progress updates using Server-Sent Events (SSE).

    Args:
        repo_name: Name of the repository being indexed

    Returns:
        SSE stream of progress updates
    """
    async def event_generator():
        """Generate SSE events for indexing progress."""
        last_progress = -1

        while True:
            if repo_name not in indexing_progress:
                # No indexing started yet
                yield f"data: {json.dumps({'status': 'waiting', 'message': 'Waiting for indexing to start...'})}\n\n"
                await asyncio.sleep(0.5)
                continue

            progress_info = indexing_progress[repo_name].copy()
            current_progress = progress_info.get("progress", 0)

            # Send update if progress changed or status changed
            if current_progress != last_progress or progress_info.get("status") in ["complete", "error"]:
                yield f"data: {json.dumps(progress_info)}\n\n"
                last_progress = current_progress

            # Stop streaming if complete or error
            if progress_info.get("status") in ["complete", "error"]:
                break

            await asyncio.sleep(0.5)  # Poll every 500ms

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    import sys

    # Get port from environment or command line
    port = int(os.getenv("API_PORT", "8080"))
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    print(f"""
    ==========================================
    🚀 C++ AI Assistant - Full Stack Server
    ==========================================

    🌐 Web Application: http://localhost:{port}

    📱 Features:
       ✅ Generate Code - Single or multi-model code generation with smart merge
       ✅ Index Repository - Index C++ repositories for symbol search
       ✅ Manage Models - Download/remove Ollama models

    ⚙️  Technology Stack:
       • Frontend: Vanilla JS + HTML5 + CSS3
       • Backend: FastAPI (Python)
       • LLM Engine: Ollama (local models)
       • Code Analysis: libclang + FAISS

    🔧 Advanced Users (API available at):
       • Swagger UI: http://localhost:{port}/docs

    ==========================================
    """)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

