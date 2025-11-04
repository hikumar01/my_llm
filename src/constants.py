#!/usr/bin/env python3
"""
Centralized Configuration Constants for C++ AI Assistant
========================================================

This module contains ALL configuration constants used throughout the application.
It provides a single source of truth for all settings, with sensible defaults
and environment variable overrides.

Usage:
    from constants import OLLAMA_URL, SYMBOL_BATCH_SIZE, CPP_EXTENSIONS

Environment Variables:
    All constants can be overridden via environment variables.
    See individual constant documentation for details.
"""

import os
from typing import Set


# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# CONTAINER_REPOS_DIR: Directory where C++ source code repositories are mounted
# Default: /repos (read-only mount from host)
# Override: Set CONTAINER_REPOS_DIR environment variable
# Example: CONTAINER_REPOS_DIR=/custom/repos
CONTAINER_REPOS_DIR = os.getenv("CONTAINER_REPOS_DIR", "/repos")

# CONTAINER_HOST_DATA_DIR: Directory for persistent data (database, indices, logs)
# Default: /app/host_data (mounted from host .host_data directory)
# Override: Set CONTAINER_HOST_DATA_DIR environment variable
# Contains: symbol_db.sqlite, faiss_index/, comparisons/, assistant.log
CONTAINER_HOST_DATA_DIR = os.getenv("CONTAINER_HOST_DATA_DIR", "/app/host_data")

# CONTAINER_SRC_DIR: Directory containing Python source code
# Default: /app/src (copied during Docker build)
# Override: Set CONTAINER_SRC_DIR environment variable
# Note: Usually not changed unless using volume mount for development
CONTAINER_SRC_DIR = os.getenv("CONTAINER_SRC_DIR", "/app/src")

# COMPARISONS_DIR: Directory for storing model comparison results
# Default: /app/host_data/comparisons
# Override: Set COMPARISONS_DIR environment variable
# Structure: comparisons/<timestamp>/<model_name>/
COMPARISONS_DIR = os.getenv("COMPARISONS_DIR", "/app/host_data/comparisons")

# FAISS_INDEX_DIR: Directory for FAISS vector index files
# Default: /app/host_data/faiss_index
# Override: Set FAISS_INDEX_DIR environment variable
# Contains: index.faiss (vector index), meta.npy (metadata)
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "/app/host_data/faiss_index")

# SYMBOL_DB_PATH: Path to SQLite database storing extracted C++ symbols
# Default: /app/host_data/symbol_db.sqlite
# Override: Set SYMBOL_DB_PATH environment variable
# Schema: symbols(id, repo, file, symbol, kind, signature, doc, line, created_at)
SYMBOL_DB_PATH = os.getenv("SYMBOL_DB_PATH", "/app/host_data/symbol_db.sqlite")


# ============================================================================
# FILE EXTENSIONS
# ============================================================================

# CPP_EXTENSIONS: Set of C++ file extensions to process during symbol extraction
# Default: All common C++ source and header file extensions
# Note: This is a Python set for O(1) lookup performance
# Used by: symbol_extractor.py, git_incremental_indexer.py
CPP_EXTENSIONS: Set[str] = {'.cpp', '.cc', '.c', '.cxx', '.hpp', '.h', '.hxx', '.hh'}

# WATCH_EXTENSIONS: File extensions to monitor for changes in file watcher
# Default: Common C++ source and header extensions (comma-separated)
# Override: Set WATCH_EXTENSIONS environment variable
# Example: WATCH_EXTENSIONS=".cpp,.hpp,.h"
# Used by: fs_watcher.py for real-time file monitoring
WATCH_EXTENSIONS: Set[str] = set(
    os.getenv("WATCH_EXTENSIONS", ".cpp,.cc,.c,.hpp,.h,.cxx,.hxx").split(',')
)


# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

# SYMBOL_BATCH_SIZE: Number of symbols to batch before database insert
# Default: 100 symbols per batch
# Override: Set SYMBOL_BATCH_SIZE environment variable
# Impact: Higher = fewer DB writes (faster), more memory usage
# Recommendation: 50-200 for most systems, 500+ for high-memory systems
# Used by: symbol_extractor.py for batch database inserts
SYMBOL_BATCH_SIZE = int(os.getenv("SYMBOL_BATCH_SIZE", "100"))

# SYMBOL_EXTRACTOR_WORKERS: Number of parallel workers for symbol extraction
# Default: 4 workers
# Override: Set SYMBOL_EXTRACTOR_WORKERS environment variable
# Impact: Higher = faster extraction, more CPU/memory usage
# Recommendation: Set to number of CPU cores (typically 2-8)
# Used by: symbol_extractor.py for parallel file processing
SYMBOL_EXTRACTOR_WORKERS = int(os.getenv("SYMBOL_EXTRACTOR_WORKERS", "4"))

# EMBEDDING_BATCH_SIZE: Number of symbols to batch for embedding generation
# Default: 32 symbols per batch
# Override: Set EMBEDDING_BATCH_SIZE environment variable
# Impact: Higher = faster embedding, more GPU/memory usage
# Recommendation: 16-64 for CPU, 64-256 for GPU
# Used by: vector_store.py for batch embedding generation
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# EMBEDDING_DIM: Dimension of embedding vectors
# Default: 384 (matches all-MiniLM-L6-v2 model)
# Override: Set EMBEDDING_DIM environment variable
# Note: Must match the embedding model's output dimension
# Used by: vector_store.py for FAISS index initialization
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# EMBEDDING_MODEL: Sentence transformer model for code embeddings
# Default: all-MiniLM-L6-v2 (fast, 384-dim, good quality)
# Override: Set EMBEDDING_MODEL environment variable
# Alternatives: all-mpnet-base-v2 (768-dim, better quality, slower)
#               paraphrase-MiniLM-L6-v2 (384-dim, similar to default)
# Used by: vector_store.py for semantic code search
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# WATCH_DEBOUNCE_SECONDS: Delay before processing file changes (debouncing)
# Default: 2.0 seconds
# Override: Set WATCH_DEBOUNCE_SECONDS environment variable
# Impact: Higher = fewer re-indexes (less CPU), slower updates
# Recommendation: 1.0-5.0 seconds depending on edit frequency
# Used by: fs_watcher.py to batch rapid file changes
WATCH_DEBOUNCE_SECONDS = float(os.getenv("WATCH_DEBOUNCE_SECONDS", "2.0"))


# ============================================================================
# API SERVER CONFIGURATION
# ============================================================================

# API_PORT: Port for REST API server
# Default: 8080
# Override: Set API_PORT environment variable
# Example: API_PORT=9000
# Used by: api_server.py for FastAPI server
API_PORT = int(os.getenv("API_PORT", "8080"))


# ============================================================================
# OLLAMA CONFIGURATION
# ============================================================================

# OLLAMA_URL: Base URL for Ollama API server
# Default: http://ollama:11434 (Docker service name)
# Override: Set OLLAMA_URL environment variable
# Used by: llm_client.py for all Ollama API calls
# Note: Single Ollama instance loads models on-demand for memory efficiency
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

# OLLAMA_NUM_PARALLEL: Number of parallel requests to Ollama
# Default: 1 (single Ollama instance, sequential model loading)
# Override: Set OLLAMA_NUM_PARALLEL environment variable
# Impact: Higher values may cause memory issues on systems with < 16 GB RAM
# Recommendation: Keep at 1 for systems with limited memory
OLLAMA_NUM_PARALLEL = int(os.getenv("OLLAMA_NUM_PARALLEL", "1"))

# OLLAMA_MAX_LOADED_MODELS: Maximum models to keep loaded in memory
# Default: 1 (only one model loaded at a time for memory efficiency)
# Override: Set OLLAMA_MAX_LOADED_MODELS environment variable
# Note: Ollama automatically unloads models when memory is tight
OLLAMA_MAX_LOADED_MODELS = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "1"))


# ============================================================================
# CODE GENERATION DEFAULTS
# ============================================================================

# DEFAULT_MAX_TOKENS: Maximum tokens to generate in LLM response
# Default: 2000 tokens (~1500 words, ~100 lines of code)
# Override: Set DEFAULT_MAX_TOKENS environment variable
# Impact: Higher = longer responses, slower generation
# Recommendation: 1000-4000 depending on use case
# Used by: llm_client.py for code generation requests
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "2000"))

# DEFAULT_TEMPERATURE: Randomness in LLM output (0.0 = deterministic, 2.0 = creative)
# Default: 0.2 (low randomness, consistent code)
# Override: Set DEFAULT_TEMPERATURE environment variable
# Impact: Higher = more creative/varied, less predictable
# Recommendation: 0.1-0.3 for code, 0.7-1.0 for creative text
# Used by: llm_client.py for all generation requests
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))

# DEFAULT_TOP_P: Nucleus sampling threshold (0.0-1.0)
# Default: 0.9 (consider top 90% probability mass)
# Override: Set DEFAULT_TOP_P environment variable
# Impact: Lower = more focused, higher = more diverse
# Recommendation: 0.8-0.95 for most use cases
# Used by: llm_client.py for all generation requests
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))

# SYSTEM_PROMPT: System instruction for LLM code generation
# Default: Expert C++ programmer with C++17 focus
# Override: Set SYSTEM_PROMPT environment variable
# Example: "You are a C++20 expert focusing on modern features"
# Used by: llm_client.py to set LLM behavior
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are an expert C++ programmer. Generate clean, efficient, well-documented C++17 code."
)


# ============================================================================
# CODE QUALITY TOOLS
# ============================================================================

# CLANG_FORMAT_STYLE: Code formatting style for clang-format
# Default: Google (Google C++ Style Guide)
# Override: Set CLANG_FORMAT_STYLE environment variable
# Alternatives: LLVM, Chromium, Mozilla, WebKit, Microsoft, GNU
# Example: CLANG_FORMAT_STYLE=LLVM
# Used by: Code formatting utilities (if implemented)
CLANG_FORMAT_STYLE = os.getenv("CLANG_FORMAT_STYLE", "Google")

# CLANG_TIDY_CHECKS: Enabled checks for clang-tidy static analysis
# Default: readability, modernize, performance, bugprone checks
# Override: Set CLANG_TIDY_CHECKS environment variable
# Format: Comma-separated list with wildcards (* = all, - = disable)
# Example: CLANG_TIDY_CHECKS="-*,modernize-*,performance-*"
# Used by: Static analysis utilities (if implemented)
CLANG_TIDY_CHECKS = os.getenv(
    "CLANG_TIDY_CHECKS",
    "-*,readability-*,modernize-*,performance-*,bugprone-*"
)


# ============================================================================
# LOGGING
# ============================================================================

# LOG_LEVEL: Logging verbosity level
# Default: INFO (show info, warning, error, critical)
# Override: Set LOG_LEVEL environment variable
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
# Recommendation: INFO for production, DEBUG for development
# Used by: All modules for logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# LOG_FORMAT: Format string for log messages
# Default: Timestamp - Module - Level - Message
# Override: Set LOG_FORMAT environment variable
# Format: Python logging format string
# Used by: Logging configuration across all modules
LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# LOG_FILE: Path to application log file
# Default: /app/host_data/assistant.log
# Override: Set LOG_FILE environment variable
# Note: Persisted to host via volume mount
# Used by: Logging configuration for file output
LOG_FILE = os.getenv("LOG_FILE", "/app/host_data/assistant.log")


# ============================================================================
# CACHE SETTINGS
# ============================================================================

# CACHE_SIZE_SMALL: LRU cache size for small, frequently accessed data
# Default: 16 entries
# Used by: Small lookup tables, configuration caches
# Example: File extension checks, simple mappings
CACHE_SIZE_SMALL = 16

# CACHE_SIZE_MEDIUM: LRU cache size for medium-sized data
# Default: 128 entries
# Used by: Symbol lookups, file metadata caches
# Example: Recently accessed symbols, file paths
CACHE_SIZE_MEDIUM = 128

# CACHE_SIZE_LARGE: LRU cache size for large datasets
# Default: 1024 entries
# Used by: Embedding caches, large lookup tables
# Example: Cached embeddings, symbol signatures
CACHE_SIZE_LARGE = 1024

# AVAILABILITY_CACHE_TTL: Time-to-live for model availability cache (seconds)
# Default: 60 seconds
# Impact: Higher = fewer health checks, slower to detect model changes
# Recommendation: 30-120 seconds
# Used by: llm_client.py to cache Ollama model availability
AVAILABILITY_CACHE_TTL = 60


# ============================================================================
# TIMEOUT SETTINGS
# ============================================================================

# OLLAMA_API_TIMEOUT: Timeout for Ollama API requests (seconds)
# Default: 300 seconds (5 minutes)
# Impact: Higher = allow longer generation, risk hanging requests
# Recommendation: 60-300 depending on model size and complexity
# Used by: llm_client.py for all generation requests
OLLAMA_API_TIMEOUT = 300

# OLLAMA_HEALTH_TIMEOUT: Timeout for Ollama health checks (seconds)
# Default: 5 seconds
# Impact: Higher = more patient health checks, slower failure detection
# Recommendation: 3-10 seconds
# Used by: llm_client.py for availability checks
OLLAMA_HEALTH_TIMEOUT = 5

# CLANG_FORMAT_TIMEOUT: Timeout for clang-format execution (seconds)
# Default: 10 seconds
# Impact: Higher = allow formatting large files, risk hanging
# Recommendation: 5-30 seconds
# Used by: Code formatting utilities (if implemented)
CLANG_FORMAT_TIMEOUT = 10

# CLANG_TIDY_TIMEOUT: Timeout for clang-tidy execution (seconds)
# Default: 30 seconds
# Impact: Higher = allow analyzing large files, risk hanging
# Recommendation: 20-60 seconds
# Used by: Static analysis utilities (if implemented)
CLANG_TIDY_TIMEOUT = 30

# GIT_COMMAND_TIMEOUT: Timeout for git commands (seconds)
# Default: 10 seconds
# Impact: Higher = allow large repo operations, risk hanging
# Recommendation: 5-30 seconds
# Used by: git_incremental_indexer.py for git diff operations
GIT_COMMAND_TIMEOUT = 10


# ============================================================================
# THREAD POOL SETTINGS
# ============================================================================

# THREAD_POOL_WORKERS: Number of worker threads for parallel operations
# Default: 4 workers
# Impact: Higher = more parallelism, more CPU/memory usage
# Recommendation: Set to number of CPU cores (typically 2-8)
# Used by: compare_models.py for parallel model comparison
THREAD_POOL_WORKERS = 4


# ============================================================================
# FAISS INDEX SETTINGS
# ============================================================================

# FAISS_FLAT_INDEX_THRESHOLD: Threshold for choosing FAISS index type
# Default: 10000 vectors
# Logic: Use IndexFlatL2 (exact search) for < 10k vectors
#        Use IndexIVFFlat (approximate search) for >= 10k vectors
# Impact: Higher = use exact search longer (slower but more accurate)
# Recommendation: 5000-20000 depending on accuracy vs speed preference
# Used by: vector_store.py for index type selection
FAISS_FLAT_INDEX_THRESHOLD = 10000

# FAISS_IVF_NLIST_DIVISOR: Divisor for calculating IVF index clusters
# Default: 100 (nlist = num_vectors // 100)
# Logic: More clusters = faster search, less accurate
# Example: 10000 vectors → 100 clusters, 100000 vectors → 1000 clusters
# Recommendation: 50-200 depending on dataset size
# Used by: vector_store.py for IVF index configuration
FAISS_IVF_NLIST_DIVISOR = 100


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_faiss_index_path() -> str:
    """
    Get the full path to the FAISS vector index file.

    Returns:
        str: Full path to index.faiss file

    Example:
        >>> get_faiss_index_path()
        '/app/host_data/faiss_index/index.faiss'
    """
    return os.path.join(FAISS_INDEX_DIR, "index.faiss")


def get_faiss_meta_path() -> str:
    """
    Get the full path to the FAISS metadata file.

    The metadata file stores symbol IDs corresponding to each vector
    in the FAISS index, enabling lookup of original symbols.

    Returns:
        str: Full path to meta.npy file

    Example:
        >>> get_faiss_meta_path()
        '/app/host_data/faiss_index/meta.npy'
    """
    return os.path.join(FAISS_INDEX_DIR, "meta.npy")


def ensure_directories() -> None:
    """
    Ensure all required directories exist, creating them if necessary.

    Creates:
        - CONTAINER_HOST_DATA_DIR: Main data directory
        - COMPARISONS_DIR: Model comparison results
        - FAISS_INDEX_DIR: Vector index files
        - Parent directory of SYMBOL_DB_PATH: Database directory

    Note:
        Uses exist_ok=True to avoid errors if directories already exist.
        Should be called during application initialization.
    """
    directories = [
        CONTAINER_HOST_DATA_DIR,
        COMPARISONS_DIR,
        FAISS_INDEX_DIR,
        os.path.dirname(SYMBOL_DB_PATH),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config() -> bool:
    """
    Validate configuration settings.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    errors = []
    
    # Validate numeric ranges
    if SYMBOL_BATCH_SIZE < 1:
        errors.append("SYMBOL_BATCH_SIZE must be >= 1")
    
    if SYMBOL_EXTRACTOR_WORKERS < 1:
        errors.append("SYMBOL_EXTRACTOR_WORKERS must be >= 1")
    
    if EMBEDDING_BATCH_SIZE < 1:
        errors.append("EMBEDDING_BATCH_SIZE must be >= 1")
    
    if WATCH_DEBOUNCE_SECONDS < 0:
        errors.append("WATCH_DEBOUNCE_SECONDS must be >= 0")
    
    if DEFAULT_TEMPERATURE < 0 or DEFAULT_TEMPERATURE > 2:
        errors.append("DEFAULT_TEMPERATURE must be between 0 and 2")
    
    if DEFAULT_TOP_P < 0 or DEFAULT_TOP_P > 1:
        errors.append("DEFAULT_TOP_P must be between 0 and 1")
    
    # Print errors if any
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


# ============================================================================
# INITIALIZATION
# ============================================================================

# Validate configuration on import
if not validate_config():
    import sys
    print("Warning: Configuration validation failed. Using defaults where possible.")

