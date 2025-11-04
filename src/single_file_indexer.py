#!/usr/bin/env python3
"""
Single file indexer for incremental updates.
Used for incremental indexing when files change.
"""

import os
from symbol_extractor import extract_from_file
from database import delete_symbols_by_file, insert_symbols

# Import constants
from constants import CONTAINER_REPOS_DIR

def index_single_file(filepath, repo_name=None):
    """Index a single file and update the database."""
    if not os.path.exists(filepath):
        print(f"Error: File does not exist: {filepath}")
        return

    if not filepath.endswith(('.cpp', '.cc', '.c', '.hpp', '.h')):
        print(f"Skipping non-C++ file: {filepath}")
        return

    # Determine repo name
    if repo_name is None:
        # Try to extract from path
        if filepath.startswith(CONTAINER_REPOS_DIR + '/'):
            parts = filepath.split('/')
            # Find the index of CONTAINER_REPOS_DIR in the path
            repos_parts = CONTAINER_REPOS_DIR.split('/')
            repos_depth = len([p for p in repos_parts if p])
            if len(parts) > repos_depth + 1:
                repo_name = parts[repos_depth + 1]
            else:
                repo_name = 'unknown'
        else:
            repo_name = 'unknown'
    
    print(f"Indexing file: {filepath} (repo: {repo_name})")

    try:
        # Delete existing symbols for this file
        deleted = delete_symbols_by_file(filepath)
        if deleted > 0:
            print(f"  Removed {deleted} old symbols for {filepath}")

        # Extract new symbols
        syms = extract_from_file(repo_name, filepath)

        # Insert new symbols using database module
        if syms:
            inserted = insert_symbols(syms)
            print(f"  Indexed {inserted} symbols from {filepath}")
        else:
            print(f"  No symbols found in {filepath}")

    except Exception as e:
        print(f"Error indexing {filepath}: {e}")
        import traceback
        traceback.print_exc()

