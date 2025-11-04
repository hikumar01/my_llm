#!/usr/bin/env python3
"""
Symbol extractor for C++ codebases using libclang.
Optimized for performance with batch inserts and better error handling.
"""

import os
from pathlib import Path
from typing import List, Tuple
from clang import cindex
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import database operations
from database import init_database, insert_symbols, delete_symbols_by_repo

# Import constants
from constants import SYMBOL_BATCH_SIZE, SYMBOL_EXTRACTOR_WORKERS, CPP_EXTENSIONS, ensure_directories


def extract_from_file(repo: str, filepath: str) -> List[Tuple]:
    """
    Extract symbols from a single C++ file using libclang.

    Args:
        repo: Repository name
        filepath: Path to the file

    Returns:
        List of symbol tuples
    """
    try:
        # Parse with libclang
        index = cindex.Index.create()
        tu = index.parse(filepath, options=cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES)

        syms = []

        def walk(node):
            """Recursively walk the AST."""
            # Only extract top-level and class-level declarations
            if node.kind.name in ("FUNCTION_DECL", "CXX_METHOD", "CLASS_DECL",
                                  "STRUCT_DECL", "VAR_DECL", "ENUM_DECL",
                                  "TYPEDEF_DECL", "NAMESPACE"):
                name = node.spelling or node.displayname

                # Skip anonymous or empty names
                if not name or name.startswith('__'):
                    return

                sig = node.type.spelling if node.type else ""
                doc = node.raw_comment or ""

                syms.append((
                    repo,
                    filepath,
                    name,
                    node.kind.name,
                    sig,
                    doc,
                    node.location.line
                ))

            # Recurse into children
            for ch in node.get_children():
                walk(ch)

        walk(tu.cursor)
        return syms

    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}")
        return []


def process_file_batch(repo: str, filepaths: List[str]) -> List[Tuple]:
    """
    Process a batch of files and extract symbols.

    Args:
        repo: Repository name
        filepaths: List of file paths

    Returns:
        List of all symbols from all files
    """
    all_symbols = []
    for filepath in filepaths:
        symbols = extract_from_file(repo, filepath)
        all_symbols.extend(symbols)
    return all_symbols


def index_repo(repo_path: str, parallel: bool = True) -> bool:
    """
    Index a C++ repository and extract all symbols.
    Optimized with batch inserts and optional parallel processing.

    Args:
        repo_path: Path to the repository
        parallel: If True, use parallel processing

    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(repo_path):
        print(f"Error: Repository path does not exist: {repo_path}")
        return False

    print(f"Indexing repository: {repo_path}")
    print(f"Parallel processing: {'enabled' if parallel else 'disabled'}")

    # Collect all C++ files
    repo_path_obj = Path(repo_path)
    cpp_files = []

    for ext in CPP_EXTENSIONS:
        cpp_files.extend(repo_path_obj.rglob(f'*{ext}'))

    cpp_files = [str(f) for f in cpp_files]

    if not cpp_files:
        print(f"Warning: No C++ files found in {repo_path}")
        return False

    print(f"Found {len(cpp_files)} C++ files")

    # Initialize database
    if not init_database():
        print("Failed to initialize database")
        return False

    repo_name = os.path.basename(repo_path)
    file_count = 0
    symbol_count = 0

    try:
        # Delete old symbols for this repo
        from database import delete_symbols_by_repo
        deleted = delete_symbols_by_repo(repo_name)
        if deleted > 0:
            print(f"Cleared {deleted} old symbols for {repo_name}")

        if parallel and len(cpp_files) > 10:
            # Parallel processing for large repos
            print(f"Processing files in parallel with {SYMBOL_EXTRACTOR_WORKERS} workers...")

            # Split files into batches
            batch_size = max(1, len(cpp_files) // (SYMBOL_EXTRACTOR_WORKERS * 4))
            file_batches = [cpp_files[i:i+batch_size] for i in range(0, len(cpp_files), batch_size)]

            with ThreadPoolExecutor(max_workers=SYMBOL_EXTRACTOR_WORKERS) as executor:
                futures = []
                for batch in file_batches:
                    future = executor.submit(process_file_batch, repo_name, batch)
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        symbols = future.result()
                        if symbols:
                            # Batch insert using database module
                            inserted = insert_symbols(symbols)
                            symbol_count += inserted
                            file_count += len(symbols) // max(1, len(symbols) // 10)  # Estimate
                            print(f"Progress: {file_count}/{len(cpp_files)} files, {symbol_count} symbols")
                    except Exception as e:
                        print(f"Error processing batch: {e}")
        else:
            # Sequential processing
            symbols_batch = []
            for filepath in cpp_files:
                print(f"Processing: {filepath}")
                syms = extract_from_file(repo_name, filepath)
                symbols_batch.extend(syms)
                file_count += 1

                # Batch insert every SYMBOL_BATCH_SIZE symbols
                if len(symbols_batch) >= SYMBOL_BATCH_SIZE:
                    inserted = insert_symbols(symbols_batch)
                    symbol_count += inserted
                    symbols_batch = []
                    print(f"Progress: {file_count}/{len(cpp_files)} files, {symbol_count} symbols")

            # Insert remaining symbols
            if symbols_batch:
                inserted = insert_symbols(symbols_batch)
                symbol_count += inserted

        print(f"\n✅ Indexing complete!")
        print(f"   Files processed: {file_count}")
        print(f"   Symbols extracted: {symbol_count}")

        return True

    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        import traceback
        traceback.print_exc()
        return False
