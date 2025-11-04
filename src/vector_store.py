#!/usr/bin/env python3
"""
Vector store for semantic code search using FAISS.
Optimized for performance with batch processing and efficient indexing.
"""

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional

# Import database operations
from database import get_connection, get_symbol_count

# Import constants
from constants import (
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    EMBEDDING_BATCH_SIZE,
    FAISS_FLAT_INDEX_THRESHOLD,
    FAISS_IVF_NLIST_DIVISOR,
    get_faiss_index_path,
    get_faiss_meta_path
)

# Get paths
INDEX_PATH = get_faiss_index_path()
META_PATH = get_faiss_meta_path()

# Lazy load model to save memory
_MODEL: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Lazy load the sentence transformer model."""
    global _MODEL
    if _MODEL is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def fetch_symbols_batch(cursor, batch_size: int = 10000):
    """
    Generator to fetch symbols in batches from database cursor.

    Args:
        cursor: Database cursor
        batch_size: Number of rows to fetch per batch

    Yields:
        List of symbol tuples
    """
    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            break
        yield batch


def build_index(incremental: bool = False) -> bool:
    """
    Build FAISS index from symbols in database.
    Optimized with batch processing and efficient memory usage.

    Args:
        incremental: If True, update existing index instead of rebuilding

    Returns:
        True if successful, False otherwise
    """
    # Check if database has symbols
    total_count = get_symbol_count()

    if total_count == 0:
        print("Error: No symbols found in database")
        print("Please run symbol_extractor.py first to index your code")
        return False

    print(f"Building FAISS index from {total_count} symbols...")

    try:
        with get_connection() as conn:
            cur = conn.cursor()

            print(f"Found {total_count} symbols, building embeddings in batches...")

            # Process in batches to reduce memory usage
            cur.execute("SELECT id, repo, file, symbol, doc FROM symbols")

            all_embeddings = []
            all_metas = []
            processed = 0

            model = get_model()

            for batch in fetch_symbols_batch(cur, batch_size=10000):
                texts = []
                metas = []

                for idx, repo, file, symbol, doc in batch:
                    # Create rich text representation for better embeddings
                    text = f"{symbol} {doc or ''} in {file}"
                    texts.append(text)
                    metas.append((repo, file, symbol))

                # Encode batch
                batch_embeddings = model.encode(
                    texts,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

                all_embeddings.append(batch_embeddings)
                all_metas.extend(metas)

                processed += len(batch)
                print(f"Processed {processed}/{total_count} symbols ({100*processed//total_count}%)")

        # Combine all embeddings (outside the connection context)
        embeddings = np.vstack(all_embeddings).astype("float32")

        print(f"Building FAISS index with {len(embeddings)} vectors...")

        # Choose index type based on size
        if len(embeddings) < FAISS_FLAT_INDEX_THRESHOLD:
            # Use flat index for small datasets (exact search)
            index = faiss.IndexFlatL2(embeddings.shape[1])
        else:
            # Use IVF index for larger datasets (approximate search, faster)
            nlist = min(100, len(embeddings) // FAISS_IVF_NLIST_DIVISOR)  # Number of clusters
            quantizer = faiss.IndexFlatL2(embeddings.shape[1])
            index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist)
            print(f"Training IVF index with {nlist} clusters...")
            index.train(embeddings)

        index.add(embeddings)

        # Ensure directory exists
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

        print(f"Saving index to: {INDEX_PATH}")
        faiss.write_index(index, INDEX_PATH)
        np.save(META_PATH, np.array(all_metas, dtype=object))
        print(f"✅ Index saved successfully!")
        print(f"   - Index: {INDEX_PATH}")
        print(f"   - Metadata: {META_PATH}")
        print(f"   - Total vectors: {index.ntotal}")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error building index: {e}")
        import traceback
        traceback.print_exc()
        return False


def search_symbols(query: str, top_k: int = 10, repo_filter: Optional[str] = None) -> List[dict]:
    """
    Search for symbols using semantic similarity.

    Args:
        query: Natural language search query
        top_k: Number of results to return
        repo_filter: Optional repository name to filter results

    Returns:
        List of matching symbols with metadata
    """
    try:
        # Check if index exists
        if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
            print("Error: FAISS index not found. Please run build_index() first.")
            return []

        # Load index and metadata
        index = faiss.read_index(INDEX_PATH)
        metadata = np.load(META_PATH, allow_pickle=True)

        # Get model and encode query
        model = get_model()
        query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

        # Search index
        distances, indices = index.search(query_embedding, min(top_k * 2, index.ntotal))

        # Get results from database
        results = []
        with get_connection() as conn:
            cur = conn.cursor()

            for idx in indices[0]:
                if idx < len(metadata):
                    repo, file, symbol = metadata[idx]

                    # Apply repository filter if specified
                    if repo_filter and repo != repo_filter:
                        continue

                    # Get full symbol details from database
                    cur.execute(
                        "SELECT * FROM symbols WHERE repo = ? AND file = ? AND symbol = ? LIMIT 1",
                        (repo, file, symbol)
                    )
                    row = cur.fetchone()

                    if row:
                        results.append(dict(row))

                        # Stop if we have enough results
                        if len(results) >= top_k:
                            break

        return results

    except Exception as e:
        print(f"Error searching symbols: {e}")
        import traceback
        traceback.print_exc()
        return []
