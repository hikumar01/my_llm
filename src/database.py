#!/usr/bin/env python3
"""
Database Operations for C++ Symbol Storage
==========================================

This module provides centralized database management for storing and querying
C++ symbols extracted from source code. Features include:

- Thread-safe connection pooling
- Optimized batch operations
- Comprehensive symbol CRUD operations
- Database statistics and maintenance

Schema:
    symbols table:
        - id: INTEGER PRIMARY KEY AUTOINCREMENT
        - repo: TEXT (repository name)
        - file: TEXT (file path)
        - symbol: TEXT (symbol name)
        - kind: TEXT (symbol type: FUNCTION_DECL, CLASS_DECL, etc.)
        - signature: TEXT (type signature)
        - doc: TEXT (documentation comment)
        - line: INTEGER (line number)
        - created_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    file_index_metadata table:
        - id: INTEGER PRIMARY KEY AUTOINCREMENT
        - repo: TEXT (repository name)
        - file: TEXT (file path)
        - git_commit_hash: TEXT (git commit hash when indexed)
        - file_mtime: REAL (file modification time when indexed)
        - indexed_at: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        - UNIQUE(repo, file)

Indices:
    - idx_symbol: ON symbols(symbol) - Fast symbol name lookup
    - idx_repo: ON symbols(repo) - Fast repository filtering
    - idx_file: ON symbols(file) - Fast file filtering
    - idx_kind: ON symbols(kind) - Fast symbol type filtering
    - idx_repo_file: ON symbols(repo, file) - Fast repo+file lookup
    - idx_metadata_repo: ON file_index_metadata(repo) - Fast metadata lookup

Usage:
    from database import init_database, insert_symbols, search_symbols

    # Initialize database
    init_database()

    # Insert symbols
    symbols = [("myrepo", "main.cpp", "main", "FUNCTION_DECL", "int ()", "", 10)]
    insert_symbols(symbols)

    # Search symbols
    results = search_symbols("main")
"""

import os
import sqlite3
from typing import List, Tuple, Optional, Dict, Any
from contextlib import contextmanager
from threading import Lock

# Import database path from constants
from constants import SYMBOL_DB_PATH

# Thread-safe connection lock for concurrent access
_connection_lock = Lock()


@contextmanager
def get_connection():
    """
    Get a thread-safe database connection with automatic cleanup.

    This context manager ensures:
    - Thread-safe access via locking
    - Automatic connection cleanup
    - Row factory for dict-like access

    Yields:
        sqlite3.Connection: Database connection with row factory enabled

    Example:
        >>> with get_connection() as conn:
        ...     cur = conn.cursor()
        ...     cur.execute("SELECT * FROM symbols WHERE symbol = ?", ("main",))
        ...     results = cur.fetchall()

    Note:
        The connection is automatically closed when exiting the context,
        even if an exception occurs.
    """
    with _connection_lock:
        conn = sqlite3.connect(SYMBOL_DB_PATH)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()


def init_database() -> bool:
    """
    Initialize the symbols database with schema and indices.
    Creates tables and indices if they don't exist.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the directory exists
        db_dir = os.path.dirname(SYMBOL_DB_PATH)
        os.makedirs(db_dir, exist_ok=True)

        print(f"Initializing database at: {SYMBOL_DB_PATH}")
        
        with get_connection() as conn:
            cur = conn.cursor()
            
            # Create symbols table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo TEXT NOT NULL,
                    file TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    kind TEXT,
                    signature TEXT,
                    doc TEXT,
                    line INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create file index metadata table for tracking git commits and file changes
            cur.execute('''
                CREATE TABLE IF NOT EXISTS file_index_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo TEXT NOT NULL,
                    file TEXT NOT NULL,
                    git_commit_hash TEXT,
                    file_mtime REAL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(repo, file)
                )
            ''')

            # Create indices for faster queries
            cur.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON symbols(symbol)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_repo ON symbols(repo)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_file ON symbols(file)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_kind ON symbols(kind)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_repo_file ON symbols(repo, file)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_metadata_repo ON file_index_metadata(repo)')
            
            conn.commit()
            print(f"✅ Database initialized successfully")
            return True
            
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False


def insert_symbols(symbols: List[Tuple]) -> int:
    """
    Insert multiple symbols into the database.
    Uses batch insert for better performance.
    
    Args:
        symbols: List of tuples (repo, file, symbol, kind, signature, doc, line)
    
    Returns:
        int: Number of symbols inserted
    """
    if not symbols:
        return 0
    
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.executemany(
                'INSERT INTO symbols (repo, file, symbol, kind, signature, doc, line) VALUES (?, ?, ?, ?, ?, ?, ?)',
                symbols
            )
            conn.commit()
            return len(symbols)
    except Exception as e:
        print(f"Error inserting symbols: {e}")
        return 0


def delete_symbols_by_file(filepath: str) -> int:
    """
    Delete all symbols for a specific file.
    Used when re-indexing a file.
    
    Args:
        filepath: Path to the file
    
    Returns:
        int: Number of symbols deleted
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute('DELETE FROM symbols WHERE file = ?', (filepath,))
            deleted = cur.rowcount
            conn.commit()
            return deleted
    except Exception as e:
        print(f"Error deleting symbols: {e}")
        return 0


def delete_symbols_by_repo(repo: str) -> int:
    """
    Delete all symbols for a specific repository.

    Args:
        repo: Repository name

    Returns:
        int: Number of symbols deleted
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute('DELETE FROM symbols WHERE repo = ?', (repo,))
            deleted = cur.rowcount
            conn.commit()
            return deleted
    except Exception as e:
        print(f"Error deleting symbols: {e}")
        return 0


def upsert_file_metadata(repo: str, filepath: str, git_commit_hash: str = None, file_mtime: float = None) -> bool:
    """
    Insert or update file index metadata.

    Args:
        repo: Repository name
        filepath: Path to the file
        git_commit_hash: Git commit hash when indexed (optional)
        file_mtime: File modification time when indexed (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO file_index_metadata (repo, file, git_commit_hash, file_mtime)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(repo, file) DO UPDATE SET
                    git_commit_hash = excluded.git_commit_hash,
                    file_mtime = excluded.file_mtime,
                    indexed_at = CURRENT_TIMESTAMP
            ''', (repo, filepath, git_commit_hash, file_mtime))
            conn.commit()
            return True
    except Exception as e:
        print(f"Error upserting file metadata: {e}")
        return False


def get_file_metadata(repo: str, filepath: str) -> dict:
    """
    Get file index metadata.

    Args:
        repo: Repository name
        filepath: Path to the file

    Returns:
        dict: Metadata dict with keys: git_commit_hash, file_mtime, indexed_at
              Returns None if not found
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT git_commit_hash, file_mtime, indexed_at
                FROM file_index_metadata
                WHERE repo = ? AND file = ?
            ''', (repo, filepath))
            row = cur.fetchone()
            if row:
                return {
                    'git_commit_hash': row[0],
                    'file_mtime': row[1],
                    'indexed_at': row[2]
                }
            return None
    except Exception as e:
        print(f"Error getting file metadata: {e}")
        return None


def delete_file_metadata(repo: str, filepath: str = None) -> int:
    """
    Delete file metadata for a specific file or all files in a repo.

    Args:
        repo: Repository name
        filepath: Path to the file (optional, if None deletes all for repo)

    Returns:
        int: Number of metadata records deleted
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            if filepath:
                cur.execute('DELETE FROM file_index_metadata WHERE repo = ? AND file = ?', (repo, filepath))
            else:
                cur.execute('DELETE FROM file_index_metadata WHERE repo = ?', (repo,))
            deleted = cur.rowcount
            conn.commit()
            return deleted
    except Exception as e:
        print(f"Error deleting file metadata: {e}")
        return 0


def get_symbol_count() -> int:
    """
    Get total number of symbols in database.
    
    Returns:
        int: Total symbol count
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM symbols")
            return cur.fetchone()[0]
    except Exception as e:
        print(f"Error getting symbol count: {e}")
        return 0


def get_symbols_by_repo(repo: str) -> List[Dict[str, Any]]:
    """
    Get all symbols for a specific repository.
    
    Args:
        repo: Repository name
    
    Returns:
        List of symbol dictionaries
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM symbols WHERE repo = ?", (repo,))
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return []


def get_symbols_by_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Get all symbols for a specific file.
    
    Args:
        filepath: Path to the file
    
    Returns:
        List of symbol dictionaries
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM symbols WHERE file = ?", (filepath,))
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return []


def search_symbols(query: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Search for symbols by name (case-insensitive).
    
    Args:
        query: Search query
        limit: Maximum number of results
    
    Returns:
        List of matching symbol dictionaries
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM symbols WHERE symbol LIKE ? LIMIT ?",
                (f"%{query}%", limit)
            )
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        print(f"Error searching symbols: {e}")
        return []


def get_database_stats() -> Dict[str, Any]:
    """
    Get database statistics.

    Returns:
        Dictionary with database statistics (empty dict if database not initialized)
    """
    try:
        # Check if database file exists
        if not os.path.exists(SYMBOL_DB_PATH):
            print(f"Database file does not exist: {SYMBOL_DB_PATH}")
            return {
                "total_symbols": 0,
                "total_files": 0,
                "by_kind": {},
                "by_repo": {},
                "database_size_bytes": 0,
                "database_size_mb": 0.0,
                "database_path": SYMBOL_DB_PATH,
                "initialized": False
            }

        with get_connection() as conn:
            cur = conn.cursor()

            # Check if symbols table exists
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='symbols'")
            if not cur.fetchone():
                print("Symbols table does not exist in database")
                return {
                    "total_symbols": 0,
                    "total_files": 0,
                    "by_kind": {},
                    "by_repo": {},
                    "database_size_bytes": 0,
                    "database_size_mb": 0.0,
                    "database_path": SYMBOL_DB_PATH,
                    "initialized": False
                }

            # Total symbols
            cur.execute("SELECT COUNT(*) FROM symbols")
            total_symbols = cur.fetchone()[0]

            # Symbols by kind
            cur.execute("SELECT kind, COUNT(*) as count FROM symbols GROUP BY kind")
            by_kind = {row[0]: row[1] for row in cur.fetchall()}

            # Symbols by repo
            cur.execute("SELECT repo, COUNT(*) as count FROM symbols GROUP BY repo")
            by_repo = {row[0]: row[1] for row in cur.fetchall()}

            # Total files
            cur.execute("SELECT COUNT(DISTINCT file) FROM symbols")
            total_files = cur.fetchone()[0]

            # Database size
            db_size = os.path.getsize(SYMBOL_DB_PATH)

            return {
                "total_symbols": total_symbols,
                "total_files": total_files,
                "by_kind": by_kind,
                "by_repo": by_repo,
                "database_size_bytes": db_size,
                "database_size_mb": round(db_size / (1024 * 1024), 2),
                "database_path": SYMBOL_DB_PATH,
                "initialized": True
            }
    except Exception as e:
        print(f"Error getting database stats: {e}")
        import traceback
        traceback.print_exc()
        return {
            "total_symbols": 0,
            "total_files": 0,
            "by_kind": {},
            "by_repo": {},
            "database_size_bytes": 0,
            "database_size_mb": 0.0,
            "database_path": SYMBOL_DB_PATH,
            "initialized": False,
            "error": str(e)
        }


def get_repositories() -> List[Dict[str, Any]]:
    """
    Get list of all indexed repositories with their statistics.

    Returns:
        List of dictionaries with repository information
    """
    try:
        with get_connection() as conn:
            cur = conn.cursor()

            # Get repositories with their stats
            cur.execute("""
                SELECT
                    repo,
                    COUNT(*) as symbol_count,
                    COUNT(DISTINCT file) as file_count,
                    MAX(created_at) as last_indexed
                FROM symbols
                GROUP BY repo
                ORDER BY repo
            """)

            repositories = []
            for row in cur.fetchall():
                repositories.append({
                    "name": row[0],
                    "symbol_count": row[1],
                    "file_count": row[2],
                    "last_indexed": row[3]
                })

            return repositories
    except Exception as e:
        print(f"Error getting repositories: {e}")
        import traceback
        traceback.print_exc()
        return []


def vacuum_database() -> bool:
    """
    Optimize database by running VACUUM.
    Reclaims unused space and optimizes indices.
    
    Returns:
        bool: True if successful
    """
    try:
        with get_connection() as conn:
            conn.execute("VACUUM")
            print("✅ Database optimized successfully")
            return True
    except Exception as e:
        print(f"❌ Error optimizing database: {e}")
        return False


if __name__ == "__main__":
    """Test database operations."""
    print("Testing database operations...")
    
    # Initialize database
    if init_database():
        print("\n✅ Database initialization successful")
        
        # Get stats
        stats = get_database_stats()
        print(f"\nDatabase Statistics:")
        print(f"  Total symbols: {stats.get('total_symbols', 0)}")
        print(f"  Total files: {stats.get('total_files', 0)}")
        print(f"  Database size: {stats.get('database_size_mb', 0)} MB")
        print(f"  Database path: {stats.get('database_path', 'N/A')}")
        
        if stats.get('by_kind'):
            print(f"\n  Symbols by kind:")
            for kind, count in stats['by_kind'].items():
                print(f"    {kind}: {count}")
    else:
        print("\n❌ Database initialization failed")

