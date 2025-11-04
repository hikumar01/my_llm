#!/usr/bin/env python3
"""
Incremental watcher for git-based repositories.
Detects changed files and updates the symbol database and vector index.
"""

import os
import subprocess
from pathlib import Path
from typing import List

# Import from other modules
from symbol_extractor import extract_from_file
from database import delete_symbols_by_file, insert_symbols
from vector_store import build_index

# Import constants
from constants import CPP_EXTENSIONS, GIT_COMMAND_TIMEOUT


def get_changed_files(repo_path: str, base_ref: str = "HEAD~1", head_ref: str = "HEAD") -> List[str]:
    """
    Get list of changed C++ files between two git refs.

    Args:
        repo_path: Path to the git repository
        base_ref: Base git reference (default: HEAD~1)
        head_ref: Head git reference (default: HEAD)

    Returns:
        List of changed C++ file paths
    """
    try:
        # Save current directory
        original_dir = os.getcwd()

        # Change to repo directory
        os.chdir(repo_path)

        # Get changed files
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref, head_ref],
            capture_output=True,
            text=True,
            timeout=GIT_COMMAND_TIMEOUT
        )

        if result.returncode != 0:
            print(f"Error: git diff failed: {result.stderr}")
            return []

        # Filter for C++ files
        files = result.stdout.splitlines()
        cpp_files = [f for f in files if f.endswith(CPP_EXTENSIONS)]

        return cpp_files

    except subprocess.TimeoutExpired:
        print(f"Error: git diff timeout after {GIT_COMMAND_TIMEOUT} seconds")
        return []
    except FileNotFoundError:
        print("Error: git not found in PATH")
        return []
    except Exception as e:
        print(f"Error getting changed files: {e}")
        return []
    finally:
        # Restore original directory
        try:
            os.chdir(original_dir)
        except Exception:
            pass


def apply_changes(repo_path: str, rebuild_index: bool = True) -> bool:
    """
    Apply incremental changes to the symbol database and vector index.

    Args:
        repo_path: Path to the git repository
        rebuild_index: If True, rebuild the entire FAISS index (default: True)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get repository name
        repo_name = Path(repo_path).name

        # Get changed files
        changed_files = get_changed_files(repo_path)

        if not changed_files:
            print("No C++ files changed")
            return True

        print(f"Found {len(changed_files)} changed C++ files:")
        for f in changed_files:
            print(f"  - {f}")

        # Process each changed file
        total_symbols = 0
        for file_rel_path in changed_files:
            file_full_path = os.path.join(repo_path, file_rel_path)

            # Check if file still exists (not deleted)
            if not os.path.exists(file_full_path):
                print(f"File deleted: {file_rel_path}")
                # Delete symbols for this file
                deleted = delete_symbols_by_file(file_full_path)
                print(f"  Removed {deleted} symbols")
                continue

            print(f"Processing: {file_rel_path}")

            # Delete old symbols for this file
            deleted = delete_symbols_by_file(file_full_path)
            if deleted > 0:
                print(f"  Removed {deleted} old symbols")

            # Extract new symbols
            symbols = extract_from_file(repo_name, file_full_path)

            # Insert new symbols
            if symbols:
                inserted = insert_symbols(symbols)
                print(f"  Indexed {inserted} new symbols")
                total_symbols += inserted
            else:
                print(f"  No symbols found")

        print(f"\n✅ Processed {len(changed_files)} files, {total_symbols} symbols updated")

        # Rebuild FAISS index if requested
        if rebuild_index:
            print("\nRebuilding FAISS index...")
            if build_index():
                print("✅ Index rebuilt successfully")
            else:
                print("❌ Failed to rebuild index")
                return False

        return True

    except Exception as e:
        print(f"❌ Error applying changes: {e}")
        import traceback
        traceback.print_exc()
        return False
