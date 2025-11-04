#!/usr/bin/env python3
"""
Compare multiple local LLM models on the same prompt.
Generates code using all available models and saves results for comparison.
Optimized with parallel processing and efficient file I/O.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_client import compare_models, format_code_with_clang, check_code_with_clang_tidy

# Import constants
from constants import COMPARISONS_DIR, ensure_directories


def _process_model_result(model_key: str, result: dict, comparison_dir: Path) -> Optional[Dict]:
    """
    Process a single model's result (format, check, save).
    Used for parallel processing.

    Args:
        model_key: Model identifier
        result: Result dictionary from compare_models
        comparison_dir: Directory to save results

    Returns:
        Metadata dictionary or None if not available
    """
    if not result["available"] or not result["code"]:
        return None

    model_dir = comparison_dir / model_key
    model_dir.mkdir(exist_ok=True)

    # Save raw code
    (model_dir / "raw_code.cpp").write_text(result["code"])

    # Format code
    formatted = format_code_with_clang(result["code"])
    (model_dir / "formatted_code.cpp").write_text(formatted)

    # Check with clang-tidy
    warnings = check_code_with_clang_tidy(formatted)
    (model_dir / "clang_tidy_warnings.txt").write_text("\n".join(warnings) if warnings else "No warnings")

    # Save metadata
    metadata = {
        "model": result["model"],
        "description": result["description"],
        "generation_time": result["time"],
        "warnings_count": len(warnings),
        "code_length": len(result["code"])
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return metadata


def save_comparison_results(prompt: str, results: dict, output_dir: str = None, parallel: bool = True):
    """
    Save comparison results to files.
    Optimized with parallel processing for formatting and checking.

    Args:
        prompt: The prompt used
        results: Results from compare_models()
        output_dir: Directory to save results (default: from COMPARISONS_DIR env var)
        parallel: If True, process models in parallel
    """
    # Get output directory from constants if not specified
    if output_dir is None:
        output_dir = COMPARISONS_DIR

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create timestamp-based subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = output_path / timestamp
    comparison_dir.mkdir(exist_ok=True)

    # Save prompt
    (comparison_dir / "prompt.txt").write_text(prompt)

    # Process each model's output
    metadata_map = {}

    if parallel and len(results) > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for model_key, result in results.items():
                if result["available"] and result["code"]:
                    future = executor.submit(_process_model_result, model_key, result, comparison_dir)
                    futures[future] = model_key

            for future in as_completed(futures):
                model_key = futures[future]
                try:
                    metadata = future.result()
                    if metadata:
                        metadata_map[model_key] = metadata
                except Exception as e:
                    print(f"Error processing {model_key}: {e}")
    else:
        # Sequential processing
        for model_key, result in results.items():
            metadata = _process_model_result(model_key, result, comparison_dir)
            if metadata:
                metadata_map[model_key] = metadata
    
    # Create summary report
    summary = []
    summary.append("="*70)
    summary.append("MODEL COMPARISON REPORT")
    summary.append("="*70)
    summary.append(f"\nPrompt: {prompt}\n")
    summary.append(f"Timestamp: {timestamp}\n")
    summary.append("="*70)
    summary.append("\nRESULTS:\n")
    
    for model_key, result in results.items():
        summary.append(f"\n{model_key.upper()}")
        summary.append("-" * 70)
        if result["available"]:
            summary.append(f"Model: {result['model']}")
            summary.append(f"Description: {result['description']}")
            summary.append(f"Generation Time: {result['time']:.2f}s")
            
            # Count warnings
            model_dir = comparison_dir / model_key
            warnings_file = model_dir / "clang_tidy_warnings.txt"
            if warnings_file.exists():
                warnings_text = warnings_file.read_text()
                if warnings_text == "No warnings":
                    summary.append("Clang-tidy: ✅ No warnings")
                else:
                    warning_count = len(warnings_text.strip().split("\n"))
                    summary.append(f"Clang-tidy: ⚠️  {warning_count} warnings")
        else:
            summary.append(f"Status: ❌ Not available")
            summary.append(f"Error: {result['error']}")
        summary.append("")
    
    summary.append("="*70)
    summary.append(f"\nResults saved to: {comparison_dir}")
    summary.append("="*70)
    
    summary_text = "\n".join(summary)
    (comparison_dir / "SUMMARY.txt").write_text(summary_text)
    
    print(summary_text)
    
    return comparison_dir

