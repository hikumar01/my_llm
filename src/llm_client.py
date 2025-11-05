#!/usr/bin/env python3
"""
LLM Client for code generation using local Ollama models.
Optimized for performance with caching and connection pooling.
"""

import os
import json
import re
import requests
import time
import subprocess
import tempfile
from typing import Optional, List, Dict, Tuple, Any, Union
from functools import lru_cache

# Import constants
from constants import (
    OLLAMA_URL,
    SYSTEM_PROMPT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    OLLAMA_API_TIMEOUT,
    OLLAMA_HEALTH_TIMEOUT,
    CLANG_FORMAT_STYLE,
    CLANG_FORMAT_TIMEOUT,
    CLANG_TIDY_CHECKS,
    CLANG_TIDY_TIMEOUT,
    AVAILABILITY_CACHE_TTL,
    CACHE_SIZE_MEDIUM
)

# Configure requests session for connection pooling
SESSION = requests.Session()
SESSION.headers.update({'Content-Type': 'application/json'})


@lru_cache(maxsize=1)
def load_models_from_env():
    """
    Load model configuration from environment variables.
    Falls back to default models if environment variables are not set.
    """
    models = {}

    # Try to load models from environment variables (MODEL_1_*, MODEL_2_*, etc.)
    for i in range(1, 10):  # Support up to 9 models
        key = os.getenv(f"MODEL_{i}_KEY")
        name = os.getenv(f"MODEL_{i}_NAME")

        if key and name:
            models[key] = {
                "name": name,
                "description": os.getenv(f"MODEL_{i}_DESC", f"Model {key}"),
                "license": os.getenv(f"MODEL_{i}_LICENSE", "Unknown"),
                "size": os.getenv(f"MODEL_{i}_SIZE", "Unknown")
            }

    # If no models loaded from env, use defaults
    if not models:
        models = {
            "deepseek-coder": {
                "name": "deepseek-coder:6.7b",
                "description": "DeepSeek Coder 6.7B - Excellent for code generation",
                "license": "MIT",
                "size": "3.8 GB",
                "best_for": "code",
                "performance": {
                    "code": "⭐⭐⭐⭐⭐ (95/100)",
                    "workflow": "⭐⭐⭐⭐ (85/100)",
                    "ppt": "⭐⭐⭐ (70/100)",
                    "speed": "2.3s"
                },
                "recommended": True,
                "priority": 1
            },
            "qwen2.5-coder": {
                "name": "qwen2.5-coder:7b",
                "description": "Qwen 2.5 Coder 7B - Strong reasoning and structured output",
                "license": "Apache 2.0",
                "size": "4.7 GB",
                "best_for": "workflow",
                "performance": {
                    "code": "⭐⭐⭐⭐⭐ (93/100)",
                    "workflow": "⭐⭐⭐⭐⭐ (92/100)",
                    "ppt": "⭐⭐⭐⭐ (85/100)",
                    "speed": "2.8s"
                },
                "recommended": True,
                "priority": 2
            },
            "llama3.2": {
                "name": "llama3.2:3b",
                "description": "Llama 3.2 3B - Fast and creative for design tasks",
                "license": "Llama 3.2 Community License",
                "size": "2.0 GB",
                "best_for": "ppt",
                "performance": {
                    "code": "⭐⭐⭐ (78/100)",
                    "workflow": "⭐⭐⭐⭐ (75/100)",
                    "ppt": "⭐⭐⭐⭐ (88/100)",
                    "speed": "1.5s"
                },
                "recommended": True,
                "priority": 3
            },
            "codellama": {
                "name": "codellama:7b",
                "description": "CodeLlama 7B - Meta's code-specialized model",
                "license": "Llama 2 Community License",
                "size": "3.8 GB",
                "best_for": "code",
                "performance": {
                    "code": "⭐⭐⭐⭐ (88/100)",
                    "workflow": "⭐⭐⭐ (72/100)",
                    "ppt": "⭐⭐⭐ (65/100)",
                    "speed": "2.5s"
                },
                "recommended": False,
                "priority": 4
            },
            "starcoder2": {
                "name": "starcoder2:7b",
                "description": "StarCoder2 7B - Multi-language code model",
                "license": "BigCode OpenRAIL-M",
                "size": "4.0 GB",
                "best_for": "code",
                "performance": {
                    "code": "⭐⭐⭐⭐ (86/100)",
                    "workflow": "⭐⭐⭐ (70/100)",
                    "ppt": "⭐⭐⭐ (68/100)",
                    "speed": "2.6s"
                },
                "recommended": False,
                "priority": 5
            }
        }

    return models


# Load models from environment or use defaults
AVAILABLE_LOCAL_MODELS = load_models_from_env()


def get_default_model_key() -> str:
    """Get the first available model key as default."""
    return next(iter(AVAILABLE_LOCAL_MODELS.keys()))


# Removed get_ollama_url_for_model() - now using single Ollama instance


class OllamaClient:
    """
    Client for interacting with local Ollama models.
    Optimized with connection pooling and caching.
    """

    # Class-level cache for model availability
    _availability_cache: Dict[str, Tuple[bool, float]] = {}
    _cache_ttl = AVAILABILITY_CACHE_TTL

    def __init__(self, model: str, base_url: Optional[str] = None):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., "deepseek-coder:6.7b")
            base_url: Ollama API URL (default: from constants.OLLAMA_URL)
        """
        self.model = model
        self.base_url = base_url or OLLAMA_URL
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """
        Check if Ollama is available and model is loaded.
        Uses caching to avoid repeated API calls.
        """
        cache_key = f"{self.base_url}:{self.model}"
        current_time = time.time()

        # Check cache
        if cache_key in self._availability_cache:
            cached_result, cached_time = self._availability_cache[cache_key]
            if current_time - cached_time < self._cache_ttl:
                return cached_result

        # Make API call
        try:
            print(f"Checking availability of {self.model} at {self.base_url}...")
            response = SESSION.get(f"{self.base_url}/api/tags", timeout=OLLAMA_HEALTH_TIMEOUT)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                result = self.model in model_names
                self._availability_cache[cache_key] = (result, current_time)
                return result
            return False
        except Exception as e:
            print(f"Warning: Ollama not available at {self.base_url}: {e}")
            return False

    def generate(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE) -> str:
        """
        Generate code using Ollama.

        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate (default: from constants)
            temperature: Sampling temperature (default: from constants)

        Returns:
            Generated code as a string
        """
        if not self.available:
            return f"// Error: Model {self.model} not available in Ollama\n// Please run: ollama pull {self.model}"

        try:
            response = SESSION.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:",
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=OLLAMA_API_TIMEOUT
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"// Error: Ollama API returned status {response.status_code}"

        except requests.exceptions.Timeout:
            return f"// Error: Request timeout after {OLLAMA_API_TIMEOUT} seconds"
        except requests.exceptions.ConnectionError as e:
            return f"// Error: Connection failed - {e}"
        except Exception as e:
            return f"// Error calling Ollama API: {e}"

    def generate_stream(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
                       temperature: float = DEFAULT_TEMPERATURE):
        """
        Generate code using the LLM with streaming enabled.
        Yields tokens as they arrive from Ollama.

        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate (default: from constants)
            temperature: Sampling temperature (default: from constants)

        Yields:
            Dict containing token data and metadata
        """
        if not self.available:
            yield {
                "error": f"Model {self.model} not available in Ollama. Please run: ollama pull {self.model}",
                "done": True
            }
            return

        try:
            print(f"[STREAMING] Starting stream for model {self.model} at {self.base_url}")
            # For streaming, use a tuple timeout: (connect_timeout, read_timeout)
            # Connect timeout: 30s to establish connection
            # Read timeout: None to allow streaming to continue indefinitely
            response = SESSION.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:",
                    "stream": True,  # Enable streaming
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                },
                timeout=(30, None),  # (connect_timeout, read_timeout)
                stream=True  # Enable streaming in requests
            )

            if response.status_code != 200:
                error_msg = f"Ollama API returned status {response.status_code}"
                try:
                    error_detail = response.text
                    print(f"[STREAMING ERROR] {error_msg}: {error_detail}")
                    error_msg += f": {error_detail}"
                except:
                    print(f"[STREAMING ERROR] {error_msg}")
                yield {
                    "error": error_msg,
                    "done": True
                }
                return

            # Stream the response line by line
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        yield data

                        # Check if generation is complete
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.Timeout:
            yield {
                "error": f"Request timeout after {OLLAMA_API_TIMEOUT} seconds",
                "done": True
            }
        except requests.exceptions.ConnectionError as e:
            yield {
                "error": f"Connection failed - {e}",
                "done": True
            }
        except Exception as e:
            yield {
                "error": f"Error calling Ollama API: {e}",
                "done": True
            }


def extract_code_blocks(text: str) -> str:
    """
    Extract code code blocks from LLM response.
    Handles markdown code blocks (```cpp ... ```) and returns only the code.
    If no code blocks found, returns the original text.

    Args:
        text: LLM response that may contain explanatory text and code blocks

    Returns:
        Extracted code or original text if no code blocks found
    """
    if not text or not text.strip():
        return text

    # Pattern to match code blocks: ```cpp ... ``` or ``` ... ```
    # Supports both ```cpp and ``` variants
    patterns = [
        r'```cpp\s*\n(.*?)\n```',  # ```cpp ... ```
        r'```c\+\+\s*\n(.*?)\n```',  # ```c++ ... ```
        r'```\s*\n(.*?)\n```',      # ``` ... ``` (generic)
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Join all code blocks with newlines
            return '\n\n'.join(matches)

    # No code blocks found - return original text
    # (might be pure code without markdown formatting)
    return text


def format_code_with_clang(code: str) -> str:
    """
    Format code using clang-format.
    Automatically extracts code blocks from LLM responses before formatting.
    Optimized with better error handling and resource cleanup.

    Args:
        code: The code to format (may include explanatory text)

    Returns:
        Formatted code
    """
    if not code or not code.strip():
        return code

    # Extract only code blocks (removes explanatory text)
    code = extract_code_blocks(code)

    temp_path = None
    try:
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(code)
            temp_path = f.name

        # Run clang-format
        result = subprocess.run(
            ['clang-format', f'-style={CLANG_FORMAT_STYLE}', temp_path],
            capture_output=True,
            text=True,
            timeout=CLANG_FORMAT_TIMEOUT
        )

        if result.returncode == 0:
            return result.stdout
        else:
            print(f"clang-format error: {result.stderr}")
            return code

    except subprocess.TimeoutExpired:
        print(f"Error: clang-format timeout after {CLANG_FORMAT_TIMEOUT} seconds")
        return code
    except FileNotFoundError:
        print("Error: clang-format not found in PATH")
        return code
    except Exception as e:
        print(f"Error formatting code: {e}")
        return code
    finally:
        # Ensure cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


# Cache for installed models list
_installed_models_cache: Dict[str, Tuple[List[str], float]] = {}
_installed_models_cache_ttl = 5  # 5 seconds cache


def invalidate_models_cache():
    """Invalidate the installed models cache. Call after downloading/removing models."""
    global _installed_models_cache
    _installed_models_cache.clear()


def get_installed_models(base_url: str = None, use_cache: bool = True) -> List[str]:
    """
    Get list of installed Ollama models.
    Uses short-lived cache (5 seconds) to reduce API calls.

    Args:
        base_url: Ollama API URL (default: from constants)
        use_cache: Whether to use cached results (default: True)

    Returns:
        List of installed model names
    """
    url = base_url or OLLAMA_URL
    cache_key = url
    current_time = time.time()

    # Check cache
    if use_cache and cache_key in _installed_models_cache:
        cached_models, cached_time = _installed_models_cache[cache_key]
        if current_time - cached_time < _installed_models_cache_ttl:
            return cached_models

    # Make API call
    try:
        print(f"Getting installed models from {url}...")
        response = SESSION.get(f"{url}/api/tags", timeout=OLLAMA_HEALTH_TIMEOUT)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            # Update cache
            _installed_models_cache[cache_key] = (model_names, current_time)
            return model_names
        return []
    except Exception as e:
        print(f"Error getting installed models: {e}")
        return []


def pull_model(model_key: str, base_url: str = None) -> bool:
    """
    Pull/download an Ollama model to its dedicated container.

    Args:
        model_key: Model key (e.g., "deepseek-coder") or full name (e.g., "deepseek-coder:6.7b")
        base_url: Ollama API URL (default: model-specific URL from constants)

    Returns:
        True if successful, False otherwise
    """
    # Convert model key to full name if needed
    if model_key in AVAILABLE_LOCAL_MODELS:
        model_full_name = AVAILABLE_LOCAL_MODELS[model_key]["name"]
    else:
        # Assume it's already a full name
        model_full_name = model_key

    # Use provided base_url or default OLLAMA_URL
    url = base_url or OLLAMA_URL

    try:
        print(f"Pulling model: {model_full_name}...")
        response = SESSION.post(
            f"{url}/api/pull",
            json={"name": model_full_name},
            timeout=3600,  # 1 hour timeout for large models
            stream=True
        )

        if response.status_code == 200:
            # Stream the response to show progress
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        status = data.get("status", "")
                        if "total" in data and "completed" in data:
                            total = data["total"]
                            completed = data["completed"]
                            percent = (completed / total * 100) if total > 0 else 0
                            print(f"  {status}: {percent:.1f}%")
                        else:
                            print(f"  {status}")
                    except json.JSONDecodeError:
                        pass

            print(f"✅ Successfully pulled {model_full_name}")
            invalidate_models_cache()  # Invalidate cache after successful download
            return True
        else:
            print(f"❌ Failed to pull {model_full_name}: HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error pulling {model_full_name}: {e}")
        return False


def ensure_models_available(models: List[str] = None, base_url: str = None, auto_pull: bool = True) -> Dict[str, bool]:
    """
    Ensure required models are available, optionally pulling them if missing.

    Since all Ollama containers share the same model storage volume,
    we only need to check one instance and pull to one instance.
    All containers will automatically see the models in the shared cache.

    Args:
        models: List of model names to check (default: all AVAILABLE_LOCAL_MODELS)
        base_url: Ollama API URL (default: from constants)
        auto_pull: If True, automatically pull missing models

    Returns:
        Dict mapping model names to availability status
    """
    if models is None:
        models = list(AVAILABLE_LOCAL_MODELS.keys())

    # Check the shared model cache by querying the Ollama instance
    check_url = base_url or OLLAMA_URL

    print(f"Checking shared model cache at: {check_url}")
    installed = get_installed_models(check_url)
    results = {}

    for model_key in models:
        # Get the full model name from the config
        if model_key in AVAILABLE_LOCAL_MODELS:
            model_full_name = AVAILABLE_LOCAL_MODELS[model_key]["name"]
        else:
            # If not in config, assume it's already a full name
            model_full_name = model_key

        if model_full_name in installed:
            print(f"✅ Model already installed: {model_key}")
            results[model_key] = True
        elif auto_pull:
            print(f"⬇️  Model not found in shared cache, pulling: {model_key}")
            # Pull to Ollama instance
            pull_url = base_url or OLLAMA_URL
            print(f"   Pulling to: {pull_url}")
            results[model_key] = pull_model(model_key, pull_url)

            if results[model_key]:
                print(f"   ✅ Model {model_key} is now available in shared cache")
                print(f"   📦 All Ollama containers can now access this model")
        else:
            print(f"❌ Model not installed: {model_key}")
            results[model_key] = False

    return results


def get_all_models_with_status(base_url: str = None) -> List[Dict[str, Any]]:
    """
    Get all available models with their download status and metadata.

    Args:
        base_url: Ollama API URL (default: from constants)

    Returns:
        List of dicts with model info: name, description, size, license, downloaded
    """
    installed = get_installed_models(base_url)
    models_info = []

    for model_key, model_config in AVAILABLE_LOCAL_MODELS.items():
        # Check if the full model name is in installed models
        model_full_name = model_config["name"]
        is_downloaded = model_full_name in installed

        models_info.append({
            "key": model_key,
            "name": model_full_name,
            "description": model_config["description"],
            "size": model_config["size"],
            "license": model_config["license"],
            "downloaded": is_downloaded
        })

    return models_info


def get_downloaded_models(base_url: str = None, use_cache: bool = True, include_metadata: bool = False) -> Union[List[str], List[Dict[str, str]]]:
    """
    Get list of model keys that are actually downloaded.

    Args:
        base_url: Ollama API URL (default: from constants)
        use_cache: Whether to use cached results (default: True)
        include_metadata: If True, return list of dicts with key, name, description

    Returns:
        List of model keys (if include_metadata=False) or list of dicts (if include_metadata=True)
    """
    installed = get_installed_models(base_url, use_cache=use_cache)
    downloaded_keys = []
    downloaded_models = []

    for key, config in AVAILABLE_LOCAL_MODELS.items():
        # Check if the full model name is in installed models
        if config["name"] in installed:
            downloaded_keys.append(key)
            if include_metadata:
                downloaded_models.append({
                    "key": key,
                    "name": config["name"],
                    "description": config["description"]
                })

    return downloaded_models if include_metadata else downloaded_keys


def remove_model(model_key: str, base_url: str = None) -> bool:
    """
    Remove/delete an Ollama model from its dedicated container.

    Args:
        model_key: Model key (e.g., "deepseek-coder") or full name (e.g., "deepseek-coder:6.7b")
        base_url: Ollama API URL (default: model-specific URL from constants)

    Returns:
        True if successful, False otherwise
    """
    # Convert model key to full name if needed
    if model_key in AVAILABLE_LOCAL_MODELS:
        model_full_name = AVAILABLE_LOCAL_MODELS[model_key]["name"]
    else:
        # Assume it's already a full name
        model_full_name = model_key

    # Use provided base_url or default OLLAMA_URL
    url = base_url or OLLAMA_URL

    try:
        print(f"Removing model: {model_full_name}...")
        response = SESSION.delete(
            f"{url}/api/delete",
            json={"name": model_full_name},
            timeout=30
        )

        if response.status_code == 200:
            print(f"✅ Successfully removed {model_full_name}")
            invalidate_models_cache()  # Invalidate cache after successful removal
            return True
        else:
            print(f"❌ Failed to remove {model_full_name}: HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error removing {model_full_name}: {e}")
        return False


def check_code_with_clang_tidy(code: str) -> List[str]:
    """
    Check code with clang-tidy.
    Automatically extracts code blocks from LLM responses before checking.
    Optimized with better error handling and resource cleanup.

    Args:
        code: The code to check (may include explanatory text)

    Returns:
        List of warnings/errors
    """
    if not code or not code.strip():
        return []

    # Extract only code blocks (removes explanatory text)
    code = extract_code_blocks(code)

    temp_path = None
    try:
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(code)
            temp_path = f.name

        # Run clang-tidy with configured checks
        result = subprocess.run(
            ['clang-tidy', f'-checks={CLANG_TIDY_CHECKS}', temp_path, '--'],
            capture_output=True,
            text=True,
            timeout=CLANG_TIDY_TIMEOUT
        )

        # Parse output - extract only warning/error lines
        warnings = []
        for line in result.stdout.split('\n'):
            if 'warning:' in line or 'error:' in line:
                warnings.append(line.strip())

        return warnings

    except subprocess.TimeoutExpired:
        print(f"Error: clang-tidy timeout after {CLANG_TIDY_TIMEOUT} seconds")
        return []
    except FileNotFoundError:
        print("Error: clang-tidy not found in PATH")
        return []
    except Exception as e:
        print(f"Error checking code: {e}")
        return []
    finally:
        # Ensure cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


# Removed _generate_with_model(), compare_models(), and merge_model_responses() - multi-model functionality removed

