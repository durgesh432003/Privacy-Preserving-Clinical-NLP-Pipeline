"""
Model Registry — Auto-discovers Ollama models and manages developer configuration.
Provides a single source of truth for all available models in the pipeline.
"""

import os
import json
import time
import httpx
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CONFIG_PATH = Path(__file__).parent.parent.parent / "data" / "models_config.json"
CACHE_TTL_SECONDS = 30  # Re-query Ollama every 30s


@dataclass
class ModelInfo:
    """Metadata about a single model."""
    name: str
    display_name: str = ""
    category: str = "general"
    description: str = ""
    enabled: bool = True
    size_bytes: int = 0
    size_display: str = ""
    parameter_count: str = ""
    last_modified: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.replace("-", " ").replace(":", " ").title()
        if self.size_bytes and not self.size_display:
            gb = self.size_bytes / (1024 ** 3)
            self.size_display = f"{gb:.1f} GB" if gb >= 1.0 else f"{self.size_bytes / (1024 ** 2):.0f} MB"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """
    Discovers and manages models available via Ollama.

    - Queries Ollama /api/tags for installed models
    - Merges with developer config from data/models_config.json
    - Caches results to avoid excessive API calls
    """

    def __init__(self):
        self._cache: Dict[str, ModelInfo] = {}
        self._cache_timestamp: float = 0
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load developer model configuration from JSON file."""
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"models": {}}

    def _save_config(self):
        """Persist model configuration to JSON file."""
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(self._config, f, indent=2)

    def _query_ollama(self) -> List[Dict[str, Any]]:
        """Query Ollama API for installed models."""
        try:
            resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                return resp.json().get("models", [])
        except Exception:
            pass
        return []

    def _refresh_cache(self):
        """Refresh the model cache if TTL has expired."""
        now = time.time()
        if now - self._cache_timestamp < CACHE_TTL_SECONDS and self._cache:
            return

        self._config = self._load_config()
        ollama_models = self._query_ollama()
        new_cache: Dict[str, ModelInfo] = {}

        for m in ollama_models:
            name = m.get("name", "")
            if not name:
                continue

            # Merge Ollama metadata with developer config
            dev_cfg = self._config.get("models", {}).get(name, {})

            info = ModelInfo(
                name=name,
                display_name=dev_cfg.get("display_name", ""),
                category=dev_cfg.get("category", "general"),
                description=dev_cfg.get("description", f"Ollama model: {name}"),
                enabled=dev_cfg.get("enabled", True),
                size_bytes=m.get("size", 0),
                parameter_count=dev_cfg.get("parameter_count", ""),
                last_modified=m.get("modified_at", ""),
            )
            new_cache[name] = info

        self._cache = new_cache
        self._cache_timestamp = now

    def get_available_models(self, include_disabled: bool = False) -> List[ModelInfo]:
        """Return all available models, optionally including disabled ones."""
        self._refresh_cache()
        models = list(self._cache.values())
        if not include_disabled:
            models = [m for m in models if m.enabled]
        return sorted(models, key=lambda m: m.name)

    def get_model_names(self) -> List[str]:
        """Return just the names of enabled models."""
        return [m.name for m in self.get_available_models()]

    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get info for a specific model."""
        self._refresh_cache()
        return self._cache.get(name)

    def add_model(self, name: str, display_name: str = "", category: str = "general",
                  description: str = "", parameter_count: str = "") -> bool:
        """
        Add or update a model entry in the developer config.
        The model must already be installed in Ollama.
        """
        self._refresh_cache()

        if name not in self._cache:
            # Model not in Ollama — check if it's pullable
            return False

        if "models" not in self._config:
            self._config["models"] = {}

        self._config["models"][name] = {
            "display_name": display_name or name.replace("-", " ").replace(":", " ").title(),
            "category": category,
            "description": description or f"Ollama model: {name}",
            "parameter_count": parameter_count,
            "enabled": True,
        }

        self._save_config()
        self._cache_timestamp = 0  # Force refresh
        return True

    def toggle_model(self, name: str, enabled: bool) -> bool:
        """Enable or disable a model."""
        if "models" not in self._config:
            self._config["models"] = {}

        if name not in self._config["models"]:
            self._config["models"][name] = {}

        self._config["models"][name]["enabled"] = enabled
        self._save_config()
        self._cache_timestamp = 0
        return True

    def get_default_model(self) -> str:
        """Return the default model from env or first available."""
        default = os.getenv("OLLAMA_MODEL", "phi4-mini")
        available = self.get_model_names()
        if default in available:
            return default
        # Try partial match (e.g. "phi4-mini" matches "phi4-mini:latest")
        for name in available:
            if name.startswith(default):
                return name
        return available[0] if available else "phi4-mini"

    def get_evaluator_model(self) -> str:
        """Return the model used for evaluation (judge). Uses largest available."""
        evaluator = os.getenv("EVALUATOR_MODEL", "")
        if evaluator:
            available = self.get_model_names()
            if evaluator in available:
                return evaluator

        # Fall back to largest available model by size
        models = self.get_available_models()
        if models:
            return max(models, key=lambda m: m.size_bytes).name
        return os.getenv("OLLAMA_MODEL", "phi4-mini")

    def get_num_gpu_for_model(self, model_name: str) -> int:
        """
        Determine the appropriate num_gpu parameter for a given model.
        Forces 999 (100% GPU) for models that fit within VRAM,
        and falls back to -1 (auto-split) for models that exceed VRAM.
        """
        if not model_name:
            return -1

        import subprocess
        total_vram_bytes = 0
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            total_vram_bytes = int(res.stdout.strip()) * 1024 * 1024
        except Exception:
            pass

        if total_vram_bytes == 0:
            return -1

        model_info = self.get_model_info(model_name)
        if not model_info:
            # Fall back to partial match
            available = self.get_available_models()
            for m in available:
                if m.name == model_name or m.name.startswith(model_name) or model_name.startswith(m.name):
                    model_info = m
                    break

        if not model_info:
            return -1

        size_bytes = model_info.size_bytes
        if size_bytes == 0:
            return -1

        # 700 MB headroom for KV cache and system
        vram_headroom = 700 * 1024 * 1024
        if size_bytes + vram_headroom < total_vram_bytes:
            return 999

        return -1


# Singleton instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or create the singleton ModelRegistry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
