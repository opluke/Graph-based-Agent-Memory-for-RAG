import os
from pathlib import Path


def configure_repo_cache() -> Path:
    """Force model/cache downloads into this repository to avoid user-profile permission issues."""
    repo_root = Path(__file__).resolve().parent.parent
    cache_root = repo_root / ".cache"
    hf_home = cache_root / "huggingface"
    transformers_cache = hf_home / "transformers"

    hf_home.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(hf_home / "sentence_transformers"))

    return cache_root
