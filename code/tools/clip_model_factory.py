"""Shared Chinese-CLIP model loader.

The project uses Chinese-CLIP from the summarizer, evaluator, and Streamlit app.
Keeping a single process-local cache avoids loading the large model repeatedly.
"""

from typing import Dict, Tuple

import torch

from config import model_config


class CLIPModelFactory:
    """Process-local singleton cache for Chinese-CLIP models."""

    _cache: Dict[Tuple[str, str, str], Tuple[object, object, object]] = {}

    @classmethod
    def get_model(
        cls,
        model_name: str = None,
        device: str = None,
        download_root: str = None,
    ):
        model_name = model_name or model_config.clip_model_name
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        download_root = download_root or model_config.clip_download_root

        cache_key = (model_name, device, download_root)
        if cache_key not in cls._cache:
            import cn_clip.clip as clip
            from cn_clip.clip import load_from_name

            model, preprocess = load_from_name(
                model_name,
                device=device,
                download_root=download_root,
            )
            model.eval()
            cls._cache[cache_key] = (model, preprocess, clip)

        return cls._cache[cache_key]


def get_clip_model(model_name: str = None, device: str = None, download_root: str = None):
    """Return (model, preprocess, clip_module)."""
    return CLIPModelFactory.get_model(model_name, device, download_root)
