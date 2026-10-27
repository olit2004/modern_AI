"""
Platform, device, and configuration utilities.
Auto-detects execution environment (Google Colab, Kaggle, Local),
determines output storage directory (Google Drive or local),
sets deterministic seeds, and parses YAML/JSON configurations.
"""

import os
import sys
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml
except ImportError:
    yaml = None

def detect_platform() -> str:
    """
    Detect whether running on Google Colab, Kaggle, or a standard local environment.
    """
    if "COLAB_GPU" in os.environ or "google.colab" in sys.modules:
        return "colab"
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    return "local"

def get_default_output_dir(base_name: str = "results", mount_colab_drive: bool = False) -> Path:
    """
    Return appropriate output directory based on platform:
    - Colab: Optional Google Drive mount /content/drive/MyDrive/connect4_experiment/ or /content/results
    - Kaggle: /kaggle/working/results
    - Local: ./results
    """
    plat = detect_platform()
    if plat == "kaggle":
        out_dir = Path("/kaggle/working") / base_name
    elif plat == "colab":
        if mount_colab_drive and os.path.exists("/content/drive/MyDrive"):
            out_dir = Path("/content/drive/MyDrive/connect4_experiment") / base_name
        else:
            out_dir = Path("/content") / base_name
    else:
        out_dir = Path("./") / base_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def get_device():
    """
    Detect PyTorch device (CUDA, MPS, or CPU).
    Falls back gracefully if torch is not installed yet.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return "cpu"

def set_seed(seed: int = 42):
    """
    Set seeds across Python random, NumPy, and PyTorch (if present) for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML or JSON configuration file.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in [".yaml", ".yml"]:
            if yaml is None:
                raise ImportError("PyYAML is not installed. Please install pyyaml or use a .json config.")
            return yaml.safe_load(f)
        else:
            return json.load(f)

def save_json(data: Any, file_path: str):
    """Save serializable data to JSON with indentation."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
