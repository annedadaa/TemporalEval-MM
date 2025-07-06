import os
import json
from typing import Any, Dict


class DotDict(dict):
    """
    A dictionary that supports dot notation access, recursively.
    """
    def __getattr__(self, key):
        if key in self:
            value = self[key]
            if isinstance(value, dict):
                return DotDict(value)
            return value
        raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")


def load_config(config_path: str = None) -> DotDict:
    """
    Load the config file and return it as a DotDict for dot-notation access.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    return DotDict(config_dict)


class Config:
    """
    Main Config wrapper class.
    """
    def __init__(self, config_path: str = None):
        self._config = load_config(config_path)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieve a nested key using dot-separated path (e.g. "model.vqa_model").
        """
        keys = key_path.split('.')
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def __getattr__(self, item):
        return getattr(self._config, item)
