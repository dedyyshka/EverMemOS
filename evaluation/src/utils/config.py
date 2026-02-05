"""
Configuration loading utilities.

Supports YAML configuration file loading with environment variable substitution.
"""
import os
import yaml
import re
from pathlib import Path
from typing import Dict, Any


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        file_path: YAML file path
        
    Returns:
        Parsed configuration dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Recursively replace environment variables
    config = _replace_env_vars(config)
    
    return config


def _replace_env_vars(obj: Any) -> Any:
    """
    Recursively replace environment variables in configuration.
    
    Supported format: ${VAR_NAME}
    """
    if isinstance(obj, dict):
        return {key: _replace_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_replace_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Match ${VAR_NAME} (fallbacks запрещены)
        pattern = r'\$\{([^:}]+)(?::([^}]+))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)
            if default_value is not None:
                raise ValueError(
                    f"Запрещён fallback для env-переменной '{var_name}'. "
                    "Используй только ${VAR} без значения по умолчанию."
                )
            if var_name not in os.environ:
                raise ValueError(
                    f"Не задана обязательная env-переменная '{var_name}'."
                )
            return os.environ.get(var_name, "")
        
        return re.sub(pattern, replacer, obj)
    else:
        return obj


def save_yaml(config: Dict[str, Any], file_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        file_path: Save path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)

