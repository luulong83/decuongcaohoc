"""
Configuration Loader
Handles YAML config files with inheritance
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import copy


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Supports config inheritance via 'base_config' key
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Handle base config inheritance
    if 'base_config' in config:
        base_config_path = config_path.parent.parent / config['base_config']
        base_config = load_config(str(base_config_path))
        
        # Merge configs (config overrides base_config)
        merged_config = deep_merge(base_config, config)
        del merged_config['base_config']  # Remove base_config key
        config = merged_config
    
    return config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries
    
    Args:
        base: Base dictionary
        override: Dictionary with override values
    
    Returns:
        Merged dictionary
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"✅ Config saved to: {save_path}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If config is invalid
    """
    required_keys = [
        'project',
        'paths',
        'dataset',
        'model',
        'training',
        'evaluation'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate paths
    if 'data_dir' not in config['paths']:
        raise ValueError("Missing 'data_dir' in paths config")
    
    # Validate model config
    if 'name' not in config['model']:
        raise ValueError("Missing 'name' in model config")
    
    # Validate training config
    if 'num_epochs' not in config['training']:
        raise ValueError("Missing 'num_epochs' in training config")
    
    return True


def print_config(config: Dict[str, Any], indent: int = 0):
    """
    Pretty print configuration
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")