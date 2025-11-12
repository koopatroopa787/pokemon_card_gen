import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import os


class Config:
    """Configuration manager for the Pokemon Card Generator"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Load configuration from YAML file

        Args:
            config_path: Path to config file. If None, looks for config.yaml,
                        then falls back to config.default.yaml
        """
        if config_path is None:
            # Try to find config file
            if Path("config.yaml").exists():
                config_path = "config.yaml"
            else:
                config_path = "config.default.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Override with environment variables if they exist
        config = self._apply_env_overrides(config)

        return config

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # Example: POKEMON_CARD_GEN_BATCH_SIZE=64
        prefix = "POKEMON_CARD_GEN_"

        env_mappings = {
            f"{prefix}BATCH_SIZE": ("training", "batch_size", int),
            f"{prefix}LEARNING_RATE": ("training", "learning_rate", float),
            f"{prefix}NUM_EPOCHS": ("training", "num_epochs", int),
            f"{prefix}API_PORT": ("api", "port", int),
            f"{prefix}API_HOST": ("api", "host", str),
            f"{prefix}LATENT_DIM": ("model", "latent_dim", int),
        }

        for env_var, (section, key, type_func) in env_mappings.items():
            if env_var in os.environ:
                config[section][key] = type_func(os.environ[env_var])

        return config

    def get(self, *keys, default=None):
        """
        Get nested configuration value

        Example:
            config.get("training", "batch_size")
            config.get("model", "latent_dim", default=100)
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self.config[key]

    def __repr__(self):
        return f"Config(path={self.config_path})"


# Global config instance
_config = None


def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create global config instance"""
    global _config
    if _config is None or config_path is not None:
        _config = Config(config_path)
    return _config
