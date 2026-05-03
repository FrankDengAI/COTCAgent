"""
Configuration management with environment variable support and validation
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from validation_utils import validate_api_config, ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration with environment variable support"""

    # API Configuration
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com/v1/chat/completions"
    deepseek_model: str = "deepseek-chat"
    deepseek_max_tokens: int = 4000
    deepseek_temperature: float = 0.1
    deepseek_timeout: int = 120
    deepseek_rate_limit_requests: int = 60
    deepseek_rate_limit_window: int = 60

    # Application Settings
    log_level: str = "INFO"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    max_concurrent_requests: int = 5
    request_timeout_seconds: int = 300

    # Security Settings
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    max_input_length: int = 10000
    sensitive_data_masking: bool = True

    # Performance Settings
    enable_performance_monitoring: bool = True
    performance_log_interval: int = 60  # seconds
    memory_cleanup_interval: int = 300  # 5 minutes

    # File Paths
    disease_database_path: str = "disease_symptom_database.json"
    config_file_path: str = "config.json"
    log_file_path: str = "cotc_agent.log"

    def to_deepseek_config(self) -> 'DeepSeekConfig':
        """Convert to DeepSeekConfig object"""
        from utils import DeepSeekConfig
        return DeepSeekConfig(
            api_key=self.deepseek_api_key,
            api_base=self.deepseek_api_base,
            model=self.deepseek_model,
            max_tokens=self.deepseek_max_tokens,
            temperature=self.deepseek_temperature,
            timeout=self.deepseek_timeout,
            rate_limit_requests=self.deepseek_rate_limit_requests,
            rate_limit_window=self.deepseek_rate_limit_window
        )


class ConfigManager:
    """Configuration manager with multiple sources support"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.getenv('COTC_CONFIG_FILE', 'config.json')
        self._config = None
        self._load_config()

    def _load_config(self):
        """Load configuration from multiple sources with priority"""
        config = AppConfig()

        # 1. Load from environment variables (highest priority)
        self._load_from_env(config)

        # 2. Load from config file (medium priority)
        self._load_from_file(config)

        # 3. Apply defaults (lowest priority - already set in dataclass)

        # Validate configuration
        self._validate_config(config)

        self._config = config
        logger.info("Configuration loaded successfully")

    def _load_from_env(self, config: AppConfig):
        """Load configuration from environment variables"""
        env_mappings = {
            'DEEPSEEK_API_KEY': 'deepseek_api_key',
            'DEEPSEEK_API_BASE': 'deepseek_api_base',
            'DEEPSEEK_MODEL': 'deepseek_model',
            'DEEPSEEK_MAX_TOKENS': 'deepseek_max_tokens',
            'DEEPSEEK_TEMPERATURE': 'deepseek_temperature',
            'DEEPSEEK_TIMEOUT': 'deepseek_timeout',
            'DEEPSEEK_RATE_LIMIT_REQUESTS': 'deepseek_rate_limit_requests',
            'DEEPSEEK_RATE_LIMIT_WINDOW': 'deepseek_rate_limit_window',
            'LOG_LEVEL': 'log_level',
            'CACHE_ENABLED': 'cache_enabled',
            'CACHE_TTL_SECONDS': 'cache_ttl_seconds',
            'MAX_CONCURRENT_REQUESTS': 'max_concurrent_requests',
            'REQUEST_TIMEOUT_SECONDS': 'request_timeout_seconds',
            'ENABLE_INPUT_VALIDATION': 'enable_input_validation',
            'ENABLE_OUTPUT_SANITIZATION': 'enable_output_sanitization',
            'MAX_INPUT_LENGTH': 'max_input_length',
            'SENSITIVE_DATA_MASKING': 'sensitive_data_masking',
            'ENABLE_PERFORMANCE_MONITORING': 'enable_performance_monitoring',
            'PERFORMANCE_LOG_INTERVAL': 'performance_log_interval',
            'MEMORY_CLEANUP_INTERVAL': 'memory_cleanup_interval',
            'DISEASE_DATABASE_PATH': 'disease_database_path',
            'CONFIG_FILE_PATH': 'config_file_path',
            'LOG_FILE_PATH': 'log_file_path'
        }

        for env_var, config_attr in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(env_value, config_attr)
                setattr(config, config_attr, converted_value)
                logger.debug(f"Loaded {config_attr} from environment variable {env_var}")

    def _convert_env_value(self, value: str, attr_name: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if attr_name in [
            'cache_enabled', 'enable_input_validation', 'enable_output_sanitization',
            'sensitive_data_masking', 'enable_performance_monitoring'
        ]:
            return value.lower() in ('true', '1', 'yes', 'on')

        # Integer conversion
        if attr_name in [
            'deepseek_max_tokens', 'deepseek_timeout', 'deepseek_rate_limit_requests',
            'deepseek_rate_limit_window', 'cache_ttl_seconds', 'max_concurrent_requests',
            'request_timeout_seconds', 'max_input_length', 'performance_log_interval',
            'memory_cleanup_interval'
        ]:
            try:
                return int(value)
            except ValueError:
                logger.warning(f"Invalid integer value for {attr_name}: {value}")
                return getattr(AppConfig(), attr_name)  # Return default

        # Float conversion
        if attr_name in ['deepseek_temperature']:
            try:
                return float(value)
            except ValueError:
                logger.warning(f"Invalid float value for {attr_name}: {value}")
                return getattr(AppConfig(), attr_name)  # Return default

        # String values (default)
        return value

    def _load_from_file(self, config: AppConfig):
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_file):
            logger.info(f"Configuration file {self.config_file} not found, using defaults")
            return

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)

            # Apply file configuration
            for key, value in file_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    logger.debug(f"Loaded {key} from configuration file")

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load configuration file {self.config_file}: {e}")
            logger.warning("Continuing with environment variables and defaults")

    def _validate_config(self, config: AppConfig):
        """Validate the loaded configuration"""
        # Validate API configuration
        api_config = {
            'api_key': config.deepseek_api_key,
            'api_base': config.deepseek_api_base,
            'model': config.deepseek_model,
            'max_tokens': config.deepseek_max_tokens,
            'temperature': config.deepseek_temperature,
            'timeout': config.deepseek_timeout
        }

        validation_result = validate_api_config(api_config)
        if not validation_result.is_valid:
            raise ConfigurationError(f"API configuration validation failed: {validation_result.errors}")

        for warning in validation_result.warnings:
            logger.warning(f"API config warning: {warning}")

        # Validate other settings
        if config.max_input_length <= 0:
            raise ConfigurationError("max_input_length must be positive")

        if config.cache_ttl_seconds <= 0:
            raise ConfigurationError("cache_ttl_seconds must be positive")

        if config.max_concurrent_requests <= 0:
            raise ConfigurationError("max_concurrent_requests must be positive")

        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.log_level.upper() not in valid_log_levels:
            raise ConfigurationError(f"log_level must be one of: {valid_log_levels}")

    def get_config(self) -> AppConfig:
        """Get the loaded configuration"""
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")
        return self._config

    def save_config(self, config: Optional[AppConfig] = None):
        """Save configuration to file"""
        if config is None:
            config = self._config

        if config is None:
            raise ConfigurationError("No configuration to save")

        try:
            # Create directory if it doesn't exist
            config_dir = Path(self.config_file).parent
            config_dir.mkdir(parents=True, exist_ok=True)

            # Save to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(config), f, indent=2, ensure_ascii=False)

            logger.info(f"Configuration saved to {self.config_file}")

        except IOError as e:
            logger.error(f"Failed to save configuration: {e}")
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")

        for key, value in updates.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.info(f"Updated configuration: {key} = {value}")
            else:
                logger.warning(f"Ignoring unknown configuration key: {key}")

        # Re-validate configuration
        self._validate_config(self._config)

    def get_deepseek_config(self) -> 'DeepSeekConfig':
        """Get DeepSeek configuration object"""
        return self.get_config().to_deepseek_config()

    def create_default_config_file(self):
        """Create a default configuration file"""
        default_config = AppConfig()
        # Mask sensitive data in default config
        default_config.deepseek_api_key = "your-api-key-here"

        self.save_config(default_config)
        logger.info(f"Created default configuration file: {self.config_file}")

    @classmethod
    def from_env(cls) -> 'ConfigManager':
        """Create ConfigManager using only environment variables"""
        manager = cls()
        # Force reload to ensure env vars take precedence
        manager._load_config()
        return manager
