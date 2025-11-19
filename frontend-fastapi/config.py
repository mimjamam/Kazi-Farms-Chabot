"""
Configuration management for the Kazi Farms Frontend
"""
import os
from typing import Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class UIConfig:
    """UI Configuration settings"""
    max_messages: int = int(os.getenv("UI_MAX_MESSAGES", "100"))
    messages_per_page: int = int(os.getenv("UI_MESSAGES_PER_PAGE", "20"))
    session_timeout: int = int(os.getenv("UI_SESSION_TIMEOUT", "3600"))  # 1 hour
    enable_timestamps: bool = os.getenv("UI_ENABLE_TIMESTAMPS", "false").lower() == "true"
    enable_dark_mode: bool = os.getenv("UI_ENABLE_DARK_MODE", "false").lower() == "true"
    theme_color: str = os.getenv("UI_THEME_COLOR", "#007bff")

@dataclass
class APIConfig:
    """API Configuration settings"""
    base_url: str = os.getenv("FASTAPI_BASE_URL", os.getenv("API_BASE_URL", "http://localhost:8000"))
    timeout: int = int(os.getenv("API_TIMEOUT", "30"))
    health_timeout: int = int(os.getenv("API_HEALTH_TIMEOUT", "15"))
    max_retries: int = int(os.getenv("API_MAX_RETRIES", "3"))
    retry_delay: float = float(os.getenv("API_RETRY_DELAY", "1.0"))
    health_cache_duration: int = int(os.getenv("API_HEALTH_CACHE_DURATION", "30"))

@dataclass
class LoggingConfig:
    """Logging Configuration settings"""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: str = os.getenv("LOG_FILE_PATH", "logs/frontend.log")
    max_file_size: int = int(os.getenv("LOG_MAX_FILE_SIZE", "10485760"))  # 10MB
    backup_count: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))

@dataclass
class AppConfig:
    """Main application configuration"""
    ui: UIConfig = field(default_factory=UIConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # App metadata
    app_name: str = "Kazi Farms Assistant"
    app_version: str = "2.0.0"
    app_description: str = "AI-powered assistant for Kazi Farms"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "ui": {
                "max_messages": self.ui.max_messages,
                "messages_per_page": self.ui.messages_per_page,
                "session_timeout": self.ui.session_timeout,
                "enable_timestamps": self.ui.enable_timestamps,
                "enable_dark_mode": self.ui.enable_dark_mode,
                "theme_color": self.ui.theme_color
            },
            "api": {
                "base_url": self.api.base_url,
                "timeout": self.api.timeout,
                "health_timeout": self.api.health_timeout,
                "max_retries": self.api.max_retries,
                "retry_delay": self.api.retry_delay,
                "health_cache_duration": self.api.health_cache_duration
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file_path": self.logging.file_path,
                "max_file_size": self.logging.max_file_size,
                "backup_count": self.logging.backup_count
            },
            "app": {
                "name": self.app_name,
                "version": self.app_version,
                "description": self.app_description
            }
        }
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to file"""
        import json
        config_path = Path(file_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'AppConfig':
        """Load configuration from file"""
        import json
        
        if not Path(file_path).exists():
            return cls()
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        
        # Update UI config
        if 'ui' in data:
            for key, value in data['ui'].items():
                if hasattr(config.ui, key):
                    setattr(config.ui, key, value)
        
        # Update API config
        if 'api' in data:
            for key, value in data['api'].items():
                if hasattr(config.api, key):
                    setattr(config.api, key, value)
        
        # Update logging config
        if 'logging' in data:
            for key, value in data['logging'].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)
        
        return config

# Global configuration instance
config = AppConfig()
