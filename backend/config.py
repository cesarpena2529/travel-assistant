"""
Configuration and environment setup for the travel assistant API
"""
import os
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    APP_NAME: str = "Travel Assistant API"
    VERSION: str = "2.0.0"
    
    # API Keys
    OPENAI_API_KEY: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str = "us-east-1"
    BEDROCK_ENDPOINT: str = "https://bedrock.us-east-1.amazonaws.com"
    
    # Database (legacy OpenSearch - now optional)
    OPENSEARCH_HOST: str = ""
    OPENSEARCH_USERNAME: str = ""
    OPENSEARCH_PASSWORD: str = ""
    
    # Models
    EMBED_MODEL: str = "text-embedding-ada-002"
    CLAUDE_MODEL: str = "anthropic.claude-3-haiku-20240307-v1:0"
    
    # Limits
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.7
    MAX_RESULTS: int = 20
    DEFAULT_RESULTS: int = 10
    
    class Config:
        env_file = ".env"

# Configuration
EMBED_MODEL = "text-embedding-3-small"
OPENSEARCH_INDEX = "travel-kb-index"
MAX_RESULTS = 20
DEFAULT_RESULTS = 10

# Claude Configuration
CLAUDE_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"  # Claude 3 Haiku (standard)
MAX_TOKENS = 1000
TEMPERATURE = 0.7

# File paths
CURRENT_DIR = Path(__file__).parent
METADATA_FILE = CURRENT_DIR.parent / "data" / "metadata_lookup.json"

# Global settings instance
_settings = None

def get_config():
    """Get the global configuration instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def validate_environment() -> None:
    """Validate required environment variables and files"""
    errors = []
    
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY", "OPENSEARCH_HOST", "OPENSEARCH_USERNAME", "OPENSEARCH_PASSWORD", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing environment variable: {var}")
    
    # Check required files
    if not METADATA_FILE.exists():
        errors.append(f"Missing metadata file: {METADATA_FILE}")
    
    if errors:
        error_msg = "Configuration errors:\n" + "\n".join(f"- {err}" for err in errors)
        raise RuntimeError(error_msg)

def get_opensearch_config() -> Dict[str, Any]:
    """Get OpenSearch client configuration"""
    return {
        "hosts": [{"host": os.getenv("OPENSEARCH_HOST"), "port": 443}],
        "http_auth": (os.getenv("OPENSEARCH_USERNAME"), os.getenv("OPENSEARCH_PASSWORD")),
        "use_ssl": True,
        "verify_certs": True,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 60,
        "max_retries": 3,
        "retry_on_timeout": True
    }
