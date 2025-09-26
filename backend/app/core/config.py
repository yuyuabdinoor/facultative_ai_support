"""
Application configuration settings
"""
from pydantic_settings import BaseSettings
from typing import Optional, List
import secrets

class Settings(BaseSettings):
    # Application
    app_name: str = "AI Facultative Reinsurance System" 
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/reinsurance"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    celery_task_serializer: str = "json"
    celery_result_serializer: str = "json"
    celery_accept_content: List[str] = ["json"]
    celery_timezone: str = "UTC"
    
    # File Storage
    upload_dir: str = "./uploads"
    max_file_size: int = 104857600  # 100MB
    allowed_file_types: List[str] = ["pdf", "docx", "xlsx", "msg", "eml", "jpg", "png"]
    
    # Security
    secret_key: str = secrets.token_urlsafe(32)
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    
    # API Configuration
    api_v1_prefix: str = "/api/v1"
    api_title: str = "AI Facultative Reinsurance System API"
    api_description: str = "AI-Powered Facultative Reinsurance Decision Support System"
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst_size: int = 100
    
    # CORS
    cors_origins: List[str] = [
        "http://localhost:3005",
        "https://localhost:3005",
        "http://127.0.0.1:3005"
    ]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: List[str] = ["Content-Type", "Authorization", "X-API-Key"]
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    # AI/ML Configuration
    ocr_confidence_threshold: float = 0.8
    risk_score_low_threshold: float = 0.3
    risk_score_medium_threshold: float = 0.6
    risk_score_high_threshold: float = 0.8
    
    # Processing Timeouts (seconds)
    ocr_processing_timeout: int = 300
    data_extraction_timeout: int = 180
    risk_analysis_timeout: int = 240
    decision_engine_timeout: int = 120
    
    # Business Rules
    max_asset_value: float = 1000000000.0  # $1B
    min_asset_value: float = 100000.0      # $100K
    max_coverage_ratio: float = 0.9        # 90%
    
    # Email Configuration (for notifications)
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    
    # Email Polling Configuration
    email_imap_server: Optional[str] = None
    email_user: Optional[str] = None
    email_password: Optional[str] = None
    email_processed_folder: str = "Processed"
    email_error_folder: str = "Error"
    email_download_dir: str = "./email_attachments"
    check_interval: int = 300  # seconds
    max_attachment_size: int = 52428800  # 50MB in bytes
    email_polling_enabled: bool = False
    
    # External Services
    virus_scan_enabled: bool = True
    virus_scan_api_key: Optional[str] = None
    
    # Currency / FX
    exchange_rate_api_key: Optional[str] = None
    exchange_rate_base: str = "KES"
    
    # Model cache paths (optional). If set, main.py will export these to os.environ on startup.
    hf_home: Optional[str] = None
    transformers_cache: Optional[str] = None
    sentence_transformers_cache: Optional[str] = None
    torch_home: Optional[str] = None
    
    # Backup Configuration
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    backup_retention_days: int = 30
    
    # Audit Configuration
    audit_log_enabled: bool = True
    audit_log_retention_days: int = 90
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings()

# Validate critical settings
def validate_settings():
    """Validate critical configuration settings"""
    errors = []
    
    # if not settings.secret_key or len(settings.secret_key) < 32:
    #     errors.append("SECRET_KEY must be at least 32 characters long")
    
    if not settings.database_url:
        errors.append("DATABASE_URL is required")
    
    if not settings.redis_url:
        errors.append("REDIS_URL is required")
    
    if settings.max_file_size > 1024 * 1024 * 1024:  # 1GB
        errors.append("MAX_FILE_SIZE should not exceed 1GB")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")

# Validate settings on import
validate_settings()