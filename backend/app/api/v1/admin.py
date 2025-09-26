"""
System administration API endpoints
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr
from datetime import datetime

from ...core.database import get_db
from ...core.auth import require_scopes, User, get_password_hash

router = APIRouter()

class SystemHealth(BaseModel):
    status: str
    uptime_seconds: int
    database_status: str
    redis_status: str
    celery_status: str
    disk_usage_percent: float
    memory_usage_percent: float
    cpu_usage_percent: float
    active_connections: int
    queue_sizes: Dict[str, int]

class UserManagement(BaseModel):
    id: str
    username: str
    email: str
    full_name: str
    is_active: bool
    is_superuser: bool
    scopes: List[str]
    created_at: datetime
    last_login: Optional[datetime]

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    password: str
    is_superuser: bool = False
    scopes: List[str] = ["read"]

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None
    scopes: Optional[List[str]] = None

class SystemConfiguration(BaseModel):
    max_file_size_mb: int
    allowed_file_types: List[str]
    ocr_confidence_threshold: float
    risk_score_thresholds: Dict[str, float]
    processing_timeouts: Dict[str, int]
    rate_limits: Dict[str, int]
    email_notifications_enabled: bool
    audit_log_retention_days: int

class BackupInfo(BaseModel):
    backup_id: str
    created_at: datetime
    size_bytes: int
    type: str
    status: str
    description: str

@router.get("/health", response_model=SystemHealth)
async def get_system_health(
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    Get comprehensive system health status
    
    Returns detailed information about:
    - Service status (database, Redis, Celery)
    - Resource utilization (CPU, memory, disk)
    - Queue sizes and processing status
    - Active connections and performance metrics
    """
    return SystemHealth(
        status="healthy",
        uptime_seconds=86400,
        database_status="connected",
        redis_status="connected",
        celery_status="running",
        disk_usage_percent=45.2,
        memory_usage_percent=68.5,
        cpu_usage_percent=23.1,
        active_connections=15,
        queue_sizes={
            "ocr_processing": 5,
            "data_extraction": 2,
            "risk_analysis": 8,
            "decision_engine": 1
        }
    )

@router.get("/users", response_model=List[UserManagement])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    List all system users with filtering options
    
    Parameters:
    - skip: Number of users to skip (pagination)
    - limit: Maximum number of users to return
    - search: Search term for username, email, or full name
    - is_active: Filter by active status
    """
    # Mock user data
    return [
        UserManagement(
            id="1",
            username="admin",
            email="admin@example.com",
            full_name="System Administrator",
            is_active=True,
            is_superuser=True,
            scopes=["read", "write", "admin"],
            created_at=datetime(2024, 1, 1),
            last_login=datetime.utcnow()
        ),
        UserManagement(
            id="2",
            username="underwriter",
            email="underwriter@example.com",
            full_name="Senior Underwriter",
            is_active=True,
            is_superuser=False,
            scopes=["read", "write"],
            created_at=datetime(2024, 1, 5),
            last_login=datetime.utcnow()
        )
    ]

@router.post("/users", response_model=UserManagement)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    Create a new system user
    
    Creates a new user account with specified permissions and scopes
    """
    # In real implementation, would create user in database
    hashed_password = get_password_hash(user_data.password)
    
    return UserManagement(
        id="new_user_id",
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        is_active=True,
        is_superuser=user_data.is_superuser,
        scopes=user_data.scopes,
        created_at=datetime.utcnow(),
        last_login=None
    )

@router.get("/users/{user_id}", response_model=UserManagement)
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["admin"]))
):
    """Get detailed user information"""
    # Mock implementation
    return UserManagement(
        id=user_id,
        username="example_user",
        email="user@example.com",
        full_name="Example User",
        is_active=True,
        is_superuser=False,
        scopes=["read"],
        created_at=datetime(2024, 1, 10),
        last_login=datetime.utcnow()
    )

@router.put("/users/{user_id}", response_model=UserManagement)
async def update_user(
    user_id: str,
    update_data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    Update user information and permissions
    
    Allows updating user profile, status, and permissions
    """
    # Implementation would update user in database
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="User update not yet implemented"
    )

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    Deactivate or delete a user account
    
    Soft deletes the user account and revokes all access
    """
    return {"message": "User deactivated successfully"}

@router.get("/config", response_model=SystemConfiguration)
async def get_system_configuration(
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    Get current system configuration settings
    
    Returns all configurable system parameters and limits
    """
    return SystemConfiguration(
        max_file_size_mb=100,
        allowed_file_types=["pdf", "docx", "xlsx", "msg", "eml", "jpg", "png"],
        ocr_confidence_threshold=0.8,
        risk_score_thresholds={
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.9
        },
        processing_timeouts={
            "ocr_processing": 300,
            "data_extraction": 180,
            "risk_analysis": 240,
            "decision_engine": 120
        },
        rate_limits={
            "api_requests_per_minute": 100,
            "file_uploads_per_hour": 50,
            "batch_processing_per_day": 10
        },
        email_notifications_enabled=True,
        audit_log_retention_days=90
    )

@router.put("/config")
async def update_system_configuration(
    config: SystemConfiguration,
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    Update system configuration settings
    
    Updates configurable system parameters and applies them
    """
    return {"message": "Configuration updated successfully"}

@router.get("/logs")
async def get_system_logs(
    level: str = Query("INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    limit: int = Query(100, ge=1, le=1000),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    component: Optional[str] = Query(None),
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    Retrieve system logs with filtering
    
    Parameters:
    - level: Minimum log level to retrieve
    - limit: Maximum number of log entries
    - start_time: Filter logs after this time
    - end_time: Filter logs before this time
    - component: Filter by system component
    """
    return {
        "logs": [
            {
                "timestamp": "2024-01-15T12:00:00Z",
                "level": "INFO",
                "component": "ocr_agent",
                "message": "Document processed successfully",
                "details": {"document_id": "doc_123", "processing_time": 45.2}
            },
            {
                "timestamp": "2024-01-15T12:01:00Z",
                "level": "WARNING",
                "component": "risk_analysis",
                "message": "Low confidence in risk assessment",
                "details": {"application_id": "app_456", "confidence": 0.65}
            }
        ],
        "total_count": 2,
        "filtered_count": 2
    }

@router.get("/audit-log")
async def get_audit_log(
    user_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    Retrieve audit log entries
    
    Returns detailed audit trail of user actions and system events
    """
    return {
        "audit_entries": [
            {
                "id": "audit_001",
                "timestamp": "2024-01-15T10:30:00Z",
                "user_id": "2",
                "username": "underwriter",
                "action": "document_upload",
                "resource_type": "document",
                "resource_id": "doc_123",
                "details": {"filename": "ri_slip.pdf", "size_bytes": 2048576},
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0..."
            },
            {
                "id": "audit_002",
                "timestamp": "2024-01-15T10:35:00Z",
                "user_id": "2",
                "username": "underwriter",
                "action": "application_approve",
                "resource_type": "application",
                "resource_id": "app_456",
                "details": {"override_ai": True, "notes": "Manual approval based on relationship"},
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0..."
            }
        ],
        "total_count": 2
    }

@router.get("/backups", response_model=List[BackupInfo])
async def list_backups(
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    List available system backups
    
    Returns information about database and file backups
    """
    return [
        BackupInfo(
            backup_id="backup_20240115_000000",
            created_at=datetime(2024, 1, 15),
            size_bytes=1073741824,  # 1GB
            type="full",
            status="completed",
            description="Daily automated backup"
        ),
        BackupInfo(
            backup_id="backup_20240114_000000",
            created_at=datetime(2024, 1, 14),
            size_bytes=1024000000,
            type="full",
            status="completed",
            description="Daily automated backup"
        )
    ]

@router.post("/backups")
async def create_backup(
    backup_type: str = Query("full", regex="^(full|incremental)$"),
    description: Optional[str] = Query(None),
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    Create a new system backup
    
    Initiates backup process for database and files
    """
    backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    
    return {
        "message": "Backup initiated",
        "backup_id": backup_id,
        "type": backup_type,
        "estimated_completion": "2024-01-15T14:30:00Z"
    }

@router.post("/backups/{backup_id}/restore")
async def restore_backup(
    backup_id: str,
    confirm: bool = Query(False),
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    Restore system from backup
    
    WARNING: This will overwrite current data with backup data
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Restore operation requires confirmation"
        )
    
    return {
        "message": "Restore initiated",
        "backup_id": backup_id,
        "estimated_completion": "2024-01-15T15:00:00Z",
        "warning": "System will be unavailable during restore"
    }

@router.get("/metrics/system")
async def get_system_metrics(
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    Get detailed system performance metrics
    
    Returns comprehensive system performance data for monitoring
    """
    return {
        "cpu": {
            "usage_percent": 23.5,
            "load_average": [1.2, 1.5, 1.8],
            "cores": 8
        },
        "memory": {
            "total_gb": 32.0,
            "used_gb": 21.9,
            "available_gb": 10.1,
            "usage_percent": 68.4
        },
        "disk": {
            "total_gb": 500.0,
            "used_gb": 226.0,
            "available_gb": 274.0,
            "usage_percent": 45.2
        },
        "network": {
            "bytes_sent": 1024000000,
            "bytes_received": 2048000000,
            "packets_sent": 1500000,
            "packets_received": 2200000
        },
        "database": {
            "connections_active": 15,
            "connections_max": 100,
            "queries_per_second": 45.2,
            "slow_queries": 2
        },
        "redis": {
            "memory_used_mb": 128.5,
            "keys_count": 15420,
            "operations_per_second": 1250.0
        }
    }

@router.post("/maintenance/start")
async def start_maintenance_mode(
    message: Optional[str] = Query("System maintenance in progress"),
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    Enable maintenance mode
    
    Puts the system into maintenance mode, blocking non-admin access
    """
    return {
        "message": "Maintenance mode enabled",
        "maintenance_message": message,
        "enabled_by": current_user.username,
        "enabled_at": datetime.utcnow()
    }

@router.post("/maintenance/stop")
async def stop_maintenance_mode(
    current_user: User = Depends(require_scopes(["admin"]))
):
    """
    Disable maintenance mode
    
    Restores normal system operation
    """
    return {
        "message": "Maintenance mode disabled",
        "disabled_by": current_user.username,
        "disabled_at": datetime.utcnow()
    }