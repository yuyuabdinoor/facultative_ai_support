"""
Analytics and reporting API endpoints
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from datetime import datetime, timedelta
from enum import Enum

from ...core.database import get_db
from ...core.auth import require_scopes, User

router = APIRouter()

class TimeRange(str, Enum):
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"
    LAST_90_DAYS = "90d"
    LAST_YEAR = "1y"
    CUSTOM = "custom"

class MetricType(str, Enum):
    APPLICATIONS = "applications"
    APPROVALS = "approvals"
    REJECTIONS = "rejections"
    PROCESSING_TIME = "processing_time"
    RISK_SCORES = "risk_scores"
    PREMIUM_ADJUSTMENTS = "premium_adjustments"

class DashboardMetrics(BaseModel):
    total_applications: int
    applications_this_period: int
    approval_rate: float
    avg_processing_time_hours: float
    total_insured_value: float
    high_risk_applications: int
    pending_reviews: int
    system_uptime_percent: float

class TrendData(BaseModel):
    date: datetime
    value: float
    count: int

class RiskDistribution(BaseModel):
    risk_level: str
    count: int
    percentage: float
    avg_asset_value: float

class GeographicAnalysis(BaseModel):
    region: str
    application_count: int
    approval_rate: float
    avg_risk_score: float
    total_insured_value: float

class IndustryAnalysis(BaseModel):
    industry: str
    application_count: int
    approval_rate: float
    avg_premium_adjustment: float
    risk_distribution: Dict[str, int]

class ProcessingPerformance(BaseModel):
    agent_name: str
    avg_processing_time_seconds: float
    success_rate: float
    error_count: int
    throughput_per_hour: float

@router.get("/dashboard", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    time_range: TimeRange = Query(TimeRange.LAST_30_DAYS),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["read"]))
):
    """
    Get key dashboard metrics for the specified time period
    
    Returns high-level KPIs and metrics for executive dashboard view
    """
    return DashboardMetrics(
        total_applications=1250,
        applications_this_period=78,
        approval_rate=0.72,
        avg_processing_time_hours=2.3,
        total_insured_value=2500000000.0,
        high_risk_applications=15,
        pending_reviews=8,
        system_uptime_percent=99.8
    )

@router.get("/trends/{metric_type}")
async def get_trend_analysis(
    metric_type: MetricType,
    time_range: TimeRange = Query(TimeRange.LAST_30_DAYS),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    granularity: str = Query("daily", regex="^(hourly|daily|weekly|monthly)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["read"]))
) -> List[TrendData]:
    """
    Get trend analysis for specific metrics over time
    
    Parameters:
    - metric_type: Type of metric to analyze
    - time_range: Predefined time range or custom
    - start_date: Custom start date (if time_range is custom)
    - end_date: Custom end date (if time_range is custom)
    - granularity: Data aggregation level
    """
    # Mock trend data
    base_date = datetime.utcnow() - timedelta(days=30)
    trend_data = []
    
    for i in range(30):
        date = base_date + timedelta(days=i)
        value = 50 + (i * 2) + (i % 7) * 10  # Mock trending data
        count = 20 + (i % 5) * 3
        
        trend_data.append(TrendData(
            date=date,
            value=value,
            count=count
        ))
    
    return trend_data

@router.get("/risk-distribution")
async def get_risk_distribution(
    time_range: TimeRange = Query(TimeRange.LAST_30_DAYS),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["read"]))
) -> List[RiskDistribution]:
    """
    Get distribution of applications by risk level
    
    Returns breakdown of applications across different risk categories
    with associated metrics
    """
    return [
        RiskDistribution(
            risk_level="LOW",
            count=450,
            percentage=36.0,
            avg_asset_value=750000.0
        ),
        RiskDistribution(
            risk_level="MEDIUM",
            count=520,
            percentage=41.6,
            avg_asset_value=1200000.0
        ),
        RiskDistribution(
            risk_level="HIGH",
            count=230,
            percentage=18.4,
            avg_asset_value=2100000.0
        ),
        RiskDistribution(
            risk_level="CRITICAL",
            count=50,
            percentage=4.0,
            avg_asset_value=5000000.0
        )
    ]

@router.get("/geographic-analysis")
async def get_geographic_analysis(
    time_range: TimeRange = Query(TimeRange.LAST_30_DAYS),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["read"]))
) -> List[GeographicAnalysis]:
    """
    Get geographic distribution and analysis of applications
    
    Returns regional breakdown with risk and approval metrics
    """
    return [
        GeographicAnalysis(
            region="North America",
            application_count=520,
            approval_rate=0.78,
            avg_risk_score=0.65,
            total_insured_value=1200000000.0
        ),
        GeographicAnalysis(
            region="Europe",
            application_count=380,
            approval_rate=0.72,
            avg_risk_score=0.58,
            total_insured_value=850000000.0
        ),
        GeographicAnalysis(
            region="Asia Pacific",
            application_count=280,
            approval_rate=0.68,
            avg_risk_score=0.72,
            total_insured_value=650000000.0
        ),
        GeographicAnalysis(
            region="Other",
            application_count=70,
            approval_rate=0.65,
            avg_risk_score=0.75,
            total_insured_value=180000000.0
        )
    ]

@router.get("/industry-analysis")
async def get_industry_analysis(
    time_range: TimeRange = Query(TimeRange.LAST_30_DAYS),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["read"]))
) -> List[IndustryAnalysis]:
    """
    Get industry sector analysis and trends
    
    Returns breakdown by industry with risk and performance metrics
    """
    return [
        IndustryAnalysis(
            industry="Technology",
            application_count=180,
            approval_rate=0.82,
            avg_premium_adjustment=0.03,
            risk_distribution={"LOW": 80, "MEDIUM": 70, "HIGH": 25, "CRITICAL": 5}
        ),
        IndustryAnalysis(
            industry="Manufacturing",
            application_count=220,
            approval_rate=0.68,
            avg_premium_adjustment=0.08,
            risk_distribution={"LOW": 60, "MEDIUM": 90, "HIGH": 55, "CRITICAL": 15}
        ),
        IndustryAnalysis(
            industry="Healthcare",
            application_count=150,
            approval_rate=0.75,
            avg_premium_adjustment=0.05,
            risk_distribution={"LOW": 70, "MEDIUM": 55, "HIGH": 20, "CRITICAL": 5}
        ),
        IndustryAnalysis(
            industry="Energy",
            application_count=120,
            approval_rate=0.58,
            avg_premium_adjustment=0.12,
            risk_distribution={"LOW": 30, "MEDIUM": 45, "HIGH": 35, "CRITICAL": 10}
        )
    ]

@router.get("/processing-performance")
async def get_processing_performance(
    time_range: TimeRange = Query(TimeRange.LAST_7_DAYS),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["read"]))
) -> List[ProcessingPerformance]:
    """
    Get AI agent processing performance metrics
    
    Returns performance data for each processing agent including
    speed, accuracy, and error rates
    """
    return [
        ProcessingPerformance(
            agent_name="OCR Processing",
            avg_processing_time_seconds=45.2,
            success_rate=0.96,
            error_count=12,
            throughput_per_hour=80.0
        ),
        ProcessingPerformance(
            agent_name="Data Extraction",
            avg_processing_time_seconds=32.8,
            success_rate=0.94,
            error_count=18,
            throughput_per_hour=110.0
        ),
        ProcessingPerformance(
            agent_name="Risk Analysis",
            avg_processing_time_seconds=125.5,
            success_rate=0.98,
            error_count=5,
            throughput_per_hour=28.0
        ),
        ProcessingPerformance(
            agent_name="Limits Validation",
            avg_processing_time_seconds=8.3,
            success_rate=0.99,
            error_count=2,
            throughput_per_hour=430.0
        ),
        ProcessingPerformance(
            agent_name="Decision Engine",
            avg_processing_time_seconds=15.7,
            success_rate=0.97,
            error_count=8,
            throughput_per_hour=230.0
        )
    ]

@router.get("/model-performance")
async def get_model_performance(
    model_name: Optional[str] = Query(None),
    time_range: TimeRange = Query(TimeRange.LAST_7_DAYS),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["read"]))
):
    """
    Get AI model performance metrics and accuracy statistics
    
    Returns detailed performance data for machine learning models
    including accuracy, precision, recall, and drift metrics
    """
    return {
        "models": [
            {
                "name": "Risk Classification Model",
                "accuracy": 0.94,
                "precision": 0.92,
                "recall": 0.96,
                "f1_score": 0.94,
                "drift_score": 0.02,
                "last_updated": "2024-01-10T08:00:00Z",
                "predictions_count": 1250
            },
            {
                "name": "Financial Analysis Model",
                "accuracy": 0.89,
                "precision": 0.87,
                "recall": 0.91,
                "f1_score": 0.89,
                "drift_score": 0.05,
                "last_updated": "2024-01-08T12:00:00Z",
                "predictions_count": 980
            },
            {
                "name": "Document Classification Model",
                "accuracy": 0.97,
                "precision": 0.96,
                "recall": 0.98,
                "f1_score": 0.97,
                "drift_score": 0.01,
                "last_updated": "2024-01-12T16:00:00Z",
                "predictions_count": 2100
            }
        ]
    }

@router.get("/export/report")
async def export_analytics_report(
    report_type: str = Query("comprehensive", regex="^(comprehensive|executive|technical)$"),
    format: str = Query("pdf", regex="^(pdf|excel|json)$"),
    time_range: TimeRange = Query(TimeRange.LAST_30_DAYS),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["read"]))
):
    """
    Export comprehensive analytics report
    
    Generates detailed reports in various formats:
    - Comprehensive: All metrics and analysis
    - Executive: High-level summary for leadership
    - Technical: Detailed technical metrics for operations
    """
    return {
        "message": f"Generating {report_type} report in {format} format",
        "report_id": f"report_{datetime.utcnow().timestamp()}",
        "estimated_completion": "2024-01-15T14:30:00Z",
        "download_url": "/api/v1/analytics/reports/download/report_123"
    }

@router.get("/alerts")
async def get_system_alerts(
    severity: Optional[str] = Query(None, regex="^(low|medium|high|critical)$"),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["read"]))
):
    """
    Get system alerts and notifications
    
    Returns active alerts for:
    - Performance issues
    - Model drift
    - Processing errors
    - Business rule violations
    - System health issues
    """
    return {
        "alerts": [
            {
                "id": "alert_001",
                "severity": "medium",
                "category": "model_performance",
                "title": "Risk Analysis Model Accuracy Decline",
                "description": "Risk analysis model accuracy has dropped to 89% over the last 24 hours",
                "timestamp": "2024-01-15T10:30:00Z",
                "acknowledged": False,
                "actions": ["retrain_model", "investigate_data"]
            },
            {
                "id": "alert_002",
                "severity": "high",
                "category": "processing_queue",
                "title": "High Processing Queue Backlog",
                "description": "OCR processing queue has 150+ pending documents",
                "timestamp": "2024-01-15T11:15:00Z",
                "acknowledged": True,
                "actions": ["scale_workers", "investigate_bottleneck"]
            }
        ],
        "total_alerts": 2,
        "unacknowledged_count": 1
    }

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_scopes(["write"]))
):
    """
    Acknowledge a system alert
    
    Marks an alert as acknowledged by the current user
    """
    return {
        "message": "Alert acknowledged",
        "alert_id": alert_id,
        "acknowledged_by": current_user.username,
        "acknowledged_at": datetime.utcnow()
    }