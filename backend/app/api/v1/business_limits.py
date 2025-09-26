"""
Business Limits API Endpoints

FastAPI endpoints for business limits validation and management.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ...core.database import get_db
from ...services.business_limits_service import BusinessLimitsService
from ...models.schemas import (
    BusinessLimit, BusinessLimitCreate, BusinessLimitUpdate,
    ValidationResult
)
from ...agents.business_limits_agent import LimitCheckResult


router = APIRouter(prefix="/business-limits", tags=["business-limits"])


def get_business_limits_service(db: Session = Depends(get_db)) -> BusinessLimitsService:
    """Get business limits service instance"""
    return BusinessLimitsService(db)


@router.post("/validate/{application_id}")
async def validate_application_limits(
    application_id: str,
    service: BusinessLimitsService = Depends(get_business_limits_service)
) -> Dict[str, Any]:
    """
    Validate an application against business limits
    
    Args:
        application_id: ID of the application to validate
        
    Returns:
        Validation results with violations and recommendations
    """
    try:
        result = await service.validate_application_limits(application_id)
        
        return {
            "application_id": application_id,
            "passed": result.passed,
            "violations": [
                {
                    "limit_type": v.limit_type,
                    "limit_id": v.limit_id,
                    "severity": v.violation_severity.value,
                    "message": v.message,
                    "current_value": v.current_value,
                    "limit_value": v.limit_value,
                    "recommendation": v.recommendation,
                    "metadata": v.metadata
                }
                for v in result.violations
            ],
            "warnings": result.warnings,
            "recommendations": result.recommendations,
            "metadata": result.metadata
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating business limits: {str(e)}"
        )


@router.get("/validate/{application_id}/summary")
async def get_validation_summary(
    application_id: str,
    service: BusinessLimitsService = Depends(get_business_limits_service)
) -> Dict[str, Any]:
    """
    Get a summary of limit violations for an application
    
    Args:
        application_id: ID of the application
        
    Returns:
        Summary of violations grouped by type and severity
    """
    try:
        return await service.get_limit_violations_summary(application_id)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating validation summary: {str(e)}"
        )


@router.get("/validate/{application_id}/type/{limit_type}")
async def check_specific_limit_type(
    application_id: str,
    limit_type: str,
    service: BusinessLimitsService = Depends(get_business_limits_service)
) -> Dict[str, Any]:
    """
    Check a specific type of business limit for an application
    
    Args:
        application_id: ID of the application
        limit_type: Type of limit to check
        
    Returns:
        Specific limit type validation results
    """
    try:
        return await service.check_specific_limit_type(application_id, limit_type)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking specific limit type: {str(e)}"
        )


@router.get("/", response_model=List[BusinessLimit])
async def get_business_limits(
    limit_type: Optional[str] = None,
    active_only: bool = True,
    service: BusinessLimitsService = Depends(get_business_limits_service)
) -> List[BusinessLimit]:
    """
    Get business limits with optional filtering
    
    Args:
        limit_type: Filter by limit type (optional)
        active_only: Only return active limits (default: True)
        
    Returns:
        List of business limits
    """
    try:
        return await service.get_business_limits(
            limit_type=limit_type,
            active_only=active_only
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching business limits: {str(e)}"
        )


@router.post("/", response_model=BusinessLimit)
async def create_business_limit(
    limit_data: BusinessLimitCreate,
    service: BusinessLimitsService = Depends(get_business_limits_service)
) -> BusinessLimit:
    """
    Create a new business limit
    
    Args:
        limit_data: Business limit data
        
    Returns:
        Created business limit
    """
    try:
        return await service.create_business_limit(limit_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating business limit: {str(e)}"
        )


@router.put("/{limit_id}", response_model=BusinessLimit)
async def update_business_limit(
    limit_id: str,
    limit_data: BusinessLimitUpdate,
    service: BusinessLimitsService = Depends(get_business_limits_service)
) -> BusinessLimit:
    """
    Update an existing business limit
    
    Args:
        limit_id: ID of the limit to update
        limit_data: Updated limit data
        
    Returns:
        Updated business limit
    """
    try:
        result = await service.update_business_limit(limit_id, limit_data)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Business limit not found: {limit_id}"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating business limit: {str(e)}"
        )


@router.delete("/{limit_id}")
async def delete_business_limit(
    limit_id: str,
    service: BusinessLimitsService = Depends(get_business_limits_service)
) -> Dict[str, str]:
    """
    Delete a business limit (soft delete)
    
    Args:
        limit_id: ID of the limit to delete
        
    Returns:
        Success message
    """
    try:
        result = await service.delete_business_limit(limit_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Business limit not found: {limit_id}"
            )
        
        return {"message": f"Business limit {limit_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting business limit: {str(e)}"
        )


@router.get("/configuration/summary")
async def get_configuration_summary(
    service: BusinessLimitsService = Depends(get_business_limits_service)
) -> Dict[str, Any]:
    """
    Get a summary of current business limit configuration
    
    Returns:
        Summary of limit configuration including counts by type
    """
    try:
        return await service.get_limit_configuration_summary()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating configuration summary: {str(e)}"
        )


@router.get("/types")
async def get_limit_types() -> Dict[str, List[str]]:
    """
    Get available business limit types
    
    Returns:
        Dictionary of available limit types and their descriptions
    """
    from ...agents.business_limits_agent import LimitType
    
    return {
        "limit_types": [limit_type.value for limit_type in LimitType],
        "descriptions": {
            LimitType.ASSET_VALUE.value: "Maximum asset value limits",
            LimitType.COVERAGE_LIMIT.value: "Maximum coverage amount limits",
            LimitType.GEOGRAPHIC_EXPOSURE.value: "Regional exposure limits",
            LimitType.INDUSTRY_SECTOR.value: "Industry sector restrictions",
            LimitType.CONSTRUCTION_TYPE.value: "Construction type restrictions",
            LimitType.OCCUPANCY_TYPE.value: "Occupancy type restrictions",
            LimitType.FINANCIAL_STRENGTH.value: "Financial strength requirements",
            LimitType.REGULATORY_COMPLIANCE.value: "Regulatory compliance rules",
            LimitType.CATASTROPHE_EXPOSURE.value: "Catastrophe exposure limits",
            LimitType.AGGREGATE_EXPOSURE.value: "Aggregate portfolio exposure limits"
        }
    }


@router.get("/severity-levels")
async def get_severity_levels() -> Dict[str, List[str]]:
    """
    Get available violation severity levels
    
    Returns:
        Dictionary of severity levels and their descriptions
    """
    from ...agents.business_limits_agent import ViolationSeverity
    
    return {
        "severity_levels": [severity.value for severity in ViolationSeverity],
        "descriptions": {
            ViolationSeverity.INFO.value: "Informational - no action required",
            ViolationSeverity.WARNING.value: "Warning - review recommended",
            ViolationSeverity.ERROR.value: "Error - action required",
            ViolationSeverity.CRITICAL.value: "Critical - immediate action required"
        }
    }