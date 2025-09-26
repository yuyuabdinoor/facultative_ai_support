"""
Business Limits Service

Service layer for business limits validation functionality.
Provides high-level interface for business limits operations.
"""

import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session

from ..agents.business_limits_agent import BusinessLimitsAgent, LimitCheckResult
from ..models.schemas import (
    BusinessLimit, BusinessLimitCreate, BusinessLimitUpdate,
    RiskParameters, FinancialData, ValidationResult
)
from ..core.database import get_db


logger = logging.getLogger(__name__)


class BusinessLimitsService:
    """
    Service for managing business limits and validation
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.agent = BusinessLimitsAgent(db_session)
        self.logger = logging.getLogger(__name__)
    
    async def validate_application_limits(
        self,
        application_id: str,
        risk_parameters: Optional[RiskParameters] = None,
        financial_data: Optional[FinancialData] = None
    ) -> LimitCheckResult:
        """
        Validate an application against business limits
        
        Args:
            application_id: ID of the application to validate
            risk_parameters: Risk parameters (optional)
            financial_data: Financial data (optional)
            
        Returns:
            LimitCheckResult with validation results
        """
        try:
            self.logger.info(f"Validating business limits for application {application_id}")
            
            result = self.agent.validate_application(
                application_id=application_id,
                risk_parameters=risk_parameters,
                financial_data=financial_data
            )
            
            self.logger.info(
                f"Business limits validation completed for {application_id}. "
                f"Passed: {result.passed}, Violations: {len(result.violations)}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating business limits: {str(e)}")
            raise
    
    async def get_business_limits(
        self,
        limit_type: Optional[str] = None,
        active_only: bool = True
    ) -> List[BusinessLimit]:
        """
        Get business limits with optional filtering
        
        Args:
            limit_type: Filter by limit type
            active_only: Only return active limits
            
        Returns:
            List of business limits
        """
        try:
            return self.agent.get_business_limits(
                limit_type=limit_type,
                active_only=active_only
            )
        except Exception as e:
            self.logger.error(f"Error fetching business limits: {str(e)}")
            raise
    
    async def create_business_limit(self, limit_data: BusinessLimitCreate) -> BusinessLimit:
        """
        Create a new business limit
        
        Args:
            limit_data: Business limit data
            
        Returns:
            Created business limit
        """
        try:
            self.logger.info(f"Creating business limit: {limit_data.limit_type}")
            
            result = self.agent.create_business_limit(limit_data)
            
            self.logger.info(f"Created business limit: {result.id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating business limit: {str(e)}")
            raise
    
    async def update_business_limit(
        self,
        limit_id: str,
        limit_data: BusinessLimitUpdate
    ) -> Optional[BusinessLimit]:
        """
        Update an existing business limit
        
        Args:
            limit_id: ID of the limit to update
            limit_data: Updated limit data
            
        Returns:
            Updated business limit or None if not found
        """
        try:
            self.logger.info(f"Updating business limit: {limit_id}")
            
            result = self.agent.update_business_limit(limit_id, limit_data)
            
            if result:
                self.logger.info(f"Updated business limit: {limit_id}")
            else:
                self.logger.warning(f"Business limit not found: {limit_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error updating business limit: {str(e)}")
            raise
    
    async def delete_business_limit(self, limit_id: str) -> bool:
        """
        Delete a business limit (soft delete)
        
        Args:
            limit_id: ID of the limit to delete
            
        Returns:
            True if deleted successfully, False if not found
        """
        try:
            self.logger.info(f"Deleting business limit: {limit_id}")
            
            result = self.agent.delete_business_limit(limit_id)
            
            if result:
                self.logger.info(f"Deleted business limit: {limit_id}")
            else:
                self.logger.warning(f"Business limit not found: {limit_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error deleting business limit: {str(e)}")
            raise
    
    async def get_limit_violations_summary(self, application_id: str) -> Dict[str, Any]:
        """
        Get a summary of limit violations for an application
        
        Args:
            application_id: ID of the application
            
        Returns:
            Summary of violations
        """
        try:
            result = await self.validate_application_limits(application_id)
            
            # Group violations by type and severity
            violations_by_type = {}
            violations_by_severity = {}
            
            for violation in result.violations:
                # Group by type
                if violation.limit_type not in violations_by_type:
                    violations_by_type[violation.limit_type] = []
                violations_by_type[violation.limit_type].append(violation)
                
                # Group by severity
                severity = violation.violation_severity.value
                if severity not in violations_by_severity:
                    violations_by_severity[severity] = []
                violations_by_severity[severity].append(violation)
            
            return {
                "application_id": application_id,
                "overall_passed": result.passed,
                "total_violations": len(result.violations),
                "violations_by_type": {
                    limit_type: len(violations) 
                    for limit_type, violations in violations_by_type.items()
                },
                "violations_by_severity": {
                    severity: len(violations)
                    for severity, violations in violations_by_severity.items()
                },
                "warnings": result.warnings,
                "recommendations": result.recommendations,
                "metadata": result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error generating violations summary: {str(e)}")
            raise
    
    async def check_specific_limit_type(
        self,
        application_id: str,
        limit_type: str
    ) -> Dict[str, Any]:
        """
        Check a specific type of business limit for an application
        
        Args:
            application_id: ID of the application
            limit_type: Type of limit to check
            
        Returns:
            Specific limit check results
        """
        try:
            # Get full validation results
            result = await self.validate_application_limits(application_id)
            
            # Filter violations for specific limit type
            type_violations = [
                v for v in result.violations 
                if v.limit_type == limit_type
            ]
            
            return {
                "application_id": application_id,
                "limit_type": limit_type,
                "passed": len(type_violations) == 0,
                "violations": [
                    {
                        "severity": v.violation_severity.value,
                        "message": v.message,
                        "current_value": v.current_value,
                        "limit_value": v.limit_value,
                        "recommendation": v.recommendation,
                        "metadata": v.metadata
                    }
                    for v in type_violations
                ],
                "violation_count": len(type_violations)
            }
            
        except Exception as e:
            self.logger.error(f"Error checking specific limit type: {str(e)}")
            raise
    
    async def get_limit_configuration_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current business limit configuration
        
        Returns:
            Summary of limit configuration
        """
        try:
            all_limits = await self.get_business_limits(active_only=False)
            active_limits = await self.get_business_limits(active_only=True)
            
            # Group limits by type
            limits_by_type = {}
            for limit in active_limits:
                if limit.limit_type not in limits_by_type:
                    limits_by_type[limit.limit_type] = []
                limits_by_type[limit.limit_type].append(limit)
            
            return {
                "total_limits": len(all_limits),
                "active_limits": len(active_limits),
                "inactive_limits": len(all_limits) - len(active_limits),
                "limits_by_type": {
                    limit_type: len(limits)
                    for limit_type, limits in limits_by_type.items()
                },
                "limit_types": list(limits_by_type.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Error generating limit configuration summary: {str(e)}")
            raise


def get_business_limits_service(db: Session = None) -> BusinessLimitsService:
    """
    Get BusinessLimitsService instance
    
    Args:
        db: Database session (optional, will create if not provided)
        
    Returns:
        BusinessLimitsService instance
    """
    if db is None:
        db = next(get_db())
    
    return BusinessLimitsService(db)