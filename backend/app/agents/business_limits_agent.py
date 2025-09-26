"""
Business Limits Validation Agent

This agent validates reinsurance applications against business limits and constraints.
It checks various types of limits including geographic exposure, industry sector restrictions,
and regulatory compliance requirements.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..models.database import BusinessLimit, Application, RiskParameters, FinancialData
from ..models.schemas import (
    BusinessLimitCreate, BusinessLimitUpdate, BusinessLimit as BusinessLimitSchema,
    ValidationResult, RiskParameters as RiskParametersSchema,
    FinancialData as FinancialDataSchema
)
from ..core.database import get_db


logger = logging.getLogger(__name__)


class LimitType(str, Enum):
    """Types of business limits"""
    ASSET_VALUE = "asset_value"
    COVERAGE_LIMIT = "coverage_limit"
    GEOGRAPHIC_EXPOSURE = "geographic_exposure"
    INDUSTRY_SECTOR = "industry_sector"
    CONSTRUCTION_TYPE = "construction_type"
    OCCUPANCY_TYPE = "occupancy_type"
    FINANCIAL_STRENGTH = "financial_strength"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    CATASTROPHE_EXPOSURE = "catastrophe_exposure"
    AGGREGATE_EXPOSURE = "aggregate_exposure"


class ViolationSeverity(str, Enum):
    """Severity levels for limit violations"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LimitViolation:
    """Represents a business limit violation"""
    limit_type: str
    limit_id: str
    violation_severity: ViolationSeverity
    message: str
    current_value: Optional[Any] = None
    limit_value: Optional[Any] = None
    recommendation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LimitCheckResult:
    """Result of business limit checking"""
    passed: bool
    violations: List[LimitViolation]
    warnings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


class BusinessLimitsAgent:
    """
    Agent responsible for validating applications against business limits and constraints
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = logging.getLogger(__name__)
        
        # Geographic regions mapping
        self.geographic_regions = {
            "north_america": ["usa", "canada", "mexico", "united states", "north america"],
            "europe": ["uk", "germany", "france", "italy", "spain", "netherlands", "europe", "european union"],
            "asia_pacific": ["japan", "australia", "singapore", "hong kong", "china", "asia", "pacific"],
            "middle_east": ["uae", "saudi arabia", "qatar", "kuwait", "middle east"],
            "africa": ["south africa", "egypt", "nigeria", "africa"],
            "latin_america": ["brazil", "argentina", "chile", "colombia", "latin america", "south america"]
        }
        
        # Industry sector classifications
        self.industry_sectors = {
            "energy": ["oil", "gas", "petroleum", "energy", "power", "electricity", "renewable"],
            "manufacturing": ["manufacturing", "automotive", "steel", "chemicals", "pharmaceuticals"],
            "construction": ["construction", "infrastructure", "building", "civil engineering"],
            "marine": ["marine", "shipping", "cargo", "vessel", "port", "offshore"],
            "aviation": ["aviation", "aircraft", "airline", "airport", "aerospace"],
            "technology": ["technology", "software", "telecommunications", "data center"],
            "healthcare": ["healthcare", "hospital", "medical", "pharmaceutical"],
            "financial": ["bank", "financial", "insurance", "investment"],
            "retail": ["retail", "shopping", "commercial", "warehouse"],
            "agriculture": ["agriculture", "farming", "crop", "livestock"]
        }
    
    def validate_application(
        self, 
        application_id: str,
        risk_parameters: Optional[RiskParametersSchema] = None,
        financial_data: Optional[FinancialDataSchema] = None
    ) -> LimitCheckResult:
        """
        Validate an application against all applicable business limits
        
        Args:
            application_id: ID of the application to validate
            risk_parameters: Risk parameters (optional, will fetch if not provided)
            financial_data: Financial data (optional, will fetch if not provided)
            
        Returns:
            LimitCheckResult with validation results
        """
        try:
            self.logger.info(f"Starting business limits validation for application {application_id}")
            
            # Fetch application data if not provided
            if not risk_parameters or not financial_data:
                app_data = self._fetch_application_data(application_id)
                risk_parameters = risk_parameters or app_data.get("risk_parameters")
                financial_data = financial_data or app_data.get("financial_data")
            
            if not risk_parameters:
                return LimitCheckResult(
                    passed=False,
                    violations=[LimitViolation(
                        limit_type="data_availability",
                        limit_id="risk_parameters",
                        violation_severity=ViolationSeverity.ERROR,
                        message="Risk parameters not available for validation"
                    )],
                    warnings=[],
                    recommendations=["Ensure risk parameters are extracted before validation"],
                    metadata={"application_id": application_id}
                )
            
            violations = []
            warnings = []
            recommendations = []
            
            # Perform all limit checks
            violations.extend(self._check_asset_value_limits(risk_parameters))
            violations.extend(self._check_coverage_limits(risk_parameters))
            violations.extend(self._check_geographic_limits(risk_parameters))
            violations.extend(self._check_industry_sector_limits(risk_parameters))
            violations.extend(self._check_construction_type_limits(risk_parameters))
            violations.extend(self._check_occupancy_limits(risk_parameters))
            
            if financial_data:
                violations.extend(self._check_financial_strength_limits(financial_data))
            
            violations.extend(self._check_regulatory_compliance(risk_parameters, financial_data))
            violations.extend(self._check_catastrophe_exposure(risk_parameters))
            violations.extend(self._check_aggregate_exposure(application_id, risk_parameters))
            
            # Generate warnings and recommendations
            warnings, recommendations = self._generate_warnings_and_recommendations(violations)
            
            # Determine overall pass/fail
            critical_violations = [v for v in violations if v.violation_severity in [ViolationSeverity.ERROR, ViolationSeverity.CRITICAL]]
            passed = len(critical_violations) == 0
            
            result = LimitCheckResult(
                passed=passed,
                violations=violations,
                warnings=warnings,
                recommendations=recommendations,
                metadata={
                    "application_id": application_id,
                    "validation_timestamp": datetime.utcnow().isoformat(),
                    "total_violations": len(violations),
                    "critical_violations": len(critical_violations)
                }
            )
            
            self.logger.info(f"Business limits validation completed for application {application_id}. Passed: {passed}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during business limits validation: {str(e)}")
            return LimitCheckResult(
                passed=False,
                violations=[LimitViolation(
                    limit_type="system_error",
                    limit_id="validation_error",
                    violation_severity=ViolationSeverity.CRITICAL,
                    message=f"System error during validation: {str(e)}"
                )],
                warnings=[],
                recommendations=["Contact system administrator"],
                metadata={"application_id": application_id, "error": str(e)}
            )
    
    def _fetch_application_data(self, application_id: str) -> Dict[str, Any]:
        """Fetch application data from database"""
        try:
            application = self.db.query(Application).filter(Application.id == application_id).first()
            if not application:
                return {}
            
            return {
                "risk_parameters": application.risk_parameters,
                "financial_data": application.financial_data,
                "application": application
            }
        except Exception as e:
            self.logger.error(f"Error fetching application data: {str(e)}")
            return {}
    
    def _check_asset_value_limits(self, risk_parameters: RiskParametersSchema) -> List[LimitViolation]:
        """Check asset value against business limits"""
        violations = []
        
        if not risk_parameters.asset_value:
            return violations
        
        try:
            # Get asset value limits
            limits = self.db.query(BusinessLimit).filter(
                and_(
                    BusinessLimit.limit_type == LimitType.ASSET_VALUE,
                    BusinessLimit.active == True
                )
            ).all()
            
            for limit in limits:
                if limit.max_amount and risk_parameters.asset_value > limit.max_amount:
                    violations.append(LimitViolation(
                        limit_type=LimitType.ASSET_VALUE,
                        limit_id=str(limit.id),
                        violation_severity=ViolationSeverity.ERROR,
                        message=f"Asset value {risk_parameters.asset_value} exceeds maximum limit of {limit.max_amount}",
                        current_value=float(risk_parameters.asset_value),
                        limit_value=float(limit.max_amount),
                        recommendation="Consider reducing asset value or seeking senior approval"
                    ))
        
        except Exception as e:
            self.logger.error(f"Error checking asset value limits: {str(e)}")
        
        return violations
    
    def _check_coverage_limits(self, risk_parameters: RiskParametersSchema) -> List[LimitViolation]:
        """Check coverage limits against business constraints"""
        violations = []
        
        if not risk_parameters.coverage_limit:
            return violations
        
        try:
            # Get coverage limits
            limits = self.db.query(BusinessLimit).filter(
                and_(
                    BusinessLimit.limit_type == LimitType.COVERAGE_LIMIT,
                    BusinessLimit.active == True
                )
            ).all()
            
            for limit in limits:
                if limit.max_amount and risk_parameters.coverage_limit > limit.max_amount:
                    violations.append(LimitViolation(
                        limit_type=LimitType.COVERAGE_LIMIT,
                        limit_id=str(limit.id),
                        violation_severity=ViolationSeverity.ERROR,
                        message=f"Coverage limit {risk_parameters.coverage_limit} exceeds maximum limit of {limit.max_amount}",
                        current_value=float(risk_parameters.coverage_limit),
                        limit_value=float(limit.max_amount),
                        recommendation="Consider reducing coverage limit or applying for exception"
                    ))
        
        except Exception as e:
            self.logger.error(f"Error checking coverage limits: {str(e)}")
        
        return violations
    
    def _check_geographic_limits(self, risk_parameters: RiskParametersSchema) -> List[LimitViolation]:
        """Check geographic exposure against regional limits"""
        violations = []
        
        if not risk_parameters.location:
            return violations
        
        try:
            location_lower = risk_parameters.location.lower()
            detected_region = self._detect_geographic_region(location_lower)
            
            if detected_region:
                # Get geographic limits for this region
                limits = self.db.query(BusinessLimit).filter(
                    and_(
                        BusinessLimit.limit_type == LimitType.GEOGRAPHIC_EXPOSURE,
                        BusinessLimit.geographic_region == detected_region,
                        BusinessLimit.active == True
                    )
                ).all()
                
                for limit in limits:
                    if limit.max_amount and risk_parameters.asset_value and risk_parameters.asset_value > limit.max_amount:
                        violations.append(LimitViolation(
                            limit_type=LimitType.GEOGRAPHIC_EXPOSURE,
                            limit_id=str(limit.id),
                            violation_severity=ViolationSeverity.WARNING,
                            message=f"Asset value {risk_parameters.asset_value} in region '{detected_region}' exceeds regional limit of {limit.max_amount}",
                            current_value=float(risk_parameters.asset_value),
                            limit_value=float(limit.max_amount),
                            recommendation=f"Review regional exposure limits for {detected_region}",
                            metadata={"detected_region": detected_region, "location": risk_parameters.location}
                        ))
        
        except Exception as e:
            self.logger.error(f"Error checking geographic limits: {str(e)}")
        
        return violations
    
    def _check_industry_sector_limits(self, risk_parameters: RiskParametersSchema) -> List[LimitViolation]:
        """Check industry sector restrictions"""
        violations = []
        
        if not risk_parameters.industry_sector:
            return violations
        
        try:
            sector_lower = risk_parameters.industry_sector.lower()
            detected_sector = self._detect_industry_sector(sector_lower)
            
            if detected_sector:
                # Get industry sector limits
                limits = self.db.query(BusinessLimit).filter(
                    and_(
                        BusinessLimit.limit_type == LimitType.INDUSTRY_SECTOR,
                        BusinessLimit.industry_sector == detected_sector,
                        BusinessLimit.active == True
                    )
                ).all()
                
                for limit in limits:
                    # Check if sector is restricted
                    if limit.max_amount == 0:
                        violations.append(LimitViolation(
                            limit_type=LimitType.INDUSTRY_SECTOR,
                            limit_id=str(limit.id),
                            violation_severity=ViolationSeverity.CRITICAL,
                            message=f"Industry sector '{detected_sector}' is restricted",
                            current_value=detected_sector,
                            limit_value=0,
                            recommendation="This industry sector is not acceptable for reinsurance",
                            metadata={"detected_sector": detected_sector, "original_sector": risk_parameters.industry_sector}
                        ))
                    elif limit.max_amount and risk_parameters.asset_value and risk_parameters.asset_value > limit.max_amount:
                        violations.append(LimitViolation(
                            limit_type=LimitType.INDUSTRY_SECTOR,
                            limit_id=str(limit.id),
                            violation_severity=ViolationSeverity.WARNING,
                            message=f"Asset value {risk_parameters.asset_value} in sector '{detected_sector}' exceeds sector limit of {limit.max_amount}",
                            current_value=float(risk_parameters.asset_value),
                            limit_value=float(limit.max_amount),
                            recommendation=f"Review sector exposure limits for {detected_sector}",
                            metadata={"detected_sector": detected_sector, "original_sector": risk_parameters.industry_sector}
                        ))
        
        except Exception as e:
            self.logger.error(f"Error checking industry sector limits: {str(e)}")
        
        return violations
    
    def _check_construction_type_limits(self, risk_parameters: RiskParametersSchema) -> List[LimitViolation]:
        """Check construction type restrictions"""
        violations = []
        
        if not risk_parameters.construction_type:
            return violations
        
        try:
            # Get construction type limits
            limits = self.db.query(BusinessLimit).filter(
                and_(
                    BusinessLimit.limit_type == LimitType.CONSTRUCTION_TYPE,
                    BusinessLimit.active == True
                )
            ).all()
            
            construction_lower = risk_parameters.construction_type.lower()
            
            for limit in limits:
                if limit.category and limit.category.lower() in construction_lower:
                    if limit.max_amount == 0:
                        violations.append(LimitViolation(
                            limit_type=LimitType.CONSTRUCTION_TYPE,
                            limit_id=str(limit.id),
                            violation_severity=ViolationSeverity.ERROR,
                            message=f"Construction type '{risk_parameters.construction_type}' is restricted",
                            current_value=risk_parameters.construction_type,
                            limit_value=0,
                            recommendation="This construction type is not acceptable"
                        ))
                    elif limit.max_amount and risk_parameters.asset_value and risk_parameters.asset_value > limit.max_amount:
                        violations.append(LimitViolation(
                            limit_type=LimitType.CONSTRUCTION_TYPE,
                            limit_id=str(limit.id),
                            violation_severity=ViolationSeverity.WARNING,
                            message=f"Asset value {risk_parameters.asset_value} for construction type '{risk_parameters.construction_type}' exceeds limit of {limit.max_amount}",
                            current_value=float(risk_parameters.asset_value),
                            limit_value=float(limit.max_amount),
                            recommendation="Review construction type exposure limits"
                        ))
        
        except Exception as e:
            self.logger.error(f"Error checking construction type limits: {str(e)}")
        
        return violations
    
    def _check_occupancy_limits(self, risk_parameters: RiskParametersSchema) -> List[LimitViolation]:
        """Check occupancy type restrictions"""
        violations = []
        
        if not risk_parameters.occupancy:
            return violations
        
        try:
            # Get occupancy limits
            limits = self.db.query(BusinessLimit).filter(
                and_(
                    BusinessLimit.limit_type == LimitType.OCCUPANCY_TYPE,
                    BusinessLimit.active == True
                )
            ).all()
            
            occupancy_lower = risk_parameters.occupancy.lower()
            
            for limit in limits:
                if limit.category and limit.category.lower() in occupancy_lower:
                    if limit.max_amount == 0:
                        violations.append(LimitViolation(
                            limit_type=LimitType.OCCUPANCY_TYPE,
                            limit_id=str(limit.id),
                            violation_severity=ViolationSeverity.ERROR,
                            message=f"Occupancy type '{risk_parameters.occupancy}' is restricted",
                            current_value=risk_parameters.occupancy,
                            limit_value=0,
                            recommendation="This occupancy type is not acceptable"
                        ))
                    elif limit.max_amount and risk_parameters.asset_value and risk_parameters.asset_value > limit.max_amount:
                        violations.append(LimitViolation(
                            limit_type=LimitType.OCCUPANCY_TYPE,
                            limit_id=str(limit.id),
                            violation_severity=ViolationSeverity.WARNING,
                            message=f"Asset value {risk_parameters.asset_value} for occupancy '{risk_parameters.occupancy}' exceeds limit of {limit.max_amount}",
                            current_value=float(risk_parameters.asset_value),
                            limit_value=float(limit.max_amount),
                            recommendation="Review occupancy type exposure limits"
                        ))
        
        except Exception as e:
            self.logger.error(f"Error checking occupancy limits: {str(e)}")
        
        return violations
    
    def _check_financial_strength_limits(self, financial_data: FinancialDataSchema) -> List[LimitViolation]:
        """Check financial strength requirements"""
        violations = []
        
        try:
            # Get financial strength limits
            limits = self.db.query(BusinessLimit).filter(
                and_(
                    BusinessLimit.limit_type == LimitType.FINANCIAL_STRENGTH,
                    BusinessLimit.active == True
                )
            ).all()
            
            for limit in limits:
                # Check credit rating requirements
                if limit.category == "credit_rating" and financial_data.credit_rating:
                    if not self._meets_credit_rating_requirement(financial_data.credit_rating, limit.category):
                        violations.append(LimitViolation(
                            limit_type=LimitType.FINANCIAL_STRENGTH,
                            limit_id=str(limit.id),
                            violation_severity=ViolationSeverity.WARNING,
                            message=f"Credit rating '{financial_data.credit_rating}' may not meet minimum requirements",
                            current_value=financial_data.credit_rating,
                            recommendation="Review financial strength requirements"
                        ))
                
                # Check minimum asset requirements
                if limit.category == "minimum_assets" and financial_data.assets:
                    if limit.max_amount and financial_data.assets < limit.max_amount:
                        violations.append(LimitViolation(
                            limit_type=LimitType.FINANCIAL_STRENGTH,
                            limit_id=str(limit.id),
                            violation_severity=ViolationSeverity.WARNING,
                            message=f"Assets {financial_data.assets} below minimum requirement of {limit.max_amount}",
                            current_value=float(financial_data.assets),
                            limit_value=float(limit.max_amount),
                            recommendation="Review minimum asset requirements"
                        ))
        
        except Exception as e:
            self.logger.error(f"Error checking financial strength limits: {str(e)}")
        
        return violations
    
    def _check_regulatory_compliance(
        self, 
        risk_parameters: RiskParametersSchema, 
        financial_data: Optional[FinancialDataSchema]
    ) -> List[LimitViolation]:
        """Check regulatory compliance requirements"""
        violations = []
        
        try:
            # Get regulatory compliance limits
            limits = self.db.query(BusinessLimit).filter(
                and_(
                    BusinessLimit.limit_type == LimitType.REGULATORY_COMPLIANCE,
                    BusinessLimit.active == True
                )
            ).all()
            
            for limit in limits:
                # Check sanctions and restricted entities
                if limit.category == "sanctions_check":
                    # This would integrate with sanctions databases
                    # For now, we'll do basic keyword checking
                    if self._check_sanctions_keywords(risk_parameters):
                        violations.append(LimitViolation(
                            limit_type=LimitType.REGULATORY_COMPLIANCE,
                            limit_id=str(limit.id),
                            violation_severity=ViolationSeverity.CRITICAL,
                            message="Potential sanctions or restricted entity detected",
                            recommendation="Perform detailed sanctions screening"
                        ))
                
                # Check regulatory capital requirements
                if limit.category == "capital_requirements" and financial_data:
                    if not self._meets_capital_requirements(financial_data, limit):
                        violations.append(LimitViolation(
                            limit_type=LimitType.REGULATORY_COMPLIANCE,
                            limit_id=str(limit.id),
                            violation_severity=ViolationSeverity.ERROR,
                            message="May not meet regulatory capital requirements",
                            recommendation="Review regulatory capital adequacy"
                        ))
        
        except Exception as e:
            self.logger.error(f"Error checking regulatory compliance: {str(e)}")
        
        return violations
    
    def _check_catastrophe_exposure(self, risk_parameters: RiskParametersSchema) -> List[LimitViolation]:
        """Check catastrophe exposure limits"""
        violations = []
        
        try:
            # Get catastrophe exposure limits
            limits = self.db.query(BusinessLimit).filter(
                and_(
                    BusinessLimit.limit_type == LimitType.CATASTROPHE_EXPOSURE,
                    BusinessLimit.active == True
                )
            ).all()
            
            if risk_parameters.location and risk_parameters.asset_value:
                cat_risk_level = self._assess_catastrophe_risk(risk_parameters.location)
                
                for limit in limits:
                    if limit.category == cat_risk_level and limit.max_amount:
                        if risk_parameters.asset_value > limit.max_amount:
                            violations.append(LimitViolation(
                                limit_type=LimitType.CATASTROPHE_EXPOSURE,
                                limit_id=str(limit.id),
                                violation_severity=ViolationSeverity.WARNING,
                                message=f"Asset value {risk_parameters.asset_value} in {cat_risk_level} catastrophe zone exceeds limit of {limit.max_amount}",
                                current_value=float(risk_parameters.asset_value),
                                limit_value=float(limit.max_amount),
                                recommendation=f"Review catastrophe exposure for {cat_risk_level} risk areas",
                                metadata={"catastrophe_risk_level": cat_risk_level}
                            ))
        
        except Exception as e:
            self.logger.error(f"Error checking catastrophe exposure: {str(e)}")
        
        return violations
    
    def _check_aggregate_exposure(self, application_id: str, risk_parameters: RiskParametersSchema) -> List[LimitViolation]:
        """Check aggregate exposure limits"""
        violations = []
        
        try:
            # Get aggregate exposure limits
            limits = self.db.query(BusinessLimit).filter(
                and_(
                    BusinessLimit.limit_type == LimitType.AGGREGATE_EXPOSURE,
                    BusinessLimit.active == True
                )
            ).all()
            
            if risk_parameters.asset_value:
                # Calculate current aggregate exposure (simplified)
                current_exposure = self._calculate_aggregate_exposure(risk_parameters)
                
                for limit in limits:
                    if limit.max_amount and current_exposure > limit.max_amount:
                        violations.append(LimitViolation(
                            limit_type=LimitType.AGGREGATE_EXPOSURE,
                            limit_id=str(limit.id),
                            violation_severity=ViolationSeverity.ERROR,
                            message=f"Aggregate exposure {current_exposure} exceeds limit of {limit.max_amount}",
                            current_value=float(current_exposure),
                            limit_value=float(limit.max_amount),
                            recommendation="Review aggregate exposure management"
                        ))
        
        except Exception as e:
            self.logger.error(f"Error checking aggregate exposure: {str(e)}")
        
        return violations
    
    def _detect_geographic_region(self, location: str) -> Optional[str]:
        """Detect geographic region from location string"""
        location_lower = location.lower()
        
        for region, keywords in self.geographic_regions.items():
            if any(keyword in location_lower for keyword in keywords):
                return region
        
        return None
    
    def _detect_industry_sector(self, industry: str) -> Optional[str]:
        """Detect industry sector from industry string"""
        industry_lower = industry.lower()
        
        for sector, keywords in self.industry_sectors.items():
            if any(keyword in industry_lower for keyword in keywords):
                return sector
        
        return None
    
    def _meets_credit_rating_requirement(self, credit_rating: str, requirement: str) -> bool:
        """Check if credit rating meets minimum requirements"""
        # Simplified credit rating check
        rating_scores = {
            "AAA": 10, "AA+": 9, "AA": 8, "AA-": 7,
            "A+": 6, "A": 5, "A-": 4,
            "BBB+": 3, "BBB": 2, "BBB-": 1,
            "BB+": 0, "BB": -1, "BB-": -2
        }
        
        current_score = rating_scores.get(credit_rating.upper(), -10)
        return current_score >= 2  # Minimum BBB rating
    
    def _check_sanctions_keywords(self, risk_parameters: RiskParametersSchema) -> bool:
        """Check for sanctions-related keywords (simplified)"""
        sanctions_keywords = ["sanctioned", "restricted", "blocked", "prohibited"]
        
        text_fields = [
            risk_parameters.location or "",
            risk_parameters.industry_sector or "",
            risk_parameters.occupancy or ""
        ]
        
        combined_text = " ".join(text_fields).lower()
        return any(keyword in combined_text for keyword in sanctions_keywords)
    
    def _meets_capital_requirements(self, financial_data: FinancialDataSchema, limit: BusinessLimit) -> bool:
        """Check if financial data meets capital requirements"""
        if not financial_data.assets or not financial_data.liabilities:
            return False
        
        capital_ratio = (financial_data.assets - financial_data.liabilities) / financial_data.assets
        return capital_ratio >= 0.1  # Minimum 10% capital ratio
    
    def _assess_catastrophe_risk(self, location: str) -> str:
        """Assess catastrophe risk level for location"""
        location_lower = location.lower()
        
        high_risk_areas = ["california", "florida", "japan", "caribbean", "philippines"]
        medium_risk_areas = ["texas", "new york", "italy", "turkey", "chile"]
        
        if any(area in location_lower for area in high_risk_areas):
            return "high_cat_risk"
        elif any(area in location_lower for area in medium_risk_areas):
            return "medium_cat_risk"
        else:
            return "low_cat_risk"
    
    def _calculate_aggregate_exposure(self, risk_parameters: RiskParametersSchema) -> Decimal:
        """Calculate aggregate exposure (simplified)"""
        # In a real implementation, this would sum up all current exposures
        # For now, we'll return the current asset value
        return risk_parameters.asset_value or Decimal(0)
    
    def _generate_warnings_and_recommendations(self, violations: List[LimitViolation]) -> Tuple[List[str], List[str]]:
        """Generate warnings and recommendations based on violations"""
        warnings = []
        recommendations = []
        
        # Group violations by type
        violation_types = {}
        for violation in violations:
            if violation.limit_type not in violation_types:
                violation_types[violation.limit_type] = []
            violation_types[violation.limit_type].append(violation)
        
        # Generate type-specific warnings and recommendations
        for limit_type, type_violations in violation_types.items():
            if limit_type == LimitType.ASSET_VALUE:
                warnings.append(f"Asset value limits exceeded in {len(type_violations)} cases")
                recommendations.append("Consider risk mitigation strategies or senior approval")
            
            elif limit_type == LimitType.GEOGRAPHIC_EXPOSURE:
                warnings.append(f"Geographic exposure concerns in {len(type_violations)} regions")
                recommendations.append("Review regional diversification strategy")
            
            elif limit_type == LimitType.INDUSTRY_SECTOR:
                critical_violations = [v for v in type_violations if v.violation_severity == ViolationSeverity.CRITICAL]
                if critical_violations:
                    warnings.append(f"Restricted industry sectors detected: {len(critical_violations)} cases")
                    recommendations.append("These industry sectors are not acceptable for reinsurance")
            
            elif limit_type == LimitType.REGULATORY_COMPLIANCE:
                warnings.append("Regulatory compliance issues detected")
                recommendations.append("Perform detailed compliance review before proceeding")
        
        return warnings, recommendations
    
    # Business Limits Configuration Methods
    
    def create_business_limit(self, limit_data: BusinessLimitCreate) -> BusinessLimitSchema:
        """Create a new business limit"""
        try:
            db_limit = BusinessLimit(**limit_data.dict())
            self.db.add(db_limit)
            self.db.commit()
            self.db.refresh(db_limit)
            
            self.logger.info(f"Created business limit: {db_limit.id}")
            return BusinessLimitSchema.from_orm(db_limit)
        
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Error creating business limit: {str(e)}")
            raise
    
    def update_business_limit(self, limit_id: str, limit_data: BusinessLimitUpdate) -> Optional[BusinessLimitSchema]:
        """Update an existing business limit"""
        try:
            db_limit = self.db.query(BusinessLimit).filter(BusinessLimit.id == limit_id).first()
            if not db_limit:
                return None
            
            update_data = limit_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_limit, field, value)
            
            self.db.commit()
            self.db.refresh(db_limit)
            
            self.logger.info(f"Updated business limit: {limit_id}")
            return BusinessLimitSchema.from_orm(db_limit)
        
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Error updating business limit: {str(e)}")
            raise
    
    def get_business_limits(
        self, 
        limit_type: Optional[str] = None,
        active_only: bool = True
    ) -> List[BusinessLimitSchema]:
        """Get business limits with optional filtering"""
        try:
            query = self.db.query(BusinessLimit)
            
            if limit_type:
                query = query.filter(BusinessLimit.limit_type == limit_type)
            
            if active_only:
                query = query.filter(BusinessLimit.active == True)
            
            limits = query.all()
            return [BusinessLimitSchema.from_orm(limit) for limit in limits]
        
        except Exception as e:
            self.logger.error(f"Error fetching business limits: {str(e)}")
            return []
    
    def delete_business_limit(self, limit_id: str) -> bool:
        """Delete a business limit (soft delete by setting active=False)"""
        try:
            db_limit = self.db.query(BusinessLimit).filter(BusinessLimit.id == limit_id).first()
            if not db_limit:
                return False
            
            db_limit.active = False
            self.db.commit()
            
            self.logger.info(f"Deleted business limit: {limit_id}")
            return True
        
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Error deleting business limit: {str(e)}")
            return False