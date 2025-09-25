from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
from decimal import Decimal

class HazardLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class RiskSeverity(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"

class ClaimsScore(BaseModel):
    frequency_score: float  # 0-100
    severity_score: float  # 0-100
    trend_score: float   # 0-100
    overall_score: float # 0-100
    weight: float = 0.20 # 20% weight

class PerilScore(BaseModel):
    individual_scores: Dict[str, float]
    aggregation_risk: float
    overall_score: float
    weight: float = 0.15

class GeographicalScore(BaseModel):
    cat_risk_score: float
    political_risk_score: float
    protection_score: float
    overall_score: float
    weight: float = 0.10

class RiskScore(BaseModel):
    claims_score: ClaimsScore
    peril_score: PerilScore
    geographical_score: GeographicalScore
    final_score: float
    recommendation: str
    confidence: float

class PremiumCalculation(BaseModel):
    tsi: Decimal
    gross_premium: Decimal
    net_premium: Decimal
    rate_percent: Decimal
    rate_permille: Decimal
    currency: str
    exchange_rate: Optional[Decimal]
    kes_equivalent: Decimal

class LossRatioCalculation(BaseModel):
    paid_losses: Decimal
    outstanding_reserves: Decimal
    recoveries: Decimal
    earned_premium: Decimal
    loss_ratio_percent: Decimal
    period_years: int = 3

class FacultativeShare(BaseModel):
    accepted_share_percent: Decimal
    accepted_premium: Decimal
    accepted_liability: Decimal
    accepted_loss_ratio: Optional[Decimal]

class OccupationScore(BaseModel):
    industry_type: str
    hazard_level: HazardLevel
    score: float
    weight: float = 0.10

class RetentionScore(BaseModel):
    tsi: Decimal
    cedant_retention: Decimal
    retention_ratio: float
    score: float
    weight: float = 0.10

class CedantScore(BaseModel):
    financial_rating: Optional[str]
    claims_handling_score: float
    reliability_score: float
    score: float
    weight: float = 0.10

class SurveyorScore(BaseModel):
    protection_measures: Dict[str, float]
    facility_age: Optional[int]
    maintenance_quality: float
    score: float
    weight: float = 0.10

class PMLScore(BaseModel):
    pml_ratio: float
    diversification_score: float
    score: float
    weight: float = 0.05

class TermsScore(BaseModel):
    deductible_adequacy: float
    coverage_terms: Dict[str, float]
    score: float
    weight: float = 0.05

class ESGScore(BaseModel):
    environmental_score: float
    social_score: float
    governance_score: float
    climate_risk_score: float
    score: float
    weight: float = 0.05

class PortfolioScore(BaseModel):
    concentration_risk: float
    market_cycle_position: float
    diversification_impact: float
    score: float
    weight: float = 0.05

class PremiumScore(BaseModel):
    technical_rate: Decimal
    offered_rate: Decimal
    adequacy_ratio: float
    score: float
    weight: float = 0.05

class DetailedRiskScore(RiskScore):
    occupation_score: OccupationScore
    retention_score: RetentionScore
    cedant_score: CedantScore
    surveyor_score: SurveyorScore
    pml_score: PMLScore
    terms_score: TermsScore
    esg_score: ESGScore
    portfolio_score: PortfolioScore
    premium_score: PremiumScore
