from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class ESGRating(BaseModel):
    environmental: RiskLevel
    social: RiskLevel
    governance: RiskLevel
    overall: RiskLevel
    confidence: float

class TechnicalAssessment(BaseModel):
    risk_quality_score: int  # 0-100
    protection_measures_score: int  # 0-100
    management_quality_score: int  # 0-100
    overall_technical_score: int  # 0-100
    confidence: float

class RiskRecommendation(BaseModel):
    decision: str  # "ACCEPT" | "DECLINE" | "REFER"
    max_share: float  # percentage
    conditions: List[str]
    rationale: str
    confidence: float

class RiskAnalysis(BaseModel):
    technical_assessment: TechnicalAssessment
    esg_rating: ESGRating
    recommendations: RiskRecommendation
    risk_factors: Dict[str, RiskLevel]
    overall_confidence: float
