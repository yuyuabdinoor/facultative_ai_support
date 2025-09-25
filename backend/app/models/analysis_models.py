from pydantic import BaseModel
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime

class LossRatioAssessment(BaseModel):
    three_year_ratio: float
    five_year_ratio: Optional[float]
    threshold_status: str  # "FAVORABLE", "CAUTIONARY", "UNFAVORABLE"
    recommendation: str
    mitigating_factors: List[str]

class RateAdequacy(BaseModel):
    guideline_rate: float
    offered_rate: float
    adequacy_ratio: float  # offered/guideline
    is_adequate: bool  # True if >= 60% of guideline

class ESGAssessment(BaseModel):
    environmental_score: int  # 0-100
    social_score: int  # 0-100
    governance_score: int  # 0-100
    climate_risk_exposure: str  # "LOW", "MEDIUM", "HIGH"
    sustainability_rating: str
    key_concerns: List[str]

class DetailedAnalysis(BaseModel):
    loss_ratio: LossRatioAssessment
    rate_adequacy: RateAdequacy
    esg_assessment: ESGAssessment
    recommended_share: float  # percentage
    conditions: List[str]
    exclusions: List[str]
    warranties: List[str]
