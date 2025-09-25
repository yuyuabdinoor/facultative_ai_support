from ..models.analysis_models import *
from typing import Dict, List
import numpy as np

class AnalysisService:
    def __init__(self):
        self.loss_ratio_thresholds = {
            "FAVORABLE": 0.60,
            "CAUTIONARY": 0.80,
            "DECLINE": 1.0
        }
        
    def analyze_loss_ratio(self, claims_history: List[Dict], premium_history: List[Dict]) -> LossRatioAssessment:
        # Calculate 3 and 5 year ratios
        total_claims_3yr = sum(c["amount"] for c in claims_history[-3:])
        total_premium_3yr = sum(p["amount"] for p in premium_history[-3:])
        
        three_year_ratio = total_claims_3yr / total_premium_3yr if total_premium_3yr > 0 else 0
        
        if len(claims_history) >= 5:
            total_claims_5yr = sum(c["amount"] for c in claims_history[-5:])
            total_premium_5yr = sum(p["amount"] for p in premium_history[-5:])
            five_year_ratio = total_claims_5yr / total_premium_5yr if total_premium_5yr > 0 else 0
        else:
            five_year_ratio = None
            
        # Determine threshold status
        if three_year_ratio < self.loss_ratio_thresholds["FAVORABLE"]:
            status = "FAVORABLE"
            recommendation = "ACCEPT"
        elif three_year_ratio < self.loss_ratio_thresholds["CAUTIONARY"]:
            status = "CAUTIONARY"
            recommendation = "ACCEPT WITH CONDITIONS"
        else:
            status = "UNFAVORABLE"
            recommendation = "DECLINE"
            
        return LossRatioAssessment(
            three_year_ratio=three_year_ratio,
            five_year_ratio=five_year_ratio,
            threshold_status=status,
            recommendation=recommendation,
            mitigating_factors=[]
        )
        
    def calculate_recommended_share(self, 
                                 loss_ratio: LossRatioAssessment,
                                 rate_adequacy: RateAdequacy,
                                 esg_assessment: ESGAssessment) -> float:
        # Base share calculation
        if loss_ratio.threshold_status == "FAVORABLE":
            base_share = 30.0
        elif loss_ratio.threshold_status == "CAUTIONARY":
            base_share = 20.0
        else:
            base_share = 0.0
            
        # Adjust for rate adequacy
        if rate_adequacy.adequacy_ratio >= 0.8:
            base_share += 10.0
        elif rate_adequacy.adequacy_ratio < 0.6:
            base_share = max(0, base_share - 15.0)
            
        # Adjust for ESG
        if esg_assessment.climate_risk_exposure == "LOW":
            base_share += 5.0
        elif esg_assessment.climate_risk_exposure == "HIGH":
            base_share = max(0, base_share - 10.0)
            
        return min(base_share, 40.0)  # Cap at 40%
