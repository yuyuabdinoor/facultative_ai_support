from decimal import Decimal
from ..models.scoring_models import *
import numpy as np

class RiskScorer:
    def __init__(self):
        self.peril_severity = {
            "FIRE": 0.9,
            "EARTHQUAKE": 0.9,
            "FLOOD": 0.8,
            "MACHINERY_BREAKDOWN": 0.5,
            "THEFT": 0.4
        }
        
        self.peril_severity.update({
            "CYCLONE": 0.9,
            "TERRORISM": 0.8,
            "POLITICAL_VIOLENCE": 0.8
        })
        
        self.kenya_cat_zones = {
            "NAIROBI": {"flood": 0.6, "quake": 0.7},
            "MOMBASA": {"flood": 0.8, "cyclone": 0.8},
            "RIFT_VALLEY": {"quake": 0.9}
        }
        
        self.industry_hazards = {
            "CHEMICAL": HazardLevel.HIGH,
            "OIL_GAS": HazardLevel.HIGH,
            "AVIATION": HazardLevel.HIGH,
            "MANUFACTURING": HazardLevel.MEDIUM,
            "HEALTHCARE": HazardLevel.MEDIUM,
            "OFFICE": HazardLevel.LOW,
            "EDUCATION": HazardLevel.LOW
        }
        
        self.min_technical_rates = {
            "FIRE": Decimal('0.002'),  # 0.2%
            "ENGINEERING": Decimal('0.003'),
            "LIABILITY": Decimal('0.001')
        }

    def calculate_claims_score(self, claims_history: List[Dict], tsi: Decimal) -> ClaimsScore:
        if not claims_history:
            return ClaimsScore(frequency_score=100, severity_score=100, trend_score=100, overall_score=100)
            
        frequency = len(claims_history)
        total_amount = sum(Decimal(str(claim["amount"])) for claim in claims_history)
        avg_severity = total_amount / frequency if frequency > 0 else Decimal('0')
        severity_ratio = avg_severity / tsi if tsi else Decimal('0')
        
        # Implement more sophisticated scoring logic
        frequency_score = max(0, 100 - (frequency * 25))  # Harsher penalty for frequency
        severity_score = max(0, 100 - (float(severity_ratio) * 200))  # Harsher penalty for high severity ratio
        
        # Enhanced trend analysis
        if len(claims_history) >= 3:
            sorted_claims = sorted(claims_history, key=lambda x: x["year"])
            amounts = [float(c["amount"]) for c in sorted_claims]
            trend_coefficient = np.polyfit(range(len(amounts)), amounts, 1)[0]
            trend_score = 100 - min(100, max(0, trend_coefficient / total_amount * 100))
        else:
            trend_score = 75
        
        # Additional penalty for high frequency + high severity
        if frequency >= 3 and severity_ratio > Decimal('0.1'):
            overall_penalty = 20
        else:
            overall_penalty = 0
            
        overall = (
            frequency_score * 0.4 +
            severity_score * 0.4 +
            trend_score * 0.2
        ) - overall_penalty
        
        return ClaimsScore(
            frequency_score=frequency_score,
            severity_score=severity_score,
            trend_score=trend_score,
            overall_score=max(0, overall)
        )

    # Add new methods for each scoring component
    def calculate_occupation_score(self, industry: str) -> OccupationScore:
        hazard_level = self.industry_hazards.get(industry, HazardLevel.MEDIUM)
        base_score = {
            HazardLevel.LOW: 90,
            HazardLevel.MEDIUM: 70,
            HazardLevel.HIGH: 50
        }[hazard_level]
        
        return OccupationScore(
            industry_type=industry,
            hazard_level=hazard_level,
            score=base_score
        )

    # ...implement other scoring methods...

    def calculate_final_score(self, data: dict) -> DetailedRiskScore:
        # Calculate all component scores
        claims_score = self.calculate_claims_score(
            data.get("Claims_History", []),
            Decimal(str(data.get("TSI", 0)))
        )
        occupation_score = self.calculate_occupation_score(data.get("Industry", "UNKNOWN"))
        # ...calculate other scores...
        
        # Weighted final score calculation
        final_score = sum([
            s.score * s.weight for s in [
                claims_score,
                occupation_score,
                # ...other scores...
            ]
        ])
        
        return DetailedRiskScore(
            claims_score=claims_score,
            occupation_score=occupation_score,
            # ...other scores...
            final_score=final_score,
            recommendation=self.get_recommendation(final_score),
            confidence=self.calculate_confidence(data)
        )
