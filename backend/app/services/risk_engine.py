from decimal import Decimal
from ..models.scoring_models import DetailedRiskScore, PremiumCalculation, LossRatioCalculation
from ..utils.calculations import *

class KenyaReRiskEngine:
    def __init__(self):
        self.loss_ratio_thresholds = {
            'decline': Decimal('80'),
            'refer': Decimal('60'),
            'accept': Decimal('35')
        }
        
    def recommend_share(self, risk_score: DetailedRiskScore) -> Decimal:
        """
        Recommend facultative share based on:
        - Loss ratio thresholds
        - Technical score
        - Rate adequacy
        - Portfolio impact
        """
        base_share = Decimal('0')
        
        # Start with maximum potential share
        if risk_score.technical_score >= Decimal('80'):
            base_share = Decimal('40')
        elif risk_score.technical_score >= Decimal('70'):
            base_share = Decimal('30')
        elif risk_score.technical_score >= Decimal('60'):
            base_share = Decimal('20')
        
        # Adjust for loss ratio
        if risk_score.loss_ratio.loss_ratio_percent > self.loss_ratio_thresholds['decline']:
            return Decimal('0')
        elif risk_score.loss_ratio.loss_ratio_percent > self.loss_ratio_thresholds['refer']:
            base_share = max(Decimal('0'), base_share - Decimal('15'))
        
        # Adjust for rate adequacy
        if risk_score.premium_calculation.rate_percent < Decimal('0.2'):  # Minimum 0.2%
            base_share = max(Decimal('0'), base_share - Decimal('10'))
            
        return base_share.quantize(Decimal('0.1'))
