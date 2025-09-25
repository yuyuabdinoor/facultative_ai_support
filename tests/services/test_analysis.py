import pytest
from backend.app.services.analysis_service import AnalysisService
from backend.app.models.analysis_models import *

@pytest.fixture
def analysis_service():
    return AnalysisService()

def test_loss_ratio_analysis(analysis_service):
    claims = [
        {"year": 2022, "amount": 100000},
        {"year": 2023, "amount": 150000},
        {"year": 2024, "amount": 200000}
    ]
    premiums = [
        {"year": 2022, "amount": 500000},
        {"year": 2023, "amount": 500000},
        {"year": 2024, "amount": 500000}
    ]
    
    result = analysis_service.analyze_loss_ratio(claims, premiums)
    assert result.three_year_ratio == 0.3  # (450000 / 1500000)
    assert result.threshold_status == "FAVORABLE"

def test_recommended_share_calculation(analysis_service):
    loss_ratio = LossRatioAssessment(
        three_year_ratio=0.3,
        five_year_ratio=0.35,
        threshold_status="FAVORABLE",
        recommendation="ACCEPT",
        mitigating_factors=[]
    )
    
    rate_adequacy = RateAdequacy(
        guideline_rate=0.5,
        offered_rate=0.4,
        adequacy_ratio=0.8,
        is_adequate=True
    )
    
    esg_assessment = ESGAssessment(
        environmental_score=80,
        social_score=75,
        governance_score=85,
        climate_risk_exposure="LOW",
        sustainability_rating="A",
        key_concerns=[]
    )
    
    share = analysis_service.calculate_recommended_share(
        loss_ratio, rate_adequacy, esg_assessment
    )
    
    assert 0 <= share <= 40.0
