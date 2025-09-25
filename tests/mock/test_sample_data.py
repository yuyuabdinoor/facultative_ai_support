from backend.app.mock.sample_data import get_mock_email, get_mock_excel, get_mock_risk_analysis
from backend.app.models.risk_models import RiskAnalysis
import json

def test_mock_data():
    # Test all samples
    for sample_id in [1, 2, 3]:
        # Get email data
        email_text = get_mock_email(sample_id)
        print(f"\n=== Sample {sample_id} Email ===")
        print(email_text)
        
        # Get excel data
        excel_data = get_mock_excel(sample_id)
        print(f"\n=== Sample {sample_id} Excel Data ===")
        print(json.dumps(json.loads(excel_data), indent=2))
        
        # Basic validations
        assert "Insured:" in email_text
        assert "Cedant:" in email_text
        assert "Sum Insured:" in email_text
        
        # Validate excel data
        excel_dict = json.loads(excel_data)
        assert "Insured_Name" in excel_dict
        assert "Total_Sum_Insured" in excel_dict
        assert "Claims_History" in excel_dict

def test_risk_analysis():
    # Test risk analysis for each sample
    for sample_id in [1, 2, 3]:
        risk_data = get_mock_risk_analysis(sample_id)
        
        # Validate risk analysis structure
        analysis = RiskAnalysis(**risk_data)
        
        print(f"\n=== Sample {sample_id} Risk Analysis ===")
        print(f"Technical Score: {analysis.technical_assessment.overall_technical_score}")
        print(f"ESG Rating: {analysis.esg_rating.overall}")
        print(f"Recommendation: {analysis.recommendations.decision}")
        print(f"Confidence: {analysis.overall_confidence}")
        
        # Basic validations
        assert analysis.technical_assessment.overall_technical_score >= 0
        assert analysis.technical_assessment.overall_technical_score <= 100
        assert analysis.overall_confidence > 0
        assert analysis.overall_confidence <= 1

if __name__ == "__main__":
    test_mock_data()
    test_risk_analysis()
