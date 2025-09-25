from tests.mock.test_sample_data import test_mock_data
from backend.app.services.risk_scorer import RiskScorer
import json

def test_risk_scoring():
    scorer = RiskScorer()
    with open('backend/app/mock/sample_data.py') as f:
        for sample_id in [1, 2, 3]:
            data = json.loads(get_mock_excel(sample_id))
            score = scorer.calculate_final_score(data)
            print(f"\n=== Risk Score for Sample {sample_id} ===")
            print(f"Final Score: {score.final_score:.2f}")
            print(f"Recommendation: {score.recommendation}")
            print(f"Claims Score: {score.claims_score.overall_score:.2f}")
            print(f"Confidence: {score.confidence:.2f}")

def main():
    print("Testing mock data samples...")
    try:
        test_mock_data()
        test_risk_scoring()
        print("\nAll tests passed successfully!")
    except AssertionError as e:
        print(f"\nTest failed: {str(e)}")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main()
