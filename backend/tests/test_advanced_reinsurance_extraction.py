"""
Test suite for advanced reinsurance-specific field extraction (Task 5.4)

Tests the following advanced extraction methods:
- PML % extraction with percentage and risk validation
- CAT Exposure extraction with risk validation
- Period of Insurance and Reinsurance Deductions extraction
- Claims Experience (3 years) and Share offered Percentage extraction
- Surveyor's report detection and attachment linking
- Climate Change and ESG Risk Assessment field extraction
"""

import pytest
from decimal import Decimal
from backend.app.agents.data_extraction_agent import DataExtractionAgent


class TestAdvancedReinsuranceExtraction:
    """Test advanced reinsurance-specific field extraction methods"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.agent = DataExtractionAgent()
    
    def test_extract_pml_percentage_basic(self):
        """Test basic PML percentage extraction"""
        text = "The Probable Maximum Loss (PML) is estimated at 15.5% of the total sum insured."
        
        pml_value, confidence = self.agent.extract_pml_percentage(text)
        
        assert pml_value == Decimal('15.5')
        assert confidence >= 0.8
    
    def test_extract_pml_percentage_variations(self):
        """Test PML extraction with various formats"""
        test_cases = [
            ("PML: 25%", Decimal('25'), 0.8),
            ("Maximum Probable Loss 12.75 percent", Decimal('12.75'), 0.8),
            ("The PML is 8.5%", Decimal('8.5'), 0.8),
            ("Estimated Maximum Loss: 30%", Decimal('30'), 0.6),  # Lower confidence for EML
        ]
        
        for text, expected_value, min_confidence in test_cases:
            pml_value, confidence = self.agent.extract_pml_percentage(text)
            assert pml_value == expected_value
            assert confidence >= min_confidence
    
    def test_extract_pml_percentage_validation(self):
        """Test PML percentage validation for out-of-range values"""
        # Test value over 100%
        text = "PML: 150%"
        pml_value, confidence = self.agent.extract_pml_percentage(text)
        assert pml_value == Decimal('100')  # Should be capped at 100%
        assert confidence < 0.5  # Low confidence for out-of-range
        
        # Test negative value
        text = "PML: -5%"
        pml_value, confidence = self.agent.extract_pml_percentage(text)
        assert confidence < 0.5  # Low confidence for negative values
    
    def test_extract_cat_exposure_basic(self):
        """Test basic catastrophe exposure extraction"""
        text = "The property is exposed to earthquake risk in Zone 4 according to RMS model analysis."
        
        cat_exposure, confidence = self.agent.extract_cat_exposure(text)
        
        assert cat_exposure is not None
        assert "earthquake" in cat_exposure.lower()
        assert "zone 4" in cat_exposure.lower()
        assert confidence >= 0.6
    
    def test_extract_cat_exposure_multiple_perils(self):
        """Test CAT exposure extraction with multiple perils"""
        text = "Catastrophe exposure includes hurricane, flood, and earthquake risks. Located in high windstorm zone."
        
        cat_exposure, confidence = self.agent.extract_cat_exposure(text)
        
        assert cat_exposure is not None
        assert any(peril in cat_exposure.lower() for peril in ["hurricane", "flood", "earthquake", "windstorm"])
        assert confidence >= 0.6
    
    def test_extract_cat_exposure_with_models(self):
        """Test CAT exposure extraction with catastrophe models"""
        text = "AIR model indicates significant hurricane exposure. RMS earthquake model shows moderate risk."
        
        cat_exposure, confidence = self.agent.extract_cat_exposure(text)
        
        assert cat_exposure is not None
        assert any(model in cat_exposure.lower() for model in ["air", "rms"])
        assert confidence >= 0.7  # Higher confidence with models
    
    def test_extract_period_of_insurance_date_range(self):
        """Test period of insurance extraction with date ranges"""
        text = "Period of Insurance: From 01/01/2024 to 31/12/2024"
        
        period, confidence = self.agent.extract_period_of_insurance(text)
        
        assert period is not None
        assert "01/01/2024" in period
        assert "31/12/2024" in period
        assert confidence >= 0.8
    
    def test_extract_period_of_insurance_standard_terms(self):
        """Test period extraction with standard insurance terms"""
        test_cases = [
            ("Policy period: 12 months", "12 months", 0.7),
            ("Coverage period: 1 year", "1 year", 0.7),
            ("Insurance term: 24 months", "24 months", 0.7),
        ]
        
        for text, expected_content, min_confidence in test_cases:
            period, confidence = self.agent.extract_period_of_insurance(text)
            assert period is not None
            assert expected_content in period.lower()
            assert confidence >= min_confidence
    
    def test_extract_reinsurance_deductions_percentage(self):
        """Test reinsurance deductions extraction as percentage"""
        text = "Reinsurance commission: 25% of premium"
        
        deductions, confidence = self.agent.extract_reinsurance_deductions(text)
        
        assert deductions == Decimal('25')
        assert confidence >= 0.6
    
    def test_extract_reinsurance_deductions_amount(self):
        """Test reinsurance deductions extraction as amount"""
        text = "Brokerage deduction: USD 50,000"
        
        deductions, confidence = self.agent.extract_reinsurance_deductions(text)
        
        assert deductions == Decimal('50000')
        assert confidence >= 0.6
    
    def test_extract_reinsurance_deductions_variations(self):
        """Test various reinsurance deduction formats"""
        test_cases = [
            ("Profit commission: 15%", Decimal('15')),
            ("Override commission: 5%", Decimal('5')),
            ("No claims bonus: 10%", Decimal('10')),
        ]
        
        for text, expected_value in test_cases:
            deductions, confidence = self.agent.extract_reinsurance_deductions(text)
            assert deductions == expected_value
            assert confidence >= 0.5
    
    def test_extract_claims_experience_positive(self):
        """Test claims experience extraction - positive experience"""
        text = "Claims experience for the last 3 years: Nil claims reported. Excellent loss record."
        
        claims_exp, confidence = self.agent.extract_claims_experience_3_years(text)
        
        assert claims_exp is not None
        assert "nil claims" in claims_exp.lower()
        assert "excellent" in claims_exp.lower()
        assert confidence >= 0.8  # Higher confidence for positive indicators
    
    def test_extract_claims_experience_negative(self):
        """Test claims experience extraction - adverse experience"""
        text = "Loss history shows significant claims in 2022 and 2023. Poor experience with frequency issues."
        
        claims_exp, confidence = self.agent.extract_claims_experience_3_years(text)
        
        assert claims_exp is not None
        assert any(year in claims_exp for year in ["2022", "2023"])
        assert "poor" in claims_exp.lower()
        assert confidence >= 0.6
    
    def test_extract_claims_experience_detailed(self):
        """Test detailed claims experience extraction"""
        text = "Claims record (3 years): 2021: USD 25,000 paid, 2022: No claims, 2023: USD 15,000 outstanding"
        
        claims_exp, confidence = self.agent.extract_claims_experience_3_years(text)
        
        assert claims_exp is not None
        assert all(year in claims_exp for year in ["2021", "2022", "2023"])
        assert confidence >= 0.8  # High confidence with specific years
    
    def test_extract_share_offered_percentage_basic(self):
        """Test basic share offered percentage extraction"""
        text = "Share offered: 25% of the total line"
        
        share_pct, confidence = self.agent.extract_share_offered_percentage(text)
        
        assert share_pct == Decimal('25')
        assert confidence >= 0.8
    
    def test_extract_share_offered_variations(self):
        """Test various share offered formats"""
        test_cases = [
            ("Quota share: 30%", Decimal('30')),
            ("Participation: 15 percent", Decimal('15')),
            ("Line size: 20%", Decimal('20')),
            ("Capacity offered: 40%", Decimal('40')),
        ]
        
        for text, expected_value in test_cases:
            share_pct, confidence = self.agent.extract_share_offered_percentage(text)
            assert share_pct == expected_value
            assert confidence >= 0.6
    
    def test_extract_share_offered_fractional(self):
        """Test share offered extraction with fractional values"""
        text = "Reinsurance share: 0.25"  # Should be converted to 25%
        
        share_pct, confidence = self.agent.extract_share_offered_percentage(text)
        
        # Note: This test depends on implementation - might need adjustment
        assert share_pct is not None
        assert confidence >= 0.4
    
    def test_extract_surveyors_report_basic(self):
        """Test basic surveyor's report extraction"""
        text = "Surveyor's report attached. Risk survey completed by ABC Engineering."
        
        survey_report, confidence = self.agent.extract_surveyors_report(text)
        
        assert survey_report is not None
        assert "attached" in survey_report.lower()
        assert "abc engineering" in survey_report.lower()
        assert confidence >= 0.7
    
    def test_extract_surveyors_report_with_attachments(self):
        """Test surveyor's report extraction with attachment linking"""
        text = "Please find the risk survey report in attachment: survey_report_2024.pdf"
        attachments = ["survey_report_2024.pdf", "other_document.docx"]
        
        survey_report, confidence = self.agent.extract_surveyors_report(text, attachments)
        
        assert survey_report is not None
        assert "survey_report_2024.pdf" in survey_report
        assert confidence >= 0.8  # Higher confidence with attachment match
    
    def test_extract_surveyors_report_status_indicators(self):
        """Test surveyor's report with various status indicators"""
        test_cases = [
            ("Survey report available upon request", "available"),
            ("Risk inspection completed", "completed"),
            ("Engineering survey pending", "pending"),
            ("Loss control survey provided", "provided"),
        ]
        
        for text, expected_indicator in test_cases:
            survey_report, confidence = self.agent.extract_surveyors_report(text)
            assert survey_report is not None
            assert expected_indicator in survey_report.lower()
            assert confidence >= 0.6
    
    def test_extract_climate_change_risk_basic(self):
        """Test basic climate change risk extraction"""
        text = "Climate change risk assessment: Low physical risk, moderate transition risk due to carbon pricing."
        
        climate_risk, confidence = self.agent.extract_climate_change_risk(text)
        
        assert climate_risk is not None
        assert "low physical risk" in climate_risk.lower()
        assert "transition risk" in climate_risk.lower()
        assert confidence >= 0.7
    
    def test_extract_climate_change_risk_frameworks(self):
        """Test climate risk extraction with specific frameworks"""
        text = "TCFD scenario analysis shows significant climate exposure. Physical climate risks assessed as high."
        
        climate_risk, confidence = self.agent.extract_climate_change_risk(text)
        
        assert climate_risk is not None
        assert "tcfd" in climate_risk.lower()
        assert "scenario analysis" in climate_risk.lower()
        assert confidence >= 0.8  # Higher confidence with frameworks
    
    def test_extract_climate_change_risk_levels(self):
        """Test climate risk extraction with risk levels"""
        test_cases = [
            ("Climate risk: Low impact expected", "low"),
            ("Significant climate exposure identified", "significant"),
            ("Moderate environmental risk", "moderate"),
            ("High climate change vulnerability", "high"),
        ]
        
        for text, expected_level in test_cases:
            climate_risk, confidence = self.agent.extract_climate_change_risk(text)
            assert climate_risk is not None
            assert expected_level in climate_risk.lower()
            assert confidence >= 0.7  # Higher confidence with risk levels
    
    def test_extract_esg_risk_assessment_basic(self):
        """Test basic ESG risk assessment extraction"""
        text = "ESG risk assessment: Strong governance framework, moderate environmental impact, good social practices."
        
        esg_risk, confidence = self.agent.extract_esg_risk_assessment(text)
        
        assert esg_risk is not None
        assert "governance" in esg_risk.lower()
        assert "environmental" in esg_risk.lower()
        assert "social" in esg_risk.lower()
        assert confidence >= 0.7
    
    def test_extract_esg_risk_components(self):
        """Test ESG risk extraction with multiple components"""
        text = "Environmental compliance excellent. Social responsibility policies in place. Board diversity assessed."
        
        esg_risk, confidence = self.agent.extract_esg_risk_assessment(text)
        
        assert esg_risk is not None
        component_count = sum(1 for comp in ["environmental", "social", "diversity"] 
                            if comp in esg_risk.lower())
        assert component_count >= 2
        assert confidence >= 0.7  # Higher confidence with multiple components
    
    def test_extract_esg_risk_ratings(self):
        """Test ESG risk extraction with ratings and assessments"""
        test_cases = [
            ("ESG rating: AA- from MSCI", "rating"),
            ("Sustainability score: 85/100", "score"),
            ("ESG compliance framework implemented", "compliance"),
            ("Corporate responsibility assessment completed", "assessment"),
        ]
        
        for text, expected_indicator in test_cases:
            esg_risk, confidence = self.agent.extract_esg_risk_assessment(text)
            assert esg_risk is not None
            assert expected_indicator in esg_risk.lower()
            assert confidence >= 0.7  # Higher confidence with specific indicators
    
    def test_extract_no_match_scenarios(self):
        """Test extraction methods with text that contains no relevant information"""
        irrelevant_text = "This is a general business document with no reinsurance-specific information."
        
        # All methods should return None or empty results with low confidence
        pml_value, pml_conf = self.agent.extract_pml_percentage(irrelevant_text)
        cat_exposure, cat_conf = self.agent.extract_cat_exposure(irrelevant_text)
        period, period_conf = self.agent.extract_period_of_insurance(irrelevant_text)
        deductions, ded_conf = self.agent.extract_reinsurance_deductions(irrelevant_text)
        claims_exp, claims_conf = self.agent.extract_claims_experience_3_years(irrelevant_text)
        share_pct, share_conf = self.agent.extract_share_offered_percentage(irrelevant_text)
        survey_report, survey_conf = self.agent.extract_surveyors_report(irrelevant_text)
        climate_risk, climate_conf = self.agent.extract_climate_change_risk(irrelevant_text)
        esg_risk, esg_conf = self.agent.extract_esg_risk_assessment(irrelevant_text)
        
        # All should have None values or very low confidence
        assert pml_value is None or pml_conf < 0.3
        assert cat_exposure is None or cat_conf < 0.3
        assert period is None or period_conf < 0.3
        assert deductions is None or ded_conf < 0.3
        assert claims_exp is None or claims_conf < 0.3
        assert share_pct is None or share_conf < 0.3
        assert survey_report is None or survey_conf < 0.3
        assert climate_risk is None or climate_conf < 0.3
        assert esg_risk is None or esg_conf < 0.3
    
    def test_field_length_limits(self):
        """Test that extracted fields respect maximum length limits"""
        # Create very long text
        long_text = "Climate change risk assessment: " + "Very detailed analysis " * 100
        
        climate_risk, confidence = self.agent.extract_climate_change_risk(long_text)
        
        if climate_risk:
            assert len(climate_risk) <= 500  # Should be truncated
            if len(climate_risk) == 500:
                assert climate_risk.endswith("...")  # Should have truncation indicator
    
    def test_confidence_scoring_consistency(self):
        """Test that confidence scores are consistent and within valid range"""
        test_text = """
        PML: 15%
        Catastrophe exposure: Earthquake Zone 3
        Period of Insurance: 12 months
        Reinsurance commission: 20%
        Claims experience: Nil claims for 3 years
        Share offered: 25%
        Surveyor's report attached
        Climate change risk: Low
        ESG assessment: Good governance
        """
        
        methods = [
            self.agent.extract_pml_percentage,
            self.agent.extract_cat_exposure,
            self.agent.extract_period_of_insurance,
            self.agent.extract_reinsurance_deductions,
            self.agent.extract_claims_experience_3_years,
            self.agent.extract_share_offered_percentage,
            self.agent.extract_surveyors_report,
            self.agent.extract_climate_change_risk,
            self.agent.extract_esg_risk_assessment,
        ]
        
        for method in methods:
            if method == self.agent.extract_surveyors_report:
                # This method takes an optional attachments parameter
                value, confidence = method(test_text, [])
            else:
                value, confidence = method(test_text)
            
            # Confidence should be between 0 and 1
            assert 0.0 <= confidence <= 1.0
            
            # If a value is extracted, confidence should be reasonable
            if value is not None:
                assert confidence >= 0.3  # Minimum reasonable confidence


if __name__ == "__main__":
    pytest.main([__file__])