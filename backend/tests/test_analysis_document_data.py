"""
Tests for Analysis Document Data models and validation
"""
import pytest
from datetime import datetime
from decimal import Decimal
from pydantic import ValidationError

from app.models.schemas import AnalysisDocumentData, ANALYSIS_DOCUMENT_FIELD_MAPPING
from app.services.validation import AnalysisDocumentValidator
from app.agents.data_extraction_agent import ExtractedData


class TestAnalysisDocumentData:
    """Test Analysis Document Data model"""
    
    def test_create_empty_analysis_document_data(self):
        """Test creating empty Analysis document data"""
        data = AnalysisDocumentData()
        assert data.reference_number is None
        assert data.confidence_score == 0.0
        assert data.extraction_method == "hybrid"
        assert data.processing_notes == []
        assert data.source_documents == []
    
    def test_create_complete_analysis_document_data(self):
        """Test creating complete Analysis document data"""
        data = AnalysisDocumentData(
            reference_number="FAC-2024-001",
            date_received=datetime(2024, 1, 15),
            insured_name="ABC Manufacturing Corp",
            cedant_reinsured="XYZ Insurance Company",
            broker_name="Global Reinsurance Brokers",
            perils_covered="Fire, Explosion, Natural Catastrophes",
            geographical_limit="Worldwide excluding USA/Canada",
            situation_of_risk="Manufacturing facility in Germany",
            occupation_of_insured="Automotive parts manufacturing",
            main_activities="Production of automotive components",
            total_sums_insured=Decimal("50000000.00"),
            currency="EUR",
            excess_retention=Decimal("1000000.00"),
            premium_rates=Decimal("2.5"),
            period_of_insurance="12 months from 01/01/2024",
            pml_percentage=Decimal("15.0"),
            cat_exposure="Earthquake Zone 2, Flood Zone A",
            reinsurance_deductions=Decimal("500000.00"),
            claims_experience_3_years="No major claims in past 3 years",
            share_offered_percentage=Decimal("25.0"),
            surveyors_report="Favorable report dated 2023-12-15",
            climate_change_risk="Low risk due to inland location",
            esg_risk_assessment="Good ESG practices, certified ISO 14001",
            confidence_score=0.85
        )
        
        assert data.reference_number == "FAC-2024-001"
        assert data.insured_name == "ABC Manufacturing Corp"
        assert data.total_sums_insured == Decimal("50000000.00")
        assert data.currency == "EUR"
        assert data.pml_percentage == Decimal("15.0")
        assert data.confidence_score == 0.85
    
    def test_currency_validation(self):
        """Test currency code validation"""
        # Valid currency
        data = AnalysisDocumentData(currency="USD")
        assert data.currency == "USD"
        
        # Invalid currency length
        with pytest.raises(ValidationError):
            AnalysisDocumentData(currency="US")
        
        with pytest.raises(ValidationError):
            AnalysisDocumentData(currency="USDD")
    
    def test_percentage_validation(self):
        """Test percentage field validation"""
        # Valid percentages
        data = AnalysisDocumentData(
            premium_rates=Decimal("2.5"),
            pml_percentage=Decimal("15.0"),
            share_offered_percentage=Decimal("25.0")
        )
        assert data.premium_rates == Decimal("2.5")
        
        # Invalid percentages (negative)
        with pytest.raises(ValidationError):
            AnalysisDocumentData(premium_rates=Decimal("-1.0"))
        
        # Invalid percentages (over 100)
        with pytest.raises(ValidationError):
            AnalysisDocumentData(pml_percentage=Decimal("150.0"))
    
    def test_financial_amount_validation(self):
        """Test financial amount validation"""
        # Valid amounts
        data = AnalysisDocumentData(
            total_sums_insured=Decimal("1000000.00"),
            excess_retention=Decimal("50000.00")
        )
        assert data.total_sums_insured == Decimal("1000000.00")
        
        # Invalid amounts (negative)
        with pytest.raises(ValidationError):
            AnalysisDocumentData(total_sums_insured=Decimal("-1000.00"))
    
    def test_calculate_completeness_score(self):
        """Test completeness score calculation"""
        # Empty data
        data = AnalysisDocumentData()
        assert data.calculate_completeness_score() == 0.0
        
        # Partially complete data
        data = AnalysisDocumentData(
            reference_number="FAC-001",
            insured_name="Test Company",
            currency="USD",
            total_sums_insured=Decimal("1000000.00")
        )
        score = data.calculate_completeness_score()
        assert 0.0 < score < 1.0
        
        # Complete critical fields
        data = AnalysisDocumentData(
            reference_number="FAC-001",
            insured_name="Test Company",
            cedant_reinsured="Test Insurer",
            broker_name="Test Broker",
            perils_covered="Fire, Flood",
            total_sums_insured=Decimal("1000000.00"),
            currency="USD",
            period_of_insurance="12 months",
            pml_percentage=Decimal("10.0"),
            share_offered_percentage=Decimal("20.0")
        )
        score = data.calculate_completeness_score()
        assert score == 1.0
    
    def test_get_missing_critical_fields(self):
        """Test missing critical fields identification"""
        # Empty data
        data = AnalysisDocumentData()
        missing = data.get_missing_critical_fields()
        assert len(missing) == 10  # All critical fields missing
        
        # Partially complete data
        data = AnalysisDocumentData(
            reference_number="FAC-001",
            insured_name="Test Company"
        )
        missing = data.get_missing_critical_fields()
        assert "reference_number" not in missing
        assert "insured_name" not in missing
        assert "cedant_reinsured" in missing


class TestAnalysisDocumentValidator:
    """Test Analysis Document Validator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = AnalysisDocumentValidator()
    
    def test_validate_financial_amount(self):
        """Test financial amount validation"""
        # Valid amount
        errors = self.validator.validate_financial_amount(1000000.0, "total_sums_insured")
        assert len(errors) == 0
        
        # Negative amount
        errors = self.validator.validate_financial_amount(-1000.0, "total_sums_insured")
        assert len(errors) == 1
        assert "cannot be negative" in errors[0]
        
        # Extremely large amount
        errors = self.validator.validate_financial_amount(1e16, "total_sums_insured")
        assert len(errors) == 1
        assert "exceeds maximum" in errors[0]
    
    def test_validate_percentage(self):
        """Test percentage validation"""
        # Valid percentage
        errors = self.validator.validate_percentage(25.5, "pml_percentage")
        assert len(errors) == 0
        
        # Negative percentage
        errors = self.validator.validate_percentage(-5.0, "pml_percentage")
        assert len(errors) == 1
        assert "cannot be negative" in errors[0]
        
        # Over 100%
        errors = self.validator.validate_percentage(150.0, "pml_percentage")
        assert len(errors) == 1
        assert "cannot exceed 100%" in errors[0]
    
    def test_validate_currency(self):
        """Test currency validation"""
        # Valid currency
        errors = self.validator.validate_currency("USD")
        assert len(errors) == 0
        
        # Invalid length
        errors = self.validator.validate_currency("US")
        assert len(errors) == 1
        assert "must be exactly 3 characters" in errors[0]
        
        # Invalid currency code
        errors = self.validator.validate_currency("XXX")
        assert len(errors) == 1
        assert "Invalid currency code" in errors[0]
    
    def test_validate_complete_analysis_document(self):
        """Test complete Analysis document validation"""
        # Valid document
        data = AnalysisDocumentData(
            reference_number="FAC-001",
            insured_name="Test Company",
            cedant_reinsured="Test Insurer",
            broker_name="Test Broker",
            perils_covered="Fire, Flood",
            total_sums_insured=Decimal("1000000.00"),
            currency="USD",
            period_of_insurance="12 months",
            pml_percentage=Decimal("10.0"),
            share_offered_percentage=Decimal("20.0"),
            confidence_score=0.8
        )
        
        result = self.validator.validate_analysis_document_data(data)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_invalid_analysis_document(self):
        """Test invalid Analysis document validation"""
        # Invalid document with multiple errors
        data = AnalysisDocumentData(
            currency="INVALID",  # Invalid currency
            total_sums_insured=Decimal("-1000.00"),  # Negative amount
            pml_percentage=Decimal("150.0"),  # Over 100%
            confidence_score=1.5  # Over 1.0
        )
        
        result = self.validator.validate_analysis_document_data(data)
        assert not result.is_valid
        assert len(result.errors) > 0


class TestExtractedDataIntegration:
    """Test ExtractedData integration with Analysis document format"""
    
    def test_migrate_to_analysis_format(self):
        """Test migration from legacy format to Analysis format"""
        # Create legacy ExtractedData
        extracted = ExtractedData(
            reference_number="FAC-001",
            insured="Test Company",
            broker="Test Broker",
            reinsured="Test Insurer"
        )
        extracted.financial.sum_insured = Decimal("1000000.00")
        extracted.financial.currency = "USD"
        extracted.risk.location = "New York, USA"
        
        # Migrate to Analysis format
        extracted.migrate_to_analysis_format()
        
        # Check migration
        assert extracted.analysis_data.reference_number == "FAC-001"
        assert extracted.analysis_data.insured_name == "Test Company"
        assert extracted.analysis_data.broker_name == "Test Broker"
        assert extracted.analysis_data.cedant_reinsured == "Test Insurer"
        assert extracted.analysis_data.total_sums_insured == Decimal("1000000.00")
        assert extracted.analysis_data.currency == "USD"
        assert extracted.analysis_data.situation_of_risk == "New York, USA"
    
    def test_to_analysis_excel_row(self):
        """Test conversion to Analysis Excel row format"""
        extracted = ExtractedData(
            reference_number="FAC-001",
            insured="Test Company",
            broker="Test Broker"
        )
        extracted.financial.sum_insured = Decimal("1000000.00")
        extracted.financial.currency = "USD"
        
        # Convert to Excel row
        row = extracted.to_analysis_excel_row()
        
        # Check Excel format
        assert row['Reference Number'] == "FAC-001"
        assert row['Insured'] == "Test Company"
        assert row['Broker'] == "Test Broker"
        assert row['Total Sums Insured'] == 1000000.00
        assert row['Currency'] == "USD"
        assert 'Confidence Score' in row
        assert 'Data Completeness %' in row


class TestFieldMapping:
    """Test Analysis document field mapping"""
    
    def test_field_mapping_completeness(self):
        """Test that field mapping covers all 23 critical fields"""
        # Check that we have exactly 23 field mappings
        assert len(ANALYSIS_DOCUMENT_FIELD_MAPPING) == 23
        
        # Check that all critical fields are mapped
        expected_fields = [
            'reference_number', 'date_received', 'insured_name', 'cedant_reinsured', 'broker_name',
            'perils_covered', 'geographical_limit', 'situation_of_risk', 'occupation_of_insured',
            'main_activities', 'total_sums_insured', 'currency', 'excess_retention',
            'premium_rates', 'period_of_insurance', 'pml_percentage', 'cat_exposure',
            'reinsurance_deductions', 'claims_experience_3_years', 'share_offered_percentage',
            'surveyors_report', 'climate_change_risk', 'esg_risk_assessment'
        ]
        
        mapped_fields = set(ANALYSIS_DOCUMENT_FIELD_MAPPING.values())
        assert mapped_fields == set(expected_fields)
    
    def test_excel_column_names(self):
        """Test Excel column names are properly formatted"""
        excel_columns = list(ANALYSIS_DOCUMENT_FIELD_MAPPING.keys())
        
        # Check some key columns
        assert 'Reference Number' in excel_columns
        assert 'Total Sums Insured' in excel_columns
        assert 'PML %' in excel_columns
        assert 'Share offered %' in excel_columns
        assert 'ESG Risk Assessment' in excel_columns