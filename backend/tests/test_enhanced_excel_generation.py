"""
Test enhanced Excel report generation for Analysis document format
"""
import pytest
import tempfile
import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import pandas as pd

from app.agents.data_extraction_agent import DataExtractionAgent, ExtractedData, AnalysisDocumentData
from app.models.schemas import ValidationResult


class TestEnhancedExcelGeneration:
    """Test enhanced Excel report generation functionality"""
    
    @pytest.fixture
    def data_extraction_agent(self):
        """Create DataExtractionAgent instance"""
        return DataExtractionAgent()
    
    @pytest.fixture
    def sample_analysis_data(self):
        """Create sample AnalysisDocumentData for testing"""
        return AnalysisDocumentData(
            reference_number="TEST-001",
            date_received=datetime(2024, 1, 15),
            insured_name="Test Insurance Company Ltd",
            cedant_reinsured="Test Reinsurance Corp",
            broker_name="Test Brokers Inc",
            perils_covered="Fire, Explosion, Earthquake",
            geographical_limit="Worldwide excluding USA",
            situation_of_risk="Industrial Complex, 123 Main St, London, UK",
            occupation_of_insured="Manufacturing",
            main_activities="Production of electronic components",
            total_sums_insured=Decimal("50000000.00"),
            currency="USD",
            excess_retention=Decimal("1000000.00"),
            premium_rates=Decimal("2.5"),
            period_of_insurance="12 months from 01/01/2024",
            pml_percentage=Decimal("15.0"),
            cat_exposure="Earthquake Zone 3, Wind Zone 2",
            reinsurance_deductions=Decimal("500000.00"),
            claims_experience_3_years="No claims in past 3 years",
            share_offered_percentage=Decimal("25.0"),
            surveyors_report="Available - dated 2023-12-01",
            climate_change_risk="Low - modern construction",
            esg_risk_assessment="Good - ESG compliant operations",
            confidence_score=0.85,
            processing_notes=["Extracted from RI Slip PDF", "High confidence extraction"],
            source_documents=["ri_slip_001.pdf", "supporting_doc_001.docx"]
        )
    
    @pytest.fixture
    def sample_extracted_data_list(self, sample_analysis_data):
        """Create list of ExtractedData objects for testing"""
        # Create multiple ExtractedData objects with different completeness levels
        data_list = []
        
        # Complete data
        complete_data = ExtractedData(
            document_type="pdf",
            confidence_score=0.85,
            analysis_data=sample_analysis_data
        )
        data_list.append(complete_data)
        
        # Partial data (missing some fields)
        partial_analysis_data = AnalysisDocumentData(
            reference_number="TEST-002",
            insured_name="Partial Insurance Ltd",
            cedant_reinsured="Partial Reinsurance",
            broker_name="Partial Brokers",
            total_sums_insured=Decimal("25000000.00"),
            currency="EUR",
            confidence_score=0.65,
            processing_notes=["Partial extraction", "Some fields missing"],
            source_documents=["partial_slip_002.pdf"]
        )
        
        partial_data = ExtractedData(
            document_type="pdf",
            confidence_score=0.65,
            analysis_data=partial_analysis_data
        )
        data_list.append(partial_data)
        
        # Low confidence data
        low_conf_analysis_data = AnalysisDocumentData(
            reference_number="TEST-003",
            insured_name="Low Confidence Corp",
            total_sums_insured=Decimal("10000000.00"),
            currency="GBP",
            confidence_score=0.35,
            processing_notes=["Low confidence extraction", "OCR quality issues"],
            source_documents=["scanned_slip_003.pdf"]
        )
        
        low_conf_data = ExtractedData(
            document_type="scanned_pdf",
            confidence_score=0.35,
            analysis_data=low_conf_analysis_data
        )
        data_list.append(low_conf_data)
        
        return data_list
    
    def test_merge_multiple_ri_slips(self, data_extraction_agent, sample_extracted_data_list):
        """Test merging multiple RI Slip data with gap-filling logic"""
        # Create duplicate data with different fields filled
        base_data = sample_extracted_data_list[0]
        
        # Create a second data object with same reference but different fields
        fill_data = ExtractedData(
            document_type="email",
            confidence_score=0.75,
            analysis_data=AnalysisDocumentData(
                reference_number="TEST-001",  # Same reference
                insured_name="Test Insurance Company Ltd",
                # Missing some fields that base_data has
                # But has some fields that base_data might be missing
                premium_rates=Decimal("3.0"),  # Different value
                cat_exposure="Updated CAT exposure info",
                processing_notes=["Extracted from email attachment"],
                source_documents=["email_attachment_001.pdf"]
            )
        )
        
        # Test merging
        merged_data = data_extraction_agent._merge_multiple_ri_slips([base_data, fill_data])
        
        assert len(merged_data) == 1  # Should merge into one
        merged = merged_data[0]
        
        # Check that source documents were combined
        assert len(merged.analysis_data.source_documents) >= 2
        assert "ri_slip_001.pdf" in merged.analysis_data.source_documents
        assert "email_attachment_001.pdf" in merged.analysis_data.source_documents
        
        # Check that processing notes were combined
        assert "Merged data from 2 documents" in merged.analysis_data.processing_notes
    
    def test_validate_analysis_document_data(self, data_extraction_agent, sample_analysis_data):
        """Test validation of Analysis document data"""
        # Test valid data
        result = data_extraction_agent._validate_analysis_document_data(sample_analysis_data)
        assert result.is_valid
        assert len(result.errors) == 0
        
        # Test invalid data - negative financial amount
        invalid_data = AnalysisDocumentData(
            total_sums_insured=Decimal("-1000000.00"),  # Negative amount
            premium_rates=Decimal("150.0"),  # Over 100%
            currency="INVALID",  # Invalid currency code
            confidence_score=1.5  # Over 1.0
        )
        
        result = data_extraction_agent._validate_analysis_document_data(invalid_data)
        assert not result.is_valid
        assert len(result.errors) > 0
        
        # Check specific error messages
        error_messages = ' '.join(result.errors)
        assert "cannot be negative" in error_messages
        assert "must be between 0 and 100" in error_messages
        assert "Currency code must be 3 characters" in error_messages
    
    def test_calculate_field_confidence_scores(self, data_extraction_agent, sample_extracted_data_list):
        """Test field-level confidence score calculation"""
        data = sample_extracted_data_list[0]  # Complete data
        
        confidence_scores = data_extraction_agent._calculate_field_confidence_scores(data)
        
        # Check that confidence scores are calculated for all fields
        assert isinstance(confidence_scores, dict)
        assert len(confidence_scores) > 0
        
        # Check that commonly available fields have higher confidence
        assert confidence_scores.get('reference_number', 0) > confidence_scores.get('esg_risk_assessment', 0)
        assert confidence_scores.get('insured_name', 0) > confidence_scores.get('climate_change_risk', 0)
        
        # Check that all confidence scores are between 0 and 1
        for field, score in confidence_scores.items():
            assert 0.0 <= score <= 1.0, f"Confidence score for {field} is out of range: {score}"
    
    def test_track_field_completeness(self, data_extraction_agent, sample_analysis_data):
        """Test field completeness tracking"""
        completeness = data_extraction_agent._track_field_completeness(sample_analysis_data)
        
        # Check that completeness is tracked for all 23 critical fields
        assert isinstance(completeness, dict)
        assert len(completeness) == 23
        
        # Check that fields with data are marked as complete
        assert completeness['reference_number'] == True
        assert completeness['insured_name'] == True
        assert completeness['total_sums_insured'] == True
        
        # Test with incomplete data
        incomplete_data = AnalysisDocumentData(
            reference_number="TEST-INCOMPLETE",
            # Missing most other fields
        )
        
        completeness = data_extraction_agent._track_field_completeness(incomplete_data)
        assert completeness['reference_number'] == True
        assert completeness['insured_name'] == False
        assert completeness['total_sums_insured'] == False
    
    def test_enhanced_excel_row_format(self, data_extraction_agent, sample_extracted_data_list):
        """Test enhanced Excel row formatting"""
        data = sample_extracted_data_list[0]
        
        row = data_extraction_agent._to_enhanced_analysis_excel_row(data)
        
        # Check that all expected columns are present
        expected_columns = [
            'Reference Number', 'Date Received', 'Insured', 'Cedant/Reinsured', 'Broker',
            'Perils Covered', 'Geographical Limit', 'Situation of Risk/Voyage',
            'Occupation of Insured', 'Main Activities', 'Total Sums Insured', 'Currency',
            'Excess/Retention', 'Premium Rates (%)', 'Period of Insurance', 'PML %',
            'CAT Exposure', 'Reinsurance Deductions', 'Claims Experience (3 years)',
            'Share offered %', 'Surveyor\'s Report', 'Climate Change Risk',
            'ESG Risk Assessment', 'Confidence Score', 'Data Completeness %',
            'Processing Notes', 'Source Documents'
        ]
        
        for column in expected_columns:
            assert column in row, f"Missing column: {column}"
        
        # Check data formatting
        assert isinstance(row['Total Sums Insured'], float)
        assert row['Date Received'] == '2024-01-15'
        assert '%' in row['Confidence Score']
        assert '%' in row['Data Completeness %']
    
    def test_generate_excel_report(self, data_extraction_agent, sample_extracted_data_list):
        """Test complete Excel report generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_analysis_report.xlsx")
            
            # Generate Excel report
            result_path = data_extraction_agent.generate_excel_report(
                sample_extracted_data_list, 
                output_path
            )
            
            # Check that file was created
            assert os.path.exists(result_path)
            assert result_path == output_path
            
            # Read and validate Excel file
            excel_file = pd.ExcelFile(output_path)
            
            # Check that all expected sheets are present
            expected_sheets = [
                'Analysis Working Sheet',
                'Validation Summary',
                'Field Completeness',
                'Confidence Analysis',
                'Processing Notes'
            ]
            
            for sheet in expected_sheets:
                assert sheet in excel_file.sheet_names, f"Missing sheet: {sheet}"
            
            # Check main data sheet
            main_df = pd.read_excel(output_path, sheet_name='Analysis Working Sheet')
            assert len(main_df) == len(sample_extracted_data_list)
            
            # Check that all 23 critical fields plus metadata are present
            expected_columns = 27  # 23 critical fields + 4 metadata columns
            assert len(main_df.columns) == expected_columns
            
            # Check validation summary sheet
            validation_df = pd.read_excel(output_path, sheet_name='Validation Summary')
            assert len(validation_df) == len(sample_extracted_data_list)
            assert 'Valid' in validation_df.columns
            assert 'Error Count' in validation_df.columns
            
            # Check field completeness sheet
            completeness_df = pd.read_excel(output_path, sheet_name='Field Completeness')
            assert len(completeness_df) == 23  # All 23 critical fields
            assert 'Field Name' in completeness_df.columns
            assert 'Completeness %' in completeness_df.columns
            
            # Check confidence analysis sheet
            confidence_df = pd.read_excel(output_path, sheet_name='Confidence Analysis')
            assert 'Field Name' in confidence_df.columns
            assert 'Average Confidence' in confidence_df.columns
            
            # Check processing notes sheet
            notes_df = pd.read_excel(output_path, sheet_name='Processing Notes')
            assert 'Processing Note' in notes_df.columns
            assert 'Note Type' in notes_df.columns
    
    def test_data_validation_rules(self, data_extraction_agent):
        """Test data validation rules for financial amounts, percentages, and dates"""
        # Test financial validation
        valid_financial_data = AnalysisDocumentData(
            total_sums_insured=Decimal("1000000.00"),
            excess_retention=Decimal("50000.00"),
            reinsurance_deductions=Decimal("25000.00")
        )
        
        result = data_extraction_agent._validate_analysis_document_data(valid_financial_data)
        financial_errors = [e for e in result.errors if 'negative' in e.lower()]
        assert len(financial_errors) == 0
        
        # Test percentage validation
        valid_percentage_data = AnalysisDocumentData(
            premium_rates=Decimal("2.5"),
            pml_percentage=Decimal("15.0"),
            share_offered_percentage=Decimal("25.0")
        )
        
        result = data_extraction_agent._validate_analysis_document_data(valid_percentage_data)
        percentage_errors = [e for e in result.errors if 'between 0 and 100' in e]
        assert len(percentage_errors) == 0
        
        # Test date validation
        valid_date_data = AnalysisDocumentData(
            date_received=datetime(2024, 1, 15)
        )
        
        result = data_extraction_agent._validate_analysis_document_data(valid_date_data)
        date_errors = [e for e in result.errors if 'date' in e.lower()]
        assert len(date_errors) == 0
    
    def test_gap_filling_logic(self, data_extraction_agent):
        """Test gap-filling logic when merging multiple RI Slips"""
        # Create base data with some missing fields
        base_data = ExtractedData(
            analysis_data=AnalysisDocumentData(
                reference_number="TEST-MERGE",
                insured_name="Test Company",
                total_sums_insured=Decimal("1000000.00"),
                # Missing: broker_name, currency, etc.
            )
        )
        
        # Create fill data with the missing fields
        fill_data = ExtractedData(
            analysis_data=AnalysisDocumentData(
                reference_number="TEST-MERGE",
                broker_name="Fill Broker Ltd",
                currency="USD",
                premium_rates=Decimal("2.0"),
                # Missing: insured_name (should not overwrite)
            )
        )
        
        # Test gap filling
        merged = data_extraction_agent._fill_gaps_in_analysis_data(base_data, fill_data)
        
        # Check that gaps were filled
        assert merged.analysis_data.broker_name == "Fill Broker Ltd"
        assert merged.analysis_data.currency == "USD"
        assert merged.analysis_data.premium_rates == Decimal("2.0")
        
        # Check that existing data was not overwritten
        assert merged.analysis_data.insured_name == "Test Company"
        assert merged.analysis_data.total_sums_insured == Decimal("1000000.00")
    
    def test_processing_notes_and_source_tracking(self, data_extraction_agent, sample_extracted_data_list):
        """Test processing notes and data source tracking per field"""
        data = sample_extracted_data_list[0]
        
        # Check that processing notes are tracked
        assert len(data.analysis_data.processing_notes) > 0
        assert "Extracted from RI Slip PDF" in data.analysis_data.processing_notes
        
        # Check that source documents are tracked
        assert len(data.analysis_data.source_documents) > 0
        assert "ri_slip_001.pdf" in data.analysis_data.source_documents
        
        # Test Excel output includes this information
        row = data_extraction_agent._to_enhanced_analysis_excel_row(data)
        assert row['Processing Notes'] != ""
        assert row['Source Documents'] != ""
        assert "ri_slip_001.pdf" in row['Source Documents']


if __name__ == "__main__":
    pytest.main([__file__])