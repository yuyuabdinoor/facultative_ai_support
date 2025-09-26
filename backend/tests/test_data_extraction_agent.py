"""
Test suite for Data Extraction Agent

Tests the comprehensive data extraction capabilities including:
- Email processing with attachments
- Financial data extraction
- Risk data extraction
- Excel report generation
- Integration with OCR agent
"""

import pytest
import tempfile
import os
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from app.agents.data_extraction_agent import (
    DataExtractionAgent, ExtractedData, FinancialData, RiskData, PartyData,
    data_extraction_agent
)
from app.agents.ocr_agent import (
    EmailContent, AttachmentData, OCRResult, WordDocumentData, 
    ExcelData, PowerPointData, TextRegion
)


class TestDataExtractionAgent:
    """Test cases for DataExtractionAgent"""
    
    @pytest.fixture
    def agent(self):
        """Create a data extraction agent instance for testing"""
        return DataExtractionAgent()
    
    @pytest.fixture
    def sample_email_content(self):
        """Create sample email content for testing"""
        # Sample OCR result for attachment
        ocr_result = OCRResult(
            text="Policy Number: FAC-2024-001\nSum Insured: USD 5,000,000\nInsured: ABC Manufacturing Ltd\nLocation: New York, USA\nCoverage: Property Insurance",
            confidence=0.95,
            regions=[
                TextRegion(text="Policy Number: FAC-2024-001", confidence=0.98, bbox=[0, 0, 100, 20]),
                TextRegion(text="Sum Insured: USD 5,000,000", confidence=0.96, bbox=[0, 20, 100, 40])
            ],
            metadata={'file_type': 'pdf'},
            processing_time=2.5,
            success=True
        )
        
        # Sample attachment
        attachment = AttachmentData(
            filename="policy_details.pdf",
            content_type=".pdf",
            size=1024000,
            processed_content=ocr_result,
            extraction_success=True
        )
        
        # Sample email
        return EmailContent(
            subject="Facultative Reinsurance Placement - ABC Manufacturing",
            sender="broker@example.com",
            recipients=["underwriter@reinsurer.com"],
            body="Please find attached the placement details for ABC Manufacturing property risk.",
            attachments=[attachment],
            date=datetime(2024, 1, 15),
            metadata={'message_class': 'IPM.Note'},
            document_type="risk_placement"
        )
    
    @pytest.fixture
    def sample_word_document(self):
        """Create sample Word document data"""
        return WordDocumentData(
            text="Risk Assessment Report\nInsured: XYZ Corporation\nSum Insured: EUR 10,000,000\nLocation: London, UK",
            paragraphs=[
                "Risk Assessment Report",
                "Insured: XYZ Corporation", 
                "Sum Insured: EUR 10,000,000",
                "Location: London, UK"
            ],
            tables=[
                [
                    ["Field", "Value"],
                    ["Policy Number", "POL-2024-002"],
                    ["Premium", "EUR 50,000"],
                    ["Deductible", "EUR 25,000"]
                ]
            ],
            metadata={'file_format': '.docx'}
        )
    
    def test_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'patterns')
        assert hasattr(agent, 'keywords')
        assert hasattr(agent, 'excel_template')
        
        # Check patterns are compiled
        assert 'currency_amount' in agent.patterns
        assert 'policy_number' in agent.patterns
        
        # Check keywords are loaded
        assert 'document_types' in agent.keywords
        assert 'coverage_types' in agent.keywords
    
    def test_extract_from_email(self, agent, sample_email_content):
        """Test extraction from email content"""
        result = agent.extract_from_email(sample_email_content)
        
        assert isinstance(result, ExtractedData)
        assert result.document_type == "risk_placement"
        assert result.confidence_score > 0
        
        # Check if data was extracted from attachment
        assert result.financial.sum_insured is not None
        assert result.financial.currency == "USD"
        assert result.insured is not None
        assert result.risk.location is not None
    
    def test_extract_from_text_patterns(self, agent):
        """Test pattern-based text extraction"""
        text = """
        Policy Reference: FAC-2024-123
        Insured: Global Industries Inc.
        Sum Insured: USD 15,000,000
        Premium: USD 75,000
        Location: Chicago, Illinois
        Coverage: Property & Casualty
        Date: 15/01/2024
        """
        
        extracted_data = ExtractedData()
        agent._extract_from_text(text, extracted_data)
        
        assert extracted_data.reference_number == "FAC-2024-123"
        assert extracted_data.financial.currency == "USD"
        assert extracted_data.risk.location is not None
        assert extracted_data.date is not None
    
    def test_extract_from_table(self, agent):
        """Test extraction from table data"""
        table = [
            ["Field", "Value", "Currency"],
            ["Sum Insured", "5000000", "USD"],
            ["Premium", "25000", "USD"],
            ["Deductible", "10000", "USD"],
            ["Insured", "Test Company Ltd", ""],
            ["Broker", "Test Broker Inc", ""]
        ]
        
        extracted_data = ExtractedData()
        agent._extract_from_table(table, extracted_data)
        
        assert extracted_data.financial.sum_insured == Decimal('5000000')
        assert extracted_data.financial.premium == Decimal('25000')
        assert extracted_data.financial.deductible == Decimal('10000')
        assert extracted_data.insured == "Test Company Ltd"
        assert extracted_data.broker == "Test Broker Inc"
    
    def test_extract_from_excel_sheet(self, agent):
        """Test extraction from Excel sheet data"""
        sheet_data = [
            {
                "Policy Number": "EXL-2024-001",
                "Insured": "Excel Test Corp",
                "Sum Insured": 8000000,
                "Currency": "EUR",
                "Location": "Paris, France"
            },
            {
                "Policy Number": "EXL-2024-002", 
                "Insured": "Another Corp",
                "Sum Insured": 12000000,
                "Currency": "GBP",
                "Location": "London, UK"
            }
        ]
        
        extracted_data = ExtractedData()
        agent._extract_from_excel_sheet(sheet_data, extracted_data)
        
        assert extracted_data.insured == "Excel Test Corp"
        assert extracted_data.financial.sum_insured == Decimal('8000000')
    
    def test_parse_amount(self, agent):
        """Test monetary amount parsing"""
        test_cases = [
            ("USD 1,000,000", Decimal('1000000')),
            ("EUR 500,000.50", Decimal('500000.50')),
            ("GBP 2,500,000", Decimal('2500000')),
            ("$1,234,567.89", Decimal('1234567.89')),
            ("invalid", None),
            ("", None)
        ]
        
        for amount_text, expected in test_cases:
            result = agent._parse_amount(amount_text)
            assert result == expected
    
    def test_parse_date(self, agent):
        """Test date parsing"""
        test_cases = [
            ("15/01/2024", datetime(2024, 1, 15)),
            ("01-15-2024", datetime(2024, 1, 15)),
            ("2024-01-15", datetime(2024, 1, 15)),
            ("15 Jan 2024", datetime(2024, 1, 15)),
            ("Jan 15, 2024", datetime(2024, 1, 15)),
            ("invalid date", None)
        ]
        
        for date_text, expected in test_cases:
            result = agent._parse_date(date_text)
            if expected:
                assert result == expected
            else:
                assert result is None
    
    def test_calculate_confidence(self, agent):
        """Test confidence score calculation"""
        # Complete data
        complete_data = ExtractedData(
            reference_number="TEST-001",
            insured="Test Company",
            document_type="risk_placement",
            date=datetime.now()
        )
        complete_data.financial.sum_insured = Decimal('1000000')
        complete_data.financial.currency = "USD"
        complete_data.risk.location = "New York"
        complete_data.broker = "Test Broker"
        
        score = agent._calculate_confidence(complete_data)
        assert score > 0.8  # Should have high confidence
        
        # Minimal data
        minimal_data = ExtractedData()
        score = agent._calculate_confidence(minimal_data)
        assert score < 0.2  # Should have low confidence
    
    def test_generate_excel_report(self, agent):
        """Test Excel report generation"""
        # Create sample extracted data
        data1 = ExtractedData(
            reference_number="TEST-001",
            insured="Test Company 1",
            broker="Test Broker",
            document_type="risk_placement",
            date=datetime(2024, 1, 15),
            line_of_business="property",
            confidence_score=0.85
        )
        data1.financial.sum_insured = Decimal('1000000')
        data1.financial.currency = "USD"
        data1.financial.premium = Decimal('5000')
        data1.risk.location = "New York"
        
        data2 = ExtractedData(
            reference_number="TEST-002",
            insured="Test Company 2",
            broker="Another Broker",
            document_type="treaty_slip",
            date=datetime(2024, 1, 16),
            line_of_business="casualty",
            confidence_score=0.92
        )
        data2.financial.sum_insured = Decimal('2000000')
        data2.financial.currency = "EUR"
        data2.risk.location = "London"
        
        extracted_data_list = [data1, data2]
        
        # Generate report
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            result_path = agent.generate_excel_report(extracted_data_list, output_path)
            assert result_path == output_path
            assert os.path.exists(output_path)
            
            # Verify file is not empty
            assert os.path.getsize(output_path) > 0
            
        finally:
            # Clean up
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @patch('app.agents.data_extraction_agent.ocr_agent')
    def test_process_documents(self, mock_ocr_agent, agent):
        """Test processing multiple documents"""
        # Mock OCR agent responses
        mock_email = EmailContent(
            subject="Test Email",
            sender="test@example.com",
            body="Test body",
            attachments=[],
            date=datetime.now(),
            document_type="risk_placement"
        )
        
        mock_ocr_result = OCRResult(
            text="Test OCR text with USD 1,000,000 sum insured",
            confidence=0.9,
            regions=[],
            metadata={},
            processing_time=1.0,
            success=True
        )
        
        mock_ocr_agent.process_document.side_effect = [mock_email, mock_ocr_result]
        
        file_paths = ["test1.msg", "test2.pdf"]
        results = agent.process_documents(file_paths)
        
        assert len(results) == 2
        assert all(isinstance(result, ExtractedData) for result in results)
        assert mock_ocr_agent.process_document.call_count == 2
    
    def test_financial_data_model(self):
        """Test FinancialData model"""
        financial = FinancialData(
            sum_insured=Decimal('1000000'),
            premium=Decimal('5000'),
            currency="USD",
            coverage_type="property"
        )
        
        assert financial.sum_insured == Decimal('1000000')
        assert financial.premium == Decimal('5000')
        assert financial.currency == "USD"
        assert financial.coverage_type == "property"
    
    def test_risk_data_model(self):
        """Test RiskData model"""
        risk = RiskData(
            risk_type="property",
            location="New York",
            country="USA",
            industry="manufacturing",
            year_built=2010
        )
        
        assert risk.risk_type == "property"
        assert risk.location == "New York"
        assert risk.country == "USA"
        assert risk.industry == "manufacturing"
        assert risk.year_built == 2010
    
    def test_party_data_model(self):
        """Test PartyData model"""
        party = PartyData(
            name="John Doe",
            email="john@example.com",
            role="broker",
            phone="+1-555-0123"
        )
        
        assert party.name == "John Doe"
        assert party.email == "john@example.com"
        assert party.role == "broker"
        assert party.phone == "+1-555-0123"
    
    def test_extracted_data_model(self):
        """Test ExtractedData model"""
        data = ExtractedData(
            reference_number="TEST-001",
            document_type="risk_placement",
            date=datetime(2024, 1, 15),
            confidence_score=0.85,
            extraction_method="hybrid"
        )
        
        assert data.reference_number == "TEST-001"
        assert data.document_type == "risk_placement"
        assert data.date == datetime(2024, 1, 15)
        assert data.confidence_score == 0.85
        assert data.extraction_method == "hybrid"
        assert isinstance(data.financial, FinancialData)
        assert isinstance(data.risk, RiskData)
        assert isinstance(data.parties, list)
    
    def test_global_agent_instance(self):
        """Test global agent instance"""
        assert data_extraction_agent is not None
        assert isinstance(data_extraction_agent, DataExtractionAgent)


class TestIntegrationWithOCRAgent:
    """Integration tests with OCR Agent"""
    
    def test_email_with_word_attachment(self):
        """Test processing email with Word document attachment"""
        word_doc = WordDocumentData(
            text="Policy Details\nInsured: Integration Test Corp\nSum Insured: USD 3,000,000",
            paragraphs=["Policy Details", "Insured: Integration Test Corp"],
            tables=[],
            metadata={'file_format': '.docx'}
        )
        
        attachment = AttachmentData(
            filename="policy.docx",
            content_type=".docx",
            size=50000,
            processed_content=word_doc,
            extraction_success=True
        )
        
        email = EmailContent(
            subject="Policy Attachment Test",
            sender="test@broker.com",
            body="Please review attached policy",
            attachments=[attachment],
            date=datetime.now(),
            document_type="risk_placement"
        )
        
        agent = DataExtractionAgent()
        result = agent.extract_from_email(email)
        
        assert result.insured == "Integration Test Corp"
        assert result.financial.sum_insured == Decimal('3000000')
        assert result.financial.currency == "USD"
    
    def test_email_with_excel_attachment(self):
        """Test processing email with Excel attachment"""
        excel_data = ExcelData(
            sheets={
                "Policy Data": [
                    {
                        "Reference": "EXL-INT-001",
                        "Insured": "Excel Integration Corp",
                        "Sum Insured": 5000000,
                        "Currency": "EUR",
                        "Premium": 25000
                    }
                ]
            },
            metadata={'file_format': '.xlsx'},
            total_rows=1,
            total_sheets=1
        )
        
        attachment = AttachmentData(
            filename="data.xlsx",
            content_type=".xlsx",
            size=75000,
            processed_content=excel_data,
            extraction_success=True
        )
        
        email = EmailContent(
            subject="Excel Data Test",
            sender="data@broker.com",
            body="Excel data attached",
            attachments=[attachment],
            date=datetime.now(),
            document_type="treaty_slip"
        )
        
        agent = DataExtractionAgent()
        result = agent.extract_from_email(email)
        
        assert result.insured == "Excel Integration Corp"
        assert result.financial.sum_insured == Decimal('5000000')


if __name__ == "__main__":
    pytest.main([__file__])