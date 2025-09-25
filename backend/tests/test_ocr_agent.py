"""
Comprehensive tests for OCR Processing Agent

Tests cover all document types and processing scenarios:
- PDF text extraction
- Scanned PDF OCR
- Email parsing
- Excel processing
- Text region detection
- Error handling
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from PIL import Image
import numpy as np
from datetime import datetime

from app.agents.ocr_agent import (
    OCRProcessingAgent, 
    OCRResult, 
    EmailContent, 
    ExcelData, 
    TextRegion
)


class TestOCRProcessingAgent:
    """Test suite for OCR Processing Agent"""
    
    @pytest.fixture
    def ocr_agent(self):
        """Create OCR agent instance for testing"""
        return OCRProcessingAgent()
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing"""
        return "This is a sample PDF document.\nIt contains multiple lines of text.\nUsed for testing purposes."
    
    @pytest.fixture
    def sample_email_data(self):
        """Sample email data for testing"""
        return {
            'subject': 'Facultative Reinsurance Application',
            'sender': 'underwriter@insurance.com',
            'body': 'Please find attached the reinsurance application for review.',
            'recipients': ['analyst@reinsurance.com'],
            'date': datetime.now()
        }
    
    @pytest.fixture
    def sample_excel_data(self):
        """Sample Excel data for testing"""
        return {
            'Sheet1': [
                {'Asset': 'Building A', 'Value': 1000000, 'Location': 'New York'},
                {'Asset': 'Building B', 'Value': 2000000, 'Location': 'California'}
            ],
            'Sheet2': [
                {'Risk Factor': 'Fire', 'Probability': 0.05},
                {'Risk Factor': 'Flood', 'Probability': 0.03}
            ]
        }
    
    def create_temp_pdf(self, content: str) -> str:
        """Create a temporary PDF file for testing"""
        # This is a mock implementation - in real tests you'd use a PDF library
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(content.encode())
        temp_file.close()
        return temp_file.name
    
    def create_temp_excel(self, data: dict) -> str:
        """Create a temporary Excel file for testing"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        temp_file.close()
        
        with pd.ExcelWriter(temp_file.name, engine='openpyxl') as writer:
            for sheet_name, sheet_data in data.items():
                df = pd.DataFrame(sheet_data)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return temp_file.name
    
    def create_temp_image(self) -> str:
        """Create a temporary image file for testing"""
        # Create a simple test image
        image = Image.new('RGB', (100, 50), color='white')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        image.save(temp_file.name)
        return temp_file.name
    
    def test_initialization(self, ocr_agent):
        """Test OCR agent initialization"""
        assert ocr_agent is not None
        # Note: OCR model might be None in test environment without GPU
    
    @patch('app.agents.ocr_agent.pdfplumber')
    def test_process_pdf_success(self, mock_pdfplumber, ocr_agent, sample_pdf_content):
        """Test successful PDF text extraction"""
        # Mock pdfplumber
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = sample_pdf_content
        mock_page.width = 612
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Create temp file
        temp_file = self.create_temp_pdf(sample_pdf_content)
        
        try:
            result = ocr_agent.process_pdf(temp_file)
            
            assert isinstance(result, OCRResult)
            assert result.success is True
            assert result.text == sample_pdf_content
            assert result.confidence > 0
            assert len(result.regions) > 0
            assert result.metadata['file_type'] == 'pdf'
            assert result.metadata['extraction_method'] == 'pdfplumber'
            
        finally:
            os.unlink(temp_file)
    
    @patch('app.agents.ocr_agent.pdfplumber')
    def test_process_pdf_error(self, mock_pdfplumber, ocr_agent):
        """Test PDF processing error handling"""
        mock_pdfplumber.open.side_effect = Exception("PDF processing error")
        
        temp_file = self.create_temp_pdf("test content")
        
        try:
            result = ocr_agent.process_pdf(temp_file)
            
            assert isinstance(result, OCRResult)
            assert result.success is False
            assert result.error_message == "PDF processing error"
            assert result.text == ""
            assert result.confidence == 0.0
            
        finally:
            os.unlink(temp_file)
    
    @patch('app.agents.ocr_agent.DocumentFile')
    @patch('app.agents.ocr_agent.ocr_predictor')
    def test_process_scanned_pdf_success(self, mock_ocr_predictor, mock_doc_file, ocr_agent):
        """Test successful scanned PDF OCR processing"""
        # Mock DOCTR components
        mock_word = MagicMock()
        mock_word.value = "sample"
        mock_word.confidence = 0.95
        mock_word.geometry = [[0, 0], [50, 20]]
        
        mock_line = MagicMock()
        mock_line.words = [mock_word]
        
        mock_block = MagicMock()
        mock_block.lines = [mock_line]
        
        mock_page = MagicMock()
        mock_page.blocks = [mock_block]
        
        mock_result = MagicMock()
        mock_result.pages = [mock_page]
        
        mock_ocr_model = MagicMock()
        mock_ocr_model.return_value = mock_result
        ocr_agent.ocr_model = mock_ocr_model
        
        temp_file = self.create_temp_pdf("scanned content")
        
        try:
            result = ocr_agent.process_scanned_pdf(temp_file)
            
            assert isinstance(result, OCRResult)
            assert result.success is True
            assert "sample" in result.text
            assert result.confidence > 0
            assert len(result.regions) > 0
            assert result.metadata['file_type'] == 'scanned_pdf'
            assert result.metadata['extraction_method'] == 'doctr_ocr'
            
        finally:
            os.unlink(temp_file)
    
    def test_process_scanned_pdf_no_model(self, ocr_agent):
        """Test scanned PDF processing without OCR model"""
        ocr_agent.ocr_model = None
        
        temp_file = self.create_temp_pdf("scanned content")
        
        try:
            result = ocr_agent.process_scanned_pdf(temp_file)
            
            assert isinstance(result, OCRResult)
            assert result.success is False
            assert "OCR model not initialized" in result.error_message
            
        finally:
            os.unlink(temp_file)
    
    @patch('app.agents.ocr_agent.extract_msg')
    def test_process_email_success(self, mock_extract_msg, ocr_agent, sample_email_data):
        """Test successful email parsing"""
        # Mock extract_msg.Message
        mock_msg = MagicMock()
        mock_msg.subject = sample_email_data['subject']
        mock_msg.sender = sample_email_data['sender']
        mock_msg.body = sample_email_data['body']
        mock_msg.to = sample_email_data['recipients'][0]
        mock_msg.cc = None
        mock_msg.date = sample_email_data['date']
        mock_msg.attachments = []
        mock_msg.messageClass = 'IPM.Note'
        mock_msg.importance = 'Normal'
        mock_msg.sensitivity = 'Normal'
        
        mock_extract_msg.Message.return_value = mock_msg
        
        # Create temp .msg file (mock)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.msg')
        temp_file.close()
        
        try:
            result = ocr_agent.process_email(temp_file.name)
            
            assert isinstance(result, EmailContent)
            assert result.subject == sample_email_data['subject']
            assert result.sender == sample_email_data['sender']
            assert result.body == sample_email_data['body']
            assert len(result.recipients) > 0
            assert result.date == sample_email_data['date']
            
        finally:
            os.unlink(temp_file.name)
    
    @patch('app.agents.ocr_agent.extract_msg')
    def test_process_email_error(self, mock_extract_msg, ocr_agent):
        """Test email processing error handling"""
        mock_extract_msg.Message.side_effect = Exception("Email parsing error")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.msg')
        temp_file.close()
        
        try:
            result = ocr_agent.process_email(temp_file.name)
            
            assert isinstance(result, EmailContent)
            assert result.subject == ""
            assert result.sender == ""
            assert result.body == ""
            assert 'error' in result.metadata
            
        finally:
            os.unlink(temp_file.name)
    
    def test_process_excel_success(self, ocr_agent, sample_excel_data):
        """Test successful Excel processing"""
        temp_file = self.create_temp_excel(sample_excel_data)
        
        try:
            result = ocr_agent.process_excel(temp_file)
            
            assert isinstance(result, ExcelData)
            assert result.total_sheets == 2
            assert result.total_rows == 4  # 2 rows per sheet
            assert 'Sheet1' in result.sheets
            assert 'Sheet2' in result.sheets
            assert len(result.sheets['Sheet1']) == 2
            assert len(result.sheets['Sheet2']) == 2
            
        finally:
            os.unlink(temp_file)
    
    def test_process_excel_error(self, ocr_agent):
        """Test Excel processing error handling"""
        # Try to process non-existent file
        result = ocr_agent.process_excel("nonexistent.xlsx")
        
        assert isinstance(result, ExcelData)
        assert 'error' in result.metadata
        assert result.total_sheets == 0
        assert result.total_rows == 0
    
    @patch('app.agents.ocr_agent.DocumentFile')
    def test_extract_text_regions_from_path(self, mock_doc_file, ocr_agent):
        """Test text region extraction from image file path"""
        # Mock DOCTR components
        mock_word = MagicMock()
        mock_word.value = "test"
        mock_word.confidence = 0.9
        mock_word.geometry = [[10, 10], [50, 30]]
        
        mock_line = MagicMock()
        mock_line.words = [mock_word]
        
        mock_block = MagicMock()
        mock_block.lines = [mock_line]
        
        mock_page = MagicMock()
        mock_page.blocks = [mock_block]
        
        mock_result = MagicMock()
        mock_result.pages = [mock_page]
        
        mock_ocr_model = MagicMock()
        mock_ocr_model.return_value = mock_result
        ocr_agent.ocr_model = mock_ocr_model
        
        temp_image = self.create_temp_image()
        
        try:
            regions = ocr_agent.extract_text_regions(temp_image)
            
            assert len(regions) == 1
            assert isinstance(regions[0], TextRegion)
            assert regions[0].text == "test"
            assert regions[0].confidence == 0.9
            assert len(regions[0].bbox) == 4
            
        finally:
            os.unlink(temp_image)
    
    def test_extract_text_regions_from_array(self, ocr_agent):
        """Test text region extraction from numpy array"""
        # Create test image array
        image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Mock OCR model
        mock_word = MagicMock()
        mock_word.value = "array_test"
        mock_word.confidence = 0.85
        mock_word.geometry = [[5, 5], [45, 25]]
        
        mock_line = MagicMock()
        mock_line.words = [mock_word]
        
        mock_block = MagicMock()
        mock_block.lines = [mock_line]
        
        mock_page = MagicMock()
        mock_page.blocks = [mock_block]
        
        mock_result = MagicMock()
        mock_result.pages = [mock_page]
        
        mock_ocr_model = MagicMock()
        mock_ocr_model.return_value = mock_result
        ocr_agent.ocr_model = mock_ocr_model
        
        regions = ocr_agent.extract_text_regions(image_array)
        
        assert len(regions) == 1
        assert regions[0].text == "array_test"
        assert regions[0].confidence == 0.85
    
    def test_extract_text_regions_no_model(self, ocr_agent):
        """Test text region extraction without OCR model"""
        ocr_agent.ocr_model = None
        
        temp_image = self.create_temp_image()
        
        try:
            regions = ocr_agent.extract_text_regions(temp_image)
            assert len(regions) == 0
            
        finally:
            os.unlink(temp_image)
    
    @patch('app.agents.ocr_agent.pdfplumber')
    def test_detect_document_type_text_pdf(self, mock_pdfplumber, ocr_agent):
        """Test document type detection for text-based PDF"""
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is a long text content that should be detected as text-based PDF"
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        temp_file = self.create_temp_pdf("test content")
        
        try:
            doc_type = ocr_agent.detect_document_type(temp_file)
            assert doc_type == 'text_pdf'
            
        finally:
            os.unlink(temp_file)
    
    @patch('app.agents.ocr_agent.pdfplumber')
    def test_detect_document_type_scanned_pdf(self, mock_pdfplumber, ocr_agent):
        """Test document type detection for scanned PDF"""
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""  # No text content
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        temp_file = self.create_temp_pdf("test content")
        
        try:
            doc_type = ocr_agent.detect_document_type(temp_file)
            assert doc_type == 'scanned_pdf'
            
        finally:
            os.unlink(temp_file)
    
    @patch('app.agents.ocr_agent.pdfplumber')
    def test_detect_document_type_error(self, mock_pdfplumber, ocr_agent):
        """Test document type detection error handling"""
        mock_pdfplumber.open.side_effect = Exception("PDF error")
        
        temp_file = self.create_temp_pdf("test content")
        
        try:
            doc_type = ocr_agent.detect_document_type(temp_file)
            assert doc_type == 'scanned_pdf'  # Default fallback
            
        finally:
            os.unlink(temp_file)
    
    def test_process_document_pdf_auto_detect(self, ocr_agent):
        """Test automatic document processing for PDF"""
        with patch.object(ocr_agent, 'detect_document_type', return_value='text_pdf'), \
             patch.object(ocr_agent, 'process_pdf') as mock_process_pdf:
            
            mock_process_pdf.return_value = OCRResult(
                text="test", confidence=0.9, processing_time=1.0
            )
            
            temp_file = self.create_temp_pdf("test content")
            
            try:
                result = ocr_agent.process_document(temp_file)
                mock_process_pdf.assert_called_once_with(temp_file)
                assert isinstance(result, OCRResult)
                
            finally:
                os.unlink(temp_file)
    
    def test_process_document_email(self, ocr_agent):
        """Test automatic document processing for email"""
        with patch.object(ocr_agent, 'process_email') as mock_process_email:
            
            mock_process_email.return_value = EmailContent(
                subject="test", sender="test@example.com", body="test body"
            )
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.msg')
            temp_file.close()
            
            try:
                result = ocr_agent.process_document(temp_file.name)
                mock_process_email.assert_called_once_with(temp_file.name)
                assert isinstance(result, EmailContent)
                
            finally:
                os.unlink(temp_file.name)
    
    def test_process_document_excel(self, ocr_agent, sample_excel_data):
        """Test automatic document processing for Excel"""
        temp_file = self.create_temp_excel(sample_excel_data)
        
        try:
            result = ocr_agent.process_document(temp_file)
            assert isinstance(result, ExcelData)
            assert result.total_sheets == 2
            
        finally:
            os.unlink(temp_file)
    
    def test_process_document_unsupported_type(self, ocr_agent):
        """Test processing with unsupported document type"""
        temp_file = self.create_temp_pdf("test content")
        
        try:
            with pytest.raises(ValueError, match="Unsupported document type"):
                ocr_agent.process_document(temp_file, document_type="unsupported")
                
        finally:
            os.unlink(temp_file)


class TestOCRDataModels:
    """Test OCR data models"""
    
    def test_text_region_model(self):
        """Test TextRegion model"""
        region = TextRegion(
            text="sample text",
            confidence=0.95,
            bbox=[10, 20, 100, 40],
            page_number=1
        )
        
        assert region.text == "sample text"
        assert region.confidence == 0.95
        assert region.bbox == [10, 20, 100, 40]
        assert region.page_number == 1
    
    def test_ocr_result_model(self):
        """Test OCRResult model"""
        regions = [
            TextRegion(text="word1", confidence=0.9, bbox=[0, 0, 50, 20]),
            TextRegion(text="word2", confidence=0.95, bbox=[60, 0, 110, 20])
        ]
        
        result = OCRResult(
            text="word1 word2",
            confidence=0.925,
            regions=regions,
            metadata={"page_count": 1},
            processing_time=2.5,
            success=True
        )
        
        assert result.text == "word1 word2"
        assert result.confidence == 0.925
        assert len(result.regions) == 2
        assert result.metadata["page_count"] == 1
        assert result.processing_time == 2.5
        assert result.success is True
        assert result.error_message is None
    
    def test_email_content_model(self):
        """Test EmailContent model"""
        email = EmailContent(
            subject="Test Subject",
            sender="sender@example.com",
            recipients=["recipient1@example.com", "recipient2@example.com"],
            body="Email body content",
            attachments=[{"filename": "attachment.pdf", "size": 1024}],
            date=datetime(2024, 1, 1, 12, 0, 0),
            metadata={"importance": "high"}
        )
        
        assert email.subject == "Test Subject"
        assert email.sender == "sender@example.com"
        assert len(email.recipients) == 2
        assert email.body == "Email body content"
        assert len(email.attachments) == 1
        assert email.date.year == 2024
        assert email.metadata["importance"] == "high"
    
    def test_excel_data_model(self):
        """Test ExcelData model"""
        sheets_data = {
            "Sheet1": [{"col1": "value1", "col2": "value2"}],
            "Sheet2": [{"colA": "valueA", "colB": "valueB"}]
        }
        
        excel = ExcelData(
            sheets=sheets_data,
            metadata={"file_format": ".xlsx"},
            total_rows=2,
            total_sheets=2
        )
        
        assert len(excel.sheets) == 2
        assert "Sheet1" in excel.sheets
        assert "Sheet2" in excel.sheets
        assert excel.metadata["file_format"] == ".xlsx"
        assert excel.total_rows == 2
        assert excel.total_sheets == 2


if __name__ == "__main__":
    pytest.main([__file__])