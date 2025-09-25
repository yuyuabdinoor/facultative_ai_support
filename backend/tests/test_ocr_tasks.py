"""
Tests for OCR Celery tasks

Tests cover asynchronous task processing, error handling, and database integration
for OCR processing tasks.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
from datetime import datetime

from app.agents.tasks import (
    process_document_ocr,
    process_pdf_ocr,
    process_email_parsing,
    process_excel_parsing,
    batch_process_documents
)
from app.agents.ocr_agent import OCRResult, EmailContent, ExcelData
from app.models.database import Document


class TestOCRTasks:
    """Test suite for OCR Celery tasks"""
    
    @pytest.fixture
    def mock_document(self):
        """Mock document for testing"""
        doc = Mock(spec=Document)
        doc.id = "test-doc-id"
        doc.file_path = "/tmp/test.pdf"
        doc.document_type = "pdf"
        doc.metadata = {}
        doc.processed = False
        return doc
    
    @pytest.fixture
    def mock_ocr_result(self):
        """Mock OCR result for testing"""
        return OCRResult(
            text="Sample extracted text",
            confidence=0.95,
            regions=[],
            metadata={"page_count": 1, "extraction_method": "test"},
            processing_time=2.5,
            success=True
        )
    
    @pytest.fixture
    def mock_email_result(self):
        """Mock email parsing result for testing"""
        return EmailContent(
            subject="Test Email",
            sender="test@example.com",
            recipients=["recipient@example.com"],
            body="Email body content",
            attachments=[],
            date=datetime.now(),
            metadata={}
        )
    
    @pytest.fixture
    def mock_excel_result(self):
        """Mock Excel processing result for testing"""
        return ExcelData(
            sheets={"Sheet1": [{"col1": "value1"}]},
            metadata={"file_format": ".xlsx"},
            total_rows=1,
            total_sheets=1
        )
    
    def create_temp_file(self, suffix='.pdf'):
        """Create temporary file for testing"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(b"test content")
        temp_file.close()
        return temp_file.name
    
    @patch('app.agents.tasks.get_db')
    @patch('app.agents.tasks.ocr_agent')
    def test_process_document_ocr_success(self, mock_ocr_agent, mock_get_db, mock_document, mock_ocr_result):
        """Test successful document OCR processing"""
        # Setup mocks
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        mock_ocr_agent.process_document.return_value = mock_ocr_result
        
        # Create temp file
        temp_file = self.create_temp_file()
        
        try:
            # Create mock task
            mock_task = MagicMock()
            mock_task.request.id = "task-123"
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            # Execute task
            result = process_document_ocr(mock_task, "test-doc-id", temp_file)
            
            # Assertions
            assert result['success'] is True
            assert result['document_id'] == "test-doc-id"
            assert result['result_type'] == "OCRResult"
            assert result['processing_completed'] is True
            
            # Verify OCR agent was called
            mock_ocr_agent.process_document.assert_called_once_with(temp_file, None)
            
            # Verify database update
            assert mock_document.processed is True
            assert 'ocr_result' in mock_document.metadata
            mock_db.commit.assert_called_once()
            
        finally:
            os.unlink(temp_file)
    
    @patch('app.agents.tasks.get_db')
    @patch('app.agents.tasks.ocr_agent')
    def test_process_document_ocr_file_not_found(self, mock_ocr_agent, mock_get_db):
        """Test document OCR processing with missing file"""
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        
        mock_task = MagicMock()
        mock_task.request.id = "task-123"
        mock_task.request.retries = 0
        mock_task.max_retries = 3
        
        # Execute task with non-existent file
        result = process_document_ocr(mock_task, "test-doc-id", "/nonexistent/file.pdf")
        
        # Assertions
        assert result['success'] is False
        assert "Document file not found" in result['error']
        assert result['processing_completed'] is False
    
    @patch('app.agents.tasks.get_db')
    @patch('app.agents.tasks.ocr_agent')
    def test_process_document_ocr_processing_error(self, mock_ocr_agent, mock_get_db, mock_document):
        """Test document OCR processing with processing error"""
        # Setup mocks
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        mock_ocr_agent.process_document.side_effect = Exception("OCR processing failed")
        
        temp_file = self.create_temp_file()
        
        try:
            mock_task = MagicMock()
            mock_task.request.id = "task-123"
            mock_task.request.retries = 3  # Max retries reached
            mock_task.max_retries = 3
            
            # Execute task
            result = process_document_ocr(mock_task, "test-doc-id", temp_file)
            
            # Assertions
            assert result['success'] is False
            assert "OCR processing failed" in result['error']
            assert result['processing_completed'] is False
            
            # Verify error was stored in database
            assert 'processing_error' in mock_document.metadata
            
        finally:
            os.unlink(temp_file)
    
    @patch('app.agents.tasks.get_db')
    @patch('app.agents.tasks.ocr_agent')
    def test_process_document_ocr_retry_logic(self, mock_ocr_agent, mock_get_db, mock_document):
        """Test document OCR processing retry logic"""
        # Setup mocks
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        mock_ocr_agent.process_document.side_effect = Exception("Temporary failure")
        
        temp_file = self.create_temp_file()
        
        try:
            mock_task = MagicMock()
            mock_task.request.id = "task-123"
            mock_task.request.retries = 1  # Less than max retries
            mock_task.max_retries = 3
            mock_task.retry = MagicMock(side_effect=Exception("Retry called"))
            
            # Execute task - should raise retry exception
            with pytest.raises(Exception, match="Retry called"):
                process_document_ocr(mock_task, "test-doc-id", temp_file)
            
            # Verify retry was called with exponential backoff
            mock_task.retry.assert_called_once_with(countdown=60 * (2 ** 1))
            
        finally:
            os.unlink(temp_file)
    
    @patch('app.agents.tasks.get_db')
    @patch('app.agents.tasks.ocr_agent')
    def test_process_pdf_ocr_success(self, mock_ocr_agent, mock_get_db, mock_document, mock_ocr_result):
        """Test successful PDF OCR processing"""
        # Setup mocks
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        mock_ocr_agent.detect_document_type.return_value = 'text_pdf'
        mock_ocr_agent.process_pdf.return_value = mock_ocr_result
        
        temp_file = self.create_temp_file('.pdf')
        
        try:
            mock_task = MagicMock()
            mock_task.request.id = "task-123"
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            # Execute task
            result = process_pdf_ocr(mock_task, "test-doc-id", temp_file)
            
            # Assertions
            assert result['success'] is True
            assert result['document_id'] == "test-doc-id"
            assert result['pdf_type'] == 'text_pdf'
            assert result['confidence'] == 0.95
            
            # Verify correct processing method was called
            mock_ocr_agent.process_pdf.assert_called_once_with(temp_file)
            
        finally:
            os.unlink(temp_file)
    
    @patch('app.agents.tasks.get_db')
    @patch('app.agents.tasks.ocr_agent')
    def test_process_pdf_ocr_scanned(self, mock_ocr_agent, mock_get_db, mock_document, mock_ocr_result):
        """Test scanned PDF OCR processing"""
        # Setup mocks
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        mock_ocr_agent.detect_document_type.return_value = 'scanned_pdf'
        mock_ocr_agent.process_scanned_pdf.return_value = mock_ocr_result
        
        temp_file = self.create_temp_file('.pdf')
        
        try:
            mock_task = MagicMock()
            mock_task.request.id = "task-123"
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            # Execute task
            result = process_pdf_ocr(mock_task, "test-doc-id", temp_file)
            
            # Assertions
            assert result['success'] is True
            assert result['pdf_type'] == 'scanned_pdf'
            
            # Verify correct processing method was called
            mock_ocr_agent.process_scanned_pdf.assert_called_once_with(temp_file)
            
        finally:
            os.unlink(temp_file)
    
    @patch('app.agents.tasks.get_db')
    @patch('app.agents.tasks.ocr_agent')
    def test_process_email_parsing_success(self, mock_ocr_agent, mock_get_db, mock_document, mock_email_result):
        """Test successful email parsing"""
        # Setup mocks
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        mock_ocr_agent.process_email.return_value = mock_email_result
        
        temp_file = self.create_temp_file('.msg')
        
        try:
            mock_task = MagicMock()
            mock_task.request.id = "task-123"
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            # Execute task
            result = process_email_parsing(mock_task, "test-doc-id", temp_file)
            
            # Assertions
            assert result['success'] is True
            assert result['document_id'] == "test-doc-id"
            assert result['subject'] == "Test Email"
            assert result['sender'] == "test@example.com"
            
            # Verify processing method was called
            mock_ocr_agent.process_email.assert_called_once_with(temp_file)
            
            # Verify database update
            assert 'email_parsing_result' in mock_document.metadata
            
        finally:
            os.unlink(temp_file)
    
    @patch('app.agents.tasks.get_db')
    @patch('app.agents.tasks.ocr_agent')
    def test_process_excel_parsing_success(self, mock_ocr_agent, mock_get_db, mock_document, mock_excel_result):
        """Test successful Excel parsing"""
        # Setup mocks
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        mock_ocr_agent.process_excel.return_value = mock_excel_result
        
        temp_file = self.create_temp_file('.xlsx')
        
        try:
            mock_task = MagicMock()
            mock_task.request.id = "task-123"
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            # Execute task
            result = process_excel_parsing(mock_task, "test-doc-id", temp_file)
            
            # Assertions
            assert result['success'] is True
            assert result['document_id'] == "test-doc-id"
            assert result['total_sheets'] == 1
            assert result['total_rows'] == 1
            
            # Verify processing method was called
            mock_ocr_agent.process_excel.assert_called_once_with(temp_file)
            
            # Verify database update
            assert 'excel_processing_result' in mock_document.metadata
            
        finally:
            os.unlink(temp_file)
    
    @patch('app.agents.tasks.get_db')
    @patch('app.agents.tasks.process_document_ocr')
    def test_batch_process_documents_success(self, mock_process_task, mock_get_db):
        """Test successful batch document processing"""
        # Setup mocks
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        
        # Mock documents
        doc1 = Mock(spec=Document)
        doc1.id = "doc-1"
        doc1.file_path = "/tmp/doc1.pdf"
        doc1.document_type = "pdf"
        
        doc2 = Mock(spec=Document)
        doc2.id = "doc-2"
        doc2.file_path = "/tmp/doc2.pdf"
        doc2.document_type = "pdf"
        
        mock_db.query.return_value.filter.return_value.first.side_effect = [doc1, doc2]
        
        # Mock task results
        mock_task1 = Mock()
        mock_task1.id = "task-1"
        mock_task2 = Mock()
        mock_task2.id = "task-2"
        
        mock_process_task.delay.side_effect = [mock_task1, mock_task2]
        
        # Execute batch processing
        result = batch_process_documents(["doc-1", "doc-2"])
        
        # Assertions
        assert result['total_documents'] == 2
        assert len(result['results']) == 2
        
        # Check individual results
        results = result['results']
        assert results[0]['document_id'] == "doc-1"
        assert results[0]['task_id'] == "task-1"
        assert results[0]['status'] == 'queued'
        
        assert results[1]['document_id'] == "doc-2"
        assert results[1]['task_id'] == "task-2"
        assert results[1]['status'] == 'queued'
        
        # Verify tasks were queued
        assert mock_process_task.delay.call_count == 2
    
    @patch('app.agents.tasks.get_db')
    def test_batch_process_documents_not_found(self, mock_get_db):
        """Test batch processing with non-existent documents"""
        # Setup mocks
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        # Execute batch processing
        result = batch_process_documents(["nonexistent-doc"])
        
        # Assertions
        assert result['total_documents'] == 1
        assert len(result['results']) == 1
        assert result['results'][0]['document_id'] == "nonexistent-doc"
        assert result['results'][0]['status'] == 'not_found'
    
    @patch('app.agents.tasks.get_db')
    def test_batch_process_documents_error(self, mock_get_db):
        """Test batch processing with database error"""
        # Setup mocks to raise exception
        mock_get_db.side_effect = Exception("Database connection error")
        
        # Execute batch processing
        result = batch_process_documents(["doc-1"])
        
        # Assertions
        assert result['total_documents'] == 1
        assert len(result['results']) == 1
        assert result['results'][0]['document_id'] == "doc-1"
        assert result['results'][0]['status'] == 'error'
        assert "Database connection error" in result['results'][0]['error']


class TestTaskIntegration:
    """Integration tests for OCR tasks"""
    
    @patch('app.agents.tasks.get_db')
    @patch('app.agents.tasks.ocr_agent')
    def test_email_content_database_storage(self, mock_ocr_agent, mock_get_db):
        """Test that email content is properly stored in database"""
        # Setup mocks
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        
        mock_document = Mock(spec=Document)
        mock_document.id = "test-doc-id"
        mock_document.metadata = {}
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        
        email_result = EmailContent(
            subject="Reinsurance Application",
            sender="broker@insurance.com",
            recipients=["underwriter@reinsurance.com"],
            body="Please review the attached application",
            attachments=[{"filename": "app.pdf", "size": 1024}],
            date=datetime(2024, 1, 15, 10, 30, 0),
            metadata={"importance": "high"}
        )
        
        mock_ocr_agent.process_email.return_value = email_result
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.msg')
        temp_file.close()
        
        try:
            mock_task = MagicMock()
            mock_task.request.id = "task-123"
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            # Execute task
            result = process_email_parsing(mock_task, "test-doc-id", temp_file.name)
            
            # Verify email data was stored correctly
            stored_metadata = mock_document.metadata['email_parsing_result']
            assert stored_metadata['subject'] == "Reinsurance Application"
            assert stored_metadata['sender'] == "broker@insurance.com"
            assert stored_metadata['recipients_count'] == 1
            assert stored_metadata['attachments_count'] == 1
            assert stored_metadata['date'] == "2024-01-15T10:30:00"
            assert stored_metadata['importance'] == "high"
            
        finally:
            os.unlink(temp_file.name)
    
    @patch('app.agents.tasks.get_db')
    @patch('app.agents.tasks.ocr_agent')
    def test_ocr_result_database_storage(self, mock_ocr_agent, mock_get_db):
        """Test that OCR results are properly stored in database"""
        # Setup mocks
        mock_db = MagicMock()
        mock_get_db.return_value = iter([mock_db])
        
        mock_document = Mock(spec=Document)
        mock_document.id = "test-doc-id"
        mock_document.metadata = {}
        mock_db.query.return_value.filter.return_value.first.return_value = mock_document
        
        ocr_result = OCRResult(
            text="Extracted document text content",
            confidence=0.92,
            regions=[],
            metadata={"page_count": 3, "extraction_method": "doctr_ocr"},
            processing_time=5.2,
            success=True
        )
        
        mock_ocr_agent.process_document.return_value = ocr_result
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.close()
        
        try:
            mock_task = MagicMock()
            mock_task.request.id = "task-123"
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            # Execute task
            result = process_document_ocr(mock_task, "test-doc-id", temp_file.name)
            
            # Verify OCR data was stored correctly
            stored_metadata = mock_document.metadata['ocr_result']
            assert stored_metadata['text'] == "Extracted document text content"
            assert stored_metadata['confidence'] == 0.92
            assert stored_metadata['processing_time'] == 5.2
            assert stored_metadata['success'] is True
            assert stored_metadata['page_count'] == 3
            assert stored_metadata['extraction_method'] == "doctr_ocr"
            
        finally:
            os.unlink(temp_file.name)


if __name__ == "__main__":
    pytest.main([__file__])