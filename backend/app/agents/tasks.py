"""
Celery tasks for OCR processing and document analysis

This module contains asynchronous tasks for processing various document types
in the facultative reinsurance system.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from celery import Celery
    from app.celery import celery_app
    CELERY_AVAILABLE = True
except ImportError:
    # Celery not available, define dummy objects
    Celery = None
    celery_app = None
    CELERY_AVAILABLE = False

# Create a decorator that works with or without Celery
def celery_task(*args, **kwargs):
    """Decorator that works whether Celery is available or not"""
    def decorator(func):
        if CELERY_AVAILABLE and celery_app is not None:
            return celery_app.task(*args, **kwargs)(func)
        else:
            # Return the function as-is if Celery is not available
            return func
    return decorator
from app.agents.ocr_agent import ocr_agent, OCRResult, EmailContent, ExcelData
from app.models.database import Document
from app.core.database import get_db
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _celery_task_decorator(func):
    """Conditional celery task decorator"""
    if CELERY_AVAILABLE and celery_app:
        return celery_app.task(bind=True, max_retries=3)(func)
    else:
        return func

@_celery_task_decorator
def process_document_ocr(self, document_id: str, file_path: str, document_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Asynchronous task to process document OCR
    
    Args:
        document_id: UUID of the document in database
        file_path: Path to the document file
        document_type: Optional document type hint
        
    Returns:
        Dictionary with processing results
    """
    try:
        logger.info(f"Starting OCR processing for document {document_id}")
        
        # Verify file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")
        
        # Process the document
        result = ocr_agent.process_document(file_path, document_type)
        
        # Update database with results
        db = next(get_db())
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                # Store processing results in metadata
                if isinstance(result, OCRResult):
                    document.metadata = {
                        **document.metadata,
                        'ocr_result': {
                            'text': result.text,
                            'confidence': result.confidence,
                            'regions_count': len(result.regions),
                            'processing_time': result.processing_time,
                            'success': result.success,
                            'error_message': result.error_message,
                            **result.metadata
                        }
                    }
                elif isinstance(result, EmailContent):
                    document.metadata = {
                        **document.metadata,
                        'email_content': {
                            'subject': result.subject,
                            'sender': result.sender,
                            'recipients_count': len(result.recipients),
                            'body_length': len(result.body),
                            'attachments_count': len(result.attachments),
                            'date': result.date.isoformat() if result.date else None,
                            **result.metadata
                        }
                    }
                elif isinstance(result, ExcelData):
                    document.metadata = {
                        **document.metadata,
                        'excel_data': {
                            'total_sheets': result.total_sheets,
                            'total_rows': result.total_rows,
                            'sheet_names': list(result.sheets.keys()),
                            **result.metadata
                        }
                    }
                
                document.processed = True
                db.commit()
                
        finally:
            db.close()
        
        logger.info(f"Successfully processed document {document_id}")
        
        return {
            'document_id': document_id,
            'success': True,
            'result_type': type(result).__name__,
            'processing_completed': True
        }
        
    except Exception as exc:
        logger.error(f"Error processing document {document_id}: {str(exc)}")
        
        # Update database with error status
        try:
            db = next(get_db())
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.metadata = {
                    **document.metadata,
                    'processing_error': {
                        'error_message': str(exc),
                        'task_id': self.request.id,
                        'retry_count': self.request.retries
                    }
                }
                db.commit()
            db.close()
        except Exception as db_error:
            logger.error(f"Failed to update database with error status: {str(db_error)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying OCR processing for document {document_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries))  # Exponential backoff
        
        return {
            'document_id': document_id,
            'success': False,
            'error': str(exc),
            'processing_completed': False
        }


@celery_task(bind=True, max_retries=3)
def process_pdf_ocr(self, document_id: str, file_path: str) -> Dict[str, Any]:
    """
    Specific task for PDF OCR processing
    
    Args:
        document_id: UUID of the document
        file_path: Path to PDF file
        
    Returns:
        Processing results dictionary
    """
    try:
        logger.info(f"Starting PDF OCR processing for document {document_id}")
        
        # Detect PDF type and process accordingly
        pdf_type = ocr_agent.detect_document_type(file_path)
        
        if pdf_type == 'text_pdf':
            result = ocr_agent.process_pdf(file_path)
        else:
            result = ocr_agent.process_scanned_pdf(file_path)
        
        # Update database
        db = next(get_db())
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.metadata = {
                    **document.metadata,
                    'pdf_ocr_result': {
                        'text': result.text,
                        'confidence': result.confidence,
                        'regions_count': len(result.regions),
                        'processing_time': result.processing_time,
                        'success': result.success,
                        'pdf_type': pdf_type,
                        **result.metadata
                    }
                }
                document.processed = True
                db.commit()
        finally:
            db.close()
        
        return {
            'document_id': document_id,
            'success': result.success,
            'pdf_type': pdf_type,
            'text_length': len(result.text),
            'confidence': result.confidence
        }
        
    except Exception as exc:
        logger.error(f"Error in PDF OCR processing for document {document_id}: {str(exc)}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        return {
            'document_id': document_id,
            'success': False,
            'error': str(exc)
        }


@celery_task(bind=True, max_retries=3)
def process_email_parsing(self, document_id: str, file_path: str) -> Dict[str, Any]:
    """
    Specific task for email parsing
    
    Args:
        document_id: UUID of the document
        file_path: Path to .msg file
        
    Returns:
        Processing results dictionary
    """
    try:
        logger.info(f"Starting email parsing for document {document_id}")
        
        result = ocr_agent.process_email(file_path)
        
        # Update database
        db = next(get_db())
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.metadata = {
                    **document.metadata,
                    'email_parsing_result': {
                        'subject': result.subject,
                        'sender': result.sender,
                        'recipients': result.recipients,
                        'body_length': len(result.body),
                        'attachments_count': len(result.attachments),
                        'date': result.date.isoformat() if result.date else None,
                        **result.metadata
                    }
                }
                document.processed = True
                db.commit()
        finally:
            db.close()
        
        return {
            'document_id': document_id,
            'success': True,
            'subject': result.subject,
            'sender': result.sender,
            'body_length': len(result.body)
        }
        
    except Exception as exc:
        logger.error(f"Error in email parsing for document {document_id}: {str(exc)}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        return {
            'document_id': document_id,
            'success': False,
            'error': str(exc)
        }


@celery_task(bind=True, max_retries=3)
def process_excel_parsing(self, document_id: str, file_path: str) -> Dict[str, Any]:
    """
    Specific task for Excel file processing
    
    Args:
        document_id: UUID of the document
        file_path: Path to Excel file
        
    Returns:
        Processing results dictionary
    """
    try:
        logger.info(f"Starting Excel processing for document {document_id}")
        
        result = ocr_agent.process_excel(file_path)
        
        # Update database
        db = next(get_db())
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.metadata = {
                    **document.metadata,
                    'excel_processing_result': {
                        'total_sheets': result.total_sheets,
                        'total_rows': result.total_rows,
                        'sheet_names': list(result.sheets.keys()),
                        **result.metadata
                    }
                }
                document.processed = True
                db.commit()
        finally:
            db.close()
        
        return {
            'document_id': document_id,
            'success': True,
            'total_sheets': result.total_sheets,
            'total_rows': result.total_rows
        }
        
    except Exception as exc:
        logger.error(f"Error in Excel processing for document {document_id}: {str(exc)}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        return {
            'document_id': document_id,
            'success': False,
            'error': str(exc)
        }


@celery_task
def batch_process_documents(document_ids: list) -> Dict[str, Any]:
    """
    Process multiple documents in batch
    
    Args:
        document_ids: List of document IDs to process
        
    Returns:
        Batch processing results
    """
    results = []
    
    for doc_id in document_ids:
        try:
            # Get document info from database
            db = next(get_db())
            document = db.query(Document).filter(Document.id == doc_id).first()
            db.close()
            
            if document:
                # Queue individual processing task
                task = process_document_ocr.delay(doc_id, document.file_path, document.document_type)
                results.append({
                    'document_id': doc_id,
                    'task_id': task.id,
                    'status': 'queued'
                })
            else:
                results.append({
                    'document_id': doc_id,
                    'status': 'not_found'
                })
                
        except Exception as e:
            results.append({
                'document_id': doc_id,
                'status': 'error',
                'error': str(e)
            })
    
    return {
        'batch_id': f"batch_{len(document_ids)}_{hash(tuple(document_ids))}",
        'total_documents': len(document_ids),
        'results': results
    }
