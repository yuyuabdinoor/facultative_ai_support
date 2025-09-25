"""
Document API endpoints
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import io

from ...core.database import get_db
from ...models.schemas import (
    DocumentResponse, DocumentUpdate, ValidationResult, ProcessingResult
)
from ...services.document_service import DocumentService
from ...services.storage import create_storage_manager
from ...services.validation import DocumentValidator
from ...core.config import settings

router = APIRouter()

# Initialize services
storage_manager = create_storage_manager("local", base_path=settings.upload_dir)
validator = DocumentValidator()
document_service = DocumentService(storage_manager, validator)


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    application_id: Optional[str] = Query(None, description="Application ID to associate with document"),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a document with validation and storage
    
    Supports:
    - PDF files (text-based and scanned)
    - Email files (.msg, .eml)
    - Excel files (.xlsx, .xls, .xlsm)
    
    The system will automatically:
    - Validate file type and size
    - Perform basic security checks
    - Extract metadata
    - Store the file securely
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    return await document_service.upload_document(db, file, application_id)


@router.post("/validate", response_model=ValidationResult)
async def validate_document(
    file: UploadFile = File(...),
):
    """
    Validate a document without uploading it
    
    Returns validation results including:
    - File type validation
    - Size validation
    - Security checks
    - Warnings and errors
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    return await document_service.validate_document_file(file)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get document metadata by ID"""
    document = await document_service.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Download document file content"""
    document = await document_service.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        file_content = await document_service.get_document_content(db, document_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")
    
    # Determine content type based on document type
    content_type_mapping = {
        "pdf": "application/pdf",
        "scanned_pdf": "application/pdf",
        "email": "application/vnd.ms-outlook",
        "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }
    
    content_type = content_type_mapping.get(document.document_type, "application/octet-stream")
    
    return StreamingResponse(
        io.BytesIO(file_content),
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename={document.filename}"}
    )


@router.get("/{document_id}/status")
async def get_document_status(
    document_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get document processing status and detailed information
    
    Returns:
    - Processing status
    - File existence status
    - Metadata
    - File size
    - Upload timestamp
    """
    return await document_service.get_document_status(db, document_id)


@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    update_data: DocumentUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update document metadata and processing status"""
    document = await document_service.update_document(db, document_id, update_data)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete document and its associated file"""
    success = await document_service.delete_document(db, document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted successfully"}


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    application_id: Optional[str] = Query(None, description="Filter by application ID"),
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of documents to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    List documents with optional filtering
    
    Parameters:
    - application_id: Filter documents by application
    - skip: Pagination offset
    - limit: Maximum number of results
    """
    return await document_service.list_documents(db, application_id, skip, limit)


@router.post("/batch-upload", response_model=List[DocumentResponse])
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    application_id: Optional[str] = Query(None, description="Application ID to associate with all documents"),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload multiple documents in a batch
    
    All files will be validated and uploaded. If any file fails validation,
    the entire batch will be rejected.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    # Validate all files first
    validation_errors = []
    for i, file in enumerate(files):
        if not file.filename:
            validation_errors.append(f"File {i+1}: Filename is required")
            continue
        
        validation_result = await document_service.validate_document_file(file)
        if not validation_result.is_valid:
            validation_errors.append(f"File {i+1} ({file.filename}): {', '.join(validation_result.errors)}")
    
    if validation_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Batch validation failed",
                "errors": validation_errors
            }
        )
    
    # Upload all files
    uploaded_documents = []
    for file in files:
        try:
            document = await document_service.upload_document(db, file, application_id)
            uploaded_documents.append(document)
        except Exception as e:
            # If any upload fails, we should ideally rollback previous uploads
            # For now, we'll return what was successfully uploaded
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload {file.filename}: {str(e)}"
            )
    
    return uploaded_documents


@router.get("/{document_id}/metadata")
async def get_document_metadata(
    document_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get detailed document metadata"""
    document = await document_service.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document.metadata or {}


@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
) -> ProcessingResult:
    """
    Mark document for reprocessing
    
    This endpoint marks a document as unprocessed so it can be
    picked up by the processing agents again.
    """
    update_data = DocumentUpdate(processed=False)
    document = await document_service.update_document(db, document_id, update_data)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return ProcessingResult(
        success=True,
        message="Document marked for reprocessing",
        data={"document_id": document_id, "processed": False}
    )


@router.post("/{document_id}/ocr")
async def trigger_ocr_processing(
    document_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Manually trigger OCR processing for a document
    
    This endpoint queues the document for OCR processing using Celery.
    The processing will happen asynchronously in the background.
    
    Returns:
    - Task ID for tracking
    - Processing status
    - Queue timestamp
    """
    return await document_service.trigger_ocr_processing(db, document_id)


@router.get("/{document_id}/ocr/status")
async def get_ocr_processing_status(
    document_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed OCR processing status for a document
    
    Returns comprehensive information about:
    - Processing status and progress
    - OCR results (if completed)
    - Email parsing results (if applicable)
    - Excel processing results (if applicable)
    - Error information (if failed)
    """
    return await document_service.get_processing_status(db, document_id)


@router.get("/{document_id}/ocr/text")
async def get_extracted_text(
    document_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get extracted text content from OCR processing
    
    Returns the full text content extracted from the document
    along with confidence scores and metadata.
    """
    document = await document_service.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    metadata = document.metadata or {}
    
    # Check for OCR results
    if 'ocr_result' in metadata:
        ocr_result = metadata['ocr_result']
        return {
            'document_id': document_id,
            'text': ocr_result.get('text', ''),
            'confidence': ocr_result.get('confidence', 0.0),
            'success': ocr_result.get('success', False),
            'processing_time': ocr_result.get('processing_time', 0.0),
            'extraction_method': ocr_result.get('extraction_method', 'unknown'),
            'regions_count': ocr_result.get('regions_count', 0)
        }
    
    # Check for email content
    elif 'email_parsing_result' in metadata:
        email_result = metadata['email_parsing_result']
        return {
            'document_id': document_id,
            'text': email_result.get('body', ''),
            'subject': email_result.get('subject', ''),
            'sender': email_result.get('sender', ''),
            'recipients': email_result.get('recipients', []),
            'date': email_result.get('date'),
            'extraction_method': 'email_parsing'
        }
    
    else:
        raise HTTPException(
            status_code=404, 
            detail="No extracted text available. Document may not be processed yet."
        )


@router.get("/{document_id}/ocr/regions")
async def get_text_regions(
    document_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detected text regions from OCR processing
    
    Returns detailed information about text regions detected in the document,
    including bounding boxes and confidence scores for each region.
    """
    document = await document_service.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    metadata = document.metadata or {}
    
    if 'ocr_result' not in metadata:
        raise HTTPException(
            status_code=404,
            detail="No OCR results available. Document may not be processed yet."
        )
    
    ocr_result = metadata['ocr_result']
    
    return {
        'document_id': document_id,
        'regions_count': ocr_result.get('regions_count', 0),
        'confidence': ocr_result.get('confidence', 0.0),
        'processing_method': ocr_result.get('extraction_method', 'unknown'),
        'page_count': ocr_result.get('page_count', 0),
        'success': ocr_result.get('success', False)
    }


@router.post("/batch-ocr")
async def batch_ocr_processing(
    document_ids: List[str],
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Trigger OCR processing for multiple documents
    
    Queues multiple documents for OCR processing in batch.
    Each document will be processed independently.
    
    Parameters:
    - document_ids: List of document IDs to process
    
    Returns:
    - Batch processing status
    - Individual task IDs
    - Queue information
    """
    if not document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")
    
    if len(document_ids) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 50 documents per batch")
    
    results = []
    
    for doc_id in document_ids:
        try:
            result = await document_service.trigger_ocr_processing(db, doc_id)
            results.append(result)
        except HTTPException as e:
            results.append({
                'document_id': doc_id,
                'status': 'error',
                'error': e.detail
            })
        except Exception as e:
            results.append({
                'document_id': doc_id,
                'status': 'error',
                'error': str(e)
            })
    
    successful_count = sum(1 for r in results if r.get('status') != 'error')
    
    return {
        'batch_id': f"batch_{len(document_ids)}_{hash(tuple(document_ids))}",
        'total_documents': len(document_ids),
        'successful_queued': successful_count,
        'failed_count': len(document_ids) - successful_count,
        'results': results
    }