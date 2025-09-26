"""
Document service for handling document operations
"""
import uuid
from typing import Optional, List, Dict, Any, BinaryIO
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from fastapi import UploadFile, HTTPException

from ..models.database import Document as DocumentModel
from ..models.schemas import (
    Document, DocumentCreate, DocumentUpdate, DocumentResponse,
    ValidationResult, ProcessingResult, DocumentType
)
from .storage import StorageManager
from .validation import DocumentValidator
from ..agents.tasks import process_document_ocr


class DocumentService:
    """Service for document operations"""
    
    def __init__(self, storage_manager: StorageManager, validator: DocumentValidator):
        self.storage_manager = storage_manager
        self.validator = validator
    
    async def upload_document(self, db: AsyncSession, file: UploadFile, 
                            application_id: Optional[str] = None) -> DocumentResponse:
        """Upload and validate a document"""
        
        # Read file content
        file_content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        # Validate document
        validation_result = await self.validator.validate_document(file.filename, file_content)
        
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Document validation failed",
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
            )
        
        # Determine document type
        document_type = self.validator.get_document_type(file.filename, file_content)
        if not document_type:
            raise HTTPException(
                status_code=400,
                detail="Unable to determine document type"
            )
        
        # Extract metadata
        metadata = self.validator.extract_basic_metadata(
            file.filename, file_content, document_type
        )
        
        # Add validation warnings to metadata
        if validation_result.warnings:
            metadata['validation_warnings'] = validation_result.warnings
        
        # Store file
        try:
            file_path = await self.storage_manager.store_document(
                file.file, file.filename, document_type.value
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store document: {str(e)}"
            )
        
        # Create document record
        document_id = str(uuid.uuid4())
        document_data = DocumentCreate(
            filename=file.filename,
            document_type=document_type,
            file_path=file_path,
            metadata=metadata
        )
        
        db_document = DocumentModel(
            id=document_id,
            application_id=application_id,
            filename=document_data.filename,
            document_type=document_data.document_type,
            file_path=document_data.file_path,
            document_metadata=document_data.metadata,
            processed=False,
            upload_timestamp=datetime.utcnow()
        )
        
        db.add(db_document)
        await db.commit()
        await db.refresh(db_document)
        
        # Queue OCR processing task
        try:
            task = process_document_ocr.delay(
                document_id=document_id,
                file_path=file_path,
                document_type=document_type.value
            )
            
            # Update metadata with task information
            db_document.document_metadata = {
                **(db_document.document_metadata or {}),
                'processing_task_id': task.id,
                'processing_status': 'queued',
                'processing_queued_at': datetime.utcnow().isoformat()
            }
            await db.commit()
            
        except Exception as e:
            # Log error but don't fail the upload
            print(f"Failed to queue OCR processing for document {document_id}: {str(e)}")
        
        return DocumentResponse(
            id=str(db_document.id),
            application_id=str(db_document.application_id) if db_document.application_id else None,
            filename=db_document.filename,
            document_type=db_document.document_type,
            file_path=db_document.file_path,
            processed=db_document.processed,
            upload_timestamp=db_document.upload_timestamp,
            metadata=db_document.document_metadata or {}
        )
    
    async def get_document(self, db: AsyncSession, document_id: str) -> Optional[DocumentResponse]:
        """Get document by ID"""
        result = await db.execute(
            select(DocumentModel).where(DocumentModel.id == document_id)
        )
        db_document = result.scalar_one_or_none()
        
        if not db_document:
            return None
        
        return DocumentResponse(
            id=str(db_document.id),
            application_id=str(db_document.application_id) if db_document.application_id else None,
            filename=db_document.filename,
            document_type=db_document.document_type,
            file_path=db_document.file_path,
            processed=db_document.processed,
            upload_timestamp=db_document.upload_timestamp,
            metadata=db_document.document_metadata or {}
        )
    
    async def get_document_content(self, db: AsyncSession, document_id: str) -> bytes:
        """Get document file content"""
        document = await self.get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        try:
            return await self.storage_manager.retrieve_document(document.file_path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Document file not found")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve document: {str(e)}"
            )
    
    async def delete_document(self, db: AsyncSession, document_id: str) -> bool:
        """Delete document and its file"""
        document = await self.get_document(db, document_id)
        if not document:
            return False
        
        # Delete file from storage
        try:
            await self.storage_manager.delete_document(document.file_path)
        except Exception:
            # Log error but continue with database deletion
            pass
        
        # Delete from database
        await db.execute(
            delete(DocumentModel).where(DocumentModel.id == document_id)
        )
        await db.commit()
        
        return True
    
    async def update_document(self, db: AsyncSession, document_id: str, 
                            update_data: DocumentUpdate) -> Optional[DocumentResponse]:
        """Update document metadata"""
        result = await db.execute(
            select(DocumentModel).where(DocumentModel.id == document_id)
        )
        db_document = result.scalar_one_or_none()
        
        if not db_document:
            return None
        
        # Update fields
        if update_data.processed is not None:
            db_document.processed = update_data.processed
        
        if update_data.metadata is not None:
            # Merge metadata
            current_metadata = db_document.metadata or {}
            current_metadata.update(update_data.metadata)
            db_document.metadata = current_metadata
        
        await db.commit()
        await db.refresh(db_document)
        
        return DocumentResponse(
            id=str(db_document.id),
            application_id=str(db_document.application_id) if db_document.application_id else None,
            filename=db_document.filename,
            document_type=db_document.document_type,
            file_path=db_document.file_path,
            processed=db_document.processed,
            upload_timestamp=db_document.upload_timestamp,
            metadata=db_document.document_metadata or {}
        )
    
    async def list_documents(self, db: AsyncSession, application_id: Optional[str] = None,
                           skip: int = 0, limit: int = 100) -> List[DocumentResponse]:
        """List documents with optional filtering"""
        query = select(DocumentModel)
        
        if application_id:
            query = query.where(DocumentModel.application_id == application_id)
        
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        documents = result.scalars().all()
        
        return [
            DocumentResponse(
                id=str(doc.id),
                application_id=str(doc.application_id) if doc.application_id else None,
                filename=doc.filename,
                document_type=doc.document_type,
                file_path=doc.file_path,
                processed=doc.processed,
                upload_timestamp=doc.upload_timestamp,
                metadata=doc.document_metadata or {}
            )
            for doc in documents
        ]
    
    async def get_document_status(self, db: AsyncSession, document_id: str) -> Dict[str, Any]:
        """Get document processing status and metadata"""
        document = await self.get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if file still exists in storage
        file_exists = await self.storage_manager.document_exists(document.file_path)
        
        status = {
            "document_id": document.id,
            "filename": document.filename,
            "document_type": document.document_type,
            "processed": document.processed,
            "upload_timestamp": document.upload_timestamp,
            "file_exists": file_exists,
            "metadata": document.metadata
        }
        
        if file_exists:
            try:
                file_size = await self.storage_manager.get_document_size(document.file_path)
                status["file_size"] = file_size
            except Exception:
                status["file_size"] = None
        
        return status
    
    async def validate_document_file(self, file: UploadFile) -> ValidationResult:
        """Validate document without uploading"""
        file_content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        return await self.validator.validate_document(file.filename, file_content)
    
    async def trigger_ocr_processing(self, db: AsyncSession, document_id: str) -> Dict[str, Any]:
        """Manually trigger OCR processing for a document"""
        document = await self.get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if file exists
        file_exists = await self.storage_manager.document_exists(document.file_path)
        if not file_exists:
            raise HTTPException(status_code=404, detail="Document file not found")
        
        try:
            # Queue OCR processing task
            # Normalize document_type to string in case it's already a string from DB
            doc_type_value = (
                document.document_type.value
                if hasattr(document.document_type, 'value') else document.document_type
            )
            task = process_document_ocr.delay(
                document_id=document_id,
                file_path=document.file_path,
                document_type=doc_type_value
            )
            
            # Update document metadata
            update_data = DocumentUpdate(
                metadata={
                    'processing_task_id': task.id,
                    'processing_status': 'queued',
                    'processing_queued_at': datetime.utcnow().isoformat(),
                    'manual_trigger': True
                }
            )
            
            await self.update_document(db, document_id, update_data)
            
            return {
                'document_id': document_id,
                'task_id': task.id,
                'status': 'queued',
                'message': 'OCR processing queued successfully'
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to queue OCR processing: {str(e)}"
            )
    
    async def get_processing_status(self, db: AsyncSession, document_id: str) -> Dict[str, Any]:
        """Get detailed processing status for a document"""
        document = await self.get_document(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        metadata = document.metadata or {}
        
        status_info = {
            'document_id': document_id,
            'processed': document.processed,
            'processing_status': metadata.get('processing_status', 'unknown'),
            'task_id': metadata.get('processing_task_id'),
            'queued_at': metadata.get('processing_queued_at'),
            'manual_trigger': metadata.get('manual_trigger', False)
        }
        
        # Add OCR results if available
        if 'ocr_result' in metadata:
            ocr_result = metadata['ocr_result']
            status_info['ocr_result'] = {
                'success': ocr_result.get('success', False),
                'confidence': ocr_result.get('confidence', 0.0),
                'text_length': len(ocr_result.get('text', '')),
                'regions_count': ocr_result.get('regions_count', 0),
                'processing_time': ocr_result.get('processing_time', 0.0),
                'error_message': ocr_result.get('error_message')
            }
        
        # Add email parsing results if available
        if 'email_parsing_result' in metadata:
            email_result = metadata['email_parsing_result']
            status_info['email_result'] = {
                'subject': email_result.get('subject', ''),
                'sender': email_result.get('sender', ''),
                'recipients_count': email_result.get('recipients_count', 0),
                'body_length': email_result.get('body_length', 0),
                'attachments_count': email_result.get('attachments_count', 0),
                'date': email_result.get('date')
            }
        
        # Add Excel processing results if available
        if 'excel_processing_result' in metadata:
            excel_result = metadata['excel_processing_result']
            status_info['excel_result'] = {
                'total_sheets': excel_result.get('total_sheets', 0),
                'total_rows': excel_result.get('total_rows', 0),
                'sheet_names': excel_result.get('sheet_names', [])
            }
        
        # Add processing errors if any
        if 'processing_error' in metadata:
            error_info = metadata['processing_error']
            status_info['error'] = {
                'message': error_info.get('error_message', ''),
                'task_id': error_info.get('task_id', ''),
                'retry_count': error_info.get('retry_count', 0)
            }
        
        return status_info