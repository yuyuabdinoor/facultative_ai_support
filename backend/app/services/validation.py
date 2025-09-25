"""
Document validation service with file type validation, virus scanning, and metadata extraction
"""
import os
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, BinaryIO, Tuple
from PIL import Image
import PyPDF2
import pandas as pd
from datetime import datetime
import hashlib

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    magic = None

from ..models.schemas import DocumentType, ValidationResult


class DocumentValidator:
    """Document validation service"""
    
    # Supported file types and their MIME types
    SUPPORTED_TYPES = {
        DocumentType.PDF: [
            'application/pdf',
        ],
        DocumentType.SCANNED_PDF: [
            'application/pdf',  # Will be determined by content analysis
        ],
        DocumentType.EMAIL: [
            'application/vnd.ms-outlook',
            'message/rfc822',
        ],
        DocumentType.EXCEL: [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            'application/vnd.ms-excel.sheet.macroEnabled.12',
        ]
    }
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES = {
        DocumentType.PDF: 50 * 1024 * 1024,  # 50MB
        DocumentType.SCANNED_PDF: 100 * 1024 * 1024,  # 100MB
        DocumentType.EMAIL: 25 * 1024 * 1024,  # 25MB
        DocumentType.EXCEL: 10 * 1024 * 1024,  # 10MB
    }
    
    def __init__(self):
        # Initialize magic for MIME type detection
        if MAGIC_AVAILABLE:
            self.magic = magic.Magic(mime=True)
        else:
            self.magic = None
    
    def detect_mime_type(self, file_content: bytes) -> str:
        """Detect MIME type from file content"""
        if self.magic:
            return self.magic.from_buffer(file_content)
        else:
            # Fallback to basic detection based on file signatures
            return self._detect_mime_type_fallback(file_content)
    
    def _detect_mime_type_fallback(self, file_content: bytes) -> str:
        """Fallback MIME type detection based on file signatures"""
        if not file_content:
            return "application/octet-stream"
        
        # Check common file signatures
        if file_content.startswith(b'%PDF'):
            return "application/pdf"
        elif file_content.startswith(b'PK\x03\x04') or file_content.startswith(b'PK\x05\x06') or file_content.startswith(b'PK\x07\x08'):
            # ZIP-based formats (Excel, etc.)
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif file_content.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
            # Microsoft Office compound document (old Excel, Outlook MSG)
            return "application/vnd.ms-excel"
        elif file_content.startswith(b'From ') or b'Message-ID:' in file_content[:1024]:
            return "message/rfc822"
        else:
            return "application/octet-stream"
    
    def validate_file_extension(self, filename: str) -> Tuple[bool, Optional[DocumentType]]:
        """Validate file extension and determine document type"""
        file_ext = Path(filename).suffix.lower()
        
        extension_mapping = {
            '.pdf': DocumentType.PDF,
            '.msg': DocumentType.EMAIL,
            '.eml': DocumentType.EMAIL,
            '.xlsx': DocumentType.EXCEL,
            '.xls': DocumentType.EXCEL,
            '.xlsm': DocumentType.EXCEL,
        }
        
        if file_ext in extension_mapping:
            return True, extension_mapping[file_ext]
        
        return False, None
    
    def validate_mime_type(self, mime_type: str, expected_type: DocumentType) -> bool:
        """Validate MIME type against expected document type"""
        return mime_type in self.SUPPORTED_TYPES.get(expected_type, [])
    
    def validate_file_size(self, file_size: int, document_type: DocumentType) -> bool:
        """Validate file size against limits"""
        max_size = self.MAX_FILE_SIZES.get(document_type, 10 * 1024 * 1024)
        return file_size <= max_size
    
    def is_pdf_scanned(self, file_content: bytes) -> bool:
        """Determine if PDF is scanned (image-based) or text-based"""
        try:
            from io import BytesIO
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            
            # Check first few pages for text content
            text_content = ""
            pages_to_check = min(3, len(pdf_reader.pages))
            
            for i in range(pages_to_check):
                page = pdf_reader.pages[i]
                text_content += page.extract_text()
            
            # If very little text is found, it's likely a scanned PDF
            return len(text_content.strip()) < 50
            
        except Exception:
            # If we can't read the PDF, assume it's scanned
            return True
    
    def extract_basic_metadata(self, filename: str, file_content: bytes, 
                             document_type: DocumentType) -> Dict[str, any]:
        """Extract basic metadata from document"""
        metadata = {
            'filename': filename,
            'file_size': len(file_content),
            'mime_type': self.detect_mime_type(file_content),
            'document_type': document_type.value,
            'upload_timestamp': datetime.utcnow().isoformat(),
            'file_hash': hashlib.sha256(file_content).hexdigest(),
        }
        
        # Type-specific metadata extraction
        if document_type in [DocumentType.PDF, DocumentType.SCANNED_PDF]:
            metadata.update(self._extract_pdf_metadata(file_content))
        elif document_type == DocumentType.EXCEL:
            metadata.update(self._extract_excel_metadata(file_content))
        elif document_type == DocumentType.EMAIL:
            metadata.update(self._extract_email_metadata(file_content))
        
        return metadata
    
    def _extract_pdf_metadata(self, file_content: bytes) -> Dict[str, any]:
        """Extract PDF-specific metadata"""
        try:
            from io import BytesIO
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            
            metadata = {
                'page_count': len(pdf_reader.pages),
                'is_scanned': self.is_pdf_scanned(file_content),
            }
            
            # Extract PDF metadata if available
            if pdf_reader.metadata:
                pdf_meta = pdf_reader.metadata
                metadata.update({
                    'title': pdf_meta.get('/Title', ''),
                    'author': pdf_meta.get('/Author', ''),
                    'subject': pdf_meta.get('/Subject', ''),
                    'creator': pdf_meta.get('/Creator', ''),
                    'producer': pdf_meta.get('/Producer', ''),
                    'creation_date': str(pdf_meta.get('/CreationDate', '')),
                    'modification_date': str(pdf_meta.get('/ModDate', '')),
                })
            
            return metadata
            
        except Exception as e:
            return {'error': f'Failed to extract PDF metadata: {str(e)}'}
    
    def _extract_excel_metadata(self, file_content: bytes) -> Dict[str, any]:
        """Extract Excel-specific metadata"""
        try:
            from io import BytesIO
            
            # Try to read Excel file
            excel_file = BytesIO(file_content)
            
            # Get sheet names
            xl_file = pd.ExcelFile(excel_file)
            sheet_names = xl_file.sheet_names
            
            metadata = {
                'sheet_count': len(sheet_names),
                'sheet_names': sheet_names,
            }
            
            # Get basic info about first sheet
            if sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_names[0], nrows=0)
                metadata.update({
                    'first_sheet_columns': list(df.columns),
                    'first_sheet_column_count': len(df.columns),
                })
            
            return metadata
            
        except Exception as e:
            return {'error': f'Failed to extract Excel metadata: {str(e)}'}
    
    def _extract_email_metadata(self, file_content: bytes) -> Dict[str, any]:
        """Extract email-specific metadata"""
        try:
            # Basic email metadata extraction
            # This is a placeholder - full implementation would use extract-msg
            metadata = {
                'email_format': 'msg' if file_content[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1' else 'eml',
            }
            
            return metadata
            
        except Exception as e:
            return {'error': f'Failed to extract email metadata: {str(e)}'}
    
    def perform_virus_scan(self, file_content: bytes) -> Tuple[bool, str]:
        """Perform basic virus scanning (placeholder implementation)"""
        # This is a basic implementation - in production, integrate with ClamAV or similar
        
        # Check for suspicious patterns
        suspicious_patterns = [
            b'<script',
            b'javascript:',
            b'vbscript:',
            b'data:text/html',
        ]
        
        file_lower = file_content.lower()
        for pattern in suspicious_patterns:
            if pattern in file_lower:
                return False, f"Suspicious pattern detected: {pattern.decode('utf-8', errors='ignore')}"
        
        # Check file size (extremely large files might be suspicious)
        if len(file_content) > 500 * 1024 * 1024:  # 500MB
            return False, "File size exceeds security limits"
        
        return True, "File passed basic security checks"
    
    async def validate_document(self, filename: str, file_content: bytes) -> ValidationResult:
        """Perform comprehensive document validation"""
        errors = []
        warnings = []
        
        # 1. Validate filename
        if not filename or len(filename.strip()) == 0:
            errors.append("Filename cannot be empty")
        
        # 2. Validate file extension
        ext_valid, detected_type = self.validate_file_extension(filename)
        if not ext_valid:
            errors.append(f"Unsupported file extension: {Path(filename).suffix}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # 3. Validate file size
        if len(file_content) == 0:
            errors.append("File is empty")
        elif not self.validate_file_size(len(file_content), detected_type):
            max_size = self.MAX_FILE_SIZES.get(detected_type, 0)
            errors.append(f"File size ({len(file_content)} bytes) exceeds limit ({max_size} bytes)")
        
        # 4. Validate MIME type
        mime_type = self.detect_mime_type(file_content)
        if not self.validate_mime_type(mime_type, detected_type):
            errors.append(f"MIME type {mime_type} doesn't match expected type for {detected_type.value}")
        
        # 5. Perform virus scan
        is_safe, scan_message = self.perform_virus_scan(file_content)
        if not is_safe:
            errors.append(f"Security check failed: {scan_message}")
        else:
            warnings.append(scan_message)
        
        # 6. Type-specific validation
        if detected_type == DocumentType.PDF:
            # Check if it's actually a scanned PDF
            if self.is_pdf_scanned(file_content):
                detected_type = DocumentType.SCANNED_PDF
                warnings.append("PDF appears to be scanned/image-based")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def get_document_type(self, filename: str, file_content: bytes) -> Optional[DocumentType]:
        """Determine the document type from filename and content"""
        ext_valid, detected_type = self.validate_file_extension(filename)
        
        if not ext_valid:
            return None
        
        # For PDFs, determine if scanned or regular
        if detected_type == DocumentType.PDF and self.is_pdf_scanned(file_content):
            return DocumentType.SCANNED_PDF
        
        return detected_type