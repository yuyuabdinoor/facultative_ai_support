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
            'application/vnd.ms-office',  # Alternative MIME type for .msg files
            'message/rfc822',
        ],
        DocumentType.EXCEL: [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            'application/vnd.ms-excel.sheet.macroEnabled.12',
        ],
        DocumentType.WORD: [
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
        ],
        DocumentType.POWERPOINT: [
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'application/vnd.ms-powerpoint',
        ],
        DocumentType.CSV: [
            'text/csv',
            'application/csv',
            'text/plain',  # CSV files are sometimes detected as plain text
        ],
        DocumentType.IMAGE: [
            'image/jpeg',
            'image/png',
            'image/tiff',
            'image/bmp',
            'image/gif',
        ]
    }
    
    # Maximum file sizes (in bytes)
    MAX_FILE_SIZES = {
        DocumentType.PDF: 50 * 1024 * 1024,  # 50MB
        DocumentType.SCANNED_PDF: 100 * 1024 * 1024,  # 100MB
        DocumentType.EMAIL: 25 * 1024 * 1024,  # 25MB
        DocumentType.EXCEL: 10 * 1024 * 1024,  # 10MB
        DocumentType.WORD: 15 * 1024 * 1024,  # 15MB
        DocumentType.POWERPOINT: 20 * 1024 * 1024,  # 20MB
        DocumentType.CSV: 5 * 1024 * 1024,  # 5MB
        DocumentType.IMAGE: 10 * 1024 * 1024,  # 10MB
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
            # ZIP-based formats (modern Office documents)
            # This is a generic detection - could be Excel, Word, or PowerPoint
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif file_content.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
            # Microsoft Office compound document (old Office formats, Outlook MSG)
            # Default to MSG format since that's most common for our use case
            return "application/vnd.ms-office"
        elif file_content.startswith(b'From ') or b'Message-ID:' in file_content[:1024]:
            return "message/rfc822"
        elif file_content.startswith(b'\xff\xd8\xff'):
            return "image/jpeg"
        elif file_content.startswith(b'\x89PNG\r\n\x1a\n'):
            return "image/png"
        elif file_content.startswith(b'BM'):
            return "image/bmp"
        elif file_content.startswith(b'II*\x00') or file_content.startswith(b'MM\x00*'):
            return "image/tiff"
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
            '.docx': DocumentType.WORD,
            '.doc': DocumentType.WORD,
            '.pptx': DocumentType.POWERPOINT,
            '.ppt': DocumentType.POWERPOINT,
            '.csv': DocumentType.CSV,
            '.png': DocumentType.IMAGE,
            '.jpg': DocumentType.IMAGE,
            '.jpeg': DocumentType.IMAGE,
            '.tiff': DocumentType.IMAGE,
            '.tif': DocumentType.IMAGE,
            '.bmp': DocumentType.IMAGE,
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


class AnalysisDocumentValidator:
    """Validator for Analysis document data with 23 critical fields"""
    
    # Critical fields that must be present for a complete analysis
    CRITICAL_FIELDS = [
        'reference_number', 'insured_name', 'cedant_reinsured', 'broker_name',
        'perils_covered', 'total_sums_insured', 'currency', 'period_of_insurance',
        'pml_percentage', 'share_offered_percentage'
    ]
    
    # All 23 fields in the Analysis document
    ALL_FIELDS = [
        'reference_number', 'date_received', 'insured_name', 'cedant_reinsured', 'broker_name',
        'perils_covered', 'geographical_limit', 'situation_of_risk', 'occupation_of_insured', 
        'main_activities', 'total_sums_insured', 'currency', 'excess_retention', 
        'premium_rates', 'period_of_insurance', 'pml_percentage', 'cat_exposure', 
        'reinsurance_deductions', 'claims_experience_3_years', 'share_offered_percentage',
        'surveyors_report', 'climate_change_risk', 'esg_risk_assessment'
    ]
    
    # Valid currency codes (ISO 4217)
    VALID_CURRENCIES = [
        'USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', 'CHF', 'SEK', 'NOK', 'DKK',
        'INR', 'CNY', 'BRL', 'MXN', 'ZAR', 'SGD', 'HKD', 'NZD', 'THB', 'MYR',
        'KRW', 'TWD', 'PLN', 'CZK', 'HUF', 'RON', 'BGN', 'HRK', 'RUB', 'TRY'
    ]
    
    def validate_financial_amount(self, amount: Optional[float], field_name: str) -> List[str]:
        """Validate financial amount fields"""
        errors = []
        
        if amount is not None:
            if amount < 0:
                errors.append(f"{field_name} cannot be negative")
            elif amount > 1e15:  # 1 quadrillion limit
                errors.append(f"{field_name} exceeds maximum allowed value")
        
        return errors
    
    def validate_percentage(self, percentage: Optional[float], field_name: str) -> List[str]:
        """Validate percentage fields"""
        errors = []
        
        if percentage is not None:
            if percentage < 0:
                errors.append(f"{field_name} cannot be negative")
            elif percentage > 100:
                errors.append(f"{field_name} cannot exceed 100%")
        
        return errors
    
    def validate_currency(self, currency: Optional[str]) -> List[str]:
        """Validate currency code"""
        errors = []
        
        if currency is not None:
            if len(currency) != 3:
                errors.append("Currency code must be exactly 3 characters")
            elif currency.upper() not in self.VALID_CURRENCIES:
                errors.append(f"Invalid currency code: {currency}")
        
        return errors
    
    def validate_date_field(self, date_value: Optional[datetime], field_name: str) -> List[str]:
        """Validate date fields"""
        errors = []
        
        if date_value is not None:
            # Check if date is reasonable (not too far in past or future)
            current_year = datetime.now().year
            if date_value.year < 1900:
                errors.append(f"{field_name} year cannot be before 1900")
            elif date_value.year > current_year + 10:
                errors.append(f"{field_name} year cannot be more than 10 years in the future")
        
        return errors
    
    def validate_string_length(self, value: Optional[str], field_name: str, max_length: int) -> List[str]:
        """Validate string field length"""
        errors = []
        
        if value is not None and len(value) > max_length:
            errors.append(f"{field_name} exceeds maximum length of {max_length} characters")
        
        return errors
    
    def validate_analysis_document_data(self, data: 'AnalysisDocumentData') -> ValidationResult:
        """Validate complete Analysis document data"""
        errors = []
        warnings = []
        
        # Import here to avoid circular imports
        from ..models.schemas import AnalysisDocumentData
        
        # Validate financial amounts
        financial_fields = [
            ('total_sums_insured', data.total_sums_insured),
            ('excess_retention', data.excess_retention),
            ('reinsurance_deductions', data.reinsurance_deductions)
        ]
        
        for field_name, value in financial_fields:
            errors.extend(self.validate_financial_amount(value, field_name))
        
        # Validate percentages
        percentage_fields = [
            ('premium_rates', data.premium_rates),
            ('pml_percentage', data.pml_percentage),
            ('share_offered_percentage', data.share_offered_percentage)
        ]
        
        for field_name, value in percentage_fields:
            errors.extend(self.validate_percentage(value, field_name))
        
        # Validate currency
        errors.extend(self.validate_currency(data.currency))
        
        # Validate dates
        errors.extend(self.validate_date_field(data.date_received, 'date_received'))
        
        # Validate string lengths
        string_fields = [
            ('reference_number', data.reference_number, 50),
            ('insured_name', data.insured_name, 200),
            ('cedant_reinsured', data.cedant_reinsured, 200),
            ('broker_name', data.broker_name, 200),
            ('perils_covered', data.perils_covered, 500),
            ('geographical_limit', data.geographical_limit, 300),
            ('situation_of_risk', data.situation_of_risk, 500),
            ('occupation_of_insured', data.occupation_of_insured, 200),
            ('main_activities', data.main_activities, 500),
            ('period_of_insurance', data.period_of_insurance, 100),
            ('cat_exposure', data.cat_exposure, 300),
            ('surveyors_report', data.surveyors_report, 500),
            ('climate_change_risk', data.climate_change_risk, 500),
            ('esg_risk_assessment', data.esg_risk_assessment, 500)
        ]
        
        for field_name, value, max_length in string_fields:
            errors.extend(self.validate_string_length(value, field_name, max_length))
        
        # Check critical field completeness
        missing_critical = []
        for field in self.CRITICAL_FIELDS:
            value = getattr(data, field, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_critical.append(field)
        
        if missing_critical:
            warnings.append(f"Missing critical fields: {', '.join(missing_critical)}")
        
        # Calculate completeness score
        total_fields = len(self.ALL_FIELDS)
        completed_fields = sum(1 for field in self.ALL_FIELDS 
                             if getattr(data, field, None) is not None)
        completeness_score = (completed_fields / total_fields) * 100
        
        if completeness_score < 50:
            warnings.append(f"Low data completeness: {completeness_score:.1f}%")
        elif completeness_score < 80:
            warnings.append(f"Moderate data completeness: {completeness_score:.1f}%")
        
        # Validate confidence score
        if data.confidence_score < 0 or data.confidence_score > 1:
            errors.append("Confidence score must be between 0 and 1")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def get_field_validation_rules(self) -> Dict[str, Dict[str, any]]:
        """Get validation rules for all Analysis document fields"""
        return {
            'reference_number': {'type': 'string', 'max_length': 50, 'required': True},
            'date_received': {'type': 'datetime', 'required': False},
            'insured_name': {'type': 'string', 'max_length': 200, 'required': True},
            'cedant_reinsured': {'type': 'string', 'max_length': 200, 'required': True},
            'broker_name': {'type': 'string', 'max_length': 200, 'required': True},
            'perils_covered': {'type': 'string', 'max_length': 500, 'required': True},
            'geographical_limit': {'type': 'string', 'max_length': 300, 'required': False},
            'situation_of_risk': {'type': 'string', 'max_length': 500, 'required': False},
            'occupation_of_insured': {'type': 'string', 'max_length': 200, 'required': False},
            'main_activities': {'type': 'string', 'max_length': 500, 'required': False},
            'total_sums_insured': {'type': 'decimal', 'min': 0, 'required': True},
            'currency': {'type': 'string', 'length': 3, 'required': True, 'valid_values': self.VALID_CURRENCIES},
            'excess_retention': {'type': 'decimal', 'min': 0, 'required': False},
            'premium_rates': {'type': 'percentage', 'min': 0, 'max': 100, 'required': False},
            'period_of_insurance': {'type': 'string', 'max_length': 100, 'required': True},
            'pml_percentage': {'type': 'percentage', 'min': 0, 'max': 100, 'required': True},
            'cat_exposure': {'type': 'string', 'max_length': 300, 'required': False},
            'reinsurance_deductions': {'type': 'decimal', 'min': 0, 'required': False},
            'claims_experience_3_years': {'type': 'text', 'required': False},
            'share_offered_percentage': {'type': 'percentage', 'min': 0, 'max': 100, 'required': True},
            'surveyors_report': {'type': 'string', 'max_length': 500, 'required': False},
            'climate_change_risk': {'type': 'string', 'max_length': 500, 'required': False},
            'esg_risk_assessment': {'type': 'string', 'max_length': 500, 'required': False}
        }