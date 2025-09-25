"""
OCR Processing Agent for Facultative Reinsurance System

This module provides OCR processing capabilities for various document types:
- PDF text extraction
- Scanned PDF OCR using DOCTR
- Email (.msg) parsing
- Excel file processing
- Text region detection and extraction
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import io
import base64

# Document processing imports
import pandas as pd
import numpy as np
from PIL import Image
import fitz  # PyMuPDF for better PDF handling
import pdfplumber
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import extract_msg
from openpyxl import load_workbook

# Pydantic models for structured responses
from pydantic import BaseModel, Field
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class TextRegion(BaseModel):
    """Represents a detected text region in a document"""
    text: str
    confidence: float
    bbox: List[float] = Field(description="Bounding box coordinates [x1, y1, x2, y2]")
    page_number: Optional[int] = None


class OCRResult(BaseModel):
    """Result of OCR processing"""
    text: str
    confidence: float
    regions: List[TextRegion] = []
    metadata: Dict[str, Any] = {}
    processing_time: float
    success: bool = True
    error_message: Optional[str] = None


class EmailContent(BaseModel):
    """Parsed email content structure"""
    subject: str
    sender: str
    recipients: List[str] = []
    body: str
    attachments: List[Dict[str, Any]] = []
    date: Optional[datetime] = None
    metadata: Dict[str, Any] = {}


class ExcelData(BaseModel):
    """Excel file processing result"""
    sheets: Dict[str, List[Dict[str, Any]]] = {}
    metadata: Dict[str, Any] = {}
    total_rows: int = 0
    total_sheets: int = 0


class OCRProcessingAgent:
    """
    OCR Processing Agent that handles various document types for reinsurance applications
    """
    
    def __init__(self):
        """Initialize the OCR agent with required models and configurations"""
        self.ocr_model = None
        self._initialize_ocr_model()
        
    def _initialize_ocr_model(self):
        """Initialize DOCTR OCR model"""
        try:
            # Initialize DOCTR OCR predictor
            self.ocr_model = ocr_predictor(pretrained=True)
            logger.info("DOCTR OCR model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR model: {str(e)}")
            self.ocr_model = None
    
    def process_pdf(self, file_path: str) -> OCRResult:
        """
        Extract text from PDF files using pdfplumber for text-based PDFs
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            OCRResult with extracted text and metadata
        """
        start_time = datetime.now()
        
        try:
            text_content = []
            regions = []
            total_confidence = 0
            page_count = 0
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                        
                        # Create text regions for each line
                        lines = page_text.split('\n')
                        for i, line in enumerate(lines):
                            if line.strip():
                                regions.append(TextRegion(
                                    text=line.strip(),
                                    confidence=0.95,  # High confidence for text-based PDFs
                                    bbox=[0, i * 20, page.width, (i + 1) * 20],
                                    page_number=page_num + 1
                                ))
                        
                        total_confidence += 0.95
                        page_count += 1
            
            full_text = '\n'.join(text_content)
            avg_confidence = total_confidence / page_count if page_count > 0 else 0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                regions=regions,
                metadata={
                    'file_type': 'pdf',
                    'page_count': page_count,
                    'extraction_method': 'pdfplumber'
                },
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def process_scanned_pdf(self, file_path: str) -> OCRResult:
        """
        Process scanned PDF using DOCTR OCR
        
        Args:
            file_path: Path to the scanned PDF file
            
        Returns:
            OCRResult with OCR-extracted text and regions
        """
        start_time = datetime.now()
        
        try:
            if not self.ocr_model:
                raise Exception("OCR model not initialized")
            
            # Load document with DOCTR
            doc = DocumentFile.from_pdf(file_path)
            
            # Perform OCR
            result = self.ocr_model(doc)
            
            # Extract text and regions
            text_content = []
            regions = []
            total_confidence = 0
            word_count = 0
            
            for page_idx, page in enumerate(result.pages):
                page_text = []
                
                for block in page.blocks:
                    for line in block.lines:
                        line_text = []
                        for word in line.words:
                            word_text = word.value
                            word_confidence = word.confidence
                            
                            line_text.append(word_text)
                            total_confidence += word_confidence
                            word_count += 1
                            
                            # Create text region for each word
                            bbox = word.geometry
                            regions.append(TextRegion(
                                text=word_text,
                                confidence=word_confidence,
                                bbox=[bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]],
                                page_number=page_idx + 1
                            ))
                        
                        if line_text:
                            page_text.append(' '.join(line_text))
                
                if page_text:
                    text_content.append('\n'.join(page_text))
            
            full_text = '\n\n'.join(text_content)
            avg_confidence = total_confidence / word_count if word_count > 0 else 0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                regions=regions,
                metadata={
                    'file_type': 'scanned_pdf',
                    'page_count': len(result.pages),
                    'word_count': word_count,
                    'extraction_method': 'doctr_ocr'
                },
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error processing scanned PDF {file_path}: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def process_email(self, file_path: str) -> EmailContent:
        """
        Parse .msg email files using extract_msg
        
        Args:
            file_path: Path to the .msg file
            
        Returns:
            EmailContent with parsed email data
        """
        try:
            msg = extract_msg.Message(file_path)
            
            # Extract basic email information
            subject = msg.subject or ""
            sender = msg.sender or ""
            body = msg.body or ""
            date = msg.date
            
            # Extract recipients
            recipients = []
            if msg.to:
                recipients.extend([r.strip() for r in msg.to.split(';') if r.strip()])
            if msg.cc:
                recipients.extend([r.strip() for r in msg.cc.split(';') if r.strip()])
            
            # Extract attachments information
            attachments = []
            for attachment in msg.attachments:
                att_info = {
                    'filename': getattr(attachment, 'longFilename', '') or getattr(attachment, 'shortFilename', ''),
                    'size': getattr(attachment, 'size', 0),
                    'type': getattr(attachment, 'type', 'unknown')
                }
                attachments.append(att_info)
            
            # Metadata
            metadata = {
                'message_class': getattr(msg, 'messageClass', ''),
                'importance': getattr(msg, 'importance', ''),
                'sensitivity': getattr(msg, 'sensitivity', ''),
                'attachment_count': len(attachments)
            }
            
            msg.close()
            
            return EmailContent(
                subject=subject,
                sender=sender,
                recipients=recipients,
                body=body,
                attachments=attachments,
                date=date,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error processing email {file_path}: {str(e)}")
            return EmailContent(
                subject="",
                sender="",
                body="",
                metadata={'error': str(e)}
            )
    
    def process_excel(self, file_path: str) -> ExcelData:
        """
        Process Excel files using pandas and openpyxl
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            ExcelData with processed spreadsheet data
        """
        try:
            # Load workbook to get sheet names and metadata
            workbook = load_workbook(file_path, read_only=True)
            sheet_names = workbook.sheetnames
            
            sheets_data = {}
            total_rows = 0
            
            # Process each sheet
            for sheet_name in sheet_names:
                try:
                    # Read sheet data with pandas
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Convert to list of dictionaries
                    sheet_data = df.to_dict('records')
                    sheets_data[sheet_name] = sheet_data
                    total_rows += len(sheet_data)
                    
                except Exception as sheet_error:
                    logger.warning(f"Error processing sheet {sheet_name}: {str(sheet_error)}")
                    sheets_data[sheet_name] = []
            
            workbook.close()
            
            # Metadata
            metadata = {
                'file_format': Path(file_path).suffix.lower(),
                'sheet_names': sheet_names,
                'processing_method': 'pandas_openpyxl'
            }
            
            return ExcelData(
                sheets=sheets_data,
                metadata=metadata,
                total_rows=total_rows,
                total_sheets=len(sheet_names)
            )
            
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {str(e)}")
            return ExcelData(
                metadata={'error': str(e)}
            )
    
    def extract_text_regions(self, image: Union[np.ndarray, str]) -> List[TextRegion]:
        """
        Extract text regions from an image using DOCTR
        
        Args:
            image: Image as numpy array or path to image file
            
        Returns:
            List of TextRegion objects with detected text
        """
        try:
            if not self.ocr_model:
                raise Exception("OCR model not initialized")
            
            # Handle different input types
            if isinstance(image, str):
                # Image file path
                doc = DocumentFile.from_images(image)
            elif isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(image)
                doc = DocumentFile.from_images([pil_image])
            else:
                raise ValueError("Image must be a file path or numpy array")
            
            # Perform OCR
            result = self.ocr_model(doc)
            
            regions = []
            
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            bbox = word.geometry
                            regions.append(TextRegion(
                                text=word.value,
                                confidence=word.confidence,
                                bbox=[bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
                            ))
            
            return regions
            
        except Exception as e:
            logger.error(f"Error extracting text regions: {str(e)}")
            return []
    
    def detect_document_type(self, file_path: str) -> str:
        """
        Detect if a PDF is text-based or scanned
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            'text_pdf' or 'scanned_pdf'
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                # Check first few pages for text content
                text_found = False
                pages_to_check = min(3, len(pdf.pages))
                
                for i in range(pages_to_check):
                    page_text = pdf.pages[i].extract_text()
                    if page_text and len(page_text.strip()) > 50:
                        text_found = True
                        break
                
                return 'text_pdf' if text_found else 'scanned_pdf'
                
        except Exception as e:
            logger.error(f"Error detecting document type for {file_path}: {str(e)}")
            return 'scanned_pdf'  # Default to OCR processing
    
    def process_document(self, file_path: str, document_type: Optional[str] = None) -> Union[OCRResult, EmailContent, ExcelData]:
        """
        Process any supported document type
        
        Args:
            file_path: Path to the document
            document_type: Optional document type hint
            
        Returns:
            Appropriate result object based on document type
        """
        file_extension = Path(file_path).suffix.lower()
        
        # Auto-detect document type if not provided
        if not document_type:
            if file_extension == '.pdf':
                document_type = self.detect_document_type(file_path)
            elif file_extension == '.msg':
                document_type = 'email'
            elif file_extension in ['.xlsx', '.xls']:
                document_type = 'excel'
            else:
                document_type = 'scanned_pdf'  # Default to OCR
        
        # Process based on type
        if document_type == 'text_pdf':
            return self.process_pdf(file_path)
        elif document_type == 'scanned_pdf':
            return self.process_scanned_pdf(file_path)
        elif document_type == 'email':
            return self.process_email(file_path)
        elif document_type == 'excel':
            return self.process_excel(file_path)
        else:
            raise ValueError(f"Unsupported document type: {document_type}")


# Global OCR agent instance
ocr_agent = OCRProcessingAgent()