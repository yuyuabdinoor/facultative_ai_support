"""
RI Slip Document Identification and Prioritization Agent

This module provides specialized functionality for identifying and prioritizing
RI (Reinsurance) Slip documents within email attachments and document collections.

Key Features:
- RI Slip detection logic for PDF and DOCX files (primary formats)
- Support for Excel, PowerPoint, and image formats as secondary options
- Attachment processing priority (RI Slips first, then supporting docs)
- Document type classification specifically for RI Slips vs other attachments
- RI Slip validation and quality checks before processing

Requirements: 2.1, 2.3
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

# Pydantic models
from pydantic import BaseModel, Field

# Import OCR agent components
from .ocr_agent import (
    OCRResult, EmailContent, ExcelData, WordDocumentData, 
    PowerPointData, AttachmentData, TextRegion
)

# Configure logging
logger = logging.getLogger(__name__)


class DocumentFormat(str, Enum):
    """Document format enumeration for RI Slip identification"""
    PDF = "pdf"
    DOCX = "docx"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    IMAGE = "image"
    UNKNOWN = "unknown"


class RISlipType(str, Enum):
    """RI Slip type classification"""
    FACULTATIVE_SLIP = "facultative_slip"
    TREATY_SLIP = "treaty_slip"
    PLACEMENT_SLIP = "placement_slip"
    COVER_NOTE = "cover_note"
    BINDING_AUTHORITY = "binding_authority"
    LINE_SLIP = "line_slip"
    OPEN_COVER = "open_cover"
    UNKNOWN_RI_SLIP = "unknown_ri_slip"


class DocumentPriority(int, Enum):
    """Document processing priority levels"""
    CRITICAL_RI_SLIP = 1      # Primary RI Slips (PDF, DOCX)
    SECONDARY_RI_SLIP = 2     # Secondary RI Slips (Excel, PPT, Images)
    SUPPORTING_DOCUMENT = 3   # Supporting documents
    LOW_PRIORITY = 4          # Other attachments


class RISlipIdentificationResult(BaseModel):
    """Result of RI Slip identification process"""
    is_ri_slip: bool
    ri_slip_type: Optional[RISlipType] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    document_format: DocumentFormat
    priority: DocumentPriority
    identification_reasons: List[str] = Field(default_factory=list)
    quality_score: float = Field(ge=0.0, le=1.0, default=0.0)
    validation_issues: List[str] = Field(default_factory=list)
    extracted_indicators: Dict[str, Any] = Field(default_factory=dict)


class DocumentClassificationResult(BaseModel):
    """Result of document classification within email attachments"""
    attachment_filename: str
    classification_result: RISlipIdentificationResult
    processing_order: int
    should_process: bool = True
    skip_reason: Optional[str] = None


class EmailProcessingPlan(BaseModel):
    """Processing plan for email attachments with prioritization"""
    ri_slips: List[DocumentClassificationResult] = Field(default_factory=list)
    supporting_documents: List[DocumentClassificationResult] = Field(default_factory=list)
    low_priority_documents: List[DocumentClassificationResult] = Field(default_factory=list)
    total_documents: int = 0
    processing_order: List[str] = Field(default_factory=list)  # Ordered list of filenames


class RISlipIdentifier:
    """
    RI Slip identification and prioritization agent
    
    Identifies RI Slips from various document formats and prioritizes
    processing order for optimal data extraction efficiency.
    """
    
    def __init__(self):
        """Initialize the RI Slip identifier with patterns and keywords"""
        self._initialize_ri_slip_patterns()
        self._initialize_document_format_mapping()
        self._initialize_quality_indicators()
    
    def _initialize_ri_slip_patterns(self):
        """Initialize patterns and keywords for RI Slip identification"""
        
        # Primary RI Slip identification keywords (high confidence)
        self.primary_ri_slip_keywords = [
            # Direct RI Slip terminology
            'ri slip', 'reinsurance slip', 'facultative slip', 'fac slip',
            'treaty slip', 'placement slip', 'line slip', 'open cover',
            
            # Reinsurance-specific terms
            'facultative certificate', 'facultative reinsurance', 
            'reinsurance placement', 'risk placement', 'cover note',
            'binding authority', 'reinsurance contract', 'ri contract',
            
            # Document headers/titles
            'reinsurance slip', 'facultative placement', 'treaty placement',
            'reinsurance offer', 'facultative offer', 'ri offer'
        ]
        
        # Secondary RI Slip indicators (medium confidence)
        self.secondary_ri_slip_keywords = [
            # Business terms
            'cedant', 'reinsured', 'ceding company', 'original insurer',
            'reinsurer', 'accepting office', 'lead underwriter',
            'quota share', 'surplus', 'excess of loss', 'stop loss',
            
            # Coverage terms
            'sum insured', 'policy limit', 'attachment point', 'retention',
            'deductible', 'excess', 'coverage limit', 'indemnity limit',
            
            # Risk terms
            'perils covered', 'risks covered', 'coverage territory',
            'geographical limit', 'policy period', 'period of insurance'
        ]
        
        # Supporting document keywords (lower confidence for RI Slip classification)
        self.supporting_document_keywords = [
            # Survey and inspection reports
            'survey report', 'property survey', 'inspection report',
            'loss inspection', 'site inspection', 'building survey',
            'risk survey', 'engineering survey', 'marine survey',
            
            # Financial documents
            'financial statement', 'balance sheet', 'income statement',
            'audited accounts', 'management accounts', 'cash flow',
            
            # Certificates and valuations
            'certificate', 'valuation report', 'appraisal report',
            'property valuation', 'asset valuation', 'replacement cost',
            
            # Claims and loss history
            'loss history', 'claims history', 'loss experience',
            'claims experience', 'loss run', 'claims summary'
        ]
        
        # Regex patterns for RI Slip identification
        self.ri_slip_patterns = {
            # Reference number patterns
            'reference_pattern': re.compile(
                r'(?:ref|reference|policy|slip|fac|ri)[\s\-\#\:]*([A-Z0-9\-\/]+)',
                re.IGNORECASE
            ),
            
            # Reinsurance percentage patterns
            'percentage_pattern': re.compile(
                r'(?:share|percentage|%|quota|surplus)[\s\:]*(\d+(?:\.\d+)?%?)',
                re.IGNORECASE
            ),
            
            # Sum insured patterns
            'sum_insured_pattern': re.compile(
                r'(?:sum insured|limit|coverage|indemnity)[\s\:]*([A-Z]{3}[\s\$\€\£]*[\d,]+(?:\.\d{2})?)',
                re.IGNORECASE
            ),
            
            # Period patterns
            'period_pattern': re.compile(
                r'(?:period|from|effective)[\s\:]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
                re.IGNORECASE
            ),
            
            # Cedant/Reinsured patterns
            'cedant_pattern': re.compile(
                r'(?:cedant|reinsured|ceding company|original insurer)[\s\:]*([A-Z][a-zA-Z\s&\.,]+)',
                re.IGNORECASE
            )
        }
    
    def _initialize_document_format_mapping(self):
        """Initialize document format detection mapping"""
        self.format_extensions = {
            DocumentFormat.PDF: ['.pdf'],
            DocumentFormat.DOCX: ['.docx', '.doc'],
            DocumentFormat.EXCEL: ['.xlsx', '.xls', '.xlsm', '.csv'],
            DocumentFormat.POWERPOINT: ['.pptx', '.ppt'],
            DocumentFormat.IMAGE: ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif']
        }
        
        # Primary formats for RI Slips (highest priority)
        self.primary_formats = [DocumentFormat.PDF, DocumentFormat.DOCX]
        
        # Secondary formats for RI Slips (medium priority)
        self.secondary_formats = [DocumentFormat.EXCEL, DocumentFormat.POWERPOINT, DocumentFormat.IMAGE]
    
    def _initialize_quality_indicators(self):
        """Initialize quality indicators for RI Slip validation"""
        
        # Essential fields that should be present in a quality RI Slip
        self.essential_fields = [
            'reference_number', 'sum_insured', 'cedant', 'reinsurer',
            'period', 'coverage_type', 'percentage_share'
        ]
        
        # Quality scoring weights
        self.quality_weights = {
            'has_reference': 0.2,
            'has_financial_data': 0.25,
            'has_parties': 0.2,
            'has_coverage_details': 0.15,
            'has_period': 0.1,
            'document_structure': 0.1
        }
        
        # Minimum quality thresholds
        self.min_quality_score = 0.3  # Minimum score to consider as valid RI Slip
        self.good_quality_score = 0.7  # Score for high-quality RI Slips
    
    def detect_document_format(self, filename: str) -> DocumentFormat:
        """
        Detect document format from filename extension
        
        Args:
            filename: Name of the document file
            
        Returns:
            DocumentFormat enum value
        """
        file_ext = Path(filename).suffix.lower()
        
        for doc_format, extensions in self.format_extensions.items():
            if file_ext in extensions:
                return doc_format
        
        return DocumentFormat.UNKNOWN
    
    def extract_ri_slip_indicators(self, text_content: str) -> Dict[str, Any]:
        """
        Extract RI Slip indicators from text content
        
        Args:
            text_content: Text content to analyze
            
        Returns:
            Dictionary of extracted indicators
        """
        indicators = {
            'reference_numbers': [],
            'percentages': [],
            'sum_insured_amounts': [],
            'periods': [],
            'cedants': [],
            'primary_keywords_found': [],
            'secondary_keywords_found': [],
            'pattern_matches': {}
        }
        
        text_lower = text_content.lower()
        
        # Extract using regex patterns
        for pattern_name, pattern in self.ri_slip_patterns.items():
            matches = pattern.findall(text_content)
            if matches:
                indicators['pattern_matches'][pattern_name] = matches
                
                # Store specific extractions
                if pattern_name == 'reference_pattern':
                    indicators['reference_numbers'].extend(matches)
                elif pattern_name == 'percentage_pattern':
                    indicators['percentages'].extend(matches)
                elif pattern_name == 'sum_insured_pattern':
                    indicators['sum_insured_amounts'].extend(matches)
                elif pattern_name == 'period_pattern':
                    indicators['periods'].extend(matches)
                elif pattern_name == 'cedant_pattern':
                    indicators['cedants'].extend(matches)
        
        # Find primary keywords
        for keyword in self.primary_ri_slip_keywords:
            if keyword.lower() in text_lower:
                indicators['primary_keywords_found'].append(keyword)
        
        # Find secondary keywords
        for keyword in self.secondary_ri_slip_keywords:
            if keyword.lower() in text_lower:
                indicators['secondary_keywords_found'].append(keyword)
        
        return indicators
    
    def calculate_ri_slip_confidence(self, indicators: Dict[str, Any], 
                                   document_format: DocumentFormat) -> float:
        """
        Calculate confidence score for RI Slip identification
        
        Args:
            indicators: Extracted indicators from document
            document_format: Format of the document
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.0
        
        # Primary keywords (high weight)
        primary_keywords_score = min(len(indicators['primary_keywords_found']) * 0.3, 0.6)
        confidence += primary_keywords_score
        
        # Secondary keywords (medium weight)
        secondary_keywords_score = min(len(indicators['secondary_keywords_found']) * 0.05, 0.2)
        confidence += secondary_keywords_score
        
        # Pattern matches (medium weight)
        pattern_score = 0.0
        for pattern_name, matches in indicators['pattern_matches'].items():
            if matches:
                if pattern_name in ['reference_pattern', 'sum_insured_pattern']:
                    pattern_score += 0.1  # High value patterns
                else:
                    pattern_score += 0.05  # Other patterns
        
        confidence += min(pattern_score, 0.25)
        
        # Document format bonus (primary formats get higher confidence)
        if document_format in self.primary_formats:
            confidence += 0.1
        elif document_format in self.secondary_formats:
            confidence += 0.05
        
        # Ensure confidence is within bounds
        return min(confidence, 1.0)
    
    def classify_ri_slip_type(self, indicators: Dict[str, Any]) -> Optional[RISlipType]:
        """
        Classify the type of RI Slip based on indicators
        
        Args:
            indicators: Extracted indicators from document
            
        Returns:
            RISlipType enum value or None
        """
        primary_keywords = [kw.lower() for kw in indicators['primary_keywords_found']]
        
        # Classification based on keywords
        if any('facultative' in kw for kw in primary_keywords):
            return RISlipType.FACULTATIVE_SLIP
        elif any('treaty' in kw for kw in primary_keywords):
            return RISlipType.TREATY_SLIP
        elif any('placement' in kw for kw in primary_keywords):
            return RISlipType.PLACEMENT_SLIP
        elif any('cover note' in kw for kw in primary_keywords):
            return RISlipType.COVER_NOTE
        elif any('binding authority' in kw for kw in primary_keywords):
            return RISlipType.BINDING_AUTHORITY
        elif any('line slip' in kw for kw in primary_keywords):
            return RISlipType.LINE_SLIP
        elif any('open cover' in kw for kw in primary_keywords):
            return RISlipType.OPEN_COVER
        elif primary_keywords:  # Has RI keywords but can't classify specifically
            return RISlipType.UNKNOWN_RI_SLIP
        
        return None
    
    def calculate_quality_score(self, indicators: Dict[str, Any], 
                              text_content: str) -> Tuple[float, List[str]]:
        """
        Calculate quality score for RI Slip document
        
        Args:
            indicators: Extracted indicators from document
            text_content: Full text content for analysis
            
        Returns:
            Tuple of (quality_score, validation_issues)
        """
        quality_score = 0.0
        validation_issues = []
        
        # Check for reference number
        if indicators['reference_numbers']:
            quality_score += self.quality_weights['has_reference']
        else:
            validation_issues.append("No reference number found")
        
        # Check for financial data
        if indicators['sum_insured_amounts'] or indicators['percentages']:
            quality_score += self.quality_weights['has_financial_data']
        else:
            validation_issues.append("No financial data (sum insured/percentages) found")
        
        # Check for parties (cedant/reinsurer information)
        if indicators['cedants'] or any('reinsurer' in text_content.lower() for _ in [1]):
            quality_score += self.quality_weights['has_parties']
        else:
            validation_issues.append("No party information (cedant/reinsurer) found")
        
        # Check for coverage details
        coverage_keywords = ['coverage', 'perils', 'risks covered', 'policy']
        if any(keyword in text_content.lower() for keyword in coverage_keywords):
            quality_score += self.quality_weights['has_coverage_details']
        else:
            validation_issues.append("No coverage details found")
        
        # Check for period information
        if indicators['periods']:
            quality_score += self.quality_weights['has_period']
        else:
            validation_issues.append("No period/date information found")
        
        # Check document structure (length and organization)
        if len(text_content.split()) > 50:  # Minimum word count
            quality_score += self.quality_weights['document_structure']
        else:
            validation_issues.append("Document appears too short or poorly structured")
        
        return quality_score, validation_issues
    
    def identify_ri_slip(self, text_content: str, filename: str) -> RISlipIdentificationResult:
        """
        Identify if a document is an RI Slip and classify it
        
        Args:
            text_content: Text content of the document
            filename: Name of the document file
            
        Returns:
            RISlipIdentificationResult with identification details
        """
        # Detect document format
        document_format = self.detect_document_format(filename)
        
        # Extract indicators
        indicators = self.extract_ri_slip_indicators(text_content)
        
        # Calculate confidence score
        confidence_score = self.calculate_ri_slip_confidence(indicators, document_format)
        
        # Determine if it's an RI Slip (threshold-based)
        is_ri_slip = confidence_score >= 0.3  # Minimum threshold for RI Slip classification
        
        # Classify RI Slip type
        ri_slip_type = self.classify_ri_slip_type(indicators) if is_ri_slip else None
        
        # Calculate quality score
        quality_score, validation_issues = self.calculate_quality_score(indicators, text_content)
        
        # Determine priority
        if is_ri_slip:
            if document_format in self.primary_formats:
                priority = DocumentPriority.CRITICAL_RI_SLIP
            else:
                priority = DocumentPriority.SECONDARY_RI_SLIP
        else:
            # Check if it's a supporting document
            text_lower = text_content.lower()
            is_supporting = any(keyword in text_lower for keyword in self.supporting_document_keywords)
            priority = DocumentPriority.SUPPORTING_DOCUMENT if is_supporting else DocumentPriority.LOW_PRIORITY
        
        # Generate identification reasons
        identification_reasons = []
        if indicators['primary_keywords_found']:
            identification_reasons.append(f"Primary RI keywords found: {', '.join(indicators['primary_keywords_found'][:3])}")
        if indicators['pattern_matches']:
            identification_reasons.append(f"RI patterns detected: {len(indicators['pattern_matches'])} types")
        if document_format in self.primary_formats:
            identification_reasons.append(f"Primary document format: {document_format.value}")
        
        return RISlipIdentificationResult(
            is_ri_slip=is_ri_slip,
            ri_slip_type=ri_slip_type,
            confidence_score=confidence_score,
            document_format=document_format,
            priority=priority,
            identification_reasons=identification_reasons,
            quality_score=quality_score,
            validation_issues=validation_issues,
            extracted_indicators=indicators
        )
    
    def classify_email_attachments(self, email_content: EmailContent) -> EmailProcessingPlan:
        """
        Classify all attachments in an email and create processing plan
        
        Args:
            email_content: Parsed email with attachments
            
        Returns:
            EmailProcessingPlan with prioritized processing order
        """
        classifications = []
        
        # Process each attachment
        for attachment in email_content.attachments:
            # Extract text content from attachment
            text_content = self._extract_text_from_attachment(attachment)
            
            # Identify RI Slip
            identification_result = self.identify_ri_slip(text_content, attachment.filename)
            
            # Create classification result
            classification = DocumentClassificationResult(
                attachment_filename=attachment.filename,
                classification_result=identification_result,
                processing_order=0,  # Will be set later
                should_process=True
            )
            
            # Check if document should be skipped
            if identification_result.quality_score < self.min_quality_score and identification_result.is_ri_slip:
                classification.should_process = False
                classification.skip_reason = f"Quality score too low: {identification_result.quality_score:.2f}"
            
            classifications.append(classification)
        
        # Sort by priority and create processing plan
        processing_plan = self._create_processing_plan(classifications)
        
        return processing_plan
    
    def _extract_text_from_attachment(self, attachment: AttachmentData) -> str:
        """
        Extract text content from attachment for analysis
        
        Args:
            attachment: Attachment data with processed content
            
        Returns:
            Text content string
        """
        if not attachment.processed_content:
            return ""
        
        content = attachment.processed_content
        
        if isinstance(content, OCRResult):
            return content.text
        elif isinstance(content, WordDocumentData):
            return content.text
        elif isinstance(content, ExcelData):
            # Extract text from all sheets
            text_parts = []
            for sheet_name, sheet_data in content.sheets.items():
                for row in sheet_data:
                    text_parts.extend([str(value) for value in row.values() if value])
            return ' '.join(text_parts)
        elif isinstance(content, PowerPointData):
            return content.text
        else:
            return ""
    
    def _create_processing_plan(self, classifications: List[DocumentClassificationResult]) -> EmailProcessingPlan:
        """
        Create processing plan with prioritized order
        
        Args:
            classifications: List of document classifications
            
        Returns:
            EmailProcessingPlan with organized processing order
        """
        # Separate documents by category
        ri_slips = []
        supporting_documents = []
        low_priority_documents = []
        
        for classification in classifications:
            result = classification.classification_result
            
            if result.is_ri_slip:
                ri_slips.append(classification)
            elif result.priority == DocumentPriority.SUPPORTING_DOCUMENT:
                supporting_documents.append(classification)
            else:
                low_priority_documents.append(classification)
        
        # Sort RI Slips with stricter rules:
        # 1) Prefer FACULTATIVE_SLIP over PLACEMENT_SLIP by default
        # 2) If best PLACEMENT quality exceeds best FACULTATIVE by threshold, allow placement first
        # 3) Then other RI slips
        # Within each group, sort by (quality desc, confidence desc)
        from operator import attrgetter

        def sort_group(items):
            return sorted(
                items,
                key=lambda x: (
                    -x.classification_result.quality_score,
                    -x.classification_result.confidence_score
                )
            )

        fac_list = [c for c in ri_slips if c.classification_result.ri_slip_type == RISlipType.FACULTATIVE_SLIP]
        placement_list = [c for c in ri_slips if c.classification_result.ri_slip_type == RISlipType.PLACEMENT_SLIP]
        other_ri_list = [c for c in ri_slips if c.classification_result.ri_slip_type not in {RISlipType.FACULTATIVE_SLIP, RISlipType.PLACEMENT_SLIP}]

        fac_list = sort_group(fac_list)
        placement_list = sort_group(placement_list)
        other_ri_list = sort_group(other_ri_list)

        # Quality override threshold (placement can precede facultative if much higher quality)
        QUALITY_OVERRIDE_THRESHOLD = 0.15

        ordered_ri_slips: List[DocumentClassificationResult] = []
        if fac_list and placement_list:
            fac_top_q = fac_list[0].classification_result.quality_score
            plc_top_q = placement_list[0].classification_result.quality_score
            if plc_top_q >= fac_top_q + QUALITY_OVERRIDE_THRESHOLD:
                ordered_ri_slips = placement_list + fac_list + other_ri_list
            else:
                ordered_ri_slips = fac_list + placement_list + other_ri_list
        else:
            # If only one of the groups present, keep natural preference: fac > placement > others
            if fac_list:
                ordered_ri_slips = fac_list + placement_list + other_ri_list
            elif placement_list:
                ordered_ri_slips = placement_list + other_ri_list
            else:
                ordered_ri_slips = other_ri_list

        # Replace ri_slips with the ordered list
        ri_slips = ordered_ri_slips

        # Sort supporting documents by relevance
        supporting_documents.sort(key=lambda x: -x.classification_result.confidence_score)
        
        # Sort low priority documents
        low_priority_documents.sort(key=lambda x: x.attachment_filename)
        
        # Assign processing order
        processing_order = []
        order_counter = 1
        
        # RI Slips first
        for classification in ri_slips:
            classification.processing_order = order_counter
            processing_order.append(classification.attachment_filename)
            order_counter += 1
        
        # Supporting documents second
        for classification in supporting_documents:
            classification.processing_order = order_counter
            processing_order.append(classification.attachment_filename)
            order_counter += 1
        
        # Low priority documents last
        for classification in low_priority_documents:
            classification.processing_order = order_counter
            processing_order.append(classification.attachment_filename)
            order_counter += 1
        
        return EmailProcessingPlan(
            ri_slips=ri_slips,
            supporting_documents=supporting_documents,
            low_priority_documents=low_priority_documents,
            total_documents=len(classifications),
            processing_order=processing_order
        )
    
    def validate_ri_slip_quality(self, identification_result: RISlipIdentificationResult) -> bool:
        """
        Validate if RI Slip meets quality requirements for processing
        
        Args:
            identification_result: RI Slip identification result
            
        Returns:
            True if RI Slip meets quality requirements
        """
        if not identification_result.is_ri_slip:
            return False
        
        # Check minimum quality score
        if identification_result.quality_score < self.min_quality_score:
            return False
        
        # Check for critical validation issues
        critical_issues = [
            "No reference number found",
            "No financial data (sum insured/percentages) found"
        ]
        
        for issue in critical_issues:
            if issue in identification_result.validation_issues:
                return False
        
        return True
    
    def get_processing_recommendations(self, processing_plan: EmailProcessingPlan) -> Dict[str, Any]:
        """
        Get processing recommendations based on the processing plan
        
        Args:
            processing_plan: Email processing plan
            
        Returns:
            Dictionary with processing recommendations
        """
        recommendations = {
            'total_documents': processing_plan.total_documents,
            'ri_slips_found': len(processing_plan.ri_slips),
            'high_quality_ri_slips': len([
                doc for doc in processing_plan.ri_slips 
                if doc.classification_result.quality_score >= self.good_quality_score
            ]),
            'processing_order': processing_plan.processing_order,
            'estimated_processing_time': self._estimate_processing_time(processing_plan),
            'recommendations': []
        }
        
        # Generate specific recommendations
        if not processing_plan.ri_slips:
            recommendations['recommendations'].append(
                "No RI Slips detected. Review attachments manually to ensure correct classification."
            )
        
        low_quality_ri_slips = [
            doc for doc in processing_plan.ri_slips 
            if doc.classification_result.quality_score < self.good_quality_score
        ]
        
        if low_quality_ri_slips:
            recommendations['recommendations'].append(
                f"{len(low_quality_ri_slips)} RI Slips have quality issues. Manual review recommended."
            )
        
        if len(processing_plan.ri_slips) > 3:
            recommendations['recommendations'].append(
                "Multiple RI Slips detected. Consider batch processing for efficiency."
            )
        
        return recommendations
    
    def _estimate_processing_time(self, processing_plan: EmailProcessingPlan) -> Dict[str, float]:
        """
        Estimate processing time for the email processing plan
        
        Args:
            processing_plan: Email processing plan
            
        Returns:
            Dictionary with time estimates in seconds
        """
        # Base processing times by document type (in seconds)
        base_times = {
            DocumentFormat.PDF: 30,
            DocumentFormat.DOCX: 20,
            DocumentFormat.EXCEL: 15,
            DocumentFormat.POWERPOINT: 25,
            DocumentFormat.IMAGE: 45,  # OCR takes longer
            DocumentFormat.UNKNOWN: 10
        }
        
        total_time = 0
        ri_slip_time = 0
        supporting_time = 0
        
        # Calculate time for RI Slips
        for doc in processing_plan.ri_slips:
            doc_format = doc.classification_result.document_format
            time_estimate = base_times.get(doc_format, 20)
            ri_slip_time += time_estimate
            total_time += time_estimate
        
        # Calculate time for supporting documents
        for doc in processing_plan.supporting_documents:
            doc_format = doc.classification_result.document_format
            time_estimate = base_times.get(doc_format, 15) * 0.7  # Supporting docs process faster
            supporting_time += time_estimate
            total_time += time_estimate
        
        # Add time for low priority documents (minimal processing)
        for doc in processing_plan.low_priority_documents:
            total_time += 5  # Minimal processing time
        
        return {
            'total_estimated_seconds': total_time,
            'ri_slips_seconds': ri_slip_time,
            'supporting_documents_seconds': supporting_time,
            'estimated_minutes': total_time / 60
        }


# Create global instance for use by other modules
ri_slip_identifier = RISlipIdentifier()