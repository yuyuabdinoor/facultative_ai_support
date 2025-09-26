"""
Email Data Models for Facultative Reinsurance System

Defines data structures for processing emails and attachments.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path


class EmailAttachment(BaseModel):
    """Represents an email attachment"""
    filename: str
    filepath: str
    content_type: str
    size: int
    processing_status: str = "pending"  # pending, processing, processed, error
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmailMessage(BaseModel):
    """Represents an email message with attachments"""
    message_id: str
    subject: str
    from_address: str
    to_address: str
    date: str
    body: str
    attachments: List[EmailAttachment] = Field(default_factory=list)
    status: str = "received"  # received, processing, processed, error
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmailProcessingResult(BaseModel):
    """Result of processing an email message"""
    success: bool
    message_id: str
    extracted_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processed_attachments: List[str] = Field(default_factory=list)
    failed_attachments: List[Dict[str, str]] = Field(default_factory=list)


class EmailProcessingConfig(BaseModel):
    """Configuration for email processing"""
    imap_server: str
    email_user: str
    email_password: str
    processed_folder: str = "Processed"
    error_folder: str = "Error"
    download_dir: str = "/tmp/email_attachments"
    check_interval: int = 300  # seconds
    max_attachment_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: List[str] = [
        ".pdf", 
        ".docx", 
        ".xlsx", 
        ".xls", 
        ".msg", 
        ".jpg", 
        ".jpeg", 
        ".png"
    ]
