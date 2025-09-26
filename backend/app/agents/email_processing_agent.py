"""Email Processing Agent for Facultative Reinsurance System"""

import os
import imaplib
import email
import logging
from email.header import decode_header
from typing import List, Optional, Tuple
from pathlib import Path

# Import existing agents and models
from .data_extraction_agent import data_extraction_agent, ExtractedData
from .ri_slip_identifier import ri_slip_identifier, EmailProcessingPlan
from .email_models import EmailMessage, EmailAttachment, EmailProcessingConfig
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class EmailProcessingAgent:
    """Agent for processing emails with reinsurance applications"""
    
    def __init__(self, config: Optional[EmailProcessingConfig] = None):
        """Initialize the email processing agent"""
        self.config = config or self._load_default_config()
        self.mail = None
        # Ensure download directory exists
        try:
            Path(self.config.download_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not create download dir {self.config.download_dir}: {e}")

        # Connect to IMAP server (inline fallback since _connect_to_email is missing)
        try:
            self.mail = imaplib.IMAP4_SSL(self.config.imap_server)
            if self.config.email_user and self.config.email_password:
                self.mail.login(self.config.email_user, self.config.email_password)
        except Exception as e:
            logging.getLogger(__name__).error(f"Unable to connect/login to IMAP server {self.config.imap_server}: {e}")
            self.mail = None
        # Filtering: only process emails matching stricter patterns
        # Derived from provided sample subjects and common RI Slip cues
        # Keyword sets that must co-occur in subject (logical AND per set)
        self.subject_keyword_sets = [
            [re.compile(r"\bFacultative\b", re.IGNORECASE), re.compile(r"\bOffer\b", re.IGNORECASE)],
            [re.compile(r"\bRequest\b", re.IGNORECASE), re.compile(r"\bLine\b", re.IGNORECASE), re.compile(r"\bSupport\b", re.IGNORECASE)],
            [re.compile(r"\bCAR\b", re.IGNORECASE)],  # Construction All Risk
            [re.compile(r"\bFire\b", re.IGNORECASE), re.compile(r"\bFacultative\b", re.IGNORECASE)],
        ]
        # Optional sender allowlist patterns (comma-separated in env)
        sender_patterns_env = os.getenv("EMAIL_ALLOWED_SENDER_PATTERNS", "").strip()
        self.sender_patterns = []
        if sender_patterns_env:
            for pat in sender_patterns_env.split(","):
                pat = pat.strip()
                if pat:
                    # Treat as simple substring or regex if wrapped with /
                    if pat.startswith("/") and pat.endswith("/"):
                        self.sender_patterns.append(re.compile(pat[1:-1], re.IGNORECASE))
                    else:
                        self.sender_patterns.append(re.compile(re.escape(pat), re.IGNORECASE))
        # Body cues indicating RI-Slip-like content
        self.body_cues = [
            # Slip/form references
            re.compile(r"\bRI\s*Slip\b", re.IGNORECASE),
            re.compile(r"\bReinsurance\s+Slip\b", re.IGNORECASE),
            re.compile(r"\bPlacement\s+Slip\b", re.IGNORECASE),
            re.compile(r"\bLine\s+Slip\b", re.IGNORECASE),
            re.compile(r"\bFac(ultative)?\s+Offer\b", re.IGNORECASE),
            re.compile(r"\bFac(ultative)?\s+Support\b", re.IGNORECASE),
            re.compile(r"\bRequest\s+for\s+Line\s+Support\b", re.IGNORECASE),
            # Parties/roles
            re.compile(r"\bCedant\b|\bReinsured\b", re.IGNORECASE),
            re.compile(r"\bInsured\b", re.IGNORECASE),
            re.compile(r"\bBroker\b", re.IGNORECASE),
            # Key headings/fields
            re.compile(r"\bPerils?\s+Covered\b", re.IGNORECASE),
            re.compile(r"\bGeographical\s+Limit\b", re.IGNORECASE),
            re.compile(r"\bSituation\s+of\s+Risk|Voyage\b", re.IGNORECASE),
            re.compile(r"\bOccupation\s+of\s+Insured\b", re.IGNORECASE),
            re.compile(r"\bMain\s+Activities\b", re.IGNORECASE),
            re.compile(r"\bTotal\s+Sums?\s+Insured\b", re.IGNORECASE),
            re.compile(r"\bExcess|Retention\b", re.IGNORECASE),
            re.compile(r"\bPremium\s+Rate[s]?\b", re.IGNORECASE),
            re.compile(r"\bPeriod\s+of\s+Insurance\b", re.IGNORECASE),
            re.compile(r"\bPML\s*%?\b", re.IGNORECASE),
            re.compile(r"\bCAT\s+Exposure\b", re.IGNORECASE),
            re.compile(r"\bReinsurance\s+Deductions\b", re.IGNORECASE),
            re.compile(r"\bClaims\s+Experience\b", re.IGNORECASE),
            re.compile(r"\bShare\s+offered\s*%?\b", re.IGNORECASE),
            re.compile(r"\bSurveyor'?s\s+report\b", re.IGNORECASE),
            re.compile(r"\bClimate\s+Change\s+Risk\b|\bESG\s+Risk\b", re.IGNORECASE),
            # Financial table cues
            re.compile(r"\bCurrency\b", re.IGNORECASE),
            re.compile(r"\bDeductible\b", re.IGNORECASE),
            re.compile(r"\bLimit\b", re.IGNORECASE),
            re.compile(r"\bSum\s+Insured\b", re.IGNORECASE),
            re.compile(r"\bRate\s*%\b", re.IGNORECASE),
        ]
        # Attachment cues (filetypes or names) that indicate RI Slip/supporting docs
        self.attachment_name_cues = [
            re.compile(r"\bRI\s*Slip\b", re.IGNORECASE),
            re.compile(r"\bReinsurance\s+Slip\b", re.IGNORECASE),
            re.compile(r"\bPlacement\s+Slip\b", re.IGNORECASE),
            re.compile(r"\bLine\s+Slip\b", re.IGNORECASE),
            re.compile(r"\bOffer\b|\bQuote\b|\bQuotation\b", re.IGNORECASE),
            re.compile(r"\bSchedule\b|\bSOV\b|\bStatement\s+of\s+Values\b", re.IGNORECASE),
            re.compile(r"\bCAR\b|\bContract\s*All\s*Risk\b", re.IGNORECASE),
            re.compile(r"\bBinder\b|\bCover\s*Note\b", re.IGNORECASE),
        ]
        self.attachment_allowed_exts = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".png", ".jpg", ".jpeg"}
        # Polling status
        self.status = {
            "running": False,
            "last_run_at": None,
            "last_processed": 0,
            "last_successful": 0,
            "last_errors": 0,
            "total_processed": 0,
            "total_successful": 0,
            "total_errors": 0,
            "last_error": None,
        }
    
    def _load_default_config(self) -> EmailProcessingConfig:
        return EmailProcessingConfig(
            imap_server=os.getenv("EMAIL_IMAP_SERVER", "imap.gmail.com"),
            email_user=os.getenv("EMAIL_USER", ""),
            email_password=os.getenv("EMAIL_PASSWORD", ""),
            processed_folder="Processed",
            error_folder="Error",
            download_dir="/tmp/email_attachments",
            check_interval=int(os.getenv("CHECK_INTERVAL", "300")),  # seconds
        )
    
    def process_new_emails(self) -> List[ExtractedData]:
        """Process new unread emails"""
        self._reconnect_if_needed()
        self.status.update({
            "running": True,
            "last_run_at": datetime.utcnow().isoformat(),
            "last_processed": 0,
            "last_successful": 0,
            "last_errors": 0,
            "last_error": None,
        })
        results = []
        
        try:
            # Guard: ensure connection exists
            if not self.mail:
                logger.warning("IMAP connection not available; skipping this polling cycle")
                self.status["last_error"] = "no_connection"
                return []

            # Safely select inbox
            try:
                typ, _ = self.mail.select('inbox')
                if typ != 'OK':
                    logger.error("Failed to select inbox")
                    self.status["last_error"] = "select_failed"
                    return []
            except Exception as e:
                logger.error(f"Error selecting inbox: {e}")
                self.status["last_error"] = "select_exception"
                return []

            # Search unseen
            try:
                status, messages = self.mail.search(None, 'UNSEEN')
            except Exception as e:
                logger.error(f"Error searching emails: {e}")
                self.status["last_error"] = "search_exception"
                return []
            if status != 'OK':
                logger.error("Failed to search emails")
                self.status["last_error"] = "search_failed"
                return []
            
            for msg_id in messages[0].split():
                try:
                    if not self.mail:
                        logger.warning("IMAP connection lost mid-cycle; aborting message fetch")
                        self.status["last_error"] = "connection_lost"
                        break
                    status, msg_data = self.mail.fetch(msg_id, '(RFC822)')
                    if status == 'OK':
                        # _process_email returns (EmailMessage, success_flag)
                        email_msg, success = self._process_email(msg_data[0][1])
                        if not success or not email_msg:
                            self.status["last_error"] = "parse_failed"
                            continue

                        # Screen by sender, subject, and cues before moving
                        if not self._passes_sender_filter(email_msg.from_address):
                            self.status["last_error"] = "sender_filtered"
                            continue
                        if not self._subject_is_relevant(email_msg.subject):
                            self.status["last_error"] = "subject_not_relevant"
                            continue
                        has_cues = self._has_ri_slip_cues(email_msg)
                        if not has_cues:
                            self.status["last_error"] = "no_ri_slip_cues"
                            continue

                        extracted_data = self._extract_data(email_msg)
                        self.status["last_processed"] += 1
                        if extracted_data:
                            results.append(extracted_data)
                            self.status["last_successful"] += 1
                        else:
                            self.status["last_errors"] += 1
                        # Move only relevant emails that passed cues screening
                        self._move_email(msg_id, self.config.processed_folder)
                except Exception as e:
                    logger.error(f"Error processing email {msg_id}: {e}")
                    self.status["last_errors"] += 1
                    self.status["last_error"] = str(e)
                    # Move the problematic email to error folder
                    try:
                        self._move_email(msg_id, self.config.error_folder)
                    except Exception:
                        pass
            
            # Update totals
            self.status["total_processed"] += self.status["last_processed"]
            self.status["total_successful"] += self.status["last_successful"]
            self.status["total_errors"] += self.status["last_errors"]
            return results
        
        except Exception as e:
            logger.error(f"Error in process_new_emails: {e}")
            self.status["last_error"] = str(e)
            return []
        finally:
            self.status["running"] = False

    def _subject_is_relevant(self, subject: str) -> bool:
        if not subject:
            return False
        # Must match at least one keyword set (all terms in that set present)
        for kw_set in self.subject_keyword_sets:
            if all(p.search(subject) for p in kw_set):
                return True
        return False

    def _passes_sender_filter(self, sender: str) -> bool:
        if not self.sender_patterns:
            return True
        if not sender:
            return False
        return any(p.search(sender) for p in self.sender_patterns)

    def _has_ri_slip_cues(self, email_msg: EmailMessage) -> bool:
        # Check body cues
        body = email_msg.body or ""
        body_hit = any(p.search(body) for p in self.body_cues)
        # Check attachments cues
        attachment_hit = False
        for a in (email_msg.attachments or []):
            name = a.filename or ""
            ext = os.path.splitext(name)[1].lower()
            if ext in self.attachment_allowed_exts and any(p.search(name) for p in self.attachment_name_cues):
                attachment_hit = True
                break
        return body_hit or attachment_hit
    
    def _process_email(self, msg_data: bytes) -> Tuple[Optional[EmailMessage], bool]:
        """Process a single email message and return (email_message, success_flag)"""
        try:
            msg = email.message_from_bytes(msg_data)
            email_msg = EmailMessage(
                message_id=msg['message-id'] or "",
                subject=decode_header(msg['subject'])[0][0] if msg['subject'] else "No Subject",
                from_address=msg['from'] or "",
                to_address=msg['to'] or "",
                date=msg['date'] or "",
                body=self._extract_body(msg),
                attachments=self._extract_attachments(msg)
            )
            return email_msg, True
        except Exception as e:
            logger.error(f"Error processing email: {e}")
            return None, False
    
    def _extract_body(self, msg) -> str:
        """Extract email body text"""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        return part.get_payload(decode=True).decode('utf-8', 'ignore')
                    except:
                        pass
        return msg.get_payload(decode=True).decode('utf-8', 'ignore') if not msg.is_multipart() else ""
    
    def _extract_attachments(self, msg) -> List[EmailAttachment]:
        """Extract email attachments"""
        attachments = []
        for part in msg.walk():
            if part.get_content_disposition() == 'attachment':
                filename = part.get_filename()
                if filename:
                    try:
                        file_data = part.get_payload(decode=True)
                        filepath = os.path.join(self.config.download_dir, filename)
                        with open(filepath, 'wb') as f:
                            f.write(file_data)
                        attachments.append(EmailAttachment(
                            filename=filename,
                            filepath=filepath,
                            content_type=part.get_content_type(),
                            size=len(file_data)
                        ))
                    except Exception as e:
                        logger.error(f"Error saving attachment {filename}: {e}")
        return attachments
    
    def _extract_data(self, email_msg: EmailMessage) -> Optional[ExtractedData]:
        """Extract data from email and attachments"""
        try:
            # Process email body
            extracted_data = data_extraction_agent.extract_from_text(email_msg.body)
            
            # Process attachments
            for attachment in email_msg.attachments:
                try:
                    data = data_extraction_agent.extract_from_attachment(attachment)
                    extracted_data = self._merge_data(extracted_data, data)
                except Exception as e:
                    logger.error(f"Error processing attachment {attachment.filename}: {e}")
            
            return extracted_data
        except Exception as e:
            logger.error(f"Error extracting data: {e}")
            return None
    
    def _merge_data(self, base: ExtractedData, new: ExtractedData) -> ExtractedData:
        """Merge two ExtractedData objects"""
        if not base:
            return new
        # Simple merge logic - can be enhanced based on requirements
        for field in ["document_type", "reference_number", "date"]:
            if getattr(new, field) and not getattr(base, field):
                setattr(base, field, getattr(new, field))
        return base
    
    def _move_email(self, msg_id: str, folder: str):
        """Move email to another folder"""
        try:
            if not self.mail:
                logger.warning("Cannot move email; no IMAP connection")
                return
            self.mail.copy(msg_id, folder)
            self.mail.store(msg_id, '+FLAGS', '\\Deleted')
            self.mail.expunge()
        except Exception as e:
            logger.error(f"Error moving email: {e}")
    
    def _reconnect_if_needed(self):
        """Reconnect if connection is lost"""
        try:
            if not self.mail:
                raise Exception("no_connection")
            self.mail.noop()
        except:
            try:
                # close any stale connection first
                if self.mail:
                    try:
                        self.mail.logout()
                    except Exception:
                        pass
                # attempt fresh connection/login
                self.mail = imaplib.IMAP4_SSL(self.config.imap_server)
                if self.config.email_user and self.config.email_password:
                    self.mail.login(self.config.email_user, self.config.email_password)
                logging.getLogger(__name__).info("Reconnected to IMAP server")
            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Failed to reconnect to IMAP server {getattr(self.config, 'imap_server', 'unknown')}: {e}"
                )
                self.mail = None
    
    def __del__(self):
        """Cleanup on object deletion"""
        try:
            if self.mail:
                self.mail.logout()
        except:
            pass
