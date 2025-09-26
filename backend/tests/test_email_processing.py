"""Tests for email processing functionality."""

import pytest
from unittest.mock import MagicMock, patch
from app.agents.email_models import EmailMessage, EmailAttachment, EmailProcessingConfig
from app.agents.email_processing_agent import EmailProcessingAgent

# Test data
SAMPLE_EMAIL = {
    "message_id": "<test@example.com>",
    "subject": "Test Email",
    "from_address": "from@example.com",
    "to_address": "to@example.com",
    "date": "Wed, 1 Jan 2023 12:00:00 +0000",
    "body": "Test email body"
}

SAMPLE_ATTACHMENT = {
    "filename": "test.pdf",
    "filepath": "/tmp/test.pdf",
    "content_type": "application/pdf",
    "size": 1024
}

@pytest.fixture
def config():
    return EmailProcessingConfig(
        imap_server="imap.test.com",
        email_user="test@example.com",
        email_password="testpass"
    )

class TestEmailModels:
    def test_email_message_creation(self):
        email = EmailMessage(**SAMPLE_EMAIL)
        assert email.message_id == SAMPLE_EMAIL["message_id"]
        assert email.subject == SAMPLE_EMAIL["subject"]
        assert email.status == "received"

    def test_email_attachment_creation(self):
        attachment = EmailAttachment(**SAMPLE_ATTACHMENT)
        assert attachment.filename == SAMPLE_ATTACHMENT["filename"]
        assert attachment.processing_status == "pending"

class TestEmailProcessingAgent:
    @patch("imaplib.IMAP4_SSL")
    def test_connect_to_email(self, mock_imap, config):
        mock_imap.return_value.login.return_value = ("OK", [b"Success"])
        agent = EmailProcessingAgent(config)
        assert agent.mail is not None
        mock_imap.return_value.login.assert_called_once()

    @patch("builtins.open", new_callable=MagicMock)
    def test_extract_attachments(self, mock_open, config):
        # Setup mock email message with attachment
        msg = MagicMock()
        part = MagicMock()
        part.get_content_type.return_value = "application/pdf"
        part.get.return_value = "attachment; filename=test.pdf"
        part.get_filename.return_value = "test.pdf"
        part.get_payload.return_value = b"PDF content"
        msg.walk.return_value = [part]
        
        # Test
        agent = EmailProcessingAgent(config)
        attachments = agent._extract_attachments(msg)
        
        # Verify
        assert len(attachments) == 1
        assert attachments[0].filename == "test.pdf"
        mock_open.assert_called_once()

    @patch("app.agents.email_processing_agent.EmailProcessingAgent._extract_attachments")
    @patch("email.message_from_bytes")
    def test_process_email(self, mock_msg_from_bytes, mock_extract, config):
        # Setup
        mock_msg = MagicMock()
        mock_msg.__getitem__.side_effect = lambda x: {
            'message-id': SAMPLE_EMAIL['message_id'],
            'subject': SAMPLE_EMAIL['subject'],
            'from': SAMPLE_EMAIL['from_address'],
            'to': SAMPLE_EMAIL['to_address'],
            'date': SAMPLE_EMAIL['date']
        }.get(x, '')
        mock_msg.is_multipart.return_value = False
        mock_msg.get_payload.return_value = SAMPLE_EMAIL['body']
        mock_msg_from_bytes.return_value = mock_msg
        
        # Test
        agent = EmailProcessingAgent(config)
        email_msg, success = agent._process_email(b"test")
        
        # Verify
        assert success is True
        assert email_msg.message_id == SAMPLE_EMAIL['message_id']
        assert email_msg.subject == SAMPLE_EMAIL['subject']
