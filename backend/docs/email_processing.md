# Email Processing for Facultative Reinsurance System

This document describes the email processing functionality for the Facultative Reinsurance Decision Support System.

## Overview

The email processing system is responsible for:
1. Monitoring an email inbox for new messages
2. Processing email messages and attachments
3. Extracting reinsurance application data
4. Saving extracted data to the system
5. Organizing processed emails into folders

## Components

### 1. Email Processing Agent

Located in `app/agents/email_processing_agent.py`, this is the main class that handles email processing. It provides methods to:
- Connect to an IMAP email server
- Fetch and process new emails
- Extract and process attachments
- Move processed emails to appropriate folders

### 2. Email Models

Located in `app/agents/email_models.py`, these Pydantic models define the data structures used for email processing:
- `EmailAttachment`: Represents an email attachment
- `EmailMessage`: Represents an email message with attachments
- `EmailProcessingConfig`: Configuration for the email processing
- `EmailProcessingResult`: Result of processing an email

### 3. Processing Script

Located in `scripts/process_emails.py`, this script provides a command-line interface to run the email processing service.

## Configuration

Email processing is configured through environment variables in the `.env` file:

```ini
# Email Processing
EMAIL_IMAP_SERVER=imap.example.com
EMAIL_USER=your-email@example.com
EMAIL_PASSWORD=your-email-password
EMAIL_PROCESSED_FOLDER=Processed
EMAIL_ERROR_FOLDER=Error
EMAIL_DOWNLOAD_DIR=./email_attachments
CHECK_INTERVAL=300
MAX_ATTACHMENT_SIZE=52428800  # 50MB in bytes
```

## Running the Email Processor

1. Copy `.env.example` to `.env` and update the email settings:
   ```bash
   cp .env.example .env
   # Edit .env with your email settings
   ```

2. Install required dependencies:
   ```bash
   pip install python-dotenv imapclient
   ```

3. Run the email processor:
   ```bash
   python -m scripts.process_emails
   ```

## How It Works

1. The email processor connects to the configured IMAP server
2. It searches for unread emails in the inbox
3. For each email:
   - The email body is extracted and processed
   - Attachments are downloaded and processed
   - RI Slips are identified and prioritized
   - Data is extracted from the email and attachments
   - The email is moved to the "Processed" folder
4. If an error occurs, the email is moved to the "Error" folder
5. The process repeats after the configured interval

## Troubleshooting

- **Connection issues**: Verify your email server settings and credentials
- **Attachment processing errors**: Check the logs for specific error messages
- **Missing data**: Ensure the email format matches the expected structure

## Security Considerations

- Email credentials should be stored securely and never committed to version control
- Downloaded attachments are stored temporarily and should be cleaned up after processing
- The system should run with minimal necessary permissions

## Future Enhancements

- Support for additional email protocols (e.g., Microsoft Graph API)
- Improved error handling and retry logic
- Better support for different email formats and encodings
- Integration with the main application's notification system
