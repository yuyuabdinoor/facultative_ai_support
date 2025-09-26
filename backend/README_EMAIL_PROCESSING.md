# Email Processing for Facultative Reinsurance System

This document outlines the email processing functionality for the Facultative Reinsurance Decision Support System.

## Overview

The email processing system is responsible for:
- Monitoring an email inbox for new messages
- Processing email messages and attachments
- Extracting reinsurance application data
- Integrating with the existing data extraction pipeline
- Organizing processed emails into folders

## Components

### 1. Email Processing Agent (`email_processing_agent.py`)

The main class that handles email processing, including:
- Connecting to IMAP email servers
- Fetching and processing new emails
- Extracting and processing attachments
- Moving processed emails to appropriate folders
- Error handling and logging

### 2. Email Models (`email_models.py`)

Pydantic models for email processing:
- `EmailAttachment`: Represents an email attachment
- `EmailMessage`: Represents an email message with attachments
- `EmailProcessingConfig`: Configuration for the email processor
- `EmailProcessingResult`: Result of processing an email

### 3. Processing Script (`scripts/process_emails.py`)

Command-line interface for running the email processing service.

## Configuration

Email processing is configured through environment variables:

```ini
# Email Server Configuration
EMAIL_IMAP_SERVER=imap.example.com
EMAIL_USER=your-email@example.com
EMAIL_PASSWORD=your-email-password

# Processing Options
EMAIL_PROCESSED_FOLDER=Processed
EMAIL_ERROR_FOLDER=Error
EMAIL_DOWNLOAD_DIR=./email_attachments
CHECK_INTERVAL=300  # seconds
MAX_ATTACHMENT_SIZE=52428800  # 50MB in bytes
```

## Installation

1. Install required dependencies:
   ```bash
   pip install -r requirements-email.txt
   ```

2. Copy and configure the environment file:
   ```bash
   cp .env.example .env
   # Edit .env with your email settings
   ```

## Running the Email Processor

### Development

```bash
python -m scripts.process_emails
```

### Production

For production, consider running the processor as a systemd service or using a process manager like Supervisor.

## Testing

Run the test suite with:

```bash
pytest tests/test_email_processing.py -v
```

## Error Handling

- Failed email processing is logged with detailed error messages
- Problematic emails are moved to the error folder for manual review
- The processor automatically reconnects if the connection is lost

## Monitoring

Check the application logs for:
- Connection status
- Email processing results
- Error messages
- Performance metrics

## Security Considerations

- Email credentials are stored in environment variables
- Attachments are scanned for size limits before processing
- Temporary files are cleaned up after processing
- The processor runs with minimal necessary permissions

## Troubleshooting

Common issues and solutions:

1. **Connection refused**: Verify IMAP server address and port
2. **Authentication failed**: Check email credentials
3. **Permission denied**: Ensure the download directory is writable
4. **Timeout errors**: Increase the timeout in the configuration

## Dependencies

See `requirements-email.txt` for a complete list of dependencies.

## License

[Your License Here]
