#!/usr/bin/env python3
"""
Script to process new emails for the Facultative Reinsurance System.

This script:
1. Connects to the email server
2. Processes new unread emails
3. Extracts data from emails and attachments
4. Saves extracted data to the database
5. Moves processed emails to appropriate folders

Environment variables required:
- EMAIL_IMAP_SERVER: IMAP server address (e.g., imap.gmail.com)
- EMAIL_USER: Email address
- EMAIL_PASSWORD: Email password or app password
- EMAIL_PROCESSED_FOLDER: Folder to move processed emails to (default: "Processed")
- EMAIL_ERROR_FOLDER: Folder to move failed emails to (default: "Error")
- EMAIL_DOWNLOAD_DIR: Directory to save attachments (default: "/tmp/email_attachments")
- CHECK_INTERVAL: How often to check for new emails in seconds (default: 300)
"""

import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
load_dotenv()

from app.agents.email_processing_agent import EmailProcessingAgent, EmailProcessingConfig

def main():
    """Main function to run the email processing service"""
    # Create download directory if it doesn't exist
    download_dir = os.getenv("EMAIL_DOWNLOAD_DIR", "/tmp/email_attachments")
    os.makedirs(download_dir, exist_ok=True)
    
    # Configure the email processing agent
    config = EmailProcessingConfig(
        imap_server=os.getenv("EMAIL_IMAP_SERVER"),
        email_user=os.getenv("EMAIL_USER"),
        email_password=os.getenv("EMAIL_PASSWORD"),
        processed_folder=os.getenv("EMAIL_PROCESSED_FOLDER", "Processed"),
        error_folder=os.getenv("EMAIL_ERROR_FOLDER", "Error"),
        download_dir=download_dir,
        check_interval=int(os.getenv("CHECK_INTERVAL", "300")),
        max_attachment_size=int(os.getenv("MAX_ATTACHMENT_SIZE", str(50 * 1024 * 1024))),  # 50MB
    )
    
    # Initialize the email processing agent
    agent = EmailProcessingAgent(config)
    
    logger.info("Starting email processing service...")
    logger.info(f"Checking for new emails every {config.check_interval} seconds")
    logger.info(f"Downloading attachments to: {config.download_dir}")
    
    try:
        while True:
            try:
                # Process new emails
                results = agent.process_new_emails()
                if results:
                    logger.info(f"Processed {len(results)} emails")
                
                # Wait before checking again
                time.sleep(config.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Stopping email processing service...")
                break
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
                
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        # Clean up
        if hasattr(agent, 'close'):
            agent.close()

if __name__ == "__main__":
    # Check required environment variables
    required_vars = ["EMAIL_IMAP_SERVER", "EMAIL_USER", "EMAIL_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    main()
