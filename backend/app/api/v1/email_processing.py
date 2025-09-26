"""
Email Processing API endpoints
"""
from __future__ import annotations

import os
import asyncio
from typing import Optional, Dict, Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ...agents.email_processing_agent import EmailProcessingAgent, EmailProcessingConfig

router = APIRouter()

# Singleton agent instance for this process
_agent: Optional[EmailProcessingAgent] = None
_polling_task: Optional[asyncio.Task] = None


def get_agent() -> EmailProcessingAgent:
    global _agent
    if _agent is None:
        cfg = EmailProcessingConfig(
            imap_server=os.getenv("EMAIL_IMAP_SERVER", ""),
            email_user=os.getenv("EMAIL_USER", ""),
            email_password=os.getenv("EMAIL_PASSWORD", ""),
            processed_folder=os.getenv("EMAIL_PROCESSED_FOLDER", "Processed"),
            error_folder=os.getenv("EMAIL_ERROR_FOLDER", "Error"),
            download_dir=os.getenv("EMAIL_DOWNLOAD_DIR", "/tmp/email_attachments"),
            check_interval=int(os.getenv("CHECK_INTERVAL", "300")),
            max_attachment_size=int(os.getenv("MAX_ATTACHMENT_SIZE", str(50 * 1024 * 1024))),
        )
        _agent = EmailProcessingAgent(cfg)
    return _agent


@router.post("/emails/process-now", summary="Process unread emails now")
async def process_emails_now(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Trigger immediate processing of unread emails.
    Runs in a background task and returns a summary.
    """
    agent = get_agent()

    def _run():
        return agent.process_new_emails()

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, _run)

    ok = [r for r in results if getattr(r, "success", True)]  # agent returns list of processing results or extracted data
    return {
        "processed": len(results),
        "successful": len(ok),
    }


@router.get("/emails/config", summary="Get email processor configuration")
async def get_email_config() -> Dict[str, Any]:
    agent = get_agent()
    cfg = agent.config
    # redact password
    redacted = {
        "imap_server": cfg.imap_server,
        "email_user": cfg.email_user,
        "processed_folder": cfg.processed_folder,
        "error_folder": cfg.error_folder,
        "download_dir": cfg.download_dir,
        "check_interval": cfg.check_interval,
        "max_attachment_size": cfg.max_attachment_size,
    }
    return redacted


@router.post("/emails/polling/enable", summary="Enable periodic email polling")
async def enable_polling() -> Dict[str, Any]:
    global _polling_task
    agent = get_agent()
    if _polling_task and not _polling_task.done():
        return {"status": "already_running"}

    async def _poll_loop():
        while True:
            try:
                await asyncio.get_event_loop().run_in_executor(None, agent.process_new_emails)
            except Exception:
                # swallow errors to keep loop alive; logs occur inside agent
                pass
            await asyncio.sleep(agent.config.check_interval)

    _polling_task = asyncio.create_task(_poll_loop())
    return {"status": "enabled"}


@router.post("/emails/polling/disable", summary="Disable periodic email polling")
async def disable_polling() -> Dict[str, Any]:
    global _polling_task
    if _polling_task:
        _polling_task.cancel()
        _polling_task = None
    return {"status": "disabled"}


@router.get("/emails/polling/status", summary="Get current polling status")
async def polling_status() -> Dict[str, Any]:
    agent = get_agent()
    status = getattr(agent, "status", {})
    running = False
    try:
        running = _polling_task is not None and not _polling_task.done()
    except Exception:
        running = False
    return {
        "agent_running": status.get("running", False),
        "polling_task_running": running,
        "last_run_at": status.get("last_run_at"),
        "last_processed": status.get("last_processed", 0),
        "last_successful": status.get("last_successful", 0),
        "last_errors": status.get("last_errors", 0),
        "total_processed": status.get("total_processed", 0),
        "total_successful": status.get("total_successful", 0),
        "total_errors": status.get("total_errors", 0),
        "last_error": status.get("last_error"),
    }
