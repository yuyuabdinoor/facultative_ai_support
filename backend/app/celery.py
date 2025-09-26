"""
Celery configuration for the AI Facultative Reinsurance System
"""
import logging
from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure, task_retry
from kombu import Queue, Exchange
from app.core.config import settings

logger = logging.getLogger(__name__)

# Create Celery instance
celery_app = Celery(
    "reinsurance_system",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.agents.tasks",
        "app.agents.orchestration_tasks",
        "app.agents.workflow_tasks"
    ]
)

# Define exchanges and queues for task routing
default_exchange = Exchange('default', type='direct')
priority_exchange = Exchange('priority', type='direct')
workflow_exchange = Exchange('workflow', type='direct')

# Configure task routing and priority management
celery_app.conf.update(
    # Basic configuration
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task tracking and monitoring
    task_track_started=True,
    task_send_sent_event=True,
    task_ignore_result=False,
    result_expires=3600,  # 1 hour
    
    # Task execution limits
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Task routing configuration
    task_routes={
        'app.agents.tasks.process_document_ocr': {'queue': 'ocr_processing'},
        'app.agents.tasks.process_pdf_ocr': {'queue': 'ocr_processing'},
        'app.agents.tasks.process_email_parsing': {'queue': 'document_processing'},
        'app.agents.tasks.process_excel_parsing': {'queue': 'document_processing'},
        'app.agents.tasks.batch_process_documents': {'queue': 'batch_processing'},
        'app.agents.orchestration_tasks.*': {'queue': 'orchestration'},
        'app.agents.workflow_tasks.*': {'queue': 'workflow'},
    },
    
    # Queue configuration with priorities
    task_default_queue='default',
    task_queues=(
        Queue('default', default_exchange, routing_key='default'),
        Queue('ocr_processing', default_exchange, routing_key='ocr_processing'),
        Queue('document_processing', default_exchange, routing_key='document_processing'),
        Queue('batch_processing', default_exchange, routing_key='batch_processing'),
        Queue('orchestration', workflow_exchange, routing_key='orchestration'),
        Queue('workflow', workflow_exchange, routing_key='workflow'),
        Queue('priority', priority_exchange, routing_key='priority'),
    ),
    
    # Retry configuration
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Result backend configuration
    result_backend_transport_options={
        'master_name': 'mymaster',
        'visibility_timeout': 3600,
    },
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-expired-results': {
            'task': 'app.agents.orchestration_tasks.cleanup_expired_results',
            'schedule': 3600.0,  # Every hour
        },
        'monitor-task-health': {
            'task': 'app.agents.orchestration_tasks.monitor_task_health',
            'schedule': 300.0,  # Every 5 minutes
        },
        'aggregate-task-metrics': {
            'task': 'app.agents.orchestration_tasks.aggregate_task_metrics',
            'schedule': 600.0,  # Every 10 minutes
        },
    },
)

# Task event monitoring
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Log task start"""
    logger.info(f"Task {task.name} [{task_id}] started with args: {args}, kwargs: {kwargs}")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Log task completion"""
    logger.info(f"Task {task.name} [{task_id}] completed with state: {state}")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Log task failure"""
    logger.error(f"Task {sender.name} [{task_id}] failed: {exception}")

@task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, einfo=None, **kwds):
    """Log task retry"""
    logger.warning(f"Task {sender.name} [{task_id}] retrying: {reason}")