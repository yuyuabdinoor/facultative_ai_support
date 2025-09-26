"""
Celery orchestration tasks for multi-agent workflow management

This module contains tasks for coordinating multiple agents, managing workflows,
and handling task aggregation and monitoring.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from celery import group, chain, chord, signature
from celery.result import AsyncResult, GroupResult
from celery.exceptions import Retry, WorkerLostError
from sqlalchemy.orm import Session

try:
    from app.celery import celery_app
    CELERY_AVAILABLE = True
except ImportError:
    celery_app = None
    CELERY_AVAILABLE = False

from app.core.database import get_db
from app.models.database import Document, Application, RiskAnalysis, Recommendation
from app.models.schemas import WorkflowStatus, TaskResult, OrchestrationResult

logger = logging.getLogger(__name__)


def _celery_task_decorator(bind=True, max_retries=3, default_retry_delay=60):
    """Conditional celery task decorator with retry configuration"""
    def decorator(func):
        if CELERY_AVAILABLE and celery_app:
            return celery_app.task(
                bind=bind, 
                max_retries=max_retries,
                default_retry_delay=default_retry_delay,
                autoretry_for=(Exception,),
                retry_backoff=True,
                retry_backoff_max=600,
                retry_jitter=True
            )(func)
        else:
            return func
    return decorator


@_celery_task_decorator(bind=True, max_retries=5)
def orchestrate_document_processing_workflow(self, application_id: str, document_ids: List[str]) -> Dict[str, Any]:
    """
    Orchestrate the complete document processing workflow for an application
    
    Args:
        application_id: UUID of the application
        document_ids: List of document IDs to process
        
    Returns:
        Workflow orchestration results
    """
    try:
        logger.info(f"Starting document processing workflow for application {application_id}")
        
        # Update application status
        db = next(get_db())
        try:
            application = db.query(Application).filter(Application.id == application_id).first()
            if not application:
                raise ValueError(f"Application {application_id} not found")
            
            application.status = "processing"
            db.commit()
        finally:
            db.close()
        
        # Phase 1: OCR and Document Processing (parallel)
        ocr_tasks = []
        for doc_id in document_ids:
            db = next(get_db())
            try:
                document = db.query(Document).filter(Document.id == doc_id).first()
                if document:
                    if document.document_type == 'pdf':
                        task = signature('app.agents.tasks.process_pdf_ocr', args=[doc_id, document.file_path])
                    elif document.document_type == 'email':
                        task = signature('app.agents.tasks.process_email_parsing', args=[doc_id, document.file_path])
                    elif document.document_type == 'excel':
                        task = signature('app.agents.tasks.process_excel_parsing', args=[doc_id, document.file_path])
                    else:
                        task = signature('app.agents.tasks.process_document_ocr', args=[doc_id, document.file_path])
                    
                    ocr_tasks.append(task)
            finally:
                db.close()
        
        # Execute OCR tasks in parallel
        ocr_job = group(ocr_tasks)
        ocr_result = ocr_job.apply_async()
        
        # Wait for OCR completion with timeout
        ocr_results = ocr_result.get(timeout=1800)  # 30 minutes timeout
        
        # Phase 2: Data Extraction (sequential, depends on OCR)
        extraction_task = signature(
            'app.agents.workflow_tasks.extract_risk_data_workflow',
            args=[application_id, document_ids]
        )
        
        # Phase 3: Risk Analysis (depends on extraction)
        risk_analysis_task = signature(
            'app.agents.workflow_tasks.risk_analysis_workflow',
            args=[application_id]
        )
        
        # Phase 4: Business Limits Validation (parallel with risk analysis)
        limits_validation_task = signature(
            'app.agents.workflow_tasks.validate_business_limits_workflow',
            args=[application_id]
        )
        
        # Phase 5: Decision Generation (depends on analysis and validation)
        decision_task = signature(
            'app.agents.workflow_tasks.generate_decision_workflow',
            args=[application_id]
        )
        
        # Phase 6: Market Grouping (can run in parallel)
        market_grouping_task = signature(
            'app.agents.workflow_tasks.market_grouping_workflow',
            args=[application_id, document_ids]
        )
        
        # Create workflow chain
        analysis_chain = chain(
            extraction_task,
            group(risk_analysis_task, limits_validation_task),
            decision_task
        )
        
        # Execute analysis workflow
        analysis_result = analysis_chain.apply_async()
        
        # Execute market grouping in parallel
        market_result = market_grouping_task.apply_async()
        
        # Wait for all workflows to complete
        final_analysis = analysis_result.get(timeout=1800)
        market_analysis = market_result.get(timeout=600)
        
        # Update application status
        db = next(get_db())
        try:
            application = db.query(Application).filter(Application.id == application_id).first()
            if application:
                application.status = "completed"
                application.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()
        
        logger.info(f"Successfully completed workflow for application {application_id}")
        
        return {
            'application_id': application_id,
            'workflow_status': 'completed',
            'ocr_results': ocr_results,
            'analysis_results': final_analysis,
            'market_results': market_analysis,
            'total_documents_processed': len(document_ids),
            'completion_time': datetime.utcnow().isoformat()
        }
        
    except Exception as exc:
        logger.error(f"Workflow failed for application {application_id}: {str(exc)}")
        
        # Update application status to failed
        try:
            db = next(get_db())
            application = db.query(Application).filter(Application.id == application_id).first()
            if application:
                application.status = "failed"
                application.updated_at = datetime.utcnow()
                db.commit()
            db.close()
        except Exception as db_error:
            logger.error(f"Failed to update application status: {str(db_error)}")
        
        # Retry logic with exponential backoff
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying workflow for application {application_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        return {
            'application_id': application_id,
            'workflow_status': 'failed',
            'error': str(exc),
            'retry_count': self.request.retries
        }


@_celery_task_decorator(bind=True, max_retries=3)
def aggregate_task_results(self, task_ids: List[str], result_type: str = "workflow") -> Dict[str, Any]:
    """
    Aggregate results from multiple tasks
    
    Args:
        task_ids: List of Celery task IDs
        result_type: Type of results being aggregated
        
    Returns:
        Aggregated results dictionary
    """
    try:
        logger.info(f"Aggregating {len(task_ids)} task results of type {result_type}")
        
        results = []
        failed_tasks = []
        successful_tasks = []
        
        for task_id in task_ids:
            try:
                result = AsyncResult(task_id, app=celery_app)
                
                if result.ready():
                    if result.successful():
                        task_result = result.get()
                        results.append({
                            'task_id': task_id,
                            'status': 'success',
                            'result': task_result
                        })
                        successful_tasks.append(task_id)
                    else:
                        results.append({
                            'task_id': task_id,
                            'status': 'failed',
                            'error': str(result.info)
                        })
                        failed_tasks.append(task_id)
                else:
                    results.append({
                        'task_id': task_id,
                        'status': 'pending',
                        'state': result.state
                    })
                    
            except Exception as e:
                logger.error(f"Error getting result for task {task_id}: {str(e)}")
                results.append({
                    'task_id': task_id,
                    'status': 'error',
                    'error': str(e)
                })
                failed_tasks.append(task_id)
        
        aggregation_summary = {
            'result_type': result_type,
            'total_tasks': len(task_ids),
            'successful_tasks': len(successful_tasks),
            'failed_tasks': len(failed_tasks),
            'pending_tasks': len(task_ids) - len(successful_tasks) - len(failed_tasks),
            'success_rate': len(successful_tasks) / len(task_ids) if task_ids else 0,
            'aggregation_time': datetime.utcnow().isoformat(),
            'results': results
        }
        
        logger.info(f"Aggregation completed: {len(successful_tasks)}/{len(task_ids)} tasks successful")
        
        return aggregation_summary
        
    except Exception as exc:
        logger.error(f"Error aggregating task results: {str(exc)}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=30 * (2 ** self.request.retries))
        
        return {
            'result_type': result_type,
            'status': 'aggregation_failed',
            'error': str(exc),
            'task_ids': task_ids
        }


@_celery_task_decorator(bind=True, max_retries=2)
def monitor_task_health(self) -> Dict[str, Any]:
    """
    Monitor overall task health and system status
    
    Returns:
        System health metrics
    """
    try:
        logger.info("Monitoring task health and system status")
        
        # Get active tasks
        inspect = celery_app.control.inspect()
        
        # Get worker stats
        stats = inspect.stats()
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()
        reserved_tasks = inspect.reserved()
        
        # Calculate metrics
        total_workers = len(stats) if stats else 0
        total_active_tasks = sum(len(tasks) for tasks in active_tasks.values()) if active_tasks else 0
        total_scheduled_tasks = sum(len(tasks) for tasks in scheduled_tasks.values()) if scheduled_tasks else 0
        total_reserved_tasks = sum(len(tasks) for tasks in reserved_tasks.values()) if reserved_tasks else 0
        
        # Check database connectivity
        db_healthy = True
        try:
            db = next(get_db())
            db.execute("SELECT 1")
            db.close()
        except Exception as e:
            db_healthy = False
            logger.error(f"Database health check failed: {str(e)}")
        
        # Check Redis connectivity
        redis_healthy = True
        try:
            celery_app.backend.get('health_check')
        except Exception as e:
            redis_healthy = False
            logger.error(f"Redis health check failed: {str(e)}")
        
        health_metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_healthy': db_healthy and redis_healthy and total_workers > 0,
            'database_healthy': db_healthy,
            'redis_healthy': redis_healthy,
            'worker_count': total_workers,
            'active_tasks': total_active_tasks,
            'scheduled_tasks': total_scheduled_tasks,
            'reserved_tasks': total_reserved_tasks,
            'total_pending_tasks': total_active_tasks + total_scheduled_tasks + total_reserved_tasks,
            'worker_stats': stats,
        }
        
        logger.info(f"Health check completed: System healthy = {health_metrics['system_healthy']}")
        
        return health_metrics
        
    except Exception as exc:
        logger.error(f"Health monitoring failed: {str(exc)}")
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system_healthy': False,
            'error': str(exc),
            'monitoring_failed': True
        }


@_celery_task_decorator(bind=True, max_retries=2)
def cleanup_expired_results(self) -> Dict[str, Any]:
    """
    Clean up expired task results and old data
    
    Returns:
        Cleanup summary
    """
    try:
        logger.info("Starting cleanup of expired results")
        
        cleanup_summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'expired_results_cleaned': 0,
            'old_documents_cleaned': 0,
            'database_records_cleaned': 0
        }
        
        # Clean up old task results (older than 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Note: Actual cleanup implementation would depend on the result backend
        # For Redis, we could iterate through keys and clean old ones
        # For database backends, we could clean old result records
        
        # Clean up old processed documents metadata
        db = next(get_db())
        try:
            # Clean up old document processing metadata (older than 7 days)
            old_cutoff = datetime.utcnow() - timedelta(days=7)
            
            # This is a placeholder - actual implementation would clean specific metadata
            # old_documents = db.query(Document).filter(
            #     Document.upload_timestamp < old_cutoff,
            #     Document.processed == True
            # ).all()
            
            # cleanup_summary['old_documents_cleaned'] = len(old_documents)
            
            db.commit()
        finally:
            db.close()
        
        logger.info(f"Cleanup completed: {cleanup_summary}")
        
        return cleanup_summary
        
    except Exception as exc:
        logger.error(f"Cleanup failed: {str(exc)}")
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'cleanup_failed': True,
            'error': str(exc)
        }


@_celery_task_decorator(bind=True, max_retries=2)
def aggregate_task_metrics(self) -> Dict[str, Any]:
    """
    Aggregate task execution metrics for monitoring and optimization
    
    Returns:
        Task metrics summary
    """
    try:
        logger.info("Aggregating task execution metrics")
        
        # Get task execution statistics
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        
        metrics_summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'worker_metrics': {},
            'queue_metrics': {},
            'task_type_metrics': {}
        }
        
        if stats:
            for worker_name, worker_stats in stats.items():
                metrics_summary['worker_metrics'][worker_name] = {
                    'total_tasks': worker_stats.get('total', {}),
                    'pool_processes': worker_stats.get('pool', {}).get('processes', 0),
                    'rusage': worker_stats.get('rusage', {}),
                    'clock': worker_stats.get('clock', 0)
                }
        
        # Get queue lengths (this would need to be implemented based on broker)
        # For Redis, we could check queue lengths
        # For RabbitMQ, we could use management API
        
        logger.info("Task metrics aggregation completed")
        
        return metrics_summary
        
    except Exception as exc:
        logger.error(f"Metrics aggregation failed: {str(exc)}")
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics_failed': True,
            'error': str(exc)
        }


@_celery_task_decorator(bind=True, max_retries=3)
def handle_failed_workflow(self, application_id: str, failed_task_id: str, error_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle failed workflow recovery and notification
    
    Args:
        application_id: UUID of the application
        failed_task_id: ID of the failed task
        error_info: Error information dictionary
        
    Returns:
        Recovery action results
    """
    try:
        logger.info(f"Handling failed workflow for application {application_id}, task {failed_task_id}")
        
        # Update application status
        db = next(get_db())
        try:
            application = db.query(Application).filter(Application.id == application_id).first()
            if application:
                application.status = "failed"
                application.updated_at = datetime.utcnow()
                
                # Store error information in metadata
                if not hasattr(application, 'metadata') or application.metadata is None:
                    application.metadata = {}
                
                application.metadata['workflow_failure'] = {
                    'failed_task_id': failed_task_id,
                    'error_info': error_info,
                    'failure_time': datetime.utcnow().isoformat(),
                    'retry_count': self.request.retries
                }
                
                db.commit()
        finally:
            db.close()
        
        # Determine recovery action based on error type
        recovery_action = "manual_review"
        
        if error_info.get('error_type') == 'temporary':
            recovery_action = "retry_workflow"
        elif error_info.get('error_type') == 'data_quality':
            recovery_action = "request_additional_data"
        elif error_info.get('error_type') == 'system_error':
            recovery_action = "system_maintenance_required"
        
        recovery_result = {
            'application_id': application_id,
            'failed_task_id': failed_task_id,
            'recovery_action': recovery_action,
            'error_info': error_info,
            'handled_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Workflow failure handled: {recovery_action} for application {application_id}")
        
        return recovery_result
        
    except Exception as exc:
        logger.error(f"Failed to handle workflow failure: {str(exc)}")
        
        return {
            'application_id': application_id,
            'failed_task_id': failed_task_id,
            'recovery_failed': True,
            'error': str(exc)
        }