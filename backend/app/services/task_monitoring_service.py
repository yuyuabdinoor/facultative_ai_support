"""
Task monitoring and management service for Celery orchestration

This service provides monitoring, metrics collection, and management
capabilities for the Celery task orchestration system.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from sqlalchemy.orm import Session

try:
    from celery import Celery
    from celery.result import AsyncResult, GroupResult
    from app.celery import celery_app
    CELERY_AVAILABLE = True
except ImportError:
    celery_app = None
    CELERY_AVAILABLE = False

from app.core.database import get_db
from app.models.database import Application, Document
from app.models.schemas import (
    TaskResult, WorkflowResult, OrchestrationResult, TaskMetrics,
    SystemHealthMetrics, WorkflowStatus, TaskStatus
)

logger = logging.getLogger(__name__)


class TaskMonitoringService:
    """Service for monitoring and managing Celery tasks and workflows"""
    
    def __init__(self):
        self.celery_app = celery_app
        self._task_history = defaultdict(list)
        self._workflow_history = defaultdict(list)
    
    def get_task_status(self, task_id: str) -> TaskResult:
        """
        Get the status of a specific task
        
        Args:
            task_id: Celery task ID
            
        Returns:
            TaskResult with current task status
        """
        try:
            if not CELERY_AVAILABLE or not self.celery_app:
                return TaskResult(
                    task_id=task_id,
                    task_name="unknown",
                    status=TaskStatus.FAILURE,
                    error="Celery not available"
                )
            
            result = AsyncResult(task_id, app=self.celery_app)
            
            # Determine task status
            if result.state == 'PENDING':
                status = TaskStatus.PENDING
            elif result.state == 'STARTED':
                status = TaskStatus.STARTED
            elif result.state == 'SUCCESS':
                status = TaskStatus.SUCCESS
            elif result.state == 'FAILURE':
                status = TaskStatus.FAILURE
            elif result.state == 'RETRY':
                status = TaskStatus.RETRY
            elif result.state == 'REVOKED':
                status = TaskStatus.REVOKED
            else:
                status = TaskStatus.PENDING
            
            # Get task info
            task_info = result.info if result.info else {}
            
            task_result = TaskResult(
                task_id=task_id,
                task_name=result.name or "unknown",
                status=status,
                result=task_info if status == TaskStatus.SUCCESS else None,
                error=str(task_info) if status == TaskStatus.FAILURE else None,
                retry_count=getattr(result, 'retries', 0)
            )
            
            return task_result
            
        except Exception as e:
            logger.error(f"Error getting task status for {task_id}: {str(e)}")
            return TaskResult(
                task_id=task_id,
                task_name="unknown",
                status=TaskStatus.FAILURE,
                error=str(e)
            )
    
    def get_workflow_status(self, workflow_id: str, application_id: str) -> WorkflowResult:
        """
        Get the status of a workflow (collection of related tasks)
        
        Args:
            workflow_id: Workflow identifier
            application_id: Application ID associated with workflow
            
        Returns:
            WorkflowResult with current workflow status
        """
        try:
            # In a real implementation, we'd track workflow-task relationships
            # For now, we'll simulate based on application status
            
            db = next(get_db())
            try:
                application = db.query(Application).filter(Application.id == application_id).first()
                
                if not application:
                    return WorkflowResult(
                        workflow_id=workflow_id,
                        workflow_type="unknown",
                        status=WorkflowStatus.FAILED,
                        application_id=application_id,
                        started_at=datetime.utcnow(),
                        error_summary="Application not found"
                    )
                
                # Map application status to workflow status
                if application.status == "pending":
                    workflow_status = WorkflowStatus.PENDING
                elif application.status == "processing":
                    workflow_status = WorkflowStatus.RUNNING
                elif application.status == "completed":
                    workflow_status = WorkflowStatus.COMPLETED
                elif application.status == "failed":
                    workflow_status = WorkflowStatus.FAILED
                else:
                    workflow_status = WorkflowStatus.PENDING
                
                workflow_result = WorkflowResult(
                    workflow_id=workflow_id,
                    workflow_type="document_processing",
                    status=workflow_status,
                    application_id=application_id,
                    started_at=application.created_at,
                    completed_at=application.updated_at if workflow_status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] else None
                )
                
                return workflow_result
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting workflow status for {workflow_id}: {str(e)}")
            return WorkflowResult(
                workflow_id=workflow_id,
                workflow_type="unknown",
                status=WorkflowStatus.FAILED,
                application_id=application_id,
                started_at=datetime.utcnow(),
                error_summary=str(e)
            )
    
    def get_system_health(self) -> SystemHealthMetrics:
        """
        Get current system health metrics
        
        Returns:
            SystemHealthMetrics with current system status
        """
        try:
            if not CELERY_AVAILABLE or not self.celery_app:
                return SystemHealthMetrics(
                    timestamp=datetime.utcnow(),
                    system_healthy=False,
                    database_healthy=False,
                    redis_healthy=False,
                    worker_count=0
                )
            
            # Get Celery inspection data
            inspect = self.celery_app.control.inspect()
            
            # Get worker stats
            stats = inspect.stats() or {}
            active_tasks = inspect.active() or {}
            scheduled_tasks = inspect.scheduled() or {}
            reserved_tasks = inspect.reserved() or {}
            
            # Calculate metrics
            worker_count = len(stats)
            total_active = sum(len(tasks) for tasks in active_tasks.values())
            total_scheduled = sum(len(tasks) for tasks in scheduled_tasks.values())
            total_reserved = sum(len(tasks) for tasks in reserved_tasks.values())
            
            # Check database health
            db_healthy = True
            try:
                db = next(get_db())
                db.execute("SELECT 1")
                db.close()
            except Exception:
                db_healthy = False
            
            # Check Redis health
            redis_healthy = True
            try:
                self.celery_app.backend.get('health_check')
            except Exception:
                redis_healthy = False
            
            system_healthy = db_healthy and redis_healthy and worker_count > 0
            
            return SystemHealthMetrics(
                timestamp=datetime.utcnow(),
                system_healthy=system_healthy,
                database_healthy=db_healthy,
                redis_healthy=redis_healthy,
                worker_count=worker_count,
                active_tasks=total_active,
                scheduled_tasks=total_scheduled,
                reserved_tasks=total_reserved,
                total_pending_tasks=total_active + total_scheduled + total_reserved,
                worker_stats=stats
            )
            
        except Exception as e:
            logger.error(f"Error getting system health: {str(e)}")
            return SystemHealthMetrics(
                timestamp=datetime.utcnow(),
                system_healthy=False,
                database_healthy=False,
                redis_healthy=False,
                worker_count=0
            )
    
    def get_task_metrics(self, task_name: Optional[str] = None, 
                        time_window_hours: int = 24) -> List[TaskMetrics]:
        """
        Get task execution metrics
        
        Args:
            task_name: Specific task name to get metrics for (None for all)
            time_window_hours: Time window for metrics calculation
            
        Returns:
            List of TaskMetrics for requested tasks
        """
        try:
            # In a production system, this would query a metrics database
            # For now, we'll return simulated metrics
            
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Simulate some common task metrics
            common_tasks = [
                "app.agents.tasks.process_document_ocr",
                "app.agents.tasks.process_pdf_ocr",
                "app.agents.tasks.process_email_parsing",
                "app.agents.orchestration_tasks.orchestrate_document_processing_workflow",
                "app.agents.workflow_tasks.extract_risk_data_workflow",
                "app.agents.workflow_tasks.risk_analysis_workflow"
            ]
            
            metrics = []
            
            for task in common_tasks:
                if task_name and task != task_name:
                    continue
                
                # Simulate metrics (in production, these would come from actual data)
                total_executions = 50
                successful_executions = 45
                failed_executions = 5
                
                metrics.append(TaskMetrics(
                    task_name=task,
                    total_executions=total_executions,
                    successful_executions=successful_executions,
                    failed_executions=failed_executions,
                    average_execution_time=120.5,
                    min_execution_time=30.0,
                    max_execution_time=300.0,
                    success_rate=successful_executions / total_executions if total_executions > 0 else 0.0,
                    last_execution=datetime.utcnow() - timedelta(minutes=15)
                ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting task metrics: {str(e)}")
            return []
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task
        
        Args:
            task_id: Celery task ID to cancel
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        try:
            if not CELERY_AVAILABLE or not self.celery_app:
                logger.warning("Celery not available, cannot cancel task")
                return False
            
            self.celery_app.control.revoke(task_id, terminate=True)
            logger.info(f"Task {task_id} cancellation requested")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {str(e)}")
            return False
    
    def cancel_workflow(self, workflow_id: str, application_id: str) -> bool:
        """
        Cancel all tasks in a workflow
        
        Args:
            workflow_id: Workflow identifier
            application_id: Application ID associated with workflow
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        try:
            # In a real implementation, we'd track workflow-task relationships
            # and cancel all associated tasks
            
            # Update application status to cancelled
            db = next(get_db())
            try:
                application = db.query(Application).filter(Application.id == application_id).first()
                if application:
                    application.status = "cancelled"
                    application.updated_at = datetime.utcnow()
                    db.commit()
                    
                    logger.info(f"Workflow {workflow_id} for application {application_id} cancelled")
                    return True
                else:
                    logger.warning(f"Application {application_id} not found for workflow cancellation")
                    return False
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error cancelling workflow {workflow_id}: {str(e)}")
            return False
    
    def retry_failed_task(self, task_id: str) -> Optional[str]:
        """
        Retry a failed task
        
        Args:
            task_id: Celery task ID to retry
            
        Returns:
            New task ID if retry was successful, None otherwise
        """
        try:
            if not CELERY_AVAILABLE or not self.celery_app:
                logger.warning("Celery not available, cannot retry task")
                return None
            
            # Get the original task result
            result = AsyncResult(task_id, app=self.celery_app)
            
            if result.state != 'FAILURE':
                logger.warning(f"Task {task_id} is not in FAILURE state, cannot retry")
                return None
            
            # In a real implementation, we'd need to store task arguments
            # and recreate the task with the same parameters
            logger.info(f"Task {task_id} retry requested (implementation needed)")
            
            # This is a placeholder - actual implementation would depend on
            # how we store task metadata and arguments
            return None
            
        except Exception as e:
            logger.error(f"Error retrying task {task_id}: {str(e)}")
            return None
    
    def get_active_workflows(self) -> List[WorkflowResult]:
        """
        Get all currently active workflows
        
        Returns:
            List of active WorkflowResult objects
        """
        try:
            db = next(get_db())
            try:
                # Get applications that are currently processing
                active_applications = db.query(Application).filter(
                    Application.status.in_(["pending", "processing"])
                ).all()
                
                workflows = []
                for app in active_applications:
                    workflow = WorkflowResult(
                        workflow_id=f"workflow_{app.id}",
                        workflow_type="document_processing",
                        status=WorkflowStatus.RUNNING if app.status == "processing" else WorkflowStatus.PENDING,
                        application_id=app.id,
                        started_at=app.created_at
                    )
                    workflows.append(workflow)
                
                return workflows
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting active workflows: {str(e)}")
            return []
    
    def get_failed_tasks(self, time_window_hours: int = 24) -> List[TaskResult]:
        """
        Get all failed tasks within a time window
        
        Args:
            time_window_hours: Time window to look for failed tasks
            
        Returns:
            List of failed TaskResult objects
        """
        try:
            # In a production system, this would query task execution logs
            # For now, we'll return a simulated list
            
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Simulate some failed tasks
            failed_tasks = [
                TaskResult(
                    task_id="failed_task_1",
                    task_name="app.agents.tasks.process_document_ocr",
                    status=TaskStatus.FAILURE,
                    error="File not found",
                    started_at=datetime.utcnow() - timedelta(hours=2),
                    completed_at=datetime.utcnow() - timedelta(hours=2),
                    retry_count=3
                ),
                TaskResult(
                    task_id="failed_task_2",
                    task_name="app.agents.workflow_tasks.risk_analysis_workflow",
                    status=TaskStatus.FAILURE,
                    error="Insufficient data for analysis",
                    started_at=datetime.utcnow() - timedelta(hours=1),
                    completed_at=datetime.utcnow() - timedelta(hours=1),
                    retry_count=2
                )
            ]
            
            return failed_tasks
            
        except Exception as e:
            logger.error(f"Error getting failed tasks: {str(e)}")
            return []
    
    def purge_queue(self, queue_name: str) -> int:
        """
        Purge all tasks from a specific queue
        
        Args:
            queue_name: Name of the queue to purge
            
        Returns:
            Number of tasks purged
        """
        try:
            if not CELERY_AVAILABLE or not self.celery_app:
                logger.warning("Celery not available, cannot purge queue")
                return 0
            
            # Purge the queue
            purged_count = self.celery_app.control.purge()
            logger.info(f"Purged {purged_count} tasks from queue {queue_name}")
            return purged_count or 0
            
        except Exception as e:
            logger.error(f"Error purging queue {queue_name}: {str(e)}")
            return 0


# Global instance
task_monitoring_service = TaskMonitoringService()