"""
FastAPI endpoints for task monitoring and orchestration management

Provides REST API endpoints for monitoring Celery tasks, workflows,
and system health metrics.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from fastapi.responses import JSONResponse

from app.services.task_monitoring_service import task_monitoring_service
from app.models.schemas import (
    TaskResult, WorkflowResult, OrchestrationResult, SystemHealthMetrics,
    TaskMetrics, WorkflowStatus, TaskStatus, BatchProcessingRequest,
    BatchProcessingResult
)
from app.agents.orchestration_tasks import (
    orchestrate_document_processing_workflow,
    aggregate_task_results,
    handle_failed_workflow
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/task-monitoring", tags=["Task Monitoring"])


@router.get("/health", response_model=SystemHealthMetrics)
async def get_system_health():
    """
    Get current system health metrics
    
    Returns:
        SystemHealthMetrics: Current system health status
    """
    try:
        health_metrics = task_monitoring_service.get_system_health()
        return health_metrics
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


@router.get("/tasks/{task_id}", response_model=TaskResult)
async def get_task_status(
    task_id: str = Path(..., description="Celery task ID")
):
    """
    Get the status of a specific task
    
    Args:
        task_id: Celery task ID
        
    Returns:
        TaskResult: Current task status and details
    """
    try:
        task_result = task_monitoring_service.get_task_status(task_id)
        return task_result
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.get("/workflows/{workflow_id}", response_model=WorkflowResult)
async def get_workflow_status(
    workflow_id: str = Path(..., description="Workflow identifier"),
    application_id: str = Query(..., description="Application ID associated with workflow")
):
    """
    Get the status of a workflow
    
    Args:
        workflow_id: Workflow identifier
        application_id: Application ID associated with workflow
        
    Returns:
        WorkflowResult: Current workflow status and details
    """
    try:
        workflow_result = task_monitoring_service.get_workflow_status(workflow_id, application_id)
        return workflow_result
    except Exception as e:
        logger.error(f"Error getting workflow status for {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")


@router.get("/workflows/active", response_model=List[WorkflowResult])
async def get_active_workflows():
    """
    Get all currently active workflows
    
    Returns:
        List[WorkflowResult]: List of active workflows
    """
    try:
        active_workflows = task_monitoring_service.get_active_workflows()
        return active_workflows
    except Exception as e:
        logger.error(f"Error getting active workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active workflows: {str(e)}")


@router.get("/tasks/failed", response_model=List[TaskResult])
async def get_failed_tasks(
    time_window_hours: int = Query(24, ge=1, le=168, description="Time window in hours (1-168)")
):
    """
    Get all failed tasks within a time window
    
    Args:
        time_window_hours: Time window to look for failed tasks (1-168 hours)
        
    Returns:
        List[TaskResult]: List of failed tasks
    """
    try:
        failed_tasks = task_monitoring_service.get_failed_tasks(time_window_hours)
        return failed_tasks
    except Exception as e:
        logger.error(f"Error getting failed tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get failed tasks: {str(e)}")


@router.get("/metrics/tasks", response_model=List[TaskMetrics])
async def get_task_metrics(
    task_name: Optional[str] = Query(None, description="Specific task name (optional)"),
    time_window_hours: int = Query(24, ge=1, le=168, description="Time window in hours (1-168)")
):
    """
    Get task execution metrics
    
    Args:
        task_name: Specific task name to get metrics for (None for all tasks)
        time_window_hours: Time window for metrics calculation (1-168 hours)
        
    Returns:
        List[TaskMetrics]: Task execution metrics
    """
    try:
        metrics = task_monitoring_service.get_task_metrics(task_name, time_window_hours)
        return metrics
    except Exception as e:
        logger.error(f"Error getting task metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get task metrics: {str(e)}")


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str = Path(..., description="Celery task ID to cancel")
):
    """
    Cancel a running task
    
    Args:
        task_id: Celery task ID to cancel
        
    Returns:
        Success message or error
    """
    try:
        success = task_monitoring_service.cancel_task(task_id)
        
        if success:
            return JSONResponse(
                status_code=200,
                content={"message": f"Task {task_id} cancellation requested", "success": True}
            )
        else:
            raise HTTPException(status_code=400, detail=f"Failed to cancel task {task_id}")
            
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


@router.post("/workflows/{workflow_id}/cancel")
async def cancel_workflow(
    workflow_id: str = Path(..., description="Workflow identifier"),
    application_id: str = Query(..., description="Application ID associated with workflow")
):
    """
    Cancel all tasks in a workflow
    
    Args:
        workflow_id: Workflow identifier
        application_id: Application ID associated with workflow
        
    Returns:
        Success message or error
    """
    try:
        success = task_monitoring_service.cancel_workflow(workflow_id, application_id)
        
        if success:
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"Workflow {workflow_id} cancellation requested",
                    "success": True
                }
            )
        else:
            raise HTTPException(status_code=400, detail=f"Failed to cancel workflow {workflow_id}")
            
    except Exception as e:
        logger.error(f"Error cancelling workflow {workflow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel workflow: {str(e)}")


@router.post("/tasks/{task_id}/retry")
async def retry_failed_task(
    task_id: str = Path(..., description="Celery task ID to retry")
):
    """
    Retry a failed task
    
    Args:
        task_id: Celery task ID to retry
        
    Returns:
        New task ID or error message
    """
    try:
        new_task_id = task_monitoring_service.retry_failed_task(task_id)
        
        if new_task_id:
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"Task {task_id} retry initiated",
                    "new_task_id": new_task_id,
                    "success": True
                }
            )
        else:
            raise HTTPException(status_code=400, detail=f"Failed to retry task {task_id}")
            
    except Exception as e:
        logger.error(f"Error retrying task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retry task: {str(e)}")


@router.post("/workflows/start")
async def start_document_processing_workflow(
    application_id: str = Query(..., description="Application ID to process"),
    document_ids: List[str] = Query(..., description="List of document IDs to process")
):
    """
    Start a document processing workflow for an application
    
    Args:
        application_id: Application ID to process
        document_ids: List of document IDs to process
        
    Returns:
        Workflow initiation result
    """
    try:
        # Validate inputs
        if not application_id:
            raise HTTPException(status_code=400, detail="Application ID is required")
        
        if not document_ids:
            raise HTTPException(status_code=400, detail="At least one document ID is required")
        
        # Start the workflow (this would typically be done asynchronously)
        # For now, we'll return a success message with workflow ID
        workflow_id = f"workflow_{application_id}_{datetime.utcnow().timestamp()}"
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "Document processing workflow started",
                "workflow_id": workflow_id,
                "application_id": application_id,
                "document_ids": document_ids,
                "status": "initiated"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting workflow for application {application_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")


@router.post("/batch/process", response_model=BatchProcessingResult)
async def start_batch_processing(
    request: BatchProcessingRequest
):
    """
    Start batch processing for multiple applications
    
    Args:
        request: Batch processing request with application IDs and configuration
        
    Returns:
        BatchProcessingResult: Batch processing initiation result
    """
    try:
        # Validate request
        if not request.application_ids:
            raise HTTPException(status_code=400, detail="At least one application ID is required")
        
        if len(request.application_ids) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 applications per batch")
        
        # Generate batch ID
        batch_id = f"batch_{datetime.utcnow().timestamp()}"
        
        # In a real implementation, this would start multiple workflows
        # For now, we'll return a simulated result
        batch_result = BatchProcessingResult(
            batch_id=batch_id,
            total_applications=len(request.application_ids),
            completed_applications=0,
            failed_applications=0,
            started_at=datetime.utcnow(),
            overall_success_rate=0.0
        )
        
        return batch_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start batch processing: {str(e)}")


@router.get("/batch/{batch_id}", response_model=BatchProcessingResult)
async def get_batch_status(
    batch_id: str = Path(..., description="Batch processing ID")
):
    """
    Get the status of a batch processing operation
    
    Args:
        batch_id: Batch processing ID
        
    Returns:
        BatchProcessingResult: Current batch processing status
    """
    try:
        # In a real implementation, this would query batch status from database/cache
        # For now, we'll return a simulated result
        
        batch_result = BatchProcessingResult(
            batch_id=batch_id,
            total_applications=10,
            completed_applications=7,
            failed_applications=1,
            started_at=datetime.utcnow() - timedelta(hours=1),
            completed_at=datetime.utcnow() - timedelta(minutes=10),
            overall_success_rate=0.7
        )
        
        return batch_result
        
    except Exception as e:
        logger.error(f"Error getting batch status for {batch_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get batch status: {str(e)}")


@router.delete("/queues/{queue_name}/purge")
async def purge_queue(
    queue_name: str = Path(..., description="Queue name to purge")
):
    """
    Purge all tasks from a specific queue
    
    Args:
        queue_name: Name of the queue to purge
        
    Returns:
        Number of tasks purged
    """
    try:
        # Validate queue name
        valid_queues = [
            'default', 'ocr_processing', 'document_processing',
            'batch_processing', 'orchestration', 'workflow', 'priority'
        ]
        
        if queue_name not in valid_queues:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid queue name. Valid queues: {', '.join(valid_queues)}"
            )
        
        purged_count = task_monitoring_service.purge_queue(queue_name)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Queue {queue_name} purged successfully",
                "purged_tasks": purged_count,
                "queue_name": queue_name
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error purging queue {queue_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to purge queue: {str(e)}")


@router.get("/statistics/summary")
async def get_system_statistics():
    """
    Get comprehensive system statistics and metrics
    
    Returns:
        System statistics summary
    """
    try:
        # Get various metrics
        health_metrics = task_monitoring_service.get_system_health()
        task_metrics = task_monitoring_service.get_task_metrics(time_window_hours=24)
        active_workflows = task_monitoring_service.get_active_workflows()
        failed_tasks = task_monitoring_service.get_failed_tasks(time_window_hours=24)
        
        # Calculate summary statistics
        total_task_executions = sum(metric.total_executions for metric in task_metrics)
        total_successful_executions = sum(metric.successful_executions for metric in task_metrics)
        overall_success_rate = (
            total_successful_executions / total_task_executions
            if total_task_executions > 0 else 0.0
        )
        
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": {
                "healthy": health_metrics.system_healthy,
                "worker_count": health_metrics.worker_count,
                "active_tasks": health_metrics.active_tasks,
                "pending_tasks": health_metrics.total_pending_tasks
            },
            "task_statistics": {
                "total_executions_24h": total_task_executions,
                "successful_executions_24h": total_successful_executions,
                "failed_executions_24h": total_task_executions - total_successful_executions,
                "overall_success_rate": overall_success_rate,
                "unique_task_types": len(task_metrics)
            },
            "workflow_statistics": {
                "active_workflows": len(active_workflows),
                "failed_tasks_24h": len(failed_tasks)
            },
            "performance_metrics": {
                "average_execution_time": (
                    sum(metric.average_execution_time for metric in task_metrics) / len(task_metrics)
                    if task_metrics else 0.0
                ),
                "fastest_task_time": (
                    min(metric.min_execution_time for metric in task_metrics)
                    if task_metrics else 0.0
                ),
                "slowest_task_time": (
                    max(metric.max_execution_time for metric in task_metrics)
                    if task_metrics else 0.0
                )
            }
        }
        
        return JSONResponse(status_code=200, content=summary)
        
    except Exception as e:
        logger.error(f"Error getting system statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system statistics: {str(e)}")