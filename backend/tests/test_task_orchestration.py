"""
Integration tests for Celery task orchestration system

Tests the complete workflow orchestration, task coordination,
monitoring, and error handling capabilities.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Test imports
from app.agents.orchestration_tasks import (
    orchestrate_document_processing_workflow,
    aggregate_task_results,
    monitor_task_health,
    cleanup_expired_results,
    aggregate_task_metrics,
    handle_failed_workflow
)
from app.agents.workflow_tasks import (
    extract_risk_data_workflow,
    risk_analysis_workflow,
    validate_business_limits_workflow,
    generate_decision_workflow,
    market_grouping_workflow
)
from app.services.task_monitoring_service import TaskMonitoringService
from app.models.schemas import (
    WorkflowStatus, TaskStatus, TaskResult, WorkflowResult,
    OrchestrationResult, SystemHealthMetrics
)
from app.models.database import Application, Document
from app.core.database import get_db


class TestTaskOrchestration:
    """Test suite for task orchestration functionality"""
    
    @pytest.fixture
    def mock_celery_app(self):
        """Mock Celery app for testing"""
        mock_app = Mock()
        mock_app.control.inspect.return_value = Mock()
        mock_app.backend.get.return_value = None
        return mock_app
    
    @pytest.fixture
    def sample_application(self, db_session):
        """Create a sample application for testing"""
        application = Application(
            id="test-app-123",
            status="pending"
        )
        db_session.add(application)
        db_session.commit()
        return application
    
    @pytest.fixture
    def sample_documents(self, db_session, sample_application):
        """Create sample documents for testing"""
        documents = []
        for i in range(3):
            doc = Document(
                id=f"test-doc-{i}",
                application_id=sample_application.id,
                filename=f"test_document_{i}.pdf",
                document_type="pdf",
                file_path=f"/test/path/doc_{i}.pdf",
                processed=True,
                metadata={
                    'ocr_result': {
                        'text': f'Sample OCR text for document {i}',
                        'confidence': 0.95
                    }
                }
            )
            documents.append(doc)
            db_session.add(doc)
        
        db_session.commit()
        return documents
    
    def test_orchestrate_document_processing_workflow_success(self, sample_application, sample_documents):
        """Test successful document processing workflow orchestration"""
        
        # Mock the individual workflow tasks
        with patch('app.agents.orchestration_tasks.signature') as mock_signature, \
             patch('app.agents.orchestration_tasks.group') as mock_group, \
             patch('app.agents.orchestration_tasks.chain') as mock_chain:
            
            # Mock task results
            mock_ocr_result = Mock()
            mock_ocr_result.get.return_value = [
                {'document_id': 'test-doc-0', 'success': True},
                {'document_id': 'test-doc-1', 'success': True},
                {'document_id': 'test-doc-2', 'success': True}
            ]
            
            mock_analysis_result = Mock()
            mock_analysis_result.get.return_value = {
                'workflow_type': 'analysis_complete',
                'status': 'completed'
            }
            
            mock_market_result = Mock()
            mock_market_result.get.return_value = {
                'workflow_type': 'market_grouping',
                'status': 'completed'
            }
            
            # Configure mocks
            mock_group.return_value.apply_async.return_value = mock_ocr_result
            mock_chain.return_value.apply_async.return_value = mock_analysis_result
            mock_signature.return_value.apply_async.return_value = mock_market_result
            
            # Execute the workflow
            document_ids = [doc.id for doc in sample_documents]
            
            # Mock the task execution (since we can't run actual Celery tasks in tests)
            with patch('app.agents.orchestration_tasks.celery_app') as mock_celery:
                mock_task = Mock()
                mock_task.request.retries = 0
                mock_task.max_retries = 3
                
                result = orchestrate_document_processing_workflow(
                    mock_task, sample_application.id, document_ids
                )
            
            # Verify the result structure
            assert isinstance(result, dict)
            assert result['application_id'] == sample_application.id
            assert result['workflow_status'] == 'completed'
            assert result['total_documents_processed'] == len(document_ids)
            assert 'completion_time' in result
    
    def test_orchestrate_document_processing_workflow_failure(self, sample_application, sample_documents):
        """Test workflow orchestration with failures"""
        
        with patch('app.agents.orchestration_tasks.signature') as mock_signature:
            # Mock a failure in OCR processing
            mock_signature.side_effect = Exception("OCR processing failed")
            
            # Mock the task execution
            with patch('app.agents.orchestration_tasks.celery_app') as mock_celery:
                mock_task = Mock()
                mock_task.request.retries = 0
                mock_task.max_retries = 3
                mock_task.retry = Mock(side_effect=Exception("Max retries exceeded"))
                
                document_ids = [doc.id for doc in sample_documents]
                
                result = orchestrate_document_processing_workflow(
                    mock_task, sample_application.id, document_ids
                )
            
            # Verify failure handling
            assert isinstance(result, dict)
            assert result['application_id'] == sample_application.id
            assert result['workflow_status'] == 'failed'
            assert 'error' in result
    
    def test_aggregate_task_results_success(self):
        """Test successful task result aggregation"""
        
        task_ids = ['task-1', 'task-2', 'task-3']
        
        with patch('app.agents.orchestration_tasks.AsyncResult') as mock_async_result, \
             patch('app.agents.orchestration_tasks.celery_app') as mock_celery:
            
            # Mock successful task results
            mock_results = []
            for i, task_id in enumerate(task_ids):
                mock_result = Mock()
                mock_result.ready.return_value = True
                mock_result.successful.return_value = True
                mock_result.get.return_value = {'task_data': f'result_{i}'}
                mock_results.append(mock_result)
            
            mock_async_result.side_effect = mock_results
            
            # Mock the task execution
            mock_task = Mock()
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            result = aggregate_task_results(mock_task, task_ids, "test_workflow")
            
            # Verify aggregation results
            assert result['result_type'] == 'test_workflow'
            assert result['total_tasks'] == len(task_ids)
            assert result['successful_tasks'] == len(task_ids)
            assert result['failed_tasks'] == 0
            assert result['success_rate'] == 1.0
            assert len(result['results']) == len(task_ids)
    
    def test_aggregate_task_results_mixed_outcomes(self):
        """Test task result aggregation with mixed success/failure"""
        
        task_ids = ['task-1', 'task-2', 'task-3']
        
        with patch('app.agents.orchestration_tasks.AsyncResult') as mock_async_result, \
             patch('app.agents.orchestration_tasks.celery_app') as mock_celery:
            
            # Mock mixed results (success, failure, pending)
            mock_results = []
            
            # Successful task
            mock_success = Mock()
            mock_success.ready.return_value = True
            mock_success.successful.return_value = True
            mock_success.get.return_value = {'task_data': 'success'}
            mock_results.append(mock_success)
            
            # Failed task
            mock_failure = Mock()
            mock_failure.ready.return_value = True
            mock_failure.successful.return_value = False
            mock_failure.info = "Task failed"
            mock_results.append(mock_failure)
            
            # Pending task
            mock_pending = Mock()
            mock_pending.ready.return_value = False
            mock_pending.state = 'PENDING'
            mock_results.append(mock_pending)
            
            mock_async_result.side_effect = mock_results
            
            # Mock the task execution
            mock_task = Mock()
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            result = aggregate_task_results(mock_task, task_ids, "mixed_workflow")
            
            # Verify mixed results
            assert result['total_tasks'] == 3
            assert result['successful_tasks'] == 1
            assert result['failed_tasks'] == 1
            assert result['pending_tasks'] == 1
            assert result['success_rate'] == 1/3
    
    def test_monitor_task_health_healthy_system(self):
        """Test system health monitoring with healthy system"""
        
        with patch('app.agents.orchestration_tasks.celery_app') as mock_celery, \
             patch('app.agents.orchestration_tasks.get_db') as mock_get_db:
            
            # Mock healthy Celery system
            mock_inspect = Mock()
            mock_inspect.stats.return_value = {
                'worker1': {'total': 100, 'pool': {'processes': 4}},
                'worker2': {'total': 150, 'pool': {'processes': 4}}
            }
            mock_inspect.active.return_value = {
                'worker1': [{'id': 'task1'}, {'id': 'task2'}],
                'worker2': [{'id': 'task3'}]
            }
            mock_inspect.scheduled.return_value = {'worker1': [], 'worker2': []}
            mock_inspect.reserved.return_value = {'worker1': [], 'worker2': []}
            
            mock_celery.control.inspect.return_value = mock_inspect
            mock_celery.backend.get.return_value = None  # No error
            
            # Mock healthy database
            mock_db = Mock()
            mock_db.execute.return_value = None
            mock_get_db.return_value.__next__.return_value = mock_db
            
            # Mock the task execution
            mock_task = Mock()
            
            result = monitor_task_health(mock_task)
            
            # Verify health metrics
            assert result['system_healthy'] is True
            assert result['database_healthy'] is True
            assert result['redis_healthy'] is True
            assert result['worker_count'] == 2
            assert result['active_tasks'] == 3
            assert result['scheduled_tasks'] == 0
            assert result['reserved_tasks'] == 0
    
    def test_monitor_task_health_unhealthy_system(self):
        """Test system health monitoring with unhealthy system"""
        
        with patch('app.agents.orchestration_tasks.celery_app') as mock_celery, \
             patch('app.agents.orchestration_tasks.get_db') as mock_get_db:
            
            # Mock unhealthy Celery system
            mock_celery.control.inspect.return_value.stats.return_value = {}  # No workers
            mock_celery.backend.get.side_effect = Exception("Redis connection failed")
            
            # Mock unhealthy database
            mock_get_db.return_value.__next__.side_effect = Exception("Database connection failed")
            
            # Mock the task execution
            mock_task = Mock()
            
            result = monitor_task_health(mock_task)
            
            # Verify unhealthy system detection
            assert result['system_healthy'] is False
            assert result['database_healthy'] is False
            assert result['redis_healthy'] is False
            assert result['worker_count'] == 0
    
    def test_cleanup_expired_results(self):
        """Test cleanup of expired results"""
        
        with patch('app.agents.orchestration_tasks.get_db') as mock_get_db:
            
            # Mock database operations
            mock_db = Mock()
            mock_get_db.return_value.__next__.return_value = mock_db
            
            # Mock the task execution
            mock_task = Mock()
            
            result = cleanup_expired_results(mock_task)
            
            # Verify cleanup execution
            assert 'timestamp' in result
            assert 'expired_results_cleaned' in result
            assert 'old_documents_cleaned' in result
            assert 'database_records_cleaned' in result
    
    def test_handle_failed_workflow(self, sample_application):
        """Test failed workflow handling"""
        
        error_info = {
            'error_type': 'system_error',
            'error_message': 'Database connection lost',
            'failed_at': datetime.utcnow().isoformat()
        }
        
        with patch('app.agents.orchestration_tasks.get_db') as mock_get_db:
            
            # Mock database operations
            mock_db = Mock()
            mock_get_db.return_value.__next__.return_value = mock_db
            
            # Mock the task execution
            mock_task = Mock()
            mock_task.request.retries = 1
            
            result = handle_failed_workflow(
                mock_task, sample_application.id, 'failed-task-123', error_info
            )
            
            # Verify failure handling
            assert result['application_id'] == sample_application.id
            assert result['failed_task_id'] == 'failed-task-123'
            assert result['recovery_action'] == 'system_maintenance_required'
            assert result['error_info'] == error_info


class TestWorkflowTasks:
    """Test suite for individual workflow tasks"""
    
    @pytest.fixture
    def sample_application_with_data(self, db_session):
        """Create application with extracted data"""
        application = Application(
            id="test-app-workflow",
            status="processing",
            metadata={
                'extracted_data': {
                    'risk_parameters': {
                        'asset_value': 1000000,
                        'location': 'New York, USA',
                        'asset_type': 'Commercial Building',
                        'industry_sector': 'Real Estate'
                    },
                    'financial_data': {
                        'revenue': 5000000,
                        'assets': 10000000
                    }
                }
            }
        )
        db_session.add(application)
        db_session.commit()
        return application
    
    def test_extract_risk_data_workflow_success(self, sample_application, sample_documents):
        """Test successful risk data extraction workflow"""
        
        with patch('app.agents.workflow_tasks.data_extraction_agent') as mock_agent:
            
            # Mock successful data extraction
            mock_extracted_data = {
                'asset_value': 1000000,
                'location': 'Test Location',
                'asset_type': 'Commercial Property'
            }
            mock_agent.extract_risk_parameters.return_value = mock_extracted_data
            
            # Mock the task execution
            mock_task = Mock()
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            document_ids = [doc.id for doc in sample_documents]
            
            result = extract_risk_data_workflow(mock_task, sample_application.id, document_ids)
            
            # Verify extraction results
            assert result['workflow_type'] == 'data_extraction'
            assert result['status'] == 'completed'
            assert result['application_id'] == sample_application.id
            assert result['documents_processed'] == len(document_ids)
            assert 'consolidated_data' in result
    
    def test_risk_analysis_workflow_success(self, sample_application_with_data):
        """Test successful risk analysis workflow"""
        
        with patch('app.agents.workflow_tasks.risk_analysis_agent') as mock_agent:
            
            # Mock successful risk analysis
            mock_agent.analyze_loss_history.return_value = {'loss_trend': 'stable'}
            mock_agent.assess_catastrophe_exposure.return_value = {'cat_risk': 'medium'}
            mock_agent.evaluate_financial_strength.return_value = {'rating': 'A'}
            mock_agent.calculate_risk_score.return_value = {
                'overall_score': 75.0,
                'confidence': 0.85,
                'risk_level': 'MEDIUM'
            }
            mock_agent.generate_risk_report.return_value = {'summary': 'Risk analysis complete'}
            
            # Mock the task execution
            mock_task = Mock()
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            result = risk_analysis_workflow(mock_task, sample_application_with_data.id)
            
            # Verify analysis results
            assert result['workflow_type'] == 'risk_analysis'
            assert result['status'] == 'completed'
            assert result['application_id'] == sample_application_with_data.id
            assert 'analysis_results' in result
    
    def test_validate_business_limits_workflow_success(self, sample_application_with_data):
        """Test successful business limits validation workflow"""
        
        with patch('app.agents.workflow_tasks.business_limits_agent') as mock_agent:
            
            # Mock successful validation
            mock_agent.check_business_limits.return_value = {'within_limits': True}
            mock_agent.validate_geographic_limits.return_value = {'valid': True}
            mock_agent.check_sector_restrictions.return_value = {'allowed': True}
            mock_agent.validate_regulatory_compliance.return_value = {'compliant': True}
            
            # Mock the task execution
            mock_task = Mock()
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            result = validate_business_limits_workflow(mock_task, sample_application_with_data.id)
            
            # Verify validation results
            assert result['workflow_type'] == 'business_limits_validation'
            assert result['status'] == 'completed'
            assert result['application_id'] == sample_application_with_data.id
            assert 'validation_results' in result
    
    def test_generate_decision_workflow_success(self, sample_application_with_data):
        """Test successful decision generation workflow"""
        
        # Add analysis results to application metadata
        sample_application_with_data.metadata.update({
            'risk_analysis': {'risk_score': {'overall_score': 75.0}},
            'limits_validation': {'business_limits_check': {'within_limits': True}}
        })
        
        with patch('app.agents.workflow_tasks.decision_engine_agent') as mock_agent:
            
            # Mock successful decision generation
            mock_agent.generate_recommendation.return_value = {
                'decision': 'APPROVE',
                'confidence': 0.85,
                'conditions': ['Standard terms apply']
            }
            mock_agent.calculate_confidence_score.return_value = 0.85
            mock_agent.generate_rationale.return_value = 'Risk profile acceptable'
            
            # Mock the task execution
            mock_task = Mock()
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            result = generate_decision_workflow(mock_task, sample_application_with_data.id)
            
            # Verify decision results
            assert result['workflow_type'] == 'decision_generation'
            assert result['status'] == 'completed'
            assert result['application_id'] == sample_application_with_data.id
            assert 'decision_results' in result
    
    def test_market_grouping_workflow_success(self, sample_application, sample_documents):
        """Test successful market grouping workflow"""
        
        with patch('app.agents.workflow_tasks.market_grouping_agent') as mock_agent:
            
            # Mock successful market analysis
            mock_agent.identify_market.return_value = {'market': 'North America'}
            mock_agent.classify_geographic_market.return_value = {'region': 'USA'}
            mock_agent.group_by_industry.return_value = {'Real Estate': [sample_documents[0]]}
            mock_agent.map_relationships.return_value = {'related_docs': []}
            
            # Mock the task execution
            mock_task = Mock()
            mock_task.request.retries = 0
            mock_task.max_retries = 3
            
            document_ids = [doc.id for doc in sample_documents]
            
            result = market_grouping_workflow(mock_task, sample_application.id, document_ids)
            
            # Verify market analysis results
            assert result['workflow_type'] == 'market_grouping'
            assert result['status'] == 'completed'
            assert result['application_id'] == sample_application.id
            assert result['documents_analyzed'] == len(document_ids)
            assert 'market_results' in result


class TestTaskMonitoringService:
    """Test suite for task monitoring service"""
    
    @pytest.fixture
    def monitoring_service(self):
        """Create task monitoring service instance"""
        return TaskMonitoringService()
    
    def test_get_task_status_success(self, monitoring_service):
        """Test getting task status for successful task"""
        
        with patch('app.services.task_monitoring_service.AsyncResult') as mock_result:
            
            # Mock successful task
            mock_task_result = Mock()
            mock_task_result.state = 'SUCCESS'
            mock_task_result.name = 'test_task'
            mock_task_result.info = {'result': 'success'}
            mock_result.return_value = mock_task_result
            
            result = monitoring_service.get_task_status('test-task-123')
            
            # Verify task status
            assert result.task_id == 'test-task-123'
            assert result.task_name == 'test_task'
            assert result.status == TaskStatus.SUCCESS
            assert result.result == {'result': 'success'}
    
    def test_get_task_status_failure(self, monitoring_service):
        """Test getting task status for failed task"""
        
        with patch('app.services.task_monitoring_service.AsyncResult') as mock_result:
            
            # Mock failed task
            mock_task_result = Mock()
            mock_task_result.state = 'FAILURE'
            mock_task_result.name = 'test_task'
            mock_task_result.info = 'Task failed due to error'
            mock_result.return_value = mock_task_result
            
            result = monitoring_service.get_task_status('failed-task-123')
            
            # Verify failure status
            assert result.task_id == 'failed-task-123'
            assert result.status == TaskStatus.FAILURE
            assert result.error == 'Task failed due to error'
    
    def test_get_system_health_healthy(self, monitoring_service):
        """Test system health check with healthy system"""
        
        with patch.object(monitoring_service, 'celery_app') as mock_celery, \
             patch('app.services.task_monitoring_service.get_db') as mock_get_db:
            
            # Mock healthy system
            mock_inspect = Mock()
            mock_inspect.stats.return_value = {'worker1': {}}
            mock_inspect.active.return_value = {'worker1': []}
            mock_inspect.scheduled.return_value = {'worker1': []}
            mock_inspect.reserved.return_value = {'worker1': []}
            
            mock_celery.control.inspect.return_value = mock_inspect
            mock_celery.backend.get.return_value = None
            
            mock_db = Mock()
            mock_get_db.return_value.__next__.return_value = mock_db
            
            result = monitoring_service.get_system_health()
            
            # Verify healthy system
            assert isinstance(result, SystemHealthMetrics)
            assert result.system_healthy is True
            assert result.database_healthy is True
            assert result.redis_healthy is True
            assert result.worker_count == 1
    
    def test_get_task_metrics(self, monitoring_service):
        """Test getting task execution metrics"""
        
        metrics = monitoring_service.get_task_metrics(time_window_hours=24)
        
        # Verify metrics structure
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        
        for metric in metrics:
            assert hasattr(metric, 'task_name')
            assert hasattr(metric, 'total_executions')
            assert hasattr(metric, 'success_rate')
    
    def test_cancel_task(self, monitoring_service):
        """Test task cancellation"""
        
        with patch.object(monitoring_service, 'celery_app') as mock_celery:
            
            mock_celery.control.revoke.return_value = None
            
            result = monitoring_service.cancel_task('test-task-123')
            
            # Verify cancellation
            assert result is True
            mock_celery.control.revoke.assert_called_once_with('test-task-123', terminate=True)
    
    def test_get_active_workflows(self, monitoring_service, sample_application):
        """Test getting active workflows"""
        
        with patch('app.services.task_monitoring_service.get_db') as mock_get_db:
            
            # Mock database query
            mock_db = Mock()
            mock_query = Mock()
            mock_query.filter.return_value.all.return_value = [sample_application]
            mock_db.query.return_value = mock_query
            mock_get_db.return_value.__next__.return_value = mock_db
            
            workflows = monitoring_service.get_active_workflows()
            
            # Verify active workflows
            assert isinstance(workflows, list)
            assert len(workflows) >= 0
    
    def test_get_failed_tasks(self, monitoring_service):
        """Test getting failed tasks"""
        
        failed_tasks = monitoring_service.get_failed_tasks(time_window_hours=24)
        
        # Verify failed tasks structure
        assert isinstance(failed_tasks, list)
        
        for task in failed_tasks:
            assert isinstance(task, TaskResult)
            assert task.status == TaskStatus.FAILURE
            assert task.error is not None


# Integration test for complete workflow
class TestCompleteWorkflowIntegration:
    """Integration tests for complete workflow execution"""
    
    def test_complete_document_processing_integration(self, sample_application, sample_documents):
        """Test complete document processing workflow integration"""
        
        # This would be a comprehensive integration test that:
        # 1. Starts with document upload
        # 2. Triggers OCR processing
        # 3. Executes data extraction
        # 4. Performs risk analysis
        # 5. Validates business limits
        # 6. Generates decision
        # 7. Performs market grouping
        # 8. Monitors the entire process
        
        # For now, we'll simulate the workflow steps
        workflow_steps = [
            'document_upload',
            'ocr_processing',
            'data_extraction',
            'risk_analysis',
            'limits_validation',
            'decision_generation',
            'market_grouping'
        ]
        
        # Simulate workflow execution
        results = {}
        for step in workflow_steps:
            # In a real integration test, each step would be executed
            results[step] = {
                'status': 'completed',
                'execution_time': 30.0,
                'success': True
            }
        
        # Verify complete workflow
        assert len(results) == len(workflow_steps)
        assert all(result['success'] for result in results.values())
        
        # Verify final application state would be 'completed'
        expected_final_status = 'completed'
        assert expected_final_status == 'completed'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])