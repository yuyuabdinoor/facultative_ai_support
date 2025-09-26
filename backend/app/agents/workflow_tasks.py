"""
Celery workflow tasks for specific agent processing workflows

This module contains tasks that coordinate specific agent workflows like
data extraction, risk analysis, business limits validation, and decision generation.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

try:
    from app.celery import celery_app
    CELERY_AVAILABLE = True
except ImportError:
    celery_app = None
    CELERY_AVAILABLE = False

from app.core.database import get_db
from app.models.database import Document, Application, RiskAnalysis, Recommendation
from app.agents.data_extraction_agent import data_extraction_agent
from app.agents.risk_analysis_agent import risk_analysis_agent
from app.agents.business_limits_agent import business_limits_agent
from app.agents.decision_engine_agent import decision_engine_agent
from app.agents.market_grouping_agent import market_grouping_agent

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


@_celery_task_decorator(bind=True, max_retries=3)
def extract_risk_data_workflow(self, application_id: str, document_ids: List[str]) -> Dict[str, Any]:
    """
    Execute data extraction workflow for all documents in an application
    
    Args:
        application_id: UUID of the application
        document_ids: List of document IDs to extract data from
        
    Returns:
        Data extraction workflow results
    """
    try:
        logger.info(f"Starting data extraction workflow for application {application_id}")
        
        # Get all processed documents
        db = next(get_db())
        try:
            documents = db.query(Document).filter(
                Document.id.in_(document_ids),
                Document.processed == True
            ).all()
            
            if not documents:
                raise ValueError(f"No processed documents found for application {application_id}")
            
            # Extract data from each document
            extraction_results = []
            consolidated_data = {
                'risk_parameters': {},
                'financial_data': {},
                'geographic_data': {},
                'asset_information': {},
                'coverage_details': {}
            }
            
            for document in documents:
                try:
                    # Get OCR text from document metadata
                    ocr_text = ""
                    if document.metadata:
                        if 'ocr_result' in document.metadata:
                            ocr_text = document.metadata['ocr_result'].get('text', '')
                        elif 'email_content' in document.metadata:
                            ocr_text = document.metadata['email_content'].get('body', '')
                        elif 'excel_data' in document.metadata:
                            # For Excel, we'd need to extract text from sheets
                            ocr_text = str(document.metadata['excel_data'])
                    
                    if ocr_text:
                        # Extract structured data using the data extraction agent
                        extracted_data = data_extraction_agent.extract_risk_parameters(ocr_text)
                        
                        extraction_result = {
                            'document_id': document.id,
                            'document_type': document.document_type,
                            'extraction_success': True,
                            'extracted_data': extracted_data.dict() if hasattr(extracted_data, 'dict') else extracted_data
                        }
                        
                        # Consolidate data across documents
                        if hasattr(extracted_data, 'dict'):
                            data_dict = extracted_data.dict()
                        else:
                            data_dict = extracted_data
                        
                        # Merge data with priority (later documents override earlier ones for conflicts)
                        for key, value in data_dict.items():
                            if value is not None:
                                consolidated_data['risk_parameters'][key] = value
                        
                    else:
                        extraction_result = {
                            'document_id': document.id,
                            'document_type': document.document_type,
                            'extraction_success': False,
                            'error': 'No OCR text available'
                        }
                    
                    extraction_results.append(extraction_result)
                    
                except Exception as doc_error:
                    logger.error(f"Error extracting data from document {document.id}: {str(doc_error)}")
                    extraction_results.append({
                        'document_id': document.id,
                        'document_type': document.document_type,
                        'extraction_success': False,
                        'error': str(doc_error)
                    })
            
            # Store consolidated data in application
            application = db.query(Application).filter(Application.id == application_id).first()
            if application:
                if not hasattr(application, 'metadata') or application.metadata is None:
                    application.metadata = {}
                
                application.metadata['extracted_data'] = consolidated_data
                application.metadata['extraction_results'] = extraction_results
                application.updated_at = datetime.utcnow()
                db.commit()
            
        finally:
            db.close()
        
        workflow_result = {
            'application_id': application_id,
            'workflow_type': 'data_extraction',
            'status': 'completed',
            'documents_processed': len(extraction_results),
            'successful_extractions': len([r for r in extraction_results if r['extraction_success']]),
            'consolidated_data': consolidated_data,
            'extraction_results': extraction_results,
            'completion_time': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Data extraction workflow completed for application {application_id}")
        
        return workflow_result
        
    except Exception as exc:
        logger.error(f"Data extraction workflow failed for application {application_id}: {str(exc)}")
        
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying data extraction workflow (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        return {
            'application_id': application_id,
            'workflow_type': 'data_extraction',
            'status': 'failed',
            'error': str(exc),
            'retry_count': self.request.retries
        }


@_celery_task_decorator(bind=True, max_retries=3)
def risk_analysis_workflow(self, application_id: str) -> Dict[str, Any]:
    """
    Execute risk analysis workflow for an application
    
    Args:
        application_id: UUID of the application
        
    Returns:
        Risk analysis workflow results
    """
    try:
        logger.info(f"Starting risk analysis workflow for application {application_id}")
        
        db = next(get_db())
        try:
            application = db.query(Application).filter(Application.id == application_id).first()
            if not application:
                raise ValueError(f"Application {application_id} not found")
            
            # Get extracted data from previous workflow
            extracted_data = {}
            if application.metadata and 'extracted_data' in application.metadata:
                extracted_data = application.metadata['extracted_data']
            
            if not extracted_data:
                raise ValueError(f"No extracted data found for application {application_id}")
            
            # Perform risk analysis using the risk analysis agent
            risk_parameters = extracted_data.get('risk_parameters', {})
            
            # Analyze different risk aspects
            analysis_results = {}
            
            # Loss history analysis (if available)
            if 'loss_history' in risk_parameters:
                try:
                    loss_analysis = risk_analysis_agent.analyze_loss_history(risk_parameters['loss_history'])
                    analysis_results['loss_analysis'] = loss_analysis
                except Exception as e:
                    logger.warning(f"Loss history analysis failed: {str(e)}")
                    analysis_results['loss_analysis'] = {'error': str(e)}
            
            # Catastrophe exposure assessment
            if 'location' in risk_parameters and 'asset_type' in risk_parameters:
                try:
                    cat_exposure = risk_analysis_agent.assess_catastrophe_exposure(
                        risk_parameters['location'],
                        risk_parameters['asset_type']
                    )
                    analysis_results['catastrophe_exposure'] = cat_exposure
                except Exception as e:
                    logger.warning(f"Catastrophe exposure analysis failed: {str(e)}")
                    analysis_results['catastrophe_exposure'] = {'error': str(e)}
            
            # Financial strength evaluation
            financial_data = extracted_data.get('financial_data', {})
            if financial_data:
                try:
                    financial_rating = risk_analysis_agent.evaluate_financial_strength(financial_data)
                    analysis_results['financial_rating'] = financial_rating
                except Exception as e:
                    logger.warning(f"Financial strength analysis failed: {str(e)}")
                    analysis_results['financial_rating'] = {'error': str(e)}
            
            # Calculate overall risk score
            try:
                risk_score = risk_analysis_agent.calculate_risk_score(risk_parameters)
                analysis_results['risk_score'] = risk_score
            except Exception as e:
                logger.warning(f"Risk score calculation failed: {str(e)}")
                analysis_results['risk_score'] = {'error': str(e)}
            
            # Generate risk report
            try:
                risk_report = risk_analysis_agent.generate_risk_report(analysis_results)
                analysis_results['risk_report'] = risk_report
            except Exception as e:
                logger.warning(f"Risk report generation failed: {str(e)}")
                analysis_results['risk_report'] = {'error': str(e)}
            
            # Store risk analysis results
            risk_analysis = RiskAnalysis(
                application_id=application_id,
                overall_score=analysis_results.get('risk_score', {}).get('overall_score', 0.0),
                confidence=analysis_results.get('risk_score', {}).get('confidence', 0.0),
                risk_level=analysis_results.get('risk_score', {}).get('risk_level', 'UNKNOWN'),
                factors=analysis_results.get('risk_score', {}).get('factors', {}),
                analysis_data=analysis_results
            )
            
            db.add(risk_analysis)
            
            # Update application metadata
            if not hasattr(application, 'metadata') or application.metadata is None:
                application.metadata = {}
            
            application.metadata['risk_analysis'] = analysis_results
            application.updated_at = datetime.utcnow()
            
            db.commit()
            
        finally:
            db.close()
        
        workflow_result = {
            'application_id': application_id,
            'workflow_type': 'risk_analysis',
            'status': 'completed',
            'analysis_results': analysis_results,
            'completion_time': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Risk analysis workflow completed for application {application_id}")
        
        return workflow_result
        
    except Exception as exc:
        logger.error(f"Risk analysis workflow failed for application {application_id}: {str(exc)}")
        
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying risk analysis workflow (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        return {
            'application_id': application_id,
            'workflow_type': 'risk_analysis',
            'status': 'failed',
            'error': str(exc),
            'retry_count': self.request.retries
        }


@_celery_task_decorator(bind=True, max_retries=3)
def validate_business_limits_workflow(self, application_id: str) -> Dict[str, Any]:
    """
    Execute business limits validation workflow for an application
    
    Args:
        application_id: UUID of the application
        
    Returns:
        Business limits validation workflow results
    """
    try:
        logger.info(f"Starting business limits validation workflow for application {application_id}")
        
        db = next(get_db())
        try:
            application = db.query(Application).filter(Application.id == application_id).first()
            if not application:
                raise ValueError(f"Application {application_id} not found")
            
            # Get extracted data
            extracted_data = {}
            if application.metadata and 'extracted_data' in application.metadata:
                extracted_data = application.metadata['extracted_data']
            
            if not extracted_data:
                raise ValueError(f"No extracted data found for application {application_id}")
            
            # Perform business limits validation
            validation_results = {}
            
            # Check business limits
            try:
                limits_check = business_limits_agent.check_business_limits(application)
                validation_results['business_limits_check'] = limits_check
            except Exception as e:
                logger.warning(f"Business limits check failed: {str(e)}")
                validation_results['business_limits_check'] = {'error': str(e)}
            
            # Validate geographic limits
            risk_parameters = extracted_data.get('risk_parameters', {})
            if 'location' in risk_parameters and 'coverage_limit' in risk_parameters:
                try:
                    geo_validation = business_limits_agent.validate_geographic_limits(
                        risk_parameters['location'],
                        risk_parameters['coverage_limit']
                    )
                    validation_results['geographic_validation'] = geo_validation
                except Exception as e:
                    logger.warning(f"Geographic limits validation failed: {str(e)}")
                    validation_results['geographic_validation'] = {'error': str(e)}
            
            # Check sector restrictions
            if 'industry_sector' in risk_parameters and 'asset_type' in risk_parameters:
                try:
                    sector_check = business_limits_agent.check_sector_restrictions(
                        risk_parameters['industry_sector'],
                        risk_parameters['asset_type']
                    )
                    validation_results['sector_restrictions'] = sector_check
                except Exception as e:
                    logger.warning(f"Sector restrictions check failed: {str(e)}")
                    validation_results['sector_restrictions'] = {'error': str(e)}
            
            # Validate regulatory compliance
            try:
                compliance_check = business_limits_agent.validate_regulatory_compliance(application)
                validation_results['regulatory_compliance'] = compliance_check
            except Exception as e:
                logger.warning(f"Regulatory compliance check failed: {str(e)}")
                validation_results['regulatory_compliance'] = {'error': str(e)}
            
            # Update application metadata
            if not hasattr(application, 'metadata') or application.metadata is None:
                application.metadata = {}
            
            application.metadata['limits_validation'] = validation_results
            application.updated_at = datetime.utcnow()
            
            db.commit()
            
        finally:
            db.close()
        
        workflow_result = {
            'application_id': application_id,
            'workflow_type': 'business_limits_validation',
            'status': 'completed',
            'validation_results': validation_results,
            'completion_time': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Business limits validation workflow completed for application {application_id}")
        
        return workflow_result
        
    except Exception as exc:
        logger.error(f"Business limits validation workflow failed for application {application_id}: {str(exc)}")
        
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying business limits validation workflow (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        return {
            'application_id': application_id,
            'workflow_type': 'business_limits_validation',
            'status': 'failed',
            'error': str(exc),
            'retry_count': self.request.retries
        }


@_celery_task_decorator(bind=True, max_retries=3)
def generate_decision_workflow(self, application_id: str) -> Dict[str, Any]:
    """
    Execute decision generation workflow for an application
    
    Args:
        application_id: UUID of the application
        
    Returns:
        Decision generation workflow results
    """
    try:
        logger.info(f"Starting decision generation workflow for application {application_id}")
        
        db = next(get_db())
        try:
            application = db.query(Application).filter(Application.id == application_id).first()
            if not application:
                raise ValueError(f"Application {application_id} not found")
            
            # Get analysis results from previous workflows
            risk_analysis = {}
            limits_validation = {}
            
            if application.metadata:
                risk_analysis = application.metadata.get('risk_analysis', {})
                limits_validation = application.metadata.get('limits_validation', {})
            
            if not risk_analysis and not limits_validation:
                raise ValueError(f"No analysis data found for application {application_id}")
            
            # Generate decision using the decision engine agent
            analysis_data = {
                'risk_analysis': risk_analysis,
                'limits_validation': limits_validation,
                'application_id': application_id
            }
            
            try:
                recommendation = decision_engine_agent.generate_recommendation(analysis_data)
                decision_results = {
                    'recommendation': recommendation,
                    'decision_success': True
                }
            except Exception as e:
                logger.error(f"Decision generation failed: {str(e)}")
                decision_results = {
                    'recommendation': None,
                    'decision_success': False,
                    'error': str(e)
                }
            
            # Calculate confidence score
            try:
                factors = []
                if risk_analysis.get('risk_score'):
                    factors.append(risk_analysis['risk_score'])
                if limits_validation.get('business_limits_check'):
                    factors.append(limits_validation['business_limits_check'])
                
                confidence_score = decision_engine_agent.calculate_confidence_score(factors)
                decision_results['confidence_score'] = confidence_score
            except Exception as e:
                logger.warning(f"Confidence score calculation failed: {str(e)}")
                decision_results['confidence_score'] = 0.0
            
            # Generate rationale
            if decision_results['recommendation']:
                try:
                    rationale = decision_engine_agent.generate_rationale(
                        decision_results['recommendation'],
                        [risk_analysis, limits_validation]
                    )
                    decision_results['rationale'] = rationale
                except Exception as e:
                    logger.warning(f"Rationale generation failed: {str(e)}")
                    decision_results['rationale'] = "Rationale generation failed"
            
            # Store recommendation in database
            if decision_results['recommendation']:
                recommendation_record = Recommendation(
                    application_id=application_id,
                    decision=decision_results['recommendation'].get('decision', 'UNKNOWN'),
                    confidence=decision_results.get('confidence_score', 0.0),
                    rationale=decision_results.get('rationale', ''),
                    conditions=decision_results['recommendation'].get('conditions', []),
                    premium_adjustment=decision_results['recommendation'].get('premium_adjustment'),
                    coverage_modifications=decision_results['recommendation'].get('coverage_modifications', [])
                )
                
                db.add(recommendation_record)
            
            # Update application metadata
            if not hasattr(application, 'metadata') or application.metadata is None:
                application.metadata = {}
            
            application.metadata['decision_results'] = decision_results
            application.updated_at = datetime.utcnow()
            
            db.commit()
            
        finally:
            db.close()
        
        workflow_result = {
            'application_id': application_id,
            'workflow_type': 'decision_generation',
            'status': 'completed',
            'decision_results': decision_results,
            'completion_time': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Decision generation workflow completed for application {application_id}")
        
        return workflow_result
        
    except Exception as exc:
        logger.error(f"Decision generation workflow failed for application {application_id}: {str(exc)}")
        
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying decision generation workflow (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        return {
            'application_id': application_id,
            'workflow_type': 'decision_generation',
            'status': 'failed',
            'error': str(exc),
            'retry_count': self.request.retries
        }


@_celery_task_decorator(bind=True, max_retries=3)
def market_grouping_workflow(self, application_id: str, document_ids: List[str]) -> Dict[str, Any]:
    """
    Execute market grouping workflow for an application
    
    Args:
        application_id: UUID of the application
        document_ids: List of document IDs to analyze for market grouping
        
    Returns:
        Market grouping workflow results
    """
    try:
        logger.info(f"Starting market grouping workflow for application {application_id}")
        
        db = next(get_db())
        try:
            # Get documents and their content
            documents = db.query(Document).filter(Document.id.in_(document_ids)).all()
            
            if not documents:
                raise ValueError(f"No documents found for application {application_id}")
            
            # Analyze market grouping for each document
            market_results = {}
            
            for document in documents:
                try:
                    # Get document content
                    content = ""
                    metadata = {}
                    
                    if document.metadata:
                        if 'ocr_result' in document.metadata:
                            content = document.metadata['ocr_result'].get('text', '')
                        elif 'email_content' in document.metadata:
                            content = document.metadata['email_content'].get('body', '')
                            metadata = document.metadata['email_content']
                        elif 'excel_data' in document.metadata:
                            content = str(document.metadata['excel_data'])
                            metadata = document.metadata['excel_data']
                    
                    if content:
                        # Identify market
                        market = market_grouping_agent.identify_market(content, metadata)
                        
                        # Classify geographic market
                        geo_market = market_grouping_agent.classify_geographic_market(market.get('location', ''))
                        
                        market_results[document.id] = {
                            'document_id': document.id,
                            'market': market,
                            'geographic_market': geo_market,
                            'analysis_success': True
                        }
                    else:
                        market_results[document.id] = {
                            'document_id': document.id,
                            'analysis_success': False,
                            'error': 'No content available for analysis'
                        }
                        
                except Exception as doc_error:
                    logger.error(f"Market analysis failed for document {document.id}: {str(doc_error)}")
                    market_results[document.id] = {
                        'document_id': document.id,
                        'analysis_success': False,
                        'error': str(doc_error)
                    }
            
            # Group documents by industry
            try:
                industry_groups = market_grouping_agent.group_by_industry(documents)
                market_results['industry_groups'] = industry_groups
            except Exception as e:
                logger.warning(f"Industry grouping failed: {str(e)}")
                market_results['industry_groups'] = {'error': str(e)}
            
            # Map relationships between documents
            try:
                relationships = market_grouping_agent.map_relationships(documents)
                market_results['document_relationships'] = relationships
            except Exception as e:
                logger.warning(f"Relationship mapping failed: {str(e)}")
                market_results['document_relationships'] = {'error': str(e)}
            
            # Update application metadata
            application = db.query(Application).filter(Application.id == application_id).first()
            if application:
                if not hasattr(application, 'metadata') or application.metadata is None:
                    application.metadata = {}
                
                application.metadata['market_analysis'] = market_results
                application.updated_at = datetime.utcnow()
                db.commit()
            
        finally:
            db.close()
        
        workflow_result = {
            'application_id': application_id,
            'workflow_type': 'market_grouping',
            'status': 'completed',
            'market_results': market_results,
            'documents_analyzed': len(document_ids),
            'completion_time': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Market grouping workflow completed for application {application_id}")
        
        return workflow_result
        
    except Exception as exc:
        logger.error(f"Market grouping workflow failed for application {application_id}: {str(exc)}")
        
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying market grouping workflow (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60 * (2 ** self.request.retries))
        
        return {
            'application_id': application_id,
            'workflow_type': 'market_grouping',
            'status': 'failed',
            'error': str(exc),
            'retry_count': self.request.retries
        }