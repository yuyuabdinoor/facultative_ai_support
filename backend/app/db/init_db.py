"""
Database initialization and seed data
"""
import logging
from decimal import Decimal
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.database import get_db_session, DatabaseManager
from app.models.database import (
    Application, Document, RiskParameters, FinancialData, 
    LossEvent, RiskAnalysis, Recommendation, BusinessLimit,
    SystemConfiguration, ApplicationStatusEnum, DocumentTypeEnum,
    RiskLevelEnum, DecisionTypeEnum
)

logger = logging.getLogger(__name__)


def create_seed_data(db: Session) -> None:
    """Create seed data for the database"""
    
    try:
        # Create business limits
        business_limits = [
            BusinessLimit(
                limit_type="Property Coverage",
                category="Commercial Property",
                max_amount=Decimal("50000000.00"),
                geographic_region="North America",
                industry_sector="Manufacturing",
                active=True
            ),
            BusinessLimit(
                limit_type="Liability Coverage",
                category="General Liability",
                max_amount=Decimal("25000000.00"),
                geographic_region="Europe",
                industry_sector="Technology",
                active=True
            ),
            BusinessLimit(
                limit_type="Property Coverage",
                category="Industrial Property",
                max_amount=Decimal("100000000.00"),
                geographic_region="Global",
                industry_sector="Energy",
                active=True
            ),
            BusinessLimit(
                limit_type="Catastrophe Coverage",
                category="Natural Disasters",
                max_amount=Decimal("200000000.00"),
                geographic_region="Asia Pacific",
                industry_sector="All",
                active=True
            )
        ]
        
        for limit in business_limits:
            existing = db.query(BusinessLimit).filter(
                BusinessLimit.limit_type == limit.limit_type,
                BusinessLimit.category == limit.category,
                BusinessLimit.geographic_region == limit.geographic_region
            ).first()
            
            if not existing:
                db.add(limit)
                logger.info(f"Added business limit: {limit.limit_type} - {limit.category}")
        
        # Create system configurations
        system_configs = [
            SystemConfiguration(
                key="risk_analysis.confidence_threshold",
                value={"threshold": 0.75},
                description="Minimum confidence threshold for risk analysis",
                category="risk_analysis"
            ),
            SystemConfiguration(
                key="document_processing.max_file_size",
                value={"size_mb": 100},
                description="Maximum file size for document uploads in MB",
                category="document_processing"
            ),
            SystemConfiguration(
                key="ai_models.risk_scoring_model",
                value={"model_name": "risk_scorer_v1", "version": "1.0.0"},
                description="Current risk scoring model configuration",
                category="ai_models"
            ),
            SystemConfiguration(
                key="notification.email_enabled",
                value={"enabled": True, "smtp_server": "localhost"},
                description="Email notification settings",
                category="notification"
            ),
            SystemConfiguration(
                key="business_rules.auto_approve_threshold",
                value={"score": 85.0, "confidence": 0.9},
                description="Thresholds for automatic approval",
                category="business_rules"
            )
        ]
        
        for config in system_configs:
            existing = db.query(SystemConfiguration).filter(
                SystemConfiguration.key == config.key
            ).first()
            
            if not existing:
                db.add(config)
                logger.info(f"Added system configuration: {config.key}")
        
        # Create sample application with complete data
        sample_app = Application(
            status=ApplicationStatusEnum.COMPLETED
        )
        db.add(sample_app)
        db.flush()  # Get the ID
        
        # Add sample document
        sample_doc = Document(
            application_id=sample_app.id,
            filename="sample_property_schedule.pdf",
            file_path="/uploads/sample_property_schedule.pdf",
            document_type=DocumentTypeEnum.PDF,
            processed=True,
            document_metadata={
                "pages": 15,
                "file_size": 2048576,
                "ocr_confidence": 0.95,
                "extraction_method": "doctr"
            }
        )
        db.add(sample_doc)
        
        # Add risk parameters
        risk_params = RiskParameters(
            application_id=sample_app.id,
            asset_value=Decimal("25000000.00"),
            coverage_limit=Decimal("20000000.00"),
            asset_type="Commercial Building",
            location="123 Business District, New York, NY 10001",
            industry_sector="Technology",
            construction_type="Steel Frame",
            occupancy="Office Building"
        )
        db.add(risk_params)
        
        # Add financial data
        financial_data = FinancialData(
            application_id=sample_app.id,
            revenue=Decimal("150000000.00"),
            assets=Decimal("500000000.00"),
            liabilities=Decimal("200000000.00"),
            credit_rating="A+",
            financial_strength_rating="AA-"
        )
        db.add(financial_data)
        
        # Add loss history
        loss_events = [
            LossEvent(
                application_id=sample_app.id,
                event_date=datetime.utcnow() - timedelta(days=365),
                amount=Decimal("150000.00"),
                cause="Water Damage",
                description="Pipe burst in server room causing equipment damage"
            ),
            LossEvent(
                application_id=sample_app.id,
                event_date=datetime.utcnow() - timedelta(days=730),
                amount=Decimal("75000.00"),
                cause="Theft",
                description="Laptop computers stolen from office"
            )
        ]
        
        for event in loss_events:
            db.add(event)
        
        # Add risk analysis
        risk_analysis = RiskAnalysis(
            application_id=sample_app.id,
            overall_score=Decimal("78.50"),
            confidence=Decimal("0.87"),
            risk_level=RiskLevelEnum.MEDIUM,
            factors={
                "location_risk": 0.15,
                "construction_risk": 0.10,
                "occupancy_risk": 0.12,
                "financial_strength": 0.25,
                "loss_history": 0.18,
                "industry_risk": 0.20
            },
            analysis_data={
                "model_version": "v1.2.0",
                "processing_time": 2.34,
                "data_completeness": 0.95,
                "risk_factors_analyzed": 15
            }
        )
        db.add(risk_analysis)
        
        # Add recommendation
        recommendation = Recommendation(
            application_id=sample_app.id,
            decision=DecisionTypeEnum.CONDITIONAL,
            confidence=Decimal("0.82"),
            rationale="Property shows good financial strength and manageable risk profile. "
                     "Recommend approval with standard terms and minor premium adjustment.",
            conditions=[
                "Install additional fire suppression system",
                "Implement enhanced security measures",
                "Provide quarterly financial reports"
            ],
            premium_adjustment=Decimal("5.00"),  # 5% increase
            coverage_modifications=[
                "Exclude flood coverage",
                "Add cyber liability endorsement"
            ]
        )
        db.add(recommendation)
        
        db.commit()
        logger.info("Sample application data created successfully")
        
    except IntegrityError as e:
        db.rollback()
        logger.warning(f"Some seed data already exists: {e}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating seed data: {e}")
        raise


def init_database():
    """Initialize database with tables and seed data"""
    try:
        # Initialize database tables
        DatabaseManager.init_db()
        logger.info("Database tables initialized")
        
        # Create seed data
        db = get_db_session()
        try:
            create_seed_data(db)
            logger.info("Database initialization completed successfully")
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def reset_database():
    """Reset database and recreate with seed data"""
    try:
        # Reset database
        DatabaseManager.reset_db()
        logger.info("Database reset completed")
        
        # Create seed data
        db = get_db_session()
        try:
            create_seed_data(db)
            logger.info("Database reset and seeding completed successfully")
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database
    init_database()