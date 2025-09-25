"""
Unit tests for data models and database operations
"""
import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.database import (
    Base, Application, Document, RiskParameters, FinancialData,
    LossEvent, RiskAnalysis, Recommendation, BusinessLimit,
    ProcessingLog, SystemConfiguration,
    ApplicationStatusEnum, DocumentTypeEnum, RiskLevelEnum, DecisionTypeEnum
)
from app.models.schemas import (
    ApplicationCreate, DocumentCreate, RiskParametersCreate,
    FinancialDataCreate, LossEventCreate, RiskAnalysisCreate,
    RecommendationCreate, BusinessLimitCreate, ValidationResult
)


# Test database setup
@pytest.fixture(scope="function")
def test_db():
    """Create a test database session"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


class TestDatabaseModels:
    """Test database model functionality"""
    
    def test_application_creation(self, test_db):
        """Test creating an application"""
        app = Application(status=ApplicationStatusEnum.PENDING)
        test_db.add(app)
        test_db.commit()
        
        assert app.id is not None
        assert app.status == ApplicationStatusEnum.PENDING
        assert app.created_at is not None
        assert app.updated_at is not None
    
    def test_document_creation(self, test_db):
        """Test creating a document"""
        # Create application first
        app = Application(status=ApplicationStatusEnum.PENDING)
        test_db.add(app)
        test_db.flush()
        
        doc = Document(
            application_id=app.id,
            filename="test.pdf",
            file_path="/uploads/test.pdf",
            document_type=DocumentTypeEnum.PDF,
            processed=False,
            document_metadata={"pages": 5}
        )
        test_db.add(doc)
        test_db.commit()
        
        assert doc.id is not None
        assert doc.application_id == app.id
        assert doc.filename == "test.pdf"
        assert doc.document_type == DocumentTypeEnum.PDF
        assert doc.processed is False
        assert doc.document_metadata["pages"] == 5
    
    def test_risk_parameters_creation(self, test_db):
        """Test creating risk parameters"""
        app = Application(status=ApplicationStatusEnum.PENDING)
        test_db.add(app)
        test_db.flush()
        
        risk_params = RiskParameters(
            application_id=app.id,
            asset_value=Decimal("1000000.00"),
            coverage_limit=Decimal("800000.00"),
            asset_type="Commercial Building",
            location="123 Main St, City, State",
            industry_sector="Technology",
            construction_type="Steel Frame",
            occupancy="Office"
        )
        test_db.add(risk_params)
        test_db.commit()
        
        assert risk_params.id is not None
        assert risk_params.application_id == app.id
        assert risk_params.asset_value == Decimal("1000000.00")
        assert risk_params.asset_type == "Commercial Building"
    
    def test_financial_data_creation(self, test_db):
        """Test creating financial data"""
        app = Application(status=ApplicationStatusEnum.PENDING)
        test_db.add(app)
        test_db.flush()
        
        financial_data = FinancialData(
            application_id=app.id,
            revenue=Decimal("50000000.00"),
            assets=Decimal("100000000.00"),
            liabilities=Decimal("30000000.00"),
            credit_rating="A+",
            financial_strength_rating="AA"
        )
        test_db.add(financial_data)
        test_db.commit()
        
        assert financial_data.id is not None
        assert financial_data.application_id == app.id
        assert financial_data.revenue == Decimal("50000000.00")
        assert financial_data.credit_rating == "A+"
    
    def test_loss_event_creation(self, test_db):
        """Test creating loss events"""
        app = Application(status=ApplicationStatusEnum.PENDING)
        test_db.add(app)
        test_db.flush()
        
        loss_event = LossEvent(
            application_id=app.id,
            event_date=datetime.utcnow() - timedelta(days=30),
            amount=Decimal("50000.00"),
            cause="Fire",
            description="Kitchen fire in office building"
        )
        test_db.add(loss_event)
        test_db.commit()
        
        assert loss_event.id is not None
        assert loss_event.application_id == app.id
        assert loss_event.amount == Decimal("50000.00")
        assert loss_event.cause == "Fire"
    
    def test_risk_analysis_creation(self, test_db):
        """Test creating risk analysis"""
        app = Application(status=ApplicationStatusEnum.PENDING)
        test_db.add(app)
        test_db.flush()
        
        risk_analysis = RiskAnalysis(
            application_id=app.id,
            overall_score=Decimal("75.50"),
            confidence=Decimal("0.85"),
            risk_level=RiskLevelEnum.MEDIUM,
            factors={"location": 0.2, "construction": 0.3},
            analysis_data={"model_version": "v1.0"}
        )
        test_db.add(risk_analysis)
        test_db.commit()
        
        assert risk_analysis.id is not None
        assert risk_analysis.application_id == app.id
        assert risk_analysis.overall_score == Decimal("75.50")
        assert risk_analysis.risk_level == RiskLevelEnum.MEDIUM
        assert risk_analysis.factors["location"] == 0.2
    
    def test_recommendation_creation(self, test_db):
        """Test creating recommendations"""
        app = Application(status=ApplicationStatusEnum.PENDING)
        test_db.add(app)
        test_db.flush()
        
        recommendation = Recommendation(
            application_id=app.id,
            decision=DecisionTypeEnum.CONDITIONAL,
            confidence=Decimal("0.80"),
            rationale="Good risk profile with minor concerns",
            conditions=["Install sprinkler system", "Annual inspections"],
            premium_adjustment=Decimal("5.00"),
            coverage_modifications=["Exclude flood coverage"]
        )
        test_db.add(recommendation)
        test_db.commit()
        
        assert recommendation.id is not None
        assert recommendation.application_id == app.id
        assert recommendation.decision == DecisionTypeEnum.CONDITIONAL
        assert len(recommendation.conditions) == 2
        assert recommendation.premium_adjustment == Decimal("5.00")
    
    def test_business_limit_creation(self, test_db):
        """Test creating business limits"""
        business_limit = BusinessLimit(
            limit_type="Property Coverage",
            category="Commercial",
            max_amount=Decimal("10000000.00"),
            geographic_region="North America",
            industry_sector="Technology",
            active=True
        )
        test_db.add(business_limit)
        test_db.commit()
        
        assert business_limit.id is not None
        assert business_limit.limit_type == "Property Coverage"
        assert business_limit.max_amount == Decimal("10000000.00")
        assert business_limit.active is True
    
    def test_system_configuration_creation(self, test_db):
        """Test creating system configurations"""
        config = SystemConfiguration(
            key="test.setting",
            value={"enabled": True, "threshold": 0.8},
            description="Test configuration",
            category="testing"
        )
        test_db.add(config)
        test_db.commit()
        
        assert config.id is not None
        assert config.key == "test.setting"
        assert config.value["enabled"] is True
        assert config.category == "testing"
    
    def test_processing_log_creation(self, test_db):
        """Test creating processing logs"""
        app = Application(status=ApplicationStatusEnum.PENDING)
        test_db.add(app)
        test_db.flush()
        
        doc = Document(
            application_id=app.id,
            filename="test.pdf",
            file_path="/uploads/test.pdf",
            document_type=DocumentTypeEnum.PDF
        )
        test_db.add(doc)
        test_db.flush()
        
        log = ProcessingLog(
            document_id=doc.id,
            application_id=app.id,
            process_type="OCR",
            status="success",
            message="Document processed successfully",
            processing_time=Decimal("2.5"),
            log_metadata={"confidence": 0.95}
        )
        test_db.add(log)
        test_db.commit()
        
        assert log.id is not None
        assert log.document_id == doc.id
        assert log.process_type == "OCR"
        assert log.status == "success"
        assert log.processing_time == Decimal("2.5")
    
    def test_application_relationships(self, test_db):
        """Test application model relationships"""
        # Create application
        app = Application(status=ApplicationStatusEnum.PENDING)
        test_db.add(app)
        test_db.flush()
        
        # Add related data
        doc = Document(
            application_id=app.id,
            filename="test.pdf",
            file_path="/uploads/test.pdf",
            document_type=DocumentTypeEnum.PDF
        )
        
        risk_params = RiskParameters(
            application_id=app.id,
            asset_value=Decimal("1000000.00")
        )
        
        financial_data = FinancialData(
            application_id=app.id,
            revenue=Decimal("5000000.00")
        )
        
        loss_event = LossEvent(
            application_id=app.id,
            event_date=datetime.utcnow(),
            amount=Decimal("10000.00")
        )
        
        test_db.add_all([doc, risk_params, financial_data, loss_event])
        test_db.commit()
        
        # Test relationships
        test_db.refresh(app)
        assert len(app.documents) == 1
        assert app.risk_parameters is not None
        assert app.financial_data is not None
        assert len(app.loss_history) == 1
        assert app.documents[0].filename == "test.pdf"
        assert app.risk_parameters.asset_value == Decimal("1000000.00")


class TestPydanticSchemas:
    """Test Pydantic schema validation"""
    
    def test_application_create_schema(self):
        """Test application creation schema"""
        app_data = ApplicationCreate(status="pending")
        assert app_data.status.value == "pending"
    
    def test_document_create_schema(self):
        """Test document creation schema"""
        doc_data = DocumentCreate(
            filename="test.pdf",
            document_type="pdf",
            file_path="/uploads/test.pdf"
        )
        assert doc_data.filename == "test.pdf"
        assert doc_data.document_type.value == "pdf"
        assert doc_data.file_path == "/uploads/test.pdf"
    
    def test_risk_parameters_create_schema(self):
        """Test risk parameters creation schema"""
        risk_data = RiskParametersCreate(
            asset_value=Decimal("1000000.00"),
            coverage_limit=Decimal("800000.00"),
            asset_type="Commercial Building"
        )
        assert risk_data.asset_value == Decimal("1000000.00")
        assert risk_data.coverage_limit == Decimal("800000.00")
        assert risk_data.asset_type == "Commercial Building"
    
    def test_financial_data_create_schema(self):
        """Test financial data creation schema"""
        financial_data = FinancialDataCreate(
            revenue=Decimal("50000000.00"),
            assets=Decimal("100000000.00"),
            credit_rating="A+"
        )
        assert financial_data.revenue == Decimal("50000000.00")
        assert financial_data.assets == Decimal("100000000.00")
        assert financial_data.credit_rating == "A+"
    
    def test_loss_event_create_schema(self):
        """Test loss event creation schema"""
        loss_data = LossEventCreate(
            event_date=datetime.utcnow(),
            amount=Decimal("50000.00"),
            cause="Fire"
        )
        assert loss_data.amount == Decimal("50000.00")
        assert loss_data.cause == "Fire"
    
    def test_risk_analysis_create_schema(self):
        """Test risk analysis creation schema"""
        analysis_data = RiskAnalysisCreate(
            overall_score=Decimal("75.50"),
            confidence=Decimal("0.85"),
            risk_level="MEDIUM",
            factors={"location": 0.2}
        )
        assert analysis_data.overall_score == Decimal("75.50")
        assert analysis_data.confidence == Decimal("0.85")
        assert analysis_data.risk_level.value == "MEDIUM"
    
    def test_recommendation_create_schema(self):
        """Test recommendation creation schema"""
        rec_data = RecommendationCreate(
            decision="CONDITIONAL",
            confidence=Decimal("0.80"),
            rationale="Good profile with conditions",
            conditions=["Install sprinklers"]
        )
        assert rec_data.decision.value == "CONDITIONAL"
        assert rec_data.confidence == Decimal("0.80")
        assert len(rec_data.conditions) == 1
    
    def test_business_limit_create_schema(self):
        """Test business limit creation schema"""
        limit_data = BusinessLimitCreate(
            limit_type="Property Coverage",
            max_amount=Decimal("10000000.00"),
            active=True
        )
        assert limit_data.limit_type == "Property Coverage"
        assert limit_data.max_amount == Decimal("10000000.00")
        assert limit_data.active is True
    
    def test_validation_result_schema(self):
        """Test validation result schema"""
        result = ValidationResult(
            is_valid=False,
            errors=["Missing required field"],
            warnings=["Low confidence score"]
        )
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
    
    def test_schema_validation_errors(self):
        """Test schema validation with invalid data"""
        # Test negative asset value
        with pytest.raises(ValueError):
            RiskParametersCreate(asset_value=Decimal("-1000.00"))
        
        # Test invalid confidence range
        with pytest.raises(ValueError):
            RiskAnalysisCreate(
                overall_score=Decimal("75.00"),
                confidence=Decimal("1.5"),  # > 1.0
                risk_level="MEDIUM"
            )
        
        # Test invalid score range
        with pytest.raises(ValueError):
            RiskAnalysisCreate(
                overall_score=Decimal("150.00"),  # > 100
                confidence=Decimal("0.8"),
                risk_level="MEDIUM"
            )


class TestDataValidation:
    """Test data validation and constraints"""
    
    def test_unique_constraints(self, test_db):
        """Test unique constraints"""
        app = Application(status=ApplicationStatusEnum.PENDING)
        test_db.add(app)
        test_db.flush()
        
        # Create first risk parameters
        risk_params1 = RiskParameters(
            application_id=app.id,
            asset_value=Decimal("1000000.00")
        )
        test_db.add(risk_params1)
        test_db.commit()
        
        # Try to create second risk parameters for same application
        risk_params2 = RiskParameters(
            application_id=app.id,
            asset_value=Decimal("2000000.00")
        )
        test_db.add(risk_params2)
        
        with pytest.raises(Exception):  # Should raise integrity error
            test_db.commit()
    
    def test_foreign_key_constraints(self, test_db):
        """Test foreign key constraints"""
        # Try to create document without valid application
        doc = Document(
            application_id="00000000-0000-0000-0000-000000000000",
            filename="test.pdf",
            file_path="/uploads/test.pdf",
            document_type=DocumentTypeEnum.PDF
        )
        test_db.add(doc)
        
        with pytest.raises(Exception):  # Should raise integrity error
            test_db.commit()
    
    def test_enum_constraints(self, test_db):
        """Test enum value constraints"""
        app = Application(status=ApplicationStatusEnum.PENDING)
        test_db.add(app)
        test_db.commit()
        
        assert app.status == ApplicationStatusEnum.PENDING
        
        # Update to valid enum value
        app.status = ApplicationStatusEnum.PROCESSING
        test_db.commit()
        
        assert app.status == ApplicationStatusEnum.PROCESSING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])