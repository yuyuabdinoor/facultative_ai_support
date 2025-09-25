"""
Unit tests for the Data Extraction Agent
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
import torch

from app.agents.data_extraction_agent import DataExtractionAgent
from app.models.schemas import RiskParameters, FinancialData, ValidationResult, ExtractionResult


class TestDataExtractionAgent:
    """Test cases for DataExtractionAgent"""
    
    @pytest.fixture
    def mock_models(self):
        """Mock the Hugging Face models to avoid loading them in tests"""
        with patch('app.agents.data_extraction_agent.AutoTokenizer'), \
             patch('app.agents.data_extraction_agent.AutoModelForTokenClassification'), \
             patch('app.agents.data_extraction_agent.AutoModel'), \
             patch('app.agents.data_extraction_agent.pipeline'), \
             patch('app.agents.data_extraction_agent.SentenceTransformer'):
            yield
    
    @pytest.fixture
    def agent(self, mock_models):
        """Create a DataExtractionAgent instance with mocked models"""
        agent = DataExtractionAgent()
        
        # Mock the models to avoid actual inference
        agent.ner_model = Mock()
        agent.ner_tokenizer = Mock()
        agent.finbert_model = Mock()
        agent.finbert_tokenizer = Mock()
        agent.financial_classifier = Mock()
        agent.layout_model = Mock()
        agent.layout_tokenizer = Mock()
        agent.zero_shot_classifier = Mock()
        agent.embeddings_model = Mock()
        agent.device = torch.device("cpu")
        
        return agent
    
    @pytest.mark.asyncio
    async def test_extract_risk_parameters_basic(self, agent):
        """Test basic risk parameter extraction"""
        # Mock the helper methods
        agent._extract_ner_entities = AsyncMock(return_value=[
            {'word': 'New York', 'label': 'LOC', 'confidence': 0.95}
        ])
        agent._extract_pattern_data = Mock(return_value={
            'asset_value': Decimal('1000000'),
            'construction_type': 'steel'
        })
        agent._classify_asset_type = AsyncMock(return_value={
            'asset_type': 'commercial building',
            'asset_type_confidence': 0.85
        })
        
        text = "Commercial building in New York with asset value of $1,000,000"
        
        risk_params, confidence_scores = await agent.extract_risk_parameters(text)
        
        assert isinstance(risk_params, RiskParameters)
        assert risk_params.location == 'New York'
        assert risk_params.asset_value == Decimal('1000000')
        assert risk_params.asset_type == 'commercial building'
        assert risk_params.construction_type == 'steel'
        assert 'location' in confidence_scores
        assert confidence_scores['location'] == 0.95
    
    @pytest.mark.asyncio
    async def test_extract_financial_data_basic(self, agent):
        """Test basic financial data extraction"""
        # Mock the helper methods
        agent._extract_financial_amounts = Mock(return_value={
            'revenue': Decimal('5000000'),
            'assets': Decimal('10000000')
        })
        agent._extract_credit_rating = Mock(return_value='AA')
        agent._analyze_financial_sentiment = AsyncMock(return_value={
            'label': 'positive',
            'score': 0.85
        })
        
        text = "Company has annual revenue of $5M and total assets of $10M with AA credit rating"
        
        financial_data, confidence_scores = await agent.extract_financial_data(text)
        
        assert isinstance(financial_data, FinancialData)
        assert financial_data.revenue == Decimal('5000000')
        assert financial_data.assets == Decimal('10000000')
        assert financial_data.credit_rating == 'AA'
        assert financial_data.financial_strength_rating == 'positive'
        assert 'credit_rating' in confidence_scores
        assert confidence_scores['credit_rating'] == 0.9
    
    @pytest.mark.asyncio
    async def test_extract_geographic_info(self, agent):
        """Test geographic information extraction"""
        # Mock zero-shot classifier
        agent.zero_shot_classifier.return_value = {
            'labels': ['North America'],
            'scores': [0.85]
        }
        
        text = "Property located at 123 Main Street, New York, NY, United States"
        
        geo_info = await agent.extract_geographic_info(text)
        
        assert 'addresses' in geo_info
        assert 'cities' in geo_info
        assert 'countries' in geo_info
        assert 'region' in geo_info
        assert geo_info['region'] == 'North America'
        assert geo_info['region_confidence'] == 0.85
    
    @pytest.mark.asyncio
    async def test_classify_asset_type(self, agent):
        """Test asset type classification"""
        # Mock zero-shot classifier
        agent.zero_shot_classifier.return_value = {
            'labels': ['office building'],
            'scores': [0.75]
        }
        
        description = "Modern office building with multiple floors"
        
        result = await agent.classify_asset_type(description)
        
        assert 'asset_type' in result
        assert result['asset_type'] == 'office building'
        assert result['asset_type_confidence'] == 0.75
    
    @pytest.mark.asyncio
    async def test_validate_extracted_data_valid(self, agent):
        """Test validation with valid data"""
        data = {
            'asset_type': 'office building',
            'location': 'New York',
            'asset_value': 1000000,
            'coverage_limit': 800000,
            'credit_rating': 'AA'
        }
        
        result = await agent.validate_extracted_data(data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_extracted_data_invalid(self, agent):
        """Test validation with invalid data"""
        data = {
            'asset_value': 20_000_000_000,  # Exceeds max range
            'coverage_limit': 1000000,
            'credit_rating': 'INVALID'  # Invalid rating
        }
        
        result = await agent.validate_extracted_data(data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert len(result.errors) >= 2  # Missing required fields + invalid asset value
        assert len(result.warnings) >= 1  # Invalid credit rating
    
    @pytest.mark.asyncio
    async def test_validate_extracted_data_warnings(self, agent):
        """Test validation with data that generates warnings"""
        data = {
            'asset_type': 'office building',
            'location': 'New York',
            'asset_value': 800000,
            'coverage_limit': 1000000,  # Exceeds asset value
        }
        
        result = await agent.validate_extracted_data(data)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True  # No errors, just warnings
        assert len(result.warnings) >= 1
        assert any('exceeds asset value' in warning for warning in result.warnings)
    
    @pytest.mark.asyncio
    async def test_process_document_text_complete(self, agent):
        """Test complete document text processing"""
        # Mock all the extraction methods
        agent.extract_risk_parameters = AsyncMock(return_value=(
            RiskParameters(
                id="test",
                application_id="test",
                asset_value=Decimal('1000000'),
                asset_type='office building',
                location='New York'
            ),
            {'location': 0.95}
        ))
        
        agent.extract_financial_data = AsyncMock(return_value=(
            FinancialData(
                id="test",
                application_id="test",
                revenue=Decimal('5000000'),
                credit_rating='AA'
            ),
            {'credit_rating': 0.9}
        ))
        
        agent.extract_geographic_info = AsyncMock(return_value={
            'region': 'North America'
        })
        
        agent.validate_extracted_data = AsyncMock(return_value=ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[]
        ))
        
        text = "Office building in New York with $1M asset value, company revenue $5M, AA rated"
        
        result = await agent.process_document_text(text)
        
        assert isinstance(result, ExtractionResult)
        assert 'risk_parameters' in result.extracted_data
        assert 'financial_data' in result.extracted_data
        assert 'geographic_info' in result.extracted_data
        assert result.validation_result.is_valid is True
        assert len(result.confidence_scores) >= 2
    
    def test_parse_currency_amount_basic(self, agent):
        """Test currency amount parsing"""
        # Test basic amounts
        assert agent._parse_currency_amount("$1,000,000") == Decimal('1000000')
        assert agent._parse_currency_amount("€500,000") == Decimal('500000')
        assert agent._parse_currency_amount("1.5 million") == Decimal('1500000')
        assert agent._parse_currency_amount("2.5B") == Decimal('2500000000')
    
    def test_parse_currency_amount_edge_cases(self, agent):
        """Test currency amount parsing edge cases"""
        # Test edge cases
        assert agent._parse_currency_amount("") is None
        assert agent._parse_currency_amount("invalid") is None
        assert agent._parse_currency_amount("$0") == Decimal('0')
    
    def test_extract_pattern_data(self, agent):
        """Test pattern-based data extraction"""
        text = """
        Asset value: $2,500,000
        Coverage limit: $2,000,000
        Construction: steel frame
        Occupancy: office
        """
        
        result = agent._extract_pattern_data(text)
        
        assert 'asset_value' in result
        assert 'coverage_limit' in result
        assert 'construction_type' in result
        assert 'occupancy' in result
        assert result['construction_type'] == 'steel'
        assert result['occupancy'] == 'office'
    
    def test_extract_financial_amounts(self, agent):
        """Test financial amount extraction"""
        text = """
        Annual revenue: $10 million
        Total assets: $50M
        Liabilities: $20 million
        """
        
        result = agent._extract_financial_amounts(text)
        
        assert 'revenue' in result
        assert 'assets' in result
        assert 'liabilities' in result
        assert result['revenue'] == Decimal('10000000')
        assert result['assets'] == Decimal('50000000')
        assert result['liabilities'] == Decimal('20000000')
    
    def test_extract_credit_rating(self, agent):
        """Test credit rating extraction"""
        text = "The company has a credit rating of AA+ from Standard & Poor's"
        
        result = agent._extract_credit_rating(text)
        
        assert result == 'AA+'
    
    def test_extract_credit_rating_not_found(self, agent):
        """Test credit rating extraction when not found"""
        text = "No credit rating information available"
        
        result = agent._extract_credit_rating(text)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_extract_ner_entities_mocked(self, agent):
        """Test NER entity extraction with mocked model"""
        # Mock the tokenizer and model outputs
        agent.ner_tokenizer.return_value = {
            'input_ids': torch.tensor([[101, 1234, 5678, 102]]),  # Mock token IDs
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        agent.ner_tokenizer.convert_ids_to_tokens.return_value = ['[CLS]', 'New', 'York', '[SEP]']
        
        # Mock model output
        mock_logits = torch.randn(1, 4, 9)  # Batch, sequence, num_labels
        mock_output = Mock()
        mock_output.logits = mock_logits
        agent.ner_model.return_value = mock_output
        agent.ner_model.config.id2label = {0: 'O', 1: 'B-LOC', 2: 'I-LOC'}
        
        text = "New York"
        
        entities = await agent._extract_ner_entities(text)
        
        assert isinstance(entities, list)
        # The exact results depend on the mocked outputs, but we verify the structure
        for entity in entities:
            assert 'word' in entity
            assert 'label' in entity
            assert 'confidence' in entity
    
    @pytest.mark.asyncio
    async def test_analyze_financial_sentiment_mocked(self, agent):
        """Test financial sentiment analysis with mocked classifier"""
        agent.financial_classifier.return_value = [
            {'label': 'positive', 'score': 0.85}
        ]
        
        text = "The company reported strong financial performance with increased profits"
        
        result = await agent._analyze_financial_sentiment(text)
        
        assert result is not None
        assert result['label'] == 'positive'
        assert result['score'] == 0.85
    
    @pytest.mark.asyncio
    async def test_analyze_financial_sentiment_no_financial_content(self, agent):
        """Test financial sentiment analysis with no financial content"""
        text = "This is a general description without financial information"
        
        result = await agent._analyze_financial_sentiment(text)
        
        assert result is None
    
    def test_initialization_patterns(self, agent):
        """Test that patterns are properly initialized"""
        assert hasattr(agent, 'patterns')
        assert 'currency_amounts' in agent.patterns
        assert 'asset_values' in agent.patterns
        assert 'credit_ratings' in agent.patterns
        
        # Test that patterns are compiled regex objects
        import re
        assert isinstance(agent.patterns['currency_amounts'], re.Pattern)
    
    def test_initialization_validation_rules(self, agent):
        """Test that validation rules are properly initialized"""
        assert hasattr(agent, 'validation_rules')
        assert 'asset_value_range' in agent.validation_rules
        assert 'required_fields' in agent.validation_rules
        assert 'valid_credit_ratings' in agent.validation_rules
        
        # Test validation rule values
        assert agent.validation_rules['asset_value_range'] == (0, 10_000_000_000)
        assert 'asset_type' in agent.validation_rules['required_fields']
        assert 'AAA' in agent.validation_rules['valid_credit_ratings']


class TestDataExtractionAgentIntegration:
    """Integration tests that test multiple components together"""
    
    @pytest.fixture
    def agent(self, mock_models):
        """Create agent for integration tests"""
        agent = DataExtractionAgent()
        
        # Mock models but allow pattern matching to work
        agent.ner_model = Mock()
        agent.ner_tokenizer = Mock()
        agent.finbert_model = Mock()
        agent.finbert_tokenizer = Mock()
        agent.financial_classifier = Mock()
        agent.layout_model = Mock()
        agent.layout_tokenizer = Mock()
        agent.zero_shot_classifier = Mock()
        agent.embeddings_model = Mock()
        agent.device = torch.device("cpu")
        
        return agent
    
    def test_pattern_extraction_integration(self, agent):
        """Test that pattern extraction works with realistic text"""
        text = """
        PROPERTY DETAILS:
        Asset Value: $5,500,000
        Coverage Limit: $5,000,000
        Location: 123 Business Park Drive, Atlanta, GA
        Construction: Steel frame with concrete floors
        Occupancy: Office building
        
        FINANCIAL INFORMATION:
        Annual Revenue: $25 million
        Total Assets: $100M
        Credit Rating: A+
        """
        
        # Test pattern extraction
        pattern_data = agent._extract_pattern_data(text)
        financial_data = agent._extract_financial_amounts(text)
        credit_rating = agent._extract_credit_rating(text)
        
        # Verify extracted data
        assert pattern_data['asset_value'] == Decimal('5500000')
        assert pattern_data['coverage_limit'] == Decimal('5000000')
        assert pattern_data['construction_type'] == 'steel'
        assert pattern_data['occupancy'] == 'office'
        
        assert financial_data['revenue'] == Decimal('25000000')
        assert financial_data['assets'] == Decimal('100000000')
        
        assert credit_rating == 'A+'
    
    @pytest.mark.asyncio
    async def test_validation_integration(self, agent):
        """Test validation with realistic extracted data"""
        # Valid data
        valid_data = {
            'asset_type': 'office building',
            'location': 'Atlanta, GA',
            'asset_value': 5500000,
            'coverage_limit': 5000000,
            'revenue': 25000000,
            'credit_rating': 'A+'
        }
        
        result = await agent.validate_extracted_data(valid_data)
        assert result.is_valid is True
        assert len(result.errors) == 0
        
        # Invalid data
        invalid_data = {
            'asset_value': 15_000_000_000,  # Too high
            'coverage_limit': 5000000,
            'credit_rating': 'INVALID'
        }
        
        result = await agent.validate_extracted_data(invalid_data)
        assert result.is_valid is False
        assert len(result.errors) >= 2  # Missing required fields + invalid asset value
        assert len(result.warnings) >= 1  # Invalid credit rating


if __name__ == "__main__":
    pytest.main([__file__])