"""
Tests for Market Grouping Agent

This module tests the market classification and grouping functionality including:
- Market identification from email content and metadata
- Geographic market classification using zero-shot learning
- Industry sector grouping with document clustering
- Relationship mapping between related documents
- Market-based filtering and reporting capabilities
- Market analytics and trend analysis

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.2
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from app.agents.market_grouping_agent import (
    MarketGroupingAgent,
    MarketIdentification,
    DocumentRelationship,
    MarketGroup,
    MarketAnalytics,
    GeographicMarket,
    IndustryMarket,
    BusinessLineMarket,
    MarketType
)
from app.models.schemas import Document, DocumentType


class TestMarketGroupingAgent:
    """Test cases for MarketGroupingAgent"""
    
    @pytest.fixture
    def agent(self):
        """Create a MarketGroupingAgent instance for testing"""
        with patch('app.agents.market_grouping_agent.pipeline') as mock_pipeline, \
             patch('app.agents.market_grouping_agent.SentenceTransformer') as mock_transformer:
            
            # Mock the pipeline for zero-shot classification
            mock_classifier = Mock()
            mock_classifier.return_value = {
                'labels': ['north_america', 'europe', 'asia_pacific'],
                'scores': [0.8, 0.15, 0.05]
            }
            mock_pipeline.return_value = mock_classifier
            
            # Mock sentence transformer
            mock_transformer_instance = Mock()
            mock_transformer_instance.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            mock_transformer.return_value = mock_transformer_instance
            
            agent = MarketGroupingAgent()
            agent.zero_shot_classifier = mock_classifier
            agent.sentence_transformer = mock_transformer_instance
            
            return agent
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        documents = [
            Document(
                id="doc1",
                filename="us_property_risk.pdf",
                document_type=DocumentType.PDF,
                file_path="/uploads/doc1.pdf",
                processed=True,
                upload_timestamp=datetime.utcnow(),
                metadata={
                    "subject": "Property Insurance - New York Office Building",
                    "from": "broker@usinsurance.com",
                    "body": "Property insurance for office building in Manhattan, New York. Construction: steel frame, occupancy: office."
                }
            ),
            Document(
                id="doc2",
                filename="uk_marine_cargo.pdf",
                document_type=DocumentType.PDF,
                file_path="/uploads/doc2.pdf",
                processed=True,
                upload_timestamp=datetime.utcnow() - timedelta(days=1),
                metadata={
                    "subject": "Marine Cargo Insurance - London to Singapore",
                    "from": "underwriter@londonmarket.co.uk",
                    "body": "Marine cargo insurance for electronics shipment from London to Singapore. Vessel: container ship."
                }
            ),
            Document(
                id="doc3",
                filename="energy_facility_texas.pdf",
                document_type=DocumentType.PDF,
                file_path="/uploads/doc3.pdf",
                processed=True,
                upload_timestamp=datetime.utcnow() - timedelta(days=2),
                metadata={
                    "subject": "Energy Facility Insurance - Texas Oil Refinery",
                    "from": "agent@energyinsure.com",
                    "body": "Property and liability insurance for oil refinery in Houston, Texas. High-value petrochemical facility."
                }
            ),
            Document(
                id="doc4",
                filename="cyber_liability_tech.pdf",
                document_type=DocumentType.EMAIL,
                file_path="/uploads/doc4.msg",
                processed=True,
                upload_timestamp=datetime.utcnow() - timedelta(days=3),
                metadata={
                    "subject": "Cyber Liability - Technology Company",
                    "from": "cyber@techinsure.com",
                    "body": "Cyber liability insurance for software company. Data breach protection, network security coverage."
                }
            )
        ]
        return documents
    
    def test_identify_market_basic(self, agent, sample_documents):
        """Test basic market identification functionality"""
        doc = sample_documents[0]  # US property document
        content = doc.metadata["body"]
        metadata = doc.metadata
        
        result = agent.identify_market(content, metadata)
        
        assert isinstance(result, MarketIdentification)
        assert result.geographic_market in GeographicMarket
        assert result.industry_market in IndustryMarket
        assert result.business_line_market in BusinessLineMarket
        assert "geographic" in result.confidence_scores
        assert "industry" in result.confidence_scores
        assert "business_line" in result.confidence_scores
        assert "overall" in result.confidence_scores
        assert isinstance(result.identified_entities, dict)
        assert isinstance(result.market_indicators, list)
    
    def test_classify_geographic_market_north_america(self, agent):
        """Test geographic market classification for North America"""
        location = "New York, USA"
        content = "Property insurance for office building in Manhattan, New York"
        
        market, confidence = agent.classify_geographic_market(location, content)
        
        assert market == GeographicMarket.NORTH_AMERICA
        assert confidence > 0.0
    
    def test_classify_geographic_market_europe(self, agent):
        """Test geographic market classification for Europe"""
        location = "London, UK"
        content = "Marine insurance from London Lloyd's market"
        
        market, confidence = agent.classify_geographic_market(location, content)
        
        assert market == GeographicMarket.EUROPE
        assert confidence > 0.0
    
    def test_classify_geographic_market_unknown(self, agent):
        """Test geographic market classification for unknown location"""
        location = "Unknown Location"
        content = "Generic insurance policy"
        
        market, confidence = agent.classify_geographic_market(location, content)
        
        # Should return some classification, even if unknown
        assert market in GeographicMarket
        assert confidence >= 0.0
    
    def test_group_by_industry_basic(self, agent, sample_documents):
        """Test basic industry grouping functionality"""
        groups = agent.group_by_industry(sample_documents)
        
        assert isinstance(groups, dict)
        assert len(groups) > 0
        
        # Check that all documents are assigned to some group
        total_docs_in_groups = sum(len(docs) for docs in groups.values())
        assert total_docs_in_groups == len(sample_documents)
        
        # Check that each group contains Document objects
        for group_name, docs in groups.items():
            assert isinstance(group_name, str)
            assert isinstance(docs, list)
            for doc in docs:
                assert isinstance(doc, Document)
    
    def test_group_by_industry_energy(self, agent, sample_documents):
        """Test that energy documents are properly grouped"""
        groups = agent.group_by_industry(sample_documents)
        
        # Find energy-related documents
        energy_docs = []
        for group_name, docs in groups.items():
            if "energy" in group_name.lower() or any("energy" in doc.filename.lower() or 
                                                   "oil" in doc.metadata.get("body", "").lower() 
                                                   for doc in docs):
                energy_docs.extend(docs)
        
        # Should have at least the Texas oil refinery document
        assert len(energy_docs) > 0
    
    def test_map_relationships_basic(self, agent, sample_documents):
        """Test basic relationship mapping functionality"""
        relationships = agent.map_relationships(sample_documents)
        
        assert isinstance(relationships, list)
        
        # Check relationship objects
        for rel in relationships:
            assert isinstance(rel, DocumentRelationship)
            assert rel.document_id_1 in [doc.id for doc in sample_documents]
            assert rel.document_id_2 in [doc.id for doc in sample_documents]
            assert rel.document_id_1 != rel.document_id_2
            assert isinstance(rel.relationship_type, str)
            assert 0.0 <= rel.similarity_score <= 1.0
            assert isinstance(rel.common_entities, list)
            assert isinstance(rel.relationship_evidence, list)
    
    def test_map_relationships_same_broker(self, agent):
        """Test relationship mapping for documents from same broker"""
        docs = [
            Document(
                id="doc1",
                filename="policy1.pdf",
                document_type=DocumentType.PDF,
                file_path="/uploads/doc1.pdf",
                processed=True,
                upload_timestamp=datetime.utcnow(),
                metadata={
                    "from": "broker@samecompany.com",
                    "subject": "Property Insurance Quote",
                    "body": "Property insurance quote for client ABC Corp"
                }
            ),
            Document(
                id="doc2",
                filename="policy2.pdf",
                document_type=DocumentType.PDF,
                file_path="/uploads/doc2.pdf",
                processed=True,
                upload_timestamp=datetime.utcnow(),
                metadata={
                    "from": "broker@samecompany.com",
                    "subject": "Liability Insurance Quote",
                    "body": "Liability insurance quote for client ABC Corp"
                }
            )
        ]
        
        relationships = agent.map_relationships(docs)
        
        # Should find relationship between documents from same broker
        assert len(relationships) > 0
        
        # Check for broker relationship or same client relationship
        broker_relationships = [r for r in relationships if 
                             r.relationship_type in ["same_broker", "same_client"]]
        assert len(broker_relationships) > 0
    
    def test_create_market_groups(self, agent, sample_documents):
        """Test market group creation"""
        groups = agent.create_market_groups(sample_documents)
        
        assert isinstance(groups, list)
        assert len(groups) > 0
        
        # Check group structure
        for group in groups:
            assert isinstance(group, MarketGroup)
            assert group.group_id
            assert group.market_type in MarketType
            assert group.market_value
            assert isinstance(group.documents, list)
            assert len(group.documents) > 0
            assert isinstance(group.characteristics, dict)
            assert isinstance(group.created_at, datetime)
            assert isinstance(group.updated_at, datetime)
            
            # Verify all document IDs exist in sample documents
            sample_doc_ids = [doc.id for doc in sample_documents]
            for doc_id in group.documents:
                assert doc_id in sample_doc_ids
    
    def test_create_market_groups_types(self, agent, sample_documents):
        """Test that different market group types are created"""
        groups = agent.create_market_groups(sample_documents)
        
        market_types = [group.market_type for group in groups]
        
        # Should have multiple market types
        assert MarketType.GEOGRAPHIC in market_types
        assert MarketType.INDUSTRY in market_types
        assert MarketType.BUSINESS_LINE in market_types
    
    def test_generate_market_analytics(self, agent, sample_documents):
        """Test market analytics generation"""
        analytics = agent.generate_market_analytics(sample_documents)
        
        assert isinstance(analytics, MarketAnalytics)
        assert analytics.total_documents == len(sample_documents)
        assert isinstance(analytics.market_distribution, dict)
        assert isinstance(analytics.trend_analysis, dict)
        assert isinstance(analytics.top_markets, list)
        assert isinstance(analytics.growth_rates, dict)
        assert isinstance(analytics.seasonal_patterns, dict)
        
        # Check market distribution
        assert len(analytics.market_distribution) > 0
        for market_name, count in analytics.market_distribution.items():
            assert isinstance(market_name, str)
            assert isinstance(count, int)
            assert count > 0
        
        # Check top markets format
        for market_name, count in analytics.top_markets:
            assert isinstance(market_name, str)
            assert isinstance(count, int)
    
    def test_filter_by_market_geographic(self, agent, sample_documents):
        """Test filtering documents by geographic market"""
        # Filter for North American documents
        filtered_docs = agent.filter_by_market(
            sample_documents, 
            MarketType.GEOGRAPHIC, 
            GeographicMarket.NORTH_AMERICA.value
        )
        
        assert isinstance(filtered_docs, list)
        
        # All returned documents should be from the original list
        for doc in filtered_docs:
            assert doc in sample_documents
    
    def test_filter_by_market_industry(self, agent, sample_documents):
        """Test filtering documents by industry market"""
        # Filter for energy/utilities documents
        filtered_docs = agent.filter_by_market(
            sample_documents,
            MarketType.INDUSTRY,
            IndustryMarket.ENERGY_UTILITIES.value
        )
        
        assert isinstance(filtered_docs, list)
        
        # All returned documents should be from the original list
        for doc in filtered_docs:
            assert doc in sample_documents
    
    def test_filter_by_market_business_line(self, agent, sample_documents):
        """Test filtering documents by business line market"""
        # Filter for property documents
        filtered_docs = agent.filter_by_market(
            sample_documents,
            MarketType.BUSINESS_LINE,
            BusinessLineMarket.PROPERTY.value
        )
        
        assert isinstance(filtered_docs, list)
        
        # All returned documents should be from the original list
        for doc in filtered_docs:
            assert doc in sample_documents
    
    def test_extract_entities_basic(self, agent):
        """Test entity extraction functionality"""
        content = "John Smith from ABC Corporation in New York is requesting insurance for their facility."
        
        # Mock NER pipeline
        mock_ner_results = [
            {'entity_group': 'PERSON', 'word': 'John Smith'},
            {'entity_group': 'ORG', 'word': 'ABC Corporation'},
            {'entity_group': 'LOC', 'word': 'New York'}
        ]
        
        with patch.object(agent, 'ner_pipeline') as mock_ner:
            mock_ner.return_value = mock_ner_results
            
            entities = agent._extract_entities(content)
            
            assert isinstance(entities, dict)
            assert 'PERSON' in entities
            assert 'ORG' in entities
            assert 'LOC' in entities
            assert 'MISC' in entities
            
            assert 'John Smith' in entities['PERSON']
            assert 'ABC Corporation' in entities['ORG']
            assert 'New York' in entities['LOC']
    
    def test_extract_entities_no_ner(self, agent):
        """Test entity extraction when NER pipeline is not available"""
        content = "Some content without NER processing"
        
        # Disable NER pipeline
        agent.ner_pipeline = None
        
        entities = agent._extract_entities(content)
        
        assert isinstance(entities, dict)
        assert 'PERSON' in entities
        assert 'ORG' in entities
        assert 'LOC' in entities
        assert 'MISC' in entities
        
        # All lists should be empty
        for entity_list in entities.values():
            assert len(entity_list) == 0
    
    def test_classify_by_patterns_geographic(self, agent):
        """Test pattern-based classification for geographic markets"""
        text = "Insurance policy for property in New York, United States"
        
        market, confidence = agent._classify_by_patterns(text, agent.geographic_patterns)
        
        assert market == GeographicMarket.NORTH_AMERICA
        assert confidence > 0.0
    
    def test_classify_by_patterns_industry(self, agent):
        """Test pattern-based classification for industry markets"""
        text = "Oil refinery petrochemical facility energy production"
        
        market, confidence = agent._classify_by_patterns(text, agent.industry_patterns)
        
        assert market == IndustryMarket.ENERGY_UTILITIES
        assert confidence > 0.0
    
    def test_classify_by_patterns_no_match(self, agent):
        """Test pattern-based classification with no matches"""
        text = "Generic text with no specific patterns"
        
        market, confidence = agent._classify_by_patterns(text, agent.geographic_patterns)
        
        # Should return first option with 0 confidence
        assert market in GeographicMarket
        assert confidence == 0.0
    
    def test_get_document_content(self, agent, sample_documents):
        """Test document content extraction"""
        doc = sample_documents[0]
        
        content = agent._get_document_content(doc)
        
        assert isinstance(content, str)
        assert len(content) > 0
        assert doc.filename in content
        
        # Should include metadata if available
        if doc.metadata and 'subject' in doc.metadata:
            assert doc.metadata['subject'] in content
    
    def test_cluster_documents_by_similarity(self, agent, sample_documents):
        """Test document clustering by similarity"""
        # Mock sentence transformer to return predictable embeddings
        mock_embeddings = np.array([
            [0.1, 0.2, 0.3],  # Similar to next one
            [0.15, 0.25, 0.35],  # Similar to previous one
            [0.8, 0.9, 1.0],  # Different cluster
            [0.85, 0.95, 1.05]  # Similar to previous one
        ])
        
        with patch.object(agent.sentence_transformer, 'encode', return_value=mock_embeddings):
            clusters = agent._cluster_documents_by_similarity(sample_documents)
            
            assert isinstance(clusters, list)
            assert len(clusters) > 0
            
            # Check that all documents are in some cluster
            all_clustered_docs = []
            for cluster in clusters:
                all_clustered_docs.extend(cluster)
            
            assert len(all_clustered_docs) == len(sample_documents)
            
            # Check that each cluster contains Document objects
            for cluster in clusters:
                assert isinstance(cluster, list)
                for doc in cluster:
                    assert isinstance(doc, Document)
    
    def test_analyze_document_relationship_high_similarity(self, agent):
        """Test document relationship analysis with high similarity"""
        doc1 = Document(
            id="doc1",
            filename="policy1.pdf",
            document_type=DocumentType.PDF,
            file_path="/uploads/doc1.pdf",
            processed=True,
            upload_timestamp=datetime.utcnow()
        )
        
        doc2 = Document(
            id="doc2",
            filename="policy2.pdf",
            document_type=DocumentType.PDF,
            file_path="/uploads/doc2.pdf",
            processed=True,
            upload_timestamp=datetime.utcnow()
        )
        
        doc1_data = {
            'content': 'Property insurance for ABC Corporation in New York',
            'entities': {'ORG': ['ABC Corporation'], 'LOC': ['New York']},
            'metadata': {'subject': 'Property Insurance Quote'}
        }
        
        doc2_data = {
            'content': 'Liability insurance for ABC Corporation in New York',
            'entities': {'ORG': ['ABC Corporation'], 'LOC': ['New York']},
            'metadata': {'subject': 'Liability Insurance Quote'}
        }
        
        # Mock high similarity score
        with patch.object(agent.sentence_transformer, 'encode') as mock_encode:
            mock_encode.return_value = np.array([[0.1, 0.2], [0.15, 0.25]])
            
            relationship = agent._analyze_document_relationship(doc1, doc2, doc1_data, doc2_data)
            
            assert relationship is not None
            assert isinstance(relationship, DocumentRelationship)
            assert relationship.document_id_1 == doc1.id
            assert relationship.document_id_2 == doc2.id
            assert len(relationship.common_entities) > 0
            assert 'ABC Corporation' in relationship.common_entities
            assert 'New York' in relationship.common_entities
    
    def test_analyze_document_relationship_no_relationship(self, agent):
        """Test document relationship analysis with no significant relationship"""
        doc1 = Document(
            id="doc1",
            filename="policy1.pdf",
            document_type=DocumentType.PDF,
            file_path="/uploads/doc1.pdf",
            processed=True,
            upload_timestamp=datetime.utcnow()
        )
        
        doc2 = Document(
            id="doc2",
            filename="policy2.pdf",
            document_type=DocumentType.PDF,
            file_path="/uploads/doc2.pdf",
            processed=True,
            upload_timestamp=datetime.utcnow()
        )
        
        doc1_data = {
            'content': 'Property insurance for Company A in California',
            'entities': {'ORG': ['Company A'], 'LOC': ['California']},
            'metadata': {'subject': 'Property Insurance'}
        }
        
        doc2_data = {
            'content': 'Marine insurance for Company B in London',
            'entities': {'ORG': ['Company B'], 'LOC': ['London']},
            'metadata': {'subject': 'Marine Insurance'}
        }
        
        # Mock low similarity score
        with patch.object(agent.sentence_transformer, 'encode') as mock_encode:
            mock_encode.return_value = np.array([[0.1, 0.2], [0.8, 0.9]])
            
            relationship = agent._analyze_document_relationship(doc1, doc2, doc1_data, doc2_data)
            
            # Should return None for unrelated documents
            assert relationship is None
    
    def test_calculate_market_distribution(self, agent, sample_documents):
        """Test market distribution calculation"""
        distribution = agent._calculate_market_distribution(sample_documents)
        
        assert isinstance(distribution, dict)
        assert len(distribution) > 0
        
        # Check that we have geographic, industry, and business line distributions
        geo_keys = [k for k in distribution.keys() if k.startswith('geo_')]
        industry_keys = [k for k in distribution.keys() if k.startswith('industry_')]
        business_keys = [k for k in distribution.keys() if k.startswith('business_')]
        
        assert len(geo_keys) > 0
        assert len(industry_keys) > 0
        assert len(business_keys) > 0
        
        # Check that all counts are positive integers
        for count in distribution.values():
            assert isinstance(count, int)
            assert count > 0
    
    def test_analyze_market_trends(self, agent, sample_documents):
        """Test market trend analysis"""
        trends = agent._analyze_market_trends(sample_documents)
        
        assert isinstance(trends, dict)
        assert 'monthly_distribution' in trends
        assert 'trend_direction' in trends
        assert 'trend_slope' in trends
        assert 'total_months' in trends
        
        assert isinstance(trends['monthly_distribution'], dict)
        assert trends['trend_direction'] in ['increasing', 'decreasing', 'stable', 'insufficient_data']
        assert isinstance(trends['trend_slope'], (int, float))
        assert isinstance(trends['total_months'], int)
    
    def test_extract_domain(self, agent):
        """Test email domain extraction"""
        # Test valid email
        domain = agent._extract_domain("user@example.com")
        assert domain == "example.com"
        
        # Test email with subdomain
        domain = agent._extract_domain("broker@london.lloyds.co.uk")
        assert domain == "london.lloyds.co.uk"
        
        # Test invalid email
        domain = agent._extract_domain("invalid-email")
        assert domain is None
        
        # Test empty string
        domain = agent._extract_domain("")
        assert domain is None
    
    def test_error_handling_identify_market(self, agent):
        """Test error handling in market identification"""
        # Test with None content
        with pytest.raises(Exception):
            agent.identify_market(None, {})
        
        # Test with invalid metadata
        result = agent.identify_market("valid content", None)
        assert isinstance(result, MarketIdentification)
    
    def test_error_handling_group_by_industry(self, agent):
        """Test error handling in industry grouping"""
        # Test with empty list
        groups = agent.group_by_industry([])
        assert groups == {}
        
        # Test with None
        groups = agent.group_by_industry(None)
        assert groups == {}
    
    def test_error_handling_map_relationships(self, agent):
        """Test error handling in relationship mapping"""
        # Test with empty list
        relationships = agent.map_relationships([])
        assert relationships == []
        
        # Test with single document
        single_doc = [Document(
            id="doc1",
            filename="test.pdf",
            document_type=DocumentType.PDF,
            file_path="/test.pdf",
            processed=True,
            upload_timestamp=datetime.utcnow()
        )]
        
        relationships = agent.map_relationships(single_doc)
        assert relationships == []
    
    @pytest.mark.parametrize("market_type,market_value", [
        (MarketType.GEOGRAPHIC, GeographicMarket.NORTH_AMERICA.value),
        (MarketType.INDUSTRY, IndustryMarket.ENERGY_UTILITIES.value),
        (MarketType.BUSINESS_LINE, BusinessLineMarket.PROPERTY.value)
    ])
    def test_filter_by_market_parametrized(self, agent, sample_documents, market_type, market_value):
        """Test filtering by different market types and values"""
        filtered_docs = agent.filter_by_market(sample_documents, market_type, market_value)
        
        assert isinstance(filtered_docs, list)
        
        # All returned documents should be from the original list
        for doc in filtered_docs:
            assert doc in sample_documents
    
    def test_performance_large_document_set(self, agent):
        """Test performance with larger document set"""
        # Create a larger set of documents for performance testing
        large_doc_set = []
        
        for i in range(50):
            doc = Document(
                id=f"doc_{i}",
                filename=f"document_{i}.pdf",
                document_type=DocumentType.PDF,
                file_path=f"/uploads/doc_{i}.pdf",
                processed=True,
                upload_timestamp=datetime.utcnow() - timedelta(days=i),
                metadata={
                    "subject": f"Insurance Document {i}",
                    "body": f"Insurance content for document {i}"
                }
            )
            large_doc_set.append(doc)
        
        # Test that operations complete in reasonable time
        import time
        
        start_time = time.time()
        groups = agent.create_market_groups(large_doc_set)
        end_time = time.time()
        
        # Should complete within 30 seconds (adjust as needed)
        assert (end_time - start_time) < 30
        assert len(groups) > 0
        
        start_time = time.time()
        analytics = agent.generate_market_analytics(large_doc_set)
        end_time = time.time()
        
        # Should complete within 10 seconds
        assert (end_time - start_time) < 10
        assert analytics.total_documents == len(large_doc_set)


class TestMarketGroupingIntegration:
    """Integration tests for market grouping functionality"""
    
    @pytest.fixture
    def agent_with_real_models(self):
        """Create agent with real models for integration testing"""
        # This would use real models in a full integration test
        # For now, we'll mock them but test the integration flow
        agent = MarketGroupingAgent()
        return agent
    
    def test_end_to_end_market_analysis(self, agent_with_real_models):
        """Test complete end-to-end market analysis workflow"""
        # Create realistic test documents
        documents = [
            Document(
                id="energy_doc",
                filename="texas_refinery_insurance.pdf",
                document_type=DocumentType.PDF,
                file_path="/uploads/energy.pdf",
                processed=True,
                upload_timestamp=datetime.utcnow(),
                metadata={
                    "subject": "Property Insurance - Texas Oil Refinery",
                    "from": "broker@energyinsurance.com",
                    "body": "Comprehensive property and business interruption insurance for oil refinery facility in Houston, Texas. Petrochemical processing, high-value equipment."
                }
            ),
            Document(
                id="tech_doc",
                filename="silicon_valley_cyber.pdf",
                document_type=DocumentType.PDF,
                file_path="/uploads/tech.pdf",
                processed=True,
                upload_timestamp=datetime.utcnow(),
                metadata={
                    "subject": "Cyber Liability Insurance - Tech Startup",
                    "from": "agent@cyberinsure.com",
                    "body": "Cyber liability and data breach insurance for technology startup in Silicon Valley, California. Software development, cloud services."
                }
            )
        ]
        
        # Perform complete analysis
        market_groups = agent_with_real_models.create_market_groups(documents)
        relationships = agent_with_real_models.map_relationships(documents)
        analytics = agent_with_real_models.generate_market_analytics(documents)
        
        # Verify results
        assert len(market_groups) > 0
        assert isinstance(relationships, list)
        assert isinstance(analytics, MarketAnalytics)
        assert analytics.total_documents == len(documents)
    
    def test_market_classification_accuracy(self, agent_with_real_models):
        """Test accuracy of market classification"""
        # Test cases with known expected results
        test_cases = [
            {
                "content": "Property insurance for office building in New York City, United States",
                "expected_geo": GeographicMarket.NORTH_AMERICA,
                "expected_industry": IndustryMarket.REAL_ESTATE
            },
            {
                "content": "Marine cargo insurance for container shipment from London to Singapore",
                "expected_geo": GeographicMarket.EUROPE,  # Origin-based
                "expected_business": BusinessLineMarket.MARINE
            },
            {
                "content": "Cyber liability insurance for fintech company data breach protection",
                "expected_industry": IndustryMarket.FINANCIAL_SERVICES,
                "expected_business": BusinessLineMarket.CYBER
            }
        ]
        
        for test_case in test_cases:
            result = agent_with_real_models.identify_market(test_case["content"], {})
            
            # Check geographic classification if expected
            if "expected_geo" in test_case:
                # Allow for reasonable confidence threshold
                if result.confidence_scores["geographic"] > 0.3:
                    assert result.geographic_market == test_case["expected_geo"]
            
            # Check industry classification if expected
            if "expected_industry" in test_case:
                if result.confidence_scores["industry"] > 0.3:
                    assert result.industry_market == test_case["expected_industry"]
            
            # Check business line classification if expected
            if "expected_business" in test_case:
                if result.confidence_scores["business_line"] > 0.3:
                    assert result.business_line_market == test_case["expected_business"]