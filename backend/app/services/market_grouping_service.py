"""
Market Grouping Service

This service provides business logic for market classification and grouping operations:
- Market identification and classification
- Document grouping and relationship mapping
- Market analytics and reporting
- Caching and performance optimization

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.2
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import hashlib
from dataclasses import asdict

from ..agents.market_grouping_agent import (
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
from ..models.schemas import Document


# Configure logging
logger = logging.getLogger(__name__)


class MarketGroupingService:
    """
    Service layer for market grouping operations
    
    Provides caching, validation, and business logic for market classification
    and document grouping functionality.
    """
    
    def __init__(self):
        """Initialize the market grouping service"""
        self.logger = logging.getLogger(__name__)
        self.agent = MarketGroupingAgent()
        
        # Simple in-memory cache (in production, use Redis or similar)
        self._cache = {}
        self._cache_ttl = timedelta(hours=1)  # Cache for 1 hour
        
        # Performance tracking
        self._performance_metrics = {
            'market_identifications': 0,
            'document_groupings': 0,
            'relationship_mappings': 0,
            'analytics_generations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def identify_market(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> MarketIdentification:
        """
        Identify market classification from content and metadata with caching
        
        Args:
            content: Document content to analyze
            metadata: Optional document metadata
            
        Returns:
            MarketIdentification with classifications and confidence scores
        """
        try:
            # Create cache key
            cache_key = self._create_cache_key("market_id", content, metadata or {})
            
            # Check cache first
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self._performance_metrics['cache_hits'] += 1
                return self._deserialize_market_identification(cached_result)
            
            self._performance_metrics['cache_misses'] += 1
            
            # Validate inputs
            if not content or not isinstance(content, str):
                raise ValueError("Content must be a non-empty string")
            
            if metadata is None:
                metadata = {}
            
            # Perform market identification
            result = self.agent.identify_market(content, metadata)
            
            # Cache the result
            self._set_cache(cache_key, self._serialize_market_identification(result))
            
            # Update metrics
            self._performance_metrics['market_identifications'] += 1
            
            self.logger.info(f"Market identification completed with confidence: {result.confidence_scores.get('overall', 0)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in market identification service: {e}")
            raise
    
    async def classify_geographic_market(
        self, 
        location: str, 
        content: str = ""
    ) -> Tuple[GeographicMarket, float]:
        """
        Classify geographic market with validation and caching
        
        Args:
            location: Location string to classify
            content: Additional content for context
            
        Returns:
            Tuple of (GeographicMarket, confidence_score)
        """
        try:
            # Validate inputs
            if not location or not isinstance(location, str):
                raise ValueError("Location must be a non-empty string")
            
            # Create cache key
            cache_key = self._create_cache_key("geo_classify", location, content)
            
            # Check cache
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self._performance_metrics['cache_hits'] += 1
                return GeographicMarket(cached_result['market']), cached_result['confidence']
            
            self._performance_metrics['cache_misses'] += 1
            
            # Perform classification
            market, confidence = self.agent.classify_geographic_market(location, content)
            
            # Cache result
            self._set_cache(cache_key, {'market': market.value, 'confidence': confidence})
            
            return market, confidence
            
        except Exception as e:
            self.logger.error(f"Error in geographic market classification service: {e}")
            raise
    
    async def group_documents_by_industry(
        self, 
        documents: List[Document]
    ) -> Dict[str, List[Document]]:
        """
        Group documents by industry with validation and optimization
        
        Args:
            documents: List of documents to group
            
        Returns:
            Dictionary mapping industry sectors to document lists
        """
        try:
            # Validate inputs
            if not documents:
                return {}
            
            if not all(isinstance(doc, Document) for doc in documents):
                raise ValueError("All items must be Document objects")
            
            # Create cache key based on document IDs
            doc_ids = sorted([doc.id for doc in documents])
            cache_key = self._create_cache_key("industry_group", *doc_ids)
            
            # Check cache
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self._performance_metrics['cache_hits'] += 1
                return self._deserialize_document_groups(cached_result, documents)
            
            self._performance_metrics['cache_misses'] += 1
            
            # Perform grouping
            groups = self.agent.group_by_industry(documents)
            
            # Cache result (store document IDs, not full objects)
            serialized_groups = {}
            for industry, docs in groups.items():
                serialized_groups[industry] = [doc.id for doc in docs]
            
            self._set_cache(cache_key, serialized_groups)
            
            # Update metrics
            self._performance_metrics['document_groupings'] += 1
            
            self.logger.info(f"Grouped {len(documents)} documents into {len(groups)} industry groups")
            return groups
            
        except Exception as e:
            self.logger.error(f"Error in industry grouping service: {e}")
            raise
    
    async def map_document_relationships(
        self, 
        documents: List[Document]
    ) -> List[DocumentRelationship]:
        """
        Map relationships between documents with optimization
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            List of DocumentRelationship objects
        """
        try:
            # Validate inputs
            if len(documents) < 2:
                return []
            
            if not all(isinstance(doc, Document) for doc in documents):
                raise ValueError("All items must be Document objects")
            
            # Create cache key
            doc_ids = sorted([doc.id for doc in documents])
            cache_key = self._create_cache_key("relationships", *doc_ids)
            
            # Check cache
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self._performance_metrics['cache_hits'] += 1
                return self._deserialize_relationships(cached_result)
            
            self._performance_metrics['cache_misses'] += 1
            
            # Perform relationship mapping
            relationships = self.agent.map_relationships(documents)
            
            # Cache result
            serialized_relationships = [
                {
                    'document_id_1': rel.document_id_1,
                    'document_id_2': rel.document_id_2,
                    'relationship_type': rel.relationship_type,
                    'similarity_score': rel.similarity_score,
                    'common_entities': rel.common_entities,
                    'relationship_evidence': rel.relationship_evidence
                }
                for rel in relationships
            ]
            
            self._set_cache(cache_key, serialized_relationships)
            
            # Update metrics
            self._performance_metrics['relationship_mappings'] += 1
            
            self.logger.info(f"Found {len(relationships)} relationships among {len(documents)} documents")
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error in relationship mapping service: {e}")
            raise
    
    async def create_market_groups(
        self, 
        documents: List[Document]
    ) -> List[MarketGroup]:
        """
        Create market groups from documents with validation
        
        Args:
            documents: List of documents to group
            
        Returns:
            List of MarketGroup objects
        """
        try:
            # Validate inputs
            if not documents:
                return []
            
            if not all(isinstance(doc, Document) for doc in documents):
                raise ValueError("All items must be Document objects")
            
            # Perform grouping
            market_groups = self.agent.create_market_groups(documents)
            
            # Update metrics
            self._performance_metrics['document_groupings'] += 1
            
            self.logger.info(f"Created {len(market_groups)} market groups from {len(documents)} documents")
            return market_groups
            
        except Exception as e:
            self.logger.error(f"Error in market group creation service: {e}")
            raise
    
    async def generate_market_analytics(
        self, 
        documents: List[Document]
    ) -> MarketAnalytics:
        """
        Generate market analytics with caching and validation
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            MarketAnalytics with comprehensive market analysis
        """
        try:
            # Validate inputs
            if not documents:
                raise ValueError("At least one document is required for analytics")
            
            if not all(isinstance(doc, Document) for doc in documents):
                raise ValueError("All items must be Document objects")
            
            # Create cache key based on document IDs and timestamps
            doc_signatures = [f"{doc.id}:{doc.upload_timestamp.isoformat()}" for doc in documents]
            cache_key = self._create_cache_key("analytics", *sorted(doc_signatures))
            
            # Check cache
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self._performance_metrics['cache_hits'] += 1
                return self._deserialize_analytics(cached_result)
            
            self._performance_metrics['cache_misses'] += 1
            
            # Generate analytics
            analytics = self.agent.generate_market_analytics(documents)
            
            # Cache result
            self._set_cache(cache_key, self._serialize_analytics(analytics))
            
            # Update metrics
            self._performance_metrics['analytics_generations'] += 1
            
            self.logger.info(f"Generated analytics for {analytics.total_documents} documents")
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error in market analytics service: {e}")
            raise
    
    async def filter_documents_by_market(
        self, 
        documents: List[Document], 
        market_type: MarketType, 
        market_value: str
    ) -> List[Document]:
        """
        Filter documents by market criteria with validation
        
        Args:
            documents: List of documents to filter
            market_type: Type of market filter
            market_value: Market value to filter by
            
        Returns:
            Filtered list of documents
        """
        try:
            # Validate inputs
            if not documents:
                return []
            
            if not all(isinstance(doc, Document) for doc in documents):
                raise ValueError("All items must be Document objects")
            
            if not isinstance(market_type, MarketType):
                raise ValueError("market_type must be a MarketType enum")
            
            if not market_value or not isinstance(market_value, str):
                raise ValueError("market_value must be a non-empty string")
            
            # Perform filtering
            filtered_docs = self.agent.filter_by_market(documents, market_type, market_value)
            
            self.logger.info(f"Filtered {len(documents)} documents to {len(filtered_docs)} by {market_type.value}: {market_value}")
            return filtered_docs
            
        except Exception as e:
            self.logger.error(f"Error in market filtering service: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get service performance metrics
        
        Returns:
            Dictionary with performance metrics and cache statistics
        """
        cache_hit_rate = 0.0
        total_requests = self._performance_metrics['cache_hits'] + self._performance_metrics['cache_misses']
        
        if total_requests > 0:
            cache_hit_rate = self._performance_metrics['cache_hits'] / total_requests
        
        return {
            **self._performance_metrics,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._cache),
            'total_requests': total_requests
        }
    
    def clear_cache(self) -> None:
        """Clear the service cache"""
        self._cache.clear()
        self.logger.info("Market grouping service cache cleared")
    
    # Private helper methods
    
    def _create_cache_key(self, operation: str, *args) -> str:
        """Create a cache key from operation and arguments"""
        key_data = f"{operation}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired"""
        if key in self._cache:
            item, timestamp = self._cache[key]
            if datetime.utcnow() - timestamp < self._cache_ttl:
                return item
            else:
                # Remove expired item
                del self._cache[key]
        return None
    
    def _set_cache(self, key: str, value: Any) -> None:
        """Set item in cache with timestamp"""
        self._cache[key] = (value, datetime.utcnow())
        
        # Simple cache size management (keep last 1000 items)
        if len(self._cache) > 1000:
            # Remove oldest 100 items
            oldest_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][1])[:100]
            for old_key in oldest_keys:
                del self._cache[old_key]
    
    def _serialize_market_identification(self, result: MarketIdentification) -> Dict[str, Any]:
        """Serialize MarketIdentification for caching"""
        return {
            'geographic_market': result.geographic_market.value,
            'industry_market': result.industry_market.value,
            'business_line_market': result.business_line_market.value,
            'confidence_scores': result.confidence_scores,
            'identified_entities': result.identified_entities,
            'market_indicators': result.market_indicators
        }
    
    def _deserialize_market_identification(self, data: Dict[str, Any]) -> MarketIdentification:
        """Deserialize MarketIdentification from cache"""
        return MarketIdentification(
            geographic_market=GeographicMarket(data['geographic_market']),
            industry_market=IndustryMarket(data['industry_market']),
            business_line_market=BusinessLineMarket(data['business_line_market']),
            confidence_scores=data['confidence_scores'],
            identified_entities=data['identified_entities'],
            market_indicators=data['market_indicators']
        )
    
    def _deserialize_document_groups(
        self, 
        serialized_groups: Dict[str, List[str]], 
        documents: List[Document]
    ) -> Dict[str, List[Document]]:
        """Deserialize document groups from cache"""
        doc_map = {doc.id: doc for doc in documents}
        groups = {}
        
        for industry, doc_ids in serialized_groups.items():
            groups[industry] = [doc_map[doc_id] for doc_id in doc_ids if doc_id in doc_map]
        
        return groups
    
    def _deserialize_relationships(self, serialized_rels: List[Dict[str, Any]]) -> List[DocumentRelationship]:
        """Deserialize relationships from cache"""
        relationships = []
        
        for rel_data in serialized_rels:
            relationship = DocumentRelationship(
                document_id_1=rel_data['document_id_1'],
                document_id_2=rel_data['document_id_2'],
                relationship_type=rel_data['relationship_type'],
                similarity_score=rel_data['similarity_score'],
                common_entities=rel_data['common_entities'],
                relationship_evidence=rel_data['relationship_evidence']
            )
            relationships.append(relationship)
        
        return relationships
    
    def _serialize_analytics(self, analytics: MarketAnalytics) -> Dict[str, Any]:
        """Serialize MarketAnalytics for caching"""
        return {
            'total_documents': analytics.total_documents,
            'market_distribution': analytics.market_distribution,
            'trend_analysis': analytics.trend_analysis,
            'top_markets': analytics.top_markets,
            'growth_rates': analytics.growth_rates,
            'seasonal_patterns': analytics.seasonal_patterns
        }
    
    def _deserialize_analytics(self, data: Dict[str, Any]) -> MarketAnalytics:
        """Deserialize MarketAnalytics from cache"""
        return MarketAnalytics(
            total_documents=data['total_documents'],
            market_distribution=data['market_distribution'],
            trend_analysis=data['trend_analysis'],
            top_markets=data['top_markets'],
            growth_rates=data['growth_rates'],
            seasonal_patterns=data['seasonal_patterns']
        )