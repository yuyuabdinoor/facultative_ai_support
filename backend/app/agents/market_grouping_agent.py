"""
Market Grouping Agent for AI-Powered Facultative Reinsurance Decision Support System

This agent implements market classification and grouping functionality including:
- Market identification from email content and metadata
- Geographic market classification using zero-shot learning
- Industry sector grouping with document clustering
- Relationship mapping between related documents
- Market-based filtering and reporting capabilities
- Visualization data for market distribution and trends

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.2
"""

import logging
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, Counter
import json
from email.utils import parseaddr
from urllib.parse import urlparse

# ML and NLP imports
import os
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch

from ..models.schemas import Document, DocumentType, Application


# Configure logging
logger = logging.getLogger(__name__)


class MarketType(Enum):
    """Market type classification"""
    GEOGRAPHIC = "geographic"
    INDUSTRY = "industry"
    BUSINESS_LINE = "business_line"
    RELATIONSHIP = "relationship"


class GeographicMarket(Enum):
    """Geographic market classifications"""
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    MIDDLE_EAST_AFRICA = "middle_east_africa"
    LATIN_AMERICA = "latin_america"
    GLOBAL = "global"
    UNKNOWN = "unknown"


class IndustryMarket(Enum):
    """Industry market classifications"""
    ENERGY_UTILITIES = "energy_utilities"
    MANUFACTURING = "manufacturing"
    CONSTRUCTION_ENGINEERING = "construction_engineering"
    TRANSPORTATION = "transportation"
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIAL_SERVICES = "financial_services"
    RETAIL_CONSUMER = "retail_consumer"
    AGRICULTURE = "agriculture"
    MINING_RESOURCES = "mining_resources"
    MARINE_AVIATION = "marine_aviation"
    REAL_ESTATE = "real_estate"
    GOVERNMENT_PUBLIC = "government_public"
    OTHER = "other"


class BusinessLineMarket(Enum):
    """Business line market classifications"""
    PROPERTY = "property"
    CASUALTY = "casualty"
    MARINE = "marine"
    AVIATION = "aviation"
    CYBER = "cyber"
    POLITICAL_RISK = "political_risk"
    CREDIT_SURETY = "credit_surety"
    SPECIALTY = "specialty"
    CATASTROPHE = "catastrophe"
    OTHER = "other"


@dataclass
class MarketIdentification:
    """Market identification result"""
    geographic_market: GeographicMarket
    industry_market: IndustryMarket
    business_line_market: BusinessLineMarket
    confidence_scores: Dict[str, float]
    identified_entities: Dict[str, List[str]]
    market_indicators: List[str]


@dataclass
class DocumentRelationship:
    """Relationship between documents"""
    document_id_1: str
    document_id_2: str
    relationship_type: str  # "same_client", "same_broker", "same_risk", "follow_up", "amendment"
    similarity_score: float
    common_entities: List[str]
    relationship_evidence: List[str]


@dataclass
class MarketGroup:
    """Market group with associated documents"""
    group_id: str
    market_type: MarketType
    market_value: str
    documents: List[str]  # Document IDs
    characteristics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class MarketAnalytics:
    """Market analytics and trends"""
    total_documents: int
    market_distribution: Dict[str, int]
    trend_analysis: Dict[str, Any]
    top_markets: List[Tuple[str, int]]
    growth_rates: Dict[str, float]
    seasonal_patterns: Dict[str, List[float]]


class MarketGroupingAgent:
    """
    Market grouping agent for email and document classification
    
    This agent analyzes documents and emails to identify market segments,
    classify them by geography and industry, and establish relationships
    between related documents.
    """
    
    def __init__(self):
        """Initialize the market grouping agent with ML models and configurations"""
        self.logger = logging.getLogger(__name__)
        
        # Cache directories (strings to avoid PosixPath issues)
        # Prefer environment variables; fall back to local project .cache to avoid '/app' when running locally
        cwd = os.getcwd()
        default_hf = (
            os.environ.get("HF_HOME")
            or os.environ.get("TRANSFORMERS_CACHE")
            or os.path.join(cwd, ".cache", "huggingface")
        )
        default_st = (
            os.environ.get("SENTENCE_TRANSFORMERS_CACHE")
            or os.path.join(cwd, ".cache", "sentence-transformers")
        )

        self.hf_cache_dir = str(default_hf)
        self.st_cache_dir = str(default_st)

        # Ensure directories exist
        try:
            os.makedirs(self.hf_cache_dir, exist_ok=True)
            os.makedirs(self.st_cache_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create cache directories: {e}")
        
        # Initialize zero-shot classification model
        try:
            self.classifier_model_name = "facebook/bart-large-mnli"
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model=self.classifier_model_name,
                return_all_scores=True,
                cache_dir=str(self.hf_cache_dir)
            )
            self.logger.info("Zero-shot classification model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load zero-shot classifier: {e}")
            self.zero_shot_classifier = None
        
        # Initialize sentence transformer for document similarity
        try:
            self.embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.sentence_transformer = SentenceTransformer(self.embeddings_model_name, cache_folder=str(self.st_cache_dir))
            self.logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_transformer = None
        
        # Initialize NER model for entity extraction
        try:
            self.ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                tokenizer=self.ner_model_name,
                aggregation_strategy="simple",
                cache_dir=str(self.hf_cache_dir)
            )
            self.logger.info("NER model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load NER model: {e}")
            self.ner_pipeline = None
        
        # Geographic market patterns
        self.geographic_patterns = {
            GeographicMarket.NORTH_AMERICA: [
                "usa", "united states", "america", "canada", "mexico", "north america",
                "new york", "california", "texas", "florida", "toronto", "vancouver"
            ],
            GeographicMarket.EUROPE: [
                "europe", "european", "uk", "united kingdom", "britain", "england",
                "germany", "france", "italy", "spain", "netherlands", "switzerland",
                "london", "paris", "berlin", "madrid", "amsterdam", "zurich"
            ],
            GeographicMarket.ASIA_PACIFIC: [
                "asia", "pacific", "china", "japan", "singapore", "hong kong",
                "australia", "new zealand", "south korea", "thailand", "malaysia",
                "tokyo", "beijing", "shanghai", "sydney", "melbourne"
            ],
            GeographicMarket.MIDDLE_EAST_AFRICA: [
                "middle east", "africa", "uae", "saudi arabia", "qatar", "dubai",
                "south africa", "egypt", "nigeria", "kenya", "morocco",
                "johannesburg", "cairo", "lagos", "casablanca"
            ],
            GeographicMarket.LATIN_AMERICA: [
                "latin america", "south america", "brazil", "argentina", "chile",
                "colombia", "peru", "mexico", "sao paulo", "buenos aires",
                "santiago", "bogota", "lima"
            ]
        }
        
        # Industry market patterns
        self.industry_patterns = {
            IndustryMarket.ENERGY_UTILITIES: [
                "oil", "gas", "energy", "power", "electricity", "utility", "utilities",
                "petroleum", "refinery", "pipeline", "solar", "wind", "nuclear",
                "coal", "lng", "renewable"
            ],
            IndustryMarket.MANUFACTURING: [
                "manufacturing", "factory", "plant", "production", "assembly",
                "automotive", "steel", "chemical", "pharmaceutical", "textile",
                "electronics", "machinery", "equipment"
            ],
            IndustryMarket.CONSTRUCTION_ENGINEERING: [
                "construction", "engineering", "building", "infrastructure",
                "contractor", "civil", "structural", "bridge", "tunnel", "road",
                "railway", "airport", "port", "dam"
            ],
            IndustryMarket.TRANSPORTATION: [
                "transportation", "logistics", "shipping", "freight", "cargo",
                "airline", "aviation", "railway", "trucking", "fleet",
                "warehouse", "distribution"
            ],
            IndustryMarket.TECHNOLOGY: [
                "technology", "software", "hardware", "it", "computer", "data",
                "telecommunications", "internet", "cloud", "ai", "artificial intelligence",
                "cybersecurity", "fintech"
            ],
            IndustryMarket.HEALTHCARE: [
                "healthcare", "hospital", "medical", "pharmaceutical", "biotech",
                "clinic", "health", "medicine", "drug", "vaccine", "therapy"
            ],
            IndustryMarket.FINANCIAL_SERVICES: [
                "bank", "banking", "financial", "insurance", "investment",
                "securities", "fund", "credit", "loan", "mortgage", "fintech"
            ],
            IndustryMarket.MARINE_AVIATION: [
                "marine", "maritime", "shipping", "vessel", "ship", "cargo",
                "aviation", "aircraft", "airline", "airport", "aerospace"
            ]
        }
        
        # Business line patterns
        self.business_line_patterns = {
            BusinessLineMarket.PROPERTY: [
                "property", "building", "real estate", "fire", "explosion",
                "natural catastrophe", "earthquake", "flood", "hurricane"
            ],
            BusinessLineMarket.CASUALTY: [
                "casualty", "liability", "third party", "public liability",
                "product liability", "professional indemnity", "workers compensation"
            ],
            BusinessLineMarket.MARINE: [
                "marine", "cargo", "hull", "vessel", "ship", "maritime",
                "ocean marine", "inland marine"
            ],
            BusinessLineMarket.AVIATION: [
                "aviation", "aircraft", "airline", "airport", "aerospace",
                "hull", "liability", "passenger"
            ],
            BusinessLineMarket.CYBER: [
                "cyber", "cybersecurity", "data breach", "privacy", "technology",
                "network security", "cyber liability"
            ],
            BusinessLineMarket.POLITICAL_RISK: [
                "political risk", "political violence", "war", "terrorism",
                "confiscation", "expropriation", "currency"
            ]
        }
        
        # Relationship indicators
        self.relationship_indicators = {
            "same_client": ["re:", "fwd:", "follow up", "amendment", "renewal"],
            "same_broker": ["broker", "intermediary", "agent"],
            "same_risk": ["same location", "same project", "related risk"],
            "follow_up": ["follow up", "update", "status", "response"],
            "amendment": ["amendment", "endorsement", "change", "modification"]
        }
    
    def identify_market(self, content: str, metadata: Dict[str, Any]) -> MarketIdentification:
        """
        Identify market from document content and metadata
        
        Args:
            content: Document text content
            metadata: Document metadata (email headers, etc.)
            
        Returns:
            MarketIdentification with market classifications
        """
        try:
            # Extract entities from content
            entities = self._extract_entities(content)
            
            # Classify geographic market
            geographic_market, geo_confidence = self._classify_geographic_market(content, entities, metadata)
            
            # Classify industry market
            industry_market, industry_confidence = self._classify_industry_market(content, entities)
            
            # Classify business line market
            business_line_market, business_confidence = self._classify_business_line_market(content, entities)
            
            # Identify market indicators
            market_indicators = self._identify_market_indicators(content, entities)
            
            # Compile confidence scores
            confidence_scores = {
                "geographic": geo_confidence,
                "industry": industry_confidence,
                "business_line": business_confidence,
                "overall": (geo_confidence + industry_confidence + business_confidence) / 3
            }
            
            return MarketIdentification(
                geographic_market=geographic_market,
                industry_market=industry_market,
                business_line_market=business_line_market,
                confidence_scores=confidence_scores,
                identified_entities=entities,
                market_indicators=market_indicators
            )
            
        except Exception as e:
            self.logger.error(f"Error in market identification: {e}")
            raise
    
    def classify_geographic_market(self, location: str, content: str = "") -> Tuple[GeographicMarket, float]:
        """
        Classify geographic market using zero-shot learning
        
        Args:
            location: Location string
            content: Additional content for context
            
        Returns:
            Tuple of (GeographicMarket, confidence_score)
        """
        try:
            # Combine location and content for analysis
            text_to_analyze = f"{location} {content}".lower()
            
            # Use pattern matching first (faster)
            pattern_result = self._classify_by_patterns(text_to_analyze, self.geographic_patterns)
            if pattern_result[1] > 0.7:  # High confidence from patterns
                return pattern_result
            
            # Use zero-shot classification if available
            if self.zero_shot_classifier:
                candidate_labels = [market.value for market in GeographicMarket if market != GeographicMarket.UNKNOWN]
                
                result = self.zero_shot_classifier(text_to_analyze, candidate_labels)
                
                if result['scores'][0] > 0.3:  # Minimum confidence threshold
                    best_label = result['labels'][0]
                    confidence = result['scores'][0]
                    
                    # Map back to enum
                    for market in GeographicMarket:
                        if market.value == best_label:
                            return market, confidence
            
            # Fallback to pattern result or unknown
            return pattern_result if pattern_result[1] > 0.0 else (GeographicMarket.UNKNOWN, 0.0)
            
        except Exception as e:
            self.logger.error(f"Error in geographic market classification: {e}")
            return GeographicMarket.UNKNOWN, 0.0
    
    def group_by_industry(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Group documents by industry sector using document clustering
        
        Args:
            documents: List of documents to group
            
        Returns:
            Dictionary mapping industry sectors to document lists
        """
        try:
            if not documents:
                return {}
            
            # Extract content and classify each document
            industry_groups = defaultdict(list)
            
            for doc in documents:
                # Read document content (simplified - in real implementation would use OCR results)
                content = self._get_document_content(doc)
                
                # Classify industry
                industry_market, confidence = self._classify_industry_market(content, {})
                
                # Group by industry if confidence is reasonable
                if confidence > 0.3:
                    industry_groups[industry_market.value].append(doc)
                else:
                    industry_groups[IndustryMarket.OTHER.value].append(doc)
            
            # Use clustering for documents with low classification confidence
            if len(industry_groups[IndustryMarket.OTHER.value]) > 2:
                clustered_groups = self._cluster_documents_by_similarity(
                    industry_groups[IndustryMarket.OTHER.value]
                )
                
                # Merge clustered groups
                for i, cluster in enumerate(clustered_groups):
                    cluster_key = f"cluster_{i}"
                    industry_groups[cluster_key] = cluster
                
                # Remove the "other" category if it was split into clusters
                if clustered_groups:
                    del industry_groups[IndustryMarket.OTHER.value]
            
            return dict(industry_groups)
            
        except Exception as e:
            self.logger.error(f"Error in industry grouping: {e}")
            return {}
    
    def map_relationships(self, documents: List[Document]) -> List[DocumentRelationship]:
        """
        Map relationships between related documents
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            List of DocumentRelationship objects
        """
        try:
            relationships = []
            
            if len(documents) < 2:
                return relationships
            
            # Extract content and entities for all documents
            doc_data = {}
            for doc in documents:
                content = self._get_document_content(doc)
                entities = self._extract_entities(content)
                doc_data[doc.id] = {
                    'content': content,
                    'entities': entities,
                    'metadata': doc.metadata or {}
                }
            
            # Compare each pair of documents
            for i, doc1 in enumerate(documents):
                for j, doc2 in enumerate(documents[i+1:], i+1):
                    relationship = self._analyze_document_relationship(
                        doc1, doc2, doc_data[doc1.id], doc_data[doc2.id]
                    )
                    
                    if relationship:
                        relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error in relationship mapping: {e}")
            return []
    
    def create_market_groups(self, documents: List[Document]) -> List[MarketGroup]:
        """
        Create market groups from documents
        
        Args:
            documents: List of documents to group
            
        Returns:
            List of MarketGroup objects
        """
        try:
            market_groups = []
            
            # Group by geographic market
            geo_groups = self._group_by_geographic_market(documents)
            for market, docs in geo_groups.items():
                if docs:
                    group = MarketGroup(
                        group_id=f"geo_{market}_{datetime.utcnow().timestamp()}",
                        market_type=MarketType.GEOGRAPHIC,
                        market_value=market,
                        documents=[doc.id for doc in docs],
                        characteristics=self._analyze_group_characteristics(docs, MarketType.GEOGRAPHIC),
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    market_groups.append(group)
            
            # Group by industry market
            industry_groups = self.group_by_industry(documents)
            for market, docs in industry_groups.items():
                if docs:
                    group = MarketGroup(
                        group_id=f"industry_{market}_{datetime.utcnow().timestamp()}",
                        market_type=MarketType.INDUSTRY,
                        market_value=market,
                        documents=[doc.id for doc in docs],
                        characteristics=self._analyze_group_characteristics(docs, MarketType.INDUSTRY),
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    market_groups.append(group)
            
            # Group by business line
            business_groups = self._group_by_business_line(documents)
            for market, docs in business_groups.items():
                if docs:
                    group = MarketGroup(
                        group_id=f"business_{market}_{datetime.utcnow().timestamp()}",
                        market_type=MarketType.BUSINESS_LINE,
                        market_value=market,
                        documents=[doc.id for doc in docs],
                        characteristics=self._analyze_group_characteristics(docs, MarketType.BUSINESS_LINE),
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    market_groups.append(group)
            
            return market_groups
            
        except Exception as e:
            self.logger.error(f"Error in creating market groups: {e}")
            return []
    
    def generate_market_analytics(self, documents: List[Document]) -> MarketAnalytics:
        """
        Generate market analytics and trends
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            MarketAnalytics with comprehensive market analysis
        """
        try:
            # Basic statistics
            total_documents = len(documents)
            
            # Market distribution analysis
            market_distribution = self._calculate_market_distribution(documents)
            
            # Trend analysis (simplified - would need historical data)
            trend_analysis = self._analyze_market_trends(documents)
            
            # Top markets
            top_markets = sorted(market_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Growth rates (simplified calculation)
            growth_rates = self._calculate_growth_rates(documents)
            
            # Seasonal patterns (simplified)
            seasonal_patterns = self._analyze_seasonal_patterns(documents)
            
            return MarketAnalytics(
                total_documents=total_documents,
                market_distribution=market_distribution,
                trend_analysis=trend_analysis,
                top_markets=top_markets,
                growth_rates=growth_rates,
                seasonal_patterns=seasonal_patterns
            )
            
        except Exception as e:
            self.logger.error(f"Error in market analytics generation: {e}")
            raise
    
    def filter_by_market(
        self, 
        documents: List[Document], 
        market_type: MarketType, 
        market_value: str
    ) -> List[Document]:
        """
        Filter documents by market criteria
        
        Args:
            documents: List of documents to filter
            market_type: Type of market filter
            market_value: Market value to filter by
            
        Returns:
            Filtered list of documents
        """
        try:
            filtered_docs = []
            
            for doc in documents:
                content = self._get_document_content(doc)
                market_id = self.identify_market(content, doc.metadata or {})
                
                match = False
                if market_type == MarketType.GEOGRAPHIC:
                    match = market_id.geographic_market.value == market_value
                elif market_type == MarketType.INDUSTRY:
                    match = market_id.industry_market.value == market_value
                elif market_type == MarketType.BUSINESS_LINE:
                    match = market_id.business_line_market.value == market_value
                
                if match:
                    filtered_docs.append(doc)
            
            return filtered_docs
            
        except Exception as e:
            self.logger.error(f"Error in market filtering: {e}")
            return []
    
    # Helper methods
    
    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities from content"""
        entities = {
            'PERSON': [],
            'ORG': [],
            'LOC': [],
            'MISC': []
        }
        
        if not self.ner_pipeline or not content:
            return entities
        
        try:
            ner_results = self.ner_pipeline(content)
            
            for entity in ner_results:
                entity_type = entity['entity_group']
                entity_text = entity['word']
                
                if entity_type in entities:
                    entities[entity_type].append(entity_text)
                else:
                    entities['MISC'].append(entity_text)
            
            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
            
        except Exception as e:
            self.logger.warning(f"Error in entity extraction: {e}")
        
        return entities
    
    def _classify_by_patterns(
        self, 
        text: str, 
        patterns: Dict[Any, List[str]]
    ) -> Tuple[Any, float]:
        """Classify text using pattern matching"""
        text_lower = text.lower()
        scores = {}
        
        for market, keywords in patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight longer keywords more heavily
                    score += len(keyword.split()) * 0.1
            
            if score > 0:
                scores[market] = min(1.0, score)
        
        if scores:
            best_market = max(scores, key=scores.get)
            return best_market, scores[best_market]
        
        return list(patterns.keys())[0], 0.0  # Return first option with 0 confidence
    
    def _classify_geographic_market(
        self, 
        content: str, 
        entities: Dict[str, List[str]], 
        metadata: Dict[str, Any]
    ) -> Tuple[GeographicMarket, float]:
        """Classify geographic market from content and entities"""
        # Check email metadata for geographic clues
        geographic_clues = []
        
        # Extract from email headers if available
        if 'from' in metadata:
            from_email = metadata['from']
            domain = self._extract_domain(from_email)
            if domain:
                geographic_clues.append(domain)
        
        # Add location entities
        geographic_clues.extend(entities.get('LOC', []))
        
        # Combine all geographic information
        geo_text = f"{content} {' '.join(geographic_clues)}".lower()
        
        return self._classify_by_patterns(geo_text, self.geographic_patterns)
    
    def _classify_industry_market(
        self, 
        content: str, 
        entities: Dict[str, List[str]]
    ) -> Tuple[IndustryMarket, float]:
        """Classify industry market from content"""
        # Combine content with organization entities for better classification
        org_entities = ' '.join(entities.get('ORG', []))
        industry_text = f"{content} {org_entities}".lower()
        
        return self._classify_by_patterns(industry_text, self.industry_patterns)
    
    def _classify_business_line_market(
        self, 
        content: str, 
        entities: Dict[str, List[str]]
    ) -> Tuple[BusinessLineMarket, float]:
        """Classify business line market from content"""
        business_text = content.lower()
        
        return self._classify_by_patterns(business_text, self.business_line_patterns)
    
    def _identify_market_indicators(
        self, 
        content: str, 
        entities: Dict[str, List[str]]
    ) -> List[str]:
        """Identify market indicators from content"""
        indicators = []
        content_lower = content.lower()
        
        # Look for specific market indicators
        market_keywords = [
            "facultative", "reinsurance", "treaty", "quota share", "surplus",
            "catastrophe", "excess of loss", "stop loss", "aggregate",
            "cedant", "ceding company", "retrocession"
        ]
        
        for keyword in market_keywords:
            if keyword in content_lower:
                indicators.append(keyword)
        
        # Add significant entities as indicators
        for entity_type, entity_list in entities.items():
            if entity_type in ['ORG', 'LOC'] and entity_list:
                indicators.extend(entity_list[:3])  # Top 3 entities per type
        
        return indicators
    
    def _get_document_content(self, document: Document) -> str:
        """Get document content (simplified - would integrate with OCR results)"""
        # In a real implementation, this would retrieve processed OCR content
        # For now, return metadata or filename as content
        content_parts = [document.filename]
        
        if document.metadata:
            if 'subject' in document.metadata:
                content_parts.append(document.metadata['subject'])
            if 'body' in document.metadata:
                content_parts.append(document.metadata['body'])
        
        return ' '.join(content_parts)
    
    def _cluster_documents_by_similarity(self, documents: List[Document]) -> List[List[Document]]:
        """Cluster documents by content similarity"""
        if not self.sentence_transformer or len(documents) < 2:
            return [documents]
        
        try:
            # Get document contents
            contents = [self._get_document_content(doc) for doc in documents]
            
            # Generate embeddings
            embeddings = self.sentence_transformer.encode(contents)
            
            # Perform clustering
            if len(documents) <= 5:
                # Use simple similarity threshold for small groups
                similarity_matrix = cosine_similarity(embeddings)
                clusters = self._threshold_clustering(documents, similarity_matrix, threshold=0.7)
            else:
                # Use DBSCAN for larger groups
                clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
                cluster_labels = clustering.fit_predict(embeddings)
                
                clusters = defaultdict(list)
                for i, label in enumerate(cluster_labels):
                    clusters[label].append(documents[i])
                
                clusters = list(clusters.values())
            
            return clusters if clusters else [documents]
            
        except Exception as e:
            self.logger.warning(f"Error in document clustering: {e}")
            return [documents]
    
    def _threshold_clustering(
        self, 
        documents: List[Document], 
        similarity_matrix: np.ndarray, 
        threshold: float
    ) -> List[List[Document]]:
        """Perform threshold-based clustering"""
        clusters = []
        used_indices = set()
        
        for i in range(len(documents)):
            if i in used_indices:
                continue
            
            cluster = [documents[i]]
            used_indices.add(i)
            
            for j in range(i + 1, len(documents)):
                if j not in used_indices and similarity_matrix[i][j] >= threshold:
                    cluster.append(documents[j])
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _analyze_document_relationship(
        self, 
        doc1: Document, 
        doc2: Document, 
        doc1_data: Dict[str, Any], 
        doc2_data: Dict[str, Any]
    ) -> Optional[DocumentRelationship]:
        """Analyze relationship between two documents"""
        try:
            # Calculate content similarity
            if self.sentence_transformer:
                embeddings = self.sentence_transformer.encode([
                    doc1_data['content'], 
                    doc2_data['content']
                ])
                similarity_score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            else:
                similarity_score = 0.0
            
            # Find common entities
            common_entities = []
            for entity_type in ['PERSON', 'ORG', 'LOC']:
                entities1 = set(doc1_data['entities'].get(entity_type, []))
                entities2 = set(doc2_data['entities'].get(entity_type, []))
                common_entities.extend(list(entities1.intersection(entities2)))
            
            # Determine relationship type
            relationship_type, evidence = self._determine_relationship_type(
                doc1, doc2, doc1_data, doc2_data, common_entities
            )
            
            # Only create relationship if there's sufficient evidence
            if (similarity_score > 0.5 or len(common_entities) > 0 or 
                relationship_type != "unrelated"):
                
                return DocumentRelationship(
                    document_id_1=doc1.id,
                    document_id_2=doc2.id,
                    relationship_type=relationship_type,
                    similarity_score=similarity_score,
                    common_entities=common_entities,
                    relationship_evidence=evidence
                )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error analyzing document relationship: {e}")
            return None
    
    def _determine_relationship_type(
        self, 
        doc1: Document, 
        doc2: Document, 
        doc1_data: Dict[str, Any], 
        doc2_data: Dict[str, Any], 
        common_entities: List[str]
    ) -> Tuple[str, List[str]]:
        """Determine the type of relationship between documents"""
        evidence = []
        
        # Check for email thread indicators
        subject1 = doc1_data['metadata'].get('subject', '').lower()
        subject2 = doc2_data['metadata'].get('subject', '').lower()
        
        if any(indicator in subject1 or indicator in subject2 
               for indicator in self.relationship_indicators['follow_up']):
            evidence.append("Email thread indicators")
            return "follow_up", evidence
        
        # Check for same client indicators
        if len(common_entities) > 2:
            evidence.append(f"Common entities: {', '.join(common_entities[:3])}")
            return "same_client", evidence
        
        # Check for broker relationships
        if any(indicator in doc1_data['content'].lower() or indicator in doc2_data['content'].lower()
               for indicator in self.relationship_indicators['same_broker']):
            evidence.append("Broker relationship indicators")
            return "same_broker", evidence
        
        # Check for amendments
        if any(indicator in subject1 or indicator in subject2
               for indicator in self.relationship_indicators['amendment']):
            evidence.append("Amendment indicators")
            return "amendment", evidence
        
        # Default to unrelated if no strong indicators
        return "unrelated", evidence
    
    def _group_by_geographic_market(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Group documents by geographic market"""
        groups = defaultdict(list)
        
        for doc in documents:
            content = self._get_document_content(doc)
            geo_market, confidence = self.classify_geographic_market("", content)
            
            if confidence > 0.3:
                groups[geo_market.value].append(doc)
            else:
                groups[GeographicMarket.UNKNOWN.value].append(doc)
        
        return dict(groups)
    
    def _group_by_business_line(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Group documents by business line"""
        groups = defaultdict(list)
        
        for doc in documents:
            content = self._get_document_content(doc)
            business_market, confidence = self._classify_business_line_market(content, {})
            
            if confidence > 0.3:
                groups[business_market.value].append(doc)
            else:
                groups[BusinessLineMarket.OTHER.value].append(doc)
        
        return dict(groups)
    
    def _analyze_group_characteristics(
        self, 
        documents: List[Document], 
        market_type: MarketType
    ) -> Dict[str, Any]:
        """Analyze characteristics of a document group"""
        characteristics = {
            'document_count': len(documents),
            'document_types': Counter([doc.document_type.value for doc in documents]),
            'date_range': self._calculate_date_range(documents),
            'common_entities': self._find_common_entities(documents)
        }
        
        return characteristics
    
    def _calculate_date_range(self, documents: List[Document]) -> Dict[str, Any]:
        """Calculate date range for documents"""
        dates = [doc.upload_timestamp for doc in documents]
        
        if dates:
            return {
                'earliest': min(dates).isoformat(),
                'latest': max(dates).isoformat(),
                'span_days': (max(dates) - min(dates)).days
            }
        
        return {}
    
    def _find_common_entities(self, documents: List[Document]) -> Dict[str, List[str]]:
        """Find entities common across documents"""
        all_entities = defaultdict(list)
        
        for doc in documents:
            content = self._get_document_content(doc)
            entities = self._extract_entities(content)
            
            for entity_type, entity_list in entities.items():
                all_entities[entity_type].extend(entity_list)
        
        # Find entities that appear in multiple documents
        common_entities = {}
        for entity_type, entity_list in all_entities.items():
            entity_counts = Counter(entity_list)
            common = [entity for entity, count in entity_counts.items() if count > 1]
            if common:
                common_entities[entity_type] = common[:5]  # Top 5 common entities
        
        return common_entities
    
    def _calculate_market_distribution(self, documents: List[Document]) -> Dict[str, int]:
        """Calculate market distribution across documents"""
        distribution = defaultdict(int)
        
        for doc in documents:
            content = self._get_document_content(doc)
            market_id = self.identify_market(content, doc.metadata or {})
            
            # Count by geographic market
            distribution[f"geo_{market_id.geographic_market.value}"] += 1
            
            # Count by industry market
            distribution[f"industry_{market_id.industry_market.value}"] += 1
            
            # Count by business line
            distribution[f"business_{market_id.business_line_market.value}"] += 1
        
        return dict(distribution)
    
    def _analyze_market_trends(self, documents: List[Document]) -> Dict[str, Any]:
        """Analyze market trends (simplified implementation)"""
        # Group documents by month
        monthly_counts = defaultdict(int)
        
        for doc in documents:
            month_key = doc.upload_timestamp.strftime("%Y-%m")
            monthly_counts[month_key] += 1
        
        # Calculate trend (simplified linear trend)
        months = sorted(monthly_counts.keys())
        counts = [monthly_counts[month] for month in months]
        
        if len(counts) > 1:
            trend_slope = (counts[-1] - counts[0]) / len(counts)
            trend_direction = "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
        else:
            trend_direction = "insufficient_data"
            trend_slope = 0
        
        return {
            'monthly_distribution': dict(monthly_counts),
            'trend_direction': trend_direction,
            'trend_slope': trend_slope,
            'total_months': len(months)
        }
    
    def _calculate_growth_rates(self, documents: List[Document]) -> Dict[str, float]:
        """Calculate growth rates by market (simplified)"""
        # This is a simplified implementation
        # In practice, would need historical data for accurate growth calculation
        
        growth_rates = {}
        
        # Group by market and calculate basic growth metrics
        market_groups = self._group_by_geographic_market(documents)
        
        for market, docs in market_groups.items():
            if len(docs) > 1:
                # Simple growth rate based on document frequency over time
                dates = sorted([doc.upload_timestamp for doc in docs])
                if len(dates) > 1:
                    time_span = (dates[-1] - dates[0]).days
                    if time_span > 0:
                        growth_rate = len(docs) / (time_span / 30)  # Documents per month
                        growth_rates[market] = round(growth_rate, 2)
        
        return growth_rates
    
    def _analyze_seasonal_patterns(self, documents: List[Document]) -> Dict[str, List[float]]:
        """Analyze seasonal patterns in document flow"""
        monthly_patterns = defaultdict(list)
        
        for doc in documents:
            month = doc.upload_timestamp.month
            monthly_patterns[doc.upload_timestamp.year].append(month)
        
        # Calculate average monthly distribution
        seasonal_data = {}
        if monthly_patterns:
            all_months = []
            for year_months in monthly_patterns.values():
                all_months.extend(year_months)
            
            month_counts = Counter(all_months)
            seasonal_data['monthly_distribution'] = [
                month_counts.get(i, 0) for i in range(1, 13)
            ]
        
        return seasonal_data
    
    def _extract_domain(self, email: str) -> Optional[str]:
        """Extract domain from email address"""
        try:
            if '@' in email:
                domain = email.split('@')[1].lower()
                return domain
        except:
            pass
        return None