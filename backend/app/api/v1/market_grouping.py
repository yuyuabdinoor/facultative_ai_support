"""
Market Grouping API endpoints

This module provides REST API endpoints for market classification and grouping functionality:
- Market identification from document content
- Document grouping by market segments
- Relationship mapping between documents
- Market analytics and reporting
- Market-based filtering capabilities

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.2
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ...agents.market_grouping_agent import (
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
from ...models.schemas import Document, DocumentType
from ...services.document_service import DocumentService


# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize market grouping agent
market_agent = MarketGroupingAgent()


# Pydantic models for API requests/responses
from pydantic import BaseModel, Field


class MarketIdentificationRequest(BaseModel):
    """Request model for market identification"""
    content: str = Field(..., description="Document content to analyze")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")


class MarketIdentificationResponse(BaseModel):
    """Response model for market identification"""
    geographic_market: str
    industry_market: str
    business_line_market: str
    confidence_scores: Dict[str, float]
    identified_entities: Dict[str, List[str]]
    market_indicators: List[str]


class DocumentRelationshipResponse(BaseModel):
    """Response model for document relationships"""
    document_id_1: str
    document_id_2: str
    relationship_type: str
    similarity_score: float
    common_entities: List[str]
    relationship_evidence: List[str]


class MarketGroupResponse(BaseModel):
    """Response model for market groups"""
    group_id: str
    market_type: str
    market_value: str
    documents: List[str]
    characteristics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class MarketAnalyticsResponse(BaseModel):
    """Response model for market analytics"""
    total_documents: int
    market_distribution: Dict[str, int]
    trend_analysis: Dict[str, Any]
    top_markets: List[List[Any]]  # List of [market_name, count] pairs
    growth_rates: Dict[str, float]
    seasonal_patterns: Dict[str, List[float]]


class MarketFilterRequest(BaseModel):
    """Request model for market filtering"""
    market_type: str = Field(..., description="Type of market filter (geographic, industry, business_line)")
    market_value: str = Field(..., description="Market value to filter by")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to filter (optional)")


# Dependency to get document service
def get_document_service() -> DocumentService:
    """Get document service instance"""
    return DocumentService()


@router.post("/identify", response_model=MarketIdentificationResponse)
async def identify_market(
    request: MarketIdentificationRequest
) -> MarketIdentificationResponse:
    """
    Identify market classification from document content and metadata
    
    Args:
        request: Market identification request with content and metadata
        
    Returns:
        Market identification results with classifications and confidence scores
    """
    try:
        logger.info("Processing market identification request")
        
        # Perform market identification
        result = market_agent.identify_market(request.content, request.metadata)
        
        # Convert to response format
        response = MarketIdentificationResponse(
            geographic_market=result.geographic_market.value,
            industry_market=result.industry_market.value,
            business_line_market=result.business_line_market.value,
            confidence_scores=result.confidence_scores,
            identified_entities=result.identified_entities,
            market_indicators=result.market_indicators
        )
        
        logger.info(f"Market identification completed with overall confidence: {result.confidence_scores.get('overall', 0)}")
        return response
        
    except Exception as e:
        logger.error(f"Error in market identification: {e}")
        raise HTTPException(status_code=500, detail=f"Market identification failed: {str(e)}")


@router.get("/classify/geographic")
async def classify_geographic_market(
    location: str = Query(..., description="Location string to classify"),
    content: str = Query("", description="Additional content for context")
) -> Dict[str, Any]:
    """
    Classify geographic market using zero-shot learning
    
    Args:
        location: Location string to classify
        content: Additional content for context
        
    Returns:
        Geographic market classification with confidence score
    """
    try:
        logger.info(f"Classifying geographic market for location: {location}")
        
        market, confidence = market_agent.classify_geographic_market(location, content)
        
        return {
            "geographic_market": market.value,
            "confidence": confidence,
            "location": location
        }
        
    except Exception as e:
        logger.error(f"Error in geographic market classification: {e}")
        raise HTTPException(status_code=500, detail=f"Geographic classification failed: {str(e)}")


@router.post("/documents/group/industry")
async def group_documents_by_industry(
    document_ids: List[str] = Body(..., description="List of document IDs to group"),
    document_service: DocumentService = Depends(get_document_service)
) -> Dict[str, List[str]]:
    """
    Group documents by industry sector using document clustering
    
    Args:
        document_ids: List of document IDs to group
        document_service: Document service dependency
        
    Returns:
        Dictionary mapping industry sectors to document ID lists
    """
    try:
        logger.info(f"Grouping {len(document_ids)} documents by industry")
        
        # Retrieve documents
        documents = []
        for doc_id in document_ids:
            doc = await document_service.get_document(doc_id)
            if doc:
                documents.append(doc)
        
        if not documents:
            raise HTTPException(status_code=404, detail="No valid documents found")
        
        # Group by industry
        industry_groups = market_agent.group_by_industry(documents)
        
        # Convert to document ID mapping
        result = {}
        for industry, docs in industry_groups.items():
            result[industry] = [doc.id for doc in docs]
        
        logger.info(f"Created {len(result)} industry groups")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in industry grouping: {e}")
        raise HTTPException(status_code=500, detail=f"Industry grouping failed: {str(e)}")


@router.post("/documents/relationships", response_model=List[DocumentRelationshipResponse])
async def map_document_relationships(
    document_ids: List[str] = Body(..., description="List of document IDs to analyze"),
    document_service: DocumentService = Depends(get_document_service)
) -> List[DocumentRelationshipResponse]:
    """
    Map relationships between related documents
    
    Args:
        document_ids: List of document IDs to analyze
        document_service: Document service dependency
        
    Returns:
        List of document relationships with similarity scores and evidence
    """
    try:
        logger.info(f"Mapping relationships for {len(document_ids)} documents")
        
        # Retrieve documents
        documents = []
        for doc_id in document_ids:
            doc = await document_service.get_document(doc_id)
            if doc:
                documents.append(doc)
        
        if len(documents) < 2:
            return []
        
        # Map relationships
        relationships = market_agent.map_relationships(documents)
        
        # Convert to response format
        response = [
            DocumentRelationshipResponse(
                document_id_1=rel.document_id_1,
                document_id_2=rel.document_id_2,
                relationship_type=rel.relationship_type,
                similarity_score=rel.similarity_score,
                common_entities=rel.common_entities,
                relationship_evidence=rel.relationship_evidence
            )
            for rel in relationships
        ]
        
        logger.info(f"Found {len(response)} document relationships")
        return response
        
    except Exception as e:
        logger.error(f"Error in relationship mapping: {e}")
        raise HTTPException(status_code=500, detail=f"Relationship mapping failed: {str(e)}")


@router.post("/documents/groups", response_model=List[MarketGroupResponse])
async def create_market_groups(
    document_ids: List[str] = Body(..., description="List of document IDs to group"),
    document_service: DocumentService = Depends(get_document_service)
) -> List[MarketGroupResponse]:
    """
    Create market groups from documents
    
    Args:
        document_ids: List of document IDs to group
        document_service: Document service dependency
        
    Returns:
        List of market groups with characteristics and metadata
    """
    try:
        logger.info(f"Creating market groups for {len(document_ids)} documents")
        
        # Retrieve documents
        documents = []
        for doc_id in document_ids:
            doc = await document_service.get_document(doc_id)
            if doc:
                documents.append(doc)
        
        if not documents:
            raise HTTPException(status_code=404, detail="No valid documents found")
        
        # Create market groups
        market_groups = market_agent.create_market_groups(documents)
        
        # Convert to response format
        response = [
            MarketGroupResponse(
                group_id=group.group_id,
                market_type=group.market_type.value,
                market_value=group.market_value,
                documents=group.documents,
                characteristics=group.characteristics,
                created_at=group.created_at,
                updated_at=group.updated_at
            )
            for group in market_groups
        ]
        
        logger.info(f"Created {len(response)} market groups")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in market group creation: {e}")
        raise HTTPException(status_code=500, detail=f"Market group creation failed: {str(e)}")


@router.post("/analytics", response_model=MarketAnalyticsResponse)
async def generate_market_analytics(
    document_ids: Optional[List[str]] = Body(None, description="List of document IDs to analyze (optional - analyzes all if not provided)"),
    document_service: DocumentService = Depends(get_document_service)
) -> MarketAnalyticsResponse:
    """
    Generate market analytics and trends
    
    Args:
        document_ids: Optional list of document IDs to analyze
        document_service: Document service dependency
        
    Returns:
        Market analytics with distribution, trends, and patterns
    """
    try:
        logger.info("Generating market analytics")
        
        # Retrieve documents
        if document_ids:
            documents = []
            for doc_id in document_ids:
                doc = await document_service.get_document(doc_id)
                if doc:
                    documents.append(doc)
        else:
            # Get all documents if no specific IDs provided
            documents = await document_service.get_all_documents()
        
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found for analysis")
        
        # Generate analytics
        analytics = market_agent.generate_market_analytics(documents)
        
        # Convert to response format
        response = MarketAnalyticsResponse(
            total_documents=analytics.total_documents,
            market_distribution=analytics.market_distribution,
            trend_analysis=analytics.trend_analysis,
            top_markets=analytics.top_markets,
            growth_rates=analytics.growth_rates,
            seasonal_patterns=analytics.seasonal_patterns
        )
        
        logger.info(f"Generated analytics for {analytics.total_documents} documents")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in market analytics generation: {e}")
        raise HTTPException(status_code=500, detail=f"Market analytics generation failed: {str(e)}")


@router.post("/filter", response_model=List[str])
async def filter_documents_by_market(
    request: MarketFilterRequest,
    document_service: DocumentService = Depends(get_document_service)
) -> List[str]:
    """
    Filter documents by market criteria
    
    Args:
        request: Market filter request with type and value
        document_service: Document service dependency
        
    Returns:
        List of document IDs matching the market criteria
    """
    try:
        logger.info(f"Filtering documents by {request.market_type}: {request.market_value}")
        
        # Validate market type
        try:
            market_type = MarketType(request.market_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid market type: {request.market_type}")
        
        # Retrieve documents
        if request.document_ids:
            documents = []
            for doc_id in request.document_ids:
                doc = await document_service.get_document(doc_id)
                if doc:
                    documents.append(doc)
        else:
            # Get all documents if no specific IDs provided
            documents = await document_service.get_all_documents()
        
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found for filtering")
        
        # Filter documents
        filtered_docs = market_agent.filter_by_market(documents, market_type, request.market_value)
        
        # Return document IDs
        result = [doc.id for doc in filtered_docs]
        
        logger.info(f"Filtered to {len(result)} documents")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in market filtering: {e}")
        raise HTTPException(status_code=500, detail=f"Market filtering failed: {str(e)}")


@router.get("/markets/geographic", response_model=List[str])
async def get_geographic_markets() -> List[str]:
    """
    Get list of available geographic markets
    
    Returns:
        List of geographic market values
    """
    return [market.value for market in GeographicMarket]


@router.get("/markets/industry", response_model=List[str])
async def get_industry_markets() -> List[str]:
    """
    Get list of available industry markets
    
    Returns:
        List of industry market values
    """
    return [market.value for market in IndustryMarket]


@router.get("/markets/business-line", response_model=List[str])
async def get_business_line_markets() -> List[str]:
    """
    Get list of available business line markets
    
    Returns:
        List of business line market values
    """
    return [market.value for market in BusinessLineMarket]


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for market grouping service
    
    Returns:
        Service health status
    """
    try:
        # Test basic functionality
        test_result = market_agent.identify_market("test content", {})
        
        return {
            "status": "healthy",
            "service": "market_grouping",
            "timestamp": datetime.utcnow().isoformat(),
            "models_loaded": {
                "zero_shot_classifier": market_agent.zero_shot_classifier is not None,
                "sentence_transformer": market_agent.sentence_transformer is not None,
                "ner_pipeline": market_agent.ner_pipeline is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "market_grouping",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }