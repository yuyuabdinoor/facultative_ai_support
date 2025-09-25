import pytest
from src.agents.data_extraction.extractor import DataExtractionAgent
from src.agents.data_extraction.models import ExtractedEntity, FinancialMetrics, DocumentLayout

@pytest.fixture
async def extraction_agent():
    agent = DataExtractionAgent()
    await agent.initialize_models()
    return agent

@pytest.mark.asyncio
async def test_entity_extraction(extraction_agent):
    text = "Acme Insurance Company reported $10M in losses."
    entities = await extraction_agent.extract_entities(text)
    assert len(entities) > 0
    assert isinstance(entities[0], ExtractedEntity)
    assert entities[0].text == "Acme Insurance"
    assert entities[0].confidence > 0.5

@pytest.mark.asyncio
async def test_financial_extraction(extraction_agent):
    text = "Annual revenue: $10M, Profit margin: 15%"
    metrics = await extraction_agent.extract_financial_data(text)
    assert isinstance(metrics, FinancialMetrics)
    assert metrics.confidence_score > 0
    assert metrics.revenue is not None
    assert metrics.profit_margin is not None

@pytest.mark.asyncio
async def test_layout_analysis(extraction_agent):
    # Mock PDF document bytes
    doc_bytes = b"mock PDF content"
    layout = await extraction_agent.analyze_layout(doc_bytes)
    assert isinstance(layout, DocumentLayout)
    assert len(layout.text_blocks) > 0
    assert layout.layout_score > 0.5
    assert "text" in layout.text_blocks[0]
    assert "bbox" in layout.text_blocks[0]

@pytest.mark.asyncio
async def test_model_initialization(extraction_agent):
    assert extraction_agent.models is not None
    # This should pass even with mock models
    assert len(extraction_agent.models) > 0
