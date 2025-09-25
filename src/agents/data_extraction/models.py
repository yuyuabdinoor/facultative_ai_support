from pydantic import BaseModel
from typing import List, Dict, Optional, Union

class ExtractedEntity(BaseModel):
    text: str
    label: str
    confidence: float
    start: int
    end: int

class FinancialMetrics(BaseModel):
    revenue: Optional[float]
    profit_margin: Optional[float]
    assets: Optional[float]
    liabilities: Optional[float]
    confidence_score: float

class DocumentLayout(BaseModel):
    text_blocks: List[Dict[str, Union[str, List[int]]]]
    layout_score: float
