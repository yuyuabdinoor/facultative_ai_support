from typing import Dict, Any, List
import docx
import pandas as pd
import PyPDF2
from pathlib import Path
from transformers import pipeline
from .calculation_service import CalculationService
from decimal import Decimal

class DocumentAnalysisService:
    def __init__(self):
        self.calc_service = CalculationService()
        self.text_classifier = pipeline("zero-shot-classification")
        self.ner_pipeline = pipeline("ner")
        self.risk_analyzer = pipeline("text-classification")
        
    async def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """Analyze document and extract key information"""
        text_content = await self._extract_text(file_path)
        extracted_data = await self._extract_key_data(text_content)
        risk_analysis = await self._analyze_risk_factors(extracted_data)
        calculations = await self._perform_calculations(extracted_data)
        
        return {
            "extracted_data": extracted_data,
            "risk_analysis": risk_analysis,
            "calculations": calculations,
            "confidence_score": await self._calculate_confidence(extracted_data)
        }

    async def _extract_text(self, file_path: str) -> str:
        """Extract text from multiple document formats"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return " ".join([page.extract_text() for page in reader.pages])
                
        elif file_ext in ['.doc', '.docx']:
            doc = docx.Document(file_path)
            return " ".join([paragraph.text for paragraph in doc.paragraphs])
            
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
            return df.to_json(orient='records')
            
        else:
            with open(file_path, 'r') as file:
                return file.read()

    async def _extract_key_data(self, text: str) -> Dict[str, Any]:
        """Extract key information using NER and classification"""
        # Extract entities
        entities = self.ner_pipeline(text)
        
        # Classify document sections
        sections = await self._classify_sections(text)
        
        return {
            "insured": self._find_entity(entities, "ORG"),
            "location": self._find_entity(entities, "LOC"),
            "perils": await self._identify_perils(text),
            "financial_data": await self._extract_financial_data(text),
            "sections": sections
        }

    async def _analyze_risk_factors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk factors using LLM"""
        location = data.get("location", "")
        
        # Generate risk analysis prompt
        prompt = f"""
        Analyze insurance risk factors for:
        Location: {location}
        Industry: {data.get('insured', {}).get('industry', '')}
        Perils: {', '.join(data.get('perils', []))}
        
        Consider:
        1. Political stability
        2. Natural disaster exposure
        3. Economic conditions
        4. Infrastructure quality
        5. Climate change impact
        """
        
        # Get risk analysis from LLM
        analysis = await self._get_llm_analysis(prompt)
        
        return {
            "location_risk": analysis.get("location_risk"),
            "political_risk": analysis.get("political_risk"),
            "natural_disaster_risk": analysis.get("natural_disaster_risk"),
            "climate_risk": analysis.get("climate_risk"),
            "confidence": analysis.get("confidence", 0.85)
        }

    async def _perform_calculations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform insurance calculations"""
        financial_data = data.get("financial_data", {})
        
        # Calculate premium rate
        premium_rate = self.calc_service.calculate_premium_rate(
            Decimal(str(financial_data.get("premium", 0))),
            Decimal(str(financial_data.get("tsi", 1))),
            use_permille=False
        )
        
        # Calculate loss ratio
        loss_ratio = self.calc_service.calculate_loss_ratio(
            Decimal(str(financial_data.get("paid_losses", 0))),
            Decimal(str(financial_data.get("outstanding", 0))),
            Decimal(str(financial_data.get("recoveries", 0))),
            Decimal(str(financial_data.get("earned_premium", 1)))
        )
        
        return {
            "premium_rate": premium_rate,
            "loss_ratio": loss_ratio,
            "currency_conversions": await self._convert_currencies(financial_data)
        }

    async def _convert_currencies(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert all monetary values to KES"""
        conversions = {}
        original_currency = financial_data.get("currency", "USD")
        
        for key, value in financial_data.items():
            if isinstance(value, (int, float)):
                conversion = await self.calc_service.convert_currency(
                    Decimal(str(value)),
                    original_currency,
                    "KES"
                )
                conversions[key] = conversion
                
        return conversions
