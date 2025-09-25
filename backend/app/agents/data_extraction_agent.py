"""
Data Extraction Agent with Hugging Face Models

This agent extracts and structures risk-related information from processed documents
using various Hugging Face models for NER, financial analysis, and classification.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, AutoModel,
    pipeline, AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

from ..models.schemas import (
    RiskParameters, FinancialData, ExtractionResult, ValidationResult,
    ProcessingResult
)

logger = logging.getLogger(__name__)


class DataExtractionAgent:
    """
    Data extraction agent that uses Hugging Face models to extract and structure
    risk-related information from processed documents.
    """
    
    def __init__(self):
        """Initialize the data extraction agent with Hugging Face models."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing DataExtractionAgent on device: {self.device}")
        
        # Initialize models
        self._initialize_models()
        
        # Pattern matching rules for structured data
        self._initialize_patterns()
        
        # Validation rules
        self._initialize_validation_rules()
    
    def _initialize_models(self):
        """Initialize all Hugging Face models."""
        try:
            # NER Model for general entity extraction
            logger.info("Loading NER model: dbmdz/bert-large-cased-finetuned-conll03-english")
            self.ner_tokenizer = AutoTokenizer.from_pretrained(
                "dbmdz/bert-large-cased-finetuned-conll03-english"
            )
            self.ner_model = AutoModelForTokenClassification.from_pretrained(
                "dbmdz/bert-large-cased-finetuned-conll03-english"
            ).to(self.device)
            
            # Financial NER using FinBERT
            logger.info("Loading Financial NER model: ProsusAI/finbert")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.finbert_model = AutoModel.from_pretrained("ProsusAI/finbert").to(self.device)
            
            # Financial sentiment pipeline
            self.financial_classifier = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Document Layout Analysis
            logger.info("Loading Layout model: microsoft/layoutlmv3-base")
            self.layout_tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")
            self.layout_model = AutoModel.from_pretrained("microsoft/layoutlmv3-base").to(self.device)
            
            # Zero-shot classification
            logger.info("Loading Zero-shot classifier: facebook/bart-large-mnli")
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Sentence embeddings for similarity analysis
            logger.info("Loading Sentence Transformer: sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def _initialize_patterns(self):
        """Initialize regex patterns for structured data extraction."""
        self.patterns = {
            # Financial patterns
            'currency_amounts': re.compile(r'[\$£€¥]\s*[\d,]+(?:\.\d{2})?|\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY|million|billion|M|B)\b', re.IGNORECASE),
            'percentages': re.compile(r'\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*percent\b', re.IGNORECASE),
            
            # Asset value patterns
            'asset_values': re.compile(r'(?:asset\s+value|total\s+value|insured\s+value|sum\s+insured)[\s:]*[\$£€¥]?\s*[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B))?', re.IGNORECASE),
            'coverage_limits': re.compile(r'(?:coverage\s+limit|limit\s+of\s+liability|maximum\s+coverage)[\s:]*[\$£€¥]?\s*[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B))?', re.IGNORECASE),
            
            # Location patterns
            'addresses': re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Place|Pl)\b', re.IGNORECASE),
            'cities': re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b'),
            'countries': re.compile(r'\b(?:United States|USA|Canada|Mexico|United Kingdom|UK|Germany|France|Japan|Australia|Brazil|India|China)\b', re.IGNORECASE),
            
            # Industry patterns
            'industries': re.compile(r'\b(?:manufacturing|construction|retail|healthcare|technology|finance|energy|transportation|agriculture|mining|real estate|hospitality)\b', re.IGNORECASE),
            
            # Construction types
            'construction_types': re.compile(r'\b(?:steel|concrete|wood|brick|masonry|frame|reinforced|non-combustible|fire-resistant)\b', re.IGNORECASE),
            
            # Occupancy types
            'occupancy_types': re.compile(r'\b(?:office|warehouse|retail|manufacturing|residential|hotel|restaurant|hospital|school|church)\b', re.IGNORECASE),
            
            # Credit ratings
            'credit_ratings': re.compile(r'\b(?:AAA|AA\+?|A\+?|BBB\+?|BB\+?|B\+?|CCC\+?|CC|C|D)\b'),
            
            # Dates
            'dates': re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b', re.IGNORECASE),
        }
    
    def _initialize_validation_rules(self):
        """Initialize validation rules for extracted data."""
        self.validation_rules = {
            'asset_value_range': (0, 10_000_000_000),  # 0 to 10 billion
            'coverage_limit_range': (0, 10_000_000_000),
            'required_fields': ['asset_type', 'location'],
            'valid_risk_levels': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
            'valid_credit_ratings': ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D']
        }
    
    async def extract_risk_parameters(self, text: str) -> Tuple[RiskParameters, Dict[str, float]]:
        """
        Extract risk parameters from text using NER and pattern matching.
        
        Args:
            text: Input text to extract risk parameters from
            
        Returns:
            Tuple of RiskParameters and confidence scores
        """
        logger.info("Extracting risk parameters from text")
        
        try:
            # Initialize extracted data
            extracted_data = {}
            confidence_scores = {}
            
            # Extract using NER models
            ner_entities = await self._extract_ner_entities(text)
            
            # Extract using pattern matching
            pattern_data = self._extract_pattern_data(text)
            
            # Extract using zero-shot classification
            classification_data = await self._classify_asset_type(text)
            
            # Combine all extraction methods
            extracted_data.update(pattern_data)
            extracted_data.update(classification_data)
            
            # Process NER entities
            for entity in ner_entities:
                if entity['label'] in ['LOC', 'LOCATION']:
                    if 'location' not in extracted_data:
                        extracted_data['location'] = entity['word']
                        confidence_scores['location'] = entity['confidence']
                elif entity['label'] in ['ORG', 'ORGANIZATION']:
                    if 'industry_sector' not in extracted_data:
                        extracted_data['industry_sector'] = entity['word']
                        confidence_scores['industry_sector'] = entity['confidence']
            
            # Create RiskParameters object
            risk_params = RiskParameters(
                id="temp",  # Will be set by the database
                application_id="temp",  # Will be set by the application
                asset_value=extracted_data.get('asset_value'),
                coverage_limit=extracted_data.get('coverage_limit'),
                asset_type=extracted_data.get('asset_type'),
                location=extracted_data.get('location'),
                industry_sector=extracted_data.get('industry_sector'),
                construction_type=extracted_data.get('construction_type'),
                occupancy=extracted_data.get('occupancy')
            )
            
            logger.info(f"Successfully extracted risk parameters: {len(extracted_data)} fields")
            return risk_params, confidence_scores
            
        except Exception as e:
            logger.error(f"Error extracting risk parameters: {str(e)}")
            raise
    
    async def extract_financial_data(self, text: str) -> Tuple[FinancialData, Dict[str, float]]:
        """
        Extract financial data using FinBERT and pattern matching.
        
        Args:
            text: Input text to extract financial data from
            
        Returns:
            Tuple of FinancialData and confidence scores
        """
        logger.info("Extracting financial data from text")
        
        try:
            extracted_data = {}
            confidence_scores = {}
            
            # Extract financial amounts using patterns
            financial_amounts = self._extract_financial_amounts(text)
            extracted_data.update(financial_amounts)
            
            # Extract credit ratings
            credit_rating = self._extract_credit_rating(text)
            if credit_rating:
                extracted_data['credit_rating'] = credit_rating
                confidence_scores['credit_rating'] = 0.9  # High confidence for pattern match
            
            # Use FinBERT for financial sentiment analysis
            financial_sentiment = await self._analyze_financial_sentiment(text)
            if financial_sentiment:
                extracted_data['financial_strength_rating'] = financial_sentiment['label']
                confidence_scores['financial_strength_rating'] = financial_sentiment['score']
            
            # Create FinancialData object
            financial_data = FinancialData(
                id="temp",  # Will be set by the database
                application_id="temp",  # Will be set by the application
                revenue=extracted_data.get('revenue'),
                assets=extracted_data.get('assets'),
                liabilities=extracted_data.get('liabilities'),
                credit_rating=extracted_data.get('credit_rating'),
                financial_strength_rating=extracted_data.get('financial_strength_rating')
            )
            
            logger.info(f"Successfully extracted financial data: {len(extracted_data)} fields")
            return financial_data, confidence_scores
            
        except Exception as e:
            logger.error(f"Error extracting financial data: {str(e)}")
            raise
    
    async def extract_geographic_info(self, text: str) -> Dict[str, Any]:
        """
        Extract geographic information from text.
        
        Args:
            text: Input text to extract geographic info from
            
        Returns:
            Dictionary containing geographic information
        """
        logger.info("Extracting geographic information")
        
        try:
            geo_info = {}
            
            # Extract addresses
            addresses = self.patterns['addresses'].findall(text)
            if addresses:
                geo_info['addresses'] = addresses
            
            # Extract cities
            cities = self.patterns['cities'].findall(text)
            if cities:
                geo_info['cities'] = cities
            
            # Extract countries
            countries = self.patterns['countries'].findall(text)
            if countries:
                geo_info['countries'] = countries
            
            # Use zero-shot classification for geographic regions
            if text.strip():
                geographic_labels = [
                    "North America", "South America", "Europe", "Asia", 
                    "Africa", "Oceania", "Middle East"
                ]
                
                result = self.zero_shot_classifier(text, geographic_labels)
                if result['scores'][0] > 0.5:  # High confidence threshold
                    geo_info['region'] = result['labels'][0]
                    geo_info['region_confidence'] = result['scores'][0]
            
            logger.info(f"Extracted geographic info: {len(geo_info)} fields")
            return geo_info
            
        except Exception as e:
            logger.error(f"Error extracting geographic info: {str(e)}")
            return {}
    
    async def classify_asset_type(self, description: str) -> Dict[str, Any]:
        """
        Classify asset type using zero-shot classification.
        
        Args:
            description: Asset description text
            
        Returns:
            Dictionary containing asset classification
        """
        return await self._classify_asset_type(description)
    
    async def validate_extracted_data(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate extracted data against business rules and quality checks.
        
        Args:
            data: Dictionary of extracted data
            
        Returns:
            ValidationResult with validation status and messages
        """
        logger.info("Validating extracted data")
        
        errors = []
        warnings = []
        
        try:
            # Check required fields
            for field in self.validation_rules['required_fields']:
                if field not in data or not data[field]:
                    errors.append(f"Required field '{field}' is missing or empty")
            
            # Validate asset value range
            if 'asset_value' in data and data['asset_value'] is not None:
                min_val, max_val = self.validation_rules['asset_value_range']
                if not (min_val <= float(data['asset_value']) <= max_val):
                    errors.append(f"Asset value {data['asset_value']} is outside valid range ({min_val} - {max_val})")
            
            # Validate coverage limit range
            if 'coverage_limit' in data and data['coverage_limit'] is not None:
                min_val, max_val = self.validation_rules['coverage_limit_range']
                if not (min_val <= float(data['coverage_limit']) <= max_val):
                    errors.append(f"Coverage limit {data['coverage_limit']} is outside valid range ({min_val} - {max_val})")
            
            # Validate credit rating
            if 'credit_rating' in data and data['credit_rating'] is not None:
                if data['credit_rating'] not in self.validation_rules['valid_credit_ratings']:
                    warnings.append(f"Credit rating '{data['credit_rating']}' is not in standard format")
            
            # Check for data consistency
            if ('asset_value' in data and 'coverage_limit' in data and 
                data['asset_value'] is not None and data['coverage_limit'] is not None):
                if float(data['coverage_limit']) > float(data['asset_value']):
                    warnings.append("Coverage limit exceeds asset value")
            
            # Check for completeness
            total_fields = len(data)
            if total_fields < 5:
                warnings.append(f"Limited data extracted ({total_fields} fields). Consider manual review.")
            
            is_valid = len(errors) == 0
            
            logger.info(f"Validation complete: {'Valid' if is_valid else 'Invalid'}, {len(errors)} errors, {len(warnings)} warnings")
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                warnings=warnings
            )
    
    async def process_document_text(self, text: str) -> ExtractionResult:
        """
        Process document text and extract all relevant information.
        
        Args:
            text: Document text to process
            
        Returns:
            ExtractionResult with all extracted data
        """
        logger.info("Processing document text for data extraction")
        
        try:
            extracted_data = {}
            confidence_scores = {}
            
            # Extract risk parameters
            risk_params, risk_confidence = await self.extract_risk_parameters(text)
            extracted_data['risk_parameters'] = risk_params.dict()
            confidence_scores.update(risk_confidence)
            
            # Extract financial data
            financial_data, financial_confidence = await self.extract_financial_data(text)
            extracted_data['financial_data'] = financial_data.dict()
            confidence_scores.update(financial_confidence)
            
            # Extract geographic information
            geo_info = await self.extract_geographic_info(text)
            extracted_data['geographic_info'] = geo_info
            
            # Validate all extracted data
            all_data = {**extracted_data['risk_parameters'], **extracted_data['financial_data'], **geo_info}
            validation_result = await self.validate_extracted_data(all_data)
            
            logger.info("Document text processing completed successfully")
            
            return ExtractionResult(
                extracted_data=extracted_data,
                confidence_scores=confidence_scores,
                validation_result=validation_result
            )
            
        except Exception as e:
            logger.error(f"Error processing document text: {str(e)}")
            return ExtractionResult(
                extracted_data={},
                confidence_scores={},
                validation_result=ValidationResult(
                    is_valid=False,
                    errors=[f"Processing error: {str(e)}"],
                    warnings=[]
                )
            )
    
    # Private helper methods
    
    async def _extract_ner_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using the NER model."""
        try:
            # Tokenize and predict
            inputs = self.ner_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.ner_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_ids = torch.argmax(predictions, dim=-1)
            
            # Convert predictions to entities
            tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            entities = []
            
            for i, (token, pred_id) in enumerate(zip(tokens, predicted_ids[0])):
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    label = self.ner_model.config.id2label[pred_id.item()]
                    if label != 'O':  # Not 'Outside' label
                        confidence = predictions[0][i][pred_id].item()
                        entities.append({
                            'word': token,
                            'label': label,
                            'confidence': confidence
                        })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in NER extraction: {str(e)}")
            return []
    
    def _extract_pattern_data(self, text: str) -> Dict[str, Any]:
        """Extract data using regex patterns."""
        extracted = {}
        
        try:
            # Extract asset values
            asset_matches = self.patterns['asset_values'].findall(text)
            if asset_matches:
                # Parse the first match and convert to decimal
                amount_str = asset_matches[0]
                amount = self._parse_currency_amount(amount_str)
                if amount:
                    extracted['asset_value'] = amount
            
            # Extract coverage limits
            coverage_matches = self.patterns['coverage_limits'].findall(text)
            if coverage_matches:
                amount_str = coverage_matches[0]
                amount = self._parse_currency_amount(amount_str)
                if amount:
                    extracted['coverage_limit'] = amount
            
            # Extract construction type
            construction_matches = self.patterns['construction_types'].findall(text)
            if construction_matches:
                extracted['construction_type'] = construction_matches[0].lower()
            
            # Extract occupancy
            occupancy_matches = self.patterns['occupancy_types'].findall(text)
            if occupancy_matches:
                extracted['occupancy'] = occupancy_matches[0].lower()
            
            return extracted
            
        except Exception as e:
            logger.error(f"Error in pattern extraction: {str(e)}")
            return {}
    
    async def _classify_asset_type(self, text: str) -> Dict[str, Any]:
        """Classify asset type using zero-shot classification."""
        try:
            asset_labels = [
                "commercial building", "residential building", "industrial facility",
                "warehouse", "office building", "retail store", "manufacturing plant",
                "hospital", "school", "hotel", "restaurant", "data center"
            ]
            
            if text.strip():
                result = self.zero_shot_classifier(text, asset_labels)
                if result['scores'][0] > 0.3:  # Lower threshold for asset type
                    return {
                        'asset_type': result['labels'][0],
                        'asset_type_confidence': result['scores'][0]
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error in asset type classification: {str(e)}")
            return {}
    
    def _extract_financial_amounts(self, text: str) -> Dict[str, Any]:
        """Extract financial amounts from text."""
        extracted = {}
        
        try:
            # Look for revenue indicators
            revenue_patterns = [
                r'(?:revenue|sales|income)[\s:]*[\$£€¥]?\s*[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B))?',
                r'annual\s+(?:revenue|sales)[\s:]*[\$£€¥]?\s*[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B))?'
            ]
            
            for pattern in revenue_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    amount = self._parse_currency_amount(matches[0])
                    if amount:
                        extracted['revenue'] = amount
                        break
            
            # Look for asset indicators
            asset_patterns = [
                r'(?:total\s+assets|assets)[\s:]*[\$£€¥]?\s*[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B))?'
            ]
            
            for pattern in asset_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    amount = self._parse_currency_amount(matches[0])
                    if amount:
                        extracted['assets'] = amount
                        break
            
            # Look for liability indicators
            liability_patterns = [
                r'(?:liabilities|debt)[\s:]*[\$£€¥]?\s*[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B))?'
            ]
            
            for pattern in liability_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    amount = self._parse_currency_amount(matches[0])
                    if amount:
                        extracted['liabilities'] = amount
                        break
            
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting financial amounts: {str(e)}")
            return {}
    
    def _extract_credit_rating(self, text: str) -> Optional[str]:
        """Extract credit rating from text."""
        try:
            matches = self.patterns['credit_ratings'].findall(text)
            if matches:
                return matches[0]
            return None
            
        except Exception as e:
            logger.error(f"Error extracting credit rating: {str(e)}")
            return None
    
    async def _analyze_financial_sentiment(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze financial sentiment using FinBERT."""
        try:
            # Extract financial sentences
            sentences = text.split('.')
            financial_sentences = [
                s for s in sentences 
                if any(keyword in s.lower() for keyword in [
                    'profit', 'loss', 'revenue', 'income', 'financial', 
                    'earnings', 'cash', 'debt', 'assets', 'liabilities'
                ])
            ]
            
            if financial_sentences:
                # Analyze the most relevant financial sentence
                text_to_analyze = financial_sentences[0][:512]  # Limit length
                result = self.financial_classifier(text_to_analyze)
                
                if result and len(result) > 0:
                    return {
                        'label': result[0]['label'],
                        'score': result[0]['score']
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in financial sentiment analysis: {str(e)}")
            return None
    
    def _parse_currency_amount(self, amount_str: str) -> Optional[Decimal]:
        """Parse currency amount string to Decimal."""
        try:
            # Remove currency symbols and clean up
            cleaned = re.sub(r'[^\d.,]', '', amount_str)
            cleaned = cleaned.replace(',', '')
            
            if not cleaned:
                return None
            
            # Handle decimal points
            if '.' in cleaned:
                amount = Decimal(cleaned)
            else:
                amount = Decimal(cleaned)
            
            # Handle multipliers (million, billion)
            if 'million' in amount_str.lower() or 'm' in amount_str.lower():
                amount *= 1_000_000
            elif 'billion' in amount_str.lower() or 'b' in amount_str.lower():
                amount *= 1_000_000_000
            
            return amount
            
        except Exception as e:
            logger.error(f"Error parsing currency amount '{amount_str}': {str(e)}")
            return None