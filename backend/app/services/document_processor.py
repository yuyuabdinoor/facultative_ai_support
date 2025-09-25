from typing import Dict, Any, Optional
import docx
import pandas as pd
import PyPDF2
from transformers import pipeline
import requests
import json
import asyncio
import argparse
from pathlib import Path

class DocumentProcessor:
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.text_analyzer = pipeline("zero-shot-classification")
        self.ner_pipeline = pipeline("ner")
        self.results = {}
        
    async def process(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Process document with provided or stored file path"""
        self.file_path = file_path or self.file_path
        if not self.file_path:
            raise ValueError("No file path provided")
            
        if not Path(self.file_path).exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        # Extract and analyze
        text = await self.extract_text(self.file_path)
        location = self._extract_location(text)
        risk_analysis = await self.analyze_location_risk(location)
        
        self.results = {
            "file_path": self.file_path,
            "extracted_text": text[:500] + "..." if len(text) > 500 else text,  # Preview
            "location": location,
            "risk_analysis": risk_analysis
        }
        
        return self.results

    async def extract_text(self, file_path: str) -> str:
        """Extract text from multiple document formats"""
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = " ".join([page.extract_text() for page in reader.pages])
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text = " ".join([paragraph.text for paragraph in doc.paragraphs])
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            text = df.to_string()
        else:
            with open(file_path, 'r') as file:
                text = file.read()
        return text

    async def analyze_location_risk(self, location: str) -> Dict[str, Any]:
        """Query LLM for location risk analysis"""
        prompt = f"""
        Analyze the risk factors for insurance/reinsurance in {location}. Consider:
        1. Political stability
        2. Natural disaster risk
        3. Security situation
        4. Economic conditions
        5. Infrastructure quality
        Format the response as JSON with risk_level and factors.
        """
        # This would use your configured LLM API
        response = self.text_analyzer(
            prompt,
            candidate_labels=["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]
        )
        
        return {
            "risk_level": response["labels"][0],
            "confidence": response["scores"][0],
            "factors": self._get_location_factors(location)
        }

    def _get_location_factors(self, location: str) -> Dict[str, Any]:
        """Get real-time location data from external APIs"""
        # Example using free APIs - replace with your preferred providers
        try:
            # Get geopolitical risk data
            geopolitical = requests.get(
                f"https://api.geopolitical-risk.org/location/{location}"
            ).json()
            
            # Get natural disaster risk data
            disaster_risk = requests.get(
                f"https://api.disaster-risk.org/assessment/{location}"
            ).json()
            
            return {
                "geopolitical": geopolitical,
                "natural_disasters": disaster_risk,
                "timestamp": pd.Timestamp.now()
            }
        except Exception as e:
            return {"error": str(e)}

    def _extract_location(self, text: str) -> str:
        """Extract location from text using NER"""
        entities = self.ner_pipeline(text)
        locations = [entity for entity in entities if entity['entity'] == 'LOC']
        return locations[0]['word'] if locations else "Unknown"

    def display_results(self):
        """Display processing results in a formatted way"""
        if not self.results:
            print("No results available. Please run process() first.")
            return
            
        print("\n=== Document Analysis Results ===")
        print(f"File: {self.results['file_path']}")
        print("\nExtracted Text Preview:")
        print(self.results['extracted_text'])
        print("\nLocation Analysis:")
        print(f"Detected Location: {self.results['location']}")
        print("\nRisk Analysis:")
        for key, value in self.results['risk_analysis'].items():
            print(f"{key}: {value}")

async def main():
    parser = argparse.ArgumentParser(description='Process a document for risk analysis')
    parser.add_argument('file_path', type=str, help='Path to the document to analyze')
    args = parser.parse_args()
    
    try:
        processor = DocumentProcessor()
        results = await processor.process(args.file_path)
        processor.display_results()
    except Exception as e:
        print(f"Error processing document: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
