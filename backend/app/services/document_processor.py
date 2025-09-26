# backend/app/services/document_processor.py
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from pathlib import Path
import pandas as pd
import re
import json
import asyncio
import openai
from calculation_service import CalculationService

# Set your OpenAI API key here
openai.api_key = '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'

@dataclass
class ExtractedInfo:
    insured: Optional[str] = None
    cedant: Optional[str] = None
    broker: Optional[str] = None
    occupation_of_insured: Optional[str] = None
    main_activities: Optional[str] = None
    perils_covered: Optional[List[str]] = None
    geographical_limit: Optional[str] = None
    situation_of_risk: Optional[str] = None
    total_sum_insured: Optional[Decimal] = None
    premium: Optional[Decimal] = None
    currency: Optional[str] = "USD"
    premium_rate_percent: Optional[Decimal] = None
    paid_losses: Optional[Decimal] = None
    outstanding_reserves: Optional[Decimal] = None
    recoveries: Optional[Decimal] = None
    earned_premium: Optional[Decimal] = None
    share_offered_percent: Optional[Decimal] = None
    excess_deductible: Optional[str] = None
    retention_of_cedant: Optional[str] = None
    possible_maximum_loss: Optional[str] = None
    cat_exposure: Optional[str] = None
    claims_experience: Optional[str] = None
    reinsurance_deductions: Optional[str] = None
    period_of_insurance: Optional[str] = None
    risk_surveyor_report: Optional[str] = None
    climate_change_risk_factors: Optional[str] = None
    esg_risk_assessment: Optional[Dict[str, str]] = None
    climate_change_risk: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

class DocumentProcessor:
    def __init__(self, file_path: Optional[str] = None, exchange_csv: str = "/Users/yussufabdinoor/facultative_ai_support/exchange_rates.csv"):
        self.file_path = file_path
        self.results: Dict[str, Any] = {}
        self.calculator = CalculationService(csv_path=exchange_csv)

    async def process(self, file_path: Optional[str] = None):
        self.file_path = file_path or self.file_path
        if not self.file_path or not Path(self.file_path).exists():
            raise ValueError(f"File not found: {self.file_path}")

        data = await self._extract_structured_data_from_file(self.file_path)
        data = self._compute_fields(data)

        # Generate CAT exposure via GPT
        data["cat_exposure"] = await self._generate_cat_exposure(data)

        # ESG and climate risk
        data["esg_risk_assessment"] = await self._analyze_esg(data)
        data["climate_change_risk"] = await self._analyze_climate_risk(data)

        data["timestamp"] = datetime.utcnow().isoformat()
        self.results = data

        # Save to Excel
        self._export_to_excel("facultative_output.xlsx")
        return data

    async def _extract_structured_data_from_file(self, file_path: str) -> Dict[str, Any]:
        ext = Path(file_path).suffix.lower()
        if ext == ".xlsx":
            df = pd.read_excel(file_path)
        elif ext == ".csv":
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file type")

        df.columns = [str(c).strip().lower() for c in df.columns]
        first_row = df.iloc[0] if not df.empty else {}

        def safe_get(col, default=None):
            val = first_row.get(col, default)
            if pd.isna(val):
                return default
            return val

        data = {
            "insured": safe_get("insured", "UNKNOWN"),
            "cedant": safe_get("cedant", "UNKNOWN"),
            "broker": safe_get("broker", "UNKNOWN"),
            "perils_covered": self._extract_perils(str(safe_get("perils covered", ""))),
            "geographical_limit": safe_get("geographical limit", "UNKNOWN"),
            "situation_of_risk": safe_get("situation of risk", "UNKNOWN"),
            "occupation_of_insured": safe_get("occupation of insured", "UNKNOWN"),
            "main_activities": safe_get("main activities", "UNKNOWN"),
            "period_of_insurance": safe_get("period of insurance", "UNKNOWN"),
            "total_sum_insured": self._safe_decimal(first_row.get("total sums insured (fac ri)", 0)),
            "premium": self._safe_decimal(first_row.get("premium", 0)),
            "currency": safe_get("currency", "USD"),
            "paid_losses": self._safe_decimal(first_row.get("paid losses", 0)),
            "outstanding_reserves": self._safe_decimal(first_row.get("outstanding reserves", 0)),
            "recoveries": self._safe_decimal(first_row.get("recoveries", 0)),
            "earned_premium": self._safe_decimal(first_row.get("earned premium", 0)),
            "share_offered_percent": self._safe_decimal(first_row.get("share offered %", 0)),
            "excess_deductible": safe_get("excess", "TBD"),
            "retention_of_cedant": safe_get("retention of cedant", "TBD"),
            "possible_maximum_loss": safe_get("possible maximum loss (pml %)", "TBD"),
            "claims_experience": safe_get("claims exp for the last 3yrs", "NIL"),
            "reinsurance_deductions": safe_get("reinsurance déductions", "TBD"),
            "risk_surveyor_report": safe_get("surveyor’s report (attached)", "☐"),
            "premium_rates": safe_get("premium rates", "TBD"),
            "climate_change_risk_factors": safe_get("climate change risk factors", "TBD"),
        }
        return data

    def _compute_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        tsi = data.get("total_sum_insured", Decimal(0))
        premium = data.get("premium", Decimal(0))
        currency = data.get("currency", "USD")

        # Convert to KES
        data["total_sum_insured_kes"] = self.calculator.convert_currency(tsi, currency, "KES") if tsi else Decimal("0.00")
        data["premium_kes"] = self.calculator.convert_currency(premium, currency, "KES") if premium else Decimal("0.00")

        # Premium Rate %
        data["premium_rate"] = self.calculator.calculate_premium_rate(premium, tsi)

        # Loss ratio
        data["loss_ratio"] = self.calculator.calculate_loss_ratio(
            data.get("paid_losses", Decimal(0)),
            data.get("outstanding_reserves", Decimal(0)),
            data.get("recoveries", Decimal(0)),
            data.get("earned_premium", Decimal(0))
        )

        # Facultative share
        share = data.get("share_offered_percent", Decimal(0))
        data.update(self.calculator.calculate_facultative_share(premium, tsi, share))
        return data

    def _safe_decimal(self, val) -> Decimal:
        try:
            return Decimal(str(val)).quantize(Decimal("0.01"), ROUND_HALF_UP)
        except:
            return Decimal("0.00")

    def _extract_perils(self, text: str) -> List[str]:
        return re.findall(r"Fire|Flood|Earthquake|Cyclone|BI|Machinery Breakdown|IAR", text, flags=re.I)

    async def _generate_cat_exposure(self, data: Dict[str, Any]) -> str:
        prompt = f"Assess CAT exposure for this insurance: {data.get('main_activities','')}, location: {data.get('geographical_limit','')}. Respond in one line."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a reinsurance analyst."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    async def _analyze_esg(self, data: Dict[str, Any]) -> Dict[str, str]:
        text = f"{data.get('occupation_of_insured','')} {data.get('main_activities','')} {data.get('perils_covered','')} {data.get('geographical_limit','')}"
        prompt = f"Assess ESG risk for this business text. Return JSON with environmental, social, governance keys with reasoning. Text: {text}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a reinsurance ESG analyst."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"environmental":"MEDIUM","social":"LOW","governance":"MEDIUM"}

    async def _analyze_climate_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = f"Analyze climate change risk for {data.get('main_activities','')} in {data.get('geographical_limit','')}."
        prompt = f"Analyze climate change risk (HIGH, MODERATE, LOW) with reasoning in JSON. Text: {text}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a reinsurance climate analyst."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"risk_level":"MODERATE","reasoning":response.choices[0].message.content}

    def _export_to_excel(self, output_path: str):
        pd.DataFrame([self.results]).to_excel(output_path, index=False)

    def display_results(self, format="text"):
        if not self.results:
            print("No results yet. Run process() first.")
            return
        if format == "json":
            print(json.dumps(self.results, indent=2, default=str))
        else:
            for k,v in self.results.items():
                print(f"{k}: {v}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python document_processor.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]

    async def runner():
        processor = DocumentProcessor(file_path)
        await processor.process()
        processor.display_results()

    asyncio.run(runner())
