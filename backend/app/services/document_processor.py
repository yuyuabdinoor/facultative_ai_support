from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from datetime import datetime
import pandas as pd
import asyncio
import re
import json
import openai

from dotenv import load_dotenv; load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    timestamp: Optional[str] = None

# ------------------- Calculation Service -------------------
class CalculationService:
    def __init__(self, exchange_csv: str):
        self.rates = pd.read_csv(exchange_csv).set_index("Currency")["Rate_to_KES"].to_dict()

    def _get_exchange_rate(self, currency: str) -> Decimal:
        currency = currency.upper()
        return Decimal(str(self.rates.get(currency, 1.0)))

    async def convert_to_kes(self, amount: Decimal, currency: str) -> Decimal:
        return amount * self._get_exchange_rate(currency)

    def calculate_premium_rate(self, premium: Decimal, tsi: Decimal) -> Decimal:
        if tsi == 0:
            return Decimal("0.00")
        return ((premium / tsi) * 100).quantize(Decimal("0.001"), ROUND_HALF_UP)

    def calculate_loss_ratio(self, paid: Decimal, outstanding: Decimal, recoveries: Decimal, earned: Decimal) -> Dict[str, Any]:
        if earned == 0:
            return {"ratio": Decimal("0.00"), "status": "N/A", "recommendation": "No earned premium"}
        incurred = paid + outstanding - recoveries
        ratio = (incurred / earned * 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
        if ratio < 60:
            status, recommendation = "LOW_RISK", "ACCEPT"
        elif ratio <= 80:
            status, recommendation = "MEDIUM_RISK", "REFER"
        else:
            status, recommendation = "HIGH_RISK", "DECLINE"
        return {"ratio": ratio, "status": status, "recommendation": recommendation}

    def calculate_facultative_share(self, gross_premium: Decimal, tsi: Decimal, share_percent: Decimal) -> Dict[str, Decimal]:
        share_decimal = share_percent / Decimal("100")
        return {
            "accepted_premium": (gross_premium * share_decimal).quantize(Decimal("0.01")),
            "accepted_liability": (tsi * share_decimal).quantize(Decimal("0.01"))
        }

# ------------------- Document Processor -------------------
class DocumentProcessor:
    def __init__(self, file_path: str, exchange_csv: str):
        self.file_path = file_path
        self.calculator = CalculationService(exchange_csv)
        self.results = {}

    async def process(self) -> Dict[str, Any]:
        data = await self._extract_data()
        data = await self._compute_fields(data)
        self.results = data
        return data

    async def _extract_data(self) -> Dict[str, Any]:
        ext = Path(self.file_path).suffix.lower()
        df = pd.read_excel(self.file_path) if ext == ".xlsx" else pd.read_csv(self.file_path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        row = df.iloc[0] if not df.empty else {}

        def safe_get(col, default="UNKNOWN"):
            return row.get(col, default) or default

        return {
            "insured": safe_get("insured"),
            "cedant": safe_get("cedant"),
            "broker": safe_get("broker"),
            "perils_covered": re.findall(r"Fire|Flood|Earthquake|Cyclone|BI|Machinery Breakdown|IAR", str(safe_get("perils covered", "")), flags=re.I),
            "geographical_limit": safe_get("geographical limit"),
            "situation_of_risk": safe_get("situation of risk"),
            "occupation_of_insured": safe_get("occupation of insured"),
            "main_activities": safe_get("main activities"),
            "period_of_insurance": safe_get("period of insurance"),
            "total_sum_insured": Decimal(str(safe_get("total sums insured (fac ri)", 0))),
            "premium": Decimal(str(safe_get("premium", 0))),
            "currency": safe_get("currency", "USD"),
            "paid_losses": Decimal(str(safe_get("paid losses", 0))),
            "outstanding_reserves": Decimal(str(safe_get("outstanding reserves", 0))),
            "recoveries": Decimal(str(safe_get("recoveries", 0))),
            "earned_premium": Decimal(str(safe_get("earned premium", 0))),
            "share_offered_percent": Decimal(str(safe_get("share offered %", 0))),
            "excess_deductible": safe_get("excess", "TBD"),
            "retention_of_cedant": safe_get("retention of cedant", "TBD"),
            "possible_maximum_loss": safe_get("possible maximum loss (pml %)", "TBD"),
            "claims_experience": safe_get("claims exp for the last 3yrs", "NIL"),
            "reinsurance_deductions": safe_get("reinsurance déductions", "TBD"),
            "risk_surveyor_report": safe_get("surveyor’s report (attached)", "☐"),
            "premium_rates": safe_get("premium rates", "TBD"),
            "climate_change_risk_factors": safe_get("climate change risk factors", "TBD"),
        }

    async def _compute_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        tsi = data["total_sum_insured"]
        premium = data["premium"]
        share = data.get("share_offered_percent", Decimal("0"))

        # Convert to KES
        data["total_sum_insured_kes"] = await self.calculator.convert_to_kes(tsi, data["currency"])
        data["premium_kes"] = await self.calculator.convert_to_kes(premium, data["currency"])

        # Premium rate & loss ratio
        data["premium_rate"] = self.calculator.calculate_premium_rate(premium, tsi)
        data["loss_ratio"] = self.calculator.calculate_loss_ratio(
            data["paid_losses"], data["outstanding_reserves"], data["recoveries"], data["earned_premium"]
        )

        # Facultative share
        share_data = self.calculator.calculate_facultative_share(premium, tsi, share)
        data.update(share_data)

        # GPT-generated CAT exposure
        data["cat_exposure"] = await self._generate_cat_exposure(data)

        # GPT ESG assessment
        data["esg_risk_assessment"] = await self._generate_esg(data)

        # GPT climate change risk
        data["climate_change_risk"] = await self._generate_climate_risk(data)

        data["timestamp"] = datetime.utcnow().isoformat()
        return data

    async def _generate_cat_exposure(self, data: Dict[str, Any]) -> str:
        prompt = f"Assess CAT (catastrophe) exposure for an insurance company with these details: {json.dumps(data)}. Give a short explanation."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a reinsurance risk analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    async def _generate_esg(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = f"{data['occupation_of_insured']} {data['main_activities']} {data['perils_covered']} {data['geographical_limit']}"
        prompt = f"Assess ESG risks (environmental, social, governance) for this text. Return JSON with reasoning. Text: {text}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a reinsurance ESG analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"environmental": "Unknown", "social": "Unknown", "governance": "Unknown"}

    async def _generate_climate_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = f"{data['occupation_of_insured']} in {data['geographical_limit']}"
        prompt = f"Analyze climate change risk (HIGH, MODERATE, LOW) with reasoning for JSON output. Text: {text}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a climate risk analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"risk_level": "MODERATE", "reasoning": "Default moderate risk"}

    def display_results(self):
        if not self.results:
            print("No results to display.")
            return
        for k, v in self.results.items():
            print(f"{k}: {v}")

    def export_to_excel(self, path: str = "facultative_analysis_output.xlsx"):
        if not self.results:
            print("No data to export.")
            return
        flat = {}
        for k, v in self.results.items():
            flat[k] = json.dumps(v, indent=2) if isinstance(v, (dict, list)) else v
        pd.DataFrame([flat]).to_excel(path, index=False)
        print(f"Results exported to {path}")

# ------------------- Runner -------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python document_processor.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    exchange_csv = "/Users/yussufabdinoor/facultative_ai_support/exchange_rates.csv"

    async def runner():
        processor = DocumentProcessor(file_path, exchange_csv)
        await processor.process()
        processor.display_results()
        processor.export_to_excel("/Users/yussufabdinoor/facultative_ai_support/facultative_analysis_output.xlsx")

    asyncio.run(runner())
