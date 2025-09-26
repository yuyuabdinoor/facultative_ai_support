# backend/app/services/calculation_service.py
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional

class CalculationService:
    def __init__(self, csv_path: str = "/Users/yussufabdinoor/facultative_ai_support/exchange_rates.csv"):
        # Load exchange rates once
        try:
            self.rates = pd.read_csv(csv_path).set_index("Currency")["Rate_to_KES"].to_dict()
        except Exception as e:
            print(f"Error loading exchange rates CSV: {e}")
            self.rates = {}

    def _get_exchange_rate(self, from_currency: str, to_currency: str = "KES") -> Decimal:
        """Look up conversion rate. Only supports conversion TO KES."""
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        if to_currency != "KES":
            raise ValueError("Only conversion to KES supported.")
        rate = self.rates.get(from_currency, 1.0)
        return Decimal(str(rate))

    async def convert_currency(self, amount: Decimal, from_currency: str, to_currency: str = "KES") -> Decimal:
        if amount is None:
            amount = Decimal("0.00")
        rate = self._get_exchange_rate(from_currency, to_currency)
        return (amount * rate).quantize(Decimal("0.01"), ROUND_HALF_UP)

    def calculate_premium_rate(self, premium: Optional[Decimal], sum_insured: Optional[Decimal], use_permille: bool = False) -> Decimal:
        """Calculate premium rate safely; returns 0 if TSI is zero"""
        if premium is None:
            premium = Decimal("0.00")
        if sum_insured is None or sum_insured == 0:
            return Decimal("0.00")
        multiplier = Decimal("1000") if use_permille else Decimal("100")
        rate = (premium / sum_insured * multiplier).quantize(Decimal("0.001"), ROUND_HALF_UP)
        return rate

    def calculate_premium(self, tsi: Decimal, rate: Decimal, use_permille: bool = False) -> Decimal:
        if tsi is None:
            tsi = Decimal("0.00")
        if rate is None:
            rate = Decimal("0.00")
        divisor = Decimal("1000") if use_permille else Decimal("100")
        return (tsi * rate / divisor).quantize(Decimal("0.01"), ROUND_HALF_UP)

    def calculate_loss_ratio(self, paid_losses: Optional[Decimal], outstanding: Optional[Decimal], recoveries: Optional[Decimal], earned_premium: Optional[Decimal]) -> Dict[str, Decimal]:
        """Calculate loss ratio safely"""
        paid_losses = paid_losses or Decimal("0.00")
        outstanding = outstanding or Decimal("0.00")
        recoveries = recoveries or Decimal("0.00")
        earned_premium = earned_premium or Decimal("0.00")

        if earned_premium == 0:
            return {"ratio": Decimal("0.00"), "status": "N/A", "recommendation": "No earned premium"}

        incurred = paid_losses + outstanding - recoveries
        ratio = (incurred / earned_premium * Decimal("100")).quantize(Decimal("0.01"), ROUND_HALF_UP)

        if ratio < 60:
            status, recommendation = "LOW_RISK", "ACCEPT"
        elif ratio <= 80:
            status, recommendation = "MEDIUM_RISK", "REFER - Loss ratio requires additional review"
        else:
            status, recommendation = "HIGH_RISK", "DECLINE"

        return {"ratio": ratio, "status": status, "recommendation": recommendation}

    def calculate_facultative_share(self, gross_premium: Optional[Decimal], tsi: Optional[Decimal], share_percent: Optional[Decimal]) -> Dict[str, Decimal]:
        """Calculate facultative share safely"""
        gross_premium = gross_premium or Decimal("0.00")
        tsi = tsi or Decimal("0.00")
        share_percent = share_percent or Decimal("0.00")
        share_decimal = share_percent / Decimal("100")

        accepted_premium = (gross_premium * share_decimal).quantize(Decimal("0.01"), ROUND_HALF_UP)
        accepted_liability = (tsi * share_decimal).quantize(Decimal("0.01"), ROUND_HALF_UP)
        return {"accepted_premium": accepted_premium, "accepted_liability": accepted_liability}
