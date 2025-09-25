import asyncio
from decimal import Decimal
import pandas as pd


class CalculationService:
    def __init__(self, csv_path="../../exchange_rates.csv"):
        # Load exchange rates once when the service starts
        self.rates = pd.read_csv(csv_path).set_index("Currency")["Rate_to_KES"].to_dict()

    def _get_exchange_rate(self, from_currency: str, to_currency: str) -> Decimal:
        """Look up conversion rate from CSV. Supports only conversion to KES."""
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()

        if to_currency != "KES":
            raise ValueError("Only conversion TO KES is supported.")

        if from_currency not in self.rates:
            raise ValueError(f"Currency {from_currency} not found in exchange rates CSV.")

        return Decimal(str(self.rates[from_currency]))

    async def convert_currency(self, amount: Decimal, from_currency: str, to_currency: str) -> Decimal:
        rate = self._get_exchange_rate(from_currency, to_currency)
        return amount * rate

    def calculate_premium_rate(self, premium: Decimal, sum_insured: Decimal):
        rate = (premium / sum_insured) * 100
        min_rate = Decimal("0.2")
        is_valid = rate >= min_rate
        return {
            "rate": rate.quantize(Decimal("0.001")),
            "unit": "%",
            "validation": {
                "is_valid": is_valid,
                "min_rate": min_rate,
                "explanation": (
                    f"Premium rate {'meets' if is_valid else 'does not meet'} "
                    f"minimum technical rate of {min_rate}%"
                ),
            },
        }

    def calculate_loss_ratio(self, claims: Decimal, premiums: Decimal):
        ratio = (claims / premiums) * 100
        if ratio < 60:
            status, recommendation = "LOW_RISK", "ACCEPT"
        elif ratio <= 80:
            status, recommendation = "MEDIUM_RISK", "REFER - Loss ratio requires additional review"
        else:
            status, recommendation = "HIGH_RISK", "DECLINE"
        return {
            "ratio": ratio.quantize(Decimal("0.01")),
            "analysis": {"status": status, "recommendation": recommendation},
        }


# Test harness
async def test():
    service = CalculationService()

    premium_rate = service.calculate_premium_rate(Decimal("20000"), Decimal("1000000"))
    print("Premium Rate Test:", premium_rate)

    loss_ratio = service.calculate_loss_ratio(Decimal("230000"), Decimal("300000"))
    print("Loss Ratio Test:", loss_ratio)

    conversion = await service.convert_currency(Decimal("100"), "USD", "KES")
    print(f"100 USD = {conversion.quantize(Decimal('0.01'))} KES")

    conversion2 = await service.convert_currency(Decimal("50"), "GBP", "KES")
    print(f"50 GBP = {conversion2.quantize(Decimal('0.01'))} KES")


if __name__ == "__main__":
    asyncio.run(test())
