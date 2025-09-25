import asyncio
from decimal import Decimal
from calculation_service import CalculationService

async def main():
    service = CalculationService()

    # Example: Convert foreign currencies to KES
    usd_to_kes = await service.convert_currency(Decimal("100"), "USD", "KES")
    print(f"100 USD = {usd_to_kes.quantize(Decimal('0.01'))} KES")

    eur_to_kes = await service.convert_currency(Decimal("50"), "EUR", "KES")
    print(f"50 EUR = {eur_to_kes.quantize(Decimal('0.01'))} KES")

    # Example: Premium rate calculation
    premium_rate = service.calculate_premium_rate(Decimal("20000"), Decimal("1000000"))
    print("Premium Rate:", premium_rate)

    # Example: Loss ratio calculation
    loss_ratio = service.calculate_loss_ratio(Decimal("230000"), Decimal("300000"))
    print("Loss Ratio:", loss_ratio)

if __name__ == "__main__":
    asyncio.run(main())
