from decimal import Decimal, ROUND_HALF_UP
from typing import Dict

def calculate_premium_rate(premium: Decimal, tsi: Decimal, use_permille: bool = False) -> Decimal:
    """Calculate premium rate in percentage or permille"""
    multiplier = Decimal('1000') if use_permille else Decimal('100')
    return (premium / tsi * multiplier).quantize(Decimal('0.001'), ROUND_HALF_UP)

def calculate_premium(tsi: Decimal, rate: Decimal, is_permille: bool = False) -> Decimal:
    """Calculate premium from TSI and rate"""
    divisor = Decimal('1000') if is_permille else Decimal('100')
    return (tsi * rate / divisor).quantize(Decimal('0.01'), ROUND_HALF_UP)

def calculate_loss_ratio(
    paid_losses: Decimal,
    outstanding: Decimal,
    recoveries: Decimal,
    earned_premium: Decimal
) -> Decimal:
    """Calculate loss ratio percentage"""
    incurred = paid_losses + outstanding - recoveries
    return (incurred / earned_premium * Decimal('100')).quantize(Decimal('0.01'), ROUND_HALF_UP)

def calculate_facultative_share(
    gross_premium: Decimal,
    tsi: Decimal,
    share_percent: Decimal
) -> Dict[str, Decimal]:
    """Calculate facultative share amounts"""
    share_decimal = share_percent / Decimal('100')
    return {
        'accepted_premium': (gross_premium * share_decimal).quantize(Decimal('0.01')),
        'accepted_liability': (tsi * share_decimal).quantize(Decimal('0.01'))
    }
