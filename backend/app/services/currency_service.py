"""
Currency service for fetching and transforming FX rates to KES using ExchangeRate-API.
"""
from __future__ import annotations

import os
import logging
from typing import Dict

import requests

logger = logging.getLogger(__name__)

EXCHANGE_RATE_API_URL = "https://v6.exchangerate-api.com/v6/{api_key}/latest/{base}"


def fetch_kes_rates_from_provider(api_key: str, base: str = "USD") -> Dict[str, float]:
    """
    Fetch FX rates from ExchangeRate-API and return a mapping of currency -> KES per unit of that currency.

    Args:
        api_key: ExchangeRate-API key
        base: Base currency used by the provider endpoint (default USD)

    Returns:
        Dict mapping currency code (e.g., "USD", "EUR", ...) to KES per 1 unit of that currency.
    """
    url = EXCHANGE_RATE_API_URL.format(api_key=api_key, base=base)
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("result") != "success":
        raise RuntimeError(f"Provider returned non-success result: {data.get('result')} - {data}")

    rates = data.get("conversion_rates", {})
    if "KES" not in rates:
        raise RuntimeError("Provider response missing KES rate")

    kes_per_base = float(rates["KES"])  # KES per 1 base unit

    # Transform: for each currency C with rate_C_per_base, we want KES_per_C = kes_per_base / rate_C_per_base
    kes_rates: Dict[str, float] = {}
    for cur, rate_c_per_base in rates.items():
        try:
            r = float(rate_c_per_base)
            if r <= 0:
                continue
            kes_rates[cur] = kes_per_base / r
        except Exception:
            continue

    # Ensure KES self-rate
    kes_rates["KES"] = 1.0

    return kes_rates
