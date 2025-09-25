from typing import Dict

KENYA_REGIONS = {
    "NAIROBI": {
        "cat_risks": ["flood", "earthquake"],
        "protection_level": "HIGH",
        "political_risk": "LOW"
    },
    "MOMBASA": {
        "cat_risks": ["flood", "cyclone"],
        "protection_level": "MEDIUM",
        "political_risk": "LOW"
    },
    "RIFT_VALLEY": {
        "cat_risks": ["earthquake"],
        "protection_level": "LOW",
        "political_risk": "MEDIUM"
    }
}

KENYA_INDUSTRY_CONCENTRATIONS = {
    "NAIROBI": {
        "MANUFACTURING": 0.4,
        "OFFICE": 0.3,
        "RETAIL": 0.2
    }
}

MARKET_CYCLE_DATA = {
    "current_phase": "HARD",  # or "SOFT"
    "rate_adequacy": 1.2,  # >1 means rates are above technical
    "capacity_availability": "RESTRICTED"
}
