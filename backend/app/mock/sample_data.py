from typing import Dict, Any
import json

# -------------------------------
# SAMPLE 1: Industrial Factory
# -------------------------------
SAMPLE_EMAIL_TEXT_1 = """
From: broker@reinsurance.com
Subject: Facultative Placement - Industrial Factory Risk

Dear Underwriter,

Insured: ABC Manufacturing Ltd
Cedant: XYZ Insurance Co.
Broker: John Broker Ltd
Location: Industrial Area, Nairobi, Kenya
Sum Insured: USD 25,000,000
Period: 01/01/2024 - 31/12/2024
Perils: Fire, Machinery Breakdown
Retention: 20%
Share Offered: 80%
Claims (3yr): One claim in 2022 - USD 150,000 (machinery breakdown)

Regards,
John Broker
"""

SAMPLE_EXCEL_DATA_1 = {
    "Insured_Name": "ABC Manufacturing Ltd",
    "Cedant": "XYZ Insurance Co.",
    "Broker": "John Broker Ltd",
    "Industry": "Manufacturing",
    "Location": "Nairobi, Kenya",
    "Total_Sum_Insured": 25000000,
    "Currency": "USD",
    "Claims_History": [
        {"year": 2022, "amount": 150000, "type": "Machinery Breakdown"}
    ]
}

SAMPLE_RISK_ANALYSIS_1 = {
    "technical_assessment": {
        "risk_quality_score": 75,
        "protection_measures_score": 80,
        "management_quality_score": 70,
        "overall_technical_score": 75,
        "confidence": 0.85
    },
    "esg_rating": {
        "environmental": "MEDIUM",
        "social": "LOW",
        "governance": "LOW",
        "overall": "LOW",
        "confidence": 0.88
    },
    "recommendations": {
        "decision": "ACCEPT",
        "max_share": 25.0,
        "conditions": [
            "Install automated fire suppression system",
            "Implement regular maintenance schedule",
            "Update staff safety training"
        ],
        "rationale": "Good risk management practices with acceptable loss history",
        "confidence": 0.82
    },
    "risk_factors": {
        "fire": "MEDIUM",
        "machinery": "HIGH",
        "natural_catastrophe": "LOW"
    },
    "overall_confidence": 0.85
}

# -------------------------------
# SAMPLE 2: Airline Risk
# -------------------------------
SAMPLE_EMAIL_TEXT_2 = """
From: broker@aviationrisk.com
Subject: Facultative Placement - Airline Hull & Liability

Dear Underwriter,

Insured: EastAfrica Airways
Cedant: Continental Insurance Co.
Broker: SkyRisk Brokers
Location: Jomo Kenyatta International Airport, Nairobi
Sum Insured: USD 120,000,000
Period: 01/04/2024 - 31/03/2025
Perils: Hull All Risk, Third Party Liability
Retention: 10%
Share Offered: 90%
Claims (3yr): 2021 - USD 2,000,000 (ground damage); 2023 - USD 500,000 (bird strike)

Sincerely,
Sarah Broker
"""

SAMPLE_EXCEL_DATA_2 = {
    "Insured_Name": "EastAfrica Airways",
    "Cedant": "Continental Insurance Co.",
    "Broker": "SkyRisk Brokers",
    "Industry": "Aviation",
    "Location": "Nairobi, Kenya",
    "Total_Sum_Insured": 120000000,
    "Currency": "USD",
    "Claims_History": [
        {"year": 2021, "amount": 2000000, "type": "Ground Damage"},
        {"year": 2023, "amount": 500000, "type": "Bird Strike"}
    ]
}

SAMPLE_RISK_ANALYSIS_2 = {
    "technical_assessment": {
        "risk_quality_score": 80,
        "protection_measures_score": 85,
        "management_quality_score": 75,
        "overall_technical_score": 80,
        "confidence": 0.9
    },
    "esg_rating": {
        "environmental": "MEDIUM",
        "social": "LOW",
        "governance": "LOW",
        "overall": "LOW",
        "confidence": 0.9
    },
    "recommendations": {
        "decision": "ACCEPT",
        "max_share": 25.0,
        "conditions": [
            "Install automated fire suppression system",
            "Implement regular maintenance schedule",
            "Update staff safety training"
        ],
        "rationale": "Good risk management practices with acceptable loss history",
        "confidence": 0.85
    },
    "risk_factors": {
        "fire": "MEDIUM",
        "machinery": "HIGH",
        "natural_catastrophe": "LOW"
    },
    "overall_confidence": 0.9
}

# -------------------------------
# SAMPLE 3: Hospital Risk
# -------------------------------
SAMPLE_EMAIL_TEXT_3 = """
From: broker@healthre.com
Subject: Facultative Placement - Private Hospital Risk

Dear Underwriter,

Insured: Nairobi West Hospital
Cedant: HealthSure Insurance Ltd
Broker: MedRisk Brokers
Location: Lang’ata, Nairobi, Kenya
Sum Insured: USD 60,000,000
Period: 01/07/2024 - 30/06/2025
Perils: Fire, Equipment Breakdown, Liability
Retention: 15%
Share Offered: 85%
Claims (3yr): 2022 - USD 800,000 (fire in pharmacy); 2023 - USD 200,000 (MRI machine breakdown)

Best regards,
David Broker
"""

SAMPLE_EXCEL_DATA_3 = {
    "Insured_Name": "Nairobi West Hospital",
    "Cedant": "HealthSure Insurance Ltd",
    "Broker": "MedRisk Brokers",
    "Industry": "Healthcare",
    "Location": "Nairobi, Kenya",
    "Total_Sum_Insured": 60000000,
    "Currency": "USD",
    "Claims_History": [
        {"year": 2022, "amount": 800000, "type": "Fire"},
        {"year": 2023, "amount": 200000, "type": "Equipment Breakdown"}
    ]
}

SAMPLE_RISK_ANALYSIS_3 = {
    "technical_assessment": {
        "risk_quality_score": 70,
        "protection_measures_score": 75,
        "management_quality_score": 80,
        "overall_technical_score": 70,
        "confidence": 0.8
    },
    "esg_rating": {
        "environmental": "MEDIUM",
        "social": "LOW",
        "governance": "LOW",
        "overall": "LOW",
        "confidence": 0.8
    },
    "recommendations": {
        "decision": "ACCEPT",
        "max_share": 25.0,
        "conditions": [
            "Install automated fire suppression system",
            "Implement regular maintenance schedule",
            "Update staff safety training"
        ],
        "rationale": "Good risk management practices with acceptable loss history",
        "confidence": 0.8
    },
    "risk_factors": {
        "fire": "MEDIUM",
        "machinery": "HIGH",
        "natural_catastrophe": "LOW"
    },
    "overall_confidence": 0.8
}

# -------------------------------
# Utility functions
# -------------------------------
def get_mock_email(sample_id: int = 1) -> str:
    return {
        1: SAMPLE_EMAIL_TEXT_1,
        2: SAMPLE_EMAIL_TEXT_2,
        3: SAMPLE_EMAIL_TEXT_3,
    }.get(sample_id, SAMPLE_EMAIL_TEXT_1)

def get_mock_excel(sample_id: int = 1) -> str:
    return json.dumps({
        1: SAMPLE_EXCEL_DATA_1,
        2: SAMPLE_EXCEL_DATA_2,
        3: SAMPLE_EXCEL_DATA_3,
    }.get(sample_id, SAMPLE_EXCEL_DATA_1), indent=2)

def get_mock_risk_analysis(sample_id: int = 1) -> dict:
    return {
        1: SAMPLE_RISK_ANALYSIS_1,
        2: SAMPLE_RISK_ANALYSIS_2,
        3: SAMPLE_RISK_ANALYSIS_3,
    }.get(sample_id, SAMPLE_RISK_ANALYSIS_1)
