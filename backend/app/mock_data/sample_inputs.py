from typing import Dict, Any
import json

# -------------------- SAMPLE 1 --------------------
SAMPLE_EMAIL_TEXT_1 = """
From: broker@reinsurance.com
Subject: Facultative Placement - Industrial Factory Risk

Dear Underwriter,

Please find below details for facultative placement consideration:

Insured: ABC Manufacturing Ltd
Location: Industrial Area, Nairobi, Kenya
Sum Insured: USD 25,000,000
Period: 01/01/2024 - 31/12/2024
Perils: Fire, Machinery Breakdown
Claims (3yr): One claim in 2022 - USD 150,000 (machinery breakdown)

Best regards,
John Broker
"""

SAMPLE_EXCEL_DATA_1 = {
    "Insured_Name": "ABC Manufacturing Ltd",
    "Industry": "Manufacturing",
    "Location": "Nairobi, Kenya",
    "Total_Sum_Insured": 25000000,
    "Currency": "USD",
    "Claims_History": [
        {"year": 2022, "amount": 150000, "type": "Machinery Breakdown"}
    ]
}

# -------------------- SAMPLE 2 --------------------
SAMPLE_EMAIL_TEXT_2 = """
From: broker@globalre.com
Subject: Facultative Placement - Hotel & Hospitality Risk

Dear Underwriter,

We are submitting the following risk for your facultative review:

Insured: OceanView Hotels Group
Location: Mombasa, Kenya
Sum Insured: KES 4,500,000,000
Period: 01/06/2024 - 31/05/2025
Perils: Fire, Flood, Terrorism
Claims (5yr): Two claims - 2021 Flood (KES 120,000,000), 2023 Fire (KES 80,000,000)

Regards,
Sarah Broker
"""

SAMPLE_EXCEL_DATA_2 = {
    "Insured_Name": "OceanView Hotels Group",
    "Industry": "Hospitality",
    "Location": "Mombasa, Kenya",
    "Total_Sum_Insured": 4500000000,
    "Currency": "KES",
    "Claims_History": [
        {"year": 2021, "amount": 120000000, "type": "Flood"},
        {"year": 2023, "amount": 80000000, "type": "Fire"}
    ]
}

# -------------------- HELPERS --------------------
def get_mock_email(sample: int = 1):
    if sample == 1:
        return SAMPLE_EMAIL_TEXT_1
    elif sample == 2:
        return SAMPLE_EMAIL_TEXT_2
    else:
        raise ValueError("Sample must be 1 or 2")

def get_mock_excel(sample: int = 1):
    if sample == 1:
        return json.dumps(SAMPLE_EXCEL_DATA_1, indent=2)
    elif sample == 2:
        return json.dumps(SAMPLE_EXCEL_DATA_2, indent=2)
    else:
        raise ValueError("Sample must be 1 or 2")
