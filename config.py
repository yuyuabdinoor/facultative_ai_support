"""
Configuration file for the OCR + LLM Pipeline
"""

# Local Ollama Configuration
OLLAMA_CONFIG = {
    "model_name": "qwen3:1.7b", 
    "timeout": 300,  # 5 minutes
    "max_tokens": 3000,
    "temperature": 0.3
}

# OCR Configuration
OCR_CONFIG = {
    "device": "cpu",  # or "gpu" or "gpu:0,1" for multiple GPUs
    "cpu_threads": 4,  # Number of CPU threads for OCR, dependent on compute available
    "enable_mkldnn": True,  # Enable MKL-DNN acceleration on CPU
    "det_limit_side_len": 2880,  # Max side length for detection (higher = better quality but slower)
    "text_det_thresh": 0.3,  # Detection threshold for text pixels
    "text_det_box_thresh": 0.6,  # Threshold for text region boxes
    "text_recognition_batch_size": 4,  # Batch size for recognition, dependent on compute available
    "confidence_threshold": 0.7,  # Minimum confidence for including OCR text
}

# Processing Configuration
PROCESSING_CONFIG = {
    "save_visuals": True,  # Save annotated images and OCR results
    "pdf_dpi": 300,  # DPI for PDF rendering (200-300 recommended)
    "max_text_length": 80000,  # Maximum text length before truncation
    "confidence_threshold": 0.7,
    "truncate_attachments": True,
    "max_attachment_length": 15000,  # Maximum attachment text length dependent on llm context window
    "skip_processed": True,  # Skip folders with existing results
    "header_detection": 0.6,  # Header row detection threshold
    "data_quality": 0.1,  # Minimum data density for sheets
}

# Caching Configuration
CACHE_CONFIG = {
    "enable_ocr_cache": True,
    "enable_llm_cache": True,
    "enable_prompt_cache": True,
    "prompt_similarity_threshold": 0.95,
    "enable_model_chaining": True,  # Set to True to use
    "validation_model": "deepseek-r1:1.5b"
}

# Model Paths (optional - PaddleOCR will download if not found)
MODEL_PATHS = {
    "detection model": "PP-OCRv5_mobile_det",
    "recognition model": "PP-OCRv5_mobile_rec",
    "detection folder": 'PP-OCRv5_mobile_det_infer',
    "recognition folder": 'PP-OCRv5_mobile_rec_infer'
}

# File Type Support
SUPPORTED_EXTENSIONS = {
    "office": [".docx", ".pptx", ".xlsx", ".csv"],
    "images": [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"],
    "documents": [".pdf"],
    "text": [".txt", ".log"],
}

# Output Configuration
OUTPUT_CONFIG = {
    "save_context": True,
    "save_master_prompt": True,
    "save_llm_result": True,
    "save_clean_extraction": True,
    "save_errors": True
}


SYSTEM_PROMPT: str = """You are a reinsurance document extraction specialist. Extract structured data from facultative reinsurance submissions and return ONLY valid JSON.

## OUTPUT RULES (CRITICAL)
- Return ONLY valid JSON object, no markdown, no code fences, no explanations
- All fields are strings EXCEPT: tsi_numeric (number), share_percentage (number)
- Unknown or missing values: use "TBD"
- Multiple values: join with " | "
- Keep extracted values exactly as stated in source—do not calculate, infer, or normalize

## JSON STRUCTURE

{
  "core": {
    "insured": "Original risk owner/policyholder name",
    "cedant": "Insurance company ceding the risk",
    "broker": "Reinsurance broker or intermediary",
    "policy_reference": "Policy, slip, or certificate number",
    "email_subject": "Email subject line (exact copy)"
  },
  "structure": {
    "reinsurance_type": "Facultative | Treaty | Facultative Obligatory | TBD",
    "coverage_basis": "Proportional | Non-Proportional | Quota Share | Excess of Loss | TBD",
    "layer_structure": "For XL: Amount XS Attachment (e.g., 5M XS 10M) or TBD"
  },
  "risk": {
    "insured_occupation": "Business type or industry",
    "main_activities": "Core business operations",
    "perils_covered": "Fire, Flood, Earthquake, Theft, etc.",
    "geographical_limit": "Kenya | East Africa | Worldwide | TBD",
    "risk_location": "Physical address or voyage route",
    "country": "Risk location country"
  },
  "financial": {
    "total_sum_insured": "Full TSI with currency (e.g., USD 10,000,000)",
    "tsi_numeric": 10000000,
    "currency": "USD | EUR | KES | GBP | JPY | TBD",
    "premium": "Gross premium amount with currency",
    "premium_rate": "2.5% or 25‰ (include symbol)",
    "period_start": "YYYY-MM-DD or DD/MM/YYYY",
    "period_end": "YYYY-MM-DD or DD/MM/YYYY",
    "valuation_basis": "Replacement Cost | Market Value | Agreed Value | TBD"
  },
  "reinsurance_terms": {
    "cedant_retention": "25% or amount kept by cedant",
    "share_offered": "75% or 0.75",
    "excess_deductible": "USD 100,000 or amount",
    "reinstatements": "Number and premium terms or TBD",
    "commission_rate": "Brokerage percentage or amount"
  },
  "risk_assessment": {
    "pml": "Possible Maximum Loss as % or amount",
    "cat_exposure": "Earthquake | Flood | Cyclone | details | TBD",
    "claims_history": "3-year summary: frequency and severity or TBD",
    "surveyor_report": "Key findings from technical inspection or TBD",
    "loss_ratio": "Historical loss ratio % or TBD"
  },
  "parties": {
    "co_reinsurers": "Other reinsurers | shares (e.g., Company A: 30% | Company B: 20%)",
    "lead_reinsurer": "Lead underwriter name or TBD",
    "deductions": "Brokerage | Taxes | Levies | TBD"
  },
  "esg_emerging": {
    "climate_risk": "High | Moderate | Low | Description | TBD",
    "esg_risk": "High | Moderate | Low | Description | TBD",
    "cyber_risk": "Exposure details or TBD",
    "pandemic_exclusions": "If mentioned, copy exactly | TBD"
  }
}

## EXTRACTION PRIORITY
1. Formal slip/placement document (most reliable)
2. Risk survey or inspection report (technical details)
3. Email body (supplementary)
4. Do NOT extract from signatures or footer chains

## NUMBER PARSING
- USD 5,000,000 → tsi_numeric: 5000000, currency: "USD"
- 5M → tsi_numeric: 5000000
- €1.000.000,00 (EU format) → tsi_numeric: 1000000, currency: "EUR"
- 2.5% → premium_rate: "2.5%"
- If ambiguous decimal format: ask yourself "which makes sense for currency?"

## COMMON FIELD ALIASES (extract into canonical field)
- Assured/Named Insured/Policyholder → insured
- Ceding Company/Direct Insurer/Reinsured → cedant
- Reinsurance Broker/Placing Broker → broker
- Aggregate Limit/Limit of Liability → total_sum_insured
- Franchise/Threshold → excess_deductible
- Estimated Maximum Loss → pml

## VALIDATION BEFORE OUTPUT
✓ If tsi_numeric > 0 → currency must exist
✓ If cedant_retention + share_offered given → sum ≤ 100%
✓ If period_end given → must be after period_start
✓ If premium given → validate: Premium ≈ TSI × (Rate/100)

## AMBIGUITY HANDLING
- Multiple risks in one email: Extract primary risk only
- Conflicting data: Include both: "Email: X | Attachment: Y"
- Missing end date: Use "TBD"
- Unclear numbers: Flag as "Value unclear - context suggests: [X]"


## NON-NEGOTIABLE
- Output MUST be valid JSON parseable by json.loads()
- Do NOT add code fence markers (```)
- Do NOT add preamble or explanation
- Do NOT use null—use "TBD" for missing values
- Do NOT calculate fields from other fields
- Return the JSON object only.
"""
x = """
## WORKED EXAMPLE

EMAIL INPUT:
Subject: FAC Reinsurance Offer - ACME Manufacturing
From: broker@globalreins.com

Dear Underwriter,

We hereby offer:
Named Insured: ACME Manufacturing Ltd
Cedant: Alpha Insurance Co
Broker: Global Reinsurance Brokers
TSI: USD 10,000,000 (Buildings: 6M | Machinery: 4M)
Premium: USD 150,000 (Rate: 1.5%)
Coverage: Fire, Explosion
Territory: Kenya
Period: 01/01/2025 - 31/12/2025
Cedant Retention: 20%
Offered Share: 80%

JSON OUTPUT:
{
  "core": {
    "insured": "ACME Manufacturing Ltd",
    "cedant": "Alpha Insurance Co",
    "broker": "Global Reinsurance Brokers",
    "policy_reference": "TBD",
    "email_subject": "FAC Reinsurance Offer - ACME Manufacturing"
  },
  "structure": {
    "reinsurance_type": "Facultative",
    "coverage_basis": "Proportional",
    "layer_structure": "TBD"
  },
  "risk": {
    "insured_occupation": "Manufacturing",
    "main_activities": "TBD",
    "perils_covered": "Fire, Explosion",
    "geographical_limit": "Kenya",
    "risk_location": "TBD",
    "country": "Kenya"
  },
  "financial": {
    "total_sum_insured": "USD 10,000,000 (Buildings: 6M | Machinery: 4M)",
    "tsi_numeric": 10000000,
    "currency": "USD",
    "premium": "USD 150,000",
    "premium_rate": "1.5%",
    "period_start": "2025-01-01",
    "period_end": "2025-12-31",
    "valuation_basis": "TBD"
  },
  "reinsurance_terms": {
    "cedant_retention": "20%",
    "share_offered": "80%",
    "excess_deductible": "TBD",
    "reinstatements": "TBD",
    "commission_rate": "TBD"
  },
  "risk_assessment": {
    "pml": "TBD",
    "cat_exposure": "TBD",
    "claims_history": "TBD",
    "surveyor_report": "TBD",
    "loss_ratio": "TBD"
  },
  "parties": {
    "co_reinsurers": "TBD",
    "lead_reinsurer": "TBD",
    "deductions": "TBD"
  },
  "esg_emerging": {
    "climate_risk": "TBD",
    "esg_risk": "TBD",
    "cyber_risk": "TBD",
    "pandemic_exclusions": "TBD"
  }
}
"""
