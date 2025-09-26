# Excel-Based Analysis Report Generation

This document describes how to use the Excel-based analysis report generation feature in the AI Facultative System.

## Overview

The system can now generate comprehensive analysis reports directly from Excel files containing analysis data. This feature implements all the calculation formulas from Appendix 1 and generates Word documents that match the Analysis - AMNS IDR.docx template format.

## Excel File Format

### Required Structure
- **Row 1**: Column headers (field names)
- **Row 2**: Corresponding values for each field

### Required Fields
The following fields are mandatory for proper report generation:

| Field Name | Description | Example |
|------------|-------------|---------|
| `Insured Name` | Name of the insured company | ABC Manufacturing Ltd |
| `TSI` | Total Sum Insured | 5000000 |
| `Premium` | Gross premium amount | 125000 |
| `Currency` | Currency code | USD |
| `Share Offered` | Percentage share offered | 25 |
| `Accepted Share` | Percentage share accepted | 20 |

### Optional Fields
These fields enhance the analysis but have defaults if not provided:

| Field Name | Description | Default |
|------------|-------------|---------|
| `Risk Location` | Physical location of risk | N/A |
| `Business Type` | Type of business/industry | N/A |
| `Cover Type` | Insurance cover type | N/A |
| `Period From` | Policy start date | N/A |
| `Period To` | Policy end date | N/A |
| `Brokerage` | Brokerage % or amount | 0 |
| `Paid Losses` | Historical paid losses | 0 |
| `Outstanding Reserves` | Outstanding reserves | 0 |
| `Recoveries` | Recoveries received | 0 |
| `ESG Rating` | ESG rating | N/A |

## Calculation Formulas

The system automatically calculates the following derived fields using formulas from Appendix 1:

### 1. Premium Rate Calculations
```
Rate % = (Premium ÷ TSI) × 100
Rate ‰ = (Premium ÷ TSI) × 1000
```

### 2. Premium Calculations (when rate is given)
```
Premium = TSI × (Rate % ÷ 100)
Premium = TSI × (Rate ‰ ÷ 1000)
```

### 3. Loss Ratio
```
Loss Ratio % = (Incurred Losses ÷ Earned Premium) × 100
Where: Incurred Losses = Paid Losses + Outstanding - Recoveries
```

### 4. Facultative Share Calculations
```
Accepted Premium = Gross Premium × Accepted Share %
Accepted Liability = TSI × Accepted Share %
```

### 5. Currency Conversion
All amounts are automatically converted to KES using current exchange rates.

## API Endpoints

### 1. Generate Report from Excel File Path
```http
POST /api/v1/reports/excel-report
Content-Type: application/json

{
    "excel_file_path": "/path/to/analysis_data.xlsx",
    "output_filename": "custom_report.docx",
    "application_id": "APP-001"
}
```

### 2. Upload Excel and Generate Report
```http
POST /api/v1/reports/excel-upload-report
Content-Type: multipart/form-data

file: [Excel file]
application_id: "APP-001" (optional)
```

### 3. Get Currency Rates
```http
GET /api/v1/reports/currency-rates
```

### 4. Update Currency Rates
```http
POST /api/v1/reports/update-currency-rates
Content-Type: application/json

{
    "USD": 150.0,
    "EUR": 165.0,
    "GBP": 190.0
}
```

## Usage Examples

### Python Usage
```python
from app.services.report_generation_service import DocumentReportGenerator

# Initialize generator
generator = DocumentReportGenerator()

# Load and process Excel data
excel_data = generator.load_excel_data("analysis_data.xlsx")

# Generate report
report_path = await generator.generate_report_from_excel(
    excel_file_path="analysis_data.xlsx",
    output_path="analysis_report.docx",
    application_id="APP-001"
)
```

### cURL Examples
```bash
# Upload Excel file and generate report
curl -X POST "http://localhost:8000/api/v1/reports/excel-upload-report" \
  -F "file=@analysis_data.xlsx" \
  -F "application_id=APP-001"

# Generate from existing file
curl -X POST "http://localhost:8000/api/v1/reports/excel-report" \
  -H "Content-Type: application/json" \
  -d '{
    "excel_file_path": "/app/uploads/analysis_data.xlsx",
    "output_filename": "report.docx"
  }'
```

## Sample Excel File

A sample Excel file (`sample_analysis_data.xlsx`) is provided in the `static` directory with:
- **Analysis Data** sheet: Contains sample data with all supported fields
- **Instructions** sheet: Field descriptions and requirements

## Generated Report Features

The generated Word document includes:
- **Kenya Re branding** with logo and company information
- **Main analysis table** with all risk details
- **Calculations section** showing formulas and computed values
- **Risk assessment** with automated scoring
- **Recommendations** based on data analysis
- **Signature sections** for approval workflow

## Risk Assessment Logic

The system automatically assesses risk based on:
- **Loss Ratio**: Historical claims experience
- **TSI Amount**: Liability level assessment
- **Premium Rate**: Rate adequacy evaluation
- **Combined Risk Score**: 1-10 scale with recommendations

### Risk Scoring
- **1-3**: Low risk → Recommend acceptance
- **4-6**: Moderate risk → Conditional acceptance
- **7-10**: High risk → Enhanced terms required

## Currency Support

Supported currencies with automatic KES conversion:
- USD (US Dollar)
- EUR (Euro)
- GBP (British Pound)
- KES (Kenyan Shilling)

Exchange rates can be updated via API or CSV file in `static/currency_rates.csv`.

## Error Handling

The system handles common errors gracefully:
- **Missing required fields**: Uses default values where possible
- **Invalid data types**: Automatic type conversion and validation
- **Currency conversion**: Falls back to default rates if conversion fails
- **Calculation errors**: Returns original data with warnings

## File Locations

- **Sample Excel**: `/backend/static/sample_analysis_data.xlsx`
- **Generated reports**: `/tmp/reports/` (configurable)
- **Currency rates**: `/app/static/currency_rates.csv`
- **Kenya Re logo**: `/app/static/kenya-re.png`

## Testing

Run the test script to verify functionality:
```bash
cd backend
python test_excel_report.py
```

This generates a test report from the sample Excel file and validates all calculations.
