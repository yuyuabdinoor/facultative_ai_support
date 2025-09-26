#!/usr/bin/env python3
"""
Script to create a sample Excel file demonstrating the expected format
for the AI Facultative System analysis data input.
"""

import pandas as pd
from pathlib import Path

def create_sample_excel():
    """Create sample Excel file with proper column headers and sample data"""
    
    # Define the expected columns and sample data
    data = {
        # Basic Information
        'Insured Name': ['ABC Manufacturing Ltd'],
        'Risk Location': ['Industrial Area, Nairobi, Kenya'],
        'Business Type': ['Manufacturing - Food Processing'],
        'Cover Type': ['Property All Risks'],
        'Period From': ['2025-01-01'],
        'Period To': ['2025-12-31'],
        
        # Financial Information
        'Currency': ['USD'],
        'TSI': [5000000],  # Total Sum Insured
        'Premium': [125000],  # Gross Premium
        'Brokerage': [15],  # Brokerage percentage
        'Taxes': [5000],  # Taxes amount
        'Levies': [2000],  # Levies amount
        
        # Risk Information
        'Excess': ['USD 50,000 Each and Every Loss'],
        'Deductible': ['USD 50,000'],
        'PML': ['60% - USD 3,000,000'],
        'Probable Maximum Loss': ['USD 3,000,000'],
        
        # Share Information
        'Share Offered': [25],  # Percentage
        'Accepted Share': [20],  # Percentage
        'Share Accepted': [20],  # Alternative column name
        
        # Loss Data (for loss ratio calculations)
        'Paid Losses': [75000],  # Historical paid losses
        'Outstanding Reserves': [25000],  # Outstanding reserves
        'Outstanding': [25000],  # Alternative column name
        'Recoveries': [10000],  # Recoveries received
        'Earned Premium': [120000],  # Earned premium for loss ratio
        
        # Additional Fields
        'ESG Rating': ['B+'],
        'ESG': ['B+'],  # Alternative column name
        'Broker Name': ['XYZ Insurance Brokers'],
        'Cedant': ['Local Insurance Company Ltd'],
        'Reference Number': ['FAC/2025/001'],
        'Perils Covered': ['Fire, Lightning, Explosion, Aircraft Impact, Riot & Strike'],
        
        # Optional calculated fields (will be computed if not provided)
        'Premium Rate Percent': [2.5],  # Will be calculated if not provided
        'Rate Percent': [2.5],  # Alternative column name
        'Premium Rate Permille': [25.0],  # Will be calculated if not provided
        'Rate Permille': [25.0],  # Alternative column name
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create output path
    output_path = Path(__file__).parent / 'static' / 'sample_analysis_data.xlsx'
    output_path.parent.mkdir(exist_ok=True)
    
    # Save to Excel with proper formatting
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write the main data sheet
        df.to_excel(writer, sheet_name='Analysis Data', index=False)
        
        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Analysis Data']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Create instructions sheet
        instructions_data = {
            'Column Name': [
                'Insured Name', 'Risk Location', 'Business Type', 'Cover Type',
                'Period From', 'Period To', 'Currency', 'TSI', 'Premium',
                'Brokerage', 'Taxes', 'Levies', 'Excess', 'PML',
                'Share Offered', 'Accepted Share', 'Paid Losses',
                'Outstanding Reserves', 'Recoveries', 'Earned Premium',
                'ESG Rating', 'Broker Name', 'Cedant', 'Reference Number'
            ],
            'Description': [
                'Name of the insured company/entity',
                'Physical location of the risk',
                'Type of business/industry',
                'Type of insurance cover',
                'Policy start date (YYYY-MM-DD)',
                'Policy end date (YYYY-MM-DD)',
                'Currency code (USD, EUR, GBP, KES, etc.)',
                'Total Sum Insured amount',
                'Gross premium amount',
                'Brokerage percentage or amount',
                'Taxes amount',
                'Levies amount',
                'Excess/Deductible amount and terms',
                'Probable Maximum Loss',
                'Percentage share offered',
                'Percentage share accepted',
                'Historical paid losses (3 years)',
                'Outstanding reserves',
                'Recoveries received',
                'Earned premium for loss ratio calculation',
                'ESG rating (if available)',
                'Broker company name',
                'Cedant/Ceding company name',
                'Reference/policy number'
            ],
            'Required': [
                'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes',
                'No', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No',
                'No', 'No', 'No', 'No'
            ]
        }
        
        instructions_df = pd.DataFrame(instructions_data)
        instructions_df.to_excel(writer, sheet_name='Instructions', index=False)
        
        # Format instructions sheet
        inst_worksheet = writer.sheets['Instructions']
        for column in inst_worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 60)
            inst_worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"Sample Excel file created: {output_path}")
    return output_path

if __name__ == "__main__":
    create_sample_excel()
