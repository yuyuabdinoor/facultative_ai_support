#!/usr/bin/env python3
"""
Test script to verify Excel-based report generation functionality
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / 'app'))

async def test_excel_report_generation():
    """Test the Excel-based report generation"""
    try:
        from app.services.report_generation_service import DocumentReportGenerator
        
        # Initialize the report generator
        generator = DocumentReportGenerator()
        
        # Path to sample Excel file
        excel_path = "/mnt/win3/work/kenyare/backend/static/sample_analysis_data.xlsx"
        
        # Output path for generated report
        output_path = "/mnt/win3/work/kenyare/backend/static/test_analysis_report.docx"
        
        print("Loading Excel data...")
        # Test loading Excel data
        excel_data = generator.load_excel_data(excel_path)
        
        print("Excel data loaded successfully!")
        print("Key fields extracted:")
        for key, value in list(excel_data.items())[:10]:  # Show first 10 fields
            print(f"  {key}: {value}")
        
        print("\nCalculated fields:")
        calculated_fields = [
            'premium_rate_percent', 'premium_rate_permille', 'loss_ratio_percent',
            'accepted_liability', 'accepted_premium', 'net_premium', 'risk_score',
            'liability_assessment', 'tsi_kes', 'premium_kes'
        ]
        
        for field in calculated_fields:
            if field in excel_data:
                print(f"  {field}: {excel_data[field]}")
        
        print(f"\nGenerating Word document report...")
        # Test report generation
        generated_path = await generator.generate_report_from_excel(
            excel_file_path=excel_path,
            output_path=output_path,
            application_id="TEST-001"
        )
        
        print(f"Report generated successfully: {generated_path}")
        
        # Verify file was created
        if os.path.exists(generated_path):
            file_size = os.path.getsize(generated_path)
            print(f"Generated file size: {file_size:,} bytes")
            print("✅ Excel-based report generation test PASSED!")
        else:
            print("❌ Generated file not found!")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_excel_report_generation())
