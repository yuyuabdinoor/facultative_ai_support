"""
Reports API endpoints for generating document reports
"""

import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any
import tempfile
import os
from pathlib import Path

from ...core.database import get_db
from ...models.database import Application
from ...services.report_generation_service import document_report_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/applications/{application_id}/analysis-report")
async def generate_analysis_report(
    application_id: str,
    background_tasks: BackgroundTasks,
    include_attachments: bool = True,
    format: str = "docx",
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Generate an analysis report for an application in Word format
    
    Args:
        application_id: UUID of the application
        include_attachments: Whether to include attachment summaries
        format: Report format (currently only 'docx' supported)
        
    Returns:
        Report generation status and download info
    """
    try:
        # Validate format
        if format not in ["docx"]:
            raise HTTPException(status_code=400, detail="Only 'docx' format is currently supported")
        
        # Get application with all related data
        from sqlalchemy.orm import selectinload
        from sqlalchemy import select
        
        stmt = select(Application).options(
            selectinload(Application.documents),
            selectinload(Application.analysis_document_data),
            selectinload(Application.risk_analysis),
            selectinload(Application.recommendation),
            selectinload(Application.risk_parameters),
            selectinload(Application.financial_data)
        ).where(Application.id == application_id)
        
        result = await db.execute(stmt)
        application = result.scalar_one_or_none()
        
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        
        # Generate temporary file path
        temp_dir = tempfile.gettempdir()
        filename = f"analysis_report_{application_id}.docx"
        output_path = os.path.join(temp_dir, filename)
        
        # Generate the report
        report_path = await document_report_generator.generate_analysis_report(
            application=application,
            output_path=output_path,
            include_attachments=include_attachments
        )
        
        # Schedule cleanup of temporary file after response
        background_tasks.add_task(_cleanup_temp_file, report_path)
        
        return {
            "message": "Analysis report generated successfully",
            "application_id": application_id,
            "filename": filename,
            "download_url": f"/api/v1/reports/download/{filename}",
            "format": format,
            "include_attachments": include_attachments,
            "generated_at": application.updated_at.isoformat() if application.updated_at else None,
            "file_path": report_path  # For immediate download
        }
        
    except Exception as e:
        logger.error(f"Error generating analysis report for application {application_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.get("/download/{filename}")
async def download_report(
    filename: str,
) -> FileResponse:
    """
    Download a generated report file
    
    Args:
        filename: Name of the file to download
        
    Returns:
        File response with the report
    """
    try:
        # Security check - only allow specific patterns
        if not filename.endswith('.docx') or '..' in filename or '/' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Report file not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading report {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download report")


@router.post("/applications/{application_id}/quick-summary")
async def generate_quick_summary(
    application_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Generate a quick summary report (JSON format) for an application
    
    Args:
        application_id: UUID of the application
        
    Returns:
        Quick summary data
    """
    try:
        from sqlalchemy.orm import selectinload
        from sqlalchemy import select
        
        stmt = select(Application).options(
            selectinload(Application.documents),
            selectinload(Application.analysis_document_data),
            selectinload(Application.risk_analysis),
            selectinload(Application.recommendation)
        ).where(Application.id == application_id)
        
        result = await db.execute(stmt)
        application = result.scalar_one_or_none()
        
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        
        # Generate quick summary
        summary = _generate_quick_summary(application)
        
        return {
            "application_id": application_id,
            "summary": summary,
            "generated_at": application.updated_at.isoformat() if application.updated_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating quick summary for application {application_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@router.get("/applications/{application_id}/export-data")
async def export_application_data(
    application_id: str,
    format: str = "json",
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Export application data in various formats
    
    Args:
        application_id: UUID of the application
        format: Export format ('json' or 'excel')
        
    Returns:
        Exported data or download information
    """
    try:
        if format not in ["json", "excel"]:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'excel'")
        
        from sqlalchemy.orm import selectinload
        from sqlalchemy import select
        
        stmt = select(Application).options(
            selectinload(Application.documents),
            selectinload(Application.analysis_document_data),
            selectinload(Application.risk_analysis),
            selectinload(Application.recommendation),
            selectinload(Application.risk_parameters),
            selectinload(Application.financial_data)
        ).where(Application.id == application_id)
        
        result = await db.execute(stmt)
        application = result.scalar_one_or_none()
        
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        
        if format == "json":
            # Return JSON data directly
            return _serialize_application_data(application)
        
        elif format == "excel":
            # Generate Excel file (placeholder - would use existing excel generation)
            temp_dir = tempfile.gettempdir()
            filename = f"application_data_{application_id}.xlsx"
            output_path = os.path.join(temp_dir, filename)
            
            # Use existing Excel generation from data extraction agent
            from ...agents.data_extraction_agent import DataExtractionAgent
            extraction_agent = DataExtractionAgent()
            
            # Convert application data to ExtractedData format
            extracted_data = _convert_to_extracted_data(application)
            excel_path = extraction_agent.generate_excel_report([extracted_data], output_path)
            
            return {
                "message": "Excel export generated successfully",
                "application_id": application_id,
                "filename": filename,
                "download_url": f"/api/v1/reports/download-excel/{filename}",
                "format": format
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting application data {application_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")


@router.get("/download-excel/{filename}")
async def download_excel_export(
    filename: str,
) -> FileResponse:
    """Download Excel export file"""
    try:
        if not filename.endswith('.xlsx') or '..' in filename or '/' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Excel file not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading Excel export {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download Excel export")


def _cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary file {file_path}: {str(e)}")


def _generate_quick_summary(application: Application) -> Dict[str, Any]:
    """Generate a quick summary of the application"""
    summary = {
        "status": application.status.value,
        "created_at": application.created_at.isoformat() if application.created_at else None,
        "updated_at": application.updated_at.isoformat() if application.updated_at else None,
        "documents_count": len(application.documents) if application.documents else 0,
        "processed_documents": sum(1 for doc in (application.documents or []) if doc.processed)
    }
    
    # Add analysis data summary
    if application.analysis_document_data:
        data = application.analysis_document_data
        summary["analysis_data"] = {
            "insured_name": data.insured_name,
            "total_sum_insured": float(data.total_sums_insured) if data.total_sums_insured else None,
            "currency": data.currency,
            "perils_covered": data.perils_covered,
            "share_offered_percentage": float(data.share_offered_percentage) if data.share_offered_percentage else None
        }
    
    # Add risk analysis summary
    if application.risk_analysis:
        risk = application.risk_analysis
        summary["risk_analysis"] = {
            "risk_score": float(risk.risk_score),
            "risk_level": risk.risk_level.value,
            "confidence": float(risk.confidence)
        }
    
    # Add recommendation summary
    if application.recommendation:
        rec = application.recommendation
        summary["recommendation"] = {
            "decision": rec.decision.value,
            "confidence": float(rec.confidence),
            "rationale": rec.rationale[:200] + "..." if len(rec.rationale) > 200 else rec.rationale
        }
    
    return summary


def _serialize_application_data(application: Application) -> Dict[str, Any]:
    """Serialize application data to JSON-compatible format"""
    data = {
        "id": str(application.id),
        "status": application.status.value,
        "created_at": application.created_at.isoformat() if application.created_at else None,
        "updated_at": application.updated_at.isoformat() if application.updated_at else None
    }
    
    # Documents
    if application.documents:
        data["documents"] = [
            {
                "id": str(doc.id),
                "filename": doc.filename,
                "document_type": doc.document_type.value,
                "processed": doc.processed,
                "upload_timestamp": doc.upload_timestamp.isoformat(),
                "metadata": doc.document_metadata
            }
            for doc in application.documents
        ]
    
    # Analysis data
    if application.analysis_document_data:
        analysis_data = application.analysis_document_data
        data["analysis_document_data"] = {
            field.name: getattr(analysis_data, field.name)
            for field in analysis_data.__table__.columns
            if hasattr(analysis_data, field.name)
        }
        # Convert Decimal to float for JSON serialization
        for key, value in data["analysis_document_data"].items():
            if hasattr(value, '__float__'):
                data["analysis_document_data"][key] = float(value)
    
    # Risk analysis
    if application.risk_analysis:
        risk = application.risk_analysis
        data["risk_analysis"] = {
            "risk_score": float(risk.risk_score),
            "risk_level": risk.risk_level.value,
            "confidence": float(risk.confidence),
            "factors": risk.factors,
            "recommendations": risk.recommendations,
            "analysis_details": risk.analysis_details
        }
    
    # Recommendation
    if application.recommendation:
        rec = application.recommendation
        data["recommendation"] = {
            "decision": rec.decision.value,
            "rationale": rec.rationale,
            "confidence": float(rec.confidence),
            "conditions": rec.conditions,
            "premium_adjustment": float(rec.premium_adjustment) if rec.premium_adjustment else None,
            "decision_factors": rec.decision_factors
        }
    
    return data


def _convert_to_extracted_data(application: Application):
    """Convert application data to ExtractedData format for Excel generation"""
    # This would need to be implemented based on your existing ExtractedData structure
    # For now, return a placeholder
    from ...agents.data_extraction_agent import ExtractedData, AnalysisDocumentData as ExtractedAnalysisData
    
    # Create ExtractedData object from application data
    extracted = ExtractedData()
    
    if application.analysis_document_data:
        # Map database AnalysisDocumentData to extraction AnalysisDocumentData
        extracted.analysis_data = ExtractedAnalysisData()
        # Copy fields from database to extraction format
        for field in ['reference_number', 'insured_name', 'cedant_reinsured', 'broker_name', 
                     'perils_covered', 'total_sums_insured', 'currency', 'period_of_insurance',
                     'pml_percentage', 'share_offered_percentage']:
            if hasattr(application.analysis_document_data, field):
                setattr(extracted.analysis_data, field, getattr(application.analysis_document_data, field))
    
    return extracted


@router.post("/excel-report")
async def generate_report_from_excel(
    excel_file_path: str,
    output_filename: Optional[str] = None,
    application_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate analysis report directly from Excel file data
    
    Args:
        excel_file_path: Path to the Excel file containing analysis data
        output_filename: Optional custom filename for the generated report
        application_id: Optional application ID for tracking
        
    Returns:
        Report generation status and download info
    """
    try:
        # Validate Excel file exists
        if not os.path.exists(excel_file_path):
            raise HTTPException(status_code=404, detail="Excel file not found")
        
        # Generate output filename if not provided
        if not output_filename:
            output_filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        
        # Create temporary output path
        output_dir = Path(tempfile.gettempdir()) / "reports"
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / output_filename)
        
        # Generate report from Excel
        generated_path = await document_report_generator.generate_report_from_excel(
            excel_file_path=excel_file_path,
            output_path=output_path,
            application_id=application_id
        )
        
        return {
            "status": "success",
            "message": "Analysis report generated successfully from Excel data",
            "report_path": generated_path,
            "download_url": f"/api/v1/reports/download?file_path={generated_path}",
            "filename": output_filename,
            "format": "docx"
        }
        
    except FileNotFoundError as e:
        logger.error(f"Excel file not found: {str(e)}")
        raise HTTPException(status_code=404, detail="Excel file not found")
    except ValueError as e:
        logger.error(f"Invalid Excel data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid Excel data: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating report from Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.post("/excel-upload-report")
async def upload_and_generate_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    application_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload Excel file and generate analysis report
    
    Args:
        file: Excel file with analysis data
        application_id: Optional application ID for tracking
        
    Returns:
        Report generation status and download info
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported")
        
        # Create temporary directory for uploaded file
        upload_dir = Path(tempfile.gettempdir()) / "uploads"
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        upload_path = upload_dir / file.filename
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Generate output filename
        output_filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        output_dir = Path(tempfile.gettempdir()) / "reports"
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / output_filename)
        
        # Generate report from uploaded Excel
        generated_path = await document_report_generator.generate_report_from_excel(
            excel_file_path=str(upload_path),
            output_path=output_path,
            application_id=application_id
        )
        
        # Clean up uploaded file in background
        background_tasks.add_task(os.unlink, upload_path)
        
        return {
            "status": "success",
            "message": "Analysis report generated successfully from uploaded Excel file",
            "report_path": generated_path,
            "download_url": f"/api/v1/reports/download?file_path={generated_path}",
            "filename": output_filename,
            "format": "docx"
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded Excel file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process Excel file: {str(e)}")


@router.get("/currency-rates")
async def get_currency_rates() -> Dict[str, float]:
    """
    Get current currency conversion rates
    
    Returns:
        Dictionary of currency codes and their KES conversion rates
    """
    try:
        return document_report_generator.currency_rates
    except Exception as e:
        logger.error(f"Error retrieving currency rates: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve currency rates")


@router.post("/update-currency-rates")
async def update_currency_rates(rates: Dict[str, float]) -> Dict[str, Any]:
    """
    Update currency conversion rates
    
    Args:
        rates: Dictionary of currency codes and their KES conversion rates
        
    Returns:
        Update status
    """
    try:
        # Update in-memory rates
        document_report_generator.currency_rates.update(rates)
        
        # Optionally save to CSV file
        import pandas as pd
        rates_df = pd.DataFrame(list(rates.items()), columns=['currency', 'kes_value'])
        rates_path = Path("/app/static/currency_rates.csv")
        rates_path.parent.mkdir(exist_ok=True)
        rates_df.to_csv(rates_path, index=False)
        
        return {
            "status": "success",
            "message": f"Updated {len(rates)} currency rates",
            "updated_rates": rates
        }
        
    except Exception as e:
        logger.error(f"Error updating currency rates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update currency rates: {str(e)}")


@router.post("/refresh-currency-rates")
async def refresh_currency_rates(api_key: Optional[str] = None, base: Optional[str] = None) -> Dict[str, Any]:
    """
    Refresh currency rates from ExchangeRate-API into KES per unit.

    Args:
        api_key: Optional API key; if omitted, EXCHANGE_RATE_API_KEY env var is used
        base: Optional base code (default from EXCHANGE_RATE_BASE or 'USD')

    Returns:
        Updated rates mapping
    """
    try:
        updated = document_report_generator.refresh_rates(api_key=api_key, base=base)
        return {"status": "success", "updated_rates": updated}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error refreshing currency rates: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to refresh currency rates")
