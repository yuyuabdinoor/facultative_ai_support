"""Add analysis document data table

Revision ID: 002
Revises: 001
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    """Add analysis_document_data table with all 23 critical fields"""
    op.create_table(
        'analysis_document_data',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=sa.text('gen_random_uuid()')),
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        
        # Basic Information (Fields 1-5)
        sa.Column('reference_number', sa.String(length=50), nullable=True),
        sa.Column('date_received', sa.DateTime(), nullable=True),
        sa.Column('insured_name', sa.String(length=200), nullable=True),
        sa.Column('cedant_reinsured', sa.String(length=200), nullable=True),
        sa.Column('broker_name', sa.String(length=200), nullable=True),
        
        # Coverage Details (Fields 6-10)
        sa.Column('perils_covered', sa.String(length=500), nullable=True),
        sa.Column('geographical_limit', sa.String(length=300), nullable=True),
        sa.Column('situation_of_risk', sa.String(length=500), nullable=True),
        sa.Column('occupation_of_insured', sa.String(length=200), nullable=True),
        sa.Column('main_activities', sa.String(length=500), nullable=True),
        
        # Financial Information (Fields 11-15)
        sa.Column('total_sums_insured', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('currency', sa.String(length=3), nullable=True),
        sa.Column('excess_retention', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('premium_rates', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('period_of_insurance', sa.String(length=100), nullable=True),
        
        # Risk Assessment (Fields 16-20)
        sa.Column('pml_percentage', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('cat_exposure', sa.String(length=300), nullable=True),
        sa.Column('reinsurance_deductions', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('claims_experience_3_years', sa.Text(), nullable=True),
        sa.Column('share_offered_percentage', sa.Numeric(precision=5, scale=2), nullable=True),
        
        # Additional Information (Fields 21-23)
        sa.Column('surveyors_report', sa.String(length=500), nullable=True),
        sa.Column('climate_change_risk', sa.String(length=500), nullable=True),
        sa.Column('esg_risk_assessment', sa.String(length=500), nullable=True),
        
        # Metadata and Processing Information
        sa.Column('confidence_score', sa.Numeric(precision=3, scale=2), nullable=False, default=0.0),
        sa.Column('field_confidence_scores', sa.JSON(), nullable=True),
        sa.Column('data_completeness', sa.JSON(), nullable=True),
        sa.Column('processing_notes', sa.JSON(), nullable=True),
        sa.Column('extraction_method', sa.String(length=50), nullable=False, default='hybrid'),
        sa.Column('source_documents', sa.JSON(), nullable=True),
        
        sa.Column('created_at', sa.DateTime(), nullable=False, default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, default=sa.func.now(), onupdate=sa.func.now()),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['application_id'], ['applications.id'], ),
        sa.UniqueConstraint('application_id', name='uq_analysis_document_data_application')
    )
    
    # Create indexes
    op.create_index('idx_analysis_document_data_application_id', 'analysis_document_data', ['application_id'])
    op.create_index('idx_analysis_document_data_reference_number', 'analysis_document_data', ['reference_number'])
    op.create_index('idx_analysis_document_data_insured_name', 'analysis_document_data', ['insured_name'])
    op.create_index('idx_analysis_document_data_cedant_reinsured', 'analysis_document_data', ['cedant_reinsured'])
    op.create_index('idx_analysis_document_data_broker_name', 'analysis_document_data', ['broker_name'])
    op.create_index('idx_analysis_document_data_currency', 'analysis_document_data', ['currency'])
    op.create_index('idx_analysis_document_data_confidence_score', 'analysis_document_data', ['confidence_score'])
    op.create_index('idx_analysis_document_data_created_at', 'analysis_document_data', ['created_at'])


def downgrade():
    """Remove analysis_document_data table"""
    op.drop_index('idx_analysis_document_data_created_at', table_name='analysis_document_data')
    op.drop_index('idx_analysis_document_data_confidence_score', table_name='analysis_document_data')
    op.drop_index('idx_analysis_document_data_currency', table_name='analysis_document_data')
    op.drop_index('idx_analysis_document_data_broker_name', table_name='analysis_document_data')
    op.drop_index('idx_analysis_document_data_cedant_reinsured', table_name='analysis_document_data')
    op.drop_index('idx_analysis_document_data_insured_name', table_name='analysis_document_data')
    op.drop_index('idx_analysis_document_data_reference_number', table_name='analysis_document_data')
    op.drop_index('idx_analysis_document_data_application_id', table_name='analysis_document_data')
    op.drop_table('analysis_document_data')