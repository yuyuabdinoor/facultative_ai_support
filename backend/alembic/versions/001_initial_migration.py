"""Initial migration - create all tables

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum types if they don't exist
    op.execute("DO $$ BEGIN CREATE TYPE documenttypeenum AS ENUM ('pdf', 'scanned_pdf', 'email', 'excel'); EXCEPTION WHEN duplicate_object THEN null; END $$;")
    op.execute("DO $$ BEGIN CREATE TYPE risklevelenum AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'); EXCEPTION WHEN duplicate_object THEN null; END $$;")
    op.execute("DO $$ BEGIN CREATE TYPE decisiontypeenum AS ENUM ('APPROVE', 'REJECT', 'CONDITIONAL'); EXCEPTION WHEN duplicate_object THEN null; END $$;")
    op.execute("DO $$ BEGIN CREATE TYPE applicationstatusenum AS ENUM ('pending', 'processing', 'analyzed', 'completed', 'rejected'); EXCEPTION WHEN duplicate_object THEN null; END $$;")

    # Create applications table
    op.create_table('applications',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'PROCESSING', 'ANALYZED', 'COMPLETED', 'REJECTED', name='applicationstatusenum'), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_application_created_at', 'applications', ['created_at'])
    op.create_index('idx_application_status', 'applications', ['status'])

    # Create documents table
    op.create_table('documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('document_type', sa.Enum('PDF', 'SCANNED_PDF', 'EMAIL', 'EXCEL', name='documenttypeenum'), nullable=False),
        sa.Column('processed', sa.Boolean(), nullable=False),
        sa.Column('document_metadata', sa.JSON(), nullable=True),
        sa.Column('upload_timestamp', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['application_id'], ['applications.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_document_application_id', 'documents', ['application_id'])
    op.create_index('idx_document_processed', 'documents', ['processed'])
    op.create_index('idx_document_type', 'documents', ['document_type'])
    op.create_index('idx_document_upload_timestamp', 'documents', ['upload_timestamp'])

    # Create risk_parameters table
    op.create_table('risk_parameters',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('asset_value', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('coverage_limit', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('asset_type', sa.String(length=100), nullable=True),
        sa.Column('location', sa.Text(), nullable=True),
        sa.Column('industry_sector', sa.String(length=100), nullable=True),
        sa.Column('construction_type', sa.String(length=100), nullable=True),
        sa.Column('occupancy', sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(['application_id'], ['applications.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('application_id', name='uq_risk_parameters_application')
    )
    op.create_index('idx_risk_parameters_application_id', 'risk_parameters', ['application_id'])
    op.create_index('idx_risk_parameters_asset_type', 'risk_parameters', ['asset_type'])
    op.create_index('idx_risk_parameters_industry_sector', 'risk_parameters', ['industry_sector'])

    # Create financial_data table
    op.create_table('financial_data',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('revenue', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('assets', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('liabilities', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('credit_rating', sa.String(length=10), nullable=True),
        sa.Column('financial_strength_rating', sa.String(length=10), nullable=True),
        sa.ForeignKeyConstraint(['application_id'], ['applications.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('application_id', name='uq_financial_data_application')
    )
    op.create_index('idx_financial_data_application_id', 'financial_data', ['application_id'])
    op.create_index('idx_financial_data_credit_rating', 'financial_data', ['credit_rating'])

    # Create loss_events table
    op.create_table('loss_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_date', sa.DateTime(), nullable=False),
        sa.Column('amount', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('cause', sa.String(length=255), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['application_id'], ['applications.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_loss_event_amount', 'loss_events', ['amount'])
    op.create_index('idx_loss_event_application_id', 'loss_events', ['application_id'])
    op.create_index('idx_loss_event_date', 'loss_events', ['event_date'])

    # Create risk_analyses table
    op.create_table('risk_analyses',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('overall_score', sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column('risk_level', sa.Enum('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', name='risklevelenum'), nullable=False),
        sa.Column('factors', sa.JSON(), nullable=True),
        sa.Column('analysis_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['application_id'], ['applications.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('application_id', name='uq_risk_analysis_application')
    )
    op.create_index('idx_risk_analysis_application_id', 'risk_analyses', ['application_id'])
    op.create_index('idx_risk_analysis_created_at', 'risk_analyses', ['created_at'])
    op.create_index('idx_risk_analysis_level', 'risk_analyses', ['risk_level'])
    op.create_index('idx_risk_analysis_score', 'risk_analyses', ['overall_score'])

    # Create recommendations table
    op.create_table('recommendations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('decision', sa.Enum('APPROVE', 'REJECT', 'CONDITIONAL', name='decisiontypeenum'), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=True),
        sa.Column('conditions', sa.JSON(), nullable=True),
        sa.Column('premium_adjustment', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('coverage_modifications', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['application_id'], ['applications.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('application_id', name='uq_recommendation_application')
    )
    op.create_index('idx_recommendation_application_id', 'recommendations', ['application_id'])
    op.create_index('idx_recommendation_created_at', 'recommendations', ['created_at'])
    op.create_index('idx_recommendation_decision', 'recommendations', ['decision'])

    # Create business_limits table
    op.create_table('business_limits',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('limit_type', sa.String(length=100), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('max_amount', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('geographic_region', sa.String(length=100), nullable=True),
        sa.Column('industry_sector', sa.String(length=100), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_business_limit_active', 'business_limits', ['active'])
    op.create_index('idx_business_limit_category', 'business_limits', ['category'])
    op.create_index('idx_business_limit_region', 'business_limits', ['geographic_region'])
    op.create_index('idx_business_limit_sector', 'business_limits', ['industry_sector'])
    op.create_index('idx_business_limit_type', 'business_limits', ['limit_type'])

    # Create processing_logs table
    op.create_table('processing_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('application_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('process_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('processing_time', sa.Numeric(precision=10, scale=3), nullable=True),
        sa.Column('log_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['application_id'], ['applications.id'], ),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_processing_log_application_id', 'processing_logs', ['application_id'])
    op.create_index('idx_processing_log_created_at', 'processing_logs', ['created_at'])
    op.create_index('idx_processing_log_document_id', 'processing_logs', ['document_id'])
    op.create_index('idx_processing_log_status', 'processing_logs', ['status'])
    op.create_index('idx_processing_log_type', 'processing_logs', ['process_type'])

    # Create system_configurations table
    op.create_table('system_configurations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('key', sa.String(length=100), nullable=False),
        sa.Column('value', sa.JSON(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key')
    )
    op.create_index('idx_system_config_category', 'system_configurations', ['category'])
    op.create_index('idx_system_config_key', 'system_configurations', ['key'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('system_configurations')
    op.drop_table('processing_logs')
    op.drop_table('business_limits')
    op.drop_table('recommendations')
    op.drop_table('risk_analyses')
    op.drop_table('loss_events')
    op.drop_table('financial_data')
    op.drop_table('risk_parameters')
    op.drop_table('documents')
    op.drop_table('applications')
    
    # Drop enum types
    op.execute("DROP TYPE IF EXISTS applicationstatusenum")
    op.execute("DROP TYPE IF EXISTS decisiontypeenum")
    op.execute("DROP TYPE IF EXISTS risklevelenum")
    op.execute("DROP TYPE IF EXISTS documenttypeenum")