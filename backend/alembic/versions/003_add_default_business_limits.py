"""Add default business limits

Revision ID: 003
Revises: 002
Create Date: 2024-01-01 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade():
    """Add default business limits configuration"""
    
    # Create a connection to execute raw SQL
    connection = op.get_bind()
    
    # Default business limits data
    default_limits = [
        # Asset Value Limits
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'asset_value',
            'category': 'standard',
            'max_amount': 100000000.00,  # $100M
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'asset_value',
            'category': 'high_value',
            'max_amount': 500000000.00,  # $500M
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        
        # Coverage Limits
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'coverage_limit',
            'category': 'standard',
            'max_amount': 50000000.00,  # $50M
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        
        # Geographic Exposure Limits
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'geographic_exposure',
            'category': 'regional',
            'max_amount': 200000000.00,  # $200M per region
            'geographic_region': 'north_america',
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'geographic_exposure',
            'category': 'regional',
            'max_amount': 150000000.00,  # $150M per region
            'geographic_region': 'europe',
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'geographic_exposure',
            'category': 'regional',
            'max_amount': 100000000.00,  # $100M per region
            'geographic_region': 'asia_pacific',
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        
        # Industry Sector Limits
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'industry_sector',
            'category': 'restricted',
            'max_amount': 0.00,  # Completely restricted
            'geographic_region': None,
            'industry_sector': 'nuclear',
            'active': True,
            'created_at': datetime.utcnow()
        },
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'industry_sector',
            'category': 'high_risk',
            'max_amount': 25000000.00,  # $25M limit for high-risk sectors
            'geographic_region': None,
            'industry_sector': 'energy',
            'active': True,
            'created_at': datetime.utcnow()
        },
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'industry_sector',
            'category': 'standard',
            'max_amount': 75000000.00,  # $75M for standard sectors
            'geographic_region': None,
            'industry_sector': 'manufacturing',
            'active': True,
            'created_at': datetime.utcnow()
        },
        
        # Construction Type Limits
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'construction_type',
            'category': 'wood_frame',
            'max_amount': 10000000.00,  # $10M for wood frame
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'construction_type',
            'category': 'steel_concrete',
            'max_amount': 100000000.00,  # $100M for steel/concrete
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        
        # Occupancy Type Limits
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'occupancy_type',
            'category': 'hazardous',
            'max_amount': 5000000.00,  # $5M for hazardous occupancies
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        
        # Financial Strength Limits
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'financial_strength',
            'category': 'credit_rating',
            'max_amount': None,
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'financial_strength',
            'category': 'minimum_assets',
            'max_amount': 10000000.00,  # $10M minimum assets
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        
        # Regulatory Compliance
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'regulatory_compliance',
            'category': 'sanctions_check',
            'max_amount': None,
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'regulatory_compliance',
            'category': 'capital_requirements',
            'max_amount': None,
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        
        # Catastrophe Exposure Limits
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'catastrophe_exposure',
            'category': 'high_cat_risk',
            'max_amount': 25000000.00,  # $25M in high cat risk areas
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'catastrophe_exposure',
            'category': 'medium_cat_risk',
            'max_amount': 50000000.00,  # $50M in medium cat risk areas
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'catastrophe_exposure',
            'category': 'low_cat_risk',
            'max_amount': 100000000.00,  # $100M in low cat risk areas
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        },
        
        # Aggregate Exposure Limits
        {
            'id': str(uuid.uuid4()),
            'limit_type': 'aggregate_exposure',
            'category': 'total_portfolio',
            'max_amount': 1000000000.00,  # $1B total portfolio limit
            'geographic_region': None,
            'industry_sector': None,
            'active': True,
            'created_at': datetime.utcnow()
        }
    ]
    
    # Insert default limits
    for limit in default_limits:
        connection.execute(
            sa.text("""
                INSERT INTO business_limits 
                (id, limit_type, category, max_amount, geographic_region, industry_sector, active, created_at)
                VALUES 
                (:id, :limit_type, :category, :max_amount, :geographic_region, :industry_sector, :active, :created_at)
            """),
            limit
        )


def downgrade():
    """Remove default business limits"""
    connection = op.get_bind()
    
    # Remove all default limits (you might want to be more selective in production)
    connection.execute(
        sa.text("DELETE FROM business_limits WHERE created_at >= '2024-01-01'")
    )