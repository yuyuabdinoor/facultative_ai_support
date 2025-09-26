"""
Risk Analysis Agent for AI-Powered Facultative Reinsurance Decision Support System

This agent implements comprehensive risk analysis including:
- Loss history analysis using statistical methods
- Catastrophe exposure modeling based on geographic and asset data
- Financial strength assessment using ProsusAI/finbert
- Risk scoring algorithms with weighted factor analysis
- Risk report generation with structured output
- Confidence scoring for risk assessments

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 8.2
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import statistics
import math
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

from ..models.schemas import (
    RiskLevel, LossEvent, FinancialData, RiskParameters,
    RiskAnalysis, RiskScore, Application
)


# Configure logging
logger = logging.getLogger(__name__)


class CatastropheType(Enum):
    """Catastrophe types for exposure modeling"""
    EARTHQUAKE = "earthquake"
    FLOOD = "flood"
    HURRICANE = "hurricane"
    WILDFIRE = "wildfire"
    TORNADO = "tornado"
    HAIL = "hail"
    TERRORISM = "terrorism"
    CYBER = "cyber"
    PANDEMIC = "pandemic"


class GeographicRiskZone(Enum):
    """Geographic risk zones for catastrophe modeling"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


@dataclass
class LossAnalysisResult:
    """Result of loss history analysis"""
    total_losses: Decimal
    average_annual_loss: Decimal
    loss_frequency: float
    loss_severity_avg: Decimal
    loss_trend: str  # "increasing", "decreasing", "stable"
    volatility: float
    largest_loss: Decimal
    confidence_score: float
    statistical_metrics: Dict[str, float]


@dataclass
class CatastropheExposure:
    """Catastrophe exposure assessment result"""
    overall_cat_score: float
    geographic_risk_zone: GeographicRiskZone
    primary_perils: List[CatastropheType]
    pml_estimate: Decimal
    exposure_concentration: float
    mitigation_factors: List[str]
    confidence_score: float
    detailed_analysis: Dict[str, Any]


@dataclass
class FinancialStrengthAssessment:
    """Financial strength assessment result"""
    overall_rating: str  # AAA, AA, A, BBB, BB, B, CCC, D
    financial_score: float  # 0-100
    liquidity_ratio: float
    solvency_ratio: float
    profitability_score: float
    stability_score: float
    credit_risk_level: RiskLevel
    confidence_score: float
    key_metrics: Dict[str, float]
    concerns: List[str]


@dataclass
class RiskFactorWeights:
    """Risk factor weights for scoring algorithm"""
    loss_history: float = 0.25
    catastrophe_exposure: float = 0.30
    financial_strength: float = 0.20
    asset_characteristics: float = 0.15
    geographic_factors: float = 0.10


@dataclass
class RiskReportData:
    """Comprehensive risk report data"""
    application_id: str
    overall_risk_score: float
    risk_level: RiskLevel
    confidence_score: float
    loss_analysis: LossAnalysisResult
    cat_exposure: CatastropheExposure
    financial_assessment: FinancialStrengthAssessment
    risk_factors: Dict[str, float]
    recommendations: List[str]
    concerns: List[str]
    mitigation_suggestions: List[str]
    generated_at: datetime


class RiskAnalysisAgent:
    """
    Comprehensive risk analysis agent for facultative reinsurance
    
    This agent analyzes various risk factors and generates comprehensive
    risk assessments with confidence scoring and detailed reporting.
    """
    
    def __init__(self):
        """Initialize the risk analysis agent with ML models and configurations"""
        self.logger = logging.getLogger(__name__)
        
        # Cache directory and offline flag for HF models
        self.hf_cache_dir = os.environ.get("HF_HOME", "/app/.cache/huggingface")
        self.hf_offline = os.environ.get("HF_OFFLINE") in ("1", "true", "True") or \
                          os.environ.get("TRANSFORMERS_OFFLINE") in ("1", "true", "True")
        
        # Initialize financial analysis model (ProsusAI/finbert)
        try:
            self.financial_model_name = "ProsusAI/finbert"
            if self.hf_offline:
                self.financial_classifier = None
                self.logger.info("HF offline mode enabled; skipping financial model load")
            else:
                self.financial_tokenizer = AutoTokenizer.from_pretrained(self.financial_model_name, cache_dir=str(self.hf_cache_dir))
                self.financial_model = AutoModelForSequenceClassification.from_pretrained(self.financial_model_name, cache_dir=str(self.hf_cache_dir))
                self.financial_classifier = pipeline(
                    "text-classification",
                    model=self.financial_model,
                    tokenizer=self.financial_tokenizer,
                    return_all_scores=True
                )
                self.logger.info("Financial analysis model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load financial model: {e}")
            self.financial_classifier = None
        
        # Risk factor weights
        self.risk_weights = RiskFactorWeights()
        
        # Geographic risk mappings (simplified for demonstration)
        self.geographic_risk_zones = {
            # High-risk earthquake zones
            "california": GeographicRiskZone.VERY_HIGH,
            "japan": GeographicRiskZone.EXTREME,
            "turkey": GeographicRiskZone.HIGH,
            "chile": GeographicRiskZone.HIGH,
            "new zealand": GeographicRiskZone.HIGH,
            
            # Hurricane-prone areas
            "florida": GeographicRiskZone.VERY_HIGH,
            "gulf coast": GeographicRiskZone.HIGH,
            "caribbean": GeographicRiskZone.VERY_HIGH,
            "bahamas": GeographicRiskZone.VERY_HIGH,
            
            # Flood-prone areas
            "bangladesh": GeographicRiskZone.EXTREME,
            "netherlands": GeographicRiskZone.HIGH,
            "louisiana": GeographicRiskZone.HIGH,
            
            # Default moderate risk
            "default": GeographicRiskZone.MODERATE
        }
        
        # Catastrophe type mappings by geography
        self.geographic_perils = {
            "california": [CatastropheType.EARTHQUAKE, CatastropheType.WILDFIRE],
            "florida": [CatastropheType.HURRICANE, CatastropheType.FLOOD],
            "japan": [CatastropheType.EARTHQUAKE, CatastropheType.FLOOD],
            "caribbean": [CatastropheType.HURRICANE],
            "bangladesh": [CatastropheType.FLOOD, CatastropheType.TORNADO],
            "default": [CatastropheType.FLOOD]
        }
        
        # Industry risk multipliers
        self.industry_risk_multipliers = {
            "oil_gas": 1.8,
            "chemical": 1.7,
            "mining": 1.6,
            "construction": 1.4,
            "manufacturing": 1.2,
            "technology": 0.8,
            "services": 0.9,
            "retail": 1.0,
            "healthcare": 0.7,
            "education": 0.6
        }
    
    def analyze_loss_history(self, loss_events: List[LossEvent]) -> LossAnalysisResult:
        """
        Analyze historical loss patterns and trends using statistical methods
        
        Args:
            loss_events: List of historical loss events
            
        Returns:
            LossAnalysisResult with comprehensive loss analysis
        """
        try:
            if not loss_events:
                return LossAnalysisResult(
                    total_losses=Decimal('0'),
                    average_annual_loss=Decimal('0'),
                    loss_frequency=0.0,
                    loss_severity_avg=Decimal('0'),
                    loss_trend="stable",
                    volatility=0.0,
                    largest_loss=Decimal('0'),
                    confidence_score=0.0,
                    statistical_metrics={}
                )
            
            # Convert to pandas DataFrame for analysis
            df = pd.DataFrame([
                {
                    'date': event.event_date,
                    'amount': float(event.amount),
                    'cause': event.cause or 'unknown'
                }
                for event in loss_events
            ])
            
            # Sort by date
            df = df.sort_values('date')
            
            # Calculate basic statistics
            total_losses = Decimal(str(df['amount'].sum()))
            largest_loss = Decimal(str(df['amount'].max()))
            
            # Calculate time span and annual metrics
            date_range = (df['date'].max() - df['date'].min()).days
            years_span = max(date_range / 365.25, 1.0)  # Minimum 1 year
            
            average_annual_loss = total_losses / Decimal(str(years_span))
            loss_frequency = len(loss_events) / years_span
            loss_severity_avg = total_losses / len(loss_events) if loss_events else Decimal('0')
            
            # Calculate volatility (coefficient of variation)
            amounts = df['amount'].values
            volatility = float(np.std(amounts) / np.mean(amounts)) if np.mean(amounts) > 0 else 0.0
            
            # Trend analysis using linear regression
            if len(df) >= 3:
                # Convert dates to numeric for regression
                df['date_numeric'] = pd.to_numeric(df['date'])
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df['date_numeric'], df['amount']
                )
                
                if p_value < 0.05:  # Statistically significant
                    if slope > 0:
                        loss_trend = "increasing"
                    else:
                        loss_trend = "decreasing"
                else:
                    loss_trend = "stable"
                
                trend_confidence = abs(r_value)
            else:
                loss_trend = "stable"
                trend_confidence = 0.0
            
            # Statistical metrics
            statistical_metrics = {
                'mean': float(np.mean(amounts)),
                'median': float(np.median(amounts)),
                'std_dev': float(np.std(amounts)),
                'skewness': float(stats.skew(amounts)),
                'kurtosis': float(stats.kurtosis(amounts)),
                'percentile_95': float(np.percentile(amounts, 95)),
                'percentile_99': float(np.percentile(amounts, 99)),
                'trend_slope': slope if len(df) >= 3 else 0.0,
                'trend_r_squared': r_value**2 if len(df) >= 3 else 0.0
            }
            
            # Calculate confidence score based on data quality
            confidence_score = self._calculate_loss_analysis_confidence(
                len(loss_events), years_span, volatility, trend_confidence if len(df) >= 3 else 0.0
            )
            
            return LossAnalysisResult(
                total_losses=total_losses,
                average_annual_loss=average_annual_loss,
                loss_frequency=loss_frequency,
                loss_severity_avg=loss_severity_avg,
                loss_trend=loss_trend,
                volatility=volatility,
                largest_loss=largest_loss,
                confidence_score=confidence_score,
                statistical_metrics=statistical_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error in loss history analysis: {e}")
            raise
    
    def assess_catastrophe_exposure(
        self, 
        risk_parameters: RiskParameters, 
        asset_value: Optional[Decimal] = None
    ) -> CatastropheExposure:
        """
        Assess catastrophe exposure based on geographic and asset data
        
        Args:
            risk_parameters: Risk parameters including location and asset type
            asset_value: Total asset value for PML calculation
            
        Returns:
            CatastropheExposure assessment
        """
        try:
            location = (risk_parameters.location or "").lower()
            asset_type = (risk_parameters.asset_type or "").lower()
            
            # Determine geographic risk zone
            geographic_risk_zone = self._get_geographic_risk_zone(location)
            
            # Identify primary perils
            primary_perils = self._identify_primary_perils(location, asset_type)
            
            # Calculate base catastrophe score
            base_cat_score = self._calculate_base_cat_score(geographic_risk_zone, primary_perils)
            
            # Apply asset-specific modifiers
            asset_modifier = self._get_asset_cat_modifier(asset_type)
            overall_cat_score = base_cat_score * asset_modifier
            
            # Estimate PML (Probable Maximum Loss)
            pml_estimate = self._estimate_pml(asset_value, geographic_risk_zone, asset_type)
            
            # Calculate exposure concentration
            exposure_concentration = self._calculate_exposure_concentration(
                risk_parameters, geographic_risk_zone
            )
            
            # Identify mitigation factors
            mitigation_factors = self._identify_mitigation_factors(
                risk_parameters, asset_type
            )
            
            # Apply mitigation adjustments
            mitigation_adjustment = len(mitigation_factors) * 0.05  # 5% reduction per factor
            overall_cat_score = max(0.0, overall_cat_score - mitigation_adjustment)
            
            # Calculate confidence score
            confidence_score = self._calculate_cat_exposure_confidence(
                location, asset_type, risk_parameters
            )
            
            # Detailed analysis
            detailed_analysis = {
                'base_score': base_cat_score,
                'asset_modifier': asset_modifier,
                'mitigation_adjustment': mitigation_adjustment,
                'location_analysis': self._analyze_location_risk(location),
                'asset_vulnerability': self._analyze_asset_vulnerability(asset_type),
                'seasonal_factors': self._get_seasonal_factors(location, primary_perils)
            }
            
            return CatastropheExposure(
                overall_cat_score=overall_cat_score,
                geographic_risk_zone=geographic_risk_zone,
                primary_perils=primary_perils,
                pml_estimate=pml_estimate,
                exposure_concentration=exposure_concentration,
                mitigation_factors=mitigation_factors,
                confidence_score=confidence_score,
                detailed_analysis=detailed_analysis
            )
            
        except Exception as e:
            self.logger.error(f"Error in catastrophe exposure assessment: {e}")
            raise
    
    def evaluate_financial_strength(self, financial_data: FinancialData) -> FinancialStrengthAssessment:
        """
        Assess financial strength using ProsusAI/finbert and financial ratios
        
        Args:
            financial_data: Financial data for analysis
            
        Returns:
            FinancialStrengthAssessment with comprehensive financial analysis
        """
        try:
            # Calculate financial ratios
            key_metrics = self._calculate_financial_ratios(financial_data)
            
            # Assess liquidity
            liquidity_ratio = self._assess_liquidity(financial_data, key_metrics)
            
            # Assess solvency
            solvency_ratio = self._assess_solvency(financial_data, key_metrics)
            
            # Assess profitability
            profitability_score = self._assess_profitability(financial_data, key_metrics)
            
            # Assess stability
            stability_score = self._assess_financial_stability(financial_data, key_metrics)
            
            # Use FinBERT for sentiment analysis if available
            finbert_score = 0.5  # Default neutral
            if self.financial_classifier and financial_data.credit_rating:
                finbert_score = self._analyze_with_finbert(financial_data)
            
            # Calculate overall financial score
            financial_score = self._calculate_overall_financial_score(
                liquidity_ratio, solvency_ratio, profitability_score, 
                stability_score, finbert_score
            )
            
            # Determine overall rating
            overall_rating = self._determine_financial_rating(financial_score)
            
            # Determine credit risk level
            credit_risk_level = self._determine_credit_risk_level(financial_score)
            
            # Identify concerns
            concerns = self._identify_financial_concerns(
                financial_data, key_metrics, financial_score
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_financial_confidence(
                financial_data, key_metrics
            )
            
            return FinancialStrengthAssessment(
                overall_rating=overall_rating,
                financial_score=financial_score,
                liquidity_ratio=liquidity_ratio,
                solvency_ratio=solvency_ratio,
                profitability_score=profitability_score,
                stability_score=stability_score,
                credit_risk_level=credit_risk_level,
                confidence_score=confidence_score,
                key_metrics=key_metrics,
                concerns=concerns
            )
            
        except Exception as e:
            self.logger.error(f"Error in financial strength evaluation: {e}")
            raise
    
    def calculate_risk_score(
        self,
        loss_analysis: LossAnalysisResult,
        cat_exposure: CatastropheExposure,
        financial_assessment: FinancialStrengthAssessment,
        risk_parameters: RiskParameters
    ) -> RiskScore:
        """
        Calculate overall risk score using weighted factor analysis
        
        Args:
            loss_analysis: Loss history analysis results
            cat_exposure: Catastrophe exposure assessment
            financial_assessment: Financial strength assessment
            risk_parameters: Risk parameters
            
        Returns:
            RiskScore with overall assessment
        """
        try:
            # Normalize individual scores to 0-100 scale
            loss_score = self._normalize_loss_score(loss_analysis)
            cat_score = min(100.0, cat_exposure.overall_cat_score * 10)  # Scale to 0-100
            financial_score = 100.0 - financial_assessment.financial_score  # Invert (higher financial strength = lower risk)
            asset_score = self._calculate_asset_risk_score(risk_parameters)
            geographic_score = self._calculate_geographic_risk_score(risk_parameters)
            
            # Apply weighted scoring
            overall_score = (
                loss_score * self.risk_weights.loss_history +
                cat_score * self.risk_weights.catastrophe_exposure +
                financial_score * self.risk_weights.financial_strength +
                asset_score * self.risk_weights.asset_characteristics +
                geographic_score * self.risk_weights.geographic_factors
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(overall_score)
            
            # Calculate confidence based on component confidences
            confidence = (
                loss_analysis.confidence_score * self.risk_weights.loss_history +
                cat_exposure.confidence_score * self.risk_weights.catastrophe_exposure +
                financial_assessment.confidence_score * self.risk_weights.financial_strength +
                0.8 * self.risk_weights.asset_characteristics +  # Assume good asset data confidence
                0.7 * self.risk_weights.geographic_factors  # Assume moderate geographic confidence
            )
            
            # Risk factors breakdown
            factors = {
                'loss_history': loss_score,
                'catastrophe_exposure': cat_score,
                'financial_strength': financial_score,
                'asset_characteristics': asset_score,
                'geographic_factors': geographic_score,
                'weights': {
                    'loss_history': self.risk_weights.loss_history,
                    'catastrophe_exposure': self.risk_weights.catastrophe_exposure,
                    'financial_strength': self.risk_weights.financial_strength,
                    'asset_characteristics': self.risk_weights.asset_characteristics,
                    'geographic_factors': self.risk_weights.geographic_factors
                }
            }
            
            return RiskScore(
                overall_score=Decimal(str(round(overall_score, 2))),
                confidence=Decimal(str(round(confidence, 3))),
                risk_level=risk_level,
                factors=factors
            )
            
        except Exception as e:
            self.logger.error(f"Error in risk score calculation: {e}")
            raise
    
    def generate_risk_report(
        self,
        application: Application,
        loss_analysis: LossAnalysisResult,
        cat_exposure: CatastropheExposure,
        financial_assessment: FinancialStrengthAssessment,
        risk_score: RiskScore
    ) -> RiskReportData:
        """
        Generate comprehensive risk report with structured output
        
        Args:
            application: Application data
            loss_analysis: Loss analysis results
            cat_exposure: Catastrophe exposure assessment
            financial_assessment: Financial assessment
            risk_score: Overall risk score
            
        Returns:
            RiskReportData with comprehensive report
        """
        try:
            # Generate recommendations based on risk level
            recommendations = self._generate_recommendations(
                risk_score.risk_level, loss_analysis, cat_exposure, financial_assessment
            )
            
            # Identify key concerns
            concerns = self._identify_key_concerns(
                loss_analysis, cat_exposure, financial_assessment, risk_score
            )
            
            # Generate mitigation suggestions
            mitigation_suggestions = self._generate_mitigation_suggestions(
                cat_exposure, financial_assessment, application.risk_parameters
            )
            
            return RiskReportData(
                application_id=str(application.id),
                overall_risk_score=float(risk_score.overall_score),
                risk_level=risk_score.risk_level,
                confidence_score=float(risk_score.confidence),
                loss_analysis=loss_analysis,
                cat_exposure=cat_exposure,
                financial_assessment=financial_assessment,
                risk_factors=risk_score.factors,
                recommendations=recommendations,
                concerns=concerns,
                mitigation_suggestions=mitigation_suggestions,
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error in risk report generation: {e}")
            raise
    
    def perform_comprehensive_analysis(self, application: Application) -> RiskAnalysis:
        """
        Perform comprehensive risk analysis for an application
        
        Args:
            application: Complete application data
            
        Returns:
            RiskAnalysis with complete assessment
        """
        try:
            self.logger.info(f"Starting comprehensive risk analysis for application {application.id}")
            
            # Analyze loss history
            loss_analysis = self.analyze_loss_history(application.loss_history or [])
            
            # Assess catastrophe exposure
            cat_exposure = self.assess_catastrophe_exposure(
                application.risk_parameters,
                application.risk_parameters.asset_value if application.risk_parameters else None
            )
            
            # Evaluate financial strength
            financial_assessment = self.evaluate_financial_strength(
                application.financial_data or FinancialData(
                    id="", application_id=str(application.id)
                )
            )
            
            # Calculate overall risk score
            risk_score = self.calculate_risk_score(
                loss_analysis, cat_exposure, financial_assessment,
                application.risk_parameters or RiskParameters(
                    id="", application_id=str(application.id)
                )
            )
            
            # Generate comprehensive report
            risk_report = self.generate_risk_report(
                application, loss_analysis, cat_exposure, 
                financial_assessment, risk_score
            )
            
            # Create RiskAnalysis object
            analysis_data = {
                'loss_analysis': {
                    'total_losses': str(loss_analysis.total_losses),
                    'average_annual_loss': str(loss_analysis.average_annual_loss),
                    'loss_frequency': loss_analysis.loss_frequency,
                    'loss_trend': loss_analysis.loss_trend,
                    'volatility': loss_analysis.volatility,
                    'statistical_metrics': loss_analysis.statistical_metrics
                },
                'catastrophe_exposure': {
                    'overall_score': cat_exposure.overall_cat_score,
                    'risk_zone': cat_exposure.geographic_risk_zone.value,
                    'primary_perils': [p.value for p in cat_exposure.primary_perils],
                    'pml_estimate': str(cat_exposure.pml_estimate),
                    'mitigation_factors': cat_exposure.mitigation_factors
                },
                'financial_assessment': {
                    'overall_rating': financial_assessment.overall_rating,
                    'financial_score': financial_assessment.financial_score,
                    'credit_risk_level': financial_assessment.credit_risk_level.value,
                    'key_metrics': financial_assessment.key_metrics,
                    'concerns': financial_assessment.concerns
                },
                'recommendations': risk_report.recommendations,
                'concerns': risk_report.concerns,
                'mitigation_suggestions': risk_report.mitigation_suggestions
            }
            
            risk_analysis = RiskAnalysis(
                id="",  # Will be set by database
                application_id=str(application.id),
                overall_score=risk_score.overall_score,
                confidence=risk_score.confidence,
                risk_level=risk_score.risk_level,
                factors=risk_score.factors,
                analysis_data=analysis_data,
                created_at=datetime.utcnow()
            )
            
            self.logger.info(f"Risk analysis completed for application {application.id}")
            return risk_analysis
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive risk analysis: {e}")
            raise    

    # Helper methods for risk analysis calculations
    
    def _calculate_loss_analysis_confidence(
        self, 
        num_events: int, 
        years_span: float, 
        volatility: float, 
        trend_confidence: float
    ) -> float:
        """Calculate confidence score for loss analysis"""
        # Base confidence on data quantity
        data_confidence = min(1.0, num_events / 10.0)  # Full confidence with 10+ events
        
        # Adjust for time span (prefer longer history)
        time_confidence = min(1.0, years_span / 5.0)  # Full confidence with 5+ years
        
        # Adjust for volatility (lower volatility = higher confidence)
        volatility_confidence = max(0.1, 1.0 - min(volatility, 1.0))
        
        # Combine factors
        confidence = (data_confidence * 0.4 + time_confidence * 0.3 + 
                     volatility_confidence * 0.2 + trend_confidence * 0.1)
        
        return min(1.0, confidence)
    
    def _get_geographic_risk_zone(self, location: str) -> GeographicRiskZone:
        """Determine geographic risk zone from location"""
        location_lower = location.lower()
        
        for geo_key, risk_zone in self.geographic_risk_zones.items():
            if geo_key in location_lower:
                return risk_zone
        
        return self.geographic_risk_zones["default"]
    
    def _identify_primary_perils(self, location: str, asset_type: str) -> List[CatastropheType]:
        """Identify primary catastrophe perils for location and asset type"""
        location_lower = location.lower()
        
        # Get location-based perils
        perils = []
        for geo_key, geo_perils in self.geographic_perils.items():
            if geo_key in location_lower:
                perils.extend(geo_perils)
                break
        
        if not perils:
            perils = self.geographic_perils["default"]
        
        # Add asset-specific perils
        asset_lower = asset_type.lower()
        if "technology" in asset_lower or "data" in asset_lower:
            perils.append(CatastropheType.CYBER)
        if "chemical" in asset_lower or "oil" in asset_lower:
            perils.extend([CatastropheType.TERRORISM])
        
        return list(set(perils))  # Remove duplicates
    
    def _calculate_base_cat_score(
        self, 
        risk_zone: GeographicRiskZone, 
        perils: List[CatastropheType]
    ) -> float:
        """Calculate base catastrophe score"""
        zone_scores = {
            GeographicRiskZone.VERY_LOW: 1.0,
            GeographicRiskZone.LOW: 2.0,
            GeographicRiskZone.MODERATE: 4.0,
            GeographicRiskZone.HIGH: 7.0,
            GeographicRiskZone.VERY_HIGH: 9.0,
            GeographicRiskZone.EXTREME: 10.0
        }
        
        base_score = zone_scores.get(risk_zone, 4.0)
        
        # Adjust for number and severity of perils
        peril_multiplier = 1.0 + (len(perils) - 1) * 0.2  # 20% increase per additional peril
        
        return base_score * peril_multiplier
    
    def _get_asset_cat_modifier(self, asset_type: str) -> float:
        """Get catastrophe modifier based on asset type"""
        asset_lower = asset_type.lower()
        
        if "residential" in asset_lower:
            return 0.8
        elif "commercial" in asset_lower:
            return 1.0
        elif "industrial" in asset_lower:
            return 1.3
        elif "infrastructure" in asset_lower:
            return 1.5
        elif "energy" in asset_lower:
            return 1.4
        else:
            return 1.0
    
    def _estimate_pml(
        self, 
        asset_value: Optional[Decimal], 
        risk_zone: GeographicRiskZone, 
        asset_type: str
    ) -> Decimal:
        """Estimate Probable Maximum Loss"""
        if not asset_value:
            return Decimal('0')
        
        # Base PML percentages by risk zone
        pml_percentages = {
            GeographicRiskZone.VERY_LOW: 0.05,
            GeographicRiskZone.LOW: 0.10,
            GeographicRiskZone.MODERATE: 0.20,
            GeographicRiskZone.HIGH: 0.40,
            GeographicRiskZone.VERY_HIGH: 0.60,
            GeographicRiskZone.EXTREME: 0.80
        }
        
        base_pml_pct = pml_percentages.get(risk_zone, 0.20)
        
        # Adjust for asset type
        asset_modifier = self._get_asset_cat_modifier(asset_type)
        adjusted_pml_pct = min(1.0, base_pml_pct * asset_modifier)
        
        return asset_value * Decimal(str(adjusted_pml_pct))
    
    def _calculate_exposure_concentration(
        self, 
        risk_parameters: RiskParameters, 
        risk_zone: GeographicRiskZone
    ) -> float:
        """Calculate exposure concentration factor"""
        # Simple concentration calculation based on asset value and location
        base_concentration = 0.5  # Default moderate concentration
        
        # Adjust based on risk zone
        zone_adjustments = {
            GeographicRiskZone.VERY_LOW: -0.2,
            GeographicRiskZone.LOW: -0.1,
            GeographicRiskZone.MODERATE: 0.0,
            GeographicRiskZone.HIGH: 0.2,
            GeographicRiskZone.VERY_HIGH: 0.3,
            GeographicRiskZone.EXTREME: 0.4
        }
        
        adjustment = zone_adjustments.get(risk_zone, 0.0)
        return max(0.0, min(1.0, base_concentration + adjustment))
    
    def _identify_mitigation_factors(
        self, 
        risk_parameters: RiskParameters, 
        asset_type: str
    ) -> List[str]:
        """Identify risk mitigation factors"""
        mitigation_factors = []
        
        construction_type = (risk_parameters.construction_type or "").lower()
        if "concrete" in construction_type or "steel" in construction_type:
            mitigation_factors.append("Fire-resistant construction")
        if "sprinkler" in construction_type:
            mitigation_factors.append("Sprinkler system")
        
        asset_lower = asset_type.lower()
        if "technology" in asset_lower:
            mitigation_factors.append("Cyber security measures")
        if "industrial" in asset_lower:
            mitigation_factors.append("Industrial safety protocols")
        
        return mitigation_factors
    
    def _calculate_cat_exposure_confidence(
        self, 
        location: str, 
        asset_type: str, 
        risk_parameters: RiskParameters
    ) -> float:
        """Calculate confidence score for catastrophe exposure assessment"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we have detailed location info
        if location and len(location) > 10:
            confidence += 0.2
        
        # Increase confidence if we have asset type info
        if asset_type:
            confidence += 0.2
        
        # Increase confidence if we have construction details
        if risk_parameters.construction_type:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _analyze_location_risk(self, location: str) -> Dict[str, Any]:
        """Analyze location-specific risk factors"""
        return {
            'location': location,
            'risk_zone': self._get_geographic_risk_zone(location).value,
            'primary_perils': [p.value for p in self._identify_primary_perils(location, "")],
            'analysis_notes': f"Location analysis for {location}"
        }
    
    def _analyze_asset_vulnerability(self, asset_type: str) -> Dict[str, Any]:
        """Analyze asset-specific vulnerability factors"""
        return {
            'asset_type': asset_type,
            'vulnerability_score': self._get_asset_cat_modifier(asset_type),
            'specific_risks': self._get_asset_specific_risks(asset_type)
        }
    
    def _get_asset_specific_risks(self, asset_type: str) -> List[str]:
        """Get asset-specific risk factors"""
        asset_lower = asset_type.lower()
        risks = []
        
        if "technology" in asset_lower:
            risks.extend(["Cyber attacks", "Equipment failure", "Data loss"])
        if "industrial" in asset_lower:
            risks.extend(["Fire", "Explosion", "Equipment breakdown"])
        if "residential" in asset_lower:
            risks.extend(["Fire", "Theft", "Natural disasters"])
        
        return risks
    
    def _get_seasonal_factors(
        self, 
        location: str, 
        perils: List[CatastropheType]
    ) -> Dict[str, float]:
        """Get seasonal risk factors"""
        seasonal_factors = {}
        
        for peril in perils:
            if peril == CatastropheType.HURRICANE:
                seasonal_factors['hurricane_season'] = 1.5  # June-November
            elif peril == CatastropheType.WILDFIRE:
                seasonal_factors['fire_season'] = 1.3  # Summer months
            elif peril == CatastropheType.FLOOD:
                seasonal_factors['flood_season'] = 1.2  # Spring/monsoon
        
        return seasonal_factors
    
    def _calculate_financial_ratios(self, financial_data: FinancialData) -> Dict[str, float]:
        """Calculate key financial ratios"""
        ratios = {}
        
        if financial_data.assets and financial_data.liabilities:
            # Debt-to-asset ratio
            ratios['debt_to_asset'] = float(financial_data.liabilities / financial_data.assets)
            
            # Equity ratio
            equity = financial_data.assets - financial_data.liabilities
            ratios['equity_ratio'] = float(equity / financial_data.assets)
        
        if financial_data.revenue and financial_data.assets:
            # Asset turnover
            ratios['asset_turnover'] = float(financial_data.revenue / financial_data.assets)
        
        if financial_data.revenue and financial_data.liabilities:
            # Revenue to debt ratio
            ratios['revenue_to_debt'] = float(financial_data.revenue / financial_data.liabilities)
        
        return ratios
    
    def _assess_liquidity(self, financial_data: FinancialData, ratios: Dict[str, float]) -> float:
        """Assess liquidity position"""
        # Simplified liquidity assessment
        if 'equity_ratio' in ratios:
            return min(1.0, ratios['equity_ratio'] * 2)  # Scale equity ratio
        return 0.5  # Default moderate liquidity
    
    def _assess_solvency(self, financial_data: FinancialData, ratios: Dict[str, float]) -> float:
        """Assess solvency position"""
        if 'debt_to_asset' in ratios:
            return max(0.0, 1.0 - ratios['debt_to_asset'])  # Lower debt = higher solvency
        return 0.5  # Default moderate solvency
    
    def _assess_profitability(self, financial_data: FinancialData, ratios: Dict[str, float]) -> float:
        """Assess profitability"""
        if 'asset_turnover' in ratios:
            return min(1.0, ratios['asset_turnover'])
        return 0.5  # Default moderate profitability
    
    def _assess_financial_stability(self, financial_data: FinancialData, ratios: Dict[str, float]) -> float:
        """Assess financial stability"""
        stability_score = 0.5  # Base score
        
        # Adjust based on credit rating
        if financial_data.credit_rating:
            rating_scores = {
                'AAA': 1.0, 'AA': 0.9, 'A': 0.8, 'BBB': 0.7,
                'BB': 0.6, 'B': 0.4, 'CCC': 0.2, 'D': 0.0
            }
            stability_score = rating_scores.get(financial_data.credit_rating, 0.5)
        
        return stability_score
    
    def _analyze_with_finbert(self, financial_data: FinancialData) -> float:
        """Analyze financial sentiment using FinBERT"""
        if not self.financial_classifier:
            return 0.5
        
        # Create text for analysis
        text = f"Credit rating: {financial_data.credit_rating or 'Not available'}"
        if financial_data.financial_strength_rating:
            text += f" Financial strength: {financial_data.financial_strength_rating}"
        
        try:
            results = self.financial_classifier(text)
            # Convert sentiment to score (assuming positive=good, negative=bad)
            for result in results[0]:
                if result['label'].upper() in ['POSITIVE', 'POS']:
                    return result['score']
                elif result['label'].upper() in ['NEGATIVE', 'NEG']:
                    return 1.0 - result['score']
        except Exception as e:
            self.logger.warning(f"FinBERT analysis failed: {e}")
        
        return 0.5  # Default neutral
    
    def _calculate_overall_financial_score(
        self, 
        liquidity: float, 
        solvency: float, 
        profitability: float, 
        stability: float, 
        finbert_score: float
    ) -> float:
        """Calculate overall financial score"""
        weights = {
            'liquidity': 0.2,
            'solvency': 0.3,
            'profitability': 0.2,
            'stability': 0.2,
            'finbert': 0.1
        }
        
        score = (
            liquidity * weights['liquidity'] +
            solvency * weights['solvency'] +
            profitability * weights['profitability'] +
            stability * weights['stability'] +
            finbert_score * weights['finbert']
        )
        
        return min(100.0, score * 100)  # Scale to 0-100
    
    def _determine_financial_rating(self, financial_score: float) -> str:
        """Determine financial rating from score"""
        if financial_score >= 90:
            return "AAA"
        elif financial_score >= 80:
            return "AA"
        elif financial_score >= 70:
            return "A"
        elif financial_score >= 60:
            return "BBB"
        elif financial_score >= 50:
            return "BB"
        elif financial_score >= 40:
            return "B"
        elif financial_score >= 30:
            return "CCC"
        else:
            return "D"
    
    def _determine_credit_risk_level(self, financial_score: float) -> RiskLevel:
        """Determine credit risk level from financial score"""
        if financial_score >= 80:
            return RiskLevel.LOW
        elif financial_score >= 60:
            return RiskLevel.MEDIUM
        elif financial_score >= 40:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _identify_financial_concerns(
        self, 
        financial_data: FinancialData, 
        ratios: Dict[str, float], 
        score: float
    ) -> List[str]:
        """Identify financial concerns"""
        concerns = []
        
        if score < 50:
            concerns.append("Overall financial strength is below acceptable levels")
        
        if 'debt_to_asset' in ratios and ratios['debt_to_asset'] > 0.7:
            concerns.append("High debt-to-asset ratio indicates leverage risk")
        
        if not financial_data.credit_rating:
            concerns.append("No credit rating available for assessment")
        
        if financial_data.credit_rating and financial_data.credit_rating in ['CCC', 'D']:
            concerns.append("Poor credit rating indicates high default risk")
        
        return concerns
    
    def _calculate_financial_confidence(
        self, 
        financial_data: FinancialData, 
        ratios: Dict[str, float]
    ) -> float:
        """Calculate confidence in financial assessment"""
        confidence = 0.3  # Base confidence
        
        # Increase confidence based on available data
        if financial_data.revenue:
            confidence += 0.15
        if financial_data.assets:
            confidence += 0.15
        if financial_data.liabilities:
            confidence += 0.15
        if financial_data.credit_rating:
            confidence += 0.15
        if financial_data.financial_strength_rating:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _normalize_loss_score(self, loss_analysis: LossAnalysisResult) -> float:
        """Normalize loss analysis to 0-100 risk score"""
        if loss_analysis.loss_frequency == 0:
            return 10.0  # Low risk if no losses
        
        # Base score on frequency and severity
        frequency_score = min(50.0, loss_analysis.loss_frequency * 10)
        
        # Volatility adjustment
        volatility_score = min(30.0, loss_analysis.volatility * 30)
        
        # Trend adjustment
        trend_score = 0.0
        if loss_analysis.loss_trend == "increasing":
            trend_score = 20.0
        elif loss_analysis.loss_trend == "decreasing":
            trend_score = -10.0
        
        total_score = frequency_score + volatility_score + trend_score
        return max(0.0, min(100.0, total_score))
    
    def _calculate_asset_risk_score(self, risk_parameters: RiskParameters) -> float:
        """Calculate asset-specific risk score"""
        base_score = 50.0  # Default moderate risk
        
        # Adjust based on industry sector
        if risk_parameters.industry_sector:
            sector_lower = risk_parameters.industry_sector.lower()
            multiplier = self.industry_risk_multipliers.get(
                sector_lower.replace(" ", "_"), 1.0
            )
            base_score *= multiplier
        
        # Adjust based on construction type
        if risk_parameters.construction_type:
            construction_lower = risk_parameters.construction_type.lower()
            if "concrete" in construction_lower or "steel" in construction_lower:
                base_score *= 0.8  # Lower risk for fire-resistant construction
            elif "wood" in construction_lower:
                base_score *= 1.2  # Higher risk for combustible construction
        
        return min(100.0, base_score)
    
    def _calculate_geographic_risk_score(self, risk_parameters: RiskParameters) -> float:
        """Calculate geographic risk score"""
        if not risk_parameters.location:
            return 50.0  # Default moderate risk
        
        risk_zone = self._get_geographic_risk_zone(risk_parameters.location)
        
        zone_scores = {
            GeographicRiskZone.VERY_LOW: 10.0,
            GeographicRiskZone.LOW: 20.0,
            GeographicRiskZone.MODERATE: 40.0,
            GeographicRiskZone.HIGH: 70.0,
            GeographicRiskZone.VERY_HIGH: 85.0,
            GeographicRiskZone.EXTREME: 95.0
        }
        
        return zone_scores.get(risk_zone, 40.0)
    
    def _determine_risk_level(self, overall_score: float) -> RiskLevel:
        """Determine risk level from overall score"""
        if overall_score <= 25:
            return RiskLevel.LOW
        elif overall_score <= 50:
            return RiskLevel.MEDIUM
        elif overall_score <= 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _generate_recommendations(
        self,
        risk_level: RiskLevel,
        loss_analysis: LossAnalysisResult,
        cat_exposure: CatastropheExposure,
        financial_assessment: FinancialStrengthAssessment
    ) -> List[str]:
        """Generate recommendations based on risk assessment"""
        recommendations = []
        
        if risk_level == RiskLevel.LOW:
            recommendations.append("Risk profile is acceptable for standard terms")
            recommendations.append("Consider competitive pricing")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("Risk profile is acceptable with standard terms")
            recommendations.append("Monitor for changes in risk factors")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Consider conditional approval with enhanced terms")
            recommendations.append("Require additional risk mitigation measures")
            recommendations.append("Implement enhanced monitoring")
        else:  # CRITICAL
            recommendations.append("High risk - consider rejection or substantial premium adjustment")
            recommendations.append("Require comprehensive risk mitigation plan")
            recommendations.append("Consider co-insurance or risk sharing")
        
        # Specific recommendations based on analysis
        if loss_analysis.loss_trend == "increasing":
            recommendations.append("Address increasing loss trend before approval")
        
        if cat_exposure.overall_cat_score > 8.0:
            recommendations.append("High catastrophe exposure - consider aggregate limits")
        
        if financial_assessment.financial_score < 60:
            recommendations.append("Financial strength concerns - require guarantees")
        
        return recommendations
    
    def _identify_key_concerns(
        self,
        loss_analysis: LossAnalysisResult,
        cat_exposure: CatastropheExposure,
        financial_assessment: FinancialStrengthAssessment,
        risk_score: RiskScore
    ) -> List[str]:
        """Identify key risk concerns"""
        concerns = []
        
        if risk_score.overall_score > 75:
            concerns.append("Overall risk score exceeds acceptable thresholds")
        
        if loss_analysis.loss_frequency > 2.0:
            concerns.append("High loss frequency indicates systemic issues")
        
        if loss_analysis.volatility > 1.0:
            concerns.append("High loss volatility creates unpredictable exposure")
        
        if cat_exposure.overall_cat_score > 8.0:
            concerns.append("Extreme catastrophe exposure in high-risk zone")
        
        if financial_assessment.credit_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            concerns.append("Financial strength raises solvency concerns")
        
        concerns.extend(financial_assessment.concerns)
        
        return concerns
    
    def _generate_mitigation_suggestions(
        self,
        cat_exposure: CatastropheExposure,
        financial_assessment: FinancialStrengthAssessment,
        risk_parameters: Optional[RiskParameters]
    ) -> List[str]:
        """Generate risk mitigation suggestions"""
        suggestions = []
        
        # Catastrophe mitigation
        if cat_exposure.overall_cat_score > 6.0:
            suggestions.append("Implement catastrophe risk reduction measures")
            suggestions.append("Consider catastrophe insurance or reinsurance")
        
        for peril in cat_exposure.primary_perils:
            if peril == CatastropheType.EARTHQUAKE:
                suggestions.append("Seismic retrofitting and building reinforcement")
            elif peril == CatastropheType.FLOOD:
                suggestions.append("Flood barriers and drainage improvements")
            elif peril == CatastropheType.WILDFIRE:
                suggestions.append("Defensible space and fire-resistant materials")
            elif peril == CatastropheType.CYBER:
                suggestions.append("Enhanced cybersecurity measures and backup systems")
        
        # Financial mitigation
        if financial_assessment.financial_score < 70:
            suggestions.append("Improve financial position through debt reduction")
            suggestions.append("Consider financial guarantees or collateral")
        
        # Asset-specific mitigation
        if risk_parameters and risk_parameters.construction_type:
            construction_lower = risk_parameters.construction_type.lower()
            if "wood" in construction_lower:
                suggestions.append("Upgrade to fire-resistant construction materials")
        
        return suggestions