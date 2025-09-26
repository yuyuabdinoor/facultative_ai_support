"""
Decision Engine Agent for AI-Powered Facultative Reinsurance Decision Support System

This agent implements intelligent decision-making including:
- Multi-factor decision algorithms using ensemble methods
- Recommendation generation with microsoft/DialoGPT-medium
- Rationale generation using facebook/bart-base
- Confidence scoring with microsoft/deberta-v3-base
- Conditional approval terms suggestion logic
- Decision explanation and transparency features

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 8.2
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import json
import re
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM, AutoModel, BartForConditionalGeneration,
    DebertaV2ForSequenceClassification
)
import torch
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from ..models.schemas import (
    DecisionType, RiskLevel, Recommendation, RiskAnalysis,
    Application, BusinessLimit, RiskParameters, FinancialData
)


# Configure logging
logger = logging.getLogger(__name__)


class DecisionFactor(Enum):
    """Decision factors for multi-factor analysis"""
    RISK_SCORE = "risk_score"
    FINANCIAL_STRENGTH = "financial_strength"
    LOSS_HISTORY = "loss_history"
    CATASTROPHE_EXPOSURE = "catastrophe_exposure"
    BUSINESS_LIMITS = "business_limits"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    MARKET_CONDITIONS = "market_conditions"
    PORTFOLIO_CONCENTRATION = "portfolio_concentration"


class ConfidenceLevel(Enum):
    """Confidence levels for decisions"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class DecisionFactorScore:
    """Individual decision factor score"""
    factor: DecisionFactor
    score: float  # 0-100 scale
    weight: float  # Factor weight in decision
    confidence: float  # Confidence in this factor
    rationale: str  # Explanation for this factor
    supporting_data: Dict[str, Any]


@dataclass
class DecisionContext:
    """Context information for decision making"""
    application_id: str
    risk_analysis: RiskAnalysis
    business_limits_check: Dict[str, Any]
    market_conditions: Dict[str, Any]
    portfolio_context: Dict[str, Any]
    regulatory_requirements: List[str]
    underwriter_preferences: Dict[str, Any]


@dataclass
class ConditionalTerm:
    """Conditional approval term"""
    term_type: str  # "premium_adjustment", "coverage_limit", "deductible", "exclusion"
    description: str
    value: Optional[Union[float, str]]
    rationale: str
    impact_on_risk: float  # Expected risk reduction (-1 to 1)


@dataclass
class DecisionExplanation:
    """Detailed decision explanation"""
    primary_reasons: List[str]
    supporting_factors: List[str]
    risk_mitigation: List[str]
    concerns_addressed: List[str]
    alternative_considerations: List[str]
    confidence_factors: List[str]


@dataclass
class DecisionResult:
    """Complete decision result"""
    decision: DecisionType
    confidence: float
    rationale: str
    factor_scores: List[DecisionFactorScore]
    conditional_terms: List[ConditionalTerm]
    explanation: DecisionExplanation
    premium_adjustment: Optional[float]
    coverage_modifications: List[str]
    generated_at: datetime
    model_versions: Dict[str, str]


class DecisionEngineAgent:
    """
    Intelligent decision engine for facultative reinsurance recommendations
    
    This agent uses ensemble methods and multiple AI models to generate
    comprehensive recommendations with detailed rationale and confidence scoring.
    """
    
    def __init__(self):
        """Initialize the decision engine with AI models and configurations"""
        self.logger = logging.getLogger(__name__)
        
        # Cache directory for HF models
        self.hf_cache_dir = os.environ.get("HF_HOME", "/app/.cache/huggingface")
        
        # Initialize decision generation model (microsoft/DialoGPT-medium)
        try:
            self.decision_model_name = "microsoft/DialoGPT-medium"
            self.decision_tokenizer = AutoTokenizer.from_pretrained(self.decision_model_name, cache_dir=str(self.hf_cache_dir))
            self.decision_model = AutoModelForCausalLM.from_pretrained(self.decision_model_name, cache_dir=str(self.hf_cache_dir))
            
            # Add padding token if not present
            if self.decision_tokenizer.pad_token is None:
                self.decision_tokenizer.pad_token = self.decision_tokenizer.eos_token
            
            self.decision_generator = pipeline(
                "text-generation",
                model=self.decision_model,
                tokenizer=self.decision_tokenizer,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.decision_tokenizer.eos_token_id
            )
            self.logger.info("Decision generation model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load decision model: {e}")
            self.decision_generator = None
        
        # Initialize rationale generation model (facebook/bart-base)
        try:
            self.rationale_model_name = "facebook/bart-base"
            self.rationale_model = BartForConditionalGeneration.from_pretrained(self.rationale_model_name, cache_dir=str(self.hf_cache_dir))
            self.rationale_tokenizer = AutoTokenizer.from_pretrained(self.rationale_model_name, cache_dir=str(self.hf_cache_dir))
            self.rationale_generator = pipeline(
                "text2text-generation",
                model=self.rationale_model,
                tokenizer=self.rationale_tokenizer,
                max_length=256,
                do_sample=True,
                temperature=0.6
            )
            self.logger.info("Rationale generation model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load rationale model: {e}")
            self.rationale_generator = None
        
        # Initialize confidence scoring model (microsoft/deberta-v3-base)
        try:
            self.confidence_model_name = "microsoft/deberta-v3-base"
            self.confidence_model = DebertaV2ForSequenceClassification.from_pretrained(
                self.confidence_model_name, num_labels=5, cache_dir=str(self.hf_cache_dir)
            )
            self.confidence_tokenizer = AutoTokenizer.from_pretrained(self.confidence_model_name, cache_dir=str(self.hf_cache_dir))
            self.confidence_classifier = pipeline(
                "text-classification",
                model=self.confidence_model,
                tokenizer=self.confidence_tokenizer,
                return_all_scores=True
            )
            self.logger.info("Confidence scoring model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load confidence model: {e}")
            self.confidence_classifier = None
        
        # Decision factor weights (can be adjusted based on business rules)
        self.factor_weights = {
            DecisionFactor.RISK_SCORE: 0.25,
            DecisionFactor.FINANCIAL_STRENGTH: 0.20,
            DecisionFactor.LOSS_HISTORY: 0.15,
            DecisionFactor.CATASTROPHE_EXPOSURE: 0.15,
            DecisionFactor.BUSINESS_LIMITS: 0.10,
            DecisionFactor.REGULATORY_COMPLIANCE: 0.08,
            DecisionFactor.MARKET_CONDITIONS: 0.04,
            DecisionFactor.PORTFOLIO_CONCENTRATION: 0.03
        }
        
        # Decision thresholds
        self.decision_thresholds = {
            "approve_threshold": 70.0,  # Score above this = approve
            "reject_threshold": 30.0,   # Score below this = reject
            "high_confidence_threshold": 0.8,
            "low_confidence_threshold": 0.4
        }
        
        # Initialize ensemble classifier for decision making
        self.ensemble_classifier = None
        self._initialize_ensemble_classifier()
    
    def _initialize_ensemble_classifier(self):
        """Initialize ensemble classifier for decision making"""
        try:
            # Create ensemble of different classifiers
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.naive_bayes import GaussianNB
            
            classifiers = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('lr', LogisticRegression(random_state=42)),
                ('svm', SVC(probability=True, random_state=42)),
                ('nb', GaussianNB())
            ]
            
            self.ensemble_classifier = VotingClassifier(
                estimators=classifiers,
                voting='soft'  # Use probability voting
            )
            
            # Note: In production, this would be trained on historical data
            # For now, we'll use rule-based decisions with ML-style scoring
            
        except Exception as e:
            self.logger.warning(f"Could not initialize ensemble classifier: {e}")    

    def generate_recommendation(
        self,
        application: Application,
        risk_analysis: RiskAnalysis,
        business_limits_check: Dict[str, Any],
        context: Optional[DecisionContext] = None
    ) -> Recommendation:
        """
        Generate intelligent recommendation using multi-factor analysis
        
        Args:
            application: Application data
            risk_analysis: Risk analysis results
            business_limits_check: Business limits validation results
            context: Additional decision context
            
        Returns:
            Recommendation with decision, rationale, and conditions
        """
        try:
            self.logger.info(f"Generating recommendation for application {application.id}")
            
            # Create decision context if not provided
            if context is None:
                context = self._create_decision_context(
                    application, risk_analysis, business_limits_check
                )
            
            # Calculate factor scores
            factor_scores = self._calculate_factor_scores(context)
            
            # Make primary decision using ensemble approach
            decision_result = self._make_ensemble_decision(factor_scores, context)
            
            # Generate rationale using AI models
            rationale = self._generate_rationale(decision_result, factor_scores, context)
            
            # Calculate confidence score
            confidence = self._calculate_decision_confidence(decision_result, factor_scores)
            
            # Generate conditional terms if applicable
            conditional_terms = []
            coverage_modifications = []
            premium_adjustment = None
            
            if decision_result.decision == DecisionType.CONDITIONAL:
                conditional_terms = self._generate_conditional_terms(
                    factor_scores, context, risk_analysis
                )
                coverage_modifications = [term.description for term in conditional_terms 
                                        if term.term_type in ["coverage_limit", "exclusion"]]
                
                # Calculate premium adjustment
                premium_adjustment = self._calculate_premium_adjustment(
                    factor_scores, conditional_terms, risk_analysis
                )
            
            # Create recommendation
            recommendation = Recommendation(
                id="",  # Will be set by database
                application_id=str(application.id),
                decision=decision_result.decision,
                confidence=Decimal(str(round(confidence, 3))),
                rationale=rationale,
                conditions=[term.description for term in conditional_terms],
                premium_adjustment=Decimal(str(premium_adjustment)) if premium_adjustment else None,
                coverage_modifications=coverage_modifications,
                created_at=datetime.utcnow()
            )
            
            self.logger.info(f"Recommendation generated: {decision_result.decision.value} "
                           f"with confidence {confidence:.3f}")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            raise
    
    def _create_decision_context(
        self,
        application: Application,
        risk_analysis: RiskAnalysis,
        business_limits_check: Dict[str, Any]
    ) -> DecisionContext:
        """Create decision context from available data"""
        return DecisionContext(
            application_id=str(application.id),
            risk_analysis=risk_analysis,
            business_limits_check=business_limits_check,
            market_conditions=self._get_market_conditions(),
            portfolio_context=self._get_portfolio_context(),
            regulatory_requirements=self._get_regulatory_requirements(),
            underwriter_preferences=self._get_underwriter_preferences()
        )
    
    def _calculate_factor_scores(self, context: DecisionContext) -> List[DecisionFactorScore]:
        """Calculate scores for all decision factors"""
        factor_scores = []
        
        # Risk Score Factor
        risk_score_factor = self._calculate_risk_score_factor(context)
        factor_scores.append(risk_score_factor)
        
        # Financial Strength Factor
        financial_factor = self._calculate_financial_strength_factor(context)
        factor_scores.append(financial_factor)
        
        # Loss History Factor
        loss_history_factor = self._calculate_loss_history_factor(context)
        factor_scores.append(loss_history_factor)
        
        # Catastrophe Exposure Factor
        cat_exposure_factor = self._calculate_catastrophe_exposure_factor(context)
        factor_scores.append(cat_exposure_factor)
        
        # Business Limits Factor
        business_limits_factor = self._calculate_business_limits_factor(context)
        factor_scores.append(business_limits_factor)
        
        # Regulatory Compliance Factor
        regulatory_factor = self._calculate_regulatory_compliance_factor(context)
        factor_scores.append(regulatory_factor)
        
        # Market Conditions Factor
        market_factor = self._calculate_market_conditions_factor(context)
        factor_scores.append(market_factor)
        
        # Portfolio Concentration Factor
        portfolio_factor = self._calculate_portfolio_concentration_factor(context)
        factor_scores.append(portfolio_factor)
        
        return factor_scores    
 
   def _calculate_risk_score_factor(self, context: DecisionContext) -> DecisionFactorScore:
        """Calculate risk score decision factor"""
        risk_analysis = context.risk_analysis
        overall_score = float(risk_analysis.overall_score)
        
        # Convert risk score to decision score (lower risk = higher decision score)
        decision_score = max(0, 100 - overall_score)
        
        # Determine rationale based on risk level
        risk_level = risk_analysis.risk_level
        if risk_level == RiskLevel.LOW:
            rationale = "Low overall risk profile supports favorable decision"
        elif risk_level == RiskLevel.MEDIUM:
            rationale = "Moderate risk profile requires careful consideration"
        elif risk_level == RiskLevel.HIGH:
            rationale = "High risk profile raises significant concerns"
        else:  # CRITICAL
            rationale = "Critical risk level strongly indicates rejection"
        
        return DecisionFactorScore(
            factor=DecisionFactor.RISK_SCORE,
            score=decision_score,
            weight=self.factor_weights[DecisionFactor.RISK_SCORE],
            confidence=float(risk_analysis.confidence),
            rationale=rationale,
            supporting_data={
                "overall_risk_score": overall_score,
                "risk_level": risk_level.value,
                "risk_factors": risk_analysis.factors or {}
            }
        )
    
    def _calculate_financial_strength_factor(self, context: DecisionContext) -> DecisionFactorScore:
        """Calculate financial strength decision factor"""
        analysis_data = context.risk_analysis.analysis_data or {}
        financial_data = analysis_data.get('financial_assessment', {})
        
        financial_score = financial_data.get('financial_score', 50.0)
        credit_risk = financial_data.get('credit_risk_level', 'MEDIUM')
        
        # Higher financial strength = higher decision score
        decision_score = min(100, financial_score * 1.2)  # Scale up slightly
        
        # Adjust based on credit risk level
        if credit_risk == 'LOW':
            decision_score = min(100, decision_score * 1.1)
            rationale = "Strong financial position supports approval"
        elif credit_risk == 'HIGH':
            decision_score = decision_score * 0.8
            rationale = "Weak financial position raises concerns"
        elif credit_risk == 'CRITICAL':
            decision_score = decision_score * 0.6
            rationale = "Critical financial weakness strongly opposes approval"
        else:
            rationale = "Moderate financial strength requires monitoring"
        
        confidence = 0.8 if financial_data else 0.3  # Lower confidence if no data
        
        return DecisionFactorScore(
            factor=DecisionFactor.FINANCIAL_STRENGTH,
            score=decision_score,
            weight=self.factor_weights[DecisionFactor.FINANCIAL_STRENGTH],
            confidence=confidence,
            rationale=rationale,
            supporting_data=financial_data
        )
    
    def _calculate_loss_history_factor(self, context: DecisionContext) -> DecisionFactorScore:
        """Calculate loss history decision factor"""
        analysis_data = context.risk_analysis.analysis_data or {}
        loss_data = analysis_data.get('loss_analysis', {})
        
        loss_frequency = loss_data.get('loss_frequency', 0.0)
        loss_trend = loss_data.get('loss_trend', 'stable')
        volatility = loss_data.get('volatility', 0.0)
        
        # Calculate decision score based on loss characteristics
        base_score = 70.0  # Start with neutral
        
        # Adjust for frequency (lower frequency = higher score)
        if loss_frequency < 0.5:  # Less than 0.5 losses per year
            frequency_adjustment = 20
            frequency_desc = "low loss frequency"
        elif loss_frequency < 1.0:
            frequency_adjustment = 10
            frequency_desc = "moderate loss frequency"
        elif loss_frequency < 2.0:
            frequency_adjustment = -10
            frequency_desc = "high loss frequency"
        else:
            frequency_adjustment = -25
            frequency_desc = "very high loss frequency"
        
        # Adjust for trend
        if loss_trend == 'decreasing':
            trend_adjustment = 15
            trend_desc = "improving loss trend"
        elif loss_trend == 'stable':
            trend_adjustment = 0
            trend_desc = "stable loss pattern"
        else:  # increasing
            trend_adjustment = -20
            trend_desc = "worsening loss trend"
        
        # Adjust for volatility (lower volatility = higher score)
        volatility_adjustment = max(-15, -volatility * 10)
        
        decision_score = max(0, min(100, base_score + frequency_adjustment + 
                                  trend_adjustment + volatility_adjustment))
        
        rationale = f"Loss history shows {frequency_desc} and {trend_desc}"
        
        confidence = 0.9 if loss_data else 0.2
        
        return DecisionFactorScore(
            factor=DecisionFactor.LOSS_HISTORY,
            score=decision_score,
            weight=self.factor_weights[DecisionFactor.LOSS_HISTORY],
            confidence=confidence,
            rationale=rationale,
            supporting_data=loss_data
        )    

    def _calculate_catastrophe_exposure_factor(self, context: DecisionContext) -> DecisionFactorScore:
        """Calculate catastrophe exposure decision factor"""
        analysis_data = context.risk_analysis.analysis_data or {}
        cat_data = analysis_data.get('catastrophe_exposure', {})
        
        cat_score = cat_data.get('overall_score', 5.0)  # 0-10 scale
        risk_zone = cat_data.get('risk_zone', 'moderate')
        
        # Convert cat score to decision score (lower cat exposure = higher decision score)
        decision_score = max(0, 100 - (cat_score * 10))
        
        # Adjust based on risk zone
        zone_adjustments = {
            'very_low': 10,
            'low': 5,
            'moderate': 0,
            'high': -15,
            'very_high': -25,
            'extreme': -40
        }
        
        adjustment = zone_adjustments.get(risk_zone, 0)
        decision_score = max(0, min(100, decision_score + adjustment))
        
        rationale = f"Catastrophe exposure in {risk_zone} risk zone"
        if cat_score > 7:
            rationale += " raises significant concerns"
        elif cat_score < 3:
            rationale += " is manageable"
        else:
            rationale += " requires monitoring"
        
        confidence = 0.8 if cat_data else 0.4
        
        return DecisionFactorScore(
            factor=DecisionFactor.CATASTROPHE_EXPOSURE,
            score=decision_score,
            weight=self.factor_weights[DecisionFactor.CATASTROPHE_EXPOSURE],
            confidence=confidence,
            rationale=rationale,
            supporting_data=cat_data
        )
    
    def _calculate_business_limits_factor(self, context: DecisionContext) -> DecisionFactorScore:
        """Calculate business limits decision factor"""
        limits_check = context.business_limits_check
        
        violations = limits_check.get('violations', [])
        warnings = limits_check.get('warnings', [])
        
        if violations:
            decision_score = 0.0  # Hard rejection for violations
            rationale = f"Business limit violations: {', '.join(violations)}"
        elif warnings:
            decision_score = 40.0  # Conditional approval possible
            rationale = f"Business limit warnings: {', '.join(warnings)}"
        else:
            decision_score = 100.0  # Full approval from limits perspective
            rationale = "All business limits satisfied"
        
        confidence = 0.95  # High confidence in business rules
        
        return DecisionFactorScore(
            factor=DecisionFactor.BUSINESS_LIMITS,
            score=decision_score,
            weight=self.factor_weights[DecisionFactor.BUSINESS_LIMITS],
            confidence=confidence,
            rationale=rationale,
            supporting_data=limits_check
        )
    
    def _calculate_regulatory_compliance_factor(self, context: DecisionContext) -> DecisionFactorScore:
        """Calculate regulatory compliance decision factor"""
        requirements = context.regulatory_requirements
        
        # Simplified compliance check (in production, this would be more sophisticated)
        compliance_score = 85.0  # Assume generally compliant
        
        if "high_risk_jurisdiction" in requirements:
            compliance_score -= 20
            rationale = "High-risk jurisdiction requires additional scrutiny"
        elif "sanctions_check_required" in requirements:
            compliance_score -= 10
            rationale = "Sanctions screening required"
        else:
            rationale = "Standard regulatory requirements apply"
        
        decision_score = max(0, compliance_score)
        confidence = 0.7  # Moderate confidence in regulatory assessment
        
        return DecisionFactorScore(
            factor=DecisionFactor.REGULATORY_COMPLIANCE,
            score=decision_score,
            weight=self.factor_weights[DecisionFactor.REGULATORY_COMPLIANCE],
            confidence=confidence,
            rationale=rationale,
            supporting_data={"requirements": requirements}
        )
    
    def _calculate_market_conditions_factor(self, context: DecisionContext) -> DecisionFactorScore:
        """Calculate market conditions decision factor"""
        market_data = context.market_conditions
        
        market_sentiment = market_data.get('sentiment', 'neutral')
        capacity_utilization = market_data.get('capacity_utilization', 0.7)
        
        # Base score from market sentiment
        sentiment_scores = {
            'very_positive': 90,
            'positive': 80,
            'neutral': 70,
            'negative': 50,
            'very_negative': 30
        }
        
        base_score = sentiment_scores.get(market_sentiment, 70)
        
        # Adjust for capacity utilization
        if capacity_utilization > 0.9:
            capacity_adjustment = -10  # High utilization = more selective
        elif capacity_utilization < 0.5:
            capacity_adjustment = 10   # Low utilization = more aggressive
        else:
            capacity_adjustment = 0
        
        decision_score = max(0, min(100, base_score + capacity_adjustment))
        
        rationale = f"Market conditions are {market_sentiment} with {capacity_utilization:.0%} capacity utilization"
        confidence = 0.6  # Moderate confidence in market assessment
        
        return DecisionFactorScore(
            factor=DecisionFactor.MARKET_CONDITIONS,
            score=decision_score,
            weight=self.factor_weights[DecisionFactor.MARKET_CONDITIONS],
            confidence=confidence,
            rationale=rationale,
            supporting_data=market_data
        )    

    def _calculate_portfolio_concentration_factor(self, context: DecisionContext) -> DecisionFactorScore:
        """Calculate portfolio concentration decision factor"""
        portfolio_data = context.portfolio_context
        
        concentration_risk = portfolio_data.get('concentration_risk', 'low')
        diversification_score = portfolio_data.get('diversification_score', 0.8)
        
        # Score based on concentration risk
        concentration_scores = {
            'low': 90,
            'moderate': 75,
            'high': 50,
            'very_high': 25
        }
        
        base_score = concentration_scores.get(concentration_risk, 75)
        
        # Adjust for diversification
        diversification_adjustment = (diversification_score - 0.5) * 20
        
        decision_score = max(0, min(100, base_score + diversification_adjustment))
        
        rationale = f"Portfolio shows {concentration_risk} concentration risk"
        confidence = 0.5  # Lower confidence in portfolio assessment
        
        return DecisionFactorScore(
            factor=DecisionFactor.PORTFOLIO_CONCENTRATION,
            score=decision_score,
            weight=self.factor_weights[DecisionFactor.PORTFOLIO_CONCENTRATION],
            confidence=confidence,
            rationale=rationale,
            supporting_data=portfolio_data
        )
    
    def _make_ensemble_decision(
        self, 
        factor_scores: List[DecisionFactorScore], 
        context: DecisionContext
    ) -> DecisionResult:
        """Make decision using ensemble approach"""
        
        # Calculate weighted overall score
        total_weighted_score = sum(
            factor.score * factor.weight for factor in factor_scores
        )
        
        # Determine primary decision
        if total_weighted_score >= self.decision_thresholds["approve_threshold"]:
            decision = DecisionType.APPROVE
        elif total_weighted_score <= self.decision_thresholds["reject_threshold"]:
            decision = DecisionType.REJECT
        else:
            decision = DecisionType.CONDITIONAL
        
        # Check for hard rejection criteria
        for factor in factor_scores:
            if (factor.factor == DecisionFactor.BUSINESS_LIMITS and 
                factor.score == 0.0):
                decision = DecisionType.REJECT
                break
        
        # Generate explanation
        explanation = self._generate_decision_explanation(factor_scores, decision)
        
        return DecisionResult(
            decision=decision,
            confidence=0.0,  # Will be calculated separately
            rationale="",    # Will be generated separately
            factor_scores=factor_scores,
            conditional_terms=[],  # Will be generated if needed
            explanation=explanation,
            premium_adjustment=None,
            coverage_modifications=[],
            generated_at=datetime.utcnow(),
            model_versions={
                "decision_model": self.decision_model_name,
                "rationale_model": self.rationale_model_name,
                "confidence_model": self.confidence_model_name
            }
        )
    
    def _generate_rationale(
        self,
        decision_result: DecisionResult,
        factor_scores: List[DecisionFactorScore],
        context: DecisionContext
    ) -> str:
        """Generate detailed rationale using AI models"""
        
        # Create input text for rationale generation
        decision_text = f"Decision: {decision_result.decision.value}\n"
        
        # Add key factors
        key_factors = sorted(factor_scores, key=lambda x: x.score * x.weight, reverse=True)[:3]
        factors_text = "Key factors: " + "; ".join([
            f"{factor.factor.value}: {factor.rationale}" for factor in key_factors
        ])
        
        input_text = decision_text + factors_text
        
        # Generate rationale using BART model if available
        if self.rationale_generator:
            try:
                prompt = f"Explain the reinsurance decision: {input_text}"
                generated = self.rationale_generator(
                    prompt,
                    max_length=200,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.6
                )
                
                if generated and len(generated) > 0:
                    rationale = generated[0]['generated_text']
                    # Clean up the generated text
                    rationale = self._clean_generated_text(rationale, prompt)
                else:
                    rationale = self._generate_fallback_rationale(decision_result, factor_scores)
                    
            except Exception as e:
                self.logger.warning(f"Error generating AI rationale: {e}")
                rationale = self._generate_fallback_rationale(decision_result, factor_scores)
        else:
            rationale = self._generate_fallback_rationale(decision_result, factor_scores)
        
        return rationale 
   
    def _generate_fallback_rationale(
        self,
        decision_result: DecisionResult,
        factor_scores: List[DecisionFactorScore]
    ) -> str:
        """Generate fallback rationale using rule-based approach"""
        
        decision = decision_result.decision
        
        # Get top positive and negative factors
        positive_factors = [f for f in factor_scores if f.score >= 70]
        negative_factors = [f for f in factor_scores if f.score < 50]
        
        if decision == DecisionType.APPROVE:
            rationale = "Recommendation: APPROVE. "
            if positive_factors:
                rationale += "Strong supporting factors include: " + "; ".join([
                    f.rationale for f in positive_factors[:2]
                ]) + ". "
            if negative_factors:
                rationale += "Minor concerns: " + "; ".join([
                    f.rationale for f in negative_factors[:1]
                ]) + ". "
            rationale += "Overall risk profile is acceptable for standard terms."
            
        elif decision == DecisionType.REJECT:
            rationale = "Recommendation: REJECT. "
            if negative_factors:
                rationale += "Significant concerns include: " + "; ".join([
                    f.rationale for f in negative_factors[:2]
                ]) + ". "
            rationale += "Risk profile exceeds acceptable thresholds."
            
        else:  # CONDITIONAL
            rationale = "Recommendation: CONDITIONAL APPROVAL. "
            rationale += "Mixed risk profile requires specific terms and conditions. "
            if positive_factors:
                rationale += "Supporting factors: " + positive_factors[0].rationale + ". "
            if negative_factors:
                rationale += "Areas of concern: " + negative_factors[0].rationale + ". "
            rationale += "Conditional terms can mitigate identified risks."
        
        return rationale
    
    def _calculate_decision_confidence(
        self,
        decision_result: DecisionResult,
        factor_scores: List[DecisionFactorScore]
    ) -> float:
        """Calculate confidence in the decision"""
        
        # Base confidence from factor confidences
        weighted_confidence = sum(
            factor.confidence * factor.weight for factor in factor_scores
        )
        
        # Adjust based on decision clarity
        total_score = sum(factor.score * factor.weight for factor in factor_scores)
        
        if decision_result.decision == DecisionType.APPROVE:
            if total_score >= 80:
                clarity_bonus = 0.1
            elif total_score >= 70:
                clarity_bonus = 0.05
            else:
                clarity_bonus = -0.1  # Borderline approval
        elif decision_result.decision == DecisionType.REJECT:
            if total_score <= 20:
                clarity_bonus = 0.1
            elif total_score <= 30:
                clarity_bonus = 0.05
            else:
                clarity_bonus = -0.1  # Borderline rejection
        else:  # CONDITIONAL
            clarity_bonus = -0.05  # Conditional decisions are inherently less certain
        
        # Adjust for factor agreement
        factor_variance = np.var([f.score for f in factor_scores])
        if factor_variance < 100:  # Low variance = high agreement
            agreement_bonus = 0.1
        elif factor_variance > 400:  # High variance = low agreement
            agreement_bonus = -0.1
        else:
            agreement_bonus = 0.0
        
        final_confidence = weighted_confidence + clarity_bonus + agreement_bonus
        return max(0.0, min(1.0, final_confidence))
    
    def _generate_conditional_terms(
        self,
        factor_scores: List[DecisionFactorScore],
        context: DecisionContext,
        risk_analysis: RiskAnalysis
    ) -> List[ConditionalTerm]:
        """Generate conditional approval terms"""
        
        terms = []
        
        # Analyze factors to determine appropriate terms
        for factor in factor_scores:
            if factor.score < 60:  # Factors needing mitigation
                
                if factor.factor == DecisionFactor.CATASTROPHE_EXPOSURE:
                    terms.append(ConditionalTerm(
                        term_type="coverage_limit",
                        description="Catastrophe exposure limit of 80% of total sum insured",
                        value=0.8,
                        rationale="Mitigate high catastrophe exposure risk",
                        impact_on_risk=-0.2
                    ))
                    
                elif factor.factor == DecisionFactor.FINANCIAL_STRENGTH:
                    terms.append(ConditionalTerm(
                        term_type="premium_adjustment",
                        description="Premium loading of 15% for financial risk",
                        value=0.15,
                        rationale="Compensate for elevated financial risk",
                        impact_on_risk=-0.1
                    ))
                    
                elif factor.factor == DecisionFactor.LOSS_HISTORY:
                    terms.append(ConditionalTerm(
                        term_type="deductible",
                        description="Increased deductible to 5% of sum insured",
                        value=0.05,
                        rationale="Mitigate loss frequency concerns",
                        impact_on_risk=-0.15
                    ))
        
        # Add standard conditional terms based on risk level
        risk_level = risk_analysis.risk_level
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            terms.append(ConditionalTerm(
                term_type="exclusion",
                description="Exclude coverage for acts of terrorism",
                value=None,
                rationale="Standard exclusion for high-risk cases",
                impact_on_risk=-0.1
            ))
        
        return terms    

    def _calculate_premium_adjustment(
        self,
        factor_scores: List[DecisionFactorScore],
        conditional_terms: List[ConditionalTerm],
        risk_analysis: RiskAnalysis
    ) -> Optional[float]:
        """Calculate premium adjustment for conditional approval"""
        
        base_adjustment = 0.0
        
        # Add adjustments from conditional terms
        for term in conditional_terms:
            if term.term_type == "premium_adjustment" and term.value:
                base_adjustment += float(term.value)
        
        # Add risk-based adjustment
        overall_score = float(risk_analysis.overall_score)
        if overall_score > 60:  # High risk
            risk_adjustment = (overall_score - 50) * 0.01  # 1% per risk point above 50
            base_adjustment += risk_adjustment
        
        # Cap the adjustment
        final_adjustment = min(0.5, max(0.0, base_adjustment))  # 0-50% range
        
        return final_adjustment if final_adjustment > 0.01 else None
    
    def _generate_decision_explanation(
        self,
        factor_scores: List[DecisionFactorScore],
        decision: DecisionType
    ) -> DecisionExplanation:
        """Generate detailed decision explanation"""
        
        # Sort factors by impact (score * weight)
        sorted_factors = sorted(
            factor_scores, 
            key=lambda x: x.score * x.weight, 
            reverse=True
        )
        
        # Primary reasons (top factors)
        primary_reasons = [f.rationale for f in sorted_factors[:2]]
        
        # Supporting factors
        supporting_factors = [f.rationale for f in sorted_factors[2:4] if f.score >= 60]
        
        # Risk mitigation (factors that help)
        risk_mitigation = [f.rationale for f in sorted_factors if f.score >= 80]
        
        # Concerns addressed
        concerns_addressed = [f.rationale for f in sorted_factors if f.score < 50]
        
        # Alternative considerations
        alternative_considerations = [
            "Market conditions could change assessment",
            "Additional information might alter recommendation",
            "Portfolio context may influence final decision"
        ]
        
        # Confidence factors
        high_conf_factors = [f for f in factor_scores if f.confidence >= 0.8]
        confidence_factors = [
            f"High confidence in {f.factor.value}" for f in high_conf_factors
        ]
        
        return DecisionExplanation(
            primary_reasons=primary_reasons,
            supporting_factors=supporting_factors,
            risk_mitigation=risk_mitigation,
            concerns_addressed=concerns_addressed,
            alternative_considerations=alternative_considerations,
            confidence_factors=confidence_factors
        )
    
    def _clean_generated_text(self, generated_text: str, prompt: str) -> str:
        """Clean up AI-generated text"""
        # Remove the prompt from the generated text
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()
        
        # Remove common artifacts
        generated_text = re.sub(r'^[:\-\s]+', '', generated_text)
        generated_text = re.sub(r'\s+', ' ', generated_text)
        
        # Ensure proper capitalization
        if generated_text and not generated_text[0].isupper():
            generated_text = generated_text[0].upper() + generated_text[1:]
        
        # Ensure proper ending
        if generated_text and not generated_text.endswith('.'):
            generated_text += '.'
        
        return generated_text.strip()
    
    # Helper methods for context data
    
    def _get_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions (simplified)"""
        return {
            "sentiment": "neutral",
            "capacity_utilization": 0.75,
            "pricing_trend": "stable",
            "competition_level": "moderate"
        }
    
    def _get_portfolio_context(self) -> Dict[str, Any]:
        """Get portfolio context (simplified)"""
        return {
            "concentration_risk": "moderate",
            "diversification_score": 0.7,
            "total_exposure": 1000000000,  # $1B
            "sector_concentration": 0.3
        }
    
    def _get_regulatory_requirements(self) -> List[str]:
        """Get regulatory requirements (simplified)"""
        return [
            "standard_compliance",
            "kyc_verification",
            "sanctions_screening"
        ]
    
    def _get_underwriter_preferences(self) -> Dict[str, Any]:
        """Get underwriter preferences (simplified)"""
        return {
            "risk_appetite": "moderate",
            "preferred_sectors": ["technology", "healthcare"],
            "geographic_preferences": ["north_america", "europe"],
            "minimum_premium": 50000
        }


# Export the agent class
__all__ = ['DecisionEngineAgent']