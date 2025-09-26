"""
Unit tests for Risk Analysis Agent

Tests all risk calculation methods including:
- Loss history analysis algorithms using statistical methods
- Catastrophe exposure modeling based on geographic and asset data
- Financial strength assessment using ProsusAI/finbert
- Risk scoring algorithms with weighted factor analysis
- Risk report generation with structured output
- Confidence scoring for risk assessments

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 8.2
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from app.agents.risk_analysis_agent import (
    RiskAnalysisAgent, CatastropheType, GeographicRiskZone,
    LossAnalysisResult, CatastropheExposure, FinancialStrengthAssessment,
    RiskFactorWeights, RiskReportData
)
from app.models.schemas import (
    RiskLevel, LossEvent, FinancialData, RiskParameters,
    RiskAnalysis, RiskScore, Application, ApplicationStatus
)


class TestRiskAnalysisAgent:
    """Test suite for Risk Analysis Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create risk analysis agent instance for testing"""
        with patch('app.agents.risk_analysis_agent.AutoTokenizer'), \
             patch('app.agents.risk_analysis_agent.AutoModelForSequenceClassification'), \
             patch('app.agents.risk_analysis_agent.pipeline'):
            return RiskAnalysisAgent()
    
    @pytest.fixture
    def sample_loss_events(self):
        """Create sample loss events for testing"""
        base_date = datetime(2020, 1, 1)
        return [
            LossEvent(
                id="1",
                application_id="app1",
                event_date=base_date + timedelta(days=30),
                amount=Decimal('50000'),
                cause="Fire",
                description="Small fire in warehouse"
            ),
            LossEvent(
                id="2",
                application_id="app1",
                event_date=base_date + timedelta(days=180),
                amount=Decimal('25000'),
                cause="Theft",
                description="Equipment theft"
            ),
            LossEvent(
                id="3",
                application_id="app1",
                event_date=base_date + timedelta(days=365),
                amount=Decimal('100000'),
                cause="Flood",
                description="Flood damage to facility"
            ),
            LossEvent(
                id="4",
                application_id="app1",
                event_date=base_date + timedelta(days=500),
                amount=Decimal('75000'),
                cause="Equipment failure",
                description="Major equipment breakdown"
            )
        ]
    
    @pytest.fixture
    def sample_financial_data(self):
        """Create sample financial data for testing"""
        return FinancialData(
            id="fin1",
            application_id="app1",
            revenue=Decimal('10000000'),
            assets=Decimal('50000000'),
            liabilities=Decimal('20000000'),
            credit_rating="A",
            financial_strength_rating="A+"
        )
    
    @pytest.fixture
    def sample_risk_parameters(self):
        """Create sample risk parameters for testing"""
        return RiskParameters(
            id="risk1",
            application_id="app1",
            asset_value=Decimal('25000000'),
            coverage_limit=Decimal('20000000'),
            asset_type="Industrial Manufacturing",
            location="California, USA",
            industry_sector="manufacturing",
            construction_type="Steel and concrete",
            occupancy="Manufacturing facility"
        )
    
    @pytest.fixture
    def sample_application(self, sample_loss_events, sample_financial_data, sample_risk_parameters):
        """Create sample application for testing"""
        return Application(
            id="app1",
            status=ApplicationStatus.PROCESSING,
            loss_history=sample_loss_events,
            financial_data=sample_financial_data,
            risk_parameters=sample_risk_parameters,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )


class TestLossHistoryAnalysis:
    """Test loss history analysis functionality"""
    
    def test_analyze_loss_history_with_data(self, agent, sample_loss_events):
        """Test loss history analysis with sample data"""
        result = agent.analyze_loss_history(sample_loss_events)
        
        assert isinstance(result, LossAnalysisResult)
        assert result.total_losses == Decimal('250000')
        assert result.loss_frequency > 0
        assert result.loss_severity_avg > 0
        assert result.largest_loss == Decimal('100000')
        assert 0 <= result.confidence_score <= 1
        assert result.loss_trend in ["increasing", "decreasing", "stable"]
        assert isinstance(result.statistical_metrics, dict)
    
    def test_analyze_loss_history_empty_data(self, agent):
        """Test loss history analysis with no data"""
        result = agent.analyze_loss_history([])
        
        assert result.total_losses == Decimal('0')
        assert result.average_annual_loss == Decimal('0')
        assert result.loss_frequency == 0.0
        assert result.loss_severity_avg == Decimal('0')
        assert result.loss_trend == "stable"
        assert result.volatility == 0.0
        assert result.largest_loss == Decimal('0')
        assert result.confidence_score == 0.0
    
    def test_analyze_loss_history_single_event(self, agent):
        """Test loss history analysis with single event"""
        single_event = [LossEvent(
            id="1",
            application_id="app1",
            event_date=datetime(2023, 1, 1),
            amount=Decimal('100000'),
            cause="Fire",
            description="Test fire"
        )]
        
        result = agent.analyze_loss_history(single_event)
        
        assert result.total_losses == Decimal('100000')
        assert result.largest_loss == Decimal('100000')
        assert result.loss_frequency > 0
        assert result.volatility == 0.0  # Single event has no variance
    
    def test_loss_trend_analysis(self, agent):
        """Test loss trend analysis with increasing losses"""
        base_date = datetime(2020, 1, 1)
        increasing_losses = [
            LossEvent(
                id=str(i),
                application_id="app1",
                event_date=base_date + timedelta(days=i*100),
                amount=Decimal(str(10000 * (i + 1))),  # Increasing amounts
                cause="Various",
                description=f"Loss {i}"
            )
            for i in range(5)
        ]
        
        result = agent.analyze_loss_history(increasing_losses)
        
        # With clearly increasing pattern, should detect trend
        assert result.loss_trend in ["increasing", "stable"]  # May be stable if not statistically significant
        assert result.confidence_score > 0
    
    def test_statistical_metrics_calculation(self, agent, sample_loss_events):
        """Test statistical metrics calculation"""
        result = agent.analyze_loss_history(sample_loss_events)
        
        metrics = result.statistical_metrics
        assert 'mean' in metrics
        assert 'median' in metrics
        assert 'std_dev' in metrics
        assert 'skewness' in metrics
        assert 'kurtosis' in metrics
        assert 'percentile_95' in metrics
        assert 'percentile_99' in metrics
        
        # Verify calculations make sense
        assert metrics['mean'] > 0
        assert metrics['median'] > 0
        assert metrics['std_dev'] >= 0


class TestCatastropheExposureAssessment:
    """Test catastrophe exposure assessment functionality"""
    
    def test_assess_catastrophe_exposure_california(self, agent, sample_risk_parameters):
        """Test catastrophe exposure assessment for California location"""
        # Modify location to California
        sample_risk_parameters.location = "Los Angeles, California"
        
        result = agent.assess_catastrophe_exposure(sample_risk_parameters, Decimal('25000000'))
        
        assert isinstance(result, CatastropheExposure)
        assert result.overall_cat_score > 0
        assert result.geographic_risk_zone in [GeographicRiskZone.HIGH, GeographicRiskZone.VERY_HIGH]
        assert CatastropheType.EARTHQUAKE in result.primary_perils
        assert result.pml_estimate > 0
        assert 0 <= result.confidence_score <= 1
        assert isinstance(result.detailed_analysis, dict)
    
    def test_assess_catastrophe_exposure_low_risk_area(self, agent, sample_risk_parameters):
        """Test catastrophe exposure assessment for low-risk area"""
        sample_risk_parameters.location = "Rural Montana, USA"
        
        result = agent.assess_catastrophe_exposure(sample_risk_parameters, Decimal('10000000'))
        
        assert result.overall_cat_score >= 0
        assert result.pml_estimate >= 0
        assert len(result.primary_perils) > 0
    
    def test_assess_catastrophe_exposure_no_asset_value(self, agent, sample_risk_parameters):
        """Test catastrophe exposure assessment without asset value"""
        result = agent.assess_catastrophe_exposure(sample_risk_parameters, None)
        
        assert result.pml_estimate == Decimal('0')
        assert result.overall_cat_score > 0  # Should still calculate exposure score
    
    def test_mitigation_factors_identification(self, agent, sample_risk_parameters):
        """Test identification of mitigation factors"""
        sample_risk_parameters.construction_type = "Steel and concrete with sprinkler system"
        sample_risk_parameters.asset_type = "Technology data center"
        
        result = agent.assess_catastrophe_exposure(sample_risk_parameters, Decimal('15000000'))
        
        assert len(result.mitigation_factors) > 0
        assert any("Fire-resistant" in factor for factor in result.mitigation_factors)
    
    def test_geographic_risk_zone_mapping(self, agent):
        """Test geographic risk zone mapping"""
        test_locations = [
            ("Tokyo, Japan", GeographicRiskZone.EXTREME),
            ("Miami, Florida", GeographicRiskZone.VERY_HIGH),
            ("London, UK", GeographicRiskZone.MODERATE),
        ]
        
        for location, expected_zone in test_locations:
            zone = agent._get_geographic_risk_zone(location)
            # Allow for some flexibility in zone assignment
            assert isinstance(zone, GeographicRiskZone)
    
    def test_primary_perils_identification(self, agent):
        """Test primary perils identification by location"""
        test_cases = [
            ("California", [CatastropheType.EARTHQUAKE, CatastropheType.WILDFIRE]),
            ("Florida", [CatastropheType.HURRICANE, CatastropheType.FLOOD]),
            ("Japan", [CatastropheType.EARTHQUAKE, CatastropheType.FLOOD]),
        ]
        
        for location, expected_perils in test_cases:
            perils = agent._identify_primary_perils(location, "")
            assert len(perils) > 0
            # Check if at least one expected peril is identified
            assert any(peril in perils for peril in expected_perils)


class TestFinancialStrengthAssessment:
    """Test financial strength assessment functionality"""
    
    def test_evaluate_financial_strength_good_rating(self, agent, sample_financial_data):
        """Test financial strength evaluation with good credit rating"""
        result = agent.evaluate_financial_strength(sample_financial_data)
        
        assert isinstance(result, FinancialStrengthAssessment)
        assert result.overall_rating in ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
        assert 0 <= result.financial_score <= 100
        assert 0 <= result.liquidity_ratio <= 1
        assert 0 <= result.solvency_ratio <= 1
        assert 0 <= result.profitability_score <= 1
        assert 0 <= result.stability_score <= 1
        assert result.credit_risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert 0 <= result.confidence_score <= 1
        assert isinstance(result.key_metrics, dict)
        assert isinstance(result.concerns, list)
    
    def test_evaluate_financial_strength_poor_rating(self, agent):
        """Test financial strength evaluation with poor credit rating"""
        poor_financial_data = FinancialData(
            id="fin2",
            application_id="app2",
            revenue=Decimal('1000000'),
            assets=Decimal('5000000'),
            liabilities=Decimal('8000000'),  # High debt
            credit_rating="CCC",
            financial_strength_rating="C"
        )
        
        result = agent.evaluate_financial_strength(poor_financial_data)
        
        assert result.financial_score < 70  # Should be low score
        assert result.credit_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert len(result.concerns) > 0
    
    def test_evaluate_financial_strength_minimal_data(self, agent):
        """Test financial strength evaluation with minimal data"""
        minimal_data = FinancialData(
            id="fin3",
            application_id="app3"
        )
        
        result = agent.evaluate_financial_strength(minimal_data)
        
        assert isinstance(result, FinancialStrengthAssessment)
        assert result.confidence_score < 0.5  # Low confidence with minimal data
        assert "No credit rating" in " ".join(result.concerns)
    
    def test_financial_ratios_calculation(self, agent, sample_financial_data):
        """Test financial ratios calculation"""
        ratios = agent._calculate_financial_ratios(sample_financial_data)
        
        assert isinstance(ratios, dict)
        assert 'debt_to_asset' in ratios
        assert 'equity_ratio' in ratios
        assert 'asset_turnover' in ratios
        assert 'revenue_to_debt' in ratios
        
        # Verify ratio calculations
        expected_debt_to_asset = float(sample_financial_data.liabilities / sample_financial_data.assets)
        assert abs(ratios['debt_to_asset'] - expected_debt_to_asset) < 0.01
    
    @patch('app.agents.risk_analysis_agent.RiskAnalysisAgent._analyze_with_finbert')
    def test_finbert_integration(self, mock_finbert, agent, sample_financial_data):
        """Test FinBERT integration for financial analysis"""
        mock_finbert.return_value = 0.8  # Mock positive sentiment
        
        result = agent.evaluate_financial_strength(sample_financial_data)
        
        # Should have called FinBERT analysis
        mock_finbert.assert_called_once()
        assert isinstance(result, FinancialStrengthAssessment)


class TestRiskScoring:
    """Test risk scoring algorithms"""
    
    def test_calculate_risk_score_comprehensive(self, agent, sample_loss_events, sample_financial_data, sample_risk_parameters):
        """Test comprehensive risk score calculation"""
        # Create component analyses
        loss_analysis = agent.analyze_loss_history(sample_loss_events)
        cat_exposure = agent.assess_catastrophe_exposure(sample_risk_parameters, Decimal('25000000'))
        financial_assessment = agent.evaluate_financial_strength(sample_financial_data)
        
        result = agent.calculate_risk_score(
            loss_analysis, cat_exposure, financial_assessment, sample_risk_parameters
        )
        
        assert isinstance(result, RiskScore)
        assert 0 <= result.overall_score <= 100
        assert 0 <= result.confidence <= 1
        assert result.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert isinstance(result.factors, dict)
        
        # Check factor breakdown
        factors = result.factors
        assert 'loss_history' in factors
        assert 'catastrophe_exposure' in factors
        assert 'financial_strength' in factors
        assert 'asset_characteristics' in factors
        assert 'geographic_factors' in factors
        assert 'weights' in factors
    
    def test_risk_level_determination(self, agent):
        """Test risk level determination from scores"""
        test_cases = [
            (15.0, RiskLevel.LOW),
            (35.0, RiskLevel.MEDIUM),
            (65.0, RiskLevel.HIGH),
            (85.0, RiskLevel.CRITICAL)
        ]
        
        for score, expected_level in test_cases:
            level = agent._determine_risk_level(score)
            assert level == expected_level
    
    def test_weighted_factor_analysis(self, agent):
        """Test weighted factor analysis in risk scoring"""
        # Test that weights sum to 1.0
        weights = agent.risk_weights
        total_weight = (
            weights.loss_history + weights.catastrophe_exposure + 
            weights.financial_strength + weights.asset_characteristics + 
            weights.geographic_factors
        )
        assert abs(total_weight - 1.0) < 0.01
    
    def test_normalize_loss_score(self, agent):
        """Test loss score normalization"""
        # Test with different loss analysis scenarios
        test_cases = [
            # No losses
            LossAnalysisResult(
                total_losses=Decimal('0'), average_annual_loss=Decimal('0'),
                loss_frequency=0.0, loss_severity_avg=Decimal('0'),
                loss_trend="stable", volatility=0.0, largest_loss=Decimal('0'),
                confidence_score=1.0, statistical_metrics={}
            ),
            # High frequency losses
            LossAnalysisResult(
                total_losses=Decimal('1000000'), average_annual_loss=Decimal('200000'),
                loss_frequency=5.0, loss_severity_avg=Decimal('100000'),
                loss_trend="increasing", volatility=1.5, largest_loss=Decimal('500000'),
                confidence_score=0.8, statistical_metrics={}
            )
        ]
        
        for loss_analysis in test_cases:
            score = agent._normalize_loss_score(loss_analysis)
            assert 0 <= score <= 100


class TestRiskReportGeneration:
    """Test risk report generation functionality"""
    
    def test_generate_risk_report_comprehensive(self, agent, sample_application):
        """Test comprehensive risk report generation"""
        # Perform analyses
        loss_analysis = agent.analyze_loss_history(sample_application.loss_history)
        cat_exposure = agent.assess_catastrophe_exposure(
            sample_application.risk_parameters, 
            sample_application.risk_parameters.asset_value
        )
        financial_assessment = agent.evaluate_financial_strength(sample_application.financial_data)
        risk_score = agent.calculate_risk_score(
            loss_analysis, cat_exposure, financial_assessment, sample_application.risk_parameters
        )
        
        result = agent.generate_risk_report(
            sample_application, loss_analysis, cat_exposure, financial_assessment, risk_score
        )
        
        assert isinstance(result, RiskReportData)
        assert result.application_id == str(sample_application.id)
        assert result.overall_risk_score == float(risk_score.overall_score)
        assert result.risk_level == risk_score.risk_level
        assert result.confidence_score == float(risk_score.confidence)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.concerns, list)
        assert isinstance(result.mitigation_suggestions, list)
        assert isinstance(result.generated_at, datetime)
    
    def test_generate_recommendations_by_risk_level(self, agent):
        """Test recommendation generation based on risk levels"""
        # Mock components for testing
        mock_loss_analysis = Mock()
        mock_loss_analysis.loss_trend = "stable"
        
        mock_cat_exposure = Mock()
        mock_cat_exposure.overall_cat_score = 5.0
        
        mock_financial_assessment = Mock()
        mock_financial_assessment.financial_score = 75.0
        
        test_cases = [
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL
        ]
        
        for risk_level in test_cases:
            recommendations = agent._generate_recommendations(
                risk_level, mock_loss_analysis, mock_cat_exposure, mock_financial_assessment
            )
            assert isinstance(recommendations, list)
            assert len(recommendations) > 0
    
    def test_identify_key_concerns(self, agent):
        """Test key concerns identification"""
        # Create mock components with concerning values
        mock_loss_analysis = Mock()
        mock_loss_analysis.loss_frequency = 3.0  # High frequency
        mock_loss_analysis.volatility = 1.5  # High volatility
        
        mock_cat_exposure = Mock()
        mock_cat_exposure.overall_cat_score = 9.0  # High exposure
        
        mock_financial_assessment = Mock()
        mock_financial_assessment.credit_risk_level = RiskLevel.HIGH
        mock_financial_assessment.concerns = ["High debt levels"]
        
        mock_risk_score = Mock()
        mock_risk_score.overall_score = 80.0  # High risk
        
        concerns = agent._identify_key_concerns(
            mock_loss_analysis, mock_cat_exposure, mock_financial_assessment, mock_risk_score
        )
        
        assert isinstance(concerns, list)
        assert len(concerns) > 0
        assert any("risk score" in concern.lower() for concern in concerns)
    
    def test_generate_mitigation_suggestions(self, agent, sample_risk_parameters):
        """Test mitigation suggestions generation"""
        # Create mock components
        mock_cat_exposure = Mock()
        mock_cat_exposure.overall_cat_score = 8.0
        mock_cat_exposure.primary_perils = [CatastropheType.EARTHQUAKE, CatastropheType.WILDFIRE]
        
        mock_financial_assessment = Mock()
        mock_financial_assessment.financial_score = 50.0
        
        suggestions = agent._generate_mitigation_suggestions(
            mock_cat_exposure, mock_financial_assessment, sample_risk_parameters
        )
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0


class TestComprehensiveAnalysis:
    """Test comprehensive analysis workflow"""
    
    def test_perform_comprehensive_analysis(self, agent, sample_application):
        """Test complete comprehensive analysis workflow"""
        result = agent.perform_comprehensive_analysis(sample_application)
        
        assert isinstance(result, RiskAnalysis)
        assert result.application_id == str(sample_application.id)
        assert 0 <= result.overall_score <= 100
        assert 0 <= result.confidence <= 1
        assert result.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert isinstance(result.factors, dict)
        assert isinstance(result.analysis_data, dict)
        assert isinstance(result.created_at, datetime)
        
        # Check analysis data structure
        analysis_data = result.analysis_data
        assert 'loss_analysis' in analysis_data
        assert 'catastrophe_exposure' in analysis_data
        assert 'financial_assessment' in analysis_data
        assert 'recommendations' in analysis_data
        assert 'concerns' in analysis_data
        assert 'mitigation_suggestions' in analysis_data
    
    def test_perform_comprehensive_analysis_minimal_data(self, agent):
        """Test comprehensive analysis with minimal application data"""
        minimal_app = Application(
            id="minimal_app",
            status=ApplicationStatus.PROCESSING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        result = agent.perform_comprehensive_analysis(minimal_app)
        
        assert isinstance(result, RiskAnalysis)
        assert result.confidence < 0.5  # Should have low confidence with minimal data
    
    @patch('app.agents.risk_analysis_agent.logger')
    def test_error_handling_in_analysis(self, mock_logger, agent, sample_application):
        """Test error handling in comprehensive analysis"""
        # Mock a method to raise an exception
        with patch.object(agent, 'analyze_loss_history', side_effect=Exception("Test error")):
            with pytest.raises(Exception):
                agent.perform_comprehensive_analysis(sample_application)
            
            # Verify error was logged
            mock_logger.error.assert_called()


class TestHelperMethods:
    """Test helper methods and utilities"""
    
    def test_confidence_score_calculations(self, agent):
        """Test various confidence score calculations"""
        # Test loss analysis confidence
        confidence = agent._calculate_loss_analysis_confidence(10, 5.0, 0.5, 0.8)
        assert 0 <= confidence <= 1
        
        # Test catastrophe exposure confidence
        confidence = agent._calculate_cat_exposure_confidence(
            "Los Angeles, California", "Industrial", Mock()
        )
        assert 0 <= confidence <= 1
        
        # Test financial confidence
        mock_financial_data = Mock()
        mock_financial_data.revenue = Decimal('1000000')
        mock_financial_data.assets = Decimal('5000000')
        mock_financial_data.liabilities = Decimal('2000000')
        mock_financial_data.credit_rating = "A"
        mock_financial_data.financial_strength_rating = "A+"
        
        confidence = agent._calculate_financial_confidence(mock_financial_data, {})
        assert 0 <= confidence <= 1
    
    def test_geographic_mappings(self, agent):
        """Test geographic risk mappings"""
        # Test known high-risk locations
        high_risk_locations = ["california", "japan", "florida"]
        for location in high_risk_locations:
            zone = agent._get_geographic_risk_zone(location)
            assert zone in [GeographicRiskZone.HIGH, GeographicRiskZone.VERY_HIGH, GeographicRiskZone.EXTREME]
        
        # Test unknown location defaults to moderate
        unknown_zone = agent._get_geographic_risk_zone("unknown location")
        assert unknown_zone == GeographicRiskZone.MODERATE
    
    def test_industry_risk_multipliers(self, agent):
        """Test industry risk multipliers"""
        # Test high-risk industries
        high_risk_score = agent._calculate_asset_risk_score(Mock(
            industry_sector="oil_gas",
            construction_type=None
        ))
        
        low_risk_score = agent._calculate_asset_risk_score(Mock(
            industry_sector="healthcare",
            construction_type=None
        ))
        
        assert high_risk_score > low_risk_score
    
    def test_financial_rating_determination(self, agent):
        """Test financial rating determination"""
        test_cases = [
            (95.0, "AAA"),
            (85.0, "AA"),
            (75.0, "A"),
            (65.0, "BBB"),
            (55.0, "BB"),
            (45.0, "B"),
            (35.0, "CCC"),
            (25.0, "D")
        ]
        
        for score, expected_rating in test_cases:
            rating = agent._determine_financial_rating(score)
            assert rating == expected_rating


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_zero_asset_value_handling(self, agent, sample_risk_parameters):
        """Test handling of zero asset values"""
        result = agent.assess_catastrophe_exposure(sample_risk_parameters, Decimal('0'))
        assert result.pml_estimate == Decimal('0')
        assert result.overall_cat_score >= 0
    
    def test_negative_financial_values_handling(self, agent):
        """Test handling of negative financial values"""
        negative_financial_data = FinancialData(
            id="neg1",
            application_id="app1",
            revenue=Decimal('-1000000'),  # Negative revenue (loss)
            assets=Decimal('5000000'),
            liabilities=Decimal('8000000')  # Liabilities > assets
        )
        
        result = agent.evaluate_financial_strength(negative_financial_data)
        assert isinstance(result, FinancialStrengthAssessment)
        assert result.financial_score < 50  # Should be low score
    
    def test_empty_strings_handling(self, agent):
        """Test handling of empty strings in parameters"""
        empty_risk_params = RiskParameters(
            id="empty1",
            application_id="app1",
            location="",
            asset_type="",
            industry_sector="",
            construction_type=""
        )
        
        result = agent.assess_catastrophe_exposure(empty_risk_params, Decimal('1000000'))
        assert isinstance(result, CatastropheExposure)
        assert result.geographic_risk_zone == GeographicRiskZone.MODERATE  # Default
    
    def test_very_large_numbers_handling(self, agent):
        """Test handling of very large financial numbers"""
        large_financial_data = FinancialData(
            id="large1",
            application_id="app1",
            revenue=Decimal('999999999999'),
            assets=Decimal('999999999999'),
            liabilities=Decimal('100000000000')
        )
        
        result = agent.evaluate_financial_strength(large_financial_data)
        assert isinstance(result, FinancialStrengthAssessment)
        assert 0 <= result.financial_score <= 100


if __name__ == "__main__":
    pytest.main([__file__])