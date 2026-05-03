"""
Advanced Temporal Analysis Module for COTCAgent

This module implements sophisticated machine learning methods for temporal medical data analysis,
including Cox Proportional Hazards models with time-dependent covariates and Wavelet Transform Analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
import logging

# Advanced ML libraries
try:
    import lifelines
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    warnings.warn("lifelines not installed. Cox model functionality will be limited.")

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    warnings.warn("PyWavelets not installed. Wavelet analysis will be unavailable.")

try:
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not installed. Time series analysis will be limited.")

try:
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TimeDependentCovariate:
    """Represents a time-dependent covariate in the Cox model"""
    name: str
    time_points: List[datetime]
    values: List[float]
    covariate_type: str = "continuous"  # continuous, categorical, binary

@dataclass
class CoxModelResult:
    """Results from Cox Proportional Hazards model fitting"""
    coefficients: Dict[str, float]
    hazard_ratios: Dict[str, float]
    p_values: Dict[str, float]
    concordance_index: float
    log_likelihood: float
    aic: float
    bic: float
    baseline_hazard: Optional[pd.DataFrame] = None
    fitted_model: Optional[Any] = None

@dataclass
class WaveletResult:
    """Results from wavelet transform analysis"""
    coefficients: np.ndarray
    scales: np.ndarray
    frequencies: np.ndarray
    power_spectrum: np.ndarray
    coherence_matrix: Optional[np.ndarray] = None
    wavelet_type: str = "morlet"
    sampling_rate: float = 1.0

@dataclass
class TimeDependentROC:
    """Time-dependent ROC analysis results"""
    time_points: List[float]
    auc_values: List[float]
    sensitivities: List[float]
    specificities: List[float]
    optimal_thresholds: List[float]
    confidence_intervals: Optional[List[Tuple[float, float]]] = None

class AdvancedTemporalAnalyzer:
    """
    Advanced temporal analysis module implementing sophisticated ML methods
    for medical time series data analysis.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.cox_models = {}
        self.wavelet_results = {}
        self.time_series_models = {}

        # Default configurations
        self.cox_config = self.config.get('cox', {
            'penalizer': 0.01,
            'l1_ratio': 0.0,
            'robust': True,
            'show_progress': True
        })

        self.wavelet_config = self.config.get('wavelet', {
            'wavelet': 'morl',
            'scales': np.arange(1, 128),
            'sampling_period': 1.0
        })

    def fit_cox_model(self,
                      time_to_event: np.ndarray,
                      event_indicator: np.ndarray,
                      covariates: pd.DataFrame,
                      time_dependent_covariates: Optional[List[TimeDependentCovariate]] = None,
                      model_name: str = "cox_model") -> CoxModelResult:
        """
        Fit Cox Proportional Hazards model with time-dependent covariates

        Args:
            time_to_event: Array of time-to-event or censoring times
            event_indicator: Binary array indicating if event occurred (1) or censored (0)
            covariates: DataFrame of baseline covariates
            time_dependent_covariates: List of time-dependent covariates
            model_name: Name identifier for the model

        Returns:
            CoxModelResult containing fitted model parameters and statistics
        """
        if not HAS_LIFELINES:
            raise ImportError("lifelines is required for Cox model analysis")

        logger.info(f"Fitting Cox model: {model_name}")

        try:
            # Prepare data for lifelines
            df = covariates.copy()
            df['time'] = time_to_event
            df['event'] = event_indicator

            # Handle time-dependent covariates if provided
            if time_dependent_covariates:
                df = self._prepare_time_dependent_data(df, time_dependent_covariates)

            # Fit Cox model
            cph = CoxPHFitter(**self.cox_config)
            cph.fit(df, duration_col='time', event_col='event')

            # Extract results
            coefficients = cph.params_.to_dict()
            hazard_ratios = cph.hazard_ratios_.to_dict()
            p_values = cph.p_values_.to_dict()

            # Calculate model statistics
            concordance_index_val = cph.concordance_index_
            log_likelihood = cph.log_likelihood_
            aic = cph.AIC_
            bic = cph.BIC_

            # Get baseline hazard if available
            baseline_hazard = None
            try:
                baseline_hazard = cph.baseline_hazard_
            except:
                pass

            result = CoxModelResult(
                coefficients=coefficients,
                hazard_ratios=hazard_ratios,
                p_values=p_values,
                concordance_index=concordance_index_val,
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic,
                baseline_hazard=baseline_hazard,
                fitted_model=cph
            )

            self.cox_models[model_name] = result
            logger.info(f"Successfully fitted Cox model {model_name} with C-index: {concordance_index_val:.3f}")

            return result

        except Exception as e:
            logger.error(f"Error fitting Cox model {model_name}: {e}")
            raise

    def _prepare_time_dependent_data(self,
                                   df: pd.DataFrame,
                                   time_dependent_covariates: List[TimeDependentCovariate]) -> pd.DataFrame:
        """Prepare data for time-dependent covariates in Cox model"""
        # This is a simplified implementation
        # In practice, you'd need to expand the dataset for time-dependent covariates
        # following the counting process notation

        expanded_df = df.copy()

        for tdc in time_dependent_covariates:
            # For each time-dependent covariate, we need to create time intervals
            # This is a complex operation that requires careful handling of time intervals

            # Simplified: add the covariate values at baseline for now
            if len(tdc.values) > 0:
                expanded_df[tdc.name] = tdc.values[0]  # Use first value as baseline

        return expanded_df

    def perform_wavelet_transform(self,
                                time_series: np.ndarray,
                                time_points: Optional[np.ndarray] = None,
                                wavelet_type: str = "morlet",
                                scales: Optional[np.ndarray] = None,
                                sampling_rate: float = 1.0,
                                result_name: str = "wavelet_analysis") -> WaveletResult:
        """
        Perform continuous wavelet transform on medical time series

        Args:
            time_series: 1D array of time series data
            time_points: Corresponding time points (optional)
            wavelet_type: Type of wavelet to use
            scales: Scales for wavelet transform
            sampling_rate: Sampling rate of the data
            result_name: Name identifier for results

        Returns:
            WaveletResult containing transform coefficients and analysis
        """
        if not HAS_PYWT:
            raise ImportError("PyWavelets is required for wavelet analysis")

        logger.info(f"Performing wavelet transform: {result_name}")

        try:
            # Set default scales if not provided
            if scales is None:
                scales = self.wavelet_config['scales']

            # Perform continuous wavelet transform
            coefficients, frequencies = pywt.cwt(time_series, scales, wavelet_type, sampling_rate)

            # Calculate power spectrum
            power_spectrum = np.abs(coefficients) ** 2

            # Calculate frequencies corresponding to scales
            if wavelet_type == 'morlet':
                central_frequency = pywt.ContinuousWavelet(wavelet_type).center_frequency
                frequencies = central_frequency / (scales * sampling_rate)

            result = WaveletResult(
                coefficients=coefficients,
                scales=scales,
                frequencies=frequencies,
                power_spectrum=power_spectrum,
                wavelet_type=wavelet_type,
                sampling_rate=sampling_rate
            )

            self.wavelet_results[result_name] = result
            logger.info(f"Successfully performed wavelet transform {result_name}")

            return result

        except Exception as e:
            logger.error(f"Error performing wavelet transform {result_name}: {e}")
            raise

    def compute_wavelet_coherence(self,
                                series1: np.ndarray,
                                series2: np.ndarray,
                                scales: Optional[np.ndarray] = None,
                                wavelet_type: str = "morlet",
                                smoothing: int = 5) -> np.ndarray:
        """
        Compute wavelet coherence between two time series

        Args:
            series1: First time series
            series2: Second time series
            scales: Scales for analysis
            wavelet_type: Type of wavelet
            smoothing: Smoothing parameter for coherence calculation

        Returns:
            Coherence matrix
        """
        if not HAS_PYWT:
            raise ImportError("PyWavelets is required for wavelet coherence")

        if scales is None:
            scales = self.wavelet_config['scales']

        # Perform wavelet transforms
        coef1, _ = pywt.cwt(series1, scales, wavelet_type)
        coef2, _ = pywt.cwt(series2, scales, wavelet_type)

        # Compute cross-wavelet transform
        cross_wavelet = coef1 * np.conj(coef2)

        # Compute individual wavelet transforms magnitude squared
        w1_sq = np.abs(coef1) ** 2
        w2_sq = np.abs(coef2) ** 2

        # Smooth the spectra
        from scipy.ndimage import gaussian_filter1d
        w1_sq_smooth = gaussian_filter1d(w1_sq, smoothing, axis=1)
        w2_sq_smooth = gaussian_filter1d(w2_sq, smoothing, axis=1)
        cross_smooth = gaussian_filter1d(cross_wavelet.real, smoothing, axis=1) + \
                      1j * gaussian_filter1d(cross_wavelet.imag, smoothing, axis=1)

        # Compute wavelet coherence
        coherence = np.abs(cross_smooth) ** 2 / (w1_sq_smooth * w2_sq_smooth)

        return coherence

    def time_dependent_roc_analysis(self,
                                  predicted_risks: np.ndarray,
                                  actual_events: np.ndarray,
                                  event_times: np.ndarray,
                                  censoring_times: np.ndarray,
                                  evaluation_times: Optional[List[float]] = None) -> TimeDependentROC:
        """
        Perform time-dependent ROC analysis for dynamic risk prediction

        Args:
            predicted_risks: Predicted risk scores
            actual_events: Binary indicators of events
            event_times: Times of events or censoring
            censoring_times: Censoring times
            evaluation_times: Time points for ROC evaluation

        Returns:
            TimeDependentROC with AUC values over time
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for ROC analysis")

        logger.info("Performing time-dependent ROC analysis")

        if evaluation_times is None:
            # Use quantiles of event times as evaluation points
            event_times_sorted = np.sort(event_times[actual_events == 1])
            evaluation_times = np.quantile(event_times_sorted,
                                         [0.25, 0.5, 0.75, 0.9]) if len(event_times_sorted) > 0 else []

        auc_values = []
        sensitivities = []
        specificities = []
        optimal_thresholds = []

        for t in evaluation_times:
            # Create labels for time t
            labels_t = self._create_time_dependent_labels(
                actual_events, event_times, censoring_times, t
            )

            # Only compute ROC if we have both positive and negative cases
            if len(np.unique(labels_t)) == 2:
                # Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(labels_t, predicted_risks)

                # Calculate AUC
                auc_t = auc(fpr, tpr)
                auc_values.append(auc_t)

                # Find optimal threshold (Youden's J statistic)
                youden_j = tpr - fpr
                optimal_idx = np.argmax(youden_j)
                optimal_threshold = thresholds[optimal_idx]
                optimal_thresholds.append(optimal_threshold)

                # Store sensitivity and specificity at optimal threshold
                sensitivities.append(tpr[optimal_idx])
                specificities.append(1 - fpr[optimal_idx])
            else:
                auc_values.append(np.nan)
                sensitivities.append(np.nan)
                specificities.append(np.nan)
                optimal_thresholds.append(np.nan)

        result = TimeDependentROC(
            time_points=evaluation_times,
            auc_values=auc_values,
            sensitivities=sensitivities,
            specificities=specificities,
            optimal_thresholds=optimal_thresholds
        )

        logger.info(f"Time-dependent ROC analysis completed. Mean AUC: {np.nanmean(auc_values):.3f}")
        return result

    def _create_time_dependent_labels(self,
                                    events: np.ndarray,
                                    event_times: np.ndarray,
                                    censoring_times: np.ndarray,
                                    evaluation_time: float) -> np.ndarray:
        """
        Create binary labels for time-dependent ROC analysis

        Args:
            events: Event indicators
            event_times: Event times
            censoring_times: Censoring times
            evaluation_time: Time point for evaluation

        Returns:
            Binary labels for ROC analysis
        """
        labels = np.zeros(len(events))

        for i in range(len(events)):
            if events[i] == 1 and event_times[i] <= evaluation_time:
                # Event occurred before or at evaluation time
                labels[i] = 1
            elif censoring_times[i] > evaluation_time:
                # Still at risk at evaluation time (censored after t)
                labels[i] = 0
            else:
                # Censored before evaluation time - exclude from analysis
                labels[i] = -1  # Will be filtered out

        # Remove cases censored before evaluation time
        valid_mask = labels != -1
        return labels[valid_mask]

    def fit_arima_model(self,
                       time_series: np.ndarray,
                       order: Tuple[int, int, int] = (1, 1, 1),
                       seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                       model_name: str = "arima_model") -> Any:
        """
        Fit ARIMA model for time series forecasting

        Args:
            time_series: Time series data
            order: (p, d, q) parameters for ARIMA
            seasonal_order: Seasonal parameters (P, D, Q, s)
            model_name: Name identifier for the model

        Returns:
            Fitted ARIMA model
        """
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels is required for ARIMA modeling")

        logger.info(f"Fitting ARIMA model: {model_name}")

        try:
            model = ARIMA(time_series, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit()

            self.time_series_models[model_name] = fitted_model

            logger.info(f"Successfully fitted ARIMA{order} model {model_name}")
            logger.info(f"AIC: {fitted_model.aic:.2f}, BIC: {fitted_model.bic:.2f}")

            return fitted_model

        except Exception as e:
            logger.error(f"Error fitting ARIMA model {model_name}: {e}")
            raise

    def analyze_temporal_patterns(self,
                                patient_data: Dict,
                                symptom_name: str) -> Dict[str, Any]:
        """
        Comprehensive temporal pattern analysis for a specific symptom

        Args:
            patient_data: Patient temporal data
            symptom_name: Name of the symptom to analyze

        Returns:
            Dictionary containing various temporal analysis results
        """
        logger.info(f"Analyzing temporal patterns for symptom: {symptom_name}")

        results = {
            'symptom': symptom_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'methods_applied': []
        }

        try:
            # Extract temporal data for the symptom
            temporal_data = self._extract_symptom_temporal_data(patient_data, symptom_name)

            if not temporal_data:
                results['error'] = f"No temporal data found for symptom {symptom_name}"
                return results

            # Convert to time series
            time_series, time_points = self._create_time_series_from_temporal_data(temporal_data)

            # Apply various analysis methods
            if len(time_series) > 10:  # Minimum data requirement

                # Wavelet analysis
                try:
                    wavelet_result = self.perform_wavelet_transform(
                        time_series,
                        time_points=time_points,
                        result_name=f"{symptom_name}_wavelet"
                    )
                    results['wavelet_analysis'] = {
                        'power_spectrum_shape': wavelet_result.power_spectrum.shape,
                        'dominant_scales': self._find_dominant_scales(wavelet_result),
                        'frequency_bands': self._analyze_frequency_bands(wavelet_result)
                    }
                    results['methods_applied'].append('wavelet_transform')
                except Exception as e:
                    logger.warning(f"Wavelet analysis failed for {symptom_name}: {e}")

                # ARIMA modeling
                try:
                    arima_model = self.fit_arima_model(
                        time_series,
                        model_name=f"{symptom_name}_arima"
                    )
                    results['arima_model'] = {
                        'aic': arima_model.aic,
                        'bic': arima_model.bic,
                        'order': arima_model.model.order,
                        'residuals_std': np.std(arima_model.resid)
                    }
                    results['methods_applied'].append('arima_modeling')
                except Exception as e:
                    logger.warning(f"ARIMA modeling failed for {symptom_name}: {e}")

                # Statistical analysis
                results['statistical_analysis'] = self._compute_temporal_statistics(time_series)
                results['methods_applied'].append('statistical_analysis')

            # Trend analysis
            results['trend_analysis'] = self._analyze_trends(temporal_data)
            results['methods_applied'].append('trend_analysis')

        except Exception as e:
            logger.error(f"Error in temporal pattern analysis for {symptom_name}: {e}")
            results['error'] = str(e)

        return results

    def _extract_symptom_temporal_data(self, patient_data: Dict, symptom_name: str) -> List[Dict]:
        """Extract temporal data for a specific symptom from patient data"""
        temporal_data = []

        # Look in basic signs section
        basic_signs = patient_data.get('temporal_indicators', {}).get('basic_signs', {})
        if symptom_name in basic_signs:
            symptom_info = basic_signs[symptom_name]
            for timestamp, severity in zip(symptom_info['timestamps'], symptom_info['severity_levels']):
                temporal_data.append({
                    'timestamp': timestamp,
                    'value': self._severity_to_numeric(severity),
                    'type': 'severity'
                })

        return temporal_data

    def _severity_to_numeric(self, severity: str) -> float:
        """Convert severity level to numeric value"""
        severity_map = {
            'None': 0.0,
            'Mild': 1.0,
            'Minor': 1.5,
            'Moderate': 2.0,
            'Medium': 2.5,
            'Severe': 3.0,
            'Extreme': 4.0,
            'Critical': 5.0
        }
        return severity_map.get(severity, 0.0)

    def _create_time_series_from_temporal_data(self, temporal_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Create time series arrays from temporal data"""
        # Sort by timestamp
        sorted_data = sorted(temporal_data, key=lambda x: x['timestamp'])

        # Extract values and times
        values = np.array([item['value'] for item in sorted_data])
        timestamps = [datetime.fromisoformat(item['timestamp']) for item in sorted_data]

        # Convert timestamps to numeric (days since first observation)
        if timestamps:
            start_time = timestamps[0]
            time_points = np.array([(t - start_time).total_seconds() / (24 * 3600) for t in timestamps])
        else:
            time_points = np.array([])

        return values, time_points

    def _find_dominant_scales(self, wavelet_result: WaveletResult) -> List[int]:
        """Find dominant scales in wavelet power spectrum"""
        # Find scales with maximum power
        mean_power = np.mean(wavelet_result.power_spectrum, axis=1)
        dominant_indices = np.argsort(mean_power)[-3:]  # Top 3 scales
        return wavelet_result.scales[dominant_indices].tolist()

    def _analyze_frequency_bands(self, wavelet_result: WaveletResult) -> Dict[str, Any]:
        """Analyze different frequency bands in wavelet spectrum"""
        frequencies = wavelet_result.frequencies

        # Define frequency bands (adjust based on your data sampling rate)
        bands = {
            'very_low': (0, 0.01),
            'low': (0.01, 0.1),
            'medium': (0.1, 1.0),
            'high': (1.0, 10.0)
        }

        band_analysis = {}
        for band_name, (f_min, f_max) in bands.items():
            mask = (frequencies >= f_min) & (frequencies <= f_max)
            if np.any(mask):
                band_power = np.mean(wavelet_result.power_spectrum[mask, :])
                band_analysis[band_name] = {
                    'power': band_power,
                    'scale_count': np.sum(mask)
                }

        return band_analysis

    def _compute_temporal_statistics(self, time_series: np.ndarray) -> Dict[str, float]:
        """Compute statistical measures for time series"""
        stats = {
            'mean': float(np.mean(time_series)),
            'std': float(np.std(time_series)),
            'min': float(np.min(time_series)),
            'max': float(np.max(time_series)),
            'median': float(np.median(time_series)),
            'skewness': float(self._compute_skewness(time_series)),
            'kurtosis': float(self._compute_kurtosis(time_series))
        }

        # Autocorrelation if statsmodels available
        if HAS_STATSMODELS and len(time_series) > 10:
            try:
                autocorr = acf(time_series, nlags=min(10, len(time_series)-1))
                stats['autocorrelation_lag1'] = float(autocorr[1]) if len(autocorr) > 1 else None
            except:
                stats['autocorrelation_lag1'] = None

        return stats

    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _analyze_trends(self, temporal_data: List[Dict]) -> Dict[str, Any]:
        """Analyze trends in temporal data"""
        if not temporal_data:
            return {'trend_type': 'insufficient_data'}

        # Sort by time
        sorted_data = sorted(temporal_data, key=lambda x: x['timestamp'])
        values = [item['value'] for item in sorted_data]

        # Simple linear trend
        if len(values) > 1:
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)

            trend_type = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'

            return {
                'trend_type': trend_type,
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(self._compute_r_squared(x, values, slope, intercept)),
                'data_points': len(values)
            }
        else:
            return {
                'trend_type': 'single_point',
                'data_points': 1
            }

    def _compute_r_squared(self, x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
        """Compute R-squared for linear fit"""
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    def generate_comprehensive_report(self, patient_data: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive temporal analysis report for patient

        Args:
            patient_data: Complete patient data dictionary

        Returns:
            Comprehensive analysis report
        """
        logger.info("Generating comprehensive temporal analysis report")

        report = {
            'patient_id': patient_data.get('patient_info', {}).get('id', 'Unknown'),
            'analysis_timestamp': datetime.now().isoformat(),
            'temporal_analysis': {},
            'risk_assessment': {},
            'recommendations': []
        }

        # Analyze all symptoms with temporal data
        temporal_indicators = patient_data.get('temporal_indicators', {})

        # Analyze basic signs
        basic_signs = temporal_indicators.get('basic_signs', {})
        for symptom_name in basic_signs.keys():
            analysis = self.analyze_temporal_patterns(patient_data, symptom_name)
            report['temporal_analysis'][symptom_name] = analysis

        # Generate risk assessment based on temporal patterns
        report['risk_assessment'] = self._generate_risk_assessment(report['temporal_analysis'])

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)

        logger.info(f"Comprehensive report generated for patient {report['patient_id']}")
        return report

    def _generate_risk_assessment(self, temporal_analysis: Dict) -> Dict[str, Any]:
        """Generate risk assessment based on temporal analysis results"""
        risk_scores = {}
        risk_factors = []

        for symptom, analysis in temporal_analysis.items():
            if 'error' in analysis:
                continue

            # Assess risk based on various factors
            risk_score = 0.0
            factors = []

            # Trend analysis
            trend_info = analysis.get('trend_analysis', {})
            if trend_info.get('trend_type') == 'increasing':
                risk_score += 0.3
                factors.append(f"Increasing trend in {symptom}")

            # Statistical measures
            stats = analysis.get('statistical_analysis', {})
            if stats.get('std', 0) > 2.0:  # High variability
                risk_score += 0.2
                factors.append(f"High variability in {symptom}")

            # Wavelet analysis
            wavelet = analysis.get('wavelet_analysis', {})
            if wavelet:
                # Check for high-frequency components (might indicate acute changes)
                freq_bands = wavelet.get('frequency_bands', {})
                if freq_bands.get('high', {}).get('power', 0) > 0.5:
                    risk_score += 0.25
                    factors.append(f"High-frequency components in {symptom}")

            risk_scores[symptom] = min(risk_score, 1.0)  # Cap at 1.0
            if factors:
                risk_factors.extend(factors)

        return {
            'symptom_risk_scores': risk_scores,
            'overall_risk_score': np.mean(list(risk_scores.values())) if risk_scores else 0.0,
            'risk_factors': risk_factors,
            'risk_level': self._classify_risk_level(np.mean(list(risk_scores.values())) if risk_scores else 0.0)
        }

    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level based on score"""
        if risk_score >= 0.7:
            return 'High'
        elif risk_score >= 0.4:
            return 'Medium'
        elif risk_score >= 0.2:
            return 'Low'
        else:
            return 'Very Low'

    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate clinical recommendations based on analysis"""
        recommendations = []

        risk_assessment = report.get('risk_assessment', {})
        overall_risk = risk_assessment.get('overall_risk_score', 0.0)

        if overall_risk >= 0.7:
            recommendations.append("Immediate clinical evaluation recommended due to high temporal risk patterns")
            recommendations.append("Consider hospitalization for close monitoring")
        elif overall_risk >= 0.4:
            recommendations.append("Schedule follow-up appointment within 1-2 weeks")
            recommendations.append("Increase monitoring frequency for key symptoms")
        elif overall_risk >= 0.2:
            recommendations.append("Regular monitoring and lifestyle modifications recommended")
        else:
            recommendations.append("Continue routine care and preventive measures")

        # Add specific recommendations based on temporal patterns
        temporal_analysis = report.get('temporal_analysis', {})
        for symptom, analysis in temporal_analysis.items():
            trend = analysis.get('trend_analysis', {}).get('trend_type')
            if trend == 'increasing':
                recommendations.append(f"Monitor {symptom} closely due to worsening trend")

        return recommendations
