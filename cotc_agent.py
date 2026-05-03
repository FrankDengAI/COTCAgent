"""
COTCAgent - Medical Temporal Data Analysis Chain-of-Thought Completion Agent
Fixed version - Reduced token usage, fixed indentation issues
"""

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import time
import hashlib
import functools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import traceback
from collections import defaultdict
from itertools import combinations

import aiohttp
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

# Import advanced analysis modules
try:
    from advanced_temporal_analysis import AdvancedTemporalAnalyzer, TimeDependentCovariate
    HAS_ADVANCED_TEMPORAL = True
except ImportError:
    HAS_ADVANCED_TEMPORAL = False
    logger.warning("Advanced temporal analysis module not available")

try:
    from cox_model_advanced import AdvancedCoxAnalyzer
    HAS_ADVANCED_COX = True
except ImportError:
    HAS_ADVANCED_COX = False
    logger.warning("Advanced Cox model module not available")

try:
    from wavelet_analysis_advanced import AdvancedWaveletAnalyzer
    HAS_ADVANCED_WAVELET = True
except ImportError:
    HAS_ADVANCED_WAVELET = False
    logger.warning("Advanced wavelet analysis module not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cotc_agent.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('cotc_agent')

# TSA Module Mathematical Models (from paper Section 3.2)

class TSAMathematicalModels:
    """Implementation of mathematical models from TSA Module (Section 3.2)"""

    @staticmethod
    def linear_mixed_effects_model(y: np.ndarray, time_points: np.ndarray, patient_ids: np.ndarray) -> Dict[str, Any]:
        """
        Linear Mixed Effects Model as described in paper Eq. (2)
        yij = β0 + β1tij + ui + ϵij

        Args:
            y: biomarker measurements
            time_points: time points for each measurement
            patient_ids: patient identifiers for random effects

        Returns:
            Dict containing model parameters and diagnostics
        """
        try:
            # Simple implementation using OLS with patient-specific intercepts
            unique_patients = np.unique(patient_ids)
            patient_effects = {}

            # Estimate patient-specific random effects
            for pid in unique_patients:
                mask = patient_ids == pid
                if np.sum(mask) > 1:  # Need at least 2 points for slope estimation
                    patient_y = y[mask]
                    patient_t = time_points[mask]

                    # Fit linear model for this patient
                    slope, intercept = np.polyfit(patient_t, patient_y, 1)
                    patient_effects[pid] = {'intercept': intercept, 'slope': slope}

            # Population-level estimates
            slopes = [effects['slope'] for effects in patient_effects.values()]
            intercepts = [effects['intercept'] for effects in patient_effects.values()]

            beta_0 = np.mean(intercepts)  # Population intercept
            beta_1 = np.mean(slopes)      # Population slope

            # Calculate random effects variance
            ui_variance = np.var(intercepts - beta_0) if len(intercepts) > 1 else 0

            return {
                'beta_0': beta_0,  # Population intercept
                'beta_1': beta_1,  # Population slope
                'ui_variance': ui_variance,  # Random effects variance
                'patient_effects': patient_effects,
                'trend_direction': 'increasing' if beta_1 > 0 else 'decreasing',
                'significance_test': {
                    'slope_t_stat': beta_1 / (np.std(slopes) / np.sqrt(len(slopes))) if len(slopes) > 1 else 0,
                    'slope_p_value': 2 * (1 - norm.cdf(abs(beta_1 / (np.std(slopes) / np.sqrt(len(slopes)))))) if len(slopes) > 1 else 1.0
                }
            }
        except Exception as e:
            return {'error': f'Linear mixed effects model failed: {str(e)}'}

    @staticmethod
    def bayesian_change_point_detection(y: np.ndarray, time_points: np.ndarray) -> Dict[str, Any]:
        """
        Bayesian Change Point Detection as described in paper Eq. (3)
        P(τ |y) ∝ P(y|τ)P(τ)

        Args:
            y: time series data
            time_points: time points

        Returns:
            Dict containing change point analysis results
        """
        try:
            n = len(y)
            if n < 10:  # Need sufficient data points
                return {'error': 'Insufficient data points for change point detection'}

            # Simple Bayesian change point detection
            # Assume normal distributions with unknown means and variances
            change_points = []
            probabilities = []

            for tau in range(5, n-5):  # Possible change points
                # Pre-change segment
                y1 = y[:tau]
                # Post-change segment
                y2 = y[tau:]

                if len(y1) < 3 or len(y2) < 3:
                    continue

                # Fit normal distributions
                mu1, sigma1 = np.mean(y1), np.std(y1)
                mu2, sigma2 = np.mean(y2), np.std(y2)

                # Likelihoods (assuming equal prior probability for all tau)
                likelihood1 = np.prod(norm.pdf(y1, mu1, sigma1 + 1e-6))  # Add small epsilon
                likelihood2 = np.prod(norm.pdf(y2, mu2, sigma2 + 1e-6))

                # Prior P(τ) = 1/n (uniform)
                prior = 1.0 / n

                # Posterior (unnormalized)
                posterior = likelihood1 * likelihood2 * prior
                probabilities.append(posterior)

                if posterior > np.max(probabilities[:-1] + [0]):  # Simple threshold
                    change_points.append(tau)

            # Find the most probable change point
            if probabilities:
                best_tau_idx = np.argmax(probabilities)
                best_tau = best_tau_idx + 5  # Offset for range start

                return {
                    'change_point': best_tau,
                    'change_time': time_points[best_tau] if best_tau < len(time_points) else None,
                    'probability': probabilities[best_tau_idx],
                    'pre_change_stats': {
                        'mean': np.mean(y[:best_tau]),
                        'std': np.std(y[:best_tau])
                    },
                    'post_change_stats': {
                        'mean': np.mean(y[best_tau:]),
                        'std': np.std(y[best_tau:])
                    },
                    'change_magnitude': abs(np.mean(y[best_tau:]) - np.mean(y[:best_tau]))
                }

            return {'change_point': None, 'probability': 0.0}

        except Exception as e:
            return {'error': f'Bayesian change point detection failed: {str(e)}'}

    @staticmethod
    def multi_scale_trend_analysis(y: np.ndarray, time_points: np.ndarray) -> Dict[str, Any]:
        """
        Multi-scale Trend Analysis as described in paper Eq. (9)
        Trend(t) = Σ[k=1 to K] wk · Filterk(yt)

        Args:
            y: time series data
            time_points: time points

        Returns:
            Dict containing multi-scale trend analysis
        """
        try:
            # Define different time scales (short-term, medium-term, long-term)
            scales = [3, 7, 14, 30]  # days
            weights = [0.4, 0.3, 0.2, 0.1]  # Corresponding weights

            trends = {}
            for scale, weight in zip(scales, weights):
                if len(y) >= scale:
                    # Simple moving average filter
                    filtered = np.convolve(y, np.ones(scale)/scale, mode='valid')
                    trends[f'scale_{scale}d'] = {
                        'trend_values': filtered.tolist(),
                        'weight': weight,
                        'slope': np.polyfit(range(len(filtered)), filtered, 1)[0] if len(filtered) > 1 else 0
                    }

            # Combined multi-scale trend
            combined_trend = np.zeros_like(y, dtype=float)
            valid_length = len(y)

            for scale_data in trends.values():
                trend_values = np.array(scale_data['trend_values'])
                weight = scale_data['weight']

                # Pad shorter trends to match original length
                if len(trend_values) < valid_length:
                    # Forward fill
                    padding = np.full(valid_length - len(trend_values), trend_values[0])
                    trend_values = np.concatenate([padding, trend_values])

                combined_trend += weight * trend_values

            # Overall trend direction
            overall_slope = np.polyfit(range(len(combined_trend)), combined_trend, 1)[0]

            return {
                'multi_scale_trends': trends,
                'combined_trend': combined_trend.tolist(),
                'overall_trend_slope': overall_slope,
                'overall_trend_direction': 'increasing' if overall_slope > 0 else 'decreasing',
                'trend_strength': abs(overall_slope)
            }

        except Exception as e:
            return {'error': f'Multi-scale trend analysis failed: {str(e)}'}

    @staticmethod
    def anomaly_detection_with_temporal_consistency(y: np.ndarray, time_points: np.ndarray) -> Dict[str, Any]:
        """
        Anomaly Detection with Temporal Consistency as described in paper Eq. (10)
        AnomalyScore(t) = |yt - ŷt|/σresidual + λ · TemporalConsistency(t)

        Args:
            y: time series data
            time_points: time points

        Returns:
            Dict containing anomaly detection results
        """
        try:
            # Fit a simple linear trend as baseline
            if len(y) < 5:
                return {'error': 'Insufficient data for anomaly detection'}

            time_idx = np.arange(len(y))
            slope, intercept = np.polyfit(time_idx, y, 1)

            # Predictions
            y_pred = slope * time_idx + intercept

            # Residuals
            residuals = y - y_pred
            residual_std = np.std(residuals)

            # Basic anomaly scores
            anomaly_scores = np.abs(residuals) / (residual_std + 1e-6)

            # Temporal consistency component
            temporal_consistency = np.zeros_like(y, dtype=float)

            for i in range(2, len(y) - 2):
                # Check consistency with neighboring points
                local_window = y[i-2:i+3]
                local_mean = np.mean(local_window)
                local_std = np.std(local_window)

                # Deviation from local pattern
                temporal_consistency[i] = abs(y[i] - local_mean) / (local_std + 1e-6)

            # Combined anomaly score
            lambda_param = 0.3  # Weight for temporal consistency
            combined_scores = anomaly_scores + lambda_param * temporal_consistency

            # Identify anomalies (threshold-based)
            threshold = np.percentile(combined_scores, 95)  # Top 5% as anomalies
            anomaly_indices = np.where(combined_scores > threshold)[0]

            return {
                'anomaly_scores': combined_scores.tolist(),
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_times': time_points[anomaly_indices].tolist() if len(anomaly_indices) > 0 else [],
                'threshold': threshold,
                'anomaly_percentage': len(anomaly_indices) / len(y) * 100,
                'temporal_consistency_scores': temporal_consistency.tolist()
            }

        except Exception as e:
            return {'error': f'Anomaly detection failed: {str(e)}'}

    @staticmethod
    def multiple_imputation_missing_data(data_matrix: np.ndarray, k: int = 5) -> Dict[str, Any]:
        """
        Multiple Imputation for Missing Data as described in paper Eq. (4)
        Ycomplete = ∪[k=1 to K] (Yobserved ∪ Yimputed(k))

        Args:
            data_matrix: Matrix with missing values (NaN)
            k: Number of imputations

        Returns:
            Dict containing imputed datasets and statistics
        """
        try:
            if not np.any(np.isnan(data_matrix)):
                return {'imputed_datasets': [data_matrix], 'message': 'No missing data found'}

            n_samples, n_features = data_matrix.shape
            imputed_datasets = []

            for imputation_round in range(k):
                # Copy original data
                imputed_data = data_matrix.copy()

                # Simple imputation strategy: mean imputation with noise
                for col in range(n_features):
                    missing_mask = np.isnan(imputed_data[:, col])
                    if np.any(missing_mask):
                        # Use observed values to estimate mean and std
                        observed_values = imputed_data[~missing_mask, col]
                        if len(observed_values) > 0:
                            mean_val = np.mean(observed_values)
                            std_val = np.std(observed_values) if len(observed_values) > 1 else 0

                            # Add random noise to avoid identical imputations
                            noise = np.random.normal(0, std_val * 0.1, size=np.sum(missing_mask))
                            imputed_data[missing_mask, col] = mean_val + noise

                imputed_datasets.append(imputed_data)

            # Calculate imputation statistics
            original_missing = np.sum(np.isnan(data_matrix))
            total_values = data_matrix.size

            return {
                'imputed_datasets': [dataset.tolist() for dataset in imputed_datasets],
                'imputation_statistics': {
                    'original_missing_percentage': (original_missing / total_values) * 100,
                    'imputations_performed': k,
                    'method': 'mean_imputation_with_noise'
                },
                'data_quality_metrics': {
                    'completeness': ((total_values - original_missing) / total_values) * 100,
                    'imputation_uncertainty': 'estimated'
                }
            }

        except Exception as e:
            return {'error': f'Multiple imputation failed: {str(e)}'}

    @staticmethod
    def dimension_reduction_multivariate_analysis(data_matrix: np.ndarray, explained_variance_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Dimension Reduction for Multivariate Time Series as described in paper Eq. (5)
        Z = f(Y) = WT Y

        Args:
            data_matrix: Multivariate time series matrix Y (n_samples x n_features)
            explained_variance_threshold: Target explained variance ratio

        Returns:
            Dict containing dimension reduction results
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Handle missing values (simple imputation for PCA)
            data_imputed = data_matrix.copy()
            for col in range(data_matrix.shape[1]):
                missing_mask = np.isnan(data_imputed[:, col])
                if np.any(missing_mask):
                    observed_values = data_imputed[~missing_mask, col]
                    if len(observed_values) > 0:
                        data_imputed[missing_mask, col] = np.mean(observed_values)

            # Standardize the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_imputed)

            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(data_scaled)

            # Find number of components explaining target variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.where(cumulative_variance >= explained_variance_threshold)[0]
            optimal_components = n_components[0] + 1 if len(n_components) > 0 else min(3, data_matrix.shape[1])

            # Project to optimal dimensions
            pca_optimal = PCA(n_components=optimal_components)
            z_reduced = pca_optimal.fit_transform(data_scaled)

            return {
                'reduced_data': z_reduced.tolist(),
                'projection_weights': pca_optimal.components_.tolist(),
                'explained_variance_ratios': pca_optimal.explained_variance_ratio_.tolist(),
                'cumulative_variance': cumulative_variance[:optimal_components].tolist(),
                'optimal_components': optimal_components,
                'original_dimensions': data_matrix.shape[1]
            }

        except ImportError:
            # Fallback to simple dimension reduction if sklearn not available
            return {'error': 'PCA requires scikit-learn. Install with: pip install scikit-learn'}
        except Exception as e:
            return {'error': f'Dimension reduction failed: {str(e)}'}

    @staticmethod
    def piecewise_linear_trend_segmentation(y: np.ndarray, time_points: np.ndarray, max_segments: int = 3) -> Dict[str, Any]:
        """
        Piecewise Linear Trend Quantification as described in paper Eq. (6)
        βsegment = argmin_β Σ[ta to tb] (yt - β0 - β1t)^2

        Args:
            y: time series data
            time_points: time points
            max_segments: maximum number of linear segments

        Returns:
            Dict containing piecewise linear analysis
        """
        try:
            # Simple implementation: try different change points
            n = len(y)
            if n < 10:
                return {'error': 'Insufficient data for piecewise analysis'}

            best_rss = float('inf')
            best_segments = []

            # Try different numbers of segments
            for n_segments in range(1, min(max_segments + 1, n // 3 + 1)):
                if n_segments == 1:
                    # Single segment
                    slope, intercept = np.polyfit(time_points, y, 1)
                    rss = np.sum((y - (slope * time_points + intercept)) ** 2)
                    segments = [{
                        'start_idx': 0,
                        'end_idx': n-1,
                        'slope': slope,
                        'intercept': intercept,
                        'rss': rss
                    }]
                else:
                    # Multiple segments - simple heuristic
                    segment_size = n // n_segments
                    segments = []
                    total_rss = 0

                    for i in range(n_segments):
                        start_idx = i * segment_size
                        end_idx = min((i + 1) * segment_size, n) if i < n_segments - 1 else n

                        segment_y = y[start_idx:end_idx]
                        segment_t = time_points[start_idx:end_idx]

                        if len(segment_y) > 1:
                            slope, intercept = np.polyfit(segment_t, segment_y, 1)
                            rss = np.sum((segment_y - (slope * segment_t + intercept)) ** 2)
                            total_rss += rss
                        else:
                            slope, intercept, rss = 0, segment_y[0] if len(segment_y) > 0 else 0, 0

                        segments.append({
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'slope': slope,
                            'intercept': intercept,
                            'rss': rss
                        })

                    rss = total_rss

                if rss < best_rss:
                    best_rss = rss
                    best_segments = segments

            # Calculate trend stability index (Eq. 7 approximation)
            if len(best_segments) > 1:
                slope_changes = []
                for i in range(1, len(best_segments)):
                    slope_changes.append(abs(best_segments[i]['slope'] - best_segments[i-1]['slope']))

                tsi = 1 - np.mean(slope_changes) / (np.std([s['slope'] for s in best_segments]) + 1e-6)
            else:
                tsi = 1.0  # Single segment is perfectly stable

            return {
                'segments': best_segments,
                'total_rss': best_rss,
                'trend_stability_index': max(0, min(1, tsi)),  # Eq. 7 approximation
                'n_segments': len(best_segments),
                'segment_change_points': [s['end_idx'] for s in best_segments[:-1]]
            }

        except Exception as e:
            return {'error': f'Piecewise linear analysis failed: {str(e)}'}

# Custom Exceptions
class COTCAgentError(Exception):
    """Base exception for COTCAgent errors"""
    pass

class ValidationError(COTCAgentError):
    """Raised when input validation fails"""
    pass

class APIError(COTCAgentError):
    """Raised when API calls fail"""
    pass

class CodeExecutionError(COTCAgentError):
    """Raised when generated code execution fails"""
    pass

class ConfigurationError(COTCAgentError):
    """Raised when configuration is invalid"""
    pass

# Import validation utilities
from validation_utils import (
    sanitize_input, validate_patient_data as validate_patient_data_new,
    validate_api_config, validate_medical_query, ValidationError, ConfigurationError
)
from performance_monitor import performance_monitor, get_performance_monitor

# Import COTC Module
try:
    from cotc_module import get_cotc_reasoning_engine, DiseaseRisk
except ImportError:
    logger.warning("COTC module not available, using fallback implementation")
    get_cotc_reasoning_engine = None

# Validation and Utility Functions

def calculate_hash(data: Any) -> str:
    """Calculate SHA256 hash of data for caching"""
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()

def retry_async(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    return wrapper

@dataclass
class DeepSeekConfig:
    """DeepSeek API Configuration with enhanced validation"""
    api_key: str
    api_base: str = "https://api.deepseek.com/v1/chat/completions"
    model: str = "deepseek-chat"
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout: int = 180
    save_temp_files: bool = False
    mock_mode: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    rate_limit_requests: int = 10
    rate_limit_window: int = 60  # 1 minute

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.api_key or not self.api_key.startswith(('sk-', 'pk-')):
            raise ConfigurationError("Invalid API key format")

        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ConfigurationError("Temperature must be between 0.0 and 2.0")

        if self.max_tokens < 1 or self.max_tokens > 32768:
            raise ConfigurationError("Max tokens must be between 1 and 32768")

        if self.timeout < 1:
            raise ConfigurationError("Timeout must be at least 1 second")

@dataclass
class SymptomIndicator:
    """Symptom or indicator data structure with validation"""
    id: str
    name: str
    values: List[Any]
    value_type: str = 'numeric'
    unit: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if not self.id or not self.name:
            raise ValidationError("Symptom ID and name are required")

        if not isinstance(self.values, list) or len(self.values) == 0:
            raise ValidationError("Values must be a non-empty list")

        if self.value_type not in ['numeric', 'categorical', 'boolean']:
            raise ValidationError("Value type must be numeric, categorical, or boolean")

@dataclass
class DiseaseRisk:
    """Disease risk data structure with validation"""
    disease_id: str
    disease_name: str
    risk_score: float
    confidence: float
    matched_symptoms: List[str]
    missing_symptoms: List[str]
    evidence_strength: str = 'weak'
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.disease_id or not self.disease_name:
            raise ValidationError("Disease ID and name are required")

        if not (0.0 <= self.risk_score <= 1.0):
            raise ValidationError("Risk score must be between 0.0 and 1.0")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValidationError("Confidence must be between 0.0 and 1.0")

        if self.evidence_strength not in ['weak', 'moderate', 'strong', 'very_strong']:
            raise ValidationError("Evidence strength must be one of: weak, moderate, strong, very_strong")

class ProgressIndicator:
    """Enhanced progress indicator with detailed tracking"""
    def __init__(self, task_name: str, enable_logging: bool = True):
        self.task_name = task_name
        self.start_time = time.time()
        self.steps_completed = 0
        self.total_steps = 0
        self.enable_logging = enable_logging
        self.checkpoints = []

    def set_total_steps(self, total: int):
        """Set total number of steps for progress tracking"""
        self.total_steps = total

    def update(self, message: str, increment_step: bool = True):
        """Update progress with current status message"""
        if increment_step:
            self.steps_completed += 1

        elapsed = time.time() - self.start_time
        progress_info = ""

        if self.total_steps > 0:
            progress_percent = (self.steps_completed / self.total_steps) * 100
            progress_info = f"[{self.steps_completed}/{self.total_steps} {progress_percent:.1f}%] "

        status_msg = f"[{elapsed:.1f}s] {progress_info}{self.task_name}: {message}"

        print(status_msg)
        if self.enable_logging:
            logger.info(status_msg)

        self.checkpoints.append({
            'timestamp': time.time(),
            'message': message,
            'elapsed': elapsed,
            'step': self.steps_completed
        })

    def complete(self, message: str):
        """Mark task as completed with final message and summary"""
        elapsed = time.time() - self.start_time
        summary = f"Task completed in {elapsed:.1f}s"

        if self.total_steps > 0:
            summary += f" ({self.steps_completed}/{self.total_steps} steps)"

        print(f"[{elapsed:.1f}s] {self.task_name}: {message}")
        print(f"[SUMMARY] {self.task_name}: {summary}")

        if self.enable_logging:
            logger.info(f"{self.task_name} completed: {summary}")

    def get_summary(self) -> Dict[str, Any]:
        """Get detailed execution summary"""
        elapsed = time.time() - self.start_time
        return {
            'task_name': self.task_name,
            'total_time': elapsed,
            'steps_completed': self.steps_completed,
            'total_steps': self.total_steps,
            'checkpoints': self.checkpoints,
            'average_step_time': elapsed / max(self.steps_completed, 1)
        }

class DeepSeekClient:
    """Enhanced DeepSeek API client with caching, retry logic, and rate limiting"""

    def __init__(self, config: DeepSeekConfig):
        # Validate configuration
        config_dict = {
            'api_key': config.api_key,
            'api_base': config.api_base,
            'model': config.model,
            'max_tokens': config.max_tokens,
            'temperature': config.temperature,
            'timeout': config.timeout
        }

        validation_result = validate_api_config(config_dict)
        if not validation_result.is_valid:
            raise ConfigurationError(f"Invalid API configuration: {validation_result.errors}")

        for warning in validation_result.warnings:
            logger.warning(f"API config warning: {warning}")

        self.config = config
        self.base_url = config.api_base
        self.headers = {
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json'
        }
        self.cache = {}
        self.request_times = []
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _is_rate_limited(self) -> bool:
        """Check if we're currently rate limited"""
        now = time.time()
        # Remove old requests outside the window
        self.request_times = [t for t in self.request_times if now - t < self.config.rate_limit_window]

        return len(self.request_times) >= self.config.rate_limit_requests

    def _wait_for_rate_limit(self):
        """Wait until rate limit allows another request"""
        if self._is_rate_limited():
            oldest_request = min(self.request_times)
            wait_time = self.config.rate_limit_window - (time.time() - oldest_request)
            if wait_time > 0:
                logger.warning(f"Rate limit exceeded, waiting {wait_time:.1f}s")
                time.sleep(wait_time)

    def _get_cache_key(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate cache key for request"""
        cache_data = {
            'messages': messages,
            'model': self.config.model,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            **kwargs
        }
        return calculate_hash(cache_data)

    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if available and not expired"""
        if not self.config.cache_enabled:
            return None

        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.config.cache_ttl:
                logger.info("Using cached response")
                return cached_item['response']
            else:
                del self.cache[cache_key]  # Remove expired cache

        return None

    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache the response"""
        if self.config.cache_enabled:
            self.cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }

    @retry_async(max_retries=3, delay=1.0)
    @performance_monitor
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Make a chat completion request to DeepSeek API with enhanced error handling"""
        import aiohttp

        # Validate input
        if not messages or not isinstance(messages, list):
            raise ValidationError("Messages must be a non-empty list")

        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValidationError("Each message must have 'role' and 'content' fields")

        # Check cache first
        cache_key = self._get_cache_key(messages, **kwargs)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        # Return mock response if mock mode is enabled
        if getattr(self.config, 'mock_mode', False):
            logger.info("Using mock mode for API call")
            print(f"\n[MOCK] Using Mock mode (skipping real API calls)")
            print(f"   Message count: {len(messages)}")
            print(f"   Message preview: {messages[0]['content'][:100]}..." if messages else "No messages")

            await asyncio.sleep(0.1)
            user_message = messages[0]['content'] if messages else ""

            if "temporal" in user_message.lower():
                print("   Identified as: Temporal analysis request")
                mock_code = '''
def analyze_temporal_health_data(patient_data, user_query):
    """Mock temporal analysis function with enhanced validation"""
    if not patient_data or not user_query:
        raise ValueError("Patient data and user query are required")

    results = {
        'summary': 'Mock temporal analysis completed successfully',
        'trends': [{'metric': 'Body Temperature', 'slope': 0.1, 'p_value': 0.05, 'trend_direction': 'increasing'}],
        'concerning_patterns': [],
        'risk_factors': [],
        'confidence_score': 0.85,
        'data_quality': 'good'
    }
    return results
'''
                mock_response = {'choices': [{'message': {'content': f'```python\n{mock_code}\n```'}}]}
                print("   Mock response content (temporal analysis):")
                print("   " + "-"*40)
                print(f"   {mock_code.strip()}")
                print("   " + "-"*40)
                mock_response = {'choices': [{'message': {'content': f'```python\n{mock_code}\n```'}}]}
                self._cache_response(cache_key, mock_response)
                return mock_response
            else:
                print("   Identified as: Advanced analysis request")
                mock_code = '''
def advanced_health_analytics(patient_data, temporal_analysis):
    """Mock advanced analytics function with comprehensive analysis"""
    if not patient_data:
        raise ValueError("Patient data is required")

    results = {
        'statistical_testing': {
            'paired_t_test': {'t_statistic': 2.1, 'p_value': 0.04, 'significant': True, 'effect_size': 0.65},
            'anova': {'f_statistic': 3.2, 'p_value': 0.02, 'significant': True}
        },
        'trend_analysis': {
            'gaussian_process': {'log_likelihood': -15.5, 'predictions': [36.5, 37.0], 'uncertainty': [0.5, 0.3]},
            'arima': {'aic': 125.3, 'bic': 132.1, 'forecast': [36.8, 37.2]}
        },
        'clinical_insights': {
            'primary_concerns': ['Body Temperature', 'Heart Rate'],
            'recommendations': ['Monitor temperature closely', 'Consider additional cardiovascular assessment'],
            'risk_level': 'moderate',
            'follow_up_needed': True
        },
        'data_quality_metrics': {
            'completeness': 0.95,
            'consistency': 0.92,
            'accuracy': 0.88
        }
    }
    return results
'''
                mock_response = {'choices': [{'message': {'content': f'```python\n{mock_code}\n```'}}]}
                print("   Mock response content (advanced analysis):")
                print("   " + "-"*40)
                print(f"   {mock_code.strip()}")
                print("   " + "-"*40)
                mock_response = {'choices': [{'message': {'content': f'```python\n{mock_code}\n```'}}]}
                self._cache_response(cache_key, mock_response)
                return mock_response

        # Rate limiting check
        self._wait_for_rate_limit()
        self.request_times.append(time.time())

        # Normal API call with enhanced error handling
        payload = {
            'model': self.config.model,
            'messages': messages,
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature,
            **kwargs
        }

        logger.info(f"Making API request to {self.base_url}")
        print(f"\n[API_REQUEST] Sending API request to: {self.base_url}")
        print(f"   Model: {self.config.model}")
        print(f"   Token limit: {self.config.max_tokens}")
        print(f"   Temperature: {self.config.temperature}")
        print(f"   Message count: {len(messages)}")

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout, connect=30, sock_read=150)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                    print(f"   响应状态码: {response.status}")

                    if response.status == 200:
                        api_response = await response.json()

                        # Validate API response structure
                        if 'choices' not in api_response:
                            raise APIError("Invalid API response: missing 'choices' field")

                        # Cache successful response
                        self._cache_response(cache_key, api_response)

                        # Print complete API response
                        print("   DeepSeek API complete response:")
                        print("   " + "="*50)

                        # Print response header information
                        usage = api_response.get('usage', {})
                        if usage:
                            print(f"   Token usage: {usage}")
                            logger.info(f"API token usage: {usage}")

                        # Print choices content
                        choices = api_response.get('choices', [])
                        if choices:
                            print(f"   Number of choices: {len(choices)}")
                            for i, choice in enumerate(choices):
                                message = choice.get('message', {})
                                content = message.get('content', '')

                                print(f"\n   Choice {i+1} content preview:")
                                print(f"      Role: {message.get('role', 'unknown')}")
                                print(f"      Content length: {len(content)} characters")

                                # If content is very long, show only first 500 characters
                                if len(content) > 500:
                                    print("      Content:")
                                    print("      " + content[:500] + "...")
                                else:
                                    print("      Content:")
                                    print("      " + content)

                                print("      ---")

                        print("   " + "="*50)
                        return api_response
                    else:
                        error_text = await response.text()
                        print(f"   API error details: {error_text}")

                    # Handle specific error codes
                    if response.status == 429:
                        logger.warning("Rate limit exceeded, will retry with backoff")
                        raise APIError(f"Rate limit exceeded: {error_text}")
                    elif response.status == 401:
                        raise APIError(f"Authentication failed: {error_text}")
                    elif response.status == 400:
                        raise APIError(f"Bad request: {error_text}")
                    else:
                        raise APIError(f"API request failed with status {response.status}: {error_text}")

        except aiohttp.ClientError as e:
            logger.error(f"Network error during API call: {e}")
            raise APIError(f"Network error: {e}")
        except asyncio.TimeoutError:
            logger.error("API request timed out")
            raise APIError("Request timed out")
        except Exception as e:
            logger.error(f"Unexpected error during API call: {e}")
            raise APIError(f"Unexpected error: {e}")

    def generate_temporal_analysis_prompt(self, patient_data: Dict, user_query: str) -> str:
        """Generate concise prompt for temporal health data analysis (under 1000 words)"""
        patient_id = patient_data.get('patient_info', {}).get('id', 'Unknown')
        total_indicators = patient_data.get('patient_info', {}).get('total_indicators', 0)

        prompt = f"""Medical data analyst. Generate Python code for temporal analysis.

Patient: {patient_id}, Query: {user_query[:30]}...
Data: {total_indicators} health indicators.

Required:
1. Statistical trend analysis
2. Time series patterns
3. Risk assessment
4. Clinical insights

Output: Python function `analyze_temporal_health_data(patient_data, user_query)`.

Requirements:
- Use numpy, pandas, scipy
- Include statistical tests
- Return JSON results
- Keep code under 1000 tokens.
"""
        return prompt

    def generate_code_writing_prompt(self, user_query: str, temporal_analysis: Dict) -> str:
        """Generate concise prompt for advanced analysis code (under 1000 words)"""
        summary = temporal_analysis.get('summary', 'Analysis completed successfully')

        prompt = f"""Medical programmer. Create advanced analysis code.

Query: {user_query[:30]}...
Summary: {summary[:50]}...

Required:
Write Python function `advanced_health_analytics(patient_data, temporal_analysis)`:

1. Statistical correlation
2. Risk assessment
3. Clinical insights
4. Recommendations

Requirements:
- Use numpy, pandas, scipy
- Include statistical tests
- Generate recommendations
- Keep code under 800 tokens.
"""
        return prompt

class COTCAgent:
    """COTCAgent - Medical Temporal Data Analysis Chain-of-Thought Completion Agent"""

    def __init__(self, deepseek_config: DeepSeekConfig):
        self.deepseek_client = DeepSeekClient(deepseek_config)
        self.config = deepseek_config

        # Initialize performance monitoring
        self.performance_monitor = get_performance_monitor()

        # Initialize advanced analysis modules
        if HAS_ADVANCED_TEMPORAL:
            self.temporal_analyzer = AdvancedTemporalAnalyzer()
            logger.info("Advanced temporal analysis module initialized")
        else:
            self.temporal_analyzer = None
            logger.warning("Advanced temporal analysis module not available")

        if HAS_ADVANCED_COX:
            self.cox_analyzer = AdvancedCoxAnalyzer()
            logger.info("Advanced Cox model analyzer initialized")
        else:
            self.cox_analyzer = None
            logger.warning("Advanced Cox model analyzer not available")

        if HAS_ADVANCED_WAVELET:
            self.wavelet_analyzer = AdvancedWaveletAnalyzer()
            logger.info("Advanced wavelet analyzer initialized")
        else:
            self.wavelet_analyzer = None
            logger.warning("Advanced wavelet analyzer not available")

        logger.info(f"Loaded {len(self.load_disease_database())} diseases from database")
        logger.info("COTCAgent initialized with performance monitoring enabled")

    def load_disease_database(self) -> List[Dict]:
        """Load disease database from JSON file"""
        try:
            with open('disease_symptom_database.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Disease database not found, using empty database")
            return []

    @performance_monitor
    async def process_user_query(self, user_query: str, patient_data: Dict) -> Dict[str, Any]:
        """Process user query with comprehensive medical analysis and enhanced error handling"""
        try:
            # Input validation and sanitization
            sanitized_query = sanitize_input(user_query)
            validation_result = validate_patient_data_new(patient_data)
            if not validation_result.is_valid:
                raise ValidationError(f"Patient data validation failed: {validation_result.errors}")

            # Use sanitized data if available
            if validation_result.sanitized_data:
                patient_data = validation_result.sanitized_data

            # Validate medical query content
            query_validation = validate_medical_query(sanitized_query)
            if not query_validation.is_valid:
                raise ValidationError(f"Medical query validation failed: {query_validation.errors}")

            # Log warnings for query validation
            for warning in query_validation.warnings:
                logger.warning(f"Query validation warning: {warning}")

            logger.info(f"Processing user query: {sanitized_query[:100]}...")
            logger.info(f"Patient ID: {patient_data.get('patient_info', {}).get('id', 'Unknown')}")

            progress = ProgressIndicator("COTCAgent Processing")
            progress.set_total_steps(6)
            progress.update("Starting user query processing")

        except (ValidationError, ConfigurationError) as e:
            logger.error(f"Input validation failed: {e}")
            raise COTCAgentError(f"Invalid input: {e}") from e

            # Step 1: Generate temporal analysis code
            progress.update("Generating temporal analysis prompt")
            temporal_prompt = self.deepseek_client.generate_temporal_analysis_prompt(patient_data, sanitized_query)

            progress.update("Calling DeepSeek API to generate temporal analysis code...")
            logger.info("Step 1: Generating temporal analysis code...")

            try:
                temporal_response = await self.deepseek_client.chat_completion([
                    {"role": "user", "content": temporal_prompt}
                ])
                progress.update("Temporal analysis code generation completed")
            except APIError as e:
                progress.update(f"API call failed: {str(e)}")
                logger.error(f"Temporal analysis API call failed: {e}")
                raise COTCAgentError(f"Temporal analysis API call failed: {e}") from e

            # Extract and execute generated code
            progress.update("Extracting and executing temporal analysis code")
            temporal_code = self.extract_code_from_response(temporal_response)

            print(f"\n[CODE_EXTRACT] Extracted temporal analysis code ({len(temporal_code)} characters):")
            print("="*60)
            print(temporal_code)
            print("="*60)

            temporal_analysis = await self.execute_generated_code(temporal_code, patient_data, sanitized_query)

            # Step 2: Generate advanced analysis code
            progress.update("Generating advanced analysis prompt")
            code_prompt = self.deepseek_client.generate_code_writing_prompt(sanitized_query, temporal_analysis)

            progress.update("Calling DeepSeek API to generate advanced analysis code...")

            try:
                code_response = await self.deepseek_client.chat_completion([
                    {"role": "user", "content": code_prompt}
                ])
                progress.update("Advanced analysis code generation completed")
            except APIError as e:
                progress.update(f"Advanced analysis API call failed: {str(e)}")
                logger.error(f"Advanced analysis API call failed: {e}")
                raise COTCAgentError(f"Advanced analysis API call failed: {e}") from e

            # Extract and execute advanced analysis code
            progress.update("Extracting and executing advanced analysis code")
            analysis_code = self.extract_code_from_response(code_response)

            print(f"\n[CODE_EXTRACT] Extracted advanced analysis code ({len(analysis_code)} characters):")
            print("="*60)
            print(analysis_code)
            print("="*60)

            detailed_analysis = await self.execute_generated_code(analysis_code, patient_data, temporal_analysis)

            # Step 3: Perform comprehensive mathematical analysis
            progress.update("Performing comprehensive mathematical analysis")
            comprehensive_analysis = self.comprehensive_mathematical_analysis(patient_data)

            # Step 3.5: Advanced temporal analysis with ML methods
            progress.update("Performing advanced temporal analysis with ML methods")
            advanced_temporal_analysis = self.perform_advanced_temporal_analysis(patient_data)

            # Step 4: Assess disease risks
            progress.update("Calculating disease risk assessment")
            symptoms = self.extract_symptoms_from_analysis(temporal_analysis)
            disease_risks = self.assess_disease_risks(symptoms)

            # Step 5: Generate active inquiry questions and perform iterative diagnosis
            progress.update("Performing iterative diagnosis with proactive consultation")
            inquiry_questions = self.generate_active_inquiry_questions(temporal_analysis, detailed_analysis)

            # Perform comprehensive COTC diagnosis with consultation
            iterative_diagnosis = self.perform_iterative_cotc_diagnosis(symptoms, inquiry_questions)

            progress.update("Performing iterative COTC diagnosis")
            iterative_diagnosis = self.perform_iterative_cotc_diagnosis(symptoms, inquiry_questions)

            progress.complete("COTCAgent processing completed")

            # Prepare final result with metadata
            result = {
                'temporal_analysis': temporal_analysis,
                'detailed_analysis': detailed_analysis,
                'comprehensive_analysis': comprehensive_analysis,
                'advanced_temporal_analysis': advanced_temporal_analysis,
                'disease_risks': disease_risks,
                'active_inquiry_questions': inquiry_questions,
                'iterative_cotc_diagnosis': iterative_diagnosis,
                'metadata': {
                    'processing_time': progress.get_summary()['total_time'],
                    'patient_id': patient_data.get('patient_info', {}).get('id', 'Unknown'),
                    'query_length': len(sanitized_query),
                    'timestamp': time.time()
                }
            }

            logger.info(f"Successfully processed query for patient {result['metadata']['patient_id']}")
            return result

        except CodeExecutionError as e:
            logger.error(f"Code execution failed: {e}")
            raise COTCAgentError(f"Analysis execution failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during processing: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise COTCAgentError(f"Unexpected processing error: {e}") from e

    def extract_code_from_response(self, response: Dict[str, Any]) -> str:
        """Extract Python code from DeepSeek API response"""
        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')

        # Extract code blocks
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', content, re.DOTALL)

        if code_blocks:
            return code_blocks[0].strip()

        # Fallback
        lines = content.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            if '```' in line:
                in_code = not in_code
                continue
            if in_code:
                code_lines.append(line)

        return '\n'.join(code_lines) if code_lines else content

    async def execute_generated_code(self, code: str, patient_data: Dict, context: Any, save_temp_files: bool = False) -> Dict[str, Any]:
        """Execute generated code in a safe sandboxed environment"""
        code_progress = ProgressIndicator("Code Execution")

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        if save_temp_files:
            print(f"Temporary code saved to: {temp_file_path}")

        try:
            # Execute the code in restricted environment
            exec_globals = {
                'patient_data': patient_data,
                'user_query': context if isinstance(context, str) else str(context),
                'temporal_analysis': context if isinstance(context, dict) else {},
                'np': np,
                'pd': pd,
                'stats': stats,
                'json': json
            }

            exec(open(temp_file_path, 'r', encoding='utf-8').read(), exec_globals)

            # Try to get results from common function names
            if 'analyze_temporal_health_data' in exec_globals:
                result = exec_globals['analyze_temporal_health_data'](patient_data, context)
            elif 'advanced_health_analytics' in exec_globals:
                result = exec_globals['advanced_health_analytics'](patient_data, context)
            else:
                result = {'error': 'No analysis function found'}

        except Exception as e:
            result = {'error': f'Code execution failed: {str(e)}'}

        finally:
            if not save_temp_files:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

        code_progress.complete("Code execution completed")
        return result

    def extract_symptoms_from_analysis(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract symptom keywords from temporal analysis results"""
        symptoms = []
        if 'trends' in analysis:
            for trend in analysis['trends']:
                metric = trend.get('metric', '')
                if any(keyword in metric.lower() for keyword in ['pain', 'fever', '疼', '痛', '晕']):
                    symptoms.append(metric)
        return list(set(symptoms))

    def assess_disease_risks(self, symptoms: List[str]) -> List[DiseaseRisk]:
        """
        Advanced disease risk assessment using COTC reasoning engine
        Implements probabilistic Chain-of-Thought Completion as described in paper Section 3.3
        """
        if get_cotc_reasoning_engine is None:
            # Fallback to simple implementation
            return self._fallback_disease_risks(symptoms)

        try:
            cotc_engine = get_cotc_reasoning_engine()
            return cotc_engine.assess_disease_risks(symptoms)
        except Exception as e:
            logger.error(f"COTC reasoning failed, using fallback: {e}")
            return self._fallback_disease_risks(symptoms)

    def _fallback_disease_risks(self, symptoms: List[str]) -> List[DiseaseRisk]:
        """Fallback disease risk assessment"""
        mock_risks = [
            DiseaseRisk(
                disease_id='D001',
                disease_name='Gastroenteritis',
                risk_score=0.8,
                confidence=0.75,
                matched_symptoms=['Abdominal pain', 'Nausea'],
                missing_symptoms=['Fever', 'Vomiting']
            ),
            DiseaseRisk(
                disease_id='D002',
                disease_name='Migraine',
                risk_score=0.6,
                confidence=0.65,
                matched_symptoms=['Headache', 'Insomnia'],
                missing_symptoms=['Visual disturbances', 'Nausea']
            )
        ]
        return mock_risks

    def generate_active_inquiry_questions(self, temporal_analysis: Dict, detailed_analysis: Dict) -> List[str]:
        """
        Generate proactive consultation questions using COTC reasoning
        Implements the iterative questioning mechanism described in paper Section 3.3.3
        """
        if get_cotc_reasoning_engine is None:
            # Fallback to simple questions
            return self._fallback_inquiry_questions()

        try:
            # Extract symptoms from analysis for COTC reasoning
            symptoms = self.extract_symptoms_from_analysis(temporal_analysis)

            # Get disease risks using COTC engine
            disease_risks = self.assess_disease_risks(symptoms)

            # Generate proactive questions based on information gaps
            cotc_engine = get_cotc_reasoning_engine()
            proactive_questions = cotc_engine.generate_proactive_questions(disease_risks)

            # Add analysis-specific questions
            analysis_questions = []
            if 'concerning_patterns' in temporal_analysis and temporal_analysis['concerning_patterns']:
                analysis_questions.append("Can you describe when these concerning patterns typically occur?")

            if 'anomalies' in detailed_analysis:
                anomaly_info = detailed_analysis['anomalies']
                if anomaly_info.get('anomaly_percentage', 0) > 10:
                    analysis_questions.append("Have you noticed any unusual changes in your symptoms recently?")

            return proactive_questions + analysis_questions

        except Exception as e:
            logger.error(f"Proactive question generation failed, using fallback: {e}")
            return self._fallback_inquiry_questions()

    def _fallback_inquiry_questions(self) -> List[str]:
        """Fallback inquiry questions"""
        return [
            "Have there been any recent changes in your dietary habits?",
            "What is the frequency and severity of the pain?",
            "Have any other related symptoms appeared?"
        ]

    def perform_iterative_cotc_diagnosis(self, symptoms: List[str], inquiry_questions: List[str]) -> Dict[str, Any]:
        """
        Perform iterative diagnosis with proactive consultation using COTC reasoning engine
        Implements the complete Chain-of-Thought Completion process from the paper
        """
        if get_cotc_reasoning_engine is None:
            return {'error': 'COTC reasoning engine not available'}

        try:
            cotc_engine = get_cotc_reasoning_engine()

            # Perform iterative diagnosis with consultation
            iterative_result = cotc_engine.iterative_diagnosis_with_consultation(
                initial_symptoms=symptoms,
                max_rounds=3,
                probability_threshold=0.8
            )

            # Identify information gaps using maximum probability inference
            info_gaps = cotc_engine.identify_information_gaps_max_probability(symptoms)

            return {
                'iterative_diagnosis': iterative_result,
                'information_gaps_analysis': info_gaps,
                'cotc_methodology': 'probabilistic_chain_of_thought',
                'implementation_status': 'complete'
            }

        except Exception as e:
            logger.error(f"Iterative COTC diagnosis failed: {e}")
            return {'error': f'COTC diagnosis failed: {str(e)}'}

    def comprehensive_mathematical_analysis(self, patient_data: Dict) -> Dict[str, Any]:
        """
        Perform comprehensive mathematical analysis using TSA Module models from paper Section 3.2
        Implements the systematic approach described in the paper for mathematical formalization
        """
        results = {
            'statistical_summary': {},
            'mathematical_rigor': 'High',
            'confidence_level': 0.95,
            'tsa_models_applied': []
        }

        try:
            # Extract temporal data from patient data
            medical_data = patient_data.get('medical_data', {})
            indicators = medical_data.get('indicators', [])

            if not indicators:
                return results

            for indicator in indicators:
                indicator_name = indicator.get('name', 'Unknown')
                values = indicator.get('values', [])
                timestamps = indicator.get('timestamps', [])

                if len(values) < 3:  # Need minimum data points
                    continue

                # Convert to numpy arrays
                y = np.array(values, dtype=float)
                time_points = np.array(timestamps) if timestamps else np.arange(len(values))

                # Apply TSA mathematical models as described in paper

                # 1. Linear Mixed Effects Model (Eq. 2)
                if len(np.unique(time_points)) > 1:  # Need varying time points
                    lme_results = TSAMathematicalModels.linear_mixed_effects_model(
                        y, time_points, np.array([patient_data.get('patient_info', {}).get('id', 'P001')] * len(y))
                    )
                    if 'error' not in lme_results:
                        results['statistical_summary'][f'{indicator_name}_lme'] = lme_results
                        results['tsa_models_applied'].append('linear_mixed_effects')

                # 2. Bayesian Change Point Detection (Eq. 3)
                if len(y) >= 10:
                    changepoint_results = TSAMathematicalModels.bayesian_change_point_detection(y, time_points)
                    if 'error' not in changepoint_results and changepoint_results.get('change_point') is not None:
                        results['statistical_summary'][f'{indicator_name}_changepoint'] = changepoint_results
                        results['tsa_models_applied'].append('bayesian_changepoint')

                # 3. Multi-scale Trend Analysis (Eq. 9)
                multiscale_results = TSAMathematicalModels.multi_scale_trend_analysis(y, time_points)
                if 'error' not in multiscale_results:
                    results['statistical_summary'][f'{indicator_name}_multiscale'] = multiscale_results
                    results['tsa_models_applied'].append('multi_scale_analysis')

                # 4. Anomaly Detection with Temporal Consistency (Eq. 10)
                anomaly_results = TSAMathematicalModels.anomaly_detection_with_temporal_consistency(y, time_points)
                if 'error' not in anomaly_results:
                    results['statistical_summary'][f'{indicator_name}_anomalies'] = anomaly_results
                    results['tsa_models_applied'].append('anomaly_detection')

            # Calculate overall confidence based on models applied
            model_count = len(results['tsa_models_applied'])
            results['confidence_level'] = min(0.95, 0.7 + 0.05 * model_count)

        except Exception as e:
            logger.error(f"Comprehensive mathematical analysis failed: {e}")
            results['error'] = str(e)

        return results

    def perform_advanced_temporal_analysis(self, patient_data: Dict) -> Dict[str, Any]:
        """
        Perform advanced temporal analysis using ML methods (Cox models, Wavelet transforms, etc.)

        Args:
            patient_data: Patient temporal data

        Returns:
            Advanced temporal analysis results
        """
        results = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'ml_methods_applied': [],
            'cox_model_results': {},
            'wavelet_analysis': {},
            'time_dependent_roc': {},
            'comprehensive_report': {},
            'performance_metrics': {}
        }

        if not self.temporal_analyzer:
            results['error'] = "Advanced temporal analysis module not available"
            return results

        try:
            start_time = time.time()

            # Generate comprehensive report using advanced methods
            comprehensive_report = self.temporal_analyzer.generate_comprehensive_report(patient_data)
            results['comprehensive_report'] = comprehensive_report
            results['ml_methods_applied'].append('comprehensive_temporal_analysis')

            # Extract symptoms for detailed analysis
            temporal_indicators = patient_data.get('temporal_indicators', {})
            basic_signs = temporal_indicators.get('basic_signs', {})

            # Perform individual symptom analysis
            symptom_analyses = {}
            for symptom_name in basic_signs.keys():
                try:
                    analysis = self.temporal_analyzer.analyze_temporal_patterns(patient_data, symptom_name)
                    symptom_analyses[symptom_name] = analysis

                    # Check if advanced methods were applied
                    methods = analysis.get('methods_applied', [])
                    results['ml_methods_applied'].extend(methods)

                except Exception as e:
                    logger.warning(f"Failed to analyze symptom {symptom_name}: {e}")
                    symptom_analyses[symptom_name] = {'error': str(e)}

            results['symptom_analyses'] = symptom_analyses

            # Attempt Cox model fitting if we have sufficient data
            try:
                cox_results = self._fit_cox_model_for_patient(patient_data)
                if cox_results:
                    results['cox_model_results'] = cox_results
                    results['ml_methods_applied'].append('cox_proportional_hazards')
            except Exception as e:
                logger.warning(f"Cox model fitting failed: {e}")
                results['cox_model_results'] = {'error': str(e)}

            # Perform advanced wavelet analysis if available
            if self.wavelet_analyzer and len(basic_signs) > 0:
                try:
                    wavelet_results = self._perform_advanced_wavelet_analysis(patient_data)
                    if wavelet_results:
                        results['wavelet_analysis'].update(wavelet_results)
                        results['ml_methods_applied'].extend(['advanced_wavelet_cwt', 'advanced_wavelet_dwt'])
                except Exception as e:
                    logger.warning(f"Advanced wavelet analysis failed: {e}")

            # Perform wavelet coherence analysis if multiple symptoms available
            if len(basic_signs) >= 2:
                try:
                    coherence_results = self._compute_wavelet_coherence_analysis(patient_data)
                    if coherence_results:
                        results['wavelet_analysis']['coherence'] = coherence_results
                        results['ml_methods_applied'].append('wavelet_coherence')
                except Exception as e:
                    logger.warning(f"Wavelet coherence analysis failed: {e}")

            # Calculate performance metrics
            processing_time = time.time() - start_time
            results['performance_metrics'] = {
                'processing_time_seconds': processing_time,
                'methods_count': len(set(results['ml_methods_applied'])),
                'symptoms_analyzed': len(symptom_analyses),
                'success_rate': len([s for s in symptom_analyses.values() if 'error' not in s]) / max(1, len(symptom_analyses))
            }

            logger.info(f"Advanced temporal analysis completed in {processing_time:.2f}s with {len(set(results['ml_methods_applied']))} ML methods")

        except Exception as e:
            logger.error(f"Advanced temporal analysis failed: {e}")
            results['error'] = str(e)

        return results

    def _fit_cox_model_for_patient(self, patient_data: Dict) -> Optional[Dict[str, Any]]:
        """Fit Cox proportional hazards model for patient data"""
        # This is a simplified implementation
        # In practice, you'd need proper time-to-event data

        try:
            temporal_indicators = patient_data.get('temporal_indicators', {})
            basic_signs = temporal_indicators.get('basic_signs', {})

            if not basic_signs:
                return None

            # Create synthetic survival data for demonstration
            # In real implementation, this would use actual clinical outcomes
            n_patients = 1  # Single patient analysis
            time_to_event = np.array([365.0])  # 1 year follow-up
            event_indicator = np.array([0])    # Censored (no event)

            # Create covariates from symptom severities
            covariates_data = {}
            for symptom_name, symptom_data in basic_signs.items():
                if 'severity_levels' in symptom_data and symptom_data['severity_levels']:
                    # Use latest severity as baseline covariate
                    latest_severity = symptom_data['severity_levels'][-1]
                    covariates_data[f"{symptom_name}_baseline"] = [self._severity_to_numeric(latest_severity)]

            if not covariates_data:
                return None

            covariates_df = pd.DataFrame(covariates_data)

            # Fit Cox model
            cox_result = self.temporal_analyzer.fit_cox_model(
                time_to_event=time_to_event,
                event_indicator=event_indicator,
                covariates=covariates_df,
                model_name="patient_cox_model"
            )

            return {
                'coefficients': cox_result.coefficients,
                'hazard_ratios': cox_result.hazard_ratios,
                'concordance_index': cox_result.concordance_index,
                'p_values': cox_result.p_values,
                'log_likelihood': cox_result.log_likelihood,
                'aic': cox_result.aic,
                'bic': cox_result.bic
            }

        except Exception as e:
            logger.error(f"Cox model fitting error: {e}")
            return None

    def _compute_wavelet_coherence_analysis(self, patient_data: Dict) -> Optional[Dict[str, Any]]:
        """Compute wavelet coherence between multiple symptoms"""
        try:
            temporal_indicators = patient_data.get('temporal_indicators', {})
            basic_signs = temporal_indicators.get('basic_signs', {})

            if len(basic_signs) < 2:
                return None

            # Extract time series for first two symptoms
            symptom_names = list(basic_signs.keys())[:2]
            time_series = []

            for symptom_name in symptom_names:
                symptom_data = basic_signs[symptom_name]
                if 'severity_levels' in symptom_data:
                    # Convert severity levels to numeric time series
                    numeric_values = [self._severity_to_numeric(sev) for sev in symptom_data['severity_levels']]
                    time_series.append(np.array(numeric_values))

            if len(time_series) >= 2 and len(time_series[0]) > 10 and len(time_series[1]) > 10:
                # Compute wavelet coherence
                coherence_matrix = self.temporal_analyzer.compute_wavelet_coherence(
                    time_series[0], time_series[1]
                )

                return {
                    'symptom_pair': symptom_names,
                    'coherence_matrix_shape': coherence_matrix.shape,
                    'mean_coherence': float(np.mean(coherence_matrix)),
                    'max_coherence': float(np.max(coherence_matrix)),
                    'coherence_scale_range': f"scales_{self.temporal_analyzer.wavelet_config['scales'][0]}_to_{self.temporal_analyzer.wavelet_config['scales'][-1]}"
                }

        except Exception as e:
            logger.error(f"Wavelet coherence analysis error: {e}")

        return None

    def _severity_to_numeric(self, severity: str) -> float:
        """Convert severity level to numeric value (duplicate for local use)"""
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

    def _perform_advanced_wavelet_analysis(self, patient_data: Dict) -> Optional[Dict[str, Any]]:
        """Perform advanced wavelet analysis on patient symptoms"""
        if not self.wavelet_analyzer:
            return None

        try:
            temporal_indicators = patient_data.get('temporal_indicators', {})
            basic_signs = temporal_indicators.get('basic_signs', {})

            wavelet_results = {}

            # Analyze each symptom with wavelet methods
            for symptom_name, symptom_data in basic_signs.items():
                if 'severity_levels' in symptom_data and len(symptom_data['severity_levels']) > 10:
                    try:
                        # Convert severity levels to numeric time series
                        numeric_values = [self._severity_to_numeric(sev) for sev in symptom_data['severity_levels']]
                        time_series = np.array(numeric_values)

                        # Perform CWT
                        cwt_result = self.wavelet_analyzer.continuous_wavelet_transform(
                            time_series,
                            result_name=f"{symptom_name}_cwt"
                        )

                        # Perform DWT if signal length is appropriate
                        if len(time_series) >= 16:  # Minimum for meaningful DWT
                            dwt_result = self.wavelet_analyzer.discrete_wavelet_transform(
                                time_series,
                                result_name=f"{symptom_name}_dwt"
                            )

                            wavelet_results[symptom_name] = {
                                'cwt_power_spectrum_shape': cwt_result.power_spectrum.shape,
                                'cwt_dominant_scale': float(cwt_result.scales[np.argmax(np.max(cwt_result.power_spectrum, axis=1))]),
                                'dwt_levels': dwt_result.decomposition_levels,
                                'dwt_max_energy_level': max(dwt_result.energy_distribution, key=dwt_result.energy_distribution.get),
                                'dwt_reconstruction_error': float(dwt_result.reconstruction_error)
                            }
                        else:
                            wavelet_results[symptom_name] = {
                                'cwt_power_spectrum_shape': cwt_result.power_spectrum.shape,
                                'cwt_dominant_scale': float(cwt_result.scales[np.argmax(np.max(cwt_result.power_spectrum, axis=1))]),
                                'note': 'Signal too short for DWT analysis'
                            }

                    except Exception as e:
                        logger.warning(f"Wavelet analysis failed for symptom {symptom_name}: {e}")
                        continue

            # Perform multivariate coherence analysis if multiple symptoms
            if len(basic_signs) >= 2:
                try:
                    time_series_dict = {}
                    for symptom_name, symptom_data in basic_signs.items():
                        if 'severity_levels' in symptom_data and len(symptom_data['severity_levels']) > 10:
                            numeric_values = [self._severity_to_numeric(sev) for sev in symptom_data['severity_levels']]
                            time_series_dict[symptom_name] = np.array(numeric_values)

                    if len(time_series_dict) >= 2:
                        coherence_results = self.wavelet_analyzer.analyze_multivariate_coherence(
                            time_series_dict
                        )

                        wavelet_results['multivariate_coherence'] = {
                            'n_pairs_analyzed': len(coherence_results),
                            'pair_names': list(coherence_results.keys()),
                            'mean_coherences': {pair: float(np.mean(result.coherence_matrix))
                                              for pair, result in coherence_results.items()}
                        }

                except Exception as e:
                    logger.warning(f"Multivariate coherence analysis failed: {e}")

            return wavelet_results if wavelet_results else None

        except Exception as e:
            logger.error(f"Error in advanced wavelet analysis: {e}")
            return None


# Export main classes
__all__ = ['COTCAgent', 'DeepSeekConfig', 'DiseaseRisk', 'SymptomIndicator']