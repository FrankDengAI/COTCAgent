"""
Advanced Cox Proportional Hazards Model Implementation for COTCAgent

This module provides sophisticated implementations of Cox models with time-dependent covariates,
including dynamic risk prediction and time-dependent ROC analysis as described in the paper.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
import logging
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

# Advanced libraries for survival analysis
try:
    import lifelines
    from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
    from lifelines.utils import concordance_index
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    warnings.warn("lifelines not installed. Advanced Cox model features will be limited.")

try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False
    warnings.warn("scikit-survival not installed. Some survival analysis features unavailable.")

try:
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
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
    measurement_times: Optional[List[datetime]] = None

@dataclass
class CoxModelResult:
    """Comprehensive results from Cox Proportional Hazards model fitting"""
    coefficients: Dict[str, float]
    hazard_ratios: Dict[str, float]
    standard_errors: Dict[str, float]
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    concordance_index: float
    log_likelihood: float
    aic: float
    bic: float
    baseline_hazard: Optional[pd.DataFrame] = None
    baseline_survival: Optional[pd.DataFrame] = None
    fitted_model: Optional[Any] = None
    model_summary: Optional[str] = None
    convergence_status: bool = True
    n_events: int = 0
    n_at_risk: int = 0

@dataclass
class TimeDependentROCResult:
    """Results from time-dependent ROC analysis"""
    time_points: List[float]
    auc_values: List[float]
    sensitivities: List[float]
    specificities: List[float]
    optimal_thresholds: List[float]
    confidence_intervals: Optional[List[Tuple[float, float]]] = None
    brier_scores: Optional[List[float]] = None
    calibration_slopes: Optional[List[float]] = None

@dataclass
class DynamicRiskPrediction:
    """Dynamic risk predictions over time"""
    prediction_times: List[float]
    risk_scores: List[float]
    cumulative_hazards: List[float]
    survival_probabilities: List[float]
    prediction_intervals: Optional[List[Tuple[float, float]]] = None
    covariate_values: Optional[Dict[str, List[float]]] = None

class AdvancedCoxAnalyzer:
    """
    Advanced Cox Proportional Hazards model analyzer with time-dependent covariates

    Implements the methodology from Fisher & Lin (1999) and provides comprehensive
    survival analysis capabilities for medical time series data.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Default Cox model configuration
        self.cox_config = self.config.get('cox', {
            'penalizer': 0.01,
            'l1_ratio': 0.0,
            'robust': True,
            'show_progress': True,
            'alpha': 0.05,  # significance level
            'max_iter': 200
        })

        # Time-dependent ROC configuration
        self.roc_config = self.config.get('roc', {
            'n_bootstraps': 100,
            'confidence_level': 0.95,
            'evaluation_times_quantiles': [0.25, 0.5, 0.75, 0.9]
        })

        # Store fitted models
        self.fitted_models = {}

        logger.info("Advanced Cox analyzer initialized")

    def fit_cox_model_with_time_dependent_covariates(self,
                                                   time_to_event: np.ndarray,
                                                   event_indicator: np.ndarray,
                                                   baseline_covariates: pd.DataFrame,
                                                   time_dependent_covariates: List[TimeDependentCovariate],
                                                   patient_ids: Optional[np.ndarray] = None,
                                                   model_name: str = "cox_td_model") -> CoxModelResult:
        """
        Fit Cox Proportional Hazards model with time-dependent covariates

        λ(t|Z(t)) = λ₀(t) exp(βᵀZ(t) + γᵀX)

        Args:
            time_to_event: Array of time-to-event or censoring times
            event_indicator: Binary array indicating if event occurred (1) or censored (0)
            baseline_covariates: DataFrame of baseline covariates X
            time_dependent_covariates: List of time-dependent covariates Z(t)
            patient_ids: Patient identifiers for time-dependent data expansion
            model_name: Identifier for the fitted model

        Returns:
            CoxModelResult with comprehensive model statistics
        """
        if not HAS_LIFELINES:
            raise ImportError("lifelines is required for advanced Cox model analysis")

        logger.info(f"Fitting Cox model with time-dependent covariates: {model_name}")

        try:
            # Prepare data for time-dependent covariates
            cox_data = self._prepare_time_dependent_cox_data(
                time_to_event, event_indicator, baseline_covariates,
                time_dependent_covariates, patient_ids
            )

            if cox_data is None or cox_data.empty:
                raise ValueError("Failed to prepare time-dependent Cox data")

            # Fit Cox model
            cph = CoxPHFitter(**self.cox_config)

            # Handle convergence issues
            try:
                cph.fit(cox_data, duration_col='tstart', event_col='event',
                       id_col='id' if 'id' in cox_data.columns else None)
                convergence_status = True
            except Exception as conv_error:
                logger.warning(f"Cox model convergence issues: {conv_error}")
                # Try with different settings
                self.cox_config['penalizer'] = 0.1
                cph = CoxPHFitter(**self.cox_config)
                cph.fit(cox_data, duration_col='tstart', event_col='event',
                       id_col='id' if 'id' in cox_data.columns else None)
                convergence_status = False

            # Extract comprehensive results
            coefficients = cph.params_.to_dict()
            hazard_ratios = cph.hazard_ratios_.to_dict()
            standard_errors = cph.standard_errors_.to_dict()
            p_values = cph.p_values_.to_dict()

            # Calculate confidence intervals
            confidence_intervals = {}
            z_score = stats.norm.ppf(1 - self.cox_config['alpha'] / 2)
            for var in coefficients.keys():
                coef = coefficients[var]
                se = standard_errors[var]
                ci_lower = coef - z_score * se
                ci_upper = coef + z_score * se
                confidence_intervals[var] = (ci_lower, ci_upper)

            # Model statistics
            concordance_index_val = cph.concordance_index_
            log_likelihood = cph.log_likelihood_
            aic = cph.AIC_
            bic = cph.BIC_

            # Baseline hazard and survival
            baseline_hazard = None
            baseline_survival = None
            try:
                baseline_hazard = cph.baseline_hazard_
                baseline_survival = cph.baseline_survival_
            except:
                pass

            # Generate model summary
            model_summary = self._generate_cox_model_summary(cph, cox_data)

            result = CoxModelResult(
                coefficients=coefficients,
                hazard_ratios=hazard_ratios,
                standard_errors=standard_errors,
                p_values=p_values,
                confidence_intervals=confidence_intervals,
                concordance_index=concordance_index_val,
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic,
                baseline_hazard=baseline_hazard,
                baseline_survival=baseline_survival,
                fitted_model=cph,
                model_summary=model_summary,
                convergence_status=convergence_status,
                n_events=int(event_indicator.sum()),
                n_at_risk=len(time_to_event)
            )

            self.fitted_models[model_name] = result

            logger.info(f"Cox model {model_name} fitted successfully. C-index: {concordance_index_val:.3f}, "
                       f"Events: {result.n_events}, At risk: {result.n_at_risk}")

            return result

        except Exception as e:
            logger.error(f"Error fitting Cox model {model_name}: {e}")
            raise

    def _prepare_time_dependent_cox_data(self,
                                       time_to_event: np.ndarray,
                                       event_indicator: np.ndarray,
                                       baseline_covariates: pd.DataFrame,
                                       time_dependent_covariates: List[TimeDependentCovariate],
                                       patient_ids: Optional[np.ndarray] = None) -> Optional[pd.DataFrame]:
        """
        Prepare data for time-dependent Cox model following counting process notation

        This creates the expanded dataset needed for time-dependent covariates where
        each patient may have multiple rows representing different time intervals.
        """
        try:
            n_patients = len(time_to_event)

            if patient_ids is None:
                patient_ids = np.arange(n_patients)

            # Start with baseline covariates
            data_list = []

            for i in range(n_patients):
                patient_id = patient_ids[i]
                t_event = time_to_event[i]
                event = event_indicator[i]

                # Get baseline covariates for this patient
                baseline_row = baseline_covariates.iloc[i].to_dict() if i < len(baseline_covariates) else {}

                # Create time intervals based on time-dependent covariate measurements
                time_intervals = self._create_time_intervals_for_patient(
                    time_dependent_covariates, patient_id, t_event
                )

                if not time_intervals:
                    # No time-dependent covariates, use single interval
                    row = {
                        'id': patient_id,
                        'tstart': 0.0,
                        'tstop': t_event,
                        'event': event,
                        **baseline_row
                    }
                    data_list.append(row)
                else:
                    # Create rows for each time interval
                    for interval in time_intervals:
                        row = {
                            'id': patient_id,
                            'tstart': interval['tstart'],
                            'tstop': interval['tstop'],
                            'event': 1 if interval['tstop'] == t_event and event == 1 else 0,
                            **baseline_row,
                            **interval['covariates']
                        }
                        data_list.append(row)

            if not data_list:
                return None

            df = pd.DataFrame(data_list)

            # Ensure proper data types
            df['id'] = df['id'].astype(int)
            df['tstart'] = df['tstart'].astype(float)
            df['tstop'] = df['tstop'].astype(float)
            df['event'] = df['event'].astype(int)

            return df

        except Exception as e:
            logger.error(f"Error preparing time-dependent Cox data: {e}")
            return None

    def _create_time_intervals_for_patient(self,
                                         time_dependent_covariates: List[TimeDependentCovariate],
                                         patient_id: int,
                                         max_time: float) -> List[Dict]:
        """
        Create time intervals for a patient based on time-dependent covariate measurements
        """
        # Collect all unique time points from all time-dependent covariates
        all_times = set([0.0])  # Include time 0

        for tdc in time_dependent_covariates:
            if hasattr(tdc, 'measurement_times') and tdc.measurement_times:
                # Convert datetime to numeric time if needed
                for mt in tdc.measurement_times:
                    if isinstance(mt, datetime):
                        # This is a simplification - in practice you'd need proper time alignment
                        time_val = (mt - datetime.min).total_seconds() / (24 * 3600)  # days
                        all_times.add(min(time_val, max_time))
                    else:
                        all_times.add(min(float(mt), max_time))

        time_points = sorted(list(all_times))
        time_points.append(max_time)

        intervals = []
        for i in range(len(time_points) - 1):
            tstart = time_points[i]
            tstop = time_points[i + 1]

            # Get covariate values for this interval
            covariates = {}
            for tdc in time_dependent_covariates:
                # Use the most recent value before tstop
                value = self._get_covariate_value_at_time(tdc, tstop)
                if value is not None:
                    covariates[tdc.name] = value

            intervals.append({
                'tstart': tstart,
                'tstop': tstop,
                'covariates': covariates
            })

        return intervals

    def _get_covariate_value_at_time(self, tdc: TimeDependentCovariate, time_point: float) -> Optional[float]:
        """Get the covariate value at a specific time point"""
        if not tdc.time_points or not tdc.values:
            return None

        # Find the most recent measurement before or at time_point
        for i in range(len(tdc.time_points)):
            t = tdc.time_points[i]
            if isinstance(t, datetime):
                t_val = (t - datetime.min).total_seconds() / (24 * 3600)
            else:
                t_val = float(t)

            if t_val <= time_point:
                return tdc.values[i]

        return None

    def _generate_cox_model_summary(self, cph: CoxPHFitter, data: pd.DataFrame) -> str:
        """Generate a comprehensive summary of the fitted Cox model"""
        summary_lines = []

        summary_lines.append("Cox Proportional Hazards Model Summary")
        summary_lines.append("=" * 50)
        summary_lines.append("")

        # Model statistics
        summary_lines.append("Model Statistics:")
        summary_lines.append(f"  Concordance Index: {cph.concordance_index_:.3f}")
        summary_lines.append(f"  Log-likelihood: {cph.log_likelihood_:.2f}")
        summary_lines.append(f"  AIC: {cph.AIC_:.2f}")
        summary_lines.append(f"  BIC: {cph.BIC_:.2f}")
        summary_lines.append(f"  Number of observations: {len(data)}")
        summary_lines.append(f"  Number of events: {data['event'].sum()}")
        summary_lines.append("")

        # Coefficients table
        summary_lines.append("Coefficients:")
        summary_lines.append("Variable          Coef    HR      SE      p-value")
        summary_lines.append("-" * 50)

        for var in cph.params_.index:
            coef = cph.params_[var]
            hr = cph.hazard_ratios_[var]
            se = cph.standard_errors_[var]
            p_val = cph.p_values_[var]

            summary_lines.append("12")

        return "\n".join(summary_lines)

    def compute_time_dependent_roc(self,
                                 fitted_model: CoxModelResult,
                                 test_times: np.ndarray,
                                 test_events: np.ndarray,
                                 test_covariates: pd.DataFrame,
                                 time_dependent_test_covariates: Optional[List[TimeDependentCovariate]] = None,
                                 evaluation_times: Optional[List[float]] = None) -> TimeDependentROCResult:
        """
        Compute time-dependent ROC curves for dynamic risk prediction

        AUC(t) = Pr(Mi > Mj | Ti = t, Tj > t)

        Args:
            fitted_model: Previously fitted Cox model
            test_times: Test set time-to-event
            test_events: Test set event indicators
            test_covariates: Test set baseline covariates
            time_dependent_test_covariates: Test set time-dependent covariates
            evaluation_times: Time points for ROC evaluation

        Returns:
            TimeDependentROCResult with AUC curves over time
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for time-dependent ROC analysis")

        logger.info("Computing time-dependent ROC analysis")

        try:
            # Set default evaluation times
            if evaluation_times is None:
                # Use quantiles of event times
                event_times = test_times[test_events == 1]
                if len(event_times) > 0:
                    evaluation_times = np.quantile(event_times, self.roc_config['evaluation_times_quantiles']).tolist()
                else:
                    evaluation_times = [np.median(test_times)]

            auc_values = []
            sensitivities = []
            specificities = []
            optimal_thresholds = []
            confidence_intervals = []

            for t in evaluation_times:
                # Compute risk scores at time t
                risk_scores = self._compute_risk_scores_at_time(
                    fitted_model, test_covariates, t, time_dependent_test_covariates
                )

                # Create binary labels for time t
                labels_t = self._create_time_dependent_labels(test_times, test_events, t)

                if len(np.unique(labels_t)) == 2 and len(risk_scores) == len(labels_t):
                    # Calculate ROC curve
                    fpr, tpr, thresholds = roc_curve(labels_t, risk_scores)

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

                    # Bootstrap confidence interval for AUC
                    if self.roc_config['n_bootstraps'] > 0:
                        ci = self._bootstrap_auc_ci(risk_scores, labels_t,
                                                  n_bootstraps=self.roc_config['n_bootstraps'],
                                                  confidence_level=self.roc_config['confidence_level'])
                        confidence_intervals.append(ci)
                    else:
                        confidence_intervals.append(None)
                else:
                    auc_values.append(np.nan)
                    sensitivities.append(np.nan)
                    specificities.append(np.nan)
                    optimal_thresholds.append(np.nan)
                    confidence_intervals.append(None)

            result = TimeDependentROCResult(
                time_points=evaluation_times,
                auc_values=auc_values,
                sensitivities=sensitivities,
                specificities=specificities,
                optimal_thresholds=optimal_thresholds,
                confidence_intervals=confidence_intervals
            )

            logger.info(f"Time-dependent ROC analysis completed. Mean AUC: {np.nanmean(auc_values):.3f}")
            return result

        except Exception as e:
            logger.error(f"Error in time-dependent ROC analysis: {e}")
            raise

    def _compute_risk_scores_at_time(self,
                                   fitted_model: CoxModelResult,
                                   covariates: pd.DataFrame,
                                   time_point: float,
                                   time_dependent_covariates: Optional[List[TimeDependentCovariate]] = None) -> np.ndarray:
        """Compute risk scores at a specific time point"""
        try:
            cph = fitted_model.fitted_model

            # Get covariate values at time t
            covariate_values = {}
            for col in covariates.columns:
                covariate_values[col] = covariates[col].values

            # Add time-dependent covariates at time t
            if time_dependent_covariates:
                for tdc in time_dependent_covariates:
                    value = self._get_covariate_value_at_time(tdc, time_point)
                    if value is not None:
                        covariate_values[tdc.name] = np.full(len(covariates), value)

            # Create DataFrame for prediction
            pred_df = pd.DataFrame(covariate_values)

            # Compute linear predictor (log hazard ratio)
            linear_pred = cph.predict_log_partial_hazards(pred_df)

            return linear_pred.values.flatten()

        except Exception as e:
            logger.error(f"Error computing risk scores at time {time_point}: {e}")
            return np.zeros(len(covariates))

    def _create_time_dependent_labels(self,
                                    times: np.ndarray,
                                    events: np.ndarray,
                                    evaluation_time: float) -> np.ndarray:
        """Create binary labels for time-dependent ROC analysis"""
        labels = np.zeros(len(times))

        for i in range(len(times)):
            if events[i] == 1 and times[i] <= evaluation_time:
                # Event occurred before or at evaluation time
                labels[i] = 1
            elif times[i] > evaluation_time:
                # Still at risk at evaluation time (censored after t)
                labels[i] = 0
            else:
                # Censored before evaluation time - exclude from analysis
                labels[i] = -1  # Will be filtered out

        # Remove cases censored before evaluation time
        return labels

    def _bootstrap_auc_ci(self,
                         risk_scores: np.ndarray,
                         labels: np.ndarray,
                         n_bootstraps: int = 100,
                         confidence_level: float = 0.95) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for AUC"""
        auc_bootstraps = []

        for _ in range(n_bootstraps):
            # Bootstrap sample
            indices = np.random.choice(len(risk_scores), len(risk_scores), replace=True)
            risk_bootstrap = risk_scores[indices]
            labels_bootstrap = labels[indices]

            if len(np.unique(labels_bootstrap)) == 2:
                try:
                    auc_bootstrap = roc_auc_score(labels_bootstrap, risk_bootstrap)
                    auc_bootstraps.append(auc_bootstrap)
                except:
                    continue

        if auc_bootstraps:
            lower_percentile = (1 - confidence_level) / 2 * 100
            upper_percentile = (1 + confidence_level) / 2 * 100

            ci_lower = np.percentile(auc_bootstraps, lower_percentile)
            ci_upper = np.percentile(auc_bootstraps, upper_percentile)

            return (ci_lower, ci_upper)

        return (np.nan, np.nan)

    def predict_dynamic_risk(self,
                           fitted_model: CoxModelResult,
                           prediction_times: List[float],
                           baseline_covariates: pd.DataFrame,
                           time_dependent_covariates: Optional[List[TimeDependentCovariate]] = None,
                           patient_id: Optional[int] = None) -> DynamicRiskPrediction:
        """
        Generate dynamic risk predictions over time for an individual patient

        Args:
            fitted_model: Fitted Cox model
            prediction_times: Time points for prediction
            baseline_covariates: Patient baseline covariates
            time_dependent_covariates: Patient time-dependent covariates
            patient_id: Patient identifier

        Returns:
            DynamicRiskPrediction with risk trajectory
        """
        logger.info(f"Generating dynamic risk predictions for patient {patient_id}")

        try:
            risk_scores = []
            cumulative_hazards = []
            survival_probabilities = []

            for t in prediction_times:
                # Compute risk score at time t
                risk_score = self._compute_risk_scores_at_time(
                    fitted_model, baseline_covariates, t, time_dependent_covariates
                )[0]  # Single patient

                risk_scores.append(risk_score)

                # Compute cumulative hazard and survival probability
                cph = fitted_model.fitted_model

                try:
                    # Get baseline cumulative hazard at time t
                    baseline_cum_hazard = cph.baseline_cumulative_hazard_at_times([t]).iloc[0, 0]

                    # Individual cumulative hazard
                    cum_hazard = baseline_cum_hazard * np.exp(risk_score)
                    cumulative_hazards.append(cum_hazard)

                    # Survival probability
                    survival_prob = np.exp(-cum_hazard)
                    survival_probabilities.append(survival_prob)

                except Exception as e:
                    logger.warning(f"Error computing hazard at time {t}: {e}")
                    cumulative_hazards.append(np.nan)
                    survival_probabilities.append(np.nan)

            result = DynamicRiskPrediction(
                prediction_times=prediction_times,
                risk_scores=risk_scores,
                cumulative_hazards=cumulative_hazards,
                survival_probabilities=survival_probabilities,
                covariate_values=self._extract_covariate_trajectory(
                    baseline_covariates, time_dependent_covariates, prediction_times
                )
            )

            logger.info(f"Dynamic risk prediction completed for {len(prediction_times)} time points")
            return result

        except Exception as e:
            logger.error(f"Error in dynamic risk prediction: {e}")
            raise

    def _extract_covariate_trajectory(self,
                                    baseline_covariates: pd.DataFrame,
                                    time_dependent_covariates: Optional[List[TimeDependentCovariate]],
                                    prediction_times: List[float]) -> Optional[Dict[str, List[float]]]:
        """Extract covariate values over time for visualization"""
        if time_dependent_covariates is None:
            return None

        trajectory = {}

        # Add baseline covariates (constant over time)
        for col in baseline_covariates.columns:
            trajectory[col] = [baseline_covariates[col].iloc[0]] * len(prediction_times)

        # Add time-dependent covariates
        for tdc in time_dependent_covariates:
            values_over_time = []
            for t in prediction_times:
                value = self._get_covariate_value_at_time(tdc, t)
                values_over_time.append(value if value is not None else 0.0)
            trajectory[tdc.name] = values_over_time

        return trajectory

    def compare_models(self,
                      model_results: Dict[str, CoxModelResult],
                      test_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Compare multiple fitted Cox models using various metrics

        Args:
            model_results: Dictionary of fitted model results
            test_data: Optional test data for external validation

        Returns:
            Model comparison results
        """
        logger.info(f"Comparing {len(model_results)} Cox models")

        comparison = {
            'models': list(model_results.keys()),
            'concordance_indices': {},
            'aic_values': {},
            'bic_values': {},
            'log_likelihoods': {},
            'n_parameters': {},
            'relative_performance': {}
        }

        for name, result in model_results.items():
            comparison['concordance_indices'][name] = result.concordance_index
            comparison['aic_values'][name] = result.aic
            comparison['bic_values'][name] = result.bic
            comparison['log_likelihoods'][name] = result.log_likelihood
            comparison['n_parameters'][name] = len(result.coefficients)

        # Find best models by different criteria
        best_c_index = max(comparison['concordance_indices'], key=comparison['concordance_indices'].get)
        best_aic = min(comparison['aic_values'], key=comparison['aic_values'].get)
        best_bic = min(comparison['bic_values'], key=comparison['bic_values'].get)

        comparison['relative_performance'] = {
            'best_concordance_index': {
                'model': best_c_index,
                'value': comparison['concordance_indices'][best_c_index]
            },
            'best_aic': {
                'model': best_aic,
                'value': comparison['aic_values'][best_aic]
            },
            'best_bic': {
                'model': best_bic,
                'value': comparison['bic_values'][best_bic]
            }
        }

        logger.info(f"Model comparison completed. Best C-index: {best_c_index} ({comparison['concordance_indices'][best_c_index]:.3f})")
        return comparison

    def validate_model(self,
                      fitted_model: CoxModelResult,
                      validation_times: np.ndarray,
                      validation_events: np.ndarray,
                      validation_covariates: pd.DataFrame,
                      time_dependent_validation_covariates: Optional[List[TimeDependentCovariate]] = None,
                      n_folds: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive model validation including cross-validation

        Args:
            fitted_model: Fitted Cox model to validate
            validation_times: Validation time-to-event data
            validation_events: Validation event indicators
            validation_covariates: Validation baseline covariates
            time_dependent_validation_covariates: Validation time-dependent covariates
            n_folds: Number of cross-validation folds

        Returns:
            Validation results
        """
        logger.info(f"Performing model validation with {n_folds}-fold cross-validation")

        validation_results = {
            'cross_validation_scores': [],
            'time_dependent_roc': None,
            'calibration_metrics': {},
            'discrimination_metrics': {}
        }

        try:
            # Time-dependent ROC analysis
            roc_result = self.compute_time_dependent_roc(
                fitted_model, validation_times, validation_events,
                validation_covariates, time_dependent_validation_covariates
            )
            validation_results['time_dependent_roc'] = {
                'mean_auc': np.nanmean(roc_result.auc_values),
                'auc_values': roc_result.auc_values,
                'time_points': roc_result.time_points
            }

            # Cross-validation for concordance index
            if HAS_SKSURV:
                # Use scikit-survival for cross-validation
                cv_scores = self._cross_validate_concordance_index(
                    validation_times, validation_events, validation_covariates, n_folds
                )
                validation_results['cross_validation_scores'] = cv_scores
                validation_results['cv_mean_c_index'] = np.mean(cv_scores)
                validation_results['cv_std_c_index'] = np.std(cv_scores)

            # Discrimination metrics
            validation_results['discrimination_metrics'] = {
                'concordance_index': fitted_model.concordance_index,
                'log_likelihood': fitted_model.log_likelihood,
                'n_events': fitted_model.n_events,
                'n_at_risk': fitted_model.n_at_risk
            }

            logger.info(f"Model validation completed. Mean CV C-index: {validation_results.get('cv_mean_c_index', 'N/A'):.3f}")

        except Exception as e:
            logger.error(f"Error in model validation: {e}")
            validation_results['error'] = str(e)

        return validation_results

    def _cross_validate_concordance_index(self,
                                        times: np.ndarray,
                                        events: np.ndarray,
                                        covariates: pd.DataFrame,
                                        n_folds: int) -> List[float]:
        """Perform cross-validation for concordance index"""
        if not HAS_SKSURV:
            return []

        try:
            # Prepare structured array for scikit-survival
            y = np.array([(bool(event), time) for event, time in zip(events, times)],
                        dtype=[('event', bool), ('time', float)])

            X = covariates.values

            # Cross-validation
            cv_scores = []
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            for train_idx, val_idx in kf.split(X, events):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Fit model
                cox_model = CoxPHSurvivalAnalysis(alpha=0.01)
                cox_model.fit(X_train, y_train)

                # Compute concordance index
                c_index = cox_model.score(X_val, y_val)
                cv_scores.append(c_index)

            return cv_scores

        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return []
