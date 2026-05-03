"""
Advanced Wavelet Transform Analysis for COTCAgent

This module implements sophisticated wavelet analysis methods for medical time series,
including continuous and discrete wavelet transforms, wavelet coherence, and multiresolution analysis
as described in Daubechies (1992) and Torrence & Webster (1999).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import warnings
import logging
from scipy import signal, stats
from scipy.signal import find_peaks

# Wavelet analysis libraries
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    warnings.warn("PyWavelets not installed. Wavelet analysis will be unavailable.")

try:
    from scipy.signal import cwt, morlet2
    HAS_SCIPY_SIGNAL = True
except ImportError:
    HAS_SCIPY_SIGNAL = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class WaveletTransformResult:
    """Results from wavelet transform analysis"""
    coefficients: np.ndarray
    scales: np.ndarray
    frequencies: np.ndarray
    power_spectrum: np.ndarray
    wavelet_type: str = "morlet"
    sampling_rate: float = 1.0
    time_points: Optional[np.ndarray] = None
    cone_of_influence: Optional[np.ndarray] = None
    significance_mask: Optional[np.ndarray] = None

@dataclass
class WaveletCoherenceResult:
    """Results from wavelet coherence analysis"""
    coherence_matrix: np.ndarray
    phase_matrix: np.ndarray
    cross_wavelet_power: np.ndarray
    scales: np.ndarray
    frequencies: np.ndarray
    wavelet_type: str = "morlet"
    sampling_rate: float = 1.0
    time_points: Optional[np.ndarray] = None
    significance_level: float = 0.95
    significance_mask: Optional[np.ndarray] = None

@dataclass
class MultiresolutionAnalysis:
    """Results from multiresolution analysis"""
    approximation_coefficients: Dict[int, np.ndarray]
    detail_coefficients: Dict[int, np.ndarray]
    decomposition_levels: int
    wavelet_type: str = "db4"
    energy_distribution: Dict[int, float]
    dominant_frequencies: Dict[int, float]
    reconstruction_error: float = 0.0

@dataclass
class TimeFrequencySignature:
    """Time-frequency signature of a signal"""
    ridge_curve: np.ndarray  # Main frequency ridge
    instantaneous_frequency: np.ndarray
    instantaneous_amplitude: np.ndarray
    instantaneous_phase: np.ndarray
    time_points: np.ndarray
    modulation_index: float = 0.0
    frequency_variability: float = 0.0

class AdvancedWaveletAnalyzer:
    """
    Advanced wavelet analyzer for medical time series analysis

    Implements continuous wavelet transform (CWT), discrete wavelet transform (DWT),
    wavelet coherence, and multiresolution analysis for physiological signals.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Default wavelet configuration
        self.wavelet_config = self.config.get('wavelet', {
            'wavelet': 'morl',  # Morlet wavelet for CWT
            'dwt_wavelet': 'db4',  # Daubechies 4 for DWT
            'scales': np.arange(1, 128),
            'sampling_period': 1.0,
            'dj': 0.125,  # Scale resolution
            's0': 2.0,    # Smallest scale
            'j1': None,   # Largest scale will be computed
        })

        # Coherence analysis configuration
        self.coherence_config = self.config.get('coherence', {
            'n_surrogates': 100,
            'significance_level': 0.95,
            'smoothing_length': 5,
            'scale_smoothing': 1
        })

        # Store analysis results
        self.transform_results = {}
        self.coherence_results = {}

        logger.info("Advanced wavelet analyzer initialized")

    def continuous_wavelet_transform(self,
                                   time_series: np.ndarray,
                                   time_points: Optional[np.ndarray] = None,
                                   wavelet_type: str = "morlet",
                                   scales: Optional[np.ndarray] = None,
                                   sampling_rate: float = 1.0,
                                   result_name: str = "cwt_analysis") -> WaveletTransformResult:
        """
        Perform continuous wavelet transform on time series

        Wx(a,b) = 1/√|a| ∫ x(t) ψ*((t-b)/a) dt

        Args:
            time_series: 1D time series signal
            time_points: Corresponding time points
            wavelet_type: Type of mother wavelet
            scales: Scales for analysis
            sampling_rate: Sampling rate of the signal
            result_name: Identifier for results

        Returns:
            WaveletTransformResult with transform coefficients and analysis
        """
        if not HAS_PYWT:
            raise ImportError("PyWavelets is required for wavelet analysis")

        logger.info(f"Performing continuous wavelet transform: {result_name}")

        try:
            # Set default scales
            if scales is None:
                scales = self.wavelet_config['scales']

            # Perform CWT using PyWavelets
            coefficients, frequencies = pywt.cwt(
                time_series, scales, wavelet_type, sampling_rate
            )

            # Calculate power spectrum
            power_spectrum = np.abs(coefficients) ** 2

            # Calculate frequencies corresponding to scales
            if wavelet_type == 'morl':
                central_frequency = pywt.ContinuousWavelet(wavelet_type).center_frequency
                frequencies = central_frequency / (scales * sampling_rate)

            # Compute cone of influence (simplified)
            coi = self._compute_cone_of_influence(len(time_series), scales, wavelet_type)

            # Compute significance mask (optional, requires statistical testing)
            significance_mask = self._compute_wavelet_significance(
                power_spectrum, time_series, scales, wavelet_type
            )

            result = WaveletTransformResult(
                coefficients=coefficients,
                scales=scales,
                frequencies=frequencies,
                power_spectrum=power_spectrum,
                wavelet_type=wavelet_type,
                sampling_rate=sampling_rate,
                time_points=time_points if time_points is not None else np.arange(len(time_series)),
                cone_of_influence=coi,
                significance_mask=significance_mask
            )

            self.transform_results[result_name] = result

            logger.info(f"CWT completed for {result_name}: {coefficients.shape} coefficient matrix")

            return result

        except Exception as e:
            logger.error(f"Error in continuous wavelet transform {result_name}: {e}")
            raise

    def discrete_wavelet_transform(self,
                                 time_series: np.ndarray,
                                 wavelet_type: str = "db4",
                                 decomposition_level: Optional[int] = None,
                                 result_name: str = "dwt_analysis") -> MultiresolutionAnalysis:
        """
        Perform discrete wavelet transform with multiresolution analysis

        x(t) = ∑k Wϕ(j₀,k) ϕj₀,k(t) + ∑j≥j₀ ∑k Wψ(j,k) ψj,k(t)

        Args:
            time_series: Time series signal (must be power of 2 length)
            wavelet_type: Type of wavelet
            decomposition_level: Level of decomposition
            result_name: Identifier for results

        Returns:
            MultiresolutionAnalysis with decomposition results
        """
        if not HAS_PYWT:
            raise ImportError("PyWavelets is required for wavelet analysis")

        logger.info(f"Performing discrete wavelet transform: {result_name}")

        try:
            # Ensure signal length is appropriate
            n = len(time_series)
            if decomposition_level is None:
                decomposition_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet_type))

            # Perform multilevel DWT
            coeffs = pywt.wavedec(time_series, wavelet_type, level=decomposition_level)

            # Separate approximation and detail coefficients
            approximation_coefficients = {0: coeffs[0]}  # Approximation at coarsest level
            detail_coefficients = {}

            for i in range(1, len(coeffs)):
                detail_coefficients[i] = coeffs[i]

            # Calculate energy distribution
            energy_distribution = {}
            total_energy = np.sum(time_series ** 2)

            for level in range(1, len(coeffs)):
                level_energy = np.sum(coeffs[level] ** 2)
                energy_distribution[level] = level_energy / total_energy

            # Calculate dominant frequencies for each level
            dominant_frequencies = {}
            sampling_rate = 1.0 / self.wavelet_config['sampling_period']

            for level in range(1, len(coeffs)):
                # Dominant frequency for this level
                freq = sampling_rate / (2 ** (level + 1))
                dominant_frequencies[level] = freq

            # Reconstruction error (optional)
            reconstructed = pywt.waverec(coeffs, wavelet_type)
            reconstruction_error = np.sqrt(np.mean((time_series - reconstructed[:n]) ** 2))

            result = MultiresolutionAnalysis(
                approximation_coefficients=approximation_coefficients,
                detail_coefficients=detail_coefficients,
                decomposition_levels=decomposition_level,
                wavelet_type=wavelet_type,
                energy_distribution=energy_distribution,
                dominant_frequencies=dominant_frequencies,
                reconstruction_error=reconstruction_error
            )

            logger.info(f"DWT completed for {result_name}: {decomposition_level} levels, "
                       f"reconstruction error: {reconstruction_error:.6f}")

            return result

        except Exception as e:
            logger.error(f"Error in discrete wavelet transform {result_name}: {e}")
            raise

    def wavelet_coherence_analysis(self,
                                 series1: np.ndarray,
                                 series2: np.ndarray,
                                 time_points: Optional[np.ndarray] = None,
                                 wavelet_type: str = "morlet",
                                 scales: Optional[np.ndarray] = None,
                                 smoothing_length: int = 5,
                                 result_name: str = "coherence_analysis") -> WaveletCoherenceResult:
        """
        Compute wavelet coherence between two time series

        Rxy(a,b) = |S(a⁻¹Wxy(a,b))|² / [S(a⁻¹|Wx(a,b)|²)S(a⁻¹|Wy(a,b)|²)]

        Args:
            series1: First time series
            series2: Second time series
            time_points: Time points for series
            wavelet_type: Type of wavelet
            scales: Scales for analysis
            smoothing_length: Length of smoothing window
            result_name: Identifier for results

        Returns:
            WaveletCoherenceResult with coherence analysis
        """
        if not HAS_PYWT:
            raise ImportError("PyWavelets is required for coherence analysis")

        logger.info(f"Computing wavelet coherence: {result_name}")

        try:
            if scales is None:
                scales = self.wavelet_config['scales']

            # Perform wavelet transforms
            coef1, freqs1 = pywt.cwt(series1, scales, wavelet_type, 1.0)
            coef2, freqs2 = pywt.cwt(series2, scales, wavelet_type, 1.0)

            # Compute cross-wavelet transform
            cross_wavelet = coef1 * np.conj(coef2)

            # Compute individual wavelet power
            w1_power = np.abs(coef1) ** 2
            w2_power = np.abs(coef2) ** 2

            # Smooth the spectra
            w1_power_smooth = self._smooth_wavelet_power(w1_power, smoothing_length)
            w2_power_smooth = self._smooth_wavelet_power(w2_power, smoothing_length)
            cross_smooth = self._smooth_wavelet_power(cross_wavelet, smoothing_length)

            # Compute wavelet coherence
            coherence = np.abs(cross_smooth) ** 2 / (w1_power_smooth * w2_power_smooth)

            # Handle potential division by zero
            coherence = np.where(np.isfinite(coherence), coherence, 0.0)
            coherence = np.clip(coherence, 0.0, 1.0)

            # Compute phase difference
            phase_matrix = np.angle(cross_smooth)

            # Compute cross-wavelet power
            cross_wavelet_power = np.abs(cross_smooth) ** 2

            # Statistical significance (simplified Monte Carlo approach)
            significance_mask = self._compute_coherence_significance(
                series1, series2, scales, wavelet_type,
                n_surrogates=self.coherence_config['n_surrogates']
            )

            result = WaveletCoherenceResult(
                coherence_matrix=coherence,
                phase_matrix=phase_matrix,
                cross_wavelet_power=cross_wavelet_power,
                scales=scales,
                frequencies=freqs1,  # Assuming same frequencies for both series
                wavelet_type=wavelet_type,
                sampling_rate=1.0,
                time_points=time_points if time_points is not None else np.arange(len(series1)),
                significance_level=self.coherence_config['significance_level'],
                significance_mask=significance_mask
            )

            self.coherence_results[result_name] = result

            logger.info(f"Wavelet coherence completed for {result_name}: "
                       f"mean coherence = {np.mean(coherence):.3f}")

            return result

        except Exception as e:
            logger.error(f"Error in wavelet coherence analysis {result_name}: {e}")
            raise

    def _smooth_wavelet_power(self, power_matrix: np.ndarray, smoothing_length: int) -> np.ndarray:
        """Smooth wavelet power spectrum in time and scale dimensions"""
        from scipy.ndimage import gaussian_filter

        # Smooth in time direction
        smoothed = gaussian_filter(power_matrix, sigma=(0, smoothing_length), mode='nearest')

        # Smooth in scale direction
        smoothed = gaussian_filter(smoothed, sigma=(smoothing_length, 0), mode='nearest')

        return smoothed

    def _compute_cone_of_influence(self,
                                 signal_length: int,
                                 scales: np.ndarray,
                                 wavelet_type: str) -> np.ndarray:
        """Compute cone of influence for wavelet transform"""
        # Simplified COI calculation
        # In practice, this depends on the specific wavelet used

        coi = np.zeros((len(scales), signal_length))

        for i, scale in enumerate(scales):
            # Approximate COI as proportional to scale
            coi_width = int(scale * 2)  # Simplified
            coi_center = signal_length // 2

            start = max(0, coi_center - coi_width // 2)
            end = min(signal_length, coi_center + coi_width // 2)

            coi[i, start:end] = 1

        return coi

    def _compute_wavelet_significance(self,
                                    power_spectrum: np.ndarray,
                                    original_signal: np.ndarray,
                                    scales: np.ndarray,
                                    wavelet_type: str,
                                    alpha: float = 0.05) -> np.ndarray:
        """Compute statistical significance of wavelet power"""
        # Simplified significance testing using chi-squared distribution
        # In practice, this should use more sophisticated surrogate data methods

        try:
            # Degrees of freedom (approximate for Morlet wavelet)
            dof = 2

            # Chi-squared threshold
            threshold = stats.chi2.ppf(1 - alpha, dof) / dof

            # Significance mask
            significance_mask = power_spectrum > threshold

            return significance_mask.astype(int)

        except Exception:
            # Return all zeros if significance testing fails
            return np.zeros_like(power_spectrum, dtype=int)

    def _compute_coherence_significance(self,
                                      series1: np.ndarray,
                                      series2: np.ndarray,
                                      scales: np.ndarray,
                                      wavelet_type: str,
                                      n_surrogates: int = 100) -> np.ndarray:
        """Compute statistical significance of wavelet coherence using surrogates"""
        try:
            coherence_surrogates = []

            for _ in range(n_surrogates):
                # Generate surrogate series by phase randomization
                surrogate1 = self._phase_randomize(series1)
                surrogate2 = self._phase_randomize(series2)

                # Compute coherence for surrogates
                coef1, _ = pywt.cwt(surrogate1, scales, wavelet_type, 1.0)
                coef2, _ = pywt.cwt(surrogate2, scales, wavelet_type, 1.0)

                cross_wavelet = coef1 * np.conj(coef2)
                w1_power = np.abs(coef1) ** 2
                w2_power = np.abs(coef2) ** 2

                # Smooth surrogates
                w1_smooth = self._smooth_wavelet_power(w1_power, self.coherence_config['smoothing_length'])
                w2_smooth = self._smooth_wavelet_power(w2_power, self.coherence_config['smoothing_length'])
                cross_smooth = self._smooth_wavelet_power(cross_wavelet, self.coherence_config['smoothing_length'])

                surrogate_coherence = np.abs(cross_smooth) ** 2 / (w1_smooth * w2_smooth)
                surrogate_coherence = np.where(np.isfinite(surrogate_coherence), surrogate_coherence, 0.0)
                surrogate_coherence = np.clip(surrogate_coherence, 0.0, 1.0)

                coherence_surrogates.append(surrogate_coherence)

            # Compute significance threshold
            coherence_surrogates = np.array(coherence_surrogates)
            significance_threshold = np.percentile(coherence_surrogates, 95, axis=0)  # 95th percentile

            # Compute actual coherence for comparison
            coef1, _ = pywt.cwt(series1, scales, wavelet_type, 1.0)
            coef2, _ = pywt.cwt(series2, scales, wavelet_type, 1.0)
            cross_wavelet = coef1 * np.conj(coef2)
            w1_power = np.abs(coef1) ** 2
            w2_power = np.abs(coef2) ** 2

            w1_smooth = self._smooth_wavelet_power(w1_power, self.coherence_config['smoothing_length'])
            w2_smooth = self._smooth_wavelet_power(w2_power, self.coherence_config['smoothing_length'])
            cross_smooth = self._smooth_wavelet_power(cross_wavelet, self.coherence_config['smoothing_length'])

            actual_coherence = np.abs(cross_smooth) ** 2 / (w1_smooth * w2_smooth)
            actual_coherence = np.where(np.isfinite(actual_coherence), actual_coherence, 0.0)
            actual_coherence = np.clip(actual_coherence, 0.0, 1.0)

            # Significance mask
            significance_mask = actual_coherence > significance_threshold

            return significance_mask.astype(int)

        except Exception as e:
            logger.warning(f"Coherence significance testing failed: {e}")
            return np.zeros((len(scales), len(series1)), dtype=int)

    def _phase_randomize(self, signal: np.ndarray) -> np.ndarray:
        """Generate surrogate data by phase randomization"""
        # Compute FFT
        fft_signal = np.fft.fft(signal)

        # Randomize phases
        phases = np.angle(fft_signal)
        random_phases = np.random.uniform(0, 2*np.pi, len(phases))

        # Keep the original magnitude, randomize phases
        randomized_fft = np.abs(fft_signal) * np.exp(1j * random_phases)

        # Inverse FFT
        surrogate = np.real(np.fft.ifft(randomized_fft))

        return surrogate

    def extract_time_frequency_signature(self,
                                       time_series: np.ndarray,
                                       time_points: Optional[np.ndarray] = None,
                                       wavelet_type: str = "morlet",
                                       result_name: str = "tf_signature") -> TimeFrequencySignature:
        """
        Extract time-frequency signature including ridge curve and instantaneous measures

        Args:
            time_series: Time series signal
            time_points: Corresponding time points
            wavelet_type: Type of wavelet
            result_name: Identifier for results

        Returns:
            TimeFrequencySignature with detailed time-frequency analysis
        """
        logger.info(f"Extracting time-frequency signature: {result_name}")

        try:
            # Perform CWT
            cwt_result = self.continuous_wavelet_transform(
                time_series, time_points, wavelet_type, result_name=f"{result_name}_cwt"
            )

            # Extract ridge curve (main frequency component)
            ridge_curve = self._extract_ridge_curve(cwt_result.power_spectrum, cwt_result.scales)

            # Compute instantaneous measures
            instantaneous_frequency = self._compute_instantaneous_frequency(
                cwt_result.coefficients, cwt_result.scales, cwt_result.sampling_rate
            )

            instantaneous_amplitude = np.abs(cwt_result.coefficients[np.argmax(cwt_result.power_spectrum, axis=0),
                                                                  np.arange(cwt_result.coefficients.shape[1])])

            instantaneous_phase = np.angle(cwt_result.coefficients[np.argmax(cwt_result.power_spectrum, axis=0),
                                                                np.arange(cwt_result.coefficients.shape[1])])

            # Compute modulation index (measure of frequency modulation)
            modulation_index = self._compute_modulation_index(instantaneous_frequency)

            # Compute frequency variability
            frequency_variability = np.std(instantaneous_frequency) / np.mean(instantaneous_frequency)

            time_points_array = time_points if time_points is not None else np.arange(len(time_series))

            result = TimeFrequencySignature(
                ridge_curve=ridge_curve,
                instantaneous_frequency=instantaneous_frequency,
                instantaneous_amplitude=instantaneous_amplitude,
                instantaneous_phase=instantaneous_phase,
                time_points=time_points_array,
                modulation_index=modulation_index,
                frequency_variability=frequency_variability
            )

            logger.info(f"Time-frequency signature extracted for {result_name}: "
                       f"modulation index = {modulation_index:.3f}")

            return result

        except Exception as e:
            logger.error(f"Error extracting time-frequency signature {result_name}: {e}")
            raise

    def _extract_ridge_curve(self, power_spectrum: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Extract the ridge curve (curve of maximum power)"""
        # Find scale with maximum power at each time point
        max_scale_indices = np.argmax(power_spectrum, axis=0)

        # Convert scale indices to actual scales
        ridge_curve = scales[max_scale_indices]

        return ridge_curve

    def _compute_instantaneous_frequency(self,
                                       coefficients: np.ndarray,
                                       scales: np.ndarray,
                                       sampling_rate: float) -> np.ndarray:
        """Compute instantaneous frequency from wavelet coefficients"""
        # Simplified instantaneous frequency calculation
        # Using the scale where power is maximum at each time point

        power_spectrum = np.abs(coefficients) ** 2
        max_scale_indices = np.argmax(power_spectrum, axis=0)

        # Convert scales to frequencies
        frequencies = 1.0 / (scales * sampling_rate)  # Approximate
        instantaneous_frequency = frequencies[max_scale_indices]

        return instantaneous_frequency

    def _compute_modulation_index(self, instantaneous_frequency: np.ndarray) -> float:
        """Compute modulation index as measure of frequency variability"""
        # Modulation index based on the variance of instantaneous frequency
        mean_freq = np.mean(instantaneous_frequency)
        if mean_freq == 0:
            return 0.0

        # Normalize frequency variations
        normalized_freq = instantaneous_frequency / mean_freq

        # Compute modulation index (similar to amplitude modulation index)
        modulation_index = np.sqrt(np.mean((normalized_freq - 1) ** 2))

        return modulation_index

    def analyze_multivariate_coherence(self,
                                     time_series_dict: Dict[str, np.ndarray],
                                     time_points: Optional[np.ndarray] = None,
                                     wavelet_type: str = "morlet") -> Dict[str, WaveletCoherenceResult]:
        """
        Perform multivariate wavelet coherence analysis

        Args:
            time_series_dict: Dictionary of time series signals
            time_points: Time points for analysis
            wavelet_type: Type of wavelet

        Returns:
            Dictionary of coherence results for all pairs
        """
        logger.info(f"Analyzing multivariate coherence for {len(time_series_dict)} signals")

        coherence_results = {}
        signal_names = list(time_series_dict.keys())

        # Compute coherence for all pairs
        for i in range(len(signal_names)):
            for j in range(i + 1, len(signal_names)):
                name1, name2 = signal_names[i], signal_names[j]
                series1, series2 = time_series_dict[name1], time_series_dict[name2]

                pair_name = f"{name1}_{name2}_coherence"

                try:
                    coherence_result = self.wavelet_coherence_analysis(
                        series1, series2, time_points, wavelet_type, result_name=pair_name
                    )

                    coherence_results[pair_name] = coherence_result

                except Exception as e:
                    logger.warning(f"Coherence analysis failed for {name1} vs {name2}: {e}")

        logger.info(f"Multivariate coherence analysis completed: {len(coherence_results)} pairs analyzed")

        return coherence_results

    def detect_transient_events(self,
                              time_series: np.ndarray,
                              wavelet_result: WaveletTransformResult,
                              threshold: float = 2.0,
                              min_duration: int = 5) -> List[Dict[str, Any]]:
        """
        Detect transient events in time series using wavelet analysis

        Args:
            time_series: Original time series
            wavelet_result: Wavelet transform results
            threshold: Detection threshold (in standard deviations)
            min_duration: Minimum event duration

        Returns:
            List of detected transient events
        """
        logger.info("Detecting transient events using wavelet analysis")

        try:
            power_spectrum = wavelet_result.power_spectrum

            # Compute global wavelet spectrum
            global_ws = np.mean(power_spectrum, axis=1)

            # Find scales with significant power
            mean_power = np.mean(power_spectrum)
            std_power = np.std(power_spectrum)

            significant_mask = power_spectrum > (mean_power + threshold * std_power)

            # Detect events in time-scale plane
            events = []

            for scale_idx in range(len(wavelet_result.scales)):
                scale_power = power_spectrum[scale_idx, :]

                # Find peaks in power spectrum for this scale
                peaks, properties = find_peaks(scale_power, height=mean_power + threshold * std_power,
                                             distance=min_duration)

                for peak_idx in peaks:
                    event = {
                        'time_index': peak_idx,
                        'scale_index': scale_idx,
                        'scale': wavelet_result.scales[scale_idx],
                        'frequency': wavelet_result.frequencies[scale_idx],
                        'power': scale_power[peak_idx],
                        'duration': properties['widths'][peaks == peak_idx][0] if 'widths' in properties else min_duration,
                        'significance': (scale_power[peak_idx] - mean_power) / std_power
                    }
                    events.append(event)

            # Sort events by significance
            events.sort(key=lambda x: x['significance'], reverse=True)

            logger.info(f"Detected {len(events)} transient events")

            return events

        except Exception as e:
            logger.error(f"Error detecting transient events: {e}")
            return []

    def generate_wavelet_report(self,
                              analysis_results: Dict[str, Any],
                              signal_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive wavelet analysis report

        Args:
            analysis_results: Dictionary of analysis results
            signal_name: Name of the analyzed signal

        Returns:
            Comprehensive analysis report
        """
        logger.info(f"Generating wavelet analysis report for {signal_name}")

        report = {
            'signal_name': signal_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'wavelet_methods_applied': [],
            'key_findings': [],
            'clinical_insights': [],
            'technical_metrics': {}
        }

        try:
            # Process CWT results
            if 'cwt_analysis' in analysis_results:
                cwt_result = analysis_results['cwt_analysis']
                report['wavelet_methods_applied'].append('continuous_wavelet_transform')

                # Analyze power spectrum
                max_power_scale = cwt_result.scales[np.argmax(np.max(cwt_result.power_spectrum, axis=1))]
                dominant_frequency = cwt_result.frequencies[np.argmax(np.max(cwt_result.power_spectrum, axis=1))]

                report['technical_metrics'].update({
                    'dominant_scale': float(max_power_scale),
                    'dominant_frequency': float(dominant_frequency),
                    'mean_power': float(np.mean(cwt_result.power_spectrum)),
                    'max_power': float(np.max(cwt_result.power_spectrum))
                })

                report['key_findings'].append({
                    'type': 'dominant_frequency',
                    'description': f"Dominant frequency component at {dominant_frequency:.3f} Hz (scale {max_power_scale:.1f})",
                    'clinical_relevance': 'Indicates primary oscillatory pattern in the signal'
                })

            # Process DWT results
            if 'dwt_analysis' in analysis_results:
                dwt_result = analysis_results['dwt_analysis']
                report['wavelet_methods_applied'].append('discrete_wavelet_transform')

                # Energy distribution analysis
                max_energy_level = max(dwt_result.energy_distribution, key=dwt_result.energy_distribution.get)

                report['technical_metrics'].update({
                    'decomposition_levels': dwt_result.decomposition_levels,
                    'max_energy_level': max_energy_level,
                    'max_energy_percentage': float(dwt_result.energy_distribution[max_energy_level] * 100),
                    'reconstruction_error': float(dwt_result.reconstruction_error)
                })

                report['key_findings'].append({
                    'type': 'energy_distribution',
                    'description': f"Maximum energy at decomposition level {max_energy_level} "
                                 f"({dwt_result.energy_distribution[max_energy_level]*100:.1f}% of total energy)",
                    'clinical_relevance': 'Indicates frequency band with most signal energy'
                })

            # Process coherence results
            if 'coherence_analysis' in analysis_results:
                coherence_result = analysis_results['coherence_analysis']
                report['wavelet_methods_applied'].append('wavelet_coherence')

                mean_coherence = np.mean(coherence_result.coherence_matrix)
                max_coherence = np.max(coherence_result.coherence_matrix)

                report['technical_metrics'].update({
                    'mean_coherence': float(mean_coherence),
                    'max_coherence': float(max_coherence),
                    'coherence_significance_level': coherence_result.significance_level
                })

                if mean_coherence > 0.5:
                    report['key_findings'].append({
                        'type': 'strong_coherence',
                        'description': f"Strong wavelet coherence (mean = {mean_coherence:.3f}) between signals",
                        'clinical_relevance': 'Indicates synchronized oscillatory behavior'
                    })

            # Generate clinical insights
            report['clinical_insights'] = self._generate_clinical_insights(report)

        except Exception as e:
            logger.error(f"Error generating wavelet report for {signal_name}: {e}")
            report['error'] = str(e)

        return report

    def _generate_clinical_insights(self, report: Dict[str, Any]) -> List[str]:
        """Generate clinical insights based on wavelet analysis results"""
        insights = []

        # Analyze dominant frequency
        if 'dominant_frequency' in report['technical_metrics']:
            freq = report['technical_metrics']['dominant_frequency']
            if freq < 0.01:
                insights.append("Very low frequency components suggest long-term trends or slow physiological processes")
            elif freq < 0.1:
                insights.append("Low frequency components may indicate circadian rhythms or metabolic processes")
            elif freq < 1.0:
                insights.append("Medium frequency components could reflect autonomic nervous system activity")
            else:
                insights.append("High frequency components may indicate rapid physiological responses or artifacts")

        # Analyze energy distribution
        if 'max_energy_percentage' in report['technical_metrics']:
            energy_pct = report['technical_metrics']['max_energy_percentage']
            if energy_pct > 50:
                insights.append("Highly concentrated energy suggests dominant frequency band with potential clinical significance")
            elif energy_pct > 25:
                insights.append("Moderate energy concentration indicates multiple active frequency bands")

        # Analyze coherence
        if 'mean_coherence' in report['technical_metrics']:
            coherence = report['technical_metrics']['mean_coherence']
            if coherence > 0.7:
                insights.append("High coherence suggests strong coupling between physiological processes")
            elif coherence > 0.4:
                insights.append("Moderate coherence indicates partial synchronization of biological signals")

        return insights
