#!/usr/bin/env python3
"""
Advanced Analysis Demo for COTCAgent

This script demonstrates the advanced machine learning and statistical methods
implemented in COTCAgent, including Cox proportional hazards models with
time-dependent covariates and wavelet transform analysis.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Import COTCAgent and advanced modules
from cotc_agent import COTCAgent, DeepSeekConfig
from advanced_temporal_analysis import AdvancedTemporalAnalyzer, TimeDependentCovariate
from cox_model_advanced import AdvancedCoxAnalyzer
from wavelet_analysis_advanced import AdvancedWaveletAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_patient_data():
    """Create synthetic patient data for demonstration"""
    # Create realistic temporal medical data
    base_date = datetime(2020, 1, 1)

    # Simulate AFP (Alpha-fetoprotein) levels over time
    afp_dates = [base_date + timedelta(days=i*30) for i in range(24)]  # 2 years of data
    afp_levels = []

    # Simulate progression: Normal -> Mild -> Severe -> Critical
    for i, date in enumerate(afp_dates):
        if i < 6:
            level = "Normal"
        elif i < 12:
            level = "Mild"
        elif i < 18:
            level = "Severe"
        else:
            level = "Critical"
        afp_levels.append(level)

    # Simulate Hematemesis (vomiting blood)
    hematemesis_dates = [base_date + timedelta(days=i*45) for i in range(16)]
    hematemesis_levels = []

    for i, date in enumerate(hematemesis_dates):
        if i < 8:
            level = "Minor"
        elif i < 12:
            level = "Mild"
        else:
            level = "Critical"
        hematemesis_levels.append(level)

    # Simulate Headache patterns
    headache_dates = [base_date + timedelta(days=i*20) for i in range(36)]
    headache_levels = []

    for i, date in enumerate(headache_dates):
        # Simulate improvement over time with occasional exacerbations
        base_trend = max(0, 4 - i * 0.1)  # Improving trend
        noise = np.random.normal(0, 0.5)

        severity_value = base_trend + noise
        if severity_value < 1:
            level = "None"
        elif severity_value < 2:
            level = "Mild"
        elif severity_value < 3:
            level = "Medium"
        elif severity_value < 4:
            level = "Severe"
        else:
            level = "Extreme"
        headache_levels.append(level)

    patient_data = {
        "patient_info": {
            "id": "demo_patient_001",
            "age": 65,
            "gender": "Male",
            "confirmed_diseases": [
                {
                    "id": "D006229",
                    "name": "Mild Gouty Arthritis",
                    "description": "Chronic inflammatory arthritis affecting joints"
                },
                {
                    "id": "D007306",
                    "name": "Advanced Adrenal Cortical Insufficiency Damage",
                    "description": "Endocrine disorder affecting adrenal function"
                },
                {
                    "id": "D006392",
                    "name": "Recurrent Bacillary Dysentery Degeneration",
                    "description": "Chronic gastrointestinal infection"
                }
            ]
        },
        "temporal_indicators": {
            "basic_signs": {
                "Alpha-fetoprotein": {
                    "timestamps": [d.isoformat() for d in afp_dates],
                    "severity_levels": afp_levels
                },
                "Hematemesis": {
                    "timestamps": [d.isoformat() for d in hematemesis_dates],
                    "severity_levels": hematemesis_levels
                },
                "Headache": {
                    "timestamps": [d.isoformat() for d in headache_dates],
                    "severity_levels": headache_levels
                }
            },
            "blood_pressure_glucose": {
                "Epistaxis (BP/Glucose)": {
                    "timestamps": [
                        (base_date + timedelta(days=i*60)).isoformat() for i in range(8)
                    ],
                    "measurements": [85.5, 92.3, 88.7, 95.1, 82.4, 97.8, 89.2, 91.6]
                }
            },
            "health_advice": {
                "Dysphagia": {
                    "timestamps": [
                        (base_date + timedelta(days=i*40)).isoformat() for i in range(10)
                    ],
                    "measurements": [85.2, 78.9, 82.1, 75.4, 88.3, 79.6, 84.7, 81.2, 76.8, 83.5]
                }
            }
        }
    }

    return patient_data

def demonstrate_cox_model_analysis():
    """Demonstrate Cox proportional hazards model analysis"""
    print("\n" + "="*60)
    print("COX PROPORTIONAL HAZARDS MODEL ANALYSIS")
    print("="*60)

    try:
        # Create synthetic survival data
        np.random.seed(42)
        n_patients = 100

        # Generate baseline covariates
        age = np.random.normal(60, 10, n_patients)
        gender = np.random.binomial(1, 0.5, n_patients)  # 0=female, 1=male
        baseline_bmi = np.random.normal(25, 5, n_patients)

        # Create time-to-event data
        true_beta_age = 0.02  # Hazard ratio increases with age
        true_beta_gender = 0.5  # Males have higher hazard
        true_beta_bmi = 0.1    # Higher BMI increases hazard

        # Generate survival times
        linear_predictor = (true_beta_age * age +
                          true_beta_gender * gender +
                          true_beta_bmi * baseline_bmi)

        # Exponential survival times with censoring
        scale = 1000  # Baseline hazard scale
        u = np.random.uniform(0, 1, n_patients)
        survival_times = -scale * np.log(u) * np.exp(-linear_predictor)

        # Add censoring (30% censored)
        censoring_times = np.random.exponential(800, n_patients)
        observed_times = np.minimum(survival_times, censoring_times)
        event_indicator = (survival_times <= censoring_times).astype(int)

        # Create baseline covariates DataFrame
        covariates_df = pd.DataFrame({
            'age': age,
            'gender': gender,
            'bmi': baseline_bmi
        })

        # Create time-dependent covariates (simulated biomarker changes)
        biomarker_covariates = []

        for i in range(n_patients):
            # Simulate biomarker measurements over time
            measurement_times = np.sort(np.random.uniform(0, observed_times[i], 5))
            biomarker_values = np.random.normal(10, 2, len(measurement_times)) + \
                             0.1 * measurement_times  # Trending biomarker

            tdc = TimeDependentCovariate(
                name=f'biomarker_patient_{i}',
                time_points=measurement_times.tolist(),
                values=biomarker_values.tolist(),
                covariate_type='continuous'
            )
            biomarker_covariates.append(tdc)

        print(f"Generated synthetic survival data: {n_patients} patients")
        print(f"Events: {event_indicator.sum()}, Censored: {n_patients - event_indicator.sum()}")
        print(f"Median survival time: {np.median(observed_times):.1f} days")

        # Initialize Cox analyzer
        cox_analyzer = AdvancedCoxAnalyzer()

        # Fit Cox model
        cox_result = cox_analyzer.fit_cox_model_with_time_dependent_covariates(
            time_to_event=observed_times,
            event_indicator=event_indicator,
            baseline_covariates=covariates_df,
            time_dependent_covariates=biomarker_covariates[:5],  # Use first 5 for demo
            model_name="demo_cox_model"
        )

        print("\nCox Model Results:")
        print(f"Concordance Index: {cox_result.concordance_index:.3f}")
        print(f"Log-likelihood: {cox_result.log_likelihood:.2f}")
        print(f"AIC: {cox_result.aic:.2f}, BIC: {cox_result.bic:.2f}")
        print("\nHazard Ratios:")
        for var, hr in cox_result.hazard_ratios.items():
            ci_lower, ci_upper = cox_result.confidence_intervals[var]
            print(".3f")

        return cox_result

    except Exception as e:
        print(f"Cox model analysis failed: {e}")
        return None

def demonstrate_wavelet_analysis():
    """Demonstrate wavelet transform analysis"""
    print("\n" + "="*60)
    print("WAVELET TRANSFORM ANALYSIS")
    print("="*60)

    try:
        # Create synthetic physiological signal
        np.random.seed(42)
        t = np.linspace(0, 100, 1000)  # 100 seconds at 10 Hz sampling

        # Generate multi-component signal
        # Low frequency trend (0.1 Hz)
        trend = 0.5 * np.sin(2 * np.pi * 0.1 * t)

        # Medium frequency oscillations (1 Hz)
        oscillation1 = 0.3 * np.sin(2 * np.pi * 1.0 * t + np.pi/4)

        # High frequency noise (5 Hz)
        noise = 0.1 * np.sin(2 * np.pi * 5.0 * t)

        # Add a transient event
        transient = np.zeros_like(t)
        transient_mask = (t > 30) & (t < 35)
        transient[transient_mask] = 0.8 * np.exp(-(t[transient_mask] - 32.5)**2 / 2)

        # Combine signal components
        signal = trend + oscillation1 + noise + transient

        print(f"Generated synthetic signal: {len(signal)} samples, duration: {t[-1]:.1f} seconds")
        print("Signal components: low-frequency trend, medium-frequency oscillation, noise, transient event")

        # Initialize wavelet analyzer
        wavelet_analyzer = AdvancedWaveletAnalyzer()

        # Perform continuous wavelet transform
        cwt_result = wavelet_analyzer.continuous_wavelet_transform(
            signal,
            time_points=t,
            result_name="demo_signal_cwt"
        )

        print("
Continuous Wavelet Transform Results:")
        print(f"Coefficients shape: {cwt_result.coefficients.shape}")
        print(f"Scales range: {cwt_result.scales[0]:.1f} to {cwt_result.scales[-1]:.1f}")
        print(f"Frequencies range: {cwt_result.frequencies[0]:.3f} to {cwt_result.frequencies[-1]:.3f} Hz")

        # Find dominant scales
        mean_power_per_scale = np.mean(cwt_result.power_spectrum, axis=1)
        dominant_scale_idx = np.argmax(mean_power_per_scale)
        dominant_scale = cwt_result.scales[dominant_scale_idx]
        dominant_freq = cwt_result.frequencies[dominant_scale_idx]

        print(f"Dominant scale: {dominant_scale:.1f} (frequency: {dominant_freq:.3f} Hz)")

        # Perform discrete wavelet transform
        dwt_result = wavelet_analyzer.discrete_wavelet_transform(
            signal,
            result_name="demo_signal_dwt"
        )

        print("
Discrete Wavelet Transform Results:")
        print(f"Decomposition levels: {dwt_result.decomposition_levels}")
        print(f"Wavelet type: {dwt_result.wavelet_type}")
        print(f"Reconstruction error: {dwt_result.reconstruction_error:.6f}")

        print("\nEnergy distribution by frequency band:")
        for level, energy in dwt_result.energy_distribution.items():
            freq = dwt_result.dominant_frequencies[level]
            print(".1f")

        # Detect transient events
        transient_events = wavelet_analyzer.detect_transient_events(
            signal, cwt_result, threshold=2.0, min_duration=10
        )

        print(f"\nDetected {len(transient_events)} transient events")
        if transient_events:
            top_event = transient_events[0]
            print(f"Most significant event: time index {top_event['time_index']}, "
                  f"scale {top_event['scale']:.1f}, significance {top_event['significance']:.2f}")

        return cwt_result, dwt_result

    except Exception as e:
        print(f"Wavelet analysis failed: {e}")
        return None, None

def demonstrate_advanced_temporal_analysis():
    """Demonstrate advanced temporal analysis on patient data"""
    print("\n" + "="*60)
    print("ADVANCED TEMPORAL ANALYSIS")
    print("="*60)

    try:
        # Create synthetic patient data
        patient_data = create_synthetic_patient_data()

        print(f"Analyzing patient: {patient_data['patient_info']['id']}")
        print(f"Confirmed diseases: {len(patient_data['patient_info']['confirmed_diseases'])}")
        print(f"Temporal indicators: {len(patient_data['temporal_indicators']['basic_signs'])} symptoms")

        # Initialize temporal analyzer
        temporal_analyzer = AdvancedTemporalAnalyzer()

        # Generate comprehensive report
        comprehensive_report = temporal_analyzer.generate_comprehensive_report(patient_data)

        print("
Comprehensive Analysis Report:")
        print(f"Analysis timestamp: {comprehensive_report['analysis_timestamp']}")

        print(f"\nPatient risk assessment:")
        risk = comprehensive_report['risk_assessment']
        print(f"Overall risk score: {risk['overall_risk_score']:.3f}")
        print(f"Risk level: {risk['risk_level']}")
        print(f"Risk factors identified: {len(risk['risk_factors'])}")

        if risk['risk_factors']:
            print("Top risk factors:")
            for factor in risk['risk_factors'][:3]:
                print(f"  - {factor}")

        print(f"\nClinical recommendations: {len(comprehensive_report['recommendations'])}")
        for rec in comprehensive_report['recommendations'][:3]:
            print(f"  - {rec}")

        # Analyze individual symptoms
        temporal_analysis = comprehensive_report['temporal_analysis']
        print(f"\nAnalyzed symptoms: {len(temporal_analysis)}")

        for symptom_name, analysis in temporal_analysis.items():
            if 'methods_applied' in analysis:
                methods = analysis['methods_applied']
                print(f"  {symptom_name}: {len(methods)} advanced methods applied")
                if methods:
                    print(f"    Methods: {', '.join(methods)}")

        return comprehensive_report

    except Exception as e:
        print(f"Advanced temporal analysis failed: {e}")
        return None

def run_full_cotc_agent_demo():
    """Run complete COTCAgent analysis with advanced methods"""
    print("\n" + "="*80)
    print("COMPLETE COTCAgent ANALYSIS WITH ADVANCED ML METHODS")
    print("="*80)

    try:
        # Create synthetic patient data
        patient_data = create_synthetic_patient_data()

        # Initialize COTCAgent (mock configuration for demo)
        config = DeepSeekConfig(
            api_key="demo_key",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.1,
            max_tokens=2000
        )

        # Note: In real usage, COTCAgent would make API calls
        # For demo, we'll show the structure
        print("COTCAgent would process the following query:")
        user_query = "I have been experiencing occasional hematemesis for 6 months and worsening headaches for 1 month. Is this related to my liver condition?"

        print(f"Query: {user_query}")
        print(f"Patient ID: {patient_data['patient_info']['id']}")

        # Show what the advanced analysis would include
        print("\nAdvanced analysis components that would be performed:")

        analysis_components = [
            "1. Time Series Analysis (TSA) - Extract trends from AFP, Hematemesis, Headache",
            "2. Probabilistic Chain-of-Thought (CoT) - Match trends to disease knowledge base",
            "3. Consultation Workflow - Resolve gaps with patient responses",
            "4. Advanced ML Methods:",
            "   - Cox Proportional Hazards Model with time-dependent covariates",
            "   - Continuous Wavelet Transform (CWT) for multi-resolution analysis",
            "   - Discrete Wavelet Transform (DWT) for energy distribution",
            "   - Wavelet Coherence for biomarker relationships",
            "   - Time-dependent ROC analysis for predictive accuracy",
            "   - ARIMA modeling for temporal forecasting",
            "5. Risk Assessment - Dynamic risk prediction over time",
            "6. Clinical Recommendations - Evidence-based intervention guidance"
        ]

        for component in analysis_components:
            print(f"  {component}")

        print("
Expected output would include:")
        print("- Temporal trend analysis with statistical significance")
        print("- Disease risk ranking with confidence intervals")
        print("- Time-dependent hazard ratios and survival probabilities")
        print("- Wavelet-based frequency domain insights")
        print("- Dynamic risk trajectories over time")
        print("- Clinical decision support recommendations")

    except Exception as e:
        print(f"COTCAgent demo failed: {e}")

def main():
    """Main demonstration function"""
    print("COTCAgent Advanced Analysis Demonstration")
    print("==========================================")

    # Demonstrate individual components
    cox_result = demonstrate_cox_model_analysis()
    cwt_result, dwt_result = demonstrate_wavelet_analysis()
    temporal_report = demonstrate_advanced_temporal_analysis()

    # Show integrated analysis
    run_full_cotc_agent_demo()

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nThis demo showcased the advanced machine learning capabilities")
    print("implemented in COTCAgent, including:")
    print("• Cox Proportional Hazards Models with time-dependent covariates")
    print("• Continuous and Discrete Wavelet Transform Analysis")
    print("• Multiresolution temporal pattern analysis")
    print("• Dynamic risk prediction and clinical decision support")
    print("\nThese methods provide sophisticated analysis of temporal medical")
    print("data for improved diagnostic accuracy and patient outcomes.")

if __name__ == "__main__":
    main()
