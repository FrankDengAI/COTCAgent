"""
Input validation and data sanitization utilities for COTCAgent
"""

import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, date
import json

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass


@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Optional[Any] = None


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input text

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized text

    Raises:
        ValidationError: If input is invalid
    """
    if not isinstance(text, str):
        raise ValidationError("Input must be a string")

    if not text.strip():
        raise ValidationError("Input cannot be empty or whitespace only")

    if len(text) > max_length:
        logger.warning(f"Input length {len(text)} exceeds max_length {max_length}, truncating")
        text = text[:max_length]

    # Remove potentially dangerous characters but keep medical terminology
    # Allow alphanumeric, spaces, basic punctuation, and medical symbols
    sanitized = re.sub(r'[^\w\s\.,;:!?\-\+\=\(\)\[\]\{\}\/\%\&]', '', text)

    # Normalize whitespace
    sanitized = ' '.join(sanitized.split())

    if not sanitized.strip():
        raise ValidationError("Input contains no valid characters after sanitization")

    return sanitized


def validate_patient_data(patient_data: Dict) -> ValidationResult:
    """
    Validate patient data structure and content

    Args:
        patient_data: Dictionary containing patient information

    Returns:
        ValidationResult with validation status and details
    """
    errors = []
    warnings = []
    sanitized_data = patient_data.copy()

    # Check required fields
    required_fields = ['patient_info']
    for field in required_fields:
        if field not in patient_data:
            errors.append(f"Missing required field: {field}")
            continue

    if 'patient_info' in patient_data:
        patient_info = patient_data['patient_info']

        # Validate patient ID
        if 'id' not in patient_info:
            errors.append("Patient ID is required")
        elif not isinstance(patient_info['id'], (str, int)):
            errors.append("Patient ID must be string or integer")
        else:
            # Sanitize patient ID
            sanitized_data['patient_info']['id'] = str(patient_info['id'])

        # Validate age
        if 'age' in patient_info:
            try:
                age = int(patient_info['age'])
                if age < 0 or age > 150:
                    errors.append("Age must be between 0 and 150")
                else:
                    sanitized_data['patient_info']['age'] = age
            except (ValueError, TypeError):
                errors.append("Age must be a valid integer")

        # Validate gender
        if 'gender' in patient_info:
            gender = str(patient_info['gender']).lower()
            valid_genders = ['male', 'female', 'other', 'unknown']
            if gender not in valid_genders:
                warnings.append(f"Gender '{gender}' not in standard values: {valid_genders}")
            sanitized_data['patient_info']['gender'] = gender

        # Validate dates
        date_fields = ['birth_date', 'admission_date', 'discharge_date']
        for date_field in date_fields:
            if date_field in patient_info:
                try:
                    # Try multiple date formats
                    date_str = str(patient_info[date_field])
                    parsed_date = None

                    # Try ISO format first
                    try:
                        parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except ValueError:
                        # Try common formats
                        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d %H:%M:%S']:
                            try:
                                parsed_date = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue

                    if parsed_date is None:
                        errors.append(f"Invalid date format for {date_field}: {date_str}")
                    else:
                        sanitized_data['patient_info'][date_field] = parsed_date.date().isoformat()

                except Exception as e:
                    errors.append(f"Error parsing {date_field}: {e}")

    # Validate medical data structure
    if 'medical_data' in patient_data:
        medical_data = patient_data['medical_data']

        # Validate vital signs
        if 'vital_signs' in medical_data:
            vital_signs = medical_data['vital_signs']
            if isinstance(vital_signs, dict):
                # Check for reasonable ranges
                vital_ranges = {
                    'heart_rate': (30, 250),  # bpm
                    'blood_pressure_systolic': (70, 250),  # mmHg
                    'blood_pressure_diastolic': (40, 150),  # mmHg
                    'temperature': (30.0, 45.0),  # Celsius
                    'respiratory_rate': (5, 60),  # breaths/min
                    'oxygen_saturation': (70, 100),  # percentage
                }

                for vital, (min_val, max_val) in vital_ranges.items():
                    if vital in vital_signs:
                        try:
                            value = float(vital_signs[vital])
                            if not (min_val <= value <= max_val):
                                warnings.append(f"{vital} value {value} outside normal range [{min_val}, {max_val}]")
                        except (ValueError, TypeError):
                            errors.append(f"Invalid {vital} value: {vital_signs[vital]}")

        # Validate lab results
        if 'lab_results' in medical_data:
            lab_results = medical_data['lab_results']
            if isinstance(lab_results, list):
                for i, result in enumerate(lab_results):
                    if not isinstance(result, dict):
                        errors.append(f"Lab result {i} must be a dictionary")
                        continue

                    required_lab_fields = ['test_name', 'value']
                    for field in required_lab_fields:
                        if field not in result:
                            errors.append(f"Lab result {i} missing required field: {field}")

    # Check for sensitive data exposure
    sensitive_fields = ['ssn', 'social_security', 'credit_card', 'password']
    patient_json = json.dumps(patient_data, default=str).lower()

    for field in sensitive_fields:
        if field in patient_json:
            errors.append(f"Potential sensitive data detected: {field}")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, sanitized_data if is_valid else None)


def validate_api_config(config: Dict) -> ValidationResult:
    """
    Validate API configuration

    Args:
        config: API configuration dictionary

    Returns:
        ValidationResult with validation status
    """
    errors = []
    warnings = []

    required_fields = ['api_key', 'api_base', 'model']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required API config field: {field}")
        elif not config[field]:
            errors.append(f"API config field {field} cannot be empty")

    # Validate API key format (basic check)
    if 'api_key' in config:
        api_key = config['api_key']
        if not isinstance(api_key, str):
            errors.append("API key must be a string")
        elif len(api_key) < 10:
            errors.append("API key seems too short")
        elif api_key.startswith('sk-') and len(api_key) != 51:  # OpenAI style key
            warnings.append("API key format may be incorrect")

    # Validate URL
    if 'api_base' in config:
        api_base = config['api_base']
        if not isinstance(api_base, str):
            errors.append("API base must be a string")
        elif not api_base.startswith(('http://', 'https://')):
            errors.append("API base must be a valid HTTP/HTTPS URL")

    # Validate numeric parameters
    numeric_params = {
        'max_tokens': (1, 100000),
        'temperature': (0.0, 2.0),
        'timeout': (1, 300)
    }

    for param, (min_val, max_val) in numeric_params.items():
        if param in config:
            try:
                value = float(config[param])
                if not (min_val <= value <= max_val):
                    errors.append(f"{param} must be between {min_val} and {max_val}")
            except (ValueError, TypeError):
                errors.append(f"{param} must be a valid number")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings)


def sanitize_sql_query(query: str) -> str:
    """
    Sanitize SQL query to prevent injection attacks

    Args:
        query: SQL query string

    Returns:
        Sanitized query (basic sanitization)

    Note: This is NOT a complete SQL injection prevention solution.
    Always use parameterized queries in production.
    """
    if not isinstance(query, str):
        raise ValidationError("SQL query must be a string")

    # Remove potentially dangerous keywords
    dangerous_keywords = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
        'EXEC', 'EXECUTE', 'UNION', 'SELECT', '--', '/*', '*/'
    ]

    query_upper = query.upper()
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            logger.warning(f"Potentially dangerous SQL keyword detected: {keyword}")
            # Don't actually remove, just log for now
            break

    return query


def validate_medical_query(query: str) -> ValidationResult:
    """
    Validate medical query content and intent

    Args:
        query: Medical query string

    Returns:
        ValidationResult with validation status
    """
    errors = []
    warnings = []

    # Check for medical relevance
    medical_keywords = [
        'symptom', 'pain', 'fever', 'diagnosis', 'treatment', 'medication',
        'blood', 'heart', 'lung', 'stomach', 'head', 'chest', 'arm', 'leg',
        'doctor', 'hospital', 'clinic', 'patient', 'medical', 'health'
    ]

    query_lower = query.lower()
    medical_score = sum(1 for keyword in medical_keywords if keyword in query_lower)

    if medical_score < 2:
        warnings.append("Query may not be medically relevant")

    # Check for emergency keywords
    emergency_keywords = [
        'emergency', 'urgent', 'severe pain', 'difficulty breathing',
        'chest pain', 'unconscious', 'bleeding heavily', 'heart attack',
        'stroke', 'seizure'
    ]

    emergency_score = sum(1 for keyword in emergency_keywords if keyword in query_lower)

    if emergency_score > 0:
        warnings.append("Query contains emergency-related keywords - recommend immediate medical attention")

    # Check query length
    if len(query) < 10:
        errors.append("Query is too short for meaningful analysis")
    elif len(query) > 2000:
        errors.append("Query is too long - please be more specific")

    # Check for inappropriate content
    inappropriate_keywords = [
        'illegal', 'drug', 'weapon', 'harm', 'suicide', 'kill', 'death'
    ]

    inappropriate_score = sum(1 for keyword in inappropriate_keywords if keyword in query_lower)

    if inappropriate_score > 0:
        warnings.append("Query may contain sensitive or inappropriate content")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings)
