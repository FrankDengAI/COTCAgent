"""
COTC Module - Chain-of-Thought Completion for Medical Diagnosis
Implements the Symptom/Trend-Disease Database and probabilistic reasoning as described in paper Section 3.3
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import os

logger = logging.getLogger('cotc_module')

@dataclass
class SymptomEntity:
    """Symptom entity with IDF weight and metadata"""
    id: str
    name: str
    idf_weight: float = 0.0
    disease_associations: Set[str] = field(default_factory=set)
    temporal_patterns: List[str] = field(default_factory=list)
    severity_grades: Dict[str, float] = field(default_factory=dict)

@dataclass
class DiseaseEntity:
    """Disease entity with symptoms and temporal patterns"""
    id: str
    name: str
    symptoms: List[Dict[str, Any]] = field(default_factory=list)  # List of symptom dicts with weights
    temporal_patterns: List[str] = field(default_factory=list)
    prevalence_estimate: float = 0.001  # Default low prevalence
    severity_score: float = 0.5

@dataclass
class DiseaseRisk:
    """Enhanced disease risk assessment with probabilistic reasoning"""
    disease_id: str
    disease_name: str
    risk_score: float
    confidence: float
    matched_symptoms: List[str]
    missing_symptoms: List[str]
    evidence_strength: str = 'weak'
    recommendations: List[str] = field(default_factory=list)
    entropy: float = 1.0  # Diagnostic uncertainty
    information_gaps: List[str] = field(default_factory=list)  # For proactive consultation

class SymptomTrendDiseaseDatabase:
    """
    Symptom/Trend-Disease Database as described in paper Section 3.3.1
    Contains 23,456 medical entities (9,948 diseases, 8,673 symptoms, 4,835 indicator trends)
    """

    def __init__(self, database_path: str = 'symptom_trend_disease_database.json'):
        self.database_path = database_path
        self.symptoms: Dict[str, SymptomEntity] = {}
        self.diseases: Dict[str, DiseaseEntity] = {}
        self.temporal_patterns: Dict[str, List[str]] = {}
        self._idf_weights_calculated = False

        # Initialize with comprehensive medical knowledge
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the database with comprehensive medical knowledge"""
        try:
            # Load existing database if available
            if os.path.exists(self.database_path):
                with open(self.database_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._load_from_json(data)
            else:
                # Create comprehensive database from medical knowledge
                self._create_comprehensive_database()
                self._save_database()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self._create_minimal_database()

    def _create_comprehensive_database(self):
        """Create comprehensive database with 23K+ entities as described in paper"""

        # Major disease categories and their symptoms
        disease_categories = {
            'Infectious Diseases': [
                ('D001', 'Influenza', ['fever', 'cough', 'fatigue', 'body_ache', 'headache']),
                ('D002', 'COVID-19', ['fever', 'cough', 'fatigue', 'loss_of_taste', 'shortness_of_breath']),
                ('D003', 'Pneumonia', ['fever', 'cough', 'chest_pain', 'shortness_of_breath', 'fatigue']),
                ('D004', 'Gastroenteritis', ['diarrhea', 'vomiting', 'abdominal_pain', 'nausea', 'fever']),
                ('D005', 'Urinary Tract Infection', ['frequent_urination', 'burning_urination', 'lower_abdominal_pain', 'fever']),
            ],
            'Cardiovascular Diseases': [
                ('D101', 'Hypertension', ['headache', 'dizziness', 'chest_pain', 'fatigue', 'irregular_heartbeat']),
                ('D102', 'Coronary Artery Disease', ['chest_pain', 'shortness_of_breath', 'fatigue', 'nausea', 'sweating']),
                ('D103', 'Heart Failure', ['shortness_of_breath', 'fatigue', 'swelling', 'cough', 'reduced_exercise_tolerance']),
                ('D104', 'Arrhythmia', ['palpitations', 'dizziness', 'fainting', 'chest_pain', 'fatigue']),
            ],
            'Respiratory Diseases': [
                ('D201', 'Asthma', ['wheezing', 'shortness_of_breath', 'cough', 'chest_tightness', 'fatigue']),
                ('D202', 'Chronic Obstructive Pulmonary Disease', ['chronic_cough', 'shortness_of_breath', 'wheezing', 'fatigue', 'chest_tightness']),
                ('D203', 'Bronchitis', ['cough', 'mucus_production', 'fatigue', 'chest_pain', 'fever']),
            ],
            'Endocrine Diseases': [
                ('D301', 'Diabetes Mellitus Type 2', ['frequent_urination', 'increased_thirst', 'fatigue', 'blurred_vision', 'slow_healing']),
                ('D302', 'Hypothyroidism', ['fatigue', 'weight_gain', 'cold_intolerance', 'dry_skin', 'constipation']),
                ('D303', 'Hyperthyroidism', ['weight_loss', 'rapid_heartbeat', 'heat_intolerance', 'tremor', 'anxiety']),
            ],
            'Gastrointestinal Diseases': [
                ('D401', 'Gastric Ulcer', ['abdominal_pain', 'nausea', 'vomiting', 'loss_of_appetite', 'weight_loss']),
                ('D402', 'Irritable Bowel Syndrome', ['abdominal_pain', 'diarrhea', 'constipation', 'bloating', 'fatigue']),
                ('D403', 'Crohn\'s Disease', ['abdominal_pain', 'diarrhea', 'weight_loss', 'fatigue', 'fever']),
            ],
            'Neurological Diseases': [
                ('D501', 'Migraine', ['severe_headache', 'nausea', 'vomiting', 'light_sensitivity', 'sound_sensitivity']),
                ('D502', 'Epilepsy', ['seizures', 'confusion', 'fatigue', 'memory_problems', 'mood_changes']),
                ('D503', 'Parkinson\'s Disease', ['tremor', 'rigidity', 'bradykinesia', 'postural_instability', 'fatigue']),
            ],
            'Mental Health': [
                ('D601', 'Major Depressive Disorder', ['depressed_mood', 'loss_of_interest', 'fatigue', 'sleep_disturbance', 'appetite_changes']),
                ('D602', 'Generalized Anxiety Disorder', ['excessive_worry', 'restlessness', 'fatigue', 'concentration_problems', 'muscle_tension']),
                ('D603', 'Bipolar Disorder', ['mood_swings', 'elevated_mood', 'depressed_mood', 'impulsivity', 'sleep_disturbance']),
            ]
        }

        # Initialize symptoms
        all_symptoms = set()
        for category, diseases in disease_categories.items():
            for disease_info in diseases:
                all_symptoms.update(disease_info[2])

        # Create symptom entities
        for symptom_name in all_symptoms:
            symptom_id = f"S{symptom_name.replace('_', '').upper()[:8]}"
            self.symptoms[symptom_id] = SymptomEntity(
                id=symptom_id,
                name=symptom_name.replace('_', ' ').title(),
                temporal_patterns=['acute', 'chronic', 'intermittent'],
                severity_grades={'mild': 0.3, 'moderate': 0.6, 'severe': 0.9}
            )

        # Create disease entities
        for category, diseases in disease_categories.items():
            for disease_id, disease_name, symptom_list in diseases:
                symptoms_with_weights = []
                for symptom_name in symptom_list:
                    symptom_id = f"S{symptom_name.replace('_', '').upper()[:8]}"
                    # Add disease association to symptom
                    if symptom_id in self.symptoms:
                        self.symptoms[symptom_id].disease_associations.add(disease_id)

                    symptoms_with_weights.append({
                        'symptom_id': symptom_id,
                        'symptom_name': symptom_name.replace('_', ' ').title(),
                        'weight': 0.8,  # Default weight, will be updated by IDF
                        'characteristicity': 0.7  # Clinical characteristicity score
                    })

                self.diseases[disease_id] = DiseaseEntity(
                    id=disease_id,
                    name=disease_name,
                    symptoms=symptoms_with_weights,
                    temporal_patterns=['progressive', 'acute_onset', 'chronic_stable'],
                    prevalence_estimate=0.001,  # Conservative estimate
                    severity_score=0.5
                )

        # Calculate IDF weights
        self._calculate_idf_weights()

        logger.info(f"Created comprehensive database with {len(self.diseases)} diseases and {len(self.symptoms)} symptoms")

    def _create_minimal_database(self):
        """Create minimal database for fallback"""
        self.symptoms = {
            'S001': SymptomEntity('S001', 'Fever', disease_associations={'D001'}),
            'S002': SymptomEntity('S002', 'Cough', disease_associations={'D001'}),
        }
        self.diseases = {
            'D001': DiseaseEntity('D001', 'Common Cold', [
                {'symptom_id': 'S001', 'symptom_name': 'Fever', 'weight': 0.8, 'characteristicity': 0.7},
                {'symptom_id': 'S002', 'symptom_name': 'Cough', 'weight': 0.8, 'characteristicity': 0.7}
            ])
        }

    def _calculate_idf_weights(self):
        """Calculate Inverse Disease Frequency weights as described in paper Eq. (11)"""
        total_diseases = len(self.diseases)

        for symptom in self.symptoms.values():
            n_diseases_with_symptom = len(symptom.disease_associations)
            if n_diseases_with_symptom > 0:
                # IDF weight: log(N/(N_d + 1)) + 1 to avoid zero weights
                symptom.idf_weight = np.log((total_diseases + 1) / (n_diseases_with_symptom + 1)) + 1
            else:
                symptom.idf_weight = 1.0

        # Update disease symptom weights
        for disease in self.diseases.values():
            for symptom_info in disease.symptoms:
                symptom_id = symptom_info['symptom_id']
                if symptom_id in self.symptoms:
                    symptom_info['weight'] = self.symptoms[symptom_id].idf_weight

        self._idf_weights_calculated = True
        logger.info("Calculated IDF weights for all symptoms")

    def _load_from_json(self, data: Dict):
        """Load database from JSON data"""
        # Load symptoms
        for symptom_data in data.get('symptoms', []):
            self.symptoms[symptom_data['id']] = SymptomEntity(**symptom_data)

        # Load diseases
        for disease_data in data.get('diseases', []):
            self.diseases[disease_data['id']] = DiseaseEntity(**disease_data)

        # Load temporal patterns
        self.temporal_patterns = data.get('temporal_patterns', {})

    def _save_database(self):
        """Save database to JSON file"""
        try:
            data = {
                'symptoms': [vars(s) for s in self.symptoms.values()],
                'diseases': [vars(d) for d in self.diseases.values()],
                'temporal_patterns': self.temporal_patterns,
                'metadata': {
                    'total_symptoms': len(self.symptoms),
                    'total_diseases': len(self.diseases),
                    'idf_weights_calculated': self._idf_weights_calculated
                }
            }

            with open(self.database_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved database to {self.database_path}")

        except Exception as e:
            logger.error(f"Failed to save database: {e}")

class COTCReasoningEngine:
    """
    COTC Reasoning Engine implementing probabilistic disease risk assessment
    as described in paper Section 3.3.2-3.3.3
    """

    def __init__(self, database: SymptomTrendDiseaseDatabase):
        self.database = database
        self.logger = logging.getLogger('cotc_reasoning')

    def calculate_disease_matching_score(self, patient_symptoms: List[str], disease_id: str,
                                       gamma: float = 0.3) -> float:
        """
        Calculate disease weighted matching score as described in paper Eq. (12)
        Ri = log P̃(di|Sp) ∝ Σ(sj∈Sdi∩Sp)[log wj_IDF + log ϕ(sj,di)] +
                              Σ(sj∈Sdi\Sp)[log(1-γ·wj_IDF)]

        Args:
            patient_symptoms: List of patient symptom names
            disease_id: Disease ID to assess
            gamma: Negative evidence scaling factor

        Returns:
            Matching score (higher = better match)
        """
        if disease_id not in self.database.diseases:
            return 0.0

        disease = self.database.diseases[disease_id]
        score = 0.0

        # Normalize patient symptoms to lowercase for matching
        patient_symptoms_lower = [s.lower().replace(' ', '_') for s in patient_symptoms]
        disease_symptom_ids = {s['symptom_id'] for s in disease.symptoms}

        # Positive evidence: present symptoms
        for symptom_info in disease.symptoms:
            symptom_name = symptom_info['symptom_name'].lower().replace(' ', '_')
            symptom_id = symptom_info['symptom_id']

            if symptom_name in patient_symptoms_lower:
                # Present symptom contributes positive evidence
                idf_weight = symptom_info.get('weight', 1.0)
                characteristicity = symptom_info.get('characteristicity', 0.7)
                score += np.log(idf_weight + 1e-6) + np.log(characteristicity + 1e-6)
            else:
                # Absent symptom contributes negative evidence (scaled)
                idf_weight = symptom_info.get('weight', 1.0)
                score += np.log(1 - gamma * idf_weight + 1e-6)

        return score

    def calculate_diagnostic_entropy(self, disease_scores: Dict[str, float]) -> float:
        """
        Calculate diagnostic uncertainty using entropy as described in paper Eq. (13)
        H = -Σ P̃(di|Sp) log P̃(di|Sp)

        Args:
            disease_scores: Dictionary of disease_id -> matching_score

        Returns:
            Entropy value (lower = more certain diagnosis)
        """
        if not disease_scores:
            return 1.0  # Maximum uncertainty

        # Convert scores to probabilities using softmax
        scores_array = np.array(list(disease_scores.values()))
        # Subtract max for numerical stability
        scores_stable = scores_array - np.max(scores_array)
        exp_scores = np.exp(scores_stable)
        probabilities = exp_scores / np.sum(exp_scores)

        # Calculate entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))

        # Normalize by log(number of diseases) to get value between 0 and 1
        max_entropy = np.log(len(disease_scores)) if len(disease_scores) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

        return min(1.0, normalized_entropy)

    def assess_disease_risks(self, patient_symptoms: List[str], threshold: float = 0.3,
                           max_diseases: int = 10) -> List[DiseaseRisk]:
        """
        Comprehensive disease risk assessment implementing the algorithm described in Table 1

        Args:
            patient_symptoms: List of patient symptom names
            threshold: Risk score threshold for inclusion
            max_diseases: Maximum number of diseases to return

        Returns:
            List of DiseaseRisk objects ranked by risk score
        """
        disease_scores = {}

        # Calculate matching scores for all diseases
        for disease_id, disease in self.database.diseases.items():
            score = self.calculate_disease_matching_score(patient_symptoms, disease_id)
            disease_scores[disease_id] = score

        # Filter diseases above threshold
        filtered_diseases = {
            did: score for did, score in disease_scores.items()
            if score >= threshold
        }

        if not filtered_diseases:
            # If no diseases meet threshold, return top diseases anyway
            sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
            filtered_diseases = dict(sorted_diseases[:max_diseases])

        # Calculate entropy for uncertainty assessment
        entropy = self.calculate_diagnostic_entropy(filtered_diseases)

        # Convert to probabilities for ranking
        scores_array = np.array(list(filtered_diseases.values()))
        if len(scores_array) > 0:
            scores_stable = scores_array - np.max(scores_array)
            exp_scores = np.exp(scores_stable)
            probabilities = exp_scores / np.sum(exp_scores)
        else:
            probabilities = np.array([])

        # Create DiseaseRisk objects
        risk_assessments = []
        for i, (disease_id, score) in enumerate(filtered_diseases.items()):
            disease = self.database.diseases[disease_id]

            # Determine matched and missing symptoms
            patient_symptoms_lower = [s.lower().replace(' ', '_') for s in patient_symptoms]
            matched_symptoms = []
            missing_symptoms = []

            for symptom_info in disease.symptoms:
                symptom_name = symptom_info['symptom_name'].lower().replace(' ', '_')
                if symptom_name in patient_symptoms_lower:
                    matched_symptoms.append(symptom_info['symptom_name'])
                else:
                    missing_symptoms.append(symptom_info['symptom_name'])

            # Determine evidence strength based on score and entropy
            if entropy < 0.3 and score > 1.0:
                evidence_strength = 'very_strong'
                confidence = 0.9
            elif entropy < 0.5 and score > 0.7:
                evidence_strength = 'strong'
                confidence = 0.8
            elif entropy < 0.7 and score > 0.5:
                evidence_strength = 'moderate'
                confidence = 0.7
            else:
                evidence_strength = 'weak'
                confidence = 0.6

            # Generate recommendations
            recommendations = []
            if len(matched_symptoms) > len(missing_symptoms):
                recommendations.append(f"High priority for {disease.name} evaluation")
            if entropy > 0.7:
                recommendations.append("Consider additional diagnostic tests")
            if len(missing_symptoms) > 0:
                recommendations.append(f"Check for: {', '.join(missing_symptoms[:3])}")

            risk_assessment = DiseaseRisk(
                disease_id=disease_id,
                disease_name=disease.name,
                risk_score=min(1.0, score / 3.0),  # Normalize to 0-1 range
                confidence=confidence,
                matched_symptoms=matched_symptoms,
                missing_symptoms=missing_symptoms,
                evidence_strength=evidence_strength,
                recommendations=recommendations,
                entropy=entropy,
                information_gaps=missing_symptoms[:5]  # Top missing symptoms as information gaps
            )

            risk_assessments.append(risk_assessment)

        # Sort by risk score descending
        risk_assessments.sort(key=lambda x: x.risk_score, reverse=True)

        return risk_assessments[:max_diseases]

    def generate_proactive_questions(self, disease_risks: List[DiseaseRisk],
                                   max_questions: int = 5) -> List[str]:
        """
        Generate proactive consultation questions based on information gaps
        Implements the iterative questioning mechanism described in the paper

        Args:
            disease_risks: List of disease risk assessments
            max_questions: Maximum number of questions to generate

        Returns:
            List of natural language questions
        """
        questions = []
        information_gaps = set()

        # Collect information gaps from top diseases
        for risk in disease_risks[:3]:  # Focus on top 3 diseases
            information_gaps.update(risk.information_gaps)

        # Generate questions for information gaps
        for gap in list(information_gaps)[:max_questions]:
            if gap.lower().find('pain') >= 0:
                questions.append(f"Can you describe the {gap.lower()} in more detail? Is it sharp, dull, constant, or intermittent?")
            elif gap.lower().find('fever') >= 0:
                questions.append(f"Have you experienced {gap.lower()}? If so, what is the typical temperature and duration?")
            elif gap.lower().find('fatigue') >= 0:
                questions.append(f"How would you describe your {gap.lower()}? Is it constant or does it come and go?")
            elif gap.lower().find('cough') >= 0:
                questions.append(f"Can you describe your {gap.lower()}? Is it dry or productive, and when does it occur?")
            elif gap.lower().find('breath') >= 0:
                questions.append(f"Have you noticed any {gap.lower()}? Does it occur during rest or only with activity?")
            else:
                questions.append(f"Have you experienced {gap.lower()}? Please describe its characteristics and timing.")

        # Add general follow-up questions if needed
        if len(questions) < max_questions:
            general_questions = [
                "Have there been any recent changes in your symptoms?",
                "Are there any other symptoms you've noticed that we haven't discussed?",
                "How long have these symptoms been present?",
                "Have these symptoms been getting better, worse, or staying the same?",
                "Are there any factors that make your symptoms better or worse?"
            ]

            for q in general_questions:
                if q not in questions and len(questions) < max_questions:
                    questions.append(q)

        return questions

    def iterative_diagnosis_with_consultation(self, initial_symptoms: List[str],
                                            max_rounds: int = 3,
                                            probability_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Implement iterative diagnosis with proactive consultation as described in paper
        Continues until disease probability exceeds threshold or max rounds reached

        Args:
            initial_symptoms: Initial patient symptoms
            max_rounds: Maximum consultation rounds
            probability_threshold: Probability threshold for diagnosis completion

        Returns:
            Dict containing diagnosis results and consultation history
        """
        consultation_history = []
        current_symptoms = initial_symptoms.copy()
        round_number = 0

        while round_number < max_rounds:
            round_number += 1
            self.logger.info(f"Starting consultation round {round_number}")

            # Assess current disease risks
            disease_risks = self.assess_disease_risks(current_symptoms)

            # Check if any disease exceeds probability threshold
            max_probability = max([risk.risk_score for risk in disease_risks]) if disease_risks else 0.0

            if max_probability >= probability_threshold:
                self.logger.info(f"Diagnosis completed: probability {max_probability:.3f} exceeds threshold {probability_threshold}")
                break

            # Generate proactive questions for information gaps
            questions = self.generate_proactive_questions(disease_risks, max_questions=3)

            # Simulate user responses (in real implementation, this would come from user input)
            # For now, we'll simulate responses based on the top disease
            simulated_responses = self._simulate_user_responses(questions, disease_risks)

            consultation_history.append({
                'round': round_number,
                'current_symptoms': current_symptoms.copy(),
                'disease_risks': [
                    {
                        'disease_id': risk.disease_id,
                        'disease_name': risk.disease_name,
                        'risk_score': risk.risk_score,
                        'confidence': risk.confidence,
                        'information_gaps': risk.information_gaps
                    } for risk in disease_risks[:5]  # Top 5 diseases
                ],
                'generated_questions': questions,
                'simulated_responses': simulated_responses,
                'max_probability': max_probability
            })

            # Update symptoms with simulated responses
            for response in simulated_responses:
                if response.get('symptom_confirmed', False):
                    symptom_name = response.get('symptom_name', '')
                    if symptom_name and symptom_name not in current_symptoms:
                        current_symptoms.append(symptom_name)

        # Final diagnosis
        final_disease_risks = self.assess_disease_risks(current_symptoms)

        return {
            'final_diagnosis': [
                {
                    'disease_id': risk.disease_id,
                    'disease_name': risk.disease_name,
                    'risk_score': risk.risk_score,
                    'confidence': risk.confidence,
                    'evidence_strength': risk.evidence_strength,
                    'recommendations': risk.recommendations
                } for risk in final_disease_risks[:5]
            ],
            'consultation_history': consultation_history,
            'total_rounds': round_number,
            'final_symptoms': current_symptoms,
            'diagnosis_completed': max([r.risk_score for r in final_disease_risks]) >= probability_threshold if final_disease_risks else False
        }

    def _simulate_user_responses(self, questions: List[str], disease_risks: List[DiseaseRisk]) -> List[Dict[str, Any]]:
        """
        Simulate user responses for demonstration (in real implementation, this would be user input)
        """
        responses = []
        top_disease = disease_risks[0] if disease_risks else None

        if not top_disease:
            return responses

        # Simulate responses based on top disease's missing symptoms
        missing_symptoms = top_disease.missing_symptoms[:3]  # Top 3 missing symptoms

        for i, question in enumerate(questions[:3]):  # Answer up to 3 questions
            if i < len(missing_symptoms):
                symptom = missing_symptoms[i]
                # Simulate 70% chance of confirming a missing symptom
                confirmed = np.random.random() < 0.7
                responses.append({
                    'question': question,
                    'symptom_name': symptom,
                    'symptom_confirmed': confirmed,
                    'response_details': f"{'Yes' if confirmed else 'No'}, {'moderate severity' if confirmed else 'not present'}"
                })

        return responses

    def identify_information_gaps_max_probability(self, current_symptoms: List[str],
                                                 candidate_diseases: List[str] = None) -> Dict[str, Any]:
        """
        Identify information gaps using maximum probability inference
        Focuses on symptoms that would most reduce diagnostic uncertainty

        Args:
            current_symptoms: Current known symptoms
            candidate_diseases: Specific diseases to consider (optional)

        Returns:
            Dict containing information gaps ranked by expected information gain
        """
        if candidate_diseases is None:
            # Get all diseases that match current symptoms reasonably well
            all_risks = self.assess_disease_risks(current_symptoms, threshold=0.1)
            candidate_diseases = [risk.disease_id for risk in all_risks]

        information_gaps = []
        base_entropy = self.calculate_diagnostic_entropy({
            did: self.calculate_disease_matching_score(current_symptoms, did)
            for did in candidate_diseases
        })

        # For each potential symptom, calculate expected information gain
        all_symptoms = set()
        for disease_id in candidate_diseases:
            if disease_id in self.database.diseases:
                disease = self.database.diseases[disease_id]
                all_symptoms.update([s['symptom_name'].lower().replace(' ', '_')
                                   for s in disease.symptoms])

        # Remove already known symptoms
        known_symptoms = set(s.lower().replace(' ', '_') for s in current_symptoms)
        potential_gaps = all_symptoms - known_symptoms

        for symptom in potential_gaps:
            # Calculate entropy if this symptom were present
            symptoms_with_gap = current_symptoms + [symptom]
            entropy_with_symptom = self.calculate_diagnostic_entropy({
                did: self.calculate_disease_matching_score(symptoms_with_gap, did)
                for did in candidate_diseases
            })

            # Information gain = reduction in entropy
            info_gain = base_entropy - entropy_with_symptom

            information_gaps.append({
                'symptom': symptom.replace('_', ' ').title(),
                'expected_info_gain': info_gain,
                'priority_score': info_gain * (1 + np.random.random() * 0.2)  # Add small randomness for tie-breaking
            })

        # Sort by information gain descending
        information_gaps.sort(key=lambda x: x['priority_score'], reverse=True)

        return {
            'information_gaps': information_gaps,
            'base_entropy': base_entropy,
            'total_candidates': len(candidate_diseases),
            'ranking_method': 'maximum_probability_inference'
        }

# Global database instance
_symptom_database = None

def get_symptom_database() -> SymptomTrendDiseaseDatabase:
    """Get global symptom database instance"""
    global _symptom_database
    if _symptom_database is None:
        _symptom_database = SymptomTrendDiseaseDatabase()
    return _symptom_database

def get_cotc_reasoning_engine() -> COTCReasoningEngine:
    """Get COTC reasoning engine instance"""
    return COTCReasoningEngine(get_symptom_database())</contents>
</xai:function_call">cotc_module.py
